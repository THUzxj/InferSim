import math

from comm.comm import Comm
from flops.flops import get_attn_gflops, get_moe_gflops
from hardware.gpu import gpu_map
from kvcache.kvcache import get_kvcache_size, get_states_size
from layers.attn import create_attention
from layers.linear_attn import create_linear_attn
from layers.moe import MoE
from params.params import (get_attn_params_size, get_expert_params_size,
                           get_linear_attn_params_size)


class HybridModel:
    def __init__(self, args, config):
        self.gpu = gpu_map[args.device_type]
        self.args = args
        self.config = config

    def print_weights_info(self):
        print("{s:{c}^{n}}".format(s="Model Weights", n=50, c="-"))
        full_attn_params_bytes = get_attn_params_size(
            self.config, self.args.use_fp8_gemm
        )
        linear_attn_params_bytes = get_linear_attn_params_size(
            self.config, self.args.use_fp8_gemm
        )
        expert_params_bytes = get_expert_params_size(
            self.config, self.args.use_fp8_gemm
        )
        print(
            "{:<40} {:<10.2f}".format(
                "One full attn params size (MB):", full_attn_params_bytes / 1024 / 1024
            )
        )
        print(
            "{:<40} {:<10.2f}".format(
                "One linear attn params size (MB):",
                linear_attn_params_bytes / 1024 / 1024,
            )
        )
        print(
            "{:<40} {:<10.2f}".format(
                "One expert params size (MB):", expert_params_bytes / 1024 / 1024
            )
        )
        params_per_gpu = expert_params_bytes * (
            self.config.num_shared_experts
            + self.config.num_routed_experts / self.args.world_size
        )
        params_per_gpu *= self.config.num_hidden_layers
        params_per_gpu += self.config.num_full_attn_layers * full_attn_params_bytes
        params_per_gpu += self.config.num_linear_attn_layers * full_attn_params_bytes

        params_per_gpu = params_per_gpu / 1024 / 1024 / 1024
        self.kvcache_mem = (
            self.gpu.mem - params_per_gpu - 15 - 5
        )  # 15GB for runtime, 5GB for encoder
        print("{:<40} {:<10.2f}".format("Per GPU params size (GB):", params_per_gpu))

    def get_weights_info(self):
        """Get weights information as a dictionary."""
        self.print_weights_info()
        full_attn_params_bytes = get_attn_params_size(
            self.config, self.args.use_fp8_gemm
        )
        linear_attn_params_bytes = get_linear_attn_params_size(
            self.config, self.args.use_fp8_gemm
        )
        expert_params_bytes = get_expert_params_size(
            self.config, self.args.use_fp8_gemm
        )
        params_per_gpu = expert_params_bytes * (
            self.config.num_shared_experts
            + self.config.num_routed_experts / self.args.world_size
        )
        params_per_gpu *= self.config.num_hidden_layers
        params_per_gpu += self.config.num_full_attn_layers * full_attn_params_bytes
        params_per_gpu += self.config.num_linear_attn_layers * full_attn_params_bytes
        params_per_gpu = params_per_gpu / 1024 / 1024 / 1024
        return {
            "one_full_attn_params_size_mb": full_attn_params_bytes / 1024 / 1024,
            "one_linear_attn_params_size_mb": linear_attn_params_bytes / 1024 / 1024,
            "one_expert_params_size_mb": expert_params_bytes / 1024 / 1024,
            "per_gpu_params_size_gb": params_per_gpu,
        }

    def print_kvcache_info(self):
        print("{s:{c}^{n}}".format(s="KV Cache", n=50, c="-"))
        print("{:<40} {:<10.2f}".format("KV cache space (GB):", self.kvcache_mem))
        context_len = self.args.target_isl + self.args.target_osl

        if self.args.decode_bs is None:
            target_bs = math.ceil(self.args.target_tgs * self.args.target_tpot / 1000)
        else:
            target_bs = self.args.decode_bs
        print("{:<40} {:<10}".format("Input seq len:", self.args.target_isl))
        print("{:<40} {:<10}".format("Output seq len:", self.args.target_osl))
        print("{:<40} {:<10}".format("Target decode batchsize:", target_bs))
        target_kvcache_bytes = self.kvcache_mem * 1024 * 1024 * 1024 / target_bs
        kvcache_bytes = (
            get_kvcache_size(self.config, self.args.use_fp8_kv)
            / self.config.num_hidden_layers
        )
        kvcache_bytes *= self.config.num_full_attn_layers * context_len
        print(
            "{:<40} {:<10.2f}".format(
                "Target per-req KV cache size (MB):", target_kvcache_bytes / 1024 / 1024
            )
        )
        print(
            "{:<40} {:<10.2f}".format(
                "Current per-req full attn KV cache size (MB):",
                kvcache_bytes / 1024 / 1024,
            )
        )
        states_bytes = get_states_size(self.config)
        print(
            "{:<40} {:<10.2f}".format(
                "Current per-req states size (MB):", states_bytes / 1024 / 1024
            )
        )
        print(
            "{:<40} {:<10.2f}".format(
                "Current per-req cache size (MB):",
                (kvcache_bytes + states_bytes) / 1024 / 1024,
            )
        )
        if kvcache_bytes + states_bytes > target_kvcache_bytes:
            print("!Error: need smaller kvcache")
        self.kvcache_bytes = kvcache_bytes / context_len
        self.states_bytes = states_bytes
        self.target_bs = target_bs

    def get_kvcache_info(self):
        """Get KV cache information as a dictionary."""
        self.print_kvcache_info()
        context_len = self.args.target_isl + self.args.target_osl
        if self.args.decode_bs is None:
            target_bs = math.ceil(self.args.target_tgs * self.args.target_tpot / 1000)
        else:
            target_bs = self.args.decode_bs
        target_kvcache_bytes = self.kvcache_mem * 1024 * 1024 * 1024 / target_bs
        kvcache_bytes = (
            get_kvcache_size(self.config, self.args.use_fp8_kv)
            / self.config.num_hidden_layers
        )
        kvcache_bytes *= self.config.num_full_attn_layers * context_len
        states_bytes = get_states_size(self.config)
        return {
            "kv_cache_space_gb": self.kvcache_mem,
            "input_seq_len": self.args.target_isl,
            "output_seq_len": self.args.target_osl,
            "target_decode_batchsize": target_bs,
            "target_per_req_kvcache_size_mb": target_kvcache_bytes / 1024 / 1024,
            "current_per_req_full_attn_kvcache_size_mb": kvcache_bytes / 1024 / 1024,
            "current_per_req_states_size_mb": states_bytes / 1024 / 1024,
            "current_per_req_cache_size_mb": (kvcache_bytes + states_bytes) / 1024 / 1024,
            "kvcache_error": kvcache_bytes + states_bytes > target_kvcache_bytes,
        }

    def print_flops_info(self):
        print("{s:{c}^{n}}".format(s="FLOPs", n=50, c="-"))
        print(
            "{:<40} {:<10}".format("Num hidden layers:", self.config.num_hidden_layers)
        )
        # per-token per-layer gflops
        self.avg_context_len = int(self.args.target_isl + self.args.target_osl / 2)
        attn_core_gflops, other_gflops = get_attn_gflops(
            self.config, self.avg_context_len, absorb=True
        )
        moe_gflops = get_moe_gflops(self.config)
        print(
            "{:<40} {:<10.2f}".format(
                "Per-token per-layer full attn core (GFLOPs):", attn_core_gflops
            )
        )
        print(
            "{:<40} {:<10.2f}".format(
                "Per-token per-layer MoE/FFN (GFLOPs):", moe_gflops
            )
        )
        print(
            "{:<40} {:<10.2f}".format(
                "Per-token per-layer others (GFLOPs):", other_gflops
            )
        )
        print(
            "{:<40} {:<10.2f}".format(
                "Per-token full attn core (GFLOPs):",
                attn_core_gflops * self.config.num_full_attn_layers,
            )
        )
        print(
            "{:<40} {:<10.2f}".format(
                "Per-token MoE (GFLOPs):", moe_gflops * self.config.num_hidden_layers
            )
        )
        print(
            "{:<40} {:<10.2f}".format(
                "Per-token others (GFLOPs):",
                other_gflops * self.config.num_hidden_layers,
            )
        )
        print(
            "{:<40} {:<10.2f}".format(
                "Per-token total (GFLOPs):",
                (attn_core_gflops + moe_gflops + other_gflops)
                * self.config.num_hidden_layers,
            )
        )

    def get_flops_info(self):
        """Get FLOPs information as a dictionary."""
        self.print_flops_info()
        self.avg_context_len = int(self.args.target_isl + self.args.target_osl / 2)
        attn_core_gflops, other_gflops = get_attn_gflops(
            self.config, self.avg_context_len, absorb=True
        )
        moe_gflops = get_moe_gflops(self.config)
        return {
            "num_hidden_layers": self.config.num_hidden_layers,
            "avg_context_len": self.avg_context_len,
            "per_token_per_layer_full_attn_core_gflops": attn_core_gflops,
            "per_token_per_layer_moe_ffn_gflops": moe_gflops,
            "per_token_per_layer_others_gflops": other_gflops,
            "per_token_full_attn_core_gflops": attn_core_gflops * self.config.num_full_attn_layers,
            "per_token_moe_gflops": moe_gflops * self.config.num_hidden_layers,
            "per_token_others_gflops": other_gflops * self.config.num_hidden_layers,
            "per_token_total_gflops": (attn_core_gflops + moe_gflops + other_gflops)
            * self.config.num_hidden_layers,
        }

    def prefill(self):
        print("{s:{c}^{n}}".format(s="Prefilling", n=50, c="-"))
        print(
            "{:<40} {:<10}".format("Max prefill tokens:", self.args.max_prefill_tokens)
        )
        # full attn
        full_attn = create_attention(
            self.config, self.args.use_fp8_gemm, self.args.use_fp8_kv
        )
        t_full_attn_core = full_attn.prefill_attn_core(
            self.args.target_isl, self.kvcache_bytes, self.args.device_type
        )
        t_full_attn_others = full_attn.prefill_attn_others(
            self.args.max_prefill_tokens, self.args.device_type
        )
        t_full_attn_core *= self.args.max_prefill_tokens / self.args.target_isl

        # linear attn
        linear_attn = create_linear_attn(self.config, self.args.use_fp8_gemm)
        t_linear_attn_core = linear_attn.prefill_attn_core(
            self.args.target_isl, self.states_bytes, self.args.device_type
        )
        t_linear_attn_core *= self.args.max_prefill_tokens / self.args.target_isl
        t_linear_attn_others = linear_attn.prefill_attn_others(
            self.args.max_prefill_tokens, self.args.device_type
        )

        # moe
        moe = MoE(self.config, self.args.use_fp8_gemm)
        t_moe = moe.prefill_moe(
            self.args.max_prefill_tokens, self.args.device_type, self.args.world_size
        )

        comm = Comm(
            self.config,
            self.gpu,
            self.args.world_size,
            self.args.num_nodes,
            self.args.enable_deepep,
        )
        comm_t1, comm_t2 = comm.prefill_comm(self.args.max_prefill_tokens)
        print("{:<40} {:<10.2f}".format("Comm before MoE/FFN (us):", comm_t1 * 1e6))
        print("{:<40} {:<10.2f}".format("Comm after MoE/FFN (us):", comm_t2 * 1e6))

        num_tokens = self.args.max_prefill_tokens
        ttft = (
            t_full_attn_core + t_full_attn_others
        ) * self.config.num_full_attn_layers
        ttft += (
            t_linear_attn_core + t_linear_attn_others
        ) * self.config.num_linear_attn_layers
        ttft += (t_moe + comm_t1 + comm_t2) * self.config.num_hidden_layers
        ttft *= 1000  # convert to ms
        ttft += 30  # for scheduler

        print("{:<40} {:<10.2f}".format("TTFT (ms):", ttft))
        throughput = num_tokens / (ttft / 1000)
        print(
            "{:<40} {:<10.0f}".format(
                "Throughput (TGS:tok/GPU/s):", throughput
            )
        )
        return {
            "max_prefill_tokens": self.args.max_prefill_tokens,
            "comm_before_moe_ffn_us": comm_t1 * 1e6,
            "comm_after_moe_ffn_us": comm_t2 * 1e6,
            "ttft_ms": ttft,
            "throughput_tgs": throughput,
        }

    def get_prefill_result(self):
        """Get prefill simulation results as a dictionary."""
        return self.prefill()

    def decoding(self):
        print("{s:{c}^{n}}".format(s="Decoding", n=50, c="-"))
        # full attn
        full_attn = create_attention(
            self.config, self.args.use_fp8_gemm, self.args.use_fp8_kv
        )
        t_full_attn_core = full_attn.decode_attn_core(
            self.target_bs,
            self.avg_context_len,
            self.kvcache_bytes,
            self.args.device_type,
        )
        t_full_attn_others = full_attn.decode_attn_others(
            self.target_bs, self.args.device_type
        )

        # linear attn
        linear_attn = create_linear_attn(self.config, self.args.use_fp8_gemm)
        t_linear_attn_core = linear_attn.decode_attn_core(
            self.target_bs, self.states_bytes, self.args.device_type
        )
        t_linear_attn_others = linear_attn.decode_attn_others(
            self.target_bs, self.args.device_type
        )

        moe = MoE(self.config, self.args.use_fp8_gemm)
        t_moe = moe.decode_moe(
            self.target_bs, self.args.device_type, self.args.world_size
        )

        comm = Comm(
            self.config,
            self.gpu,
            self.args.world_size,
            self.args.num_nodes,
            self.args.enable_deepep,
        )
        comm_t1, comm_t2 = comm.decode_comm(self.target_bs)
        print("{:<40} {:<10.2f}".format("Comm before MoE/FFN (us):", comm_t1 * 1e6))
        print("{:<40} {:<10.2f}".format("Comm after MoE/FFN (us):", comm_t2 * 1e6))

        num_tokens = self.target_bs
        tpot = (
            t_full_attn_core + t_full_attn_others
        ) * self.config.num_full_attn_layers
        tpot += (
            t_linear_attn_core + t_linear_attn_others
        ) * self.config.num_linear_attn_layers
        tpot += (t_moe + comm_t1 + comm_t2) * self.config.num_hidden_layers
        tpot *= 1000  # convert to ms
        tpot += 5  # for scheduler

        print("{:<40} {:<10.2f}".format("TPOT (ms):", tpot))
        throughput = num_tokens / tpot * 1000
        print("{:<40} {:<10.0f}".format("Throughput (TGS):", throughput))
        tpot_error = tpot > self.args.target_tpot
        if tpot_error:
            print("!Error: TPOT > SLO, need smaller GFLOPs to speedup")
        return {
            "comm_before_moe_ffn_us": comm_t1 * 1e6,
            "comm_after_moe_ffn_us": comm_t2 * 1e6,
            "tpot_ms": tpot,
            "throughput_tgs": throughput,
            "tpot_error": tpot_error,
        }

    def get_decode_result(self):
        """Get decoding simulation results as a dictionary."""
        return self.decoding()
