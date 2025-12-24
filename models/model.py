import math

from comm.comm import Comm
from flops.flops import get_attn_gflops, get_moe_gflops
from hardware.gpu import gpu_map
from kvcache.kvcache import get_kvcache_size
from layers.attn import create_attention
from layers.moe import MoE
from params.params import get_attn_params_size, get_expert_params_size


class Model:
    def __init__(self, args, config):
        self.gpu = gpu_map[args.device_type]
        self.args = args
        self.config = config

    def print_weights_info(self):
        print("{s:{c}^{n}}".format(s="Model Weights", n=50, c="-"))
        attn_params_bytes = get_attn_params_size(
            self.config, self.args.use_fp8_gemm)
        expert_params_bytes = get_expert_params_size(
            self.config, self.args.use_fp8_gemm
        )
        print(
            "{:<40} {:<10.2f}".format(
                "One attn params size (MB):", attn_params_bytes / 1024 / 1024
            )
        )
        print(
            "{:<40} {:<10.2f}".format(
                "One expert params size (MB):", expert_params_bytes /
                1024 / 1024
            )
        )
        params_per_gpu = attn_params_bytes + expert_params_bytes * (
            self.config.num_shared_experts
            + self.config.num_routed_experts / self.args.world_size
        )
        params_per_gpu = params_per_gpu / 1024 / 1024 / 1024
        params_per_gpu *= self.config.num_hidden_layers
        self.kvcache_mem = (
            self.gpu.mem - params_per_gpu - 15 - 5
        )  # 15GB for runtime, 5GB for encoder
        print("{:<40} {:<10.2f}".format(
            "Per GPU params size (GB):", params_per_gpu))

    def get_weights_info(self):
        """Get weights information as a dictionary."""
        self.print_weights_info()
        attn_params_bytes = get_attn_params_size(
            self.config, self.args.use_fp8_gemm)
        expert_params_bytes = get_expert_params_size(
            self.config, self.args.use_fp8_gemm
        )
        params_per_gpu = attn_params_bytes + expert_params_bytes * (
            self.config.num_shared_experts
            + self.config.num_routed_experts / self.args.world_size
        )
        params_per_gpu = params_per_gpu / 1024 / 1024 / 1024
        params_per_gpu *= self.config.num_hidden_layers
        return {
            "one_attn_params_size_mb": attn_params_bytes / 1024 / 1024,
            "one_expert_params_size_mb": expert_params_bytes / 1024 / 1024,
            "per_gpu_params_size_gb": params_per_gpu,
        }

    def print_kvcache_info(self):
        print("{s:{c}^{n}}".format(s="KV Cache", n=50, c="-"))
        print("{:<40} {:<10.2f}".format(
            "KV cache space (GB):", self.kvcache_mem))
        context_len = self.args.target_isl + self.args.target_osl

        if self.args.decode_bs is None:
            target_bs = math.ceil(self.args.target_tgs *
                                  self.args.target_tpot / 1000)
        else:
            target_bs = self.args.decode_bs
        print("{:<40} {:<10}".format("Input seq len:", self.args.target_isl))
        print("{:<40} {:<10}".format("Output seq len:", self.args.target_osl))
        print("{:<40} {:<10}".format("Target decode batchsize:", target_bs))
        target_kvcache_bytes = (
            self.kvcache_mem * 1024 * 1024 * 1024 / target_bs / context_len
        )
        kvcache_bytes = get_kvcache_size(self.config, self.args.use_fp8_kv)
        print(
            "{:<40} {:<10.2f}".format(
                "Target per-token KV cache size (KB):", target_kvcache_bytes / 1024
            )
        )
        print(
            "{:<40} {:<10.2f}".format(
                "Current per-token KV cache size (KB):", kvcache_bytes / 1024
            )
        )
        if kvcache_bytes > target_kvcache_bytes:
            print("!Error: need smaller kvcache")
        self.kvcache_bytes = kvcache_bytes
        self.target_bs = target_bs

    def get_kvcache_info(self):
        """Get KV cache information as a dictionary."""
        self.print_kvcache_info()
        context_len = self.args.target_isl + self.args.target_osl
        if self.args.decode_bs is None:
            target_bs = math.ceil(self.args.target_tgs *
                                  self.args.target_tpot / 1000)
        else:
            target_bs = self.args.decode_bs
        target_kvcache_bytes = (
            self.kvcache_mem * 1024 * 1024 * 1024 / target_bs / context_len
        )
        kvcache_bytes = get_kvcache_size(self.config, self.args.use_fp8_kv)
        return {
            "kv_cache_space_gb": self.kvcache_mem,
            "input_seq_len": self.args.target_isl,
            "output_seq_len": self.args.target_osl,
            "target_decode_batchsize": target_bs,
            "target_per_token_kvcache_size_kb": target_kvcache_bytes / 1024,
            "current_per_token_kvcache_size_kb": kvcache_bytes / 1024,
            "kvcache_error": kvcache_bytes > target_kvcache_bytes,
        }

    def print_flops_info(self):
        print("{s:{c}^{n}}".format(s="FLOPs", n=50, c="-"))
        print(
            "{:<40} {:<10}".format("Num hidden layers:",
                                   self.config.num_hidden_layers)
        )
        # per-token per-layer gflops
        self.avg_context_len = int(
            self.args.target_isl + self.args.target_osl / 2)
        attn_core_gflops, other_gflops = get_attn_gflops(
            self.config, self.avg_context_len, absorb=True
        )
        moe_gflops = get_moe_gflops(self.config)
        print(
            "{:<40} {:<10.2f}".format(
                "Per-token per-layer attn core (GFLOPs):", attn_core_gflops
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
                "Per-token attn core (GFLOPs):",
                attn_core_gflops * self.config.num_hidden_layers,
            )
        )
        print(
            "{:<40} {:<10.2f}".format(
                "Per-token MoE (GFLOPs):", moe_gflops *
                self.config.num_hidden_layers
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
        self.avg_context_len = int(
            self.args.target_isl + self.args.target_osl / 2)
        attn_core_gflops, other_gflops = get_attn_gflops(
            self.config, self.avg_context_len, absorb=True
        )
        moe_gflops = get_moe_gflops(self.config)
        return {
            "num_hidden_layers": self.config.num_hidden_layers,
            "avg_context_len": self.avg_context_len,
            "per_token_per_layer_attn_core_gflops": attn_core_gflops,
            "per_token_per_layer_moe_ffn_gflops": moe_gflops,
            "per_token_per_layer_others_gflops": other_gflops,
            "per_token_attn_core_gflops": attn_core_gflops * self.config.num_hidden_layers,
            "per_token_moe_gflops": moe_gflops * self.config.num_hidden_layers,
            "per_token_others_gflops": other_gflops * self.config.num_hidden_layers,
            "per_token_total_gflops": (attn_core_gflops + moe_gflops + other_gflops)
            * self.config.num_hidden_layers,
        }

    def prefill(self):
        print("{s:{c}^{n}}".format(s="Prefilling", n=50, c="-"))
        print(
            "{:<40} {:<10}".format("Max prefill tokens:",
                                   self.args.max_prefill_tokens)
        )
        attn = create_attention(
            self.config, self.args.use_fp8_gemm, self.args.use_fp8_kv
        )
        attn_core_time = attn.prefill_attn_core(
            self.args.target_isl, self.kvcache_bytes, self.args.device_type
        )
        attn_other_time = attn.prefill_attn_others(
            self.args.max_prefill_tokens, self.args.device_type
        )
        attn_core_time *= math.ceil(self.args.max_prefill_tokens /
                                    self.args.target_isl)

        moe = MoE(self.config, self.args.use_fp8_gemm)
        moe_time = moe.prefill_moe(
            self.args.max_prefill_tokens, self.args.device_type, self.args.world_size
        )

        comm = Comm(
            self.config,
            self.gpu,
            self.args.world_size,
            self.args.num_nodes,
            self.args.enable_deepep,
        )
        comm_time1, comm_time2 = comm.prefill_comm(
            self.args.max_prefill_tokens)
        print("{:<40} {:<10.2f}".format(
            "Comm before MoE/FFN (us):", comm_time1 * 1e6))
        print("{:<40} {:<10.2f}".format(
            "Comm after MoE/FFN (us):", comm_time2 * 1e6))

        num_tokens = self.args.max_prefill_tokens
        if self.args.enable_tbo:
            num_tokens *= 2
            ttft = max(
                (attn_core_time + attn_other_time) /
                self.args.sm_ratio, comm_time1
            )
            ttft += max(
                (attn_core_time + attn_other_time) /
                self.args.sm_ratio, comm_time2
            )
            ttft += max(moe_time / self.args.sm_ratio, comm_time1)
            ttft += max(moe_time / self.args.sm_ratio, comm_time2)
        else:
            ttft = attn_core_time
            ttft += moe_time
            ttft += attn_other_time
            ttft += comm_time1 + comm_time2
        ttft *= self.config.num_hidden_layers
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
            "comm_before_moe_ffn_us": comm_time1 * 1e6,
            "comm_after_moe_ffn_us": comm_time2 * 1e6,
            "ttft_ms": ttft,
            "throughput_tgs": throughput,
        }

    def decoding(self):
        print("{s:{c}^{n}}".format(s="Decoding", n=50, c="-"))
        attn = create_attention(
            self.config, self.args.use_fp8_gemm, self.args.use_fp8_kv
        )
        attn_core_time = attn.decode_attn_core(
            self.target_bs,
            self.avg_context_len,
            self.kvcache_bytes,
            self.args.device_type,
        )
        attn_other_time = attn.decode_attn_others(
            self.target_bs, self.args.device_type)

        moe = MoE(self.config, self.args.use_fp8_gemm)
        moe_time = moe.decode_moe(
            self.target_bs, self.args.device_type, self.args.world_size
        )

        comm = Comm(
            self.config,
            self.gpu,
            self.args.world_size,
            self.args.num_nodes,
            self.args.enable_deepep,
        )
        comm_time1, comm_time2 = comm.decode_comm(self.target_bs)
        print("{:<40} {:<10.2f}".format(
            "Comm before MoE/FFN (us):", comm_time1 * 1e6))
        print("{:<40} {:<10.2f}".format(
            "Comm after MoE/FFN (us):", comm_time2 * 1e6))

        num_tokens = self.target_bs
        # base per-layer TPOT (in seconds) before multiplying by num_layers and scheduler
        if self.args.enable_tbo:
            # Two micro-batch overlapping
            base_tpot_per_layer = max(
                attn_core_time + attn_other_time, moe_time + comm_time1 + comm_time2
            ) * 2
        else:
            base_tpot_per_layer = (
                attn_core_time
                + attn_other_time
                + moe_time
                + comm_time1
                + comm_time2
            )

        if self.args.enable_tbo:
            num_tokens *= 2

        if not self.args.enable_tbo:
            attn_ratio = (attn_core_time + attn_other_time) / base_tpot_per_layer
            moe_ratio = moe_time / base_tpot_per_layer
            comm_ratio = (comm_time1 + comm_time2) / base_tpot_per_layer
            print("{:<40} {:<10.2f}%".format(
                "Attention ratio (%):", attn_ratio * 100))
            print("{:<40} {:<10.2f}%".format(
                "MoE ratio (%):", moe_ratio * 100))
            print("{:<40} {:<10.2f}%".format(
                "Communication ratio (%):", comm_ratio * 100))
    
        tpot = base_tpot_per_layer * self.config.num_hidden_layers
        tpot *= 1000  # convert to ms
        tpot += 5  # for scheduler        
        print("{:<40} {:<10.2f}".format("TPOT (ms):", tpot))
        throughput = num_tokens / tpot * 1000
        print("{:<40} {:<10.0f}".format("Throughput (TGS):", throughput))
        tpot_error = tpot > self.args.target_tpot
        if tpot_error:
            print("!Error: TPOT > SLO, need smaller GFLOPs to speedup")
        result = {
            "comm_before_moe_ffn_us": comm_time1 * 1e6,
            "comm_after_moe_ffn_us": comm_time2 * 1e6,
            "tpot_ms": tpot,
            "throughput_tgs": throughput,
            "tpot_error": tpot_error,
        }

        # Optionally log detailed TPOT breakdown into the result
        if getattr(self.args, "log_tpot_detail", False):
            # 构造数学表达式形式的 TPOT 计算公式（符号形式 + 数值展开）
            if self.args.enable_tbo:
                tpot_equation = (
                    "tpot_ms = (max(attn_core_time_s + attn_other_time_s, "
                    "moe_time_s + comm_before_s + comm_after_s) * 2 "
                    "* num_layers) * 1000 + scheduler_overhead_ms"
                )
                tpot_equation_numeric = (
                    "tpot_ms = (max("
                    f"{attn_core_time:.6f} + {attn_other_time:.6f}, "
                    f"{moe_time:.6f} + {comm_time1:.6f} + {comm_time2:.6f}"
                    f") * 2 * {self.config.num_hidden_layers}) * 1000"
                    f" + 5.0 = {tpot:.6f}"
                )
            else:
                tpot_equation = (
                    "tpot_ms = (attn_core_time_s + attn_other_time_s + "
                    "moe_time_s + comm_before_s + comm_after_s) "
                    "* num_layers * 1000 + scheduler_overhead_ms"
                )
                tpot_equation_numeric = (
                    "tpot_ms = ("
                    f"{attn_core_time:.6f} + {attn_other_time:.6f} + "
                    f"{moe_time:.6f} + {comm_time1:.6f} + {comm_time2:.6f}"
                    f") * {self.config.num_hidden_layers} * 1000"
                    f" + 5.0 = {tpot:.6f}"
                )

            result["tpot_detail"] = {
                "attn_core_time_s": attn_core_time,
                "attn_other_time_s": attn_other_time,
                "moe_time_s": moe_time,
                "comm_time_before_moe_ffn_s": comm_time1,
                "comm_time_after_moe_ffn_s": comm_time2,
                "enable_tbo": self.args.enable_tbo,
                "tpot_per_layer_ms_before_scheduler": base_tpot_per_layer * 1000,
                "num_layers": self.config.num_hidden_layers,
                "scheduler_overhead_ms": 5.0,
                "num_tokens": num_tokens,
                "tpot_equation": tpot_equation,
                "tpot_equation_numeric": tpot_equation_numeric,
            }

        return result
