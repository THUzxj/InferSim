import argparse
import json

from config.model_config import ModelConfig
from models.hybrid_model import HybridModel
from models.model import Model


def main(args):
    config = ModelConfig(args.config_path)

    print("\n{s:{c}^{n}}".format(s=" Simulator Result ", n=50, c="="))
    print("{:<40} {:<10}".format("Device type:", args.device_type))
    print("{:<40} {:<10}".format("World size:", args.world_size))
    print("{:<40} {:<10}".format("Attn type:", config.attn_type))
    print("{:<40} {:<10}".format("Use FP8 GEMM:", args.use_fp8_gemm))
    print("{:<40} {:<10}".format("Use FP8 KV:", args.use_fp8_kv))

    # Initialize result dictionary
    result = {
        "config": {
            "device_type": args.device_type,
            "world_size": args.world_size,
            "num_nodes": args.num_nodes,
            "attn_type": config.attn_type,
            "use_fp8_gemm": args.use_fp8_gemm,
            "use_fp8_kv": args.use_fp8_kv,
            "max_prefill_tokens": args.max_prefill_tokens,
            "target_isl": args.target_isl,
            "target_osl": args.target_osl,
            "target_tgs": args.target_tgs,
            "target_tpot": args.target_tpot,
            "enable_deepep": args.enable_deepep,
            "enable_tbo": args.enable_tbo,
            "is_hybrid_linear": config.is_hybrid_linear,
        }
    }

    if config.is_hybrid_linear:
        model = HybridModel(args, config)
    else:
        model = Model(args, config)

    weights_info = model.get_weights_info()
    result["weights"] = weights_info

    kvcache_info = model.get_kvcache_info()
    result["kvcache"] = kvcache_info

    flops_info = model.get_flops_info()
    result["flops"] = flops_info

    if not args.decode_only:
        prefill_result = model.prefill()
        result["prefill"] = prefill_result

    if not args.prefill_only:
        decode_result = model.decoding()
        result["decoding"] = decode_result

    # Save to JSON file if specified
    if args.output_json:
        with open(args.output_json, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {args.output_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-path",
        type=str,
        help="The path of the hf model config.json",
        required=True,
    )
    parser.add_argument(
        "--device-type",
        type=str,
        default="H20",
        choices=["H20", "H800", "H200", "GB200"],
        help="Device type",
    )
    parser.add_argument("--world-size", type=int,
                        default=1, help="Num of GPUs")
    parser.add_argument("--num-nodes", type=int,
                        default=1, help="Num of nodes")
    parser.add_argument(
        "--max-prefill-tokens", type=int, default=4096, help="Max prefill tokens"
    )
    parser.add_argument(
        "--decode-bs",
        type=int,
        help="Decoding batchsize. If not specified, bs = tgs * tpot.",
    )
    parser.add_argument(
        "--target-tgs", type=float, default=2560, help="Target tokens/s per GPU"
    )
    parser.add_argument("--target-tpot", type=float,
                        default=50, help="TPOT in ms")
    parser.add_argument(
        "--target-isl", type=int, default=4096, help="Input sequence length, in tokens"
    )
    parser.add_argument(
        "--target-osl", type=int, default=2048, help="Output sequence length, in tokens"
    )
    parser.add_argument(
        "--use-fp8-gemm", action="store_true", help="Use fp8 gemm")
    parser.add_argument("--use-fp8-kv", action="store_true",
                        help="Use fp8 kvcache")
    parser.add_argument("--enable-deepep",
                        action="store_true", help="Enable DeepEP")
    parser.add_argument(
        "--enable-tbo", action="store_true", help="Enable two batch overlap"
    )
    parser.add_argument(
        "--sm-ratio",
        type=float,
        default=108 / 132,
        help="In TBO DeepEP normal mode, the SM ratio used for computation",
    )
    parser.add_argument(
        "--prefill-only", action="store_true", help="Only simulate prefill"
    )
    parser.add_argument(
        "--decode-only", action="store_true", help="Only simulate decoding"
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Path to save simulation results as JSON file",
    )
    parser.add_argument(
        "--log-tpot-detail",
        action="store_true",
        help="Log detailed TPOT breakdown and save it into the decoding result",
    )
    args = parser.parse_args()
    main(args)
