#!/usr/bin/env python3
"""
Cost calculator for comparing inference configurations.

Compares GPU costs, calculates cost-per-token, and projects monthly spend
for different hardware + quantization combinations.

Usage:
    python cost_calculator.py
    python cost_calculator.py --model-size 70 --precision fp8 --gpu H100
    python cost_calculator.py --compare  # side-by-side comparison table
"""

import argparse
import json

# Baseten GPU pricing ($/minute, as of 2026)
GPU_SPECS = {
    "T4": {
        "price_per_min": 0.01052,
        "vram_gb": 16,
        "mem_bw_gbps": 300,
        "fp16_tflops": 65,
        "fp8_tflops": None,  # No FP8 support
    },
    "L4": {
        "price_per_min": 0.01414,
        "vram_gb": 24,
        "mem_bw_gbps": 300,
        "fp16_tflops": 121,
        "fp8_tflops": 242,
    },
    "A100": {
        "price_per_min": 0.06667,
        "vram_gb": 80,
        "mem_bw_gbps": 2039,
        "fp16_tflops": 312,
        "fp8_tflops": None,  # No native FP8
    },
    "H100": {
        "price_per_min": 0.10833,
        "vram_gb": 80,
        "mem_bw_gbps": 3350,
        "fp16_tflops": 990,
        "fp8_tflops": 1979,
    },
    "B200": {
        "price_per_min": 0.16633,
        "vram_gb": 192,
        "mem_bw_gbps": 8000,
        "fp16_tflops": 2250,
        "fp8_tflops": 4500,
    },
}

# Bytes per parameter at each precision
PRECISION_BYTES = {
    "fp32": 4.0,
    "fp16": 2.0,
    "bf16": 2.0,
    "fp8": 1.0,
    "fp4": 0.5,
    "int4": 0.5,
}

# Rough throughput multipliers (relative to fp16 baseline)
# These are approximate — real numbers come from benchmarking
THROUGHPUT_MULTIPLIER = {
    "fp16": 1.0,
    "fp8": 1.8,   # ~80% more throughput due to smaller memory reads + tensor cores
    "fp4": 2.5,   # ~150% more throughput (aggressive quantization)
}


def model_memory_gb(params_billions: float, precision: str) -> float:
    """Calculate GPU memory needed for model weights."""
    bytes_per_param = PRECISION_BYTES[precision]
    return params_billions * bytes_per_param  # billions × bytes = GB


def gpus_needed(params_billions: float, precision: str, gpu: str) -> int:
    """Calculate minimum GPUs needed to fit the model."""
    weight_memory = model_memory_gb(params_billions, precision)
    overhead_factor = 1.15  # 15% overhead for framework, activations
    total_needed = weight_memory * overhead_factor
    gpu_vram = GPU_SPECS[gpu]["vram_gb"]
    return max(1, -(-int(total_needed) // gpu_vram))  # ceiling division


def estimate_throughput(
    params_billions: float,
    precision: str,
    gpu: str,
    gpu_count: int,
) -> float:
    """
    Estimate system throughput (tokens/sec) for a configuration.

    This is a rough model based on memory bandwidth. Real numbers vary
    significantly — always benchmark for accurate results.
    """
    mem_bw = GPU_SPECS[gpu]["mem_bw_gbps"] * gpu_count
    weight_size_gb = model_memory_gb(params_billions, precision)

    # Theoretical max decode tokens/sec (memory-bandwidth limited)
    # Each token requires reading all weights once
    theoretical_max = mem_bw / weight_size_gb

    # Real-world efficiency (overhead, KV cache reads, batching effects)
    efficiency = 0.4  # Typical: 30-50% of theoretical

    # Quantization throughput bonus
    quant_multiplier = THROUGHPUT_MULTIPLIER.get(precision, 1.0)

    return theoretical_max * efficiency * quant_multiplier


def calculate_config(
    params_billions: float,
    precision: str,
    gpu: str,
    replicas: int = 1,
    hours_per_day: float = 24.0,
    days_per_month: int = 30,
) -> dict:
    """Calculate full cost and performance for a configuration."""

    gpu_count = gpus_needed(params_billions, precision, gpu)
    weight_mem = model_memory_gb(params_billions, precision)
    gpu_vram = GPU_SPECS[gpu]["vram_gb"] * gpu_count
    kv_cache_room = gpu_vram - (weight_mem * 1.1)  # 10% overhead

    throughput = estimate_throughput(params_billions, precision, gpu, gpu_count)
    price_per_min = GPU_SPECS[gpu]["price_per_min"] * gpu_count
    price_per_hour = price_per_min * 60

    # Cost per million tokens
    tokens_per_hour = throughput * 3600
    cost_per_million = (price_per_hour / tokens_per_hour) * 1_000_000 if tokens_per_hour > 0 else float('inf')

    # Monthly cost
    monthly_gpu_cost = price_per_hour * hours_per_day * days_per_month * replicas

    return {
        "model_params_b": params_billions,
        "precision": precision,
        "gpu": gpu,
        "gpu_count": gpu_count,
        "replicas": replicas,

        "weight_memory_gb": round(weight_mem, 1),
        "total_vram_gb": gpu_vram,
        "kv_cache_room_gb": round(max(0, kv_cache_room), 1),

        "estimated_throughput_tps": round(throughput, 0),
        "cost_per_hour": round(price_per_hour * replicas, 2),
        "cost_per_million_tokens": round(cost_per_million, 4),
        "monthly_cost": round(monthly_gpu_cost, 0),

        "note": f"{gpu_count}x {gpu}" + (f" (TP={gpu_count})" if gpu_count > 1 else ""),
    }


def print_comparison_table(configs: list[dict]):
    """Print a formatted comparison table."""

    print("\n" + "=" * 100)
    print("  CONFIGURATION COMPARISON")
    print("=" * 100)

    # Header
    header = f"{'Config':<20} {'GPUs':<10} {'Weights':<10} {'KV Room':<10} {'Throughput':<12} {'$/hr':<8} {'$/1M tok':<10} {'$/month':<10}"
    print(header)
    print("-" * 100)

    for c in configs:
        row = (
            f"{c['precision']} on {c['note']:<10}"[:20].ljust(20)
            + f"{c['total_vram_gb']}GB".ljust(10)
            + f"{c['weight_memory_gb']}GB".ljust(10)
            + f"{c['kv_cache_room_gb']}GB".ljust(10)
            + f"{c['estimated_throughput_tps']:.0f} t/s".ljust(12)
            + f"${c['cost_per_hour']:.2f}".ljust(8)
            + f"${c['cost_per_million_tokens']:.4f}".ljust(10)
            + f"${c['monthly_cost']:,.0f}".ljust(10)
        )
        print(row)

    print("=" * 100)
    print("  Note: Throughput estimates are approximate. Run benchmark.py for real numbers.")
    print()


def run_standard_comparison(params_b: float):
    """Run a standard comparison across common configurations."""

    configs = []

    # Try each GPU at each viable precision
    for precision in ["fp16", "fp8", "fp4"]:
        for gpu in ["A100", "H100", "B200"]:
            # Skip if GPU doesn't support the precision
            if precision == "fp8" and GPU_SPECS[gpu]["fp8_tflops"] is None:
                if gpu in ["T4", "A100"]:  # A100 can emulate but not great
                    continue

            gpu_count = gpus_needed(params_b, precision, gpu)

            # Skip configs that need too many GPUs
            if gpu_count > 8:
                continue

            config = calculate_config(params_b, precision, gpu)
            configs.append(config)

    return configs


def main():
    parser = argparse.ArgumentParser(description="Inference cost calculator")
    parser.add_argument("--model-size", type=float, default=70,
                        help="Model size in billions of parameters (default: 70)")
    parser.add_argument("--precision", default=None,
                        help="Quantization precision (fp16, fp8, fp4)")
    parser.add_argument("--gpu", default=None,
                        help="GPU type (T4, L4, A100, H100, B200)")
    parser.add_argument("--replicas", type=int, default=1,
                        help="Number of replicas")
    parser.add_argument("--hours", type=float, default=24,
                        help="Active hours per day (for monthly cost)")
    parser.add_argument("--compare", action="store_true",
                        help="Run full comparison across configs")
    parser.add_argument("--output", default=None,
                        help="Save results to JSON")
    args = parser.parse_args()

    if args.compare or (args.precision is None and args.gpu is None):
        # Full comparison mode
        print(f"\nComparing configurations for {args.model_size}B parameter model:")
        configs = run_standard_comparison(args.model_size)
        print_comparison_table(configs)

        # Find the best cost/token
        best = min(configs, key=lambda c: c["cost_per_million_tokens"])
        print(f"  Best cost/token: {best['precision']} on {best['note']}"
              f" (${best['cost_per_million_tokens']:.4f}/1M tokens)")

        # Find the best throughput
        fastest = max(configs, key=lambda c: c["estimated_throughput_tps"])
        print(f"  Best throughput: {fastest['precision']} on {fastest['note']}"
              f" ({fastest['estimated_throughput_tps']:.0f} tokens/sec)")

        if args.output:
            with open(args.output, "w") as f:
                json.dump(configs, f, indent=2)
            print(f"\n  Saved to {args.output}")

    else:
        # Single config mode
        config = calculate_config(
            params_billions=args.model_size,
            precision=args.precision or "fp8",
            gpu=args.gpu or "H100",
            replicas=args.replicas,
            hours_per_day=args.hours,
        )

        print(f"\n  Configuration: {config['precision']} on {config['note']}")
        print(f"  Weight memory:   {config['weight_memory_gb']} GB")
        print(f"  Total VRAM:      {config['total_vram_gb']} GB")
        print(f"  KV cache room:   {config['kv_cache_room_gb']} GB")
        print(f"  Est. throughput: {config['estimated_throughput_tps']:.0f} tokens/sec")
        print(f"  Cost/hour:       ${config['cost_per_hour']:.2f}")
        print(f"  Cost/1M tokens:  ${config['cost_per_million_tokens']:.4f}")
        print(f"  Monthly cost:    ${config['monthly_cost']:,.0f}")


if __name__ == "__main__":
    main()
