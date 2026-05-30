#!/usr/bin/env python3
"""
Compare the same model at different quantization levels.

Deploys (or benchmarks already-deployed) versions of a model at fp16, fp8,
and fp4, then runs identical benchmarks against each and produces a
comparison report.

Usage:
    # Compare against already-deployed endpoints
    python compare_quantizations.py \
        --fp16-endpoint https://model-fp16.api.baseten.co/v1 \
        --fp8-endpoint https://model-fp8.api.baseten.co/v1 \
        --fp4-endpoint https://model-fp4.api.baseten.co/v1 \
        --model meta-llama/Llama-3.1-70B-Instruct

    # Just run the cost comparison (no live endpoints needed)
    python compare_quantizations.py --cost-only --model-size 70

    # With custom prompts
    python compare_quantizations.py --prompt-file prompts.jsonl
"""

import argparse
import asyncio
import json
import os
import sys
import time

import numpy as np

# Import from our benchmark script
sys.path.insert(0, os.path.dirname(__file__))
from benchmark import run_benchmark, compute_summary, print_report
from cost_calculator import calculate_config, GPU_SPECS


def print_quantization_comparison(results: dict, model_name: str):
    """Print a side-by-side comparison of quantization levels."""

    print("\n" + "=" * 80)
    print(f"  QUANTIZATION COMPARISON: {model_name}")
    print("=" * 80)

    precisions = list(results.keys())

    # Header
    header = f"{'Metric':<25}"
    for p in precisions:
        header += f"{p:>15}"
    print(header)
    print("-" * 80)

    # Metrics to compare
    metrics = [
        ("TTFT p50 (ms)", "ttft_p50_ms"),
        ("TTFT p95 (ms)", "ttft_p95_ms"),
        ("TTFT p99 (ms)", "ttft_p99_ms"),
        ("Total Latency p50 (ms)", "total_latency_p50_ms"),
        ("Total Latency p95 (ms)", "total_latency_p95_ms"),
        ("TPOT p50 (ms)", "tpot_p50_ms"),
        ("System Throughput (t/s)", "system_throughput_tps"),
        ("Per-req Throughput (t/s)", "avg_request_throughput_tps"),
        ("Avg Output Tokens", "output_tokens_mean"),
        ("Error Rate (%)", "error_rate_pct"),
    ]

    baseline_key = precisions[0]

    for label, key in metrics:
        row = f"{label:<25}"
        baseline_val = results[baseline_key].get(key)

        for p in precisions:
            val = results[p].get(key)
            if val is None:
                row += f"{'N/A':>15}"
            elif p == baseline_key:
                row += f"{val:>15.1f}"
            else:
                # Show value + percentage change from baseline
                if baseline_val and baseline_val != 0:
                    pct = ((val - baseline_val) / baseline_val) * 100
                    sign = "+" if pct > 0 else ""
                    row += f"{val:>8.1f} ({sign}{pct:.0f}%)"
                else:
                    row += f"{val:>15.1f}"
        print(row)

    print("=" * 80)

    # Recommendations
    print("\n  Recommendations:")

    if len(precisions) >= 2:
        fp16 = results.get("fp16", {})
        fp8 = results.get("fp8", {})

        if fp16 and fp8:
            ttft_improvement = (1 - fp8.get("ttft_p50_ms", 0) / max(fp16.get("ttft_p50_ms", 1), 1)) * 100
            throughput_improvement = (fp8.get("system_throughput_tps", 0) / max(fp16.get("system_throughput_tps", 1), 1) - 1) * 100

            print(f"  - fp8 vs fp16: {ttft_improvement:.0f}% lower TTFT, {throughput_improvement:.0f}% higher throughput")
            print(f"  - fp8 is recommended for most production workloads (minimal quality loss)")

    if "fp4" in results:
        print(f"  - fp4: Maximum throughput, but evaluate quality on YOUR specific prompts")
        print(f"  - fp4 is best for: high-volume batch processing, cost-sensitive workloads")

    print()


def run_cost_comparison(model_size: float, gpu: str = "H100"):
    """Run a cost-only comparison (no live endpoints needed)."""

    print(f"\n  Cost Comparison: {model_size}B model on {gpu}")
    print("=" * 80)

    configs = []
    for precision in ["fp16", "fp8", "fp4"]:
        config = calculate_config(model_size, precision, gpu)
        configs.append(config)

    header = f"{'Metric':<30}{'fp16':>15}{'fp8':>15}{'fp4':>15}"
    print(header)
    print("-" * 80)

    metrics = [
        ("Weight Memory (GB)", "weight_memory_gb"),
        ("GPUs Needed", "gpu_count"),
        ("KV Cache Room (GB)", "kv_cache_room_gb"),
        ("Est. Throughput (t/s)", "estimated_throughput_tps"),
        ("Cost/Hour ($)", "cost_per_hour"),
        ("Cost/1M Tokens ($)", "cost_per_million_tokens"),
        ("Monthly Cost ($)", "monthly_cost"),
    ]

    for label, key in metrics:
        row = f"{label:<30}"
        for config in configs:
            val = config[key]
            if isinstance(val, float) and val > 100:
                row += f"${val:>13,.0f}" if "cost" in key.lower() or "monthly" in key.lower() else f"{val:>15,.0f}"
            elif isinstance(val, float):
                row += f"{val:>15.2f}"
            else:
                row += f"{val:>15}"
        print(row)

    print("=" * 80)

    # Cost savings summary
    fp16_monthly = configs[0]["monthly_cost"]
    fp8_monthly = configs[1]["monthly_cost"]
    fp4_monthly = configs[2]["monthly_cost"]

    print(f"\n  Monthly savings vs fp16:")
    print(f"    fp8:  ${fp16_monthly - fp8_monthly:,.0f}/month ({(1 - fp8_monthly/fp16_monthly)*100:.0f}% savings)")
    print(f"    fp4:  ${fp16_monthly - fp4_monthly:,.0f}/month ({(1 - fp4_monthly/fp16_monthly)*100:.0f}% savings)")
    print()


async def run_live_comparison(
    endpoints: dict[str, str],
    model: str,
    prompts: list[str],
    num_requests: int,
    concurrency: int,
    api_key: str | None,
):
    """Benchmark multiple endpoints and compare results."""

    results = {}

    for precision, endpoint in endpoints.items():
        if not endpoint:
            continue

        print(f"\n{'='*40}")
        print(f"  Benchmarking {precision}...")
        print(f"  Endpoint: {endpoint}")
        print(f"{'='*40}")

        raw_results = await run_benchmark(
            endpoint=endpoint,
            model=model,
            prompts=prompts,
            num_requests=num_requests,
            concurrency=concurrency,
            api_key=api_key,
            max_tokens=256,
        )

        results[precision] = compute_summary(raw_results)

        # Brief pause between benchmarks to avoid interference
        if len(endpoints) > 1:
            print("  Cooling down for 5 seconds...")
            await asyncio.sleep(5)

    return results


# Default prompts for comparison
DEFAULT_PROMPTS = [
    "Explain the concept of database sharding and when you would use it.",
    "Write a Python class that implements a thread-safe LRU cache.",
    "What are the tradeoffs between microservices and monolithic architectures?",
    "Describe how TLS 1.3 handshake works, step by step.",
    "Write a SQL query to find all products that have never been ordered.",
]


def main():
    parser = argparse.ArgumentParser(description="Compare model quantization levels")
    parser.add_argument("--fp16-endpoint", default=None, help="Endpoint for fp16 model")
    parser.add_argument("--fp8-endpoint", default=None, help="Endpoint for fp8 model")
    parser.add_argument("--fp4-endpoint", default=None, help="Endpoint for fp4 model")
    parser.add_argument("--model", default="meta-llama/Llama-3.1-70B-Instruct")
    parser.add_argument("--model-size", type=float, default=70, help="Model size in billions")
    parser.add_argument("--gpu", default="H100", help="GPU type for cost comparison")
    parser.add_argument("--requests", type=int, default=50)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--prompt-file", default=None)
    parser.add_argument("--cost-only", action="store_true", help="Only run cost comparison (no live benchmark)")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    # Always show cost comparison
    run_cost_comparison(args.model_size, args.gpu)

    if args.cost_only:
        return

    # Check if any endpoints are provided
    endpoints = {}
    if args.fp16_endpoint:
        endpoints["fp16"] = args.fp16_endpoint
    if args.fp8_endpoint:
        endpoints["fp8"] = args.fp8_endpoint
    if args.fp4_endpoint:
        endpoints["fp4"] = args.fp4_endpoint

    if not endpoints:
        print("  No endpoints provided. Use --fp16-endpoint, --fp8-endpoint, --fp4-endpoint")
        print("  to benchmark live deployments, or use --cost-only for cost comparison only.")
        return

    # Load prompts
    if args.prompt_file:
        prompts = []
        with open(args.prompt_file) as f:
            for line in f:
                data = json.loads(line)
                prompts.append(data.get("prompt", data.get("text", "")))
    else:
        prompts = DEFAULT_PROMPTS

    api_key = os.environ.get("BASETEN_API_KEY")

    # Run live benchmarks
    results = asyncio.run(run_live_comparison(
        endpoints=endpoints,
        model=args.model,
        prompts=prompts,
        num_requests=args.requests,
        concurrency=args.concurrency,
        api_key=api_key,
    ))

    # Print comparison
    print_quantization_comparison(results, args.model)

    # Save results
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Results saved to {args.output}")


if __name__ == "__main__":
    main()
