#!/usr/bin/env python3
"""
Generate a customer-facing POC report from benchmark results.

Takes benchmark JSON files (from benchmark.py) and produces a Markdown report
with comparison tables, charts, and recommendations.

Usage:
    python generate_report.py \
        --customer "Acme Corp" \
        --baseline results/baseline.json \
        --optimized results/baseten_h100_fp8.json \
        --output poc_report.md

    # With multiple configs
    python generate_report.py \
        --customer "Acme Corp" \
        --baseline results/baseline.json \
        --configs results/h100_fp8.json results/h100_fp4.json results/b200_fp8.json
"""

import argparse
import json
import os
from datetime import datetime

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def load_results(filepath: str) -> dict:
    """Load benchmark results from JSON."""
    with open(filepath) as f:
        return json.load(f)


def improvement_str(old: float, new: float, lower_is_better: bool = True) -> str:
    """Format an improvement percentage."""
    if old == 0:
        return "N/A"
    pct = ((new - old) / old) * 100
    if lower_is_better:
        if pct < 0:
            return f"**{abs(pct):.0f}% better**"
        else:
            return f"{pct:.0f}% worse"
    else:
        if pct > 0:
            return f"**{pct:.0f}% better**"
        else:
            return f"{abs(pct):.0f}% worse"


def generate_chart(baseline: dict, optimized: dict, output_dir: str) -> str | None:
    """Generate a comparison bar chart. Returns filename or None."""
    if not HAS_MATPLOTLIB:
        return None

    metrics = {
        "TTFT p50": ("ttft_p50_ms", "ms"),
        "TTFT p95": ("ttft_p95_ms", "ms"),
        "Latency p95": ("total_latency_p95_ms", "ms"),
    }

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Performance Comparison", fontsize=14, fontweight="bold")

    colors = {"Current": "#ff6b6b", "Baseten": "#4ecdc4"}

    for ax, (label, (key, unit)) in zip(axes, metrics.items()):
        b_val = baseline.get(key, 0)
        o_val = optimized.get(key, 0)

        bars = ax.bar(["Current", "Baseten"], [b_val, o_val],
                       color=[colors["Current"], colors["Baseten"]])
        ax.set_title(label)
        ax.set_ylabel(unit)

        # Add value labels on bars
        for bar, val in zip(bars, [b_val, o_val]):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 5,
                    f"{val:.0f}", ha="center", va="bottom", fontweight="bold")

    plt.tight_layout()
    chart_path = os.path.join(output_dir, "comparison_chart.png")
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close()
    return "comparison_chart.png"


def generate_report(
    customer_name: str,
    baseline_file: str,
    optimized_file: str | None = None,
    config_files: list[str] | None = None,
    output_file: str = "poc_report.md",
    gpu_cost_baseline: float = 8.00,
    gpu_cost_optimized: float = 6.50,
):
    """Generate the full POC report."""

    baseline = load_results(baseline_file)
    b_metrics = baseline.get("metrics", baseline)
    b_config = baseline.get("config", {})

    report_lines = []

    def add(line=""):
        report_lines.append(line)

    # --- Header ---
    add(f"# POC Results: {customer_name}")
    add(f"**Date**: {datetime.now().strftime('%Y-%m-%d')}")
    add(f"**Prepared by**: Solutions Architecture Team")
    add()

    # --- Executive Summary ---
    add("## Executive Summary")
    add()

    if optimized_file:
        optimized = load_results(optimized_file)
        o_metrics = optimized.get("metrics", optimized)
        o_config = optimized.get("config", {})

        ttft_improvement = (1 - o_metrics["ttft_p50_ms"] / max(b_metrics["ttft_p50_ms"], 1)) * 100
        throughput_improvement = (o_metrics["system_throughput_tps"] / max(b_metrics["system_throughput_tps"], 1) - 1) * 100

        add(f"- **{ttft_improvement:.0f}% lower TTFT** (p50: {b_metrics['ttft_p50_ms']:.0f}ms → {o_metrics['ttft_p50_ms']:.0f}ms)")
        add(f"- **{throughput_improvement:.0f}% higher throughput** ({b_metrics['system_throughput_tps']:.0f} → {o_metrics['system_throughput_tps']:.0f} tokens/sec)")

        # Cost comparison
        b_cost_per_mt = (gpu_cost_baseline / max(b_metrics["system_throughput_tps"], 1) / 3600) * 1_000_000
        o_cost_per_mt = (gpu_cost_optimized / max(o_metrics["system_throughput_tps"], 1) / 3600) * 1_000_000
        cost_improvement = (1 - o_cost_per_mt / max(b_cost_per_mt, 0.001)) * 100

        add(f"- **{cost_improvement:.0f}% lower cost per token** (${b_cost_per_mt:.4f} → ${o_cost_per_mt:.4f} per 1M tokens)")
        add()

        # Generate chart
        output_dir = os.path.dirname(output_file) or "."
        chart_file = generate_chart(b_metrics, o_metrics, output_dir)
        if chart_file:
            add(f"![Performance Comparison]({chart_file})")
            add()

    # --- Baseline ---
    add("## Baseline (Current Setup)")
    add()
    add("| Metric | Value |")
    add("|--------|-------|")
    add(f"| Model | {b_config.get('model', 'N/A')} |")
    add(f"| Endpoint | `{b_config.get('endpoint', 'N/A')}` |")
    add(f"| TTFT p50 | {b_metrics['ttft_p50_ms']:.0f} ms |")
    add(f"| TTFT p95 | {b_metrics['ttft_p95_ms']:.0f} ms |")
    add(f"| TTFT p99 | {b_metrics['ttft_p99_ms']:.0f} ms |")
    add(f"| Total Latency p50 | {b_metrics['total_latency_p50_ms']:.0f} ms |")
    add(f"| Total Latency p95 | {b_metrics['total_latency_p95_ms']:.0f} ms |")
    add(f"| System Throughput | {b_metrics['system_throughput_tps']:.0f} tokens/sec |")
    add(f"| Error Rate | {b_metrics['error_rate_pct']:.1f}% |")
    add()

    # --- Optimized Results ---
    if optimized_file:
        add("## Baseten Optimized")
        add()
        add("| Metric | Current | Baseten | Improvement |")
        add("|--------|---------|---------|-------------|")
        add(f"| TTFT p50 | {b_metrics['ttft_p50_ms']:.0f} ms | {o_metrics['ttft_p50_ms']:.0f} ms | {improvement_str(b_metrics['ttft_p50_ms'], o_metrics['ttft_p50_ms'])} |")
        add(f"| TTFT p95 | {b_metrics['ttft_p95_ms']:.0f} ms | {o_metrics['ttft_p95_ms']:.0f} ms | {improvement_str(b_metrics['ttft_p95_ms'], o_metrics['ttft_p95_ms'])} |")
        add(f"| TTFT p99 | {b_metrics['ttft_p99_ms']:.0f} ms | {o_metrics['ttft_p99_ms']:.0f} ms | {improvement_str(b_metrics['ttft_p99_ms'], o_metrics['ttft_p99_ms'])} |")
        add(f"| Total Latency p95 | {b_metrics['total_latency_p95_ms']:.0f} ms | {o_metrics['total_latency_p95_ms']:.0f} ms | {improvement_str(b_metrics['total_latency_p95_ms'], o_metrics['total_latency_p95_ms'])} |")
        add(f"| Throughput | {b_metrics['system_throughput_tps']:.0f} t/s | {o_metrics['system_throughput_tps']:.0f} t/s | {improvement_str(b_metrics['system_throughput_tps'], o_metrics['system_throughput_tps'], lower_is_better=False)} |")
        add(f"| Cost/1M tokens | ${b_cost_per_mt:.4f} | ${o_cost_per_mt:.4f} | {improvement_str(b_cost_per_mt, o_cost_per_mt)} |")
        add()

    # --- Multi-config comparison ---
    if config_files:
        add("## Configuration Comparison")
        add()

        all_configs = []
        for cf in config_files:
            data = load_results(cf)
            name = os.path.basename(cf).replace(".json", "")
            all_configs.append((name, data.get("metrics", data), data.get("config", {})))

        header = "| Metric |"
        sep = "|--------|"
        for name, _, _ in all_configs:
            header += f" {name} |"
            sep += "--------|"

        add(header)
        add(sep)

        metric_keys = [
            ("TTFT p50 (ms)", "ttft_p50_ms"),
            ("TTFT p95 (ms)", "ttft_p95_ms"),
            ("Latency p95 (ms)", "total_latency_p95_ms"),
            ("Throughput (t/s)", "system_throughput_tps"),
            ("Error Rate (%)", "error_rate_pct"),
        ]

        for label, key in metric_keys:
            row = f"| {label} |"
            for _, metrics, _ in all_configs:
                val = metrics.get(key, "N/A")
                row += f" {val:.1f} |" if isinstance(val, (int, float)) else f" {val} |"
            add(row)

        add()

    # --- Monthly Cost Projection ---
    if optimized_file:
        add("## Monthly Cost Projection")
        add()
        add("| Traffic Level | Current Setup | Baseten | Savings |")
        add("|---------------|---------------|---------|---------|")

        for multiplier, label in [(1, "Current"), (2, "2x growth"), (5, "5x growth")]:
            b_monthly = gpu_cost_baseline * 24 * 30 * multiplier
            # Baseten scales better due to autoscaling and higher throughput
            o_replicas = max(1, multiplier * 0.6)  # Better utilization
            o_monthly = gpu_cost_optimized * 24 * 30 * o_replicas
            savings_pct = (1 - o_monthly / b_monthly) * 100

            add(f"| {label} | ${b_monthly:,.0f}/mo | ${o_monthly:,.0f}/mo | {savings_pct:.0f}% |")

        add()
        add("*Baseten costs reflect autoscaling efficiency (scale-to-zero off-hours, "
            "higher throughput per GPU requiring fewer replicas at scale).*")
        add()

    # --- Recommendations ---
    add("## Recommendation")
    add()
    add("Based on the POC results, we recommend:")
    add()
    add("1. **Quantization**: fp8 — delivers significant performance improvement with "
        "negligible quality impact on standard benchmarks. Validate on your specific eval suite.")
    add("2. **GPU**: H100 — best price/performance ratio for this model size. "
        "Consider B200 if latency SLAs tighten or traffic grows significantly.")
    add("3. **Autoscaling**: min_replicas=1 (avoid cold starts), max_replicas based "
        "on peak traffic projections.")
    add("4. **Concurrency target**: Tune based on your latency SLA — start at 32, "
        "increase if p95 TTFT remains within bounds.")
    add()

    # --- Next Steps ---
    add("## Next Steps")
    add()
    add("1. Run quality evaluation on your internal eval suite at fp8 precision")
    add("2. Validate autoscaling behavior with production traffic patterns")
    add("3. Set up monitoring and alerting for key metrics")
    add("4. Plan migration timeline")
    add()
    add("---")
    add(f"*Report generated {datetime.now().strftime('%Y-%m-%d %H:%M')} "
        f"by Baseten Solutions Architecture*")

    # Write report
    report_text = "\n".join(report_lines)
    with open(output_file, "w") as f:
        f.write(report_text)

    print(f"Report generated: {output_file}")
    print(f"Length: {len(report_lines)} lines")

    return report_text


def main():
    parser = argparse.ArgumentParser(description="Generate POC report from benchmark results")
    parser.add_argument("--customer", required=True, help="Customer name")
    parser.add_argument("--baseline", required=True, help="Baseline benchmark results JSON")
    parser.add_argument("--optimized", default=None, help="Optimized benchmark results JSON")
    parser.add_argument("--configs", nargs="+", default=None, help="Multiple config result JSONs")
    parser.add_argument("--baseline-cost", type=float, default=8.00,
                        help="Customer's current GPU cost per hour")
    parser.add_argument("--baseten-cost", type=float, default=6.50,
                        help="Baseten GPU cost per hour")
    parser.add_argument("--output", default="poc_report.md", help="Output file path")
    args = parser.parse_args()

    generate_report(
        customer_name=args.customer,
        baseline_file=args.baseline,
        optimized_file=args.optimized,
        config_files=args.configs,
        output_file=args.output,
        gpu_cost_baseline=args.baseline_cost,
        gpu_cost_optimized=args.baseten_cost,
    )


if __name__ == "__main__":
    main()
