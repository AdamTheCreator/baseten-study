# 05 — Scripting for SAs

> Scripting was called out as an emphasis for this role. As an SA, you write
> scripts to automate POCs, benchmark models, collect metrics, and generate
> customer-facing reports. This section covers the patterns and tools you
> need to know.

---

## Why Scripting Matters for This Role

An SA at Baseten isn't just presenting slides — you're running hands-on
POCs that produce concrete performance data. That means:

1. **Deploying models programmatically** (Truss CLI + Python SDK)
2. **Benchmarking endpoints** (load testing, metrics collection)
3. **Analyzing results** (statistics, percentiles, cost calculations)
4. **Generating reports** (tables, charts, customer-facing output)
5. **Automating repetitive POC workflows** (sweep configs, compare results)

The SA who can script a full POC in an afternoon delivers results faster
than one who clicks through dashboards manually.

---

## Core Python Skills Needed

### 1. Async HTTP Requests (aiohttp / httpx)

Load testing requires sending many concurrent requests:

```python
import asyncio
import httpx
import time

async def send_request(client: httpx.AsyncClient, endpoint: str, payload: dict) -> dict:
    """Send a single inference request and capture timing."""
    start = time.perf_counter()
    first_token_time = None
    output_chunks = []

    # Streaming request to measure TTFT
    async with client.stream("POST", endpoint, json=payload) as response:
        async for chunk in response.aiter_text():
            if first_token_time is None:
                first_token_time = time.perf_counter()
            output_chunks.append(chunk)

    end = time.perf_counter()

    return {
        "ttft_ms": (first_token_time - start) * 1000 if first_token_time else None,
        "total_ms": (end - start) * 1000,
        "status": response.status_code,
        "output_length": len("".join(output_chunks)),
    }


async def run_load_test(endpoint: str, prompts: list, concurrency: int = 16):
    """Run concurrent requests and collect metrics."""
    semaphore = asyncio.Semaphore(concurrency)
    results = []

    async def bounded_request(client, prompt):
        async with semaphore:
            payload = {
                "messages": [{"role": "user", "content": prompt}],
                "model": "meta-llama/Llama-3.1-70B-Instruct",
                "stream": True,
            }
            result = await send_request(client, endpoint, payload)
            results.append(result)

    async with httpx.AsyncClient(timeout=120) as client:
        tasks = [bounded_request(client, p) for p in prompts]
        await asyncio.gather(*tasks)

    return results
```

### 2. Statistics and Percentile Calculations

```python
import numpy as np

def compute_metrics(results: list[dict]) -> dict:
    """Compute summary statistics from benchmark results."""
    ttfts = [r["ttft_ms"] for r in results if r["ttft_ms"] is not None]
    totals = [r["total_ms"] for r in results]
    errors = sum(1 for r in results if r["status"] != 200)

    return {
        "total_requests": len(results),
        "error_count": errors,
        "error_rate": errors / len(results),

        "ttft_p50_ms": np.percentile(ttfts, 50),
        "ttft_p90_ms": np.percentile(ttfts, 90),
        "ttft_p95_ms": np.percentile(ttfts, 95),
        "ttft_p99_ms": np.percentile(ttfts, 99),
        "ttft_mean_ms": np.mean(ttfts),

        "latency_p50_ms": np.percentile(totals, 50),
        "latency_p95_ms": np.percentile(totals, 95),
        "latency_p99_ms": np.percentile(totals, 99),

        "throughput_rps": len(results) / (max(totals) / 1000),
    }
```

### 3. OpenAI-Compatible API Usage

Baseten exposes an OpenAI-compatible API. Know how to use it:

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://bridge.baseten.co/v1/direct",
    api_key=os.environ["BASETEN_API_KEY"],
)

# Standard completion
response = client.chat.completions.create(
    model="meta-llama/Llama-3.1-70B-Instruct",
    messages=[{"role": "user", "content": "Hello"}],
    max_tokens=256,
    temperature=0.7,
)

# Streaming (for TTFT measurement)
stream = client.chat.completions.create(
    model="meta-llama/Llama-3.1-70B-Instruct",
    messages=[{"role": "user", "content": "Hello"}],
    stream=True,
)

first_token_time = None
for chunk in stream:
    if first_token_time is None:
        first_token_time = time.perf_counter()
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

### 4. Truss Programmatic Deployment

```python
import truss
import yaml

def deploy_model(
    model_name: str,
    gpu: str = "H100",
    quantization: str = "fp8",
    concurrency: int = 48,
):
    """Deploy a model on Baseten with specified configuration."""

    config = {
        "model_name": model_name,
        "runtime": {"predict_concurrency": concurrency},
        "resources": {
            "accelerator": gpu,
            "use_gpu": True,
        },
        "trt_llm": {
            "build": {
                "base_model": "llama",
                "quantization_type": quantization,
                "max_seq_len": 8192,
            }
        }
    }

    # Write config
    with open("config.yaml", "w") as f:
        yaml.dump(config, f)

    # Deploy via CLI
    import subprocess
    result = subprocess.run(
        ["uvx", "truss", "push"],
        capture_output=True, text=True
    )

    return result.stdout
```

### 5. Data Visualization (matplotlib)

```python
import matplotlib.pyplot as plt

def plot_latency_vs_throughput(sweep_results: dict):
    """Plot the latency-throughput tradeoff curve."""
    concurrencies = sorted(sweep_results.keys())
    ttfts = [sweep_results[c]["ttft_p95_ms"] for c in concurrencies]
    throughputs = [sweep_results[c]["throughput_rps"] for c in concurrencies]

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.set_xlabel("Concurrency Target")
    ax1.set_ylabel("TTFT p95 (ms)", color="red")
    ax1.plot(concurrencies, ttfts, "r-o", label="TTFT p95")

    ax2 = ax1.twinx()
    ax2.set_ylabel("Throughput (req/s)", color="blue")
    ax2.plot(concurrencies, throughputs, "b-s", label="Throughput")

    plt.title("Latency vs Throughput Tradeoff")
    fig.legend(loc="upper left", bbox_to_anchor=(0.15, 0.85))
    plt.savefig("latency_throughput.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_comparison_bar(baseline: dict, optimized: dict):
    """Side-by-side bar chart comparing baseline vs Baseten."""
    metrics = ["TTFT p50", "TTFT p95", "Cost/1M tok"]
    baseline_vals = [
        baseline["ttft_p50_ms"],
        baseline["ttft_p95_ms"],
        baseline["cost_per_million_tokens"],
    ]
    optimized_vals = [
        optimized["ttft_p50_ms"],
        optimized["ttft_p95_ms"],
        optimized["cost_per_million_tokens"],
    ]

    x = range(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([i - width/2 for i in x], baseline_vals, width, label="Current", color="#ff6b6b")
    ax.bar([i + width/2 for i in x], optimized_vals, width, label="Baseten", color="#4ecdc4")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.set_title("Current vs Baseten Optimized")
    plt.savefig("comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
```

---

## Core Bash Skills Needed

### Quick health checks

```bash
#!/bin/bash
# health_check.sh — Quick endpoint latency check

ENDPOINT="${1:-https://bridge.baseten.co/v1/direct}"
API_KEY="${BASETEN_API_KEY}"

echo "=== Endpoint Health Check ==="
echo "Target: $ENDPOINT"
echo ""

# Simple latency test (5 requests)
for i in $(seq 1 5); do
    START=$(python3 -c "import time; print(time.time())")

    RESPONSE=$(curl -s -w "\n%{http_code}\n%{time_total}" \
        -X POST "$ENDPOINT/chat/completions" \
        -H "Authorization: Bearer $API_KEY" \
        -H "Content-Type: application/json" \
        -d '{
            "model": "meta-llama/Llama-3.1-70B-Instruct",
            "messages": [{"role": "user", "content": "Say hello"}],
            "max_tokens": 10
        }')

    HTTP_CODE=$(echo "$RESPONSE" | tail -2 | head -1)
    TOTAL_TIME=$(echo "$RESPONSE" | tail -1)

    echo "Request $i: HTTP $HTTP_CODE, ${TOTAL_TIME}s total"
done
```

### Deployment automation

```bash
#!/bin/bash
# deploy_sweep.sh — Deploy same model with different configs

MODEL="meta-llama/Llama-3.1-70B-Instruct"
CONFIGS=("fp16:H100:1" "fp8:H100:1" "fp4:H100:1" "fp8:B200:1" "fp8:H100:2")

for config in "${CONFIGS[@]}"; do
    IFS=':' read -r quant gpu count <<< "$config"
    echo "=== Deploying: $quant on ${count}x $gpu ==="

    # Generate config.yaml
    cat > config.yaml << EOF
model_name: $MODEL
runtime:
  predict_concurrency: 48
resources:
  accelerator: "${gpu}:${count}"
  use_gpu: true
trt_llm:
  build:
    base_model: llama
    quantization_type: $quant
    max_seq_len: 8192
EOF

    uvx truss push 2>&1 | tail -5
    echo "---"
done
```

### Log analysis

```bash
#!/bin/bash
# analyze_logs.sh — Parse benchmark results

RESULTS_DIR="${1:-./benchmarks}"

echo "=== Benchmark Summary ==="
echo ""

for f in "$RESULTS_DIR"/*.json; do
    name=$(basename "$f" .json)
    echo "--- $name ---"

    # Extract key metrics with python
    python3 -c "
import json
with open('$f') as fh:
    data = json.load(fh)
metrics = data.get('metrics', data)
print(f\"  TTFT p50:  {metrics.get('ttft_p50_ms', 'N/A'):.1f}ms\")
print(f\"  TTFT p95:  {metrics.get('ttft_p95_ms', 'N/A'):.1f}ms\")
print(f\"  Throughput: {metrics.get('throughput_rps', 'N/A'):.1f} req/s\")
print(f\"  Errors:    {metrics.get('error_count', 0)}\")
"
    echo ""
done
```

---

## Scripting Patterns for Common SA Tasks

### Pattern 1: A/B Comparison Script

Compare two endpoints head-to-head:

```python
async def ab_compare(
    endpoint_a: str,  # Customer's current
    endpoint_b: str,  # Baseten
    prompts: list[str],
    concurrency: int = 16,
) -> dict:
    """Run identical load against two endpoints and compare."""
    print("Running against Endpoint A (baseline)...")
    results_a = await run_load_test(endpoint_a, prompts, concurrency)
    metrics_a = compute_metrics(results_a)

    print("Running against Endpoint B (Baseten)...")
    results_b = await run_load_test(endpoint_b, prompts, concurrency)
    metrics_b = compute_metrics(results_b)

    # Calculate improvements
    comparison = {}
    for key in metrics_a:
        if isinstance(metrics_a[key], (int, float)) and metrics_a[key] > 0:
            pct_change = ((metrics_b[key] - metrics_a[key]) / metrics_a[key]) * 100
            comparison[key] = {
                "baseline": metrics_a[key],
                "baseten": metrics_b[key],
                "change_pct": pct_change,
                "improved": pct_change < 0 if "latency" in key or "ttft" in key
                           else pct_change > 0,
            }

    return comparison
```

### Pattern 2: Cost Calculator

```python
def calculate_cost(
    gpu: str,
    gpu_count: int,
    replicas: int,
    hours_per_day: float = 24,
    days_per_month: int = 30,
) -> dict:
    """Calculate monthly infrastructure cost."""

    gpu_rates = {  # $/minute
        "T4": 0.01052,
        "L4": 0.01414,
        "A100": 0.06667,
        "H100": 0.10833,
        "B200": 0.16633,
    }

    rate_per_min = gpu_rates[gpu] * gpu_count
    daily_cost = rate_per_min * 60 * hours_per_day * replicas
    monthly_cost = daily_cost * days_per_month

    return {
        "gpu": gpu,
        "gpu_count": gpu_count,
        "replicas": replicas,
        "rate_per_hour": rate_per_min * 60,
        "daily_cost": daily_cost,
        "monthly_cost": monthly_cost,
        "annual_cost": monthly_cost * 12,
    }


def cost_per_token(
    gpu_cost_per_hour: float,
    tokens_per_second: float,
) -> float:
    """Calculate cost per million tokens."""
    tokens_per_hour = tokens_per_second * 3600
    return (gpu_cost_per_hour / tokens_per_hour) * 1_000_000
```

### Pattern 3: Prompt Distribution Analysis

Understanding the customer's prompt distribution helps you tune:

```python
def analyze_prompt_distribution(prompts_file: str) -> dict:
    """Analyze input/output token lengths from a prompt file."""
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")

    input_lengths = []
    for line in open(prompts_file):
        prompt = json.loads(line)["prompt"]
        input_lengths.append(len(enc.encode(prompt)))

    return {
        "count": len(input_lengths),
        "input_tokens_mean": np.mean(input_lengths),
        "input_tokens_median": np.median(input_lengths),
        "input_tokens_p95": np.percentile(input_lengths, 95),
        "input_tokens_max": max(input_lengths),
        "recommendation": (
            "Short prompts (<512 tokens) — TTFT will be fast, "
            "focus optimization on decode throughput"
            if np.median(input_lengths) < 512
            else "Long prompts (>512 tokens) — TTFT is the bottleneck, "
                 "prioritize compute (H100/B200) and consider prompt caching"
        ),
    }
```

### Pattern 4: Automated Report Generation

```python
from tabulate import tabulate

def generate_poc_report(
    customer_name: str,
    baseline: dict,
    baseten_results: dict,
    output_file: str = "poc_report.md",
):
    """Generate a customer-facing POC report."""

    report = f"""# POC Results: {customer_name}
## Generated: {datetime.now().strftime('%Y-%m-%d')}

### Executive Summary

| Metric | Current | Baseten | Improvement |
|--------|---------|---------|-------------|
| TTFT p50 | {baseline['ttft_p50_ms']:.0f}ms | {baseten_results['ttft_p50_ms']:.0f}ms | {improvement(baseline['ttft_p50_ms'], baseten_results['ttft_p50_ms'])} |
| TTFT p95 | {baseline['ttft_p95_ms']:.0f}ms | {baseten_results['ttft_p95_ms']:.0f}ms | {improvement(baseline['ttft_p95_ms'], baseten_results['ttft_p95_ms'])} |
| Throughput | {baseline['throughput_rps']:.0f} req/s | {baseten_results['throughput_rps']:.0f} req/s | {improvement(baseline['throughput_rps'], baseten_results['throughput_rps'], higher_is_better=True)} |
| Monthly Cost | ${baseline['monthly_cost']:,.0f} | ${baseten_results['monthly_cost']:,.0f} | {improvement(baseline['monthly_cost'], baseten_results['monthly_cost'])} |
"""

    with open(output_file, "w") as f:
        f.write(report)

    print(f"Report written to {output_file}")


def improvement(old: float, new: float, higher_is_better: bool = False) -> str:
    pct = ((new - old) / old) * 100
    if higher_is_better:
        return f"+{pct:.0f}%" if pct > 0 else f"{pct:.0f}%"
    else:
        return f"{pct:.0f}%" if pct < 0 else f"+{pct:.0f}%"
```

---

## Tools You Should Know

| Tool | What For | Install |
|------|----------|---------|
| `truss` | Deploy models to Baseten | `pip install truss` |
| `openai` SDK | Call Baseten endpoints (OpenAI-compatible) | `pip install openai` |
| `httpx` | Async HTTP for load testing | `pip install httpx` |
| `numpy` | Statistics (percentiles, means) | `pip install numpy` |
| `matplotlib` | Charts for reports | `pip install matplotlib` |
| `tabulate` | Pretty-print tables | `pip install tabulate` |
| `tiktoken` | Count tokens for prompt analysis | `pip install tiktoken` |
| `jq` | Parse JSON in bash | `brew install jq` |
| `curl` | Quick endpoint testing | Built-in |
| `wrk` / `hey` | HTTP load testing from CLI | `brew install wrk` / `brew install hey` |

---

## Interview Scripting Questions to Prepare For

1. **"Write a script to benchmark an endpoint and report p50/p95 TTFT"**
   → See the `run_load_test` + `compute_metrics` patterns above

2. **"How would you automate deploying the same model at different configs?"**
   → See `deploy_sweep.sh` — iterate over quantization/GPU combos

3. **"How do you measure TTFT for a streaming endpoint?"**
   → Time from request send to first SSE chunk received

4. **"Write a cost comparison between two GPU configurations"**
   → See `calculate_cost` + `cost_per_token` functions

5. **"How would you simulate realistic production traffic?"**
   → See `simulate_production_traffic` — variable QPS, burst patterns

6. **"How do you parse and analyze benchmark results?"**
   → numpy percentiles, matplotlib visualization, tabulate for reports
