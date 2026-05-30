# 04 — The POC Playbook

> This is the operational guide for running a proof-of-concept as a
> Solutions Architect. This is your core deliverable — proving to a
> customer that Baseten delivers better throughput, lower latency,
> and/or lower cost than their current setup.

---

## POC Philosophy

A POC is not a demo. It's a controlled experiment with measurable outcomes.

**What you're proving**: "On YOUR model, with YOUR prompts, at YOUR scale,
Baseten delivers X% better throughput / Y% lower latency / Z% lower cost."

**What makes a POC credible**:
1. Uses the customer's actual model (not a similar one)
2. Uses the customer's actual prompts (or a representative sample)
3. Matches the customer's actual traffic patterns
4. Compares apples-to-apples (same quality/accuracy bar)
5. Shows tradeoffs, not just the best case

---

## Phase 1: Discovery & Baseline (Day 1)

### Gathering the customer's baseline

Before you can show improvement, you need to know where they are:

```python
# Questions to answer:
baseline = {
    # Model
    "model": "meta-llama/Llama-3.1-70B-Instruct",
    "current_precision": "fp16",
    "serving_framework": "vLLM 0.4.2",

    # Hardware
    "current_gpu": "A100 80GB",
    "gpu_count": 2,  # TP=2
    "current_cost_per_hour": 8.00,  # 2x A100

    # Performance (ask for these or help them measure)
    "current_ttft_p50_ms": 450,
    "current_ttft_p95_ms": 1200,
    "current_throughput_tokens_sec": 2800,
    "current_tpot_ms": 35,

    # Workload
    "avg_input_tokens": 1500,
    "avg_output_tokens": 300,
    "peak_qps": 50,
    "avg_qps": 15,

    # Requirements
    "latency_sla_p95_ms": 1000,   # "p95 TTFT must be under 1s"
    "min_throughput_tokens_sec": 3000,
    "monthly_budget": 15000,
    "quality_bar": "no degradation on our eval suite",
}
```

### If they don't have baseline numbers:

Many customers don't have precise metrics. Help them measure:

```bash
# Quick baseline measurement script (run against their current endpoint)
python scripts/benchmark.py \
    --endpoint "http://their-vllm-server:8000/v1" \
    --model "meta-llama/Llama-3.1-70B-Instruct" \
    --prompt-file customer_prompts.jsonl \
    --concurrency 16 \
    --requests 500 \
    --output baseline_results.json
```

**This is a trust-building moment**: You're helping them understand their
current system, even before they commit to Baseten. Provide value early.

---

## Phase 2: Deploy on Baseten (Day 1-2)

### Step 1: Initialize the Truss project

```bash
pip install --upgrade truss
uvx truss login
uvx truss init customer-poc
cd customer-poc
```

### Step 2: Configure the model

For a standard model (config-only deployment):

```yaml
# config.yaml
model_name: meta-llama/Llama-3.1-70B-Instruct

runtime:
  predict_concurrency: 48

resources:
  accelerator: H100:1
  use_gpu: true

trt_llm:
  build:
    base_model: llama
    quantization_type: fp8
    max_seq_len: 8192
    max_batch_size: 64
  serve:
    engine_repository: default
    tensor_parallel_count: 1
```

### Step 3: Deploy

```bash
uvx truss push
# Wait for build (TRT-LLM compilation takes 15-30 min)
# You get a deployment URL when done
```

### Step 4: Verify functional correctness

```python
# Quick sanity check — same prompt, compare outputs
from openai import OpenAI

# Hit both endpoints with identical prompts
baseline_client = OpenAI(base_url="http://their-server:8000/v1", api_key="x")
baseten_client = OpenAI(
    base_url="https://bridge.baseten.co/v1/direct",
    api_key=os.environ["BASETEN_API_KEY"]
)

prompt = "Explain quantum computing in 3 sentences."

baseline_response = baseline_client.chat.completions.create(
    model="meta-llama/Llama-3.1-70B-Instruct",
    messages=[{"role": "user", "content": prompt}]
)

baseten_response = baseten_client.chat.completions.create(
    model="meta-llama/Llama-3.1-70B-Instruct",
    messages=[{"role": "user", "content": prompt}]
)

# Compare: outputs should be equivalent quality
# (Not identical — different implementations may sample differently)
```

---

## Phase 3: Optimization Sweep (Day 2-4)

### Sweep 1: Quantization

Deploy the same model at multiple precision levels and benchmark each:

```bash
# Run the quantization comparison script
python scripts/compare_quantizations.py \
    --model "meta-llama/Llama-3.1-70B-Instruct" \
    --gpu H100 \
    --prompt-file customer_prompts.jsonl \
    --output quant_comparison.json
```

Expected results pattern:
```
┌───────────┬──────────┬────────────┬──────────────┬────────────┐
│ Precision │ TTFT p50 │ Tokens/sec │ Cost/1M tok  │ Quality    │
├───────────┼──────────┼────────────┼──────────────┼────────────┤
│ fp16      │ 380ms    │ 3,200      │ $0.56        │ Baseline   │
│ fp8       │ 280ms    │ 4,800      │ $0.38        │ -0.3% acc  │
│ fp4       │ 210ms    │ 6,100      │ $0.30        │ -2.1% acc  │
└───────────┴──────────┴────────────┴──────────────┴────────────┘
```

**The conversation**: "fp8 gives you 50% more throughput at 32% lower
cost per token, with essentially no quality loss. fp4 pushes further
but we should evaluate quality on your specific use case."

### Sweep 2: Hardware

Deploy on different GPUs:

```bash
python scripts/benchmark.py --gpu A100 --results a100_results.json
python scripts/benchmark.py --gpu H100 --results h100_results.json
python scripts/benchmark.py --gpu B200 --results b200_results.json
```

### Sweep 3: Concurrency

Find the optimal concurrency target:

```bash
for concurrency in 1 4 8 16 32 64 128; do
    python scripts/benchmark.py \
        --concurrency $concurrency \
        --output "concurrency_${concurrency}.json"
done
```

This produces a latency-vs-throughput curve:

```
Throughput (tokens/sec)
    │
8K  │                          ●────── saturation (GPU maxed)
    │                     ●
6K  │                ●
    │           ●
4K  │      ●              ← sweet spot (good throughput, acceptable latency)
    │  ●
2K  │●
    │
    └──────────────────────────────
    0   4   8   16  32  64  128
         Concurrency Target

TTFT p95 (ms)
    │
2000│                              ●  ← unacceptable for chat
    │                         ●
1500│
    │                    ●
1000│── ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─  ← customer's SLA line
    │               ●
 500│      ●   ●
    │  ●
    └──────────────────────────────
    0   4   8   16  32  64  128
         Concurrency Target

Optimal concurrency = highest throughput where p95 TTFT < SLA
In this example: concurrency_target = 32
```

### Sweep 4: Engine Features

Test Baseten-specific optimizations:

```yaml
# Lookahead decoding (for structured output)
trt_llm:
  serve:
    lookahead_windows_size: 3  # Try 2, 3, 4, 5

# DFlash (custom attention)
# Enabled automatically by engine-builder for supported models
```

---

## Phase 4: Production Load Test (Day 5-6)

### Simulate real traffic

Don't just blast requests — simulate the actual traffic pattern:

```python
# production_load_test.py
import asyncio
import random
import time

async def simulate_production_traffic(
    endpoint: str,
    duration_minutes: int = 30,
    avg_qps: float = 15,
    burst_qps: float = 50,
    burst_probability: float = 0.1,  # 10% of seconds have burst
):
    """Simulate realistic production traffic patterns."""
    results = []
    end_time = time.time() + (duration_minutes * 60)

    while time.time() < end_time:
        # Determine this second's QPS
        if random.random() < burst_probability:
            current_qps = burst_qps
        else:
            current_qps = avg_qps * random.uniform(0.5, 1.5)

        # Send requests for this second
        tasks = []
        for _ in range(int(current_qps)):
            prompt = random.choice(customer_prompts)
            tasks.append(send_request(endpoint, prompt))

        batch_results = await asyncio.gather(*tasks)
        results.extend(batch_results)

        await asyncio.sleep(1)

    return results
```

### Capture comprehensive metrics

```python
# For each request, capture:
result = {
    "timestamp": time.time(),
    "input_tokens": len(prompt_tokens),
    "output_tokens": len(output_tokens),
    "ttft_ms": first_token_time - request_start_time,
    "total_latency_ms": end_time - request_start_time,
    "tpot_ms": (end_time - first_token_time) / output_tokens,
    "status_code": response.status_code,
    "error": None,
}
```

---

## Phase 5: Generate the Report (Day 7)

### Report structure:

```markdown
# POC Results: [Customer Name]
## Date: [Date]
## Model: [Model Name]

### Executive Summary
- [Key finding 1]: e.g., "43% higher throughput at 28% lower cost"
- [Key finding 2]: e.g., "p95 TTFT reduced from 1200ms to 380ms"
- [Key finding 3]: e.g., "fp8 quantization shows <0.5% quality impact"

### Baseline (Customer's Current Setup)
| Metric | Value |
|--------|-------|
| Hardware | 2× A100 80GB (TP=2) |
| Framework | vLLM 0.4.2 |
| Precision | fp16 |
| TTFT p50 | 450ms |
| TTFT p95 | 1,200ms |
| Throughput | 2,800 tok/s |
| Cost/hr | $8.00 |
| Cost/1M tokens | $0.79 |

### Baseten Optimized
| Metric | Value | vs Baseline |
|--------|-------|-------------|
| Hardware | 1× H100 80GB | Fewer GPUs |
| Engine | TRT-LLM + DFlash | Compiled + custom kernel |
| Precision | fp8 | Lower memory, same quality |
| TTFT p50 | 180ms | -60% |
| TTFT p95 | 380ms | -68% |
| Throughput | 4,800 tok/s | +71% |
| Cost/hr | $6.50 | -19% |
| Cost/1M tokens | $0.38 | -52% |

### Quality Evaluation
[Show results on customer's eval suite at each precision level]

### Scaling Behavior
[Show autoscaling response to traffic bursts]
[Show cost at different traffic levels]

### Recommendation
[Specific configuration recommendation with rationale]

### Monthly Cost Projection
| Traffic Level | Current | Baseten | Savings |
|---------------|---------|---------|---------|
| Current (15 QPS avg) | $5,760 | $4,680 | 19% |
| 2x growth | $11,520 | $6,500 | 44% |
| 5x growth | $28,800 | $13,000 | 55% |
```

---

## Phase 6: Present & Close (Day 8-10)

### Presentation tips:

1. **Lead with the metric they asked about**
   - If they said "we need lower latency" → lead with TTFT improvement
   - If they said "we're spending too much" → lead with cost/token improvement
   - If they said "we can't scale" → lead with autoscaling demo

2. **Show the dashboard live**
   - Let them see real-time metrics during a load test
   - This is more convincing than any slide deck

3. **Address the "what if" questions**:
   - "What if our traffic doubles?" → Show scaling projections
   - "What about vendor lock-in?" → OpenAI-compatible API, Truss is open-source
   - "What about cold starts?" → Show BDN caching, min replicas option
   - "What about model updates?" → Show versioned deploys, instant rollback
   - "What about data privacy?" → SOC2, HIPAA, self-hosted option

4. **Leave them with the numbers**
   - The report should stand alone — shareable with their leadership
   - Include the raw benchmark data so they can verify

---

## Common POC Pitfalls

### 1. Comparing different conditions
**Wrong**: Benchmarking Baseten with fp8 vs customer's fp16 without noting
the precision difference. They'll call it out.
**Right**: Show fp16 vs fp16 first (apples-to-apples), THEN show fp8 as
an additional optimization with quality evaluation.

### 2. Not matching traffic patterns
**Wrong**: Blasting 1000 concurrent requests when customer's peak is 50.
**Right**: Match their actual QPS, burst patterns, and input distributions.

### 3. Ignoring cold starts
**Wrong**: Only benchmarking warm replicas.
**Right**: Include cold start time in p99 metrics, or clearly state
"these metrics assume min_replicas ≥ 1."

### 4. Cherry-picking metrics
**Wrong**: "Our p50 is better!" when their p95 is worse.
**Right**: Show all percentiles. If p95 is worse, explain why and show
how to fix it (lower concurrency target, more replicas).

### 5. Forgetting quality evaluation
**Wrong**: "We're 50% faster!" without verifying output quality.
**Right**: Run the customer's eval suite at each precision level.
Show quality alongside performance.

---

## POC Checklist

```
□ Baseline metrics captured (TTFT, throughput, cost, quality)
□ Model deployed on Baseten (functionally verified)
□ Quantization sweep completed (fp16, fp8, fp4)
□ Hardware sweep completed (A100, H100, B200 as applicable)
□ Concurrency sweep completed (latency vs throughput curve)
□ Engine optimizations tested (TRT-LLM, DFlash, lookahead)
□ Production load test run (30+ min, realistic traffic)
□ Quality evaluation done at recommended precision
□ Report generated with side-by-side comparison
□ Cost projections at current and projected traffic
□ Presentation delivered
□ Follow-up questions addressed
□ Migration plan provided
```
