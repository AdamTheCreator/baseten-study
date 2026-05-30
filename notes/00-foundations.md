# 00 — Inference Engineering Foundations

> Before you can optimize anything, you need to understand what's actually
> happening when a GPU generates tokens. This section covers the physics
> of inference — the concepts that underpin every conversation you'll have
> as a Solutions Architect.

---

## The Two Phases of LLM Inference

Every LLM request has two distinct computational phases. Understanding this
split is the single most important concept in inference engineering.

### Phase 1: Prefill (Processing the Input)

When a request arrives, the model must process the entire input prompt at once.
This is **compute-bound** — the GPU is doing massive matrix multiplications
across all input tokens simultaneously.

- All input tokens are processed in parallel (not one-by-one)
- Generates the KV cache (more on this below)
- Duration scales with input length
- This phase determines **TTFT (Time to First Token)**

**Why this matters for customers**: A chatbot with a 4,000-token system prompt
pays this cost on every single request. Prompt caching (reusing prefill results)
can cut TTFT dramatically for repeated prefixes.

### Phase 2: Decode (Generating the Output)

After prefill, the model generates tokens one at a time (autoregressively).
Each new token requires attending to all previous tokens via the KV cache.
This is **memory-bandwidth-bound** — the GPU spends most of its time reading
weights and KV cache from memory, not doing math.

- Tokens generated sequentially (each depends on the previous)
- Duration scales with output length
- GPU utilization is typically LOW during decode (10-30%)
- This phase determines **tokens/second** and **TPOT (Time Per Output Token)**

**Why this matters**: The decode phase is where batching helps enormously.
While one request is waiting on its next token, the GPU can work on another
request's token. This is why high concurrency improves throughput.

```
Request lifecycle:

[---- Prefill ----][--- Decode (token by token) ---]
                   ^                                ^
                   TTFT                          Total latency

TTFT = time until first token appears in stream
TPOT = average time between subsequent tokens
Total latency = TTFT + (output_tokens × TPOT)
```

---

## The KV Cache: Why Memory Matters More Than Compute

During prefill, the model computes **Key** and **Value** tensors for every
attention layer and every input token. These are cached in GPU memory so
decode doesn't have to recompute them.

### Why this matters:

- KV cache size scales with: `layers × heads × head_dim × seq_len × 2 (K+V) × precision`
- A 70B model with 4K context in fp16: ~40GB of KV cache per concurrent request
- **This is usually what limits concurrency**, not compute
- More concurrent requests = more KV cache = more GPU memory needed

### The practical implication:

A customer asks "why can I only run 8 concurrent requests on an H100?"
Answer: The model weights take 70GB (fp16 for a 70B model). The H100 has
80GB. That leaves 10GB for KV cache. At ~5GB per request's cache, you get
roughly 2 concurrent requests. This is why **quantization matters** — fp8
cuts weight memory in half, doubling your KV cache budget.

```
H100 80GB Memory Budget (70B model):
┌─────────────────────────────────────────────────────┐
│ Model Weights (fp16): ~140GB  ← DOESN'T FIT!       │
└─────────────────────────────────────────────────────┘

H100 80GB Memory Budget (70B model, fp8 quantized):
┌──────────────────────────┬──────────────────────────┐
│ Model Weights (fp8): 70GB│ KV Cache: ~10GB          │
│                          │ (~2-8 concurrent reqs)   │
└──────────────────────────┴──────────────────────────┘

B200 192GB Memory Budget (70B model, fp8 quantized):
┌──────────────────────────┬──────────────────────────┐
│ Model Weights (fp8): 70GB│ KV Cache: ~120GB         │
│                          │ (~24-48 concurrent reqs) │
└──────────────────────────┴──────────────────────────┘
```

---

## Key Metrics — What They Mean and When They Matter

### TTFT (Time to First Token)
- Measured from: request received → first token streamed back
- Includes: queue time + prefill computation
- Affected by: input length, model size, GPU compute power, queue depth
- **When it matters most**: Chat/conversational UIs, real-time applications
- **Typical targets**: <500ms for chat, <2s for batch processing
- **Baseten reports this as**: "Time to First Byte" in their metrics dashboard

### TPOT (Time Per Output Token)
- Measured from: time between consecutive output tokens
- Affected by: model size, GPU memory bandwidth, batch size
- **When it matters most**: Streaming UIs where users read as tokens appear
- **Typical targets**: <50ms/token for smooth reading experience

### Throughput (tokens/second)
- Can mean two things — clarify which:
  - **Per-request throughput**: tokens/sec for a single request
  - **System throughput**: total tokens/sec across all concurrent requests
- System throughput is what determines cost-per-token
- **This is usually the metric that justifies Baseten's cost**

### Latency Percentiles (p50, p95, p99)
- p50: median response time (what most users experience)
- p95: 1 in 20 requests is slower than this
- p99: 1 in 100 requests is slower than this
- **p95/p99 are what SLAs are written against**
- Large gap between p50 and p99 usually means autoscaling is too slow
  or cold starts are hitting some users

### GPU Utilization
- Percentage of GPU compute being used
- During prefill: often 80-95% (compute-bound)
- During decode: often 10-30% (memory-bandwidth-bound)
- **Low utilization during decode is normal and expected** — don't let
  a customer think the GPU is "wasted"
- Batching multiple requests improves utilization during decode

### Queue Depth
- Number of requests waiting for a GPU replica
- If consistently >0, need more replicas or higher concurrency
- Directly adds to TTFT (queue time is part of TTFT)

---

## Batching: The Key to Cost-Efficient Inference

### Why Batching Works

During decode, the GPU is mostly idle (waiting on memory reads). If you
batch multiple requests together, the GPU can process multiple tokens
in the time it would have spent on one. This is called **continuous batching**.

```
Without batching (concurrency=1):
GPU: [Req1-tok1][idle][Req1-tok2][idle][Req1-tok3]...
     ████░░░░░░████░░░░░░████░░░░░░

With batching (concurrency=32):
GPU: [R1-t1,R2-t1,R3-t1,...][R1-t2,R2-t2,R3-t2,...]
     ████████████████████████████████████████████████
     (GPU stays busy serving multiple requests)
```

### The Tradeoff

Higher batch size = higher system throughput = lower cost/token
Higher batch size = higher per-request latency (TPOT increases)

**This is the fundamental tension in inference engineering.**

As an SA, your job is to find the sweet spot for each customer:
- Chat app? Prioritize low TPOT (batch size 4-16)
- Batch document processing? Maximize throughput (batch size 64-256)
- Code completion? Low TTFT is critical (batch size 8-32)

---

## Quantization: Trading Precision for Speed

Quantization reduces the numerical precision of model weights:

| Precision | Bits | Memory | Quality | Speed   | Use Case |
|-----------|------|--------|---------|---------|----------|
| fp32      | 32   | 1x     | Best    | Slowest | Training only |
| fp16/bf16 | 16   | 0.5x   | Great   | Fast    | Default inference |
| fp8       | 8    | 0.25x  | Good    | Faster  | Production sweet spot |
| fp4/int4  | 4    | 0.125x | Decent  | Fastest | Cost-sensitive, high throughput |

### Why quantization improves speed (not just memory):

1. **Smaller weights = faster memory reads** (decode is memory-bandwidth-bound)
2. **Smaller weights = more KV cache room = higher batch size = better throughput**
3. **Smaller weights = fits on fewer/cheaper GPUs**

### Baseten's quantization options:
- `fp8` — FP8 weights, 16-bit KV cache (best quality/speed balance)
- `fp8_kv` — FP8 everything (more concurrent requests)
- `fp4` — 4-bit weights (maximum throughput, some quality loss)
- `fp4_kv` — 4-bit everything (aggressive, test quality carefully)

### The SA conversation:

Customer: "We need the best quality possible."
You: "Let's benchmark fp16 vs fp8 on your actual prompts. fp8 typically
shows <1% quality degradation on standard benchmarks but gives you 2x
the throughput. I'll run both and you can evaluate quality on your
specific use case."

---

## Continuous Batching vs Static Batching

**Static batching** (old approach): Collect N requests, process all at once,
return all at once. Problem: fastest request waits for slowest.

**Continuous batching** (what Baseten/vLLM/TRT-LLM use): New requests can
join a running batch at any iteration. Completed requests leave immediately.
No request waits for another.

This is a key differentiator from naive on-prem deployments that use
basic serving frameworks without continuous batching.

---

## Speculative Decoding (Lookahead)

Use a small "draft" model to predict several tokens ahead, then verify
with the full model in one pass. If predictions are correct (common for
structured outputs like JSON/code), you generate multiple tokens per
forward pass.

Baseten supports this as "lookahead decoding" — configurable window size.

**When to recommend**: Customer generates lots of structured output (JSON,
code, SQL). Less useful for creative/conversational text.

---

## Tensor Parallelism vs Pipeline Parallelism

When a model doesn't fit on one GPU, you split it across multiple:

**Tensor Parallelism (TP)**: Split each layer across GPUs. All GPUs work
on every token. Lower latency, but requires fast GPU-to-GPU interconnect
(NVLink). Common for inference.

**Pipeline Parallelism (PP)**: Different layers on different GPUs. Tokens
flow through GPUs sequentially. Higher latency per token, but works with
slower interconnects. Common for training.

**What you need to know as an SA**:
- 70B model in fp8 = ~35GB → fits on 1 H100 (80GB)
- 70B model in fp16 = ~140GB → needs 2x H100 with TP=2
- 405B model in fp8 = ~200GB → needs 4x H100 with TP=4 (or 2x B200)
- TP across nodes (different machines) is much slower than TP within
  a node (NVLink). Baseten handles this — on-prem, you'd configure it yourself.

---

## Summary: The Mental Model

```
Customer request arrives
        │
        ▼
   ┌─────────┐     Queue time adds to TTFT
   │  Queue   │     (autoscaling + concurrency target controls this)
   └────┬────┘
        │
        ▼
   ┌─────────┐     Prefill time = f(input_length, model_size, GPU_compute)
   │ Prefill  │     This is compute-bound. Faster GPU = faster prefill.
   └────┬────┘     KV cache generated here.
        │
        ▼
   ┌─────────┐     Decode time = f(output_length, model_size, GPU_bandwidth)
   │ Decode   │     This is memory-bound. Batching helps here.
   │ (token   │     Quantization helps here (smaller reads).
   │  by      │     More concurrent requests = better GPU utilization
   │  token)  │     but higher per-request latency.
   └────┬────┘
        │
        ▼
   Response complete
   Total cost = GPU_time × GPU_price_per_minute
   Cost/token = total_cost / tokens_generated
```

Every optimization lever maps to one of these stages. When a customer
says "it's slow," your first job is figuring out which stage is the
bottleneck: queue, prefill, or decode.
