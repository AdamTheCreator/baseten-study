# 03 — Hardware Selection Guide

> How GPU choice affects throughput, latency, cost-per-token, and ultimately
> what a customer selects. This is one of the most concrete things you
> advise on as an SA.

---

## GPU Lineup Available on Baseten

| GPU | VRAM | Memory BW | FP16 TFLOPS | FP8 TFLOPS | $/min | $/hr | Best For |
|-----|------|-----------|-------------|------------|-------|------|----------|
| T4 | 16 GB | 300 GB/s | 65 | N/A | $0.011 | $0.63 | Small models, embeddings |
| L4 | 24 GB | 300 GB/s | 121 | 242 | $0.014 | $0.85 | Small-medium models, image gen |
| A10G | 24 GB | 600 GB/s | 125 | N/A | ~$0.02 | ~$1.20 | Medium models, image gen |
| A100 | 80 GB | 2,039 GB/s | 312 | N/A | $0.067 | $4.00 | Large models, high batch |
| H100 | 80 GB | 3,350 GB/s | 990 | 1,979 | $0.108 | $6.50 | Production LLMs, best perf/$ |
| B200 | 192 GB | 8,000 GB/s | 2,250 | 4,500 | $0.166 | $9.98 | Largest models, max throughput |

---

## Understanding the Specs

### VRAM (Video Memory)
**What it determines**: Which models fit, and how many concurrent requests
you can serve.

```
Model fits? → Check: model_weights + KV_cache_per_request × max_concurrency ≤ VRAM

Examples:
- Llama 8B fp16 (~16GB weights):
  T4 (16GB): Fits, but zero room for KV cache. Unusable.
  L4 (24GB): Fits with ~8GB for KV cache. Low concurrency.
  A100 (80GB): Fits with 64GB for cache. High concurrency.

- Llama 70B fp8 (~35GB weights):
  T4/L4: Doesn't fit.
  A100 (80GB): Fits with ~45GB for cache. Medium concurrency.
  H100 (80GB): Fits with ~45GB for cache. But faster memory BW.
  B200 (192GB): Fits with ~157GB for cache. Very high concurrency.

- Llama 70B fp16 (~140GB weights):
  Single GPU: Doesn't fit anywhere.
  2× H100 (160GB): Fits with TP=2, ~20GB total for cache.
  2× B200 (384GB): Fits with TP=2, ~244GB total for cache.
  1× B200 (192GB): Fits with ~52GB for cache. NO tensor parallelism needed.
```

**Key insight**: B200's 192GB VRAM means models that required 2× H100 (with
expensive NVLink communication overhead) now fit on a single B200. Eliminating
tensor parallelism overhead can improve latency 20-30%.

### Memory Bandwidth
**What it determines**: Decode speed (tokens/second per request).

Decode is memory-bandwidth-bound. The GPU needs to read the entire model's
weights from memory for each token. Faster memory = faster token generation.

```
Theoretical max tokens/sec (single request, simplified):
= memory_bandwidth / model_size_in_bytes

Llama 70B fp8 (35GB weights):
- A100: 2,039 GB/s ÷ 35 GB = ~58 tokens/sec theoretical max
- H100: 3,350 GB/s ÷ 35 GB = ~96 tokens/sec theoretical max
- B200: 8,000 GB/s ÷ 35 GB = ~229 tokens/sec theoretical max

Real-world is lower (overhead, KV cache reads, etc.) but the ratios hold.
B200 decode is roughly 2.4x faster than H100 for memory-bound workloads.
```

### Compute (TFLOPS)
**What it determines**: Prefill speed (TTFT for long inputs) and batched
decode throughput.

Prefill is compute-bound. More TFLOPS = faster prefill = lower TTFT.

```
Relative prefill speed (simplified):
- A100 fp16: 312 TFLOPS (baseline)
- H100 fp16: 990 TFLOPS (3.2x faster prefill)
- H100 fp8:  1,979 TFLOPS (6.3x faster prefill)
- B200 fp8:  4,500 TFLOPS (14.4x faster prefill)
```

### FP8 Support
- T4, A10G: **No FP8 support**. Limited to fp16/int8.
- A100: **No native FP8**. Can emulate but not efficient.
- H100, B200: **Native FP8 Tensor Cores**. This is a game-changer.

FP8 on H100/B200 means:
- 2x less memory for weights (fit larger models / more concurrency)
- 2x more compute throughput (fp8 TFLOPS >> fp16 TFLOPS)
- Minimal quality degradation (<1% on most benchmarks)

**SA talking point**: "If the customer is on A100s, the move to H100 isn't
just a speed bump — it's access to FP8, which is a fundamentally different
price/performance tier."

---

## Decision Framework: Which GPU for Which Customer

### Small Models (< 10B parameters)
**Examples**: Llama 8B, Mistral 7B, Phi-3, embedding models

| GPU | Fit? | Concurrency | Cost | Recommendation |
|-----|------|-------------|------|----------------|
| T4 | Tight (fp16) | Very low | $0.63/hr | Only for embeddings |
| L4 | Yes | Low-medium | $0.85/hr | Good for low-traffic |
| A100 | Yes | Very high | $4.00/hr | Overkill unless high QPS |
| H100 | Yes | Very high | $6.50/hr | Only if TTFT-critical |

**Recommendation**: L4 for low traffic, A100 for high traffic.
H100 only justified if sub-100ms TTFT is required.

### Medium Models (10-40B parameters)
**Examples**: Llama 3.1 70B (quantized), Mixtral 8x7B, CodeLlama 34B

| GPU | Fit? | Concurrency | Cost | Recommendation |
|-----|------|-------------|------|----------------|
| L4 | fp4 only | Very low | $0.85/hr | Not recommended |
| A100 | fp8: yes | Medium | $4.00/hr | Budget option |
| H100 | fp8: yes | Medium-high | $6.50/hr | Best perf/$ |
| B200 | fp8: yes | Very high | $9.98/hr | High concurrency needs |

**Recommendation**: H100 is the sweet spot. A100 if cost-constrained.

### Large Models (70B+ parameters, full precision)
**Examples**: Llama 3.1 70B fp16, Llama 3.1 405B, DeepSeek V3

| GPU | Fit? | Setup | Cost | Recommendation |
|-----|------|-------|------|----------------|
| A100 | 2× for 70B fp16 | TP=2 | $8.00/hr | Possible but slow |
| H100 | 2× for 70B fp16 | TP=2 | $13.00/hr | Good for 70B |
| H100 | 8× for 405B fp8 | TP=8 | $52.00/hr | Only option pre-B200 |
| B200 | 1× for 70B fp16 | No TP! | $9.98/hr | Better than 2× H100 |
| B200 | 2× for 405B fp8 | TP=2 | $19.96/hr | Game-changer |

**Key insight**: B200 eliminates tensor parallelism for 70B models.
1× B200 ($9.98/hr) beats 2× H100 ($13.00/hr) on both cost AND latency
(no cross-GPU communication overhead).

### MoE Models (DeepSeek V3, Mixtral)
**Examples**: DeepSeek V3 (671B total, ~37B active), Mixtral 8x22B

MoE models are unique — they have huge total parameter counts but only
activate a subset (experts) per token. This means:
- Weights are large (need lots of VRAM to store all experts)
- Active compute is moderate (only some experts fire)
- Memory bandwidth is critical (reading expert weights)

Baseten's BIS-LLM engine is optimized for MoE with smart expert routing.

---

## H100 vs B200: The Customer Decision

This will be one of your most common conversations as an SA.

### H100 Advantages:
- **More available**: Easier to get, more regions
- **Lower unit cost**: $6.50/hr vs $9.98/hr
- **Well-understood**: Customers have benchmarks, engineers know it
- **Sufficient for most models**: 80GB fits 70B fp8 comfortably

### B200 Advantages:
- **2.4x memory bandwidth**: Directly translates to higher tokens/sec
- **2.4x VRAM**: 192GB vs 80GB — fits 70B fp16 on ONE GPU
- **2.3x compute**: Faster prefill, lower TTFT
- **Eliminates TP for 70B**: 1× B200 < 2× H100 in cost AND latency
- **Higher concurrency**: More KV cache room per GPU
- **Better cost/token at scale**: Despite higher $/hr, tokens/hr/$ is better

### When to recommend H100:
- Small/medium models that fit comfortably in 80GB
- Cost-sensitive customers who don't need peak performance
- Workloads where latency requirements are moderate
- Customers who want proven/stable hardware

### When to recommend B200:
- Large models (70B fp16, 405B)
- Latency-critical applications (real-time chat, code completion)
- High-concurrency workloads (need lots of KV cache)
- Cost optimization at scale (better tokens/$/hr math)
- Customer is currently using 2× H100 for TP (consolidate to 1× B200)

### The Cost Math (Example: Llama 70B fp8)

```
Scenario: 1000 requests/hr, avg 500 input + 200 output tokens

H100 (1 GPU):
- Throughput: ~4,000 tokens/sec system-wide at batch=32
- Tokens/hr: 14.4M
- Cost/hr: $6.50
- Cost per 1M tokens: $0.45
- Can handle 1000 req/hr? Yes (1000 × 700 = 700K tokens/hr << 14.4M)
- Replicas needed: 1 (with headroom)
- Monthly cost: ~$4,680

B200 (1 GPU):
- Throughput: ~9,500 tokens/sec system-wide at batch=64
- Tokens/hr: 34.2M
- Cost/hr: $9.98
- Cost per 1M tokens: $0.29
- Can handle 1000 req/hr? Yes, with massive headroom
- Replicas needed: 1
- Monthly cost: ~$7,186

At 1000 req/hr → H100 wins on total cost ($4,680 vs $7,186)

But at 5000 req/hr:
- H100: needs 2 replicas → $9,360/mo
- B200: still 1 replica → $7,186/mo
- B200 wins!

The crossover point depends on model, concurrency, and load pattern.
THIS IS WHAT YOU CALCULATE IN THE POC.
```

---

## Hardware + Quantization Matrix

```
                    Quality ◄────────────────────► Speed/Cost
                    fp16        fp8         fp4

    Premium     ┌─────────┬─────────┬─────────┐
    (B200)      │ Best    │ Sweet   │ Max     │
                │ quality │ spot    │ thruput │
                │ $$$     │ $$      │ $       │
                ├─────────┼─────────┼─────────┤
    Standard    │ Good    │ Great   │ Budget  │
    (H100)      │ quality │ value   │ option  │
                │ $$$     │ $$      │ $       │
                ├─────────┼─────────┼─────────┤
    Budget      │ Fits    │ Better  │ Most    │
    (A100)      │ small   │ value   │ models  │
                │ models  │ here    │ only    │
                └─────────┴─────────┴─────────┘

    Most customers land on H100 + fp8. It's the best quality/cost ratio
    for models up to 70B. Push B200 for 70B+ or latency-critical workloads.
```

---

## Multi-GPU Configurations

When a model doesn't fit on one GPU:

| Model Size (fp8) | 1× H100 | 2× H100 | 4× H100 | 1× B200 | 2× B200 |
|-------------------|---------|---------|---------|---------|---------|
| 8B (~4GB) | ✅ | Overkill | Overkill | Overkill | Overkill |
| 70B (~35GB) | ✅ | Faster | Overkill | ✅ (more room) | Overkill |
| 70B fp16 (~140GB) | ❌ | ✅ TP=2 | Overkill | ✅ NO TP! | Overkill |
| 405B (~200GB) | ❌ | ❌ | ✅ TP=4 | ❌ | ✅ TP=2 |

**Cost comparison for 70B fp16**:
- 2× H100: $13.00/hr + NVLink overhead → higher latency
- 1× B200: $9.98/hr + no TP overhead → lower latency
- B200 wins on both cost (-23%) and latency (-20-30%)

**This is often the "aha moment" in a POC**: show the customer that B200
isn't just "a bigger GPU" — it's a fundamentally different architecture
for models in the 70-150B range.

---

## Cold Start Considerations

When autoscaling adds a new replica, the model must load into GPU memory.

| GPU | Typical Cold Start (70B fp8) | Mitigation |
|-----|------------------------------|------------|
| A100 | 45-90 seconds | BDN caching |
| H100 | 30-60 seconds | BDN caching |
| B200 | 20-45 seconds | BDN caching, faster memory |

**BDN (Baseten Delivery Network)**: Model weights cached at the edge.
After first deploy, subsequent cold starts pull from cache (not S3/HF).

**SA recommendation**: For latency-sensitive workloads, set `min_replicas ≥ 1`
to avoid cold starts entirely. The cost of keeping one warm replica is
usually less than the cost of user-facing cold start latency.

---

## What Customers Actually Choose (Patterns)

Based on common SA conversations:

1. **Startup building a chatbot (Series A, cost-conscious)**:
   Llama 3.1 8B on L4, fp16. Scale-to-zero. ~$0.85/hr when active.

2. **Mid-market SaaS adding AI features (100K daily users)**:
   Llama 3.1 70B on H100, fp8. Min 2 replicas. ~$13/hr continuous.

3. **Enterprise document processing (millions of docs/day)**:
   Llama 3.1 70B on H100 × 4, fp8. High concurrency, batch-optimized.
   Throughput matters more than latency. ~$26/hr, but cost/doc is pennies.

4. **AI-native company (inference is the product)**:
   Custom fine-tuned 70B on B200, fp8. Latency-critical. Dedicated compute.
   ~$10/hr per replica, 5+ replicas. Premium performance is the product.

5. **Research lab evaluating models**:
   405B on 4× H100, fp8. Low volume, scale-to-zero. Pay only when testing.
