# 07 — Blind Spots & Glossary

> Things that trip up people new to inference engineering. If you're coming
> from software engineering, product, or sales — these are the gaps to fill.
> This section also covers interview-relevant gotchas.

---

## Blind Spot #1: Latency ≠ Throughput

This is the most common confusion. They are inversely correlated when
you optimize for one at the expense of the other.

- **Low latency config**: concurrency=1, one request at a time.
  That single request is FAST. But your GPU sits idle between requests.
  Throughput is terrible. Cost per token is astronomical.

- **High throughput config**: concurrency=128, many requests batched.
  Each individual request is slower (higher TPOT). But your GPU
  processes way more tokens per second total. Cost per token is great.

**The SA skill**: Find the knee of the curve — the concurrency level
where throughput is high AND latency is within the customer's SLA.

---

## Blind Spot #2: Tokens Per Second Can Mean Two Things

Always clarify:
- **Per-request tokens/sec**: How fast one user sees tokens stream in.
  This is what end users perceive. Lower is worse UX.
- **System tokens/sec**: Total tokens generated across all concurrent
  requests. This determines cost-per-token. Higher is cheaper.

A system doing 5,000 tokens/sec across 50 concurrent requests is generating
100 tokens/sec per request. Both numbers are valid — but they answer
different questions.

---

## Blind Spot #3: GPU Utilization Is Misleading

A dashboard showing "30% GPU utilization" doesn't mean the GPU is wasted.

During **decode** (token generation), the bottleneck is **memory bandwidth**,
not compute. The GPU is reading model weights from memory constantly but
only doing a small amount of math. GPU utilization measures compute usage,
not memory bandwidth saturation.

A decode-heavy workload at 30% GPU utilization with 85% memory bandwidth
utilization is fully loaded. Don't let a customer think they're paying
for an idle GPU.

**What to watch instead**: memory bandwidth utilization, tokens/sec output,
and queue depth. If tokens/sec is maxed and queue is growing, the GPU is
fully saturated — regardless of what the utilization % says.

---

## Blind Spot #4: Model Size ≠ File Size

"Llama 70B" means 70 billion parameters, not 70 gigabytes.

Memory required depends on precision:
- fp32: 70B × 4 bytes = 280 GB
- fp16/bf16: 70B × 2 bytes = 140 GB
- fp8: 70B × 1 byte = 70 GB
- int4/fp4: 70B × 0.5 bytes = 35 GB

Plus KV cache, plus activation memory, plus framework overhead.
Rule of thumb: budget 10-20% more than raw weight size.

---

## Blind Spot #5: Cold Start Is Not Just "Loading the Model"

Cold start includes:
1. **Container scheduling**: Finding a GPU node, starting the container
2. **Container initialization**: Starting the Python runtime, loading libraries
3. **Weight loading**: Downloading or loading model weights from storage
4. **Model compilation**: TRT-LLM compilation (if not cached), engine warmup
5. **Warm-up inference**: First inference is slower (JIT compilation, cache warming)

Total cold start for a 70B model: 30-90 seconds depending on GPU and
caching. This is why `min_replicas ≥ 1` matters for production.

Baseten mitigates this with BDN (weights cached at edge) and pre-compiled
engine caching. But you can't eliminate it entirely.

---

## Blind Spot #6: Quantization Quality Loss Is Use-Case Dependent

"fp8 has less than 1% quality loss" is a general benchmark statement.
On specific tasks, it can vary:

- **Simple Q&A, summarization**: Usually negligible quality loss with fp8
- **Math/reasoning**: More sensitive to quantization — test carefully
- **Code generation**: Usually fine with fp8, more sensitive with fp4
- **Creative writing**: Usually fine at any quantization
- **Rare languages**: More sensitive — calibration data may not cover them

**Always run the customer's eval suite at each precision level.** Never
assume quality is preserved — verify it.

---

## Blind Spot #7: Autoscaling Has Lag

Autoscaling is not instant. The timeline:

```
Traffic spike detected (t=0)
  → Autoscaler evaluates metrics (t=10-30s, depends on window)
    → Requests new replica (t=30s)
      → Container scheduled to GPU node (t=30-60s)
        → Model loaded and warmed up (t=60-120s)
          → Replica ready to serve (t=2-4 min total)
```

During those 2-4 minutes, existing replicas absorb the spike. If the
spike is large enough, queue depth grows and latency increases.

**Mitigation strategies**:
- **Headroom**: Set `target_utilization` to 60-70% (buffer capacity)
- **Min replicas**: Keep enough warm replicas for expected peaks
- **Shorter scaling window**: React faster (but risk oscillation)
- **Predictive scaling**: If traffic is predictable (e.g., business hours),
  pre-scale based on schedule

---

## Blind Spot #8: Networking Matters for Multi-GPU

When a model spans multiple GPUs (tensor parallelism), those GPUs need to
communicate constantly during inference. The interconnect speed directly
affects performance:

- **NVLink (within a node)**: 900 GB/s (H100). Fast enough that TP=2 or TP=4
  within a single machine adds minimal overhead.
- **InfiniBand (across nodes)**: 400 Gb/s (50 GB/s). Much slower than NVLink.
  Cross-node TP is possible but adds significant latency.
- **Ethernet**: 10-100 Gb/s. Way too slow for TP. Don't even try.

**Why this matters**: If a customer needs TP=8 for a 405B model, all 8 GPUs
should ideally be in the same node (connected via NVLink). Baseten handles
this scheduling — on-prem, you'd need to ensure proper GPU topology.

---

## Blind Spot #9: MoE Models Are Different

Mixture of Experts (MoE) models like DeepSeek V3 or Mixtral break the
normal rules:

- **Total parameters ≠ Active parameters**: DeepSeek V3 has 671B total
  but only ~37B active per token. You need VRAM for all 671B but compute
  only processes 37B per token.
- **Expert routing adds overhead**: The router must decide which experts
  to activate, adding latency.
- **Memory access is irregular**: Different experts are activated for
  different tokens, making caching less effective.
- **Baseten's BIS-LLM engine**: Purpose-built for MoE with optimized
  expert routing and memory management.

Don't benchmark MoE models the same way as dense models. They have
different bottlenecks.

---

## Blind Spot #10: The Customer's Eval Suite Matters More Than Benchmarks

MMLU, HumanEval, and other public benchmarks tell you about general
model capability. They do NOT tell you how the model performs on the
customer's specific task.

**Always push to evaluate on the customer's data.** A model that scores
75% on MMLU but 95% on the customer's internal eval is better than one
that scores 80% MMLU and 88% on their eval.

Similarly for performance: synthetic benchmarks (uniform prompt lengths,
artificial concurrency) don't reflect real workloads. Use the customer's
actual prompts and traffic patterns.

---

## Blind Spot #11: Prompt Caching Is a Huge Deal

If the customer sends the same system prompt (or prefix) with every request,
prompt caching can dramatically reduce TTFT and cost:

- **Without caching**: Every request re-processes the full system prompt
  during prefill. 4,000-token system prompt = 4,000 tokens of compute
  every single request.
- **With caching**: System prompt is processed once, KV cache is stored.
  Subsequent requests skip prefill for the cached prefix. TTFT drops
  from ~500ms to ~50ms for the cached portion.

Baseten's model APIs support this (see pricing: "cached input" is
significantly cheaper). Ask every customer: "Do you use a system prompt?
How long is it? Does it change between requests?"

---

## Blind Spot #12: The "Works in Demo, Fails in Production" Gap

Things that work in a POC demo but fail at production scale:

- **Autoscaling oscillation**: Works fine at steady state, but production
  traffic is bursty. Replicas scale up/down rapidly, causing instability.
- **Memory leaks**: KV cache accumulation over hours. Works for 30-min
  benchmark, OOMs after 8 hours.
- **Tail latency**: p50 is great, p99 is terrible. You only see this
  with sustained load over time.
- **Error handling**: What happens when a GPU fails? When a request
  times out? When the model generates too many tokens?

**In your POC, run sustained load for at least 30 minutes.** Quick bursts
hide these problems.

---

## Glossary of Terms You'll Use Daily

| Term | Definition |
|------|-----------|
| **TTFT** | Time to First Token — latency before first output token appears |
| **TPOT** | Time Per Output Token — time between consecutive output tokens |
| **TPS** | Tokens Per Second — either per-request or system-wide (clarify!) |
| **QPS** | Queries Per Second — request rate |
| **p50/p95/p99** | Latency percentiles (median, 95th, 99th percentile) |
| **KV Cache** | Key-Value cache — stored attention computations from prefill |
| **Prefill** | Processing the input prompt (compute-bound phase) |
| **Decode** | Generating output tokens one-by-one (memory-bandwidth-bound) |
| **TP** | Tensor Parallelism — splitting a model across GPUs within a layer |
| **PP** | Pipeline Parallelism — splitting a model across GPUs across layers |
| **Continuous Batching** | Adding/removing requests from a batch mid-inference |
| **TRT-LLM** | TensorRT-LLM — NVIDIA's compiled inference engine |
| **vLLM** | Popular open-source LLM serving framework (PyTorch-based) |
| **DFlash** | Baseten's custom attention kernel (proprietary) |
| **BDN** | Baseten Delivery Network — CDN for model weights |
| **MoE** | Mixture of Experts — model architecture with sparse activation |
| **SFT** | Supervised Fine-Tuning — training on instruction/response pairs |
| **DPO** | Direct Preference Optimization — training from human preferences |
| **LoRA** | Low-Rank Adaptation — parameter-efficient fine-tuning method |
| **QLoRA** | Quantized LoRA — LoRA on a quantized base model |
| **NVLink** | High-speed GPU-to-GPU interconnect (within a node) |
| **InfiniBand** | High-speed network interconnect (across nodes) |
| **FP8/FP4** | 8-bit / 4-bit floating point — quantization precision levels |
| **BF16** | Brain Float 16 — 16-bit format with more exponent bits than FP16 |
| **OOM** | Out of Memory — when model + KV cache exceeds GPU VRAM |
| **Scale-to-zero** | Removing all replicas when idle (no cost) |
| **Cold start** | Time to spin up a new replica from zero |
| **Noisy neighbor** | Performance degradation from shared infrastructure |
| **SLA** | Service Level Agreement — contractual performance guarantees |
| **TCO** | Total Cost of Ownership — full cost including ops, people, waste |

---

## Interview Prep: Questions They Might Ask You

### Technical Understanding

**"Explain the difference between prefill and decode"**
→ See 00-foundations.md. Prefill = compute-bound, processes all input at once,
generates KV cache. Decode = memory-bandwidth-bound, generates tokens one at a
time using KV cache. TTFT is determined by prefill. TPOT is determined by decode.

**"Why does quantization improve throughput?"**
→ Three reasons: (1) smaller weights read faster from memory (decode is
memory-bandwidth-bound), (2) smaller weights leave more VRAM for KV cache
(higher concurrency), (3) fp8 uses Tensor Cores more efficiently on H100/B200.

**"What's the difference between TTFT and total latency?"**
→ TTFT = queue time + prefill. Total = TTFT + (output_tokens × TPOT).
TTFT matters for UX (user sees response start). Total matters for batch
workloads. They optimize differently.

**"When would you recommend B200 over H100?"**
→ See 03-hardware.md. Three scenarios: (1) model needs 2× H100 for TP but
fits on 1× B200 (cheaper AND faster), (2) latency-critical workloads (2.4x
memory bandwidth), (3) high-concurrency needs (2.4x VRAM for KV cache).

### SA Scenario Questions

**"A customer says your platform is too expensive. How do you respond?"**
→ "Let's look at total cost of ownership, not just GPU price. What are you
spending on infrastructure engineering, idle GPU time, and over-provisioning?
Then let's benchmark — if we can show 40% better throughput, the cost per
token is lower even if the GPU price per hour is the same."

**"A customer's POC shows worse latency than their current setup. What do you do?"**
→ Diagnose: (1) Check if it's queue time (need more replicas or higher
concurrency target). (2) Check if TRT-LLM compilation completed (pre-compiled
engines are faster). (3) Compare configs — are they running fp16 and we're
also fp16? (4) Check network latency (are they comparing local network
vs internet roundtrip?). Be transparent — if their setup is genuinely
faster for their specific workload, say so and explain what we can improve.

**"How would you scope a POC for a customer you've never worked with?"**
→ See 04-poc-playbook.md. Start with discovery: understand their model,
workload, current metrics, and success criteria. Deploy their model on
Baseten (1-2 days). Run optimization sweeps (2-3 days). Production load
test (1 day). Generate comparison report (1 day). Present (1 day).
Total: 7-10 days.

**"A customer wants to run a 405B model but is budget-constrained"**
→ Options: (1) fp8 quantization (halves GPU requirements), (2) fp4 if
quality allows, (3) consider if a 70B fine-tuned model could match 405B
quality on their specific task (much cheaper to serve), (4) scale-to-zero
if usage is intermittent, (5) consider Baseten's pre-optimized API endpoints
(pay per token, no GPU commitment).

### Scripting / Technical Depth

**"Write pseudocode for measuring TTFT of a streaming endpoint"**
```python
start = time.perf_counter()
stream = client.chat.completions.create(messages=msgs, stream=True)
first_chunk = next(iter(stream))
ttft = time.perf_counter() - start
```

**"How would you automate a quantization comparison?"**
→ Script that: (1) deploys model at fp16, fp8, fp4 (loop over configs),
(2) runs same benchmark against each, (3) collects metrics, (4) generates
comparison table. See scripts/compare_quantizations.py.

**"How do you calculate cost per token?"**
→ `cost_per_million_tokens = (gpu_cost_per_hour / system_tokens_per_second / 3600) × 1,000,000`

---

## Things to Study If You Have More Time

1. **Read TRT-LLM docs**: https://github.com/NVIDIA/TensorRT-LLM
   Understand compilation, quantization, and engine concepts.

2. **Read vLLM docs**: https://docs.vllm.ai
   Know the competitor's serving framework. Many customers start here.

3. **Understand PagedAttention**: The algorithm that enables continuous
   batching and efficient KV cache management. Published by vLLM team.

4. **Know NVIDIA GPU architectures**: Hopper (H100), Blackwell (B200).
   Understand Tensor Cores, NVLink, HBM3 vs HBM3e.

5. **Read Baseten blog posts**: Especially DFlash, performance benchmarks,
   and customer case studies. These are the talking points you'll use.

6. **Practice with the Truss CLI**: Deploy a model, change configs,
   measure the difference. Hands-on experience is more valuable than
   reading about it.
