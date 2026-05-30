# 08 — Corrections & Additions from "Inference Engineering" (Philip Kiely, Baseten Books 2026)

> This is THE official Baseten book. Philip Kiely has been at Baseten 4 years.
> Everything below is what our original study guide missed, got wrong, or
> oversimplified. **Read this file alongside the originals — it's the errata sheet.**

---

## CRITICAL FRAMEWORK CORRECTION: Three Layers, Not Four Pillars

The book defines Baseten's architecture as **three layers**, not four pillars:

1. **Runtime** — Optimizing a single model on a single GPU-backed instance
   - Model performance techniques: Batching, Caching, Quantization,
     Speculation, Parallelism, **Disaggregation**
   - Software stack: CUDA → PyTorch → Inference Engines (vLLM, SGLang, TensorRT-LLM)
   - This is what we called "Model Performance" in 02-core-pillars.md

2. **Infrastructure** — Scaling across clusters, regions, and clouds
   - Routing, Load Balancing, Autoscaling, Multi-Cloud Capacity Management
   - GPUs, Storage, Networking
   - This is what we called "MCM/Infra"

3. **Tooling** — Developer experience with the right level of abstraction
   - This is what we called "DevUI & Truss"

**What we had as "Pillar 4: Post-Training" is NOT a separate pillar in
the book's framework.** It's a prerequisite (Ch 1.3) — model selection,
fine-tuning, and distillation happen BEFORE inference engineering.

**Use the book's three-layer framing in your interview.** It maps directly
to how Baseten thinks about the problem.

---

## MAJOR GAPS: Topics We Completely Missed

### 1. Disaggregation (Ch 5.5) — A Major Technique We Didn't Cover

Disaggregation = separating prefill and decode onto **separate GPUs or nodes**.

Why: Prefill is compute-bound. Decode is memory-bandwidth-bound. When they
run on the same GPU under heavy traffic, they compete for resources.

How it works:
1. Prefill engine processes input → generates KV cache + first token
2. KV cache is sent to decode engine via hardware interconnect
3. Decode engine generates all subsequent tokens

**Conditional disaggregation**: Decode engine first checks if input is already
cached or short enough to handle locally. Only sends to prefill engine if needed.

**When to use**: Large volume (100M-1B+ tokens/day), large models (100B+),
prefill-heavy traffic (long input sequences). A code editor where developers
pass large code contexts is the textbook use case.

**NVIDIA Dynamo** enables production-ready dynamic disaggregation — configurable
xPyD ratios (e.g., 5P3D = 5 prefill engines, 3 decode engines), runtime
adjustable.

**This is a big deal at Baseten's scale.** You need to know this for the interview.

### 2. SGLang — A Major Inference Engine We Undersold

The book treats SGLang as a **co-equal** to vLLM and TensorRT-LLM, not a minor player:

| Engine | vLLM | SGLang | TensorRT-LLM |
|--------|------|--------|-------------|
| Performance | Good | Good | Best |
| Ease of use | Easy | Easy | Hard |
| Model support | Most | Most | Some |
| Hardware | GPU, TPU | NVIDIA, AMD | NVIDIA only |

- SGLang rose to prominence alongside Chinese open models (DeepSeek, Qwen, Kimi)
- Engine of choice at xAI
- Strong MoE support on large systems like GB200 NVL72
- Supports image/video gen via "SGLang Diffusion"
- **EAGLE speculative decoding** is built-in (see example command on p108)
- Baseten uses all three engines and selects per-deployment

### 3. NVIDIA Dynamo (Ch 4.4) — Orchestration Layer We Missed

NVIDIA Dynamo sits **on top of inference engines** to power large-scale
distributed serving. It's not an inference engine itself — it orchestrates them.

Key capabilities:
- Production-ready disaggregation with dynamic xPyD configuration
- Prefill queue management
- Conditional disaggregation routing
- NIXL-based KV transfer between prefill/decode engines
- KVBM (KV Block Manager) for KV cache offloading across memory tiers

### 4. Speculative Decoding — Much More Nuanced Than "Lookahead"

Our guide only mentioned "lookahead decoding." The book covers FOUR approaches:

| Method | How It Works | Best For |
|--------|-------------|----------|
| **Draft-Target** | Separate smaller model generates drafts | Quick setup, no fine-tuning needed |
| **Medusa** | Fine-tuned extra decoder heads on target model | Inspired EAGLE, not widely used now |
| **EAGLE** | Purpose-built draft model from hidden states | Go-to for general use (best acceptance rate) |
| **N-gram / Lookahead** | Dictionary from input text, no draft model | Code completion (input ≈ output) |

Key insight: Speculation only improves **TPS/ITL, NOT TTFT**. It only
helps during decode. Acceptance rate drops at higher batch sizes (compute
needed for verification), so speculation is dynamically disabled under load.

### 5. Ops:Byte Ratio and Arithmetic Intensity (Ch 2.4)

This is a foundational concept we didn't cover:

**Ops:byte ratio** = GPU's compute FLOPS ÷ memory bandwidth
- H100 in FP16: 989 TFLOPS ÷ 3.35 TB/s = **~295 ops:byte**
- This means: for perfectly balanced inference, the system needs 295
  floating point operations for every byte it reads from memory

**Arithmetic intensity** = work (compute ops) ÷ memory traffic (bytes moved)
- If arithmetic intensity > ops:byte ratio → **compute bound** (prefill)
- If arithmetic intensity < ops:byte ratio → **memory bound** (decode)

This is how you formally prove WHY prefill is compute-bound and decode
is memory-bound. The book works through the full math on p62-66.

**For the interview**: Being able to explain ops:byte ratio shows you
understand the physics, not just the rules of thumb.

### 6. Quantization Sensitivity Hierarchy (Ch 5.1.2)

The book provides a specific ordering of what's safe to quantize:

1. **Weights** (least sensitive) — linear layers quantize well
2. **Activations** — intermediate outputs, only somewhat sensitive
3. **KV cache** — moderately sensitive, compounds over sequence
4. **Attention** (most sensitive) — softmax is very sensitive to precision

Most production quantization: FP8 for weights + activations + KV cache,
but attention stays in original precision. Components of the attention
layer are **rarely quantized** even in aggressive schemes.

Also: **MXFP8** and **NVFP4** are new microscaling formats on Blackwell
with blockwise scale factors (every 32 params for MX, every 16 for NVFP4).
These preserve quality better than naive FP4/FP8. We mentioned "fp4" and
"fp8" generically — the specific format matters.

### 7. KV Cache Storage Tiers (Ch 5.3.2)

KV cache doesn't have to live in GPU VRAM:

| Level | Memory Type | Speed | Size |
|-------|-----------|-------|------|
| G1 | GPU VRAM | TB/s | 10s-100s GB |
| G2 | Host CPU RAM | 10s-100s GB/s | 100s GB - TBs |
| G3 | Local SSD | 5-10 GB/s | TBs |
| G4 | Networked SSD | GB/s | 10s TB |

NVIDIA Dynamo KVBM manages movement between tiers. Grace CPUs provide
much faster G2 access due to NVLink Chip-to-Chip.

### 8. Cache-Aware Routing (Ch 5.3.3)

Standard load balancing distributes requests evenly. **Cache-aware routing**
routes requests to the replica most likely to have a KV cache hit for
that request's prefix.

Example: User in a multi-turn conversation should always hit the same
replica to reuse the KV cache from prior turns.

This is an infrastructure technique, not a model technique. We didn't mention it.

### 9. Multi-Instance GPU (MIG) (Ch 3.3.2)

When the GPU is too BIG for the model:
- H100 can be split into up to **7 compute slices** (7 SMs each)
- Each slice gets a portion of compute, VRAM, and interconnect
- A 3/7 slice ≈ half the GPU compute, ~40GB VRAM
- Great for small models (Orpheus TTS at 3B params)
- Better perf/$ than using older GPUs (fractional H100 > full T4)

### 10. H200 and B300 — GPUs We Missed

The book includes GPUs our guide didn't mention:

| GPU | Compute (FP8) | Memory | Bandwidth |
|-----|--------------|--------|-----------|
| H200 | 1,979 TFLOPS | **141 GB** | **4.8 TB/s** |
| B300 | ~5 petaFLOPS | **288 GB** | up to 8 TB/s |

**H200** is same compute as H100 but with 76% more memory and 43% more
bandwidth. It's a Hopper chip with HBM3e (vs HBM3 on H100).

**B300** is the larger Blackwell chip (288GB vs B200's 192GB).

### 11. Rubin Architecture (Ch 3.2.4) — Next Gen After Blackwell

Launching 2026. Key features:
- HBM4 (faster memory bandwidth than HBM3e)
- CPX chip purpose-built for prefill
- Vera CPU replaces Grace CPU
- Feynman architecture follows in 2028

### 12. Alternative Accelerators (Ch 3.4)

The book covers non-NVIDIA options — relevant because customers may ask:

| Company | Product | Edge |
|---------|---------|------|
| AMD | MI350 | Competitive specs, own software stack |
| AWS | Inferentia / Trainium | Purpose-built for AWS inference |
| Google | TPU | ASIC for inference and training |
| Groq | LPU | SRAM-based, extreme memory bandwidth |
| Cerebras | WSE-3 | Wafer-scale, removes decode bottleneck |

### 13. Model Selection Is the #1 Optimization (Ch 1.3)

The book is emphatic: **the most important decision in model performance
optimization is which model you choose**, not the runtime engine or
speculation algorithm.

"Find — or create — the smallest, easiest-to-run model that's smart
enough to handle the task at hand."

This reframes the SA conversation: before optimizing a 70B model,
ask if a fine-tuned 8B could do the job. That's a bigger perf win
than any amount of quantization or speculation.

### 14. Distillation vs Fine-Tuning (Ch 1.3.3)

Our guide mentioned fine-tuning but not distillation:

- **Fine-tuning**: Changes weights to perform better on a specific domain
- **Distillation**: Trains a smaller "student" to emulate a larger "teacher"
  model's probability distributions (not just its final answers)

DeepSeek R1's distilled versions (on Llama 3 and Qwen 2.5) are among
the most popular models on HuggingFace. This is a real SA conversation:
"Can we distill your 70B model to an 8B that's good enough for your task?"

### 15. Online vs Offline Workloads (Ch 1.2.2)

The book distinguishes two fundamentally different deployment patterns:

- **Online**: Real-time, latency-sensitive (chat, code completion, voice)
  → Optimize for TTFT and per-user TPS
- **Offline**: Batch processing, throughput-sensitive (transcription, embedding, moderation)
  → Optimize for total system throughput and cost/token

"It's possible to have a single model used for both online and offline
jobs... Assuming both use cases have enough volume, it will be more
cost effective to create two separate deployments."

### 16. Consumer vs B2B Inference (Ch 1.2.3)

Different priorities:
- **Consumer**: Cost-sensitive, unpredictable traffic, viral spikes. Optimize cost + flexibility.
- **B2B**: Better margins, stable traffic, high availability required. Optimize latency + uptime.

### 17. Measuring Quality Impact of Quantization (Ch 5.1.3)

Three methods, in order of rigor:
1. **Perplexity** — Quick sanity check (lower = better, compare to original)
2. **Intelligence benchmarks** — MMLU, SWE-bench (compare to original scores)
3. **Custom evals** — Product-specific evaluation suite (gold standard)

"The standard for production-ready quantization is zero perceptible quality loss."

### 18. TensorRT-LLM V0 vs V1 (Ch 4.3.3)

Important version distinction:
- **V0 (0.X.Y)**: Plugin for NVIDIA TensorRT. Builds a compiled engine.
- **V1 (1.X.Y)**: Standalone, PyTorch-based. No TensorRT dependency.

V1 was released summer 2025. Both versions are in production. The developer
experience is now similar to vLLM/SGLang (`trtllm-serve` command with flags).

### 19. Kernel Fusion (Ch 4.1.3)

We didn't explain WHY FlashAttention is fast. Kernel fusion:
- Combining two sequential GPU operations into one
- Eliminates intermediate memory reads/writes
- During decode (memory-bound), every wasted read/write costs latency
- FlashAttention = fused attention kernel. Not a new algorithm — same math,
  fewer memory round trips.

### 20. Model File Formats (Ch 4.2.2)

- **Safetensors**: Dominant format. Weights only, no executable code. Memory-mapped.
- **ONNX**: Weights + execution graph bundled together. More portable.
- **GGUF**: Popular for local/edge inference. Stores highly quantized models.

---

## CORRECTIONS TO OUR GUIDE

### Hardware Numbers Were Off

Our guide (03-hardware.md) had some inaccurate specs:

| GPU | Our Guide | Book (Correct) |
|-----|-----------|---------------|
| B200 VRAM | 192 GB | 192 GB (correct!) |
| B200 Compute | 2,250 FP16 TFLOPS | ~5 petaFLOPS FP8 (different metric) |
| B200 Bandwidth | 8,000 GB/s | "up to 8 TB/s" (same, correct) |
| H100 Bandwidth | 3,350 GB/s | 3.35 TB/s (same, correct) |
| H200 | NOT MENTIONED | 141 GB, 4.8 TB/s, same compute as H100 |
| B300 | NOT MENTIONED | 288 GB, up to 8 TB/s |

### NVLink Speed Was Too Low

Our guide said 900 GB/s for NVLink. The book says:
- Hopper NVLink: 900 GB/s
- **Blackwell NVLink: up to 1800 GB/s** (double)

### "Shared vs Dedicated" Framing

The book has a clearer framing than our "pre-optimized API vs custom deploy":

| | Shared Inference | Dedicated Inference |
|---|---|---|
| Pricing | Per million tokens | Per GPU hour |
| Cold starts | None | Possible (scale-to-zero) |
| Engineering | Minimal (API key) | Moderate (Truss config) |
| Control | None | Full |
| Best for | Early stage, low volume | Scale, specialization, orchestration |

Three reasons to switch to dedicated:
1. **Scale** — Enough volume that per-GPU is cheaper than per-token
2. **Specialization** — Custom/fine-tuned model, specific latency needs
3. **Orchestration** — Multi-model pipelines need coordinated deployment

### Speculative Decoding Nuance

Our guide said "lookahead decoding" as if it's the main speculation method.
**EAGLE is the go-to** for general use. N-gram/lookahead is specifically
for code completion where output mirrors input.

### "Quantization Improves Speed Not Just Memory" — More Nuanced

The book clarifies: going from 16-bit to 8-bit does NOT linearly double
performance. "Quantization down a single level of precision generally offers
30 to 50 percent better performance for LLMs" — not 2x.

### VRAM Budget Rule of Thumb

Our guide said "budget 10-20% more than raw weight size." The book says:
"The VRAM should hold the model weights, **plus at least 50 percent headroom**
for KV cache (more for long context, high batch sizes, or video generation models)."

50% headroom, not 10-20%. Big difference.

---

## TOPICS FROM THE BOOK TO STUDY FURTHER

If you have time before the interview, prioritize reading these chapters in full:

1. **Ch 5.5 Disaggregation** (p148-151) — Short, critical, you need to know this
2. **Ch 5.3 Caching** (p136-141) — Prefix caching, KV storage tiers, cache-aware routing
3. **Ch 5.2 Speculative Decoding** (p129-136) — EAGLE especially
4. **Ch 7 Production** (p177-208) — Autoscaling, cold starts, zero-downtime deploys, NIMs
5. **Ch 1.2 About Your App** (p27-30) — Online vs offline, consumer vs B2B
6. **Appendix A: Glossary** (p209-230) — Their official definitions

---

## KEY QUOTES FOR THE INTERVIEW

> "Inference is the most valuable category in the AI industry." (Preface, p9)

> "The most important decision in model performance optimization isn't the
> runtime engine or speculation algorithm, it's which model you choose to
> work with in the first place." (Ch 1.3, p31)

> "The more constraints you can introduce in your inference system, the better
> performance you'll achieve." (Ch 5, p119)

> "Finding the right combination of techniques and configurations takes patient
> experimentation. I remember during an internal hackathon one of Baseten's
> inference engineers tried 77 different configurations via a handwritten script
> before finding a non-obvious solution that doubled TPS for a customer's model."
> (Ch 5, p119)

> "Switching to open models unlocks the opportunity to use inference engineering
> to make the models powering AI products better in new dimensions: Latency,
> Availability, and Cost." (Preface, p12)

These quotes demonstrate you've read the book and internalized the philosophy.
