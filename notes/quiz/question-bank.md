# Question Bank

Grouped by theme, ordered easy → hard. Each question has a **grading key** —
the points an answer should hit. SA-applied questions also note the
*customer move* the answer should demonstrate. Page/chapter refs map to the
book via `../08-book-corrections.md`.

Legend: ⚙️ = pure technical · 🤝 = customer/SA-applied · 🎯 = blended

---

## A. The Two Phases (prefill / decode)

1. ⚙️ Name the two phases of LLM inference and state which resource each is
   bound by. *(Key: prefill = compute-bound, parallel over input tokens, sets
   TTFT; decode = memory-bandwidth-bound, autoregressive, sets TPOT.)*
2. 🎯 A customer says "our chatbot feels slow." What's your first diagnostic
   question and why? *(Key: split the problem — is it TTFT or TPOT? Then
   queue vs prefill vs decode. Don't optimize blindly.)*
3. ⚙️ Why is decode memory-bandwidth-bound rather than compute-bound?
   *(Key: one token at a time → tiny matmuls, GPU mostly waits reading weights
   + KV cache from memory; low arithmetic intensity.)*
4. 🎯 A customer has a 4,000-token system prompt on every request and complains
   about TTFT. What lever do you reach for? *(Key: prefix/prompt caching —
   reuse prefill KV for the shared prefix.)*

## B. KV Cache & Memory Budget

5. ⚙️ What is the KV cache and why does it usually limit concurrency before
   compute does? *(Key: cached K/V tensors per layer/token so decode doesn't
   recompute; grows with seq_len × layers × heads × precision; eats VRAM left
   after weights.)*
6. 🎯 "Why can I only run ~2 concurrent requests on an H100 with my 70B model?"
   Walk the customer through the math. *(Key: 70B fp16 ≈ 140GB > 80GB — doesn't
   even fit; fp8 ≈ 70GB leaves ~10GB for KV → few concurrent. Quantize or go
   bigger-memory GPU.)*
7. ⚙️ Book's VRAM rule of thumb for headroom? *(Key: weights + **at least 50%**
   headroom for KV cache, more for long context / high batch / video.)*
8. ⚙️ Name the four KV-cache storage tiers G1–G4 and the tradeoff.
   *(Key: G1 GPU VRAM (TB/s), G2 host RAM, G3 local SSD, G4 networked SSD;
   bigger but slower as you descend. Dynamo KVBM moves blocks across tiers.)*

## C. Batching

9. ⚙️ Continuous vs static batching — the difference and why it matters.
   *(Key: static waits for slowest in batch; continuous lets requests join/leave
   a running batch each iteration → no head-of-line blocking.)*
10. 🎯 State the fundamental batching tradeoff and how you'd tune it for (a) a
    chat app vs (b) overnight document processing. *(Key: ↑batch = ↑throughput
    ↓cost/token but ↑TPOT. Chat → small batch (low TPOT); offline → big batch
    (max throughput).)*

## D. Quantization

11. ⚙️ Why does quantization speed up *decode*, not just save memory?
    *(Key: decode is memory-bound — smaller weights = fewer bytes read; also
    frees VRAM for KV → higher batch.)*
12. ⚙️ Quantization sensitivity hierarchy — what's safe to quantize, what isn't?
    *(Key: weights (safe) → activations → KV cache → attention (most sensitive,
    softmax; usually kept in higher precision).)*
13. 🎯 Customer: "We need the absolute best quality, so no quantization." How do
    you respond? *(Key: don't argue — propose a benchmark on THEIR prompts;
    fp8 typically <1% degradation for ~30–50% perf; measure with their evals.)*
14. ⚙️ Correct the myth that fp16→fp8 doubles performance. *(Key: book says one
    precision level ≈ **30–50%** better perf, not 2x.)*
15. ⚙️ What are MXFP8 / NVFP4 and why do they beat naive fp4/fp8?
    *(Key: Blackwell microscaling formats with blockwise scale factors (every 32
    / 16 params) → preserve quality better.)*
16. 🎯 How do you measure quantization quality impact, in order of rigor?
    *(Key: perplexity (quick) → intelligence benchmarks (MMLU/SWE-bench) →
    custom product evals (gold standard). Standard = zero perceptible loss.)*

## E. Speculative Decoding

17. ⚙️ One-line definition + the key limitation. *(Key: small draft predicts
    several tokens, full model verifies in one pass; helps **decode/TPS only,
    NOT TTFT**; acceptance drops at high batch → disabled under load.)*
18. ⚙️ Name the four methods and the best use case of each. *(Key: Draft-Target
    (quick, no FT); Medusa (extra heads, dated); **EAGLE** (go-to, best
    acceptance); N-gram/Lookahead (code completion, input≈output).)*
19. 🎯 Customer generates tons of JSON/code. Which speculation method and why?
    *(Key: N-gram/lookahead — output mirrors input; or EAGLE generally. Tie to
    structured output.)*

## F. Parallelism & Disaggregation

20. ⚙️ Tensor vs pipeline parallelism — split and tradeoff. *(Key: TP splits each
    layer across GPUs, needs NVLink, lower latency, common for inference; PP puts
    different layers on different GPUs, tolerates slow interconnect, common in
    training.)*
21. ⚙️ What is disaggregation and what problem does it solve? *(Key: separate
    prefill (compute-bound) and decode (memory-bound) onto different GPUs/nodes
    so they stop competing; KV cache shipped over interconnect.)*
22. 🎯 When do you recommend disaggregation to a customer? *(Key: high volume
    (100M–1B+ tok/day), large models (100B+), prefill-heavy/long-input traffic —
    e.g., code editor with big contexts.)*
23. ⚙️ What does NVIDIA Dynamo add on top of inference engines? *(Key: orchestration
    — dynamic xPyD disaggregation, prefill queue mgmt, conditional disaggregation,
    NIXL KV transfer, KVBM tiering. Not an engine itself.)*

## G. Inference Engines & Software Stack

24. ⚙️ Compare vLLM, SGLang, TensorRT-LLM on perf / ease / hardware.
    *(Key: TRT-LLM best perf but hard & NVIDIA-only; vLLM & SGLang easy, broad
    support; Baseten uses all three, picks per deployment.)*
25. ⚙️ TensorRT-LLM V0 vs V1? *(Key: V0 = TensorRT plugin, compiled engine; V1 =
    standalone PyTorch-based, no TRT dep, vLLM-like `trtllm-serve`, since summer
    2025.)*
26. ⚙️ Why is FlashAttention fast? (kernel fusion) *(Key: fuses ops into one
    kernel, eliminates intermediate memory round-trips; same math, fewer
    reads/writes — big in memory-bound decode.)*
27. ⚙️ Safetensors vs ONNX vs GGUF. *(Key: safetensors = weights only, no code,
    mmap, dominant; ONNX = weights+graph, portable; GGUF = quantized, edge/local.)*

## H. The Physics (ops:byte / arithmetic intensity)

28. ⚙️ Define ops:byte ratio and arithmetic intensity, and how they prove
    prefill is compute-bound / decode memory-bound. *(Key: ops:byte = FLOPS ÷
    bandwidth (H100 fp16 ≈ 295); arithmetic intensity = ops ÷ bytes moved.
    AI > ops:byte → compute-bound (prefill); AI < ops:byte → memory-bound
    (decode).)*

## I. Hardware

29. ⚙️ H100 vs H200 — what changed? *(Key: same compute; H200 = 141GB (vs 80),
    4.8 TB/s (vs 3.35), HBM3e. More memory/bandwidth → bigger batch / longer
    context.)*
30. 🎯 Customer running a 3B TTS model on a full H100 — what do you suggest?
    *(Key: MIG — slice the H100 (up to 7 slices); fractional H100 beats a full
    older GPU on perf/$.)*
31. ⚙️ Blackwell NVLink speed vs Hopper? *(Key: up to 1800 GB/s vs 900 GB/s.)*

## J. Metrics & SLAs

32. ⚙️ TTFT vs TPOT vs throughput — define and say when each matters most.
33. 🎯 A customer's p50 is great but p99 is 5x worse. What do you suspect?
    *(Key: slow autoscaling or cold starts hitting some users; queue depth.)*
34. ⚙️ Two meanings of "throughput" — which drives cost/token? *(Key: per-request
    vs system; **system** throughput drives cost/token.)*

## K. Model Selection & Post-Training (Ch 1.3)

35. 🎯 Book says the #1 optimization is ___? Reframe an SA conversation around it.
    *(Key: **which model you choose**. Before optimizing a 70B, ask if a
    fine-tuned/distilled 8B is smart enough — bigger win than any quant/spec.)*
36. ⚙️ Fine-tuning vs distillation. *(Key: FT changes weights for a domain;
    distillation trains a small student to mimic a large teacher's probability
    distributions. e.g., DeepSeek R1 distilled to Llama/Qwen.)*

## L. Workload Shape & Business (Ch 1.2)

37. 🎯 Online vs offline workloads — optimize for what, and when to split
    deployments? *(Key: online → TTFT/per-user TPS; offline → system
    throughput/cost. If both have volume, two deployments is cheaper.)*
38. 🎯 Consumer vs B2B inference priorities. *(Key: consumer = cost-sensitive,
    spiky/viral → optimize cost + flexibility; B2B = better margins, stable,
    HA → optimize latency + uptime.)*
39. 🎯 Shared vs Dedicated inference — pricing model and the three reasons to
    move a customer to dedicated. *(Key: shared = per-token, no cold start, API
    key; dedicated = per-GPU-hour, full control. Switch for Scale, Specialization,
    Orchestration.)*

## M. Infrastructure / Production (Ch 7)

40. ⚙️ Cache-aware routing — what and why. *(Key: route a request to the replica
    most likely to already hold its prefix's KV; e.g., pin a multi-turn convo to
    one replica. Infra technique, not model.)*
41. 🎯 Why might scale-to-zero hurt a B2B customer, and what's the tradeoff?
    *(Key: cold starts on first request after idle; B2B wants HA/low latency →
    keep min replicas warm; cost vs latency tradeoff.)*

## N. Competitive / Closing

42. 🤝 Customer says "we'll just run vLLM on our own H100s, why pay Baseten?"
    Make the case. *(Key: continuous batching is table stakes; Baseten adds
    multi-cloud capacity, autoscaling, cache-aware routing, disaggregation,
    engine selection per model, 99.99% uptime, the 77-config tuning expertise —
    perf/$ and ops burden, not just raw GPU.)*
43. 🤝 Pick one quote from the book that captures Baseten's philosophy and say
    why it matters to a customer. *(Key: e.g., "the most important decision...
    is which model you choose" or "the more constraints you introduce, the
    better performance.")*
