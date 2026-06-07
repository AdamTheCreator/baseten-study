# SA Discovery Cheat-Sheet
### "How many · How big · How fast · How good"

> Goal of discovery: gather the inputs that size the deployment and reveal the
> bottleneck. **Don't prescribe before you've asked.** Ask input/output token
> lengths *separately* — they drive different phases of inference.

---

## 1. HOW MANY — *volume* → drives **# replicas / GPUs**
- [ ] Requests per day?
- [ ] **Peak** requests/sec (not just the daily average)?
- [ ] Traffic shape — **steady, spiky, or viral**? Daily cycle? Overnight lull?
- [ ] Growth expected (2×? 10×?) over the next 6–12 months?

→ *Drives throughput → how many GPUs you need. Spiky traffic → autoscaling story (idle cost).*

## 2. HOW BIG — *size* → drives **KV cache, concurrency, memory, prefill**
- [ ] Typical **input** tokens per request? (prompt + system + any RAG context)
- [ ] Typical **output** tokens per request?
- [ ] **p95** of each (not just average — heavy requests blow the memory budget)?
- [ ] Any very long contexts (RAG, docs, code)?

→ *Input → prefill / TTFT + starting KV. Output → decode / how long a request holds
a slot + growing KV. **This is the #1 most-missed question.***

## 3. HOW FAST — *latency SLA* → drives **batch size, GPU choice**
- [ ] **TTFT** target? (chat ≈ <500ms–1s)
- [ ] **TPOT / streaming speed** target? (smooth ≈ ~50ms/token)
- [ ] Or for non-streaming (agents/tools): **total response time** target?
- [ ] What's their **current** latency (so you know the headroom)?
- [ ] **Online** (user waiting) or **offline** (batch job)?

→ *Headroom = room to trade speed for cost. Online → small batch / low TPOT;
offline → big batch / max throughput.*

## 4. HOW GOOD — *quality bar* → de-risks **quantization / smaller model**
- [ ] Quality bar — how good is "good enough"?
- [ ] Do they have **evals**? (gold standard = custom product evals)
- [ ] How sensitive is the use case to small quality drops (customer-facing?)?

→ *Evals make quantization/distillation safe to propose — you measure instead of guess.*

## 5. CURRENT STATE — *baseline* → gives you the number to beat
- [ ] What are they running **today** — model, **GPU type & count**, **precision**,
      framework (vLLM / TensorRT-LLM / SGLang)?
- [ ] On-prem, their own cloud, or a provider?
- [ ] **Current cost** ($/month or $/token)?
- [ ] What does **success** look like for this POC? (the win condition)

→ *You can't claim savings without their baseline. Always anchor to it.*

---

## How the answers connect (the chain)
```
input tokens   → prefill / TTFT  + initial KV size
output tokens  → decode / TPOT   + KV grows + slot-time
slot time      = TTFT + output_tokens × TPOT
concurrency    = peak_req/sec × slot_time          (Little's Law)
per-req KV     = (2 × layers × kv_heads × head_dim × bytes) × (input + output tokens)
# replicas     = concurrency ÷ (concurrency one replica holds at SLA)
GPUs/replica   = model size ÷ precision (does it fit on 1 card?)
──────────────────────────────────────────────────────────────────────
Monthly cost   = #replicas × GPUs/replica × $/GPU-hr × hours running
```

## The savings levers (which knob each one turns)
| Lever | Turns down… | Tradeoff |
|---|---|---|
| Quantize (fp16→fp8→fp4) | GPUs per replica (fits on fewer cards) | small quality risk → validate on evals |
| Autoscale / scale-to-zero | hours running (kill idle) | cold-start latency on first request |
| Smaller / distilled model | GPUs/replica **and** concurrency | weeks of work + eval validation |
| Batch / concurrency tuning | # replicas (more per GPU) | higher per-user TPOT |
| Bigger-mem GPU (H200/B200) | GPUs/replica (+concurrency, long ctx) | higher $/hr (but often lower $/token) |

## Golden one-liners
- *"How many, how big, how fast, how good — plus what are you running today."*
- *"Volume tells me how many GPUs; token size tells me what each GPU can hold."*
- *"I'll prove the quality on your evals before anything ships."*
