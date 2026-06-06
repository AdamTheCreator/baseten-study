# SA Roleplay Scenarios

Roleplay drills for the **core SA deliverable**: running the customer
conversation that turns inference fundamentals into a recommendation + POC.

## How to run
Say: *"Run roleplay scenario N"* (or *"give me a new roleplay"*).

- **Claude plays the customer** (persona + hidden facts below). It only reveals
  a fact when the SA actually asks for it — discovery is part of the test.
- **You play the SA.** Drive the conversation through four moves:
  1. **Discovery** — ask the questions that surface the bottleneck
  2. **Diagnose** — name what's actually wrong
  3. **Recommend** — levers + tradeoffs, quick wins before slow ones
  4. **Prove** — frame the POC/benchmark that demonstrates the win
- Claude grades each move (1–5) and logs to `scorecard.md`.

## Grading rubric (per scenario)
| Move | What "great" looks like |
|---|---|
| Discovery | Asks online vs offline, workload token shape (incl. p95), latency SLA, current setup/cost, quality bar, volume. Doesn't jump to solutions before understanding. |
| Diagnose | Correctly identifies the binding constraint (memory/concurrency, prefill/TTFT, decode/TPOT, queue/infra). |
| Recommend | Right levers in right order (quick: quant, batch tune, bigger-mem GPU, caching; slow: distill/smaller model). States the tradeoff for each. Validates quality on their evals. |
| Prove | Designs a benchmark that measures the metric that matters at their SLA; frames result as cost/latency vs their baseline. |
| Comms | Plain language, no jargon dumps, confirms understanding, leads with their goal. |

## The "golden discovery questions" (SA muscle memory)
1. Is this **online** (user waiting) or **offline** (batch job)?
2. What are typical **input and output token lengths** — and the **p95**?
3. What's the **latency SLA** (TTFT and TPOT/total)?
4. What are they running **today** (model, GPU, precision, framework) and what's it **costing**?
5. What's their **quality bar**, and do they have **evals**?
6. What's the **volume** (req/day, tokens/day) and the **traffic shape** (spiky? steady?)?
7. What does **success** look like for this POC?

---

## Scenario 1 — "The expensive chatbot"  (online, right-sizing)

**Opening line (what the SA hears):**
> "Our customer-support chatbot works, but our GPU bill is brutal — like $40k/month
>  and climbing. Leadership wants it cut. Can Baseten make it cheaper without making
>  it feel slow?"

**Hidden facts (Claude reveals only when asked):**
- Model: Llama-3 70B, running **fp16**, on **2× H100** per replica (TP=2), 4 replicas.
- Precision: never quantized ("we assumed fp16 = best quality").
- Workload: support chat. Input ~300 tokens, output ~200 tokens. p95 input ~800.
- SLA: TTFT < 1s, smooth streaming. Currently TTFT ~400ms, TPOT ~30ms (plenty of headroom).
- Concurrency: each replica handles ~6 concurrent before slowing. Often underutilized.
- Volume: ~500k requests/day, steady daytime traffic, near-zero overnight.
- Quality: they have a 200-example support-quality eval set (good!).
- Cost today: ~$40k/mo.

**Ideal path:** fp16→fp8 (halves to 1× H100, validate on their eval set) is the
headline win; they have huge latency headroom so quality-validated quantization is
low-risk. Scale-to-zero / scale-down overnight (steady-day, dead-night traffic).
Right-size concurrency target via a benchmark. Possible smaller/distilled model as
strategic follow-up. POC: deploy fp8 on 1× H100, run their eval set for quality +
a concurrency sweep for cost/latency, present $/mo vs the $40k baseline.

---

## Scenario 2 — "The sluggish RAG assistant"  (online, prefill/TTFT)

**Opening line:**
> "We built an internal knowledge assistant over our docs. People complain it takes
>  5–8 seconds before it starts answering. The actual answer streams fine once it starts.
>  Help."

**Hidden facts:**
- "Starts slow, streams fine" = TTFT problem, TPOT is okay → prefill/queue, not decode.
- RAG stuffs ~16k–24k tokens of retrieved context into every prompt. Output ~300 tokens.
- Same long system+context prefix is reused across many queries (cacheable!).
- Model: Qwen2.5 32B, fp8, 1× H100. Inference-only TTFT measured ~5s; end-to-end ~6s.
- Traffic: bursty (whole team hits it 9am). Min replicas = 1.
- No prompt/prefix caching configured.

**Ideal path:** Diagnose prefill-bound TTFT (huge input). Levers: **prefix caching**
for the shared context (biggest TTFT win), shorten/retrieve fewer tokens, consider
**disaggregation** if volume grows (prefill-heavy). Check the burst → raise min replicas
to kill cold-start queueing at 9am. POC: enable prefix caching, measure TTFT before/after
on their real prompts.

---

## Scenario 3 — "The overnight transcription pipeline"  (offline, throughput/cost)

**Opening line:**
> "We transcribe ~2 million minutes of call-center audio. It's a nightly batch job,
>  nobody's waiting on any single file. Right now it's slow and expensive. Go."

**Hidden facts:**
- Offline! No latency SLA on individual requests — only total job time + cost matter.
- Model: Whisper large-v3. Currently batch size tiny (running like online).
- They've been optimizing TTFT/latency by habit — wrong metric for this job.
- Goal: finish before 6am, minimize cost.

**Ideal path:** Flip the framing — this is **offline/throughput**, so **max batch size**,
maximize total TPS, accept high per-request latency. Larger/cheaper batched deployment,
possibly cheaper GPU at high utilization. POC: measure total throughput (files/hour) and
$/1000 min at large batch vs their current setup.

---

## Scenario 4 — "We'll just self-host vLLM"  (competitive/closing)

**Opening line:**
> "Honestly, my team can just run vLLM on our own H100s in our cloud account.
>  Why would we pay Baseten a markup on top of GPUs we already have?"

**Hidden facts:**
- They have a capable ML team but it's small (3 people) and busy.
- They underestimate ops: autoscaling, multi-region, cold starts, capacity, uptime.
- Traffic is spiky; they currently over-provision (idle GPUs) to be safe.
- No continuous-batching tuning expertise; running near batch=1–2.

**Ideal path:** Don't bash vLLM (Baseten uses it). Reframe: raw GPU ≠ production
inference. Value = autoscaling + multi-cloud capacity + cache-aware routing +
engine selection + zero-downtime deploys + 99.99% uptime + the tuning expertise
(the "77 configs" story) — i.e., **perf/$ and the ops burden they'd carry**.
Quantify: their idle over-provisioning vs Baseten's autoscaling; eng-time cost.
POC: benchmark their current self-host vs Baseten on cost/token at equal SLA.
