# Scorecard

Running log of quiz attempts. Resume a session by reading this file and
continuing from the last question, prioritizing weak areas.

## Summary

- **Questions attempted:** 5
- **Strong areas:** prefill/decode split, KV cache origin, SA diagnostic instinct, leading with the precision question, FP16/FP8 memory + 50% headroom math, distillation lifecycle, per-request KV math + concurrency division (NAILED 5/5), fp8_kv lever reasoning
- **Weak areas to revisit:** TTFT vs TPOT mapping; reach for fast levers before slow ones; **BATCHING TRADEOFF (Q4, 2/5) — bigger batch = cheaper but slower per user is PHYSICS not misconfig; "increase batch to fix slowness" was backwards; weight-read amortization; online vs offline decides batch size; raise batch until TPOT SLA then stop**

## Log

| # | Q | Theme | Score (/5) | Notes |
|---|---|-------|-----------|-------|
| 1 | Two phases + SA diagnostic | Fundamentals/SA | 4 | Got prefill/decode + bottlenecks + diagnostic instinct. Slip: "both responsible for TTFT" — prefill=TTFT, decode=TPOT. |
| 2 | 70B-on-H100 concurrency call | KV cache/SA | 4 | (a) FP16/FP8 + 50% headroom math flawless; led with precision Q. (b) right intuition, missing per-request KV scaling math + "spans both phases". (c) distillation valid but led with slowest lever; missed FP4/bigger-GPU/KV-quant quick wins. |
| 3 | KV math micro-drill (40L/8KVh/128/fp16) | KV cache math | 5 | (a) 0.16384MB ✅ (b) 1.28GB ✅ (c) ~15.6 ✅ (d) fp8 doubles → ~31, said ~33 close; reasoning perfect. KV math locked in. |
| 4 | Batching throughput vs latency call | Batching/SA | 2 | Conflated tradeoff with static/dynamic config; "increase batch to fix slow tokens" backwards. Taught: weight-read amortization (batch↑ → throughput↑/cost↓ but TPOT↑); online vs offline decides; raise batch to TPOT SLA then stop. Re-test recommended. |
| 5 | Coding assistant + speculative decoding | Speculation/SA | 4 | (a) low batch = spare compute ✅ (b) n-gram/lookahead, output≈input ✅ nailed both. (c) couldn't find caveat. Taught: speculation helps TPS not TTFT; gain only on input-matching spans (edit points fall back); temperature lowers acceptance; auto-disables at high batch. |

## KEY SA LESSON (learned in Roleplay 1)
**Qualify a workload on TWO axes, not one:**
- VOLUME (how many): requests/day, peak req/s, traffic shape → drives # GPUs/replicas
- SIZE (how big): input + output tokens/request, avg AND p95 → drives KV cache, concurrency, memory
- Volume without token-size is useless for sizing. Same 500k req/day is tiny at 500 tok vs memory-bound at 24k tok (RAG).
- Ask input & output SEPARATELY: input→prefill/TTFT+starting KV; output→decode/slot-time+growing KV.
- Mnemonic: "How many, how big, how fast, how good."
- Cost model: Monthly cost = #replicas × GPUs/replica × $/GPU-hr × hours.
  - Quantize → fewer GPUs/replica; Autoscale → fewer hours; Smaller model → both; Batch tune → fewer replicas.
- Little's Law: concurrency = arrival_rate × slot_time; slot_time = TTFT + output_tokens×TPOT.

## Roleplay log
| # | Scenario | Discovery | Diagnose | Recommend | Comms | Notes |
|---|----------|-----------|----------|-----------|-------|-------|
| 1 | Expensive chatbot (in progress) | strong (asked deployment, hw, volume, traffic, model, SLA, precision, evals) | fp16→fp8 right | quant + overnight scale-down ✅; MIG misapplied; didn't quantify $ at first | good, collaborative | MISSED token-length discovery Q; reached for KV math before having token size. Paused to teach cost model. |

## Side topics covered (deep dives, not scored)
- TTFT = queue + prefill; inference-time vs end-to-end (book p39); diagram saved to ttft_anatomy.png
- p50→p99 gap = infra/variance (queue, cold starts, autoscaling), not hardware
- AWQ only in book's Recommended Reading (p251); body uses FP8/NVFP4/MXFP8
- Distillation lifecycle: trained teacher → distill into pretrained student → optional fine-tune → redeploy (offline, new version)
