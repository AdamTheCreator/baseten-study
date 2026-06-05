# Scorecard

Running log of quiz attempts. Resume a session by reading this file and
continuing from the last question, prioritizing weak areas.

## Summary

- **Questions attempted:** 2
- **Strong areas:** prefill/decode split, KV cache origin, SA diagnostic instinct, **leading with the precision question**, FP16/FP8 memory + 50% headroom math, distillation lifecycle
- **Weak areas to revisit:** TTFT vs TPOT mapping; **per-request KV cache scaling math (concurrency = KV budget ÷ per-request KV)**; KV cache spans BOTH phases (prefill writes, decode reads/grows); reach for fast levers (quant/bigger GPU) before slow ones (distillation)

## Log

| # | Q | Theme | Score (/5) | Notes |
|---|---|-------|-----------|-------|
| 1 | Two phases + SA diagnostic | Fundamentals/SA | 4 | Got prefill/decode + bottlenecks + diagnostic instinct. Slip: "both responsible for TTFT" — prefill=TTFT, decode=TPOT. |
| 2 | 70B-on-H100 concurrency call | KV cache/SA | 4 | (a) FP16/FP8 + 50% headroom math flawless; led with precision Q. (b) right intuition, missing per-request KV scaling math + "spans both phases". (c) distillation valid but led with slowest lever; missed FP4/bigger-GPU/KV-quant quick wins. |

## Side topics covered (deep dives, not scored)
- TTFT = queue + prefill; inference-time vs end-to-end (book p39); diagram saved to ttft_anatomy.png
- p50→p99 gap = infra/variance (queue, cold starts, autoscaling), not hardware
- AWQ only in book's Recommended Reading (p251); body uses FP8/NVFP4/MXFP8
- Distillation lifecycle: trained teacher → distill into pretrained student → optional fine-tune → redeploy (offline, new version)
