# Baseten Solutions Architect — Study Guide & POC Toolkit

> Preparing for a Solutions Architect IC role at Baseten.
> This guide covers the end-to-end customer journey, core platform pillars,
> POC methodology, hardware selection, scripting, and inference engineering fundamentals.

## How to Use This Guide

Work through each section in order. Each builds on the last:

1. **[00 — Inference Engineering Foundations](notes/00-foundations.md)**
   The mental models you need before touching any platform. Covers TTFT, throughput,
   batching, quantization, KV cache, prefill vs decode — the physics of inference.

2. **[01 — The Customer Journey (End-to-End)](notes/01-customer-journey.md)**
   What a customer experiences from first contact through production deployment.
   Maps to the SA role: where you add value at each stage.

3. **[02 — Baseten Core Pillars](notes/02-core-pillars.md)**
   Deep dive into the four product areas: Model Performance, MCM/Infra,
   DevUI & Truss, and Post-Training. What differentiates each from competitors
   and on-prem alternatives.

4. **[03 — Hardware Selection Guide](notes/03-hardware.md)**
   H100 vs B200 vs A100 — how hardware choice affects throughput, latency,
   cost-per-token, and what a customer ultimately selects.

5. **[04 — The POC Playbook](notes/04-poc-playbook.md)**
   How a Solutions Architect runs a proof-of-concept: model selection, deploy,
   optimize, benchmark, and present findings that prove better throughput/latency
   than what the customer has today.

6. **[05 — Scripting for SAs](notes/05-scripting.md)**
   Python and bash scripts for benchmarking, deployment automation, metrics
   collection, and customer-facing reporting. Hands-on examples.

7. **[06 — Competitive Landscape & On-Prem](notes/06-competitive.md)**
   How Baseten compares to running vLLM on your own GPUs, using Replicate/Modal/
   RunPod/Together, and what arguments win deals.

8. **[07 — Blind Spots & Glossary](notes/07-blind-spots.md)**
   Things that trip up people new to inference engineering. Terminology,
   common misconceptions, and interview-relevant gotchas.

9. **[08 — Book Corrections & Additions](notes/08-book-corrections.md)** *(READ THIS FIRST)*
   Cross-referenced against Philip Kiely's "Inference Engineering" (Baseten Books, 2026).
   20 gaps, corrections, and key additions. Covers disaggregation, EAGLE speculation,
   ops:byte ratio, SGLang, NVIDIA Dynamo, quantization sensitivity, cache-aware
   routing, H200/B300 GPUs, MIG, distillation, and more.
   **This is the errata sheet — read alongside the originals.**

## Scripts

All in `scripts/` — runnable examples that demonstrate SA-relevant workflows:

- `deploy_model.py` — Deploy a model via Truss programmatically
- `benchmark.py` — Load test an endpoint, measure TTFT/throughput/p95
- `compare_quantizations.py` — Deploy same model at fp16/fp8/fp4, compare
- `cost_calculator.py` — Calculate cost-per-token for different GPU configs
- `generate_report.py` — Generate a customer-facing POC report from benchmark data
- `health_check.sh` — Quick endpoint health/latency check (bash)

## Quick Start

```bash
# Install dependencies
pip install truss openai requests numpy tabulate matplotlib

# Authenticate with Baseten
uvx truss login

# Run your first benchmark against a pre-optimized model
python scripts/benchmark.py --model "deepseek-v3" --requests 100
```
