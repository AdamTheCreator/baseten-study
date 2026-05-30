# 06 — Competitive Landscape & On-Prem Comparison

> As an SA, you'll constantly be asked "why not just do this ourselves?"
> or "how does this compare to X?" This section prepares you for those
> conversations.

---

## Baseten vs. Self-Hosted (vLLM on Your Own GPUs)

This is your most common competitor. Many ML teams already run vLLM on
rented GPUs (AWS, GCP, Lambda Labs, CoreWeave).

### What they have to manage themselves:

```
Self-Hosted Stack (what the customer maintains):
┌─────────────────────────────────────┐
│ Application Layer                   │
│ - Load balancer (nginx/envoy)       │
│ - Request routing                   │
│ - Rate limiting                     │
│ - Authentication                    │
├─────────────────────────────────────┤
│ Serving Framework                   │
│ - vLLM / TGI installation          │
│ - Model weight management           │
│ - Quantization pipeline             │
│ - Continuous batching config        │
├─────────────────────────────────────┤
│ Container Orchestration             │
│ - Kubernetes cluster                │
│ - GPU device plugin                 │
│ - Node scheduling                   │
│ - Pod autoscaler (custom metrics)   │
│ - Health checks & restarts          │
├─────────────────────────────────────┤
│ Infrastructure                      │
│ - GPU procurement / reservation     │
│ - CUDA driver management            │
│ - Network (NVLink for TP)           │
│ - Storage (model weights)           │
│ - Monitoring (Prometheus/Grafana)   │
└─────────────────────────────────────┘

Baseten Stack (what the customer maintains):
┌─────────────────────────────────────┐
│ config.yaml (10 lines)             │
│ + API key                          │
└─────────────────────────────────────┘
```

### The TCO (Total Cost of Ownership) argument:

```
Self-hosted costs that customers often forget:

GPU rental:                    $6.50/hr per H100 (same as Baseten)
+ Kubernetes ops engineer:     $180k/yr salary = ~$86/hr
+ GPU idle time:               30-60% of hours (no scale-to-zero)
+ Over-provisioning:           20-30% headroom for spikes
+ CUDA/driver incidents:       4-8 hrs/month debugging
+ Scaling engineering:         Custom autoscaler development & maintenance
+ Monitoring stack:            Prometheus + Grafana setup & maintenance
+ On-call rotation:            Weekend/night coverage for GPU failures

Conservative estimate:
- 4 H100s, 70% utilization, 1 part-time ops engineer
- GPU: 4 × $6.50 × 24 × 30 = $18,720/mo
- Ops: $15,000/mo (0.5 FTE)
- Overprovisioning waste: ~$5,600/mo
- Total: ~$39,320/mo

Baseten equivalent:
- 4 H100s peak, autoscaling, scale-to-zero off-hours
- Effective utilization: ~85%
- GPU: ~$14,000/mo (less idle time)
- Ops: $0
- Total: ~$14,000/mo

Savings: ~$25,000/mo = $300,000/yr
```

### Performance argument:

```
vLLM (self-hosted)              vs.    Baseten (TRT-LLM + DFlash)
─────────────────                      ──────────────────────────
PyTorch execution                      Compiled TRT-LLM kernels
Flash Attention 2                      DFlash (custom, 3x faster claimed)
Manual quantization                    Automated calibration pipeline
Basic KV cache management             Optimized KV cache with paging
No speculative decoding (by default)   Lookahead decoding built-in
Manual batching config                 Adaptive continuous batching

Typical improvement: 30-50% higher throughput at same hardware
= 30-50% lower cost per token
```

### When self-hosted actually wins:

Be honest about this — credibility matters more than winning every point.

- **Extreme customization**: Custom CUDA kernels, research workloads
- **Data sovereignty**: When data cannot leave the customer's VPC at all
  (but note: Baseten offers self-hosted/hybrid for this)
- **Existing infrastructure**: If they already have idle GPU capacity paid for
- **Very low volume**: If they run <10 requests/day, the complexity of
  any platform is overkill — just use a cloud API

---

## Baseten vs. Together AI

### Together's approach:
- Serverless inference API (OpenAI-compatible)
- Pre-deployed popular models (Llama, Mixtral, etc.)
- Simple per-token pricing
- Some fine-tuning support

### Where Baseten wins:
| Factor | Baseten | Together AI |
|--------|---------|-------------|
| Custom model deployment | Full Truss framework | Limited to supported models |
| Optimization control | TRT-LLM, DFlash, quantization options | Black box |
| Compound AI | Chains framework | Not available |
| Dedicated compute | Yes (no noisy neighbors) | Shared infrastructure |
| Training integration | Loops (SFT, RL, DPO) | Basic fine-tuning only |
| Enterprise | Self-hosted, HIPAA, custom SLAs | Limited |

### Where Together wins:
- Simpler — just an API call, no deployment needed
- Competitive pricing on popular models
- Faster time-to-first-request (no deployment wait)

### SA positioning:
"Together is great for getting started. Baseten is where you go when you
need control — custom models, dedicated compute, optimization for your
specific workload, and a platform that scales with you."

---

## Baseten vs. Replicate

### Replicate's approach:
- Model marketplace (largest selection of pre-packaged models)
- Community-contributed model implementations (Cog framework)
- Per-second billing
- Focus on ease of use

### Where Baseten wins:
| Factor | Baseten | Replicate |
|--------|---------|-----------|
| Inference optimization | TRT-LLM compiled, DFlash | Whatever the model author packaged |
| Performance consistency | Dedicated infrastructure | Variable (shared, community models) |
| Enterprise readiness | SOC2, HIPAA, self-hosted | SOC2 only |
| Training | Loops SDK | None |
| Custom engines | TRT-LLM, vLLM, SGLang | Cog only |
| Compound AI | Chains | Not available |

### Where Replicate wins:
- Largest model catalog (thousands of models)
- Community ecosystem (someone already packaged most models)
- Great for prototyping (one-click deploy)
- Better for non-LLM workloads (image, audio, video models)

### SA positioning:
"Replicate is a model marketplace — great for trying models quickly.
Baseten is an inference platform — built for running models in production
with optimized performance and enterprise reliability."

---

## Baseten vs. Modal

### Modal's approach:
- General-purpose serverless GPU compute
- Python-native (decorators to define functions)
- Pay-per-second for any compute (not just ML)
- Great developer experience

### Where Baseten wins:
| Factor | Baseten | Modal |
|--------|---------|-------|
| Inference optimization | TRT-LLM, DFlash (purpose-built) | DIY (bring your own engine) |
| LLM-specific features | Continuous batching, KV cache mgmt | You build it |
| Model deployment | Config-only (no code for standard models) | Always requires code |
| OpenAI compatibility | Native | DIY |
| Autoscaling | Inference-aware (concurrency, GPU util) | Generic (request count) |

### Where Modal wins:
- More flexible (any compute, not just ML inference)
- Better for training workloads (raw GPU access)
- Developer experience is excellent
- Good for non-serving ML workloads (batch processing, data pipelines)

### SA positioning:
"Modal is a compute platform that happens to support ML. Baseten is an
inference platform that's purpose-built for running models. If your job is
to serve models in production, purpose-built wins."

---

## Baseten vs. AWS SageMaker / GCP Vertex AI

### Cloud provider approach:
- Part of a broader ML platform (training, data, serving, monitoring)
- Deep integration with cloud ecosystem
- Enterprise support and compliance
- Often complex to configure

### Where Baseten wins:
| Factor | Baseten | SageMaker/Vertex |
|--------|---------|-----------------|
| Deployment speed | Minutes (truss push) | Hours (Terraform + container build) |
| Optimization | TRT-LLM, DFlash | Basic TRT-LLM (Vertex) or none |
| Pricing | Per-minute, scale-to-zero | Endpoint minimums, hourly billing |
| Developer experience | Truss CLI, simple config | CloudFormation, IAM, VPC config |
| Vendor lock-in | OpenAI-compatible API, Truss is open-source | Proprietary APIs |
| Time to production | Days | Weeks |

### Where cloud providers win:
- Existing cloud commitment (credits, enterprise agreement)
- Regulatory requirements that mandate specific cloud providers
- Deep integration with data pipeline (S3 → SageMaker → Redshift)
- Existing team expertise in the cloud platform

### SA positioning:
"If you're already deep in AWS/GCP and just need to add inference, consider
the native option. But if inference is a core part of your product and
performance matters, a purpose-built platform will get you to production
faster and run more efficiently."

---

## Baseten vs. RunPod

### RunPod's approach:
- Cheapest raw GPU rental
- Both persistent and serverless options
- Minimal abstraction — you get a GPU and build everything else
- Community templates

### Where Baseten wins:
| Factor | Baseten | RunPod |
|--------|---------|--------|
| Optimization | TRT-LLM, DFlash, quantization pipeline | None (DIY) |
| Managed serving | Yes (autoscaling, routing, monitoring) | Serverless only (limited) |
| Enterprise | SOC2, HIPAA, SLAs | Limited |
| Developer experience | Config-only deployment | Docker containers, manual setup |

### Where RunPod wins:
- Cheapest GPU pricing (often 20-40% cheaper per GPU hour)
- Most GPU variety (including consumer GPUs like 4090)
- No minimum commitment
- Good for training and experimentation

### SA positioning:
"RunPod gives you cheap GPUs. Baseten gives you optimized inference.
The cheapest GPU doesn't always give you the cheapest inference — a $6.50/hr
H100 on Baseten serving 50% more tokens/sec is cheaper per token than a
$4/hr H100 on RunPod running unoptimized vLLM."

---

## The On-Prem Conversation (When a Customer Wants to Build In-House)

### Common customer objections and responses:

**"We have GPU capacity sitting idle"**
→ "Great — let's benchmark against your idle capacity. If your GPUs are
A100s and you're running vLLM, we can typically show 30-50% more throughput
with TRT-LLM compilation alone. That means your existing GPUs serve more
requests, and Baseten handles the overflow with autoscaling."

**"We need data to stay in our environment"**
→ "Baseten offers self-hosted deployment. Same platform, same optimizations,
runs in your VPC. You get the performance without data leaving your network."

**"GPU cloud is too expensive"**
→ "What's your current GPU utilization? Most self-hosted setups run at
40-60% utilization. With scale-to-zero and autoscaling, Baseten only charges
when GPUs are serving requests. That typically cuts your effective GPU cost
by 30-50%."

**"We don't want vendor lock-in"**
→ "Three points: (1) Truss is open-source — your model packaging works
anywhere. (2) The API is OpenAI-compatible — switching is one line of code.
(3) We earn your business every month. If we're not providing value, you
can leave without migration pain."

**"Our ML team can do this"**
→ "Absolutely — and they should focus on model quality and product features.
The question is whether building and maintaining inference infrastructure
is the best use of their time. We have a team of kernel engineers who spend
100% of their time on inference optimization. Your team shouldn't have to
become infrastructure experts."

---

## Competitive Summary Cheat Sheet

```
Customer says:              Recommend:           Over:
"Just need a quick API"     Together / Replicate  Baseten (overkill)
"Running vLLM, want better" Baseten               Self-hosted
"Need cheapest GPUs"        RunPod                Baseten (if cost is #1)
"Deep in AWS already"       Depends               Need to evaluate
"Building compound AI"      Baseten (Chains)      Everyone else
"Need to fine-tune + serve" Baseten (Loops)       Modal or RunPod
"Enterprise compliance"     Baseten               Replicate, RunPod
"Maximum performance"       Baseten (TRT-LLM)     Everyone else
"General GPU compute"       Modal                 Baseten (too specific)
```
