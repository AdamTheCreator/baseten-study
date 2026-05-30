# 01 — The Customer Journey (End-to-End)

> This maps the full lifecycle a customer goes through with Baseten,
> from discovery to production. At each stage, you'll see what the SA
> does, what the customer cares about, and how Baseten differentiates.

---

## Stage 0: The Customer's Starting Point

Before Baseten, a typical customer is in one of these situations:

### Situation A: Running inference on-prem or self-managed cloud
- Running vLLM/TGI on rented GPUs (AWS, GCP, Lambda Labs, RunPod)
- Engineering team managing CUDA drivers, container orchestration, autoscaling
- Spending 30-50% of ML engineer time on infra, not model work
- **Pain**: Ops overhead, scaling challenges, no optimization expertise

### Situation B: Using a model API provider
- Calling OpenAI/Anthropic/Together AI APIs
- Simple but expensive at scale, no customization
- Vendor lock-in concerns, can't run their own fine-tuned models
- **Pain**: Cost at scale, no control, data privacy concerns

### Situation C: Evaluating for a new AI feature
- Building their first AI-powered product
- Don't have ML infra expertise in-house
- Need to move fast but don't want to accumulate tech debt
- **Pain**: Don't know what they don't know

**Your SA lens**: Situation A is your most common and highest-value customer.
They already understand inference and have benchmarks to beat. The POC
is a direct comparison.

---

## Stage 1: Discovery & Qualification

### What happens:
- Customer finds Baseten via: referral, blog post, conference, or outbound
- Initial conversation with sales/SA to understand their workload
- SA qualifies the opportunity by understanding:

### Questions you ask:

```
Workload Understanding:
- What model(s) are you running? (architecture, parameter count)
- What's your current serving stack? (vLLM, TGI, TRT-LLM, custom)
- What hardware are you on? (A100, H100, cloud instance type)
- What are your current latency numbers? (TTFT, p50, p95)
- What's your request volume? (QPS peak and average)
- What's your current cost? ($/month, $/1M tokens)

Use Case Understanding:
- Is this real-time (chat) or batch (document processing)?
- What's your input/output token distribution?
- Do you use structured output (JSON mode)?
- Do you need streaming?
- Any compliance requirements? (HIPAA, SOC2, data residency)

Growth & Goals:
- Where are you headed? (more models, higher volume, fine-tuning)
- What would "success" look like for a Baseten evaluation?
- Who are the decision-makers? (eng lead, CTO, VP Eng)
```

### Why Baseten wins here:
- Free credits to start — low friction to try
- OpenAI-compatible API — minimal code change to switch
- SOC2 + HIPAA out of the box — enterprise customers don't need to wait

---

## Stage 2: The POC (Proof of Concept) — This Is Your Core Job

### POC Structure (typically 1-2 weeks):

#### Day 1-2: Baseline & Deploy

1. **Establish customer's baseline metrics**
   - Get their current TTFT, throughput, p95, cost numbers
   - If they don't have them, help them measure (this builds trust)

2. **Deploy their model on Baseten**
   ```bash
   # If it's a standard model (Llama, Mistral, etc.)
   uvx truss init my-poc
   # Edit config.yaml with model, GPU, quantization
   uvx truss push
   ```

3. **Verify functional correctness**
   - Same inputs produce equivalent outputs
   - Structured output works
   - Streaming works

#### Day 3-5: Optimization Sweep

4. **Run quantization sweep** (see scripts/compare_quantizations.py)
   - fp16 baseline → fp8 → fp4
   - Measure quality degradation on customer's actual prompts
   - Measure throughput improvement

5. **Hardware sweep**
   - Same model on A100 vs H100 vs B200
   - Calculate cost-per-token at each tier
   - Find the price/performance sweet spot

6. **Concurrency tuning**
   - Sweep concurrency targets (1, 8, 16, 32, 64, 128)
   - Plot latency vs throughput curve
   - Find the knee of the curve for their latency SLA

7. **Engine optimization**
   - TensorRT-LLM compilation (Baseten's engine-builder)
   - DFlash attention kernels
   - Lookahead decoding if structured output

#### Day 6-8: Benchmark & Document

8. **Run production-realistic load test**
   - Match customer's actual traffic pattern (QPS, burst, input distribution)
   - Run for sustained period (30-60 min, not just burst)
   - Capture all metrics

9. **Generate comparison report** (see scripts/generate_report.py)
   - Side-by-side: Customer's current vs Baseten optimized
   - Metrics: TTFT, throughput, p95 latency, cost/token, cost/month
   - Include quality evaluation if quantized

#### Day 9-10: Present & Iterate

10. **Present findings to customer's team**
    - Lead with the metric they care about most
    - Show the tradeoff curves (not just one config)
    - Address concerns (data privacy, vendor lock-in, migration effort)

### What makes a POC successful:
- **Concrete numbers**: "23% lower p95 latency at 40% lower cost"
- **Apples-to-apples comparison**: Same model, same prompts, same load
- **Their data**: Run on their actual prompts, not synthetic benchmarks
- **Tradeoff transparency**: Show the options, not just the best case

---

## Stage 3: Migration & Integration

### What happens:
- Customer decides to move forward
- SA helps with integration:

### Integration is usually simple:

```python
# BEFORE (vLLM self-hosted):
client = OpenAI(
    base_url="http://my-vllm-server:8000/v1",
    api_key="not-needed"
)

# AFTER (Baseten):
client = OpenAI(
    base_url="https://bridge.baseten.co/v1/direct",  # or model-specific URL
    api_key=os.environ["BASETEN_API_KEY"]
)
# Everything else stays the same — same SDK, same parameters
```

### Migration checklist:
- [ ] API endpoint swap (usually 1 line of code)
- [ ] API key management (secrets management integration)
- [ ] Error handling (Baseten returns standard HTTP codes)
- [ ] Rate limit awareness (configure based on plan)
- [ ] Streaming compatibility (SSE format, same as OpenAI)
- [ ] Monitoring integration (webhook alerts, Datadog, etc.)
- [ ] Autoscaling configuration (min/max replicas, concurrency)
- [ ] Failover/fallback strategy

### Why Baseten wins at migration:
- OpenAI-compatible API means minimal code changes
- No need to manage CUDA, drivers, containers, orchestration
- Autoscaling is built in (vs. building your own with K8s HPA)
- Versioned deployments (rollback is trivial)

---

## Stage 4: Production Operations

### What the customer gets in production:

**Observability (Dashboard)**:
- Real-time TTFT, throughput, error rate, p50/p95/p99
- GPU utilization and memory per replica
- Queue depth and autoscaling events
- Request/response size distributions
- Container restart tracking

**Autoscaling**:
- Scale-to-zero (no cost when idle)
- Scale-up based on: `load > replicas × concurrency_target × utilization%`
- Configurable scale-down delay (default 900s) with exponential back-off
- Predictive scaling for known traffic patterns

**Deployment Management**:
- Blue/green deployments (new version alongside old)
- Traffic splitting (canary releases)
- Instant rollback to previous version
- Environment promotion (dev → staging → prod)

**Cost Controls**:
- Max replica limits (cost ceiling)
- Scale-to-zero for dev/staging environments
- Per-minute billing (no hourly minimums like cloud providers)

### What the SA does post-sale:
- Quarterly reviews of performance and cost
- Optimization recommendations as models/hardware evolve
- Help with new model deployments
- Escalation path for performance issues

---

## Stage 5: Expansion

### Common expansion paths:
1. **More models**: Customer starts with 1 model, adds more for different tasks
2. **Fine-tuning**: Use Baseten Loops to train on their data, deploy directly
3. **Chains**: Build compound AI pipelines (classify → route → generate)
4. **Higher tiers**: Move from shared to dedicated compute
5. **Enterprise features**: Self-hosted, custom SLAs, data residency

### The SA's role in expansion:
- Identify opportunities from usage patterns
- Run mini-POCs for new models or configurations
- Introduce new Baseten features (DFlash, new GPU types, Chains)
- Connect customer with Baseten's Forward Deployed Engineers for deep optimization

---

## The SA Value-Add at Each Stage

| Stage | Customer Question | Your SA Response |
|-------|-------------------|------------------|
| Discovery | "Why not just use vLLM ourselves?" | "You can — let me show you what you're leaving on the table" |
| POC | "How much faster/cheaper?" | "Here are the numbers on YOUR workload" |
| Migration | "How hard is the switch?" | "One line of code — I'll walk you through it" |
| Production | "Why is p99 spiking?" | "Autoscaling window is too long — let's tune it" |
| Expansion | "We want to add a fine-tuned model" | "Train with Loops, deploy with one command, I'll set it up" |

---

## Differentiators vs. Competitors at Each Stage

### vs. Self-Hosted (vLLM on rented GPUs):
- **POC**: You show TRT-LLM compilation + DFlash beating their vLLM setup
- **Migration**: They stop managing CUDA drivers, Kubernetes, GPU scheduling
- **Production**: Autoscaling, observability, versioned deploys — all built in
- **Cost argument**: "How much does your ML engineer's time cost? $200k/yr?
  That's the hidden cost of self-hosting."

### vs. Together AI / Fireworks:
- **POC**: Deploy their custom/fine-tuned model (Together only supports select models)
- **Production**: Dedicated compute vs shared (noisy neighbor problems)
- **Expansion**: Chains for compound AI, Loops for training (full platform)

### vs. Replicate:
- **POC**: TRT-LLM optimization (Replicate uses whatever the model author packaged)
- **Production**: Enterprise features (SOC2, HIPAA, self-hosted option)
- **Cost**: Per-minute billing vs Replicate's per-second (Baseten is cheaper for sustained load)

### vs. Cloud Providers (AWS SageMaker, GCP Vertex):
- **POC**: Deploy in minutes vs hours/days of CloudFormation/Terraform
- **Migration**: No vendor lock-in to cloud-specific serving formats
- **Production**: Purpose-built for inference vs general-purpose ML platform
- **Cost**: No managed endpoint minimum fees, true scale-to-zero
