# 02 — Baseten Core Pillars

> Baseten's platform is built around four pillars. Understanding each deeply
> is critical for the SA role — you need to speak to how each creates value
> and how they work together.

---

## Pillar 1: Model Performance

**What it is**: The inference engine layer — how Baseten makes models run
faster than what you'd get running the same model yourself.

### Key Technologies:

#### TensorRT-LLM (TRT-LLM) Engine
Baseten compiles models with NVIDIA's TensorRT-LLM, which:
- Fuses operations (combines multiple GPU kernels into one)
- Optimizes memory layout for the specific GPU architecture
- Enables INT8/FP8 computation on Tensor Cores
- Adds inflight batching (continuous batching at the engine level)

**Why this matters vs. vLLM**: vLLM is a great serving framework, but
it runs models in PyTorch (interpreted). TRT-LLM compiles models into
optimized GPU code. The difference is like Python vs C — same algorithm,
10-40% faster execution.

**The tradeoff**: Compilation takes time (10-60 min depending on model size).
Baseten handles this during `truss push` — the customer never sees it.

#### DFlash (Baseten's Custom Attention Kernel)
- Custom CUDA kernel for attention computation
- Claims 3x faster inference on supported models
- Replaces standard Flash Attention with optimized memory access patterns
- This is proprietary — you can't get this running vLLM yourself

**SA talking point**: "DFlash is custom CUDA code our kernel engineers wrote
specifically for inference workloads. It's not available in any open-source
framework. This is the kind of optimization that's only economical to build
when you're running inference at Baseten's scale."

#### Quantization Pipeline
- Baseten handles calibration data and quantization automatically
- Customer specifies `fp8` or `fp4` in config — Baseten does the rest
- On-prem, you'd need to: select calibration data, run quantization,
  validate quality, handle edge cases. Hours of work per model.

#### Speculative / Lookahead Decoding
- Built into the engine, configurable via config.yaml
- Particularly effective for JSON/code generation (predictable patterns)
- Customer just sets `lookahead_windows_size: 3` — no code changes

### How to Talk About Model Performance:

**To an ML engineer**: "We compile with TRT-LLM, run DFlash for attention,
and handle quantization calibration. You get the performance of a team
that spent 6 months on kernel optimization, deployed in one config change."

**To a VP of Engineering**: "Same model, same quality, 30-50% fewer GPUs
needed. That's the model performance layer."

**To a CFO**: "We make each GPU do more work. Fewer GPUs = lower bill."

---

## Pillar 2: MCM / Infrastructure (Multi-Cluster Management)

**What it is**: The infrastructure layer that runs the GPUs, manages
autoscaling, handles networking, and keeps everything reliable.

### Key Capabilities:

#### GPU Fleet Management
- Access to T4, L4, A10G, A100, H100, B200 GPUs
- Multi-region availability
- Baseten manages the physical infrastructure, CUDA drivers, container runtime
- Customer never SSH's into a machine

**vs. On-Prem**: When you run on-prem, you manage:
- GPU procurement (6-12 month lead times for H100s)
- Data center space, power, cooling
- CUDA driver versions (breaking changes are common)
- Container orchestration (Kubernetes + GPU operator + device plugin)
- Node health monitoring (GPUs fail silently — ECC errors, thermal throttling)
- Network topology for multi-GPU (NVLink, InfiniBand)

None of this is the customer's problem on Baseten.

#### Autoscaling
```
The autoscaling equation:

Scale up when: load > replicas × concurrency_target × target_utilization

Example:
- 50 concurrent requests
- concurrency_target = 32
- target_utilization = 70%
- Current replicas = 2

Check: 50 > 2 × 32 × 0.7 = 44.8 → YES, scale up

New replicas needed: ceil(50 / (32 × 0.7)) = ceil(2.23) = 3
```

**Configurable parameters**:
- `concurrency_target`: Requests per replica before scaling (LLMs: 32-128)
- `target_utilization`: Headroom buffer (default 70%, lower = more spare capacity)
- `scale_down_delay`: Seconds before removing an idle replica (default 900s)
- `min_replicas`: Floor (0 = scale-to-zero, ≥1 = always warm)
- `max_replicas`: Ceiling (cost cap)

**vs. Self-managed autoscaling**: Building this with Kubernetes HPA is
possible but painful. K8s HPA doesn't understand GPU utilization natively.
You need custom metrics exporters, GPU-aware scheduling, and careful
tuning to avoid oscillation. Baseten's autoscaler is purpose-built for
inference workloads.

#### Networking & Routing
- Load balancing across replicas
- Request queuing with backpressure
- Health checks and automatic replica replacement
- GPU-to-GPU communication for tensor-parallel models (NVLink)
- Edge routing for low-latency global access

#### Reliability
- 99.99% uptime SLA (Enterprise)
- Automatic replica replacement on GPU failure
- No single point of failure in the serving path
- Deployment versioning with instant rollback

### How to Talk About Infra:

**To an ML engineer**: "You write `config.yaml`, we handle everything below
the model. CUDA, containers, scaling, networking, GPU failures — not your
problem."

**To a Platform engineer**: "We replace your Kubernetes GPU cluster, custom
autoscaler, and serving infrastructure with a managed service. Your team
stops being on-call for GPU driver updates."

**To a CTO**: "Your ML engineers spend their time on model quality instead
of GPU operations. We handle the undifferentiated heavy lifting."

---

## Pillar 3: DevUI & Truss

**What it is**: The developer experience layer — how engineers interact
with Baseten day-to-day.

### Truss (The CLI & Framework)

#### What makes Truss good:

1. **Config-over-code for common cases**
   ```yaml
   # This is a complete deployment config for Llama 3.1 70B:
   model_name: meta-llama/Llama-3.1-70B-Instruct
   runtime:
     predict_concurrency: 48
   resources:
     accelerator: H100:2
     use_gpu: true
   trt_llm:
     build:
       base_model: llama
       quantization_type: fp8
       max_seq_len: 8192
   ```
   No Dockerfile. No Python. No Kubernetes manifests.

2. **Escape hatch for custom logic**
   ```python
   # model/model.py — when you need custom pre/post-processing
   class Model:
       def __init__(self):
           self._model = None

       def load(self):
           # Load your model however you want
           self._model = load_my_model()

       def predict(self, request):
           # Custom preprocessing
           inputs = self.preprocess(request["prompt"])
           # Run inference
           output = self._model.generate(inputs)
           # Custom postprocessing
           return self.postprocess(output)
   ```

3. **Dev iteration loop**
   ```bash
   truss push --watch  # Live reload on file changes
   # Edit config.yaml or model.py → changes deploy automatically
   # Like `next dev` but for ML models
   ```

4. **Production promotion**
   ```bash
   truss push           # Deploy to development
   # Test, validate, benchmark
   truss push --promote # Promote to production (stable endpoint)
   ```

#### vs. Competitors:

| Feature | Truss (Baseten) | Cog (Replicate) | Modal | SageMaker |
|---------|----------------|-----------------|-------|-----------|
| Config-only deploy | Yes | No (needs predict.py) | No (needs Python) | No (needs container) |
| Live reload | Yes (--watch) | No | Yes | No |
| Local testing | Yes | Yes | Yes | Painful |
| CI/CD integration | GitHub Action | No official | No official | CodePipeline |
| Versioned deploys | Yes | Yes | Yes | Yes |
| Engine selection | Yes (TRT-LLM, vLLM) | No | No | Limited |

### The Dashboard (DevUI)

#### What it provides:
- **Deployment management**: Version history, promote/rollback, environment management
- **Real-time metrics**: TTFT, throughput, errors, GPU utilization, queue depth
- **Autoscaling controls**: Visual configuration of scaling parameters
- **Logs**: Streaming container logs, filterable by replica
- **API playground**: Test your model directly in the browser
- **Cost tracking**: Per-deployment cost breakdown

#### Why the dashboard matters for SAs:
During a POC, you share the dashboard with the customer. They can see:
- Real-time performance metrics (no custom Grafana setup needed)
- How autoscaling responds to load
- Cost accumulating per minute
- Side-by-side comparison of deployment versions

This is a sales tool as much as a dev tool. The visual proof of performance
is more convincing than a spreadsheet.

### Baseten Chains (Compound AI)

For multi-step AI pipelines:

```python
# Example: Document processing pipeline
# Step 1 (CPU): Parse PDF → text
# Step 2 (GPU, small model): Classify document type
# Step 3 (GPU, large model): Extract entities based on doc type
# Step 4 (CPU): Format and return results

# Each step gets its own hardware and autoscaling
# Step 1 scales independently of Step 3
# GPU-intensive steps get H100s, CPU steps get cheap instances
```

**vs. doing this yourself**: You'd need a message queue (SQS/Kafka),
separate services for each step, independent scaling for each, retry logic,
error handling, and observability across the pipeline. Chains gives you
type-safe interfaces, per-step scaling, and unified observability.

---

## Pillar 4: Post-Training

**What it is**: Fine-tuning and reinforcement learning on Baseten's
infrastructure, with seamless deployment to inference.

### Baseten Loops (Training SDK)

#### Supported workflows:
- **SFT (Supervised Fine-Tuning)**: Train on instruction/response pairs
- **DPO (Direct Preference Optimization)**: Learn from human preferences
- **GRPO (Group Relative Policy Optimization)**: RL with custom rewards
- **LoRA/QLoRA**: Parameter-efficient fine-tuning (less GPU memory)

#### Supported frameworks:
- **Axolotl**: Config-driven, beginner-friendly
- **TRL**: Hugging Face's training library
- **VeRL**: RL-specific training
- **MS-Swift**: Long-context and multilingual

#### The training-to-inference pipeline:

```bash
# 1. Configure training
# training_config.yaml with model, dataset, hyperparameters

# 2. Launch training job
truss train run --config training_config.yaml

# 3. Monitor (SSH access, VS Code remote attach available)
truss train status --job-id <job_id>

# 4. Deploy checkpoint directly to inference
truss train deploy_checkpoints --job-id <job_id>

# 5. Checkpoint is now a live API endpoint
# No weight transfer, no container rebuild, no waiting
```

**Why this matters**: The typical self-managed workflow is:
1. Train on GPU cluster A (Lambda Labs, AWS)
2. Download weights (hours for large models)
3. Upload weights to serving infra
4. Rebuild serving container
5. Deploy and test

Steps 2-4 can take half a day for a 70B model. On Baseten, the checkpoint
is already on the right infrastructure — step 4 above goes directly to
a live endpoint.

### BDN (Baseten Delivery Network)

- CDN for model weights
- Weights loaded from HuggingFace, S3, or GCS at deploy time
- Not bundled in the container (faster builds)
- Cached across the fleet (fast cold starts after first deploy)

**Why this matters for post-training**: Your fine-tuned weights are stored on
BDN. When you deploy a new checkpoint, it loads in seconds (cached on the
serving nodes) rather than minutes (pulling from S3).

### How to Talk About Post-Training:

**To an ML researcher**: "Train with whatever framework you know. We handle
the GPU orchestration, checkpointing, and deployment. You SSH in and debug
like it's your local machine."

**To a VP of ML**: "The typical train-to-deploy cycle is 4-8 hours of
engineering time per iteration. With Loops, it's one command. Your team
iterates 3x faster."

**To a customer evaluating fine-tuning**: "Let's fine-tune the base model
on your data during the POC. I'll set up a LoRA training job, and we'll
have a deployed endpoint with your fine-tuned model by end of day."
