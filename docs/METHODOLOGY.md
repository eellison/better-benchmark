# Methodology

## Extraction

We hook into `torch._inductor.scheduler.Scheduler.__init__` during `torch.compile` to capture
the fused IR nodes before codegen. Each captured region is serialized as a standalone FX graph
module (`Repro` class + `make_inputs()`) that can be compiled independently.

### Models

- **HF Dynamo**: 26 models from `benchmarks/dynamo/common.py` via `python extract_reductions.py "dynamo:all" --mode aten --inference-only`
- **vLLM**: 7 models extracted separately via vLLM model configs

### Deduplication

Repros are deduplicated by hashing the FX graph structure (op sequence + shapes), ignoring
concrete tensor values. This collapses identical attention layers across transformer blocks
into a single repro.

## Probing

Each deduplicated repro is compiled in a subprocess (to isolate CUDA errors) and we count
`torch._inductor.metrics.generated_kernel_count`. Subprocess isolation is critical — one
hanging or crashing repro must not block others.

Probing is parallelized across 4 GPUs using `CUDA_VISIBLE_DEVICES`.

### Results

- 442 total inference repros across 33 models
- 423 compiled successfully
- 59 produced >1 kernel (14%)
- 19 failed (CUDA OOM, timeout, or import errors)

## Classification

For each multi-kernel repro, we run with `TORCH_LOGS="fusion"` to capture the scheduler's
"cannot fuse" reasons. We then read the actual FX graph code to understand the semantic
pattern (RoPE, cross-entropy, MoE, etc.) and map each to a root cause category.

The classification is done by reading:
1. The FX graph (what ops are being computed)
2. The fusion log (why the scheduler couldn't fuse)
3. The scheduler IR dump (iteration domains, memory deps, var_ranges)

## Reproducing

```bash
# Step 1: Extract regions from HF models
python scripts/extract_reductions.py "dynamo:all" --mode aten --inference-only

# Step 2: Probe kernel counts
python scripts/probe_batch.py filelist.txt --device 0 --output results.json

# Step 3: Classify fusion failures
python scripts/classify_fusion.py

# Step 4: Investigate individual repros
python scripts/run_with_logs.py repros/multi_kernel_inference/<repro>.py 2>logs.txt
```
