# Inductor Kernel Extraction & Benchmarking Framework — Design Document

## Problem Statement

When optimizing PyTorch Inductor's codegen (heuristics, fusion decisions, tiling),
the iteration cycle is slow: modify Inductor → recompile a full model → profile →
identify which kernel regressed/improved. We need a way to:

1. Extract every compilation region from real models as standalone, runnable repros
2. Benchmark each region against speed-of-light (memcopy bandwidth)
3. Classify multi-kernel regions by root cause (what prevented fusion)
4. Re-run the full benchmark suite after a change to measure aggregate impact

The framework targets **reduction kernels** specifically because reductions are
where Inductor's heuristics have the most room for improvement (num_warps,
RBLOCK, persistent vs non-persistent, split reductions, multi-kernel fusion).

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      extract_reductions.py                       │
│                                                                 │
│  ┌──────────────────┐    ┌──────────────────────────────────┐  │
│  │  Scheduler Mode  │    │         ATen Mode                 │  │
│  │  (post-fusion)   │    │  (post-grad, pre-scheduling)      │  │
│  │                  │    │                                    │  │
│  │  Hooks into      │    │  Hooks into                       │  │
│  │  _post_fusion_   │    │  post_grad_custom_pre_pass        │  │
│  │  custom_pass     │    │                                    │  │
│  │                  │    │  Uses CapabilityBasedPartitioner   │  │
│  │  Captures fused  │    │  + is_fusible_node to find        │  │
│  │  scheduler nodes │    │  kernel-sized fusion regions       │  │
│  │  (what Inductor  │    │  (what Inductor COULD fuse)       │  │
│  │  actually fuses) │    │                                    │  │
│  └────────┬─────────┘    └──────────────┬───────────────────┘  │
│           │                              │                       │
│           ▼                              ▼                       │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              ReductionExtractor._extract_subgraph         │   │
│  │                                                           │   │
│  │  1. Collect origin FX nodes from scheduler/partition      │   │
│  │  2. Build new FX graph with only those nodes              │   │
│  │  3. External deps become placeholders (inputs)            │   │
│  │  4. Record shape, dtype, stride, device per placeholder   │   │
│  │  5. Deduplicate by structural hash (ops + shapes)         │   │
│  └─────────────────────────┬───────────────────────────────┘   │
│                             │                                    │
│                             ▼                                    │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │           _generate_standalone_script                      │   │
│  │                                                           │   │
│  │  Emits a .py file containing:                             │   │
│  │    - class Repro(nn.Module): forward(...)                 │   │
│  │    - def make_inputs(): [torch.randn(...), ...]           │   │
│  │    - def benchmark(): compile + measure + SOL comparison  │   │
│  │                                                           │   │
│  │  Non-contiguous inputs use as_strided() to preserve       │   │
│  │  the exact memory layout from the original model.         │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Extraction Modes

### Mode 1: Scheduler (post-fusion)

**What it captures:** The actual fused kernel groups that Inductor's scheduler produces.

**Hook point:** `inductor_config._post_fusion_custom_pass` — a callback invoked after
the scheduler runs fusion, receiving the list of `BaseSchedulerNode` (including
`FusedSchedulerNode` groups).

**When to use:** When you want to see what Inductor *actually* does — the real kernels
it generates. Good for SOL analysis and identifying slow kernels.

**Key detail:** `split_reductions=False` is set to prevent Inductor from splitting
large reductions into multi-pass, which would create artificially many scheduler nodes.

### Mode 2: ATen (pre-scheduling)

**What it captures:** Fusible regions from the post-grad FX graph, as partitioned by
`CapabilityBasedPartitioner` with `is_fusible_node`.

**Hook point:** `inductor_config.post_grad_custom_pre_pass` — captures the FX
`GraphModule` before scheduling begins.

**When to use:** When you want to identify fusion *opportunities* — what *could* be
fused if heuristics were perfect. Good for finding fusion gaps (regions that should
be 1 kernel but become 2+).

**Shared-input merging:** After partitioning, reduction-containing regions that share
a common activation input are merged (Union-Find). This models Inductor's mix-order
reduction fusion: e.g., LayerNorm backward has `sum(tangents, [0,1])` and
`sum(tangents, [2])` that share `tangents_1` and get fused into one kernel.

Merge heuristic: only merge when one partition is small (≤5 ops) to avoid
over-merging in weight-sharing models like ALBERT.

---

## Subgraph Extraction Algorithm

Given a set of FX nodes (from scheduler origins or ATen partitioner):

1. **Needed set** = exactly the origin nodes (no backward BFS — we don't pull in
   upstream ops that would inflate the region beyond what Inductor actually fuses)

2. **Placeholders** = any FX node consumed by a needed node but NOT in the needed
   set. These become inputs to the standalone repro.

3. **Outputs** = needed nodes that have no user within the needed set (leaf nodes).

4. **Deduplication** = hash of `{sorted op targets + input shapes + strides}`.
   Identical attention layers across transformer blocks collapse to one repro.

### Placeholder metadata (preserved per input)

```python
{
    "shape": [4, 64, 512, 513],
    "stride": [16846848, 263232, 513, 1],  # empty [] if contiguous
    "dtype": "torch.float32",
    "device": "cuda:0",
}
```

`stride` is only populated when the tensor is non-contiguous (`not val.is_contiguous()`).
Empty `stride` means contiguous — no redundant boolean needed.

**Why strides matter:** Non-contiguous inputs (from `permute`, `slice`, `as_strided`)
change the memory access pattern, affecting Inductor's tiling decisions and measured
bandwidth. A softmax over `[B, H, S, D]` with stride `[H*S*D, S*D, D, 1]` performs
very differently than the same shape with stride `[H*S*D, D, H*D, 1]` (transposed).

Generated `make_inputs()` uses `as_strided` for non-contiguous tensors:

```python
def make_inputs():
    return [
        torch.randn(67174401, dtype=torch.float32, device='cuda')
            .as_strided([4, 64, 512, 513], [16846848, 263232, 513, 1]),
    ]
```

---

## Benchmarking Methodology

Each standalone repro's `benchmark()` function measures:

### 1. Compiled (default heuristics)
Standard `torch.compile(mod)` with default Inductor config.
25 warmup + 200 timed iterations, wall-clock.

### 2. Compiled (coordinate descent tuning)
`inductor_config.coordinate_descent_tuning = True` — searches over num_warps,
XBLOCK, RBLOCK. Shows if better tile choices exist.

### 3. Memcopy SOL (speed-of-light)
Measures `dst.copy_(src)` at the same byte count as the kernel's total read+write.
This gives the theoretical minimum time if the kernel were perfectly bandwidth-bound
with zero compute overhead.

### 4. SOL gap = compiled_time / sol_time
- Gap ≈ 1.0 → kernel is bandwidth-optimal
- Gap 2-5× → usually launch overhead (for small kernels) or tiling inefficiency
- Gap >5× → significant issue (bad parallelization, strided access, etc.)

### 5. Bandwidth curve
Measures memcopy at multiple sizes (1KB → 256MB) to show where on the GPU's
bandwidth curve this kernel sits. Small kernels can't saturate bandwidth regardless
of code quality.

---

## Batch Runner (`benchmark_all.py`)

Discovers all `fused_*.py` and `region_*.py` files, runs each in a subprocess
(isolation for CUDA errors), collects JSON results, and produces a summary table:

```
Kernel                                                            Default       CD       SOL      Gap
---------------------------------------------------------------------------------------------------------
hf_albert/fused_002_welford_reduce_abc123_def456                    45.3     42.1     38.2    1.19x
llama_like/fused_007_amax_sum_789abc_012def                         67.8     61.2     12.4    5.47x
```

---

## Multi-Kernel Classification Pipeline

For regions producing >1 kernel:

1. **Probe:** `torch._inductor.metrics.generated_kernel_count` — count kernels per repro
2. **Fuse-log:** `TORCH_LOGS="fusion"` — captures scheduler's "cannot fuse" reasons
3. **Classify:** Map each (pattern, reason) pair to a root cause category

### Root cause taxonomy

| Category | Description | Example |
|---|---|---|
| `numel/rnumel mismatch` | Two reductions with different iteration domains | Cross-entropy: softmax [B*S, V] + loss sum [B*S] |
| `no shared data` | Independent computations, no common buffer reads | Q-RoPE and K-RoPE (different head counts) |
| `intermediate nodes` | Scheduling dependency prevents reorder | RMSNorm mean → subsequent mul blocks fusion |
| `NopKernel boundary` | ConcatKernel blocks all fusion | `cat` lowered as NopKernel instead of pointwise_cat |
| `numel incompatibility` | Output shapes differ | Softmax [B,H,S,513] → slice [B,H,S,512] |
| `assert_async` | Side-effect node prevents fusion | MoE bounds-check assertions |

---

## Model Coverage

### HuggingFace Dynamo (via `benchmarks/dynamo/huggingface.py`)

Supports all models from the PyTorch benchmarks dynamo suite:
```bash
python extract_reductions.py dynamo:all --mode aten --inference-only
```

### vLLM models (via custom extraction scripts)

Run vLLM with `TORCH_TRACE`, use `tlparse` to render post-grad graphs,
then feed the model runner through ATen extraction.

### Built-in synthetic models

`simple`, `rmsnorm`, `cross_entropy`, `llama_like`, `bert_like` — for quick
iteration on heuristic changes without downloading model weights.

---

## Usage Examples

```bash
# Extract all fusion regions from a single model
python extract_reductions.py llama_like --mode both

# Extract inference-only regions from all HF dynamo models
python extract_reductions.py dynamo:all --mode aten --inference-only

# Run the full benchmark suite
python benchmark_all.py output/aten_repros/

# Probe kernel counts (parallelized across GPUs)
python scripts/probe_batch.py filelist.txt --device 0 --output results.json

# Investigate a specific multi-kernel region
TORCH_LOGS="fusion" python output/aten_repros/dynamo_BertForMaskedLM_inference/region_003_sum_amax_abc123.py

# Run a single repro's benchmark
python output/aten_repros/vllm_deepseek_v3_inference/region_011_amax_sum_c0c1b9.py
```

---

## Key Design Decisions

### Why two extraction modes?

**Scheduler mode** shows reality — what Inductor actually does. But it requires a
full compilation (slow) and the fused nodes don't map cleanly back to FX subgraphs
when Inductor has rewritten the IR (CSE, constant folding, buffer reuse).

**ATen mode** is cheaper (captures the post-grad graph before scheduling) and
produces cleaner FX subgraphs. The partitioning approximates what Inductor would do
but isn't exact — it misses heuristic decisions like "don't fuse these because one
is too large."

Both are needed: ATen mode for fast iteration on fusion opportunities, scheduler
mode for ground-truth SOL measurement.

### Why deduplicate?

A 32-layer transformer has 32 identical attention blocks. Without dedup, you get 32
copies of the same softmax repro. Hashing by op structure + input shapes collapses
these to 1, making the suite manageable (59 unique multi-kernel regions across 33
models, not thousands).

### Why disable split_reductions?

`split_reductions=True` (default) tells Inductor to split large reductions into
multi-pass (reduce partially → reduce again). This creates extra scheduler nodes
that look like "multi-kernel" but are intentional. We disable it to see the logical
fusion structure, not the split-reduction implementation detail.

### Why preserve strides?

Non-contiguous inputs change everything: L2 cache hit rates, coalescing,
vectorization eligibility. A repro with `torch.randn(shape)` (always contiguous)
may benchmark 2× faster than the real workload where the input is a slice/permute
view. `as_strided` in `make_inputs()` reproduces the exact memory layout.

### Why subprocess isolation for benchmarks?

A single CUDA OOM, segfault, or infinite-loop Triton kernel would kill the entire
benchmark run. Subprocess isolation (one process per repro) ensures one bad repro
doesn't block the other 400+.

---

## Results Summary (B200, 33 models, inference)

- **442 total regions** extracted (ATen-level)
- **423 compiled** successfully
- **59 produce >1 kernel** (14% — fusion gap)
- **375 single-kernel** regions benchmarked against SOL:
  - Median SOL gap (default compile): **5.25×** ← dominated by launch overhead
  - Median SOL gap (CUDA graph): **0.96×** ← kernel code is near-optimal
  - Only **17 kernels** remain ≥2× SOL with CUDA graph ← actual code quality issues

**Conclusion:** The dominant performance bottleneck is kernel launch overhead, not
kernel code quality. Fusion (reducing kernel count) is far more impactful than
better tiling/num_warps for these workloads.

---

## File Layout

```
better_benchmark/
├── DESIGN.md                          # This document
├── extract_reductions.py              # Main extraction tool (both modes)
├── benchmark_all.py                   # Batch benchmark runner
├── capture_reductions.py              # Legacy scheduler-only extraction
├── instructions.md                    # Original project goals
├── output/
│   ├── repros/                        # Scheduler-mode extracted repros
│   │   ├── llama_like/
│   │   └── hf_albert/
│   ├── aten_repros/                   # ATen-mode extracted repros
│   │   ├── dynamo_BertForMaskedLM_inference/
│   │   ├── dynamo_AlbertForMaskedLM_inference/
│   │   ├── vllm_openai_gpt-oss-20b_inference/
│   │   └── ...
│   └── deduped_repros/                # Deduplicated across models
├── repo/
│   ├── README.md                      # Summary of findings
│   ├── docs/
│   │   ├── CLASSIFICATION.md          # Fusion failure root causes
│   │   ├── METHODOLOGY.md            # How extraction/probing works
│   │   └── SOL_ANALYSIS.md           # Single-kernel SOL gap results
│   ├── repros/                        # Curated multi-kernel repros
│   ├── scripts/                       # Pipeline scripts
│   └── analysis/                      # Machine-readable results
└── *.py                               # Supporting scripts (probe, test, tune)
```
