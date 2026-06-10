# SqueezeNet scatter_add gather-reduce elimination: end-to-end validation

**Question:** is the sum_sum_3219a09ab96a gap (3.60x -> 0.45x closed by
pytorch-work 5489b8c2bb9) real in the FULL squeezenet1_1 backward graph, or an
artifact of the capture partitioner cutting the graph at boundaries Inductor
wouldn't see?

**Verdict: PARTITIONER ARTIFACT.** The pass never fires on the full model
graph. The repro-level pattern ("scatter result consumed ONLY by masked
channel sums") only exists because the partitioner cut away the
`convolution_backward` consumer of every `where_self` node. In the full
graph each `where_*` feeds BOTH a `sum.dim_IntList` (bias grad) AND a
`convolution_backward` (weight/input grad), so the scatter output must
materialize regardless, and the pass's own profitability guard correctly
declines all 26 candidate chains. End-to-end delta: **18us out of 16.8ms
(0.1%, within noise)** vs a naive composed prediction of ~4700us.

## Setup

- B200, `CUDA_VISIBLE_DEVICES=1`, `INDUCTOR_GPU_BENCH_LOCK=1` (exclusive lock
  serialized against the running full-graph sweep), CUDAGraph capture,
  min-of-5x do_bench(min), `coordinate_descent_tuning=True`, fresh
  `TORCHINDUCTOR_CACHE_DIR` per condition, one condition per process.
- Fix gated by `TORCHINDUCTOR_SCATTER_ADD_REDUCE_ELIMINATION`
  (`config.scatter_add_reduce_elimination`, default 1; commit 5489b8c2bb9).
- Driver: `scripts/e2e_squeezenet_scatter_validation.py`.
- Full graph: `repros/models/torchbench/train/squeezenet1_1/full_graph_001.py`
  (whole backward, 67 inputs, ~4.6GB). The sweep skips this graph
  (`unsafe_integer_input`); to run it we override the int8 maxpool-offset
  inputs (getitem_1/3/5) to the valid `[0, 9)` range for a 3x3 window —
  the sidecar default (high=512 clamped to int8) trips
  `tl.device_assert(0 <= tmp4 < 9)` in the scatter kernel.

## Timings

| condition | compiled us (min-of-5) | kernels (def triton) | pass applied | max rel err vs eager |
|---|---|---|---|---|
| repro fix ON  | **99.3**  | 7 (4 in log: CD dedups) | 2 chains | 1.8e-7 |
| repro fix OFF | **737.1** | 3 | 0 | 1.4e-6 |
| full graph fix ON  | **16769** | 37 (58 incl. wrappers) | **0 chains** | 6.6e-6 |
| full graph fix OFF | **16787** | 37 | 0 | 2.1e-6 |

- Repro-level saving confirmed: 737.1 -> 99.3us = **638us saved** (0.45x vs
  204.6us oracle, reproduces the writeup).
- Kernel listings for full graph fix ON vs OFF are **byte-identical kernel
  sets** (diff of sorted `def triton` lines is empty). The pass is a no-op
  on the full model.
- Full-graph delta 18us = 0.1%, within run-to-run jitter of the rounds.

## Why the pass doesn't fire: the partitioner hid the second consumer

The full graph has 3 scatter_add sites (fire-module maxpool backwards), each
fanning out to 2 channel-sliced `where` consumers (sites 1-2) or 1 (site 3) —
the 6 textual "scatter_add" occurrences are 3 ops + 3 view lines. With
`TORCH_LOGS` + pass debug logging, fix-ON compile reports for every candidate:

```
scatter_reduce_fusion: skipping scatter_add chain - where_self has 1
non-rewritable user(s) (['convolution_backward_N']), scatter cannot be eliminated
```

(26 such lines — every ReLU-backward `where` in the net matches the chain
prefix; all are rejected.)

In the full graph (e.g. full_graph_001.py lines 129-144), each masked grad is
used twice:

```python
where_13 = where(le_13, 0, slice_10)            # masked grad
sum_14   = sum(where_13, [0,2,3])               #  -> bias gradient
convolution_backward_13 = conv_bwd(where_13, …) #  -> weight/input gradient
```

The capture partitioner cut the repro at the `where`/`sum` boundary, dropping
`convolution_backward` (not a fusible reduction pattern), which produced a
graph where the scatter output's ONLY consumers are the two sums — exactly the
"scatter can be fully eliminated" precondition that never holds in the model.

The pass's profitability guard (scatter_reduce_fusion.py:819-830: "If
non-rewritable users exist, the scatter intermediate must still materialize…
adding gather operations would be strictly worse") is doing precisely the
right thing — this validates the guard, not just the limitation.

## Is there residual headroom at the scatter sites in the full model?

Per-iteration kernel profile of the full graph (torch.profiler, fix ON):

- total ~16.9ms/iter, dominated by conv backward GEMMs + cudnn NCHW<->NHWC
  transposes (~10ms+).
- scatter-site triton kernels: zeros-init + atomic scatter ~1417us,
  where-materialize pointwise ~775us, where+sum reductions ~339us.

Even though the sum can't be detached from the scatter, there IS a smaller,
different opportunity: the bias-grad `sum(where)` kernels re-read the scratch
buffer that the conv_backward also reads. But that's bounded by the ~339us
of sum-reduction kernels (2% of step time), and the where-materialization is
needed by cuDNN anyway. The 1.4ms zeros+atomic-scatter cost is real maxpool-
backward work whose output feeds an opaque `convolution_backward` extern
kernel — eliminating it would require fusing into cuDNN, out of scope.

## Predicted vs actual

- Naive composition (3 sites, scaled by scratch+src traffic: site2 ~2.1x,
  site3 ~4.2x of site1's 638us): **~4700us predicted** saving.
- Actual: **18us (noise)**. Ratio ~0.4% of prediction.

## Generalizable lessons

1. **A repro composes to the model only if the pattern's *consumer-set*
   precondition survives recontextualization.** This pass requires "ONLY
   reduction consumers." The partitioner manufactured that property by
   slicing off the conv_backward user. Any pass keyed on "sole consumer" /
   "all users rewritable" is maximally exposed to this artifact class.
2. **Validation recipe is cheap:** grep the full graph for the pattern's
   anchor op, list ALL users of the chain nodes (here: `where_13` ->
   {sum_14, convolution_backward_13}), and check the pass's preconditions
   against that user set *before* projecting repro savings. Five minutes of
   graph reading would have predicted the no-op without GPU time.
3. **The corpus should record consumer fan-out at partition boundaries.**
   A repro whose pattern-boundary nodes had additional (dropped) users in the
   source model should carry a manifest flag (e.g. `boundary_users_dropped`),
   so "repro gap closed" claims are auto-qualified. The capture hook has this
   information at partition time.
4. **The repro is not worthless**: the rewrite is correct and 7.4x on the
   isolated pattern, and patterns where a masked scatter feeds only
   reductions do exist (the pass's other 10 rewritten repros). But its
   model-attribution to squeezenet1_1 is wrong, and corpus-level "closed
   3.60x->0.45x" accounting should not be projected onto model step time.
5. **Where the real model-level win would be:** the SqueezeNet backward is
   dominated by conv-backward GEMMs and layout transposes; the scatter sites
   are ~2.5ms/16.9ms and irreducible while conv_backward consumes the dense
   buffer. A future direction is bias-grad fusion into the conv-backward
   epilogue (sum(where) shares its input read with conv_bwd), worth <=339us.

## Artifacts

- Driver: `scripts/e2e_squeezenet_scatter_validation.py`
- Raw results: /tmp/sq_{repro,fg}_fix_{on,off}.json, stderr logs with
  output_code at /tmp/sq_*_fix_*.stderr.log (kernel listings).
- Pass: /tmp/pytorch-work torch/_inductor/fx_passes/scatter_reduce_fusion.py
  (guard at lines ~819-830), flag in torch/_inductor/config.py:1274.
