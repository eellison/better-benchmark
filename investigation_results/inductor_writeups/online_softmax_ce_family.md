# online_softmax_cross_entropy family — June 2026 investigation

Branch: pytorch-work pr-184905. Hardware: B200. Fresh cache per bench.

## Triage (fresh re-bench 2026-06-10, oracle*.py --bench)

| repro_id | oracle_us | compile_us | abs gap (us) | ratio | verdict | pattern |
|----------|-----------|------------|--------------|-------|---------|---------|
| amax_sum_sum_6fd07d12d98a | 520.3 | 738.3 | 218.0 | 1.42 | REAL | DistillGPT2 infer: shifted-label ignore-index CE mean, f32[16384, 50257] |
| amax_sum_sum_86d05d6810f4 | 521.0 | 739.0 | 218.0 | 1.42 | REAL | DistillGPT2 train: same pattern |
| amax_sum_sum_e2f518f0a274 | 91.0 | 149.2 | 58.2 | 1.64 | REAL | GPT2 shifted causal-LM CE mean, [2048, 50257] |
| amax_sum_d112f48ea917 | 75.4 | 101.0 | 25.6 | 1.34 | REAL | Reformer stable softmax (where(inf) guard) |
| amax_sum_f5253e4f250e | 337.5 | 358.0 | 20.6 | 1.06 | near floor | |
| amax_sum_sum_0e37ca9164b3 | 33.7 | 43.9 | 10.2 | 1.30 | REAL | Blenderbot ignore-index CE mean |

Skipped per task: amax_sum_9940b361e5b4 (Longformer band assembly — separate blocker).

Stale-jsonl agreement: jsonl ratios match fresh measurements within noise; family genuinely open.

## Status: WIP — blocker analysis in progress

## Blocker analysis (2026-06-10, fresh)

The triage hypothesis (pattern match not firing for shifted/ignore-index forms) is WRONG.
Verified via TORCHINDUCTOR_CACHE_DIR dump on amax_sum_sum_6fd07d12d98a:
`cross_entropy_loss_online` fires (output code kernel comments show
`%cross_entropy_loss_online_default`), and the online-softmax scalar-accumulator
kernel is generated. The 218us gap is INSIDE the generated online-softmax loop.

Per-kernel profile (compiled, 738us total):
- kernel0 (online softmax max+sum over f32[16384, 50257]): 718us  <-- all the gap
- kernel1 (target gather + per-row loss + partial mean): 16us
- kernel2 (final mean): 1.5us
Oracle does the whole thing in 520us; its row kernel is the same online logsumexp.

Microbenchmark decomposition of kernel0 body (B200, interleaved best-of-3, config XBLOCK=2/R0=2048/w4):
| variant | us | GB/s |
|---------|----|------|
| inductor body: `other=0.0` load + 2x `tl.where(r0_mask, v, -inf)` + `triton_helpers.max2` | 641.8 (721.6 @ coordesc-chosen 2/4096/w8) | 5170 |
| same but `tl.max` instead of `max2` | 537.3 | 6130 |
| `other=-inf` load, no where, `max2` | 579.4 | 5684 |
| `other=-inf` load, no where, `tl.max` | **519.7** | 6338 |
| oracle | 520.3 | 6457 |

Blockers (file:line, pytorch-work pr-184905):
1. torch/_inductor/codegen/triton.py:5018 (and :5140 CE variant) — scalar online-softmax
   combine uses `triton_helpers.max2(...)` = `tl.reduce(a, dim, maximum)` with a custom
   NaN-propagating combine fn. Compiles to a generic reduce ~104us slower than native
   `tl.max` on B200. NaN propagation through the max output is the only semantic delta;
   sum_exp output still propagates NaN via exp(NaN), so softmax/log-softmax/CE final
   outputs are NaN-identical.
2. torch/_inductor/codegen/triton.py:4158 — masked loads feeding the online-softmax
   reduction use `other=0.0` then re-mask with `tl.where(cond, value, float('-inf'))`
   inside the loop (triton.py:5009). Loading `other=-inf` directly removes the where
   (~17us with tl.max, ~62us with max2).

Fix direction: codegen change behind new config flag (default True), no custom kernels.

## Orchestrator guidance (2026-06-10, from user discussion — for the implementing agent)

Prefer GENERIC loop-body-level transformations over softmax-specific patches:

1. max2 fix, try FIRST: keep NaN semantics and use native speed —
   `tl.max(a, dim, propagate_nan=tl.PropagateNan.ALL)` (Triton 3.7.0 has
   PropagateNan; modern PTX has single-instruction NaN-propagating max).
   If that benches at native-tl.max speed, the fix is a one-line change in
   triton_helpers.max2 (or its call sites) with BITWISE-identical semantics —
   no NaN-equivalence argument needed anywhere. Only if propagate_nan ALL is
   slow fall back to the documented exp(NaN)-propagation argument, scoped to
   the online-softmax combine sites.
2. other=-inf fix: implement as generic mask-fill propagation, not a softmax
   special case — when a masked load's value is consumed ONLY through
   tl.where(same_mask, value, C), fold C into the load's `other` and elide the
   where. This is a loop-body dataflow property (value-flow + mask equality),
   useful beyond softmax (any masked reduction operand).
