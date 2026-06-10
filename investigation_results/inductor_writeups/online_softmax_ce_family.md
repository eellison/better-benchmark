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

## Status: CLOSED 2026-06-10 — pytorch-work commit a26fc2c8bf4

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

## Fix (2026-06-10, pytorch-work commit a26fc2c8bf4)

`[inductor] Fast combine for online-softmax loops` — codegen-only, behind NEW
config flag `triton.online_softmax_fast_combine` (env
`TORCHINDUCTOR_ONLINE_SOFTMAX_FAST_COMBINE`, default True). No existing
defaults changed; no custom Triton kernels.

Three coordinated changes in torch/_inductor/codegen/triton.py:
1. Block max via native `tl.max` instead of `triton_helpers.max2` in the
   scalar online-softmax combine (both `online_softmax_reduce` and
   `online_softmax_cross_entropy` looped paths) and in the persistent
   two-pass fallback's final max reduction.
2. New `TritonKernel._rewrite_masked_load_other_neg_inf`: when the value
   feeding the online-softmax reduction is the direct result of a single
   masked load (`load_var.mask_vars == reduction masks`, no `ops.masked`
   context, fp dtype), rewrite the already-emitted load line from
   `other=0.0` to `other=float("-inf")` and skip the in-loop
   `tl.where(mask, value, -inf)` re-masking.
3. `_twopass_fallback_maybe_fast_max` applies the same rewrite on the
   persistent path (Blenderbot-class kernels) and skips `_mask_value`
   for the max reduction there.

Re: orchestrator guidance — `tl.max(..., propagate_nan=...)` was tried first
but the installed Triton 3.7.0 `tl.max` does NOT accept a propagate_nan
kwarg (signature: input, axis, return_indices, ...), so the bitwise-identical
one-liner is unavailable. Fell back to the documented exp(NaN)-propagation
argument scoped to online-softmax sites: NaN inputs poison sum_exp via
exp(NaN), so softmax / log-softmax / CE outputs computed from (max, sum)
remain NaN-identical; only the raw amax side output can differ under NaN.
Verified empirically: NaN row and all--inf row produce identical eager vs
compiled outputs.

ignore_index exactness verified: ignored rows contribute 0 to the loss sum
and are excluded from the mean denominator — checked with ignored labels
present, absent, and ALL ignored (0/0 = NaN matches eager).

## Per-repro results (B200, oracle*.py --bench, fresh cache per run)

| repro_id | before us / ratio | after us / ratio | status |
|----------|-------------------|------------------|--------|
| amax_sum_sum_6fd07d12d98a | 738.3 / 1.42 | 531.5 / 1.01 | AT_FLOOR |
| amax_sum_sum_86d05d6810f4 | 739.0 / 1.42 | 531.4 / 1.01 | AT_FLOOR |
| amax_sum_sum_e2f518f0a274 | 149.2 / 1.64 | 97.2 / 1.08 | GOOD |
| amax_sum_d112f48ea917 | 101.0 / 1.34 | 77.3 / 1.03 | AT_FLOOR |
| amax_sum_sum_0e37ca9164b3 | 43.9 / 1.30 | 35.5 / 1.06 | GOOD |

Family abs gap closed: ~530us total -> ~50us total.

Regression checks (after fix):
| repro_id | ratio | note |
|----------|-------|------|
| amax_sum_f5253e4f250e (T5 plain online softmax, was at floor) | 1.03 | no regression (was 1.06) |
| amax_sum_17ab35828f89 (swin plain softmax) | 0.91 | no regression (was 0.92) |
| sum_sum_b16afac198fb | 0.95 | unchanged |
| amax_sum_9940b361e5b4 (Longformer, separate blocker) | 2.54 | unchanged (jsonl 2.60) |

test/inductor/test_online_softmax.py: 18 failed / 14 passed BOTH with and
without the change (pre-existing branch failures, verified via git stash).
test_torchinductor.py -k "cross_entropy or reduction or max": 147 passed,
1 pre-existing CPU failure (test_max_min_bool_cpu, fails on clean branch too).
softmax/logsumexp/amax subset: 19 passed.

Remaining headroom: e2f518f0a274 (1.08) and 0e37ca9164b3 (1.06) — coordinate
descent picks XBLOCK=2/R0_BLOCK=8192/w16 for the big CE kernel where
2/4096/w8 is ~10% faster (605 vs 698us kernel time); a tuning-search issue,
not a codegen issue. Small absolute gaps (<8us each).
