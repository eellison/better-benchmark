# Family: multi_output_reduction_templates (rank #1 by model attribution)

245 repros / 82 models / 21.8% avg fusible speedup. Investigation started 2026-06-10.

## Verified targets (B200, fresh 2026-06-10)

| repro | oracle us | compile us | ratio | kernels generated |
|-------|-----------|------------|-------|-------------------|
| sum_011e69da166d (PRIMARY) | 403.3 | 573.3 | 1.42x | **1** (fused pointwise+permute-store+colsum) |
| sum_0becf9609ad7 | 66.8 | 93.8 | 1.41x | **1** (looped INNER reduction) |
| sum_21c4018e2b7b | 44.6 | 56.0 | 1.26x | 2 (split reduction + finalize) |
| sum_031c8d6c51f7 (AT FLOOR) | ~51 | ~53 | 1.04x | 1 (persistent reduction) |

## Finding #1: the family name is misleading for these targets

The "multiple reductions re-reading the same input as separate kernels" failure mode
does NOT reproduce on the verified targets: Inductor already fuses the sibling
outputs into a single kernel for the primary (permuted side-store + column sum in
one `triton_red_fused_exp_le_mul_sum_where_0`), and 0bec/031c are single kernels.
The measured gaps come from **kernel-quality decisions inside the single fused
kernel**, not from missing fusion.

## Finding #2 (PRIMARY): coordinate-descent tuning lands in a local minimum

Hand-sweep of the EXACT inductor-generated kernel for sum_011e69da166d
(x=197951 cols contiguous, r=1024 rows, OUTER-style reduction, hint=DEFAULT):

- CD pick: `XBLOCK=32, R0_BLOCK=64, num_warps=16, stages=1` -> 576 us
- Best in same config space: `XBLOCK=128, R0_BLOCK=32, num_warps=8` -> **481 us**
- Oracle (col-tile loop over rows, BLOCK_N>=64 contiguous axis last, stages=3): 403 us

So 95us of the 170us gap is reachable *within the existing kernel* — CD never
finds it (must move XBLOCK 32->128 AND warps 16->8 simultaneously; radius-1
single-coordinate moves regress individually). The remaining ~78us appears
structural (tile orientation / pipelining).

## Finding #3 (ROOT CAUSE, measured): `evict_first` on x-contiguous / r-strided loads

Inductor's load eviction heuristic (`torch/_inductor/codegen/triton.py:4136-4155`
pre-fix) emits `eviction_policy='evict_first'` for ANY coalesced load inside a
looped or persistent reduction (escalating to `evict_last` only if the buffer is
re-loaded). "Coalesced" is `any(stride == 1)` over ALL tiling vars — including the
pointwise x dim. For OUTER-shaped reductions (column sums of a row-major matrix:
load index `x0 + N*r0`, stride 1 on x, stride N on r), `evict_first` is measurably
wrong on B200.

Interleaved A/B of the exact compiled modules, only sed-ing the eviction policy out:

| repro | load orientation | orig (with EP) | EP stripped | delta |
|-------|------------------|---------------|-------------|-------|
| sum_011e69da166d | **x-contiguous** (x0 + 197951*r0) | 584.7-686.8us | 416.7-617.3us | **-25%** |
| sum_0becf9609ad7 | r-contiguous (r0_1 + 169*x0 + ...) | 95.4us | 103.6us | +8% (EP GOOD) |
| sum_031c8d6c51f7 (at-floor) | r-contiguous | 38.3us | 38.9us | +2% (EP good) |
| sum_21c4018e2b7b | r-contiguous (evict_last on loads) | 57.3us | 54.6us | -5% |

Standalone sweep of the primary's exact kernel (same config, only EP varies):
- cfg (128,16,w8): evict_first 495.5us / evict_last 404.9us / none 404.8us
- cfg (32,64,w16) [the CD pick]: evict_first 571.9 / evict_last 538.1 / none 457.5

Mechanism: row stride 197951*4 B is not 128B-aligned, so every XBLOCK-column
segment straddles cache lines shared with the adjacent CTA's x-tile (alignment
shifts every r-step); `evict_first` throws those boundary lines out of L2 before
the neighbor consumes them, forcing DRAM re-reads. With r-contiguous streaming
loads the line IS dead after the CTA's own next r-step, so evict_first behaves
as intended (0bec/031c keep their win under the fix).

**Sibling diff answer**: at-floor sum_031c8d6c51f7 has reduction-contiguous loads
(stride 1 on r), where evict_first is correct. The failure doesn't generalize from
the sibling because the heuristic's coalesced test conflates x-contiguity with
r-contiguity — only the latter justifies evict_first.

## Blocker classification (per design TODO #1 taxonomy)

Class: none of (a)/(b)/(c) as framed — fusion is NOT the blocker. Closest is
**(c)** but as a mis-scoped per-load codegen heuristic, not missing codegen:
`torch/_inductor/codegen/triton.py:4140-4153` (pre-fix), `TritonKernel.load`:
the `elif self.inside_reduction and (persistent or is_loop)` branch assigns
evict_first whenever `is_coalesced` (line 4133: `any(i == 1 for i in
get_strides_of_load(...))` — satisfied by x-stride-1 alone).

The dead agent's hypothesis ("force OUTER reductions to INNER splitting to
enable sibling fusion") is REJECTED by measurement: the primary's sibling
outputs already fuse into one kernel (schedule log "fusing op0 with op1"; one
`triton_red_fused_*` kernel with 2 loads / 2 stores). No fusion is missing;
the fused kernel was just slow. MixOrderReduction acceptance was never reached
because there is only one reduction node.

Secondary blocker (NOT fixed here, logged): coordinate-descent tuning lands in
a local minimum on the primary (picks 32/64/w16 @ 576us; best same-space config
128/32/w8 @ 481us with EP, ~405us without). Escaping needs a simultaneous
2-coordinate move (XBLOCK 4x up AND warps 2x down); radius-1 CD cannot. With
the eviction fix the kernel lands at 395us compile (beats oracle), so this
secondary issue is moot for this repro but may matter elsewhere.

## Fix (committed to pytorch-work pr-184905)

`config.triton.evict_first_requires_aligned_strides` (new flag, default True),
`TritonKernel.load` + `TritonKernel.store`. Final shape after two refinement
rounds driven by counterexamples:

1. v1 (drop evict_first whenever not r-contiguous) — regressed
   sum_31bf563cdf96 0.946x and sum_2bcc7099936f 0.957x: their x-contiguous
   loads have 128B-ALIGNED strides (2304*4, 4096*4), where segments never
   straddle CTA boundaries and evict_first stays profitable.
2. v2 (require misaligned constant stride too) — fixed those, but
   sum_sum_666d023699ba (col sum [4096,30000], misaligned 30000*4 stride,
   NO side store) regressed 0.93x: isolated module A/B confirmed evict_first
   is GOOD there (292.9 vs 313.7us). Difference vs the primary: pure
   streaming reduction (no full-size store competing for L2), boundary lines
   get a chance to survive; and sum_sum_f4d29f9ee6ad showed +1.04x mostly
   from its side-store kernel.
3. v3: drop evict_first only when ALL of:
   - load is x-contiguous but NOT r-contiguous,
   - some non-unit constant stride is not 128B-aligned,
   - the kernel ALSO has an rindex-ed (full-size) store inside the reduction
     loop (`self.has_store_with_rindex`, set in `store()`, resolved via the
     existing DelayReplaceLine deferral) — i.e. exactly the fused
     multi-output "reduction + pointwise side output" shape this family is
     named for.
   Gated to CUDA cc >= 10 (validated on B200). Symbolic strides keep
   evict_first conservatively. Re-loaded buffers keep evict_last escalation.
4. v4 (final): also require the kernel's iteration space (numel*rnumel*
   itemsize) to exceed the device L2 size. sum_sum_06ebd69de9aa
   ([128,1000] = 500KB, 6us scale) matched the v3 shape but its whole
   working set is L2-resident — the eviction policy itself measured neutral
   (isolated A/B 7.2 vs 7.3us) yet flipping the policy perturbed CD into a
   worse num_warps pick (0.93x). Below-L2 kernels keep evict_first so their
   tuning is untouched.

Primary result: sum_011e69da166d 1.42x -> **0.98x** (573.3us -> 394-395us,
oracle 403.3us) — beats the hand-written oracle.

## Final measurements (B200, 2026-06-10)

Commits in /tmp/pytorch-work (branch pr-184905):
- f6ae31e6984 [WIP] v1 (r-contiguity requirement)
- e45f846f8d1 v2 (alignment requirement)
- 60431eca431 v3 (side-store gate)
- e9a67c98a8c v4 (L2 working-set gate) — FINAL

Oracle-relative (fresh cache, oracle harness):

| repro | before | after | oracle us | compile us |
|-------|--------|-------|-----------|------------|
| sum_011e69da166d | 1.42x | **0.98x AT_FLOOR** | 402.5 | 393.3 |
| sum_0becf9609ad7 | 1.41x | 1.41x (untouched, r-contiguous) | 66.4 | 93.9 |
| sum_21c4018e2b7b | 1.26x | 1.26x (untouched, r-contiguous) | 44.0 | 55.2 |
| sum_031c8d6c51f7 | 1.04x | 1.00x AT_FLOOR | 36.5 | 36.6 |

Flag-on vs flag-off interleaved A/B (19 repros, 3 rounds, same GPU/lock):

| repro | before us | after us | speedup |
|-------|-----------|----------|---------|
| sum_011e69da166d | 569.3 | 394.2 | **1.444x** |
| sum_21c4018e2b7b | 56.3 | 55.3 | 1.017x |
| sum_31bf563cdf96 | 111.6 | 110.3 | 1.012x |
| sum_0becf9609ad7 | 93.1 | 93.2 | 0.999x |
| sum_031c8d6c51f7 | 36.6 | 36.6 | 1.001x |
| sum_0c674ef4b13c | 319.3 | 319.4 | 1.000x |
| sum_27f8a4b9ab09 | 33.6 | 33.7 | 0.999x |
| sum_2bcc7099936f | 69.4 | 69.5 | 0.999x |
| sum_2e0e9617102b | 55.0 | 55.1 | 0.999x |
| sum_sum_f4d29f9ee6ad | 999.4 | 1001.4 | 0.998x |
| sum_sum_666d023699ba | 290.7 | 290.7 | 1.000x |
| sum_sum_06ebd69de9aa | 6.0 | 6.3 | 0.949x (codegen bit-identical; autotune variance at 6us) |
| sum_sum_sum_9fe928f49216 | 197.6 | 197.5 | 1.000x |
| sum_sum_sum_508eb468b8d9 | 155.6 | 155.6 | 1.000x |
| sum_sum_63e248035ceb (sentinel) | 23.5 | 23.5 | 1.001x |
| sum_sum_b16afac198fb (sentinel) | 118.7 | 118.7 | 0.999x |
| sum_sum_3219a09ab96a (sentinel) | 92.1 | 98.0 | 0.939x (codegen bit-identical; autotune variance) |
| var_mean_598830735cf6 (sentinel) | 114.4 | 114.4 | 1.000x |
| amax_sum_0561785713ab (softmax sentinel) | 98.0 | 98.0 | 1.000x |

Unit tests: all 6 `pytest test/inductor/test_torchinductor.py -k "evict or
skip_l1"` pass.

## Projected coverage of the 245-repro family

Static scan of the 306 family rows in inductor_optimization_per_repro_queue.csv
for the v4 trigger shape (sum keeping the trailing dim, reducing leading dims,
inner dim not a 32-element multiple, working set > L2 132MB): **3 direct hits**
(sum_011e69da166d 773MB, sum_sum_f4d29f9ee6ad 1.25GB, sum_sum_666d023699ba
468MB) — the latter two are gated off by the side-store requirement (their
side-store kernels are separate). So the fix is narrow-but-deep: it fully
closes the single largest-absolute-gap member (170us of 61.5ms family total)
and is precision-targeted to never regress the rest (16/19 within 0.999-1.017x).

The bulk of the remaining family (r-contiguous BN-backward-style reductions:
sum_0becf9609ad7 1.41x, sum_21c4018e2b7b 1.26x) has a DIFFERENT blocker: the
oracle uses a 2D-grid split-K (partial + finalize) while Inductor emits a
single-wave looped reduction (xnumel=1000-1184 rows on 148 SMs at ~7-8 CTAs/SM
but bandwidth-limited by the single pass-through). That is design TODO #1(b)
(INTRODUCE a split when nothing downstream fuses and rows undersaturate) — the
split heuristic at choices.py:570 only lowers the no-split threshold for
numel_hint < num_sm (1000 >> 148), so these never split. Closing them needs the
bidirectional split decision, not eviction tweaks. Left for the TODO #1
scheduler-level chooser.

## Where this fix fits the design TODOs

This is evidence FOR the "structural decisions made blind to consumers" theme:
the eviction policy was chosen per-load from local index shape only, blind to
(a) whether segments are cache-line aligned, (b) whether the kernel streams a
side store, (c) whether the working set even exceeds L2. The fix wires exactly
those three signals in. Same root principle as TODO #9/TODO #1: local
heuristics need fusion/working-set context.
