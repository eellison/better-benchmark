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

`config.triton.evict_first_requires_r_contiguous` (new flag, default True):
in the evict_first branch of `TritonKernel.load`, require stride 1 on a
*reduction* tiling var; loads only x-contiguous get no eviction hint (normal
policy). Re-loaded buffers keep the evict_last escalation. Gated to CUDA
cc >= 10 (validated on B200; same gating precedent as
split_reductions_for_undersaturated_gpu).

Primary result: sum_011e69da166d 1.42x -> **0.98x** (573.3us -> 395.0us,
oracle 403.3us) — beats the hand-written oracle.
