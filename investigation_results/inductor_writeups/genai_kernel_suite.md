# genai kernel suite — triage + fixes (2026-06-11, B200)

Suite: `repros/models/genai/{CrossEntropy,Softmax,LayerNorm,RMSNorm}{Forward,Backward}/full_graph_*.py`
(standalone kernel benchmarks; no partitioner-artifact risk).

Bench method: `scripts/bench_parallel.py --full-graphs repros/models/genai --device-kind B200`
(fresh `TORCHINDUCTOR_CACHE_DIR`, GPU lock, pytorch-work @ a85d79a900a / pr-184905).
CE-backward graph 001 needs `--allow-unsafe-full-graphs` (i64 target input, randint is fine
for this graph since targets only feed one-hot eq/gather).

Raw results: `results/genai_triage_2026-06-11.json`, `results/genai_triage_ceb001_2026-06-11.json`.

## Triage table (fresh re-measure; the old 2628/2248/1619/1597us attribution numbers predate
## several landed fixes — re-measured below)

`default_us` = inductor default autotune; `cd_us` = with coordinate-descent tuning.
SOL references: `optimal_kernels/*.py` hand kernels where they exist, else algorithmic
memcopy-SOL from bytes at 6.9 TB/s.

| graph | shape | default us | CD us | best-known / SOL us | default/SOL | CD/SOL | verdict |
|---|---|---:|---:|---:|---:|---:|---|
| SoftmaxForward/000 | 8192x262144 bf16 | 3540.9 | 1913.9 | 1915 (hand, 2-pass) | 1.85 | 1.00 | default configs bad; CD = hand kernel |
| SoftmaxBackward/000 (recompute fwd) | 8192x262144 | 3526.4 | 1401.7 | ~1245 (1 read + 1 write)¹ | 2.83 | 1.13 | default configs bad |
| SoftmaxBackward/001 (dgrad) | 8192x262144 | 2879.5 | 2458.6 | 1904 (hand, 2-pass) | 1.51 | 1.29 | gap even after CD |
| CrossEntropyForward/000 | 8192x262144 | 1969.0 | 746.4 | 689 (hand, 1-pass online) | 2.86 | 1.08 | default configs bad; CD near-hand |
| CrossEntropyBackward/000 (recompute) | 8192x262144 | 1977.4 | 747.5 | ~689¹ | 2.87 | 1.08 | default configs bad |
| CrossEntropyBackward/001 (dgrad) | 8192x262144 | 1725.3 | 1435.6 | ~1245 (1 read + 1 write)² | 1.39 | 1.15 | moderate gap after CD |
| LayerNormForward/000 | 1152000x512 | 343.8 | 343.9 | 342 (SOL; hand=inductor) | 1.01 | 1.01 | at SOL |
| LayerNormBackward/000 (recompute) | 1152000x512 | 304.1 | 268.0 | ~263³ | 1.16 | 1.02 | near SOL after CD |
| LayerNormBackward/001 (dgrad) | 1152000x512 | 675.9 | 380.6 | ~343 | 1.97 | 1.11 | default configs bad |
| RMSNormForward/000 | 1152000x512 | 344.0 | 345.8 | 342 (SOL; hand=inductor) | 1.01 | 1.01 | at SOL |
| RMSNormBackward/000 (recompute) | 1152000x512 | 304.0 | 227.0 | ~225³ | 1.35 | 1.01 | near SOL after CD |
| RMSNormBackward/001 (dgrad) | 1152000x512 | 380.8 | 372.5 | ~343 | 1.11 | 1.09 | small gap |

¹ graph 000 of the backward pairs is the forward-with-saved-stats recompute: it reduces the
  large tensor to scalars/row-stats, so minimal traffic is 1 full read (+1 full write for
  SoftmaxBackward/000 which materializes y before `.sum()` — inductor can fuse so 1 read
  + small writes; CD's 1402us ≈ 2 reads, hand floor unclear).
² sum_3 in CE dgrad is algebraically -grad[row] (sum of one-hot * grad), so 1 read of x +
  1 write of dx = 8.59 GB = 1245us floor; CD 1436us.
³ row-stat outputs only: 1 full read = 1.18 GB = 171us, plus dweight partials; measured CD
  values taken as near-floor pending kernel inspection.

## Headline findings

1. The single biggest, *systematic* problem is the default reduction config search space:
   every large-rnumel (262144) graph is 1.4x-2.9x off SOL with default autotune but lands
   at/near hand-kernel time under coordinate-descent tuning. The hand-kernel docstrings
   (optimal_kernels/*.py) independently identified the same root cause: `_reduction_configs`
   caps R0_BLOCK at 1024-2048; the winning configs use R0_BLOCK=4096-8192 with XBLOCK=1.
2. SoftmaxBackward/001 (dy*y - y*sum(dy*y)) is the only graph still >1.25x off best-known
   *after* CD — needs kernel dump + blocker analysis.
3. LN/RMS forward are at SOL (B200); the segment-aligned-split + evict_first fixes already
   landed appear to have done their job for the small-rnumel family.

## Fixes (in progress)

(to be filled in per-fix: blocker file:line, hand-A/B, commit, before/after)

## Orchestrator guidance for SoftmaxBackward/001 (dgrad, 1.29x after CD)

Graph read (full_graph_001.py): recompute y from saved stats (primals_1 bf16 +
amax/sum_1 f32), sum_3 = rowsum(dy*y), dx = fma(-y, sum_3, dy*y). Natural
2-kernel structure (K1: read x for sum_3; K2: re-read x, write dx) = 12.9GB
= 1869us floor at 6.9TB/s. Hand kernel 1904us = ~98% BW; CD 2459us = ~76% BW.
=> The gap is PER-KERNEL STREAMING EFFICIENCY, not structure. Check, in order:
(1) which of the two kernels is slow (time them separately from output_code);
(2) in-loop masking on the loads (other=0.0 + where — the a26fc2c8bf4 class)
and eviction policy on the K2 re-read (x is read in BOTH kernels: K1 should
NOT evict_first x if K2 follows soon? measure both policies);
(3) R0_BLOCK/warps config for rnumel=262144 looped rows (CD radius-1 again?
hand-sweep a few 2-coordinate moves).

CAVEAT: tangents_1 is a SCALAR (0-d expanded) => sum_3 == dy analytically
(rowsum(softmax)=1) and dx ~= 0 numerically. Do NOT exploit this identity
(benchmark-specific, numerically inexact). DO validate any change on a
tensor-tangent variant as well (build one in a standalone script) since this
graph's outputs are insensitive to bugs.

## Orchestrator finding: root cause of the default-vs-CD spread (user hypothesis confirmed)

triton_heuristics.py:4073 gates the scalar-accumulator knowledge on CD:
`has_scalar_acc = inductor_meta.get("coordinate_descent_tuning", False) and (...)`
and :4083 windows it to `8192 <= rnumel <= 131072`. Consequences for genai shapes
(rnumel=262144): the DEFAULT candidate set caps MAX_R0_BLOCK at 1024 (Blackwell)
— the good config CD finds is excluded from default autotune BY CONSTRUCTION,
and even CD's window excludes rnumel=262144 (it gets there only by walking).
Both the CD-gate and the 131072 ceiling are calibrated against the PRE-fast-
combine cost model (the old max2 finalization punished large R0); after
a26fc2c8bf4 the combine is cheap and large-R0 configs won.

FIX DIRECTION (in your territory): derive has_scalar_acc from the kernel
property only (drop the CD-gate term); re-validate the rnumel ceiling >=262144
against current codegen (the old "split heuristics collision" rationale predates
segment-aligned splits a85d79a900a — re-measure, don't assume). Validation =
this suite's default-column: SoftmaxForward 3541us -> expect ~1914; CEF 1969 ->
~750. Sentinels: CD numbers must not regress anywhere; sum_0becf9609ad7 ~0.67x;
amax_sum_sum_6fd07d12d98a ~1.02x. Gate behind a NEW flag per session rules.
