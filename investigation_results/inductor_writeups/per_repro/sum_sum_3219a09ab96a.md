# sum_sum_3219a09ab96a (SqueezeNet maxpool-backward → masked channel sums)

## Re-measured 2026-06-09 (fresh cache, CUDAGraph + GPU lock, B200)
- Oracle: 204.6 us, Compile: 736.3 us, **Ratio: 3.60x** — biggest verified gap in corpus
- Prior writeup said "Status: implemented" at 2.82x — STALE/WRONG: nothing in
  /tmp/pytorch-work eliminates this scatter (no structured pass exists; grep
  confirms), and the gap has widened.
- Same-family verified same day: sum_sum_sum_4d7a27f102ca 2.70x,
  sum_sum_d73ef74614dc 2.50x (UNet bilinear variants).

## Classification: ALGEBRAIC_ELIMINATION (reduction-over-scatter)

## Graph
```
zeros[131072, 729]
idx = _low_memory_max_pool_offsets_to_indices(i8 offsets, 3x3, 27x27, stride 2)
s   = scatter_add(zeros, dim=1, idx.view(131072,169), grad.view(131072,169))
v   = s.view(512,256,27,27)
out0 = sum(where(mask0, 0, v[:, 128:256]), [0,2,3])   # [128]
out1 = sum(where(mask1, 0, v[:, 0:128]),  [0,2,3])    # [128]
```
Scatter result has NO consumers other than the two fully-reduced masked sums.

## Inductor today: 3 kernels, ~1.1GB scratch traffic
1. zeros init (382MB write), 2. atomic scatter_add (88.6MB read + 382MB random
atomics), 3. masked reduction (382MB + 96MB reads → 2x[128]).

## The elimination
Reduction distributes over scatter_add. Iterate SOURCE elements; each grad
element's contribution is gated by the mask gathered at its computed destination:
```
for (row, j) in grad[131072,169]:
    dest = idx[row, j]; (b, c, p) = decompose(row, dest)
    out[c>=128 ? 0 : 1][c % 128] += grad[row,j] * !mask[b, c%128, p]
```
One reduction kernel, ~185MB traffic, no zeros init, no atomics on scratch.
Matches the oracle (oracle_structured_scatter_reduce.py) and explains 3.6x.
Full family analysis incl. bilinear variants and I/O accounting:
investigation_results/structured_scatter_reduce_analysis.md.

## Altitude
FX/graph-level is the correct altitude: the identity is algebraic
(playbook category 3, like cat-through-reduction). The scheduler receives IR
where the dense scratch buffer already exists as a mutation target — no fusion
decision can remove the 1.1GB. The scheduler-level alternative (fuse
atomic-scatter producer into full-reduction consumer with index-remapped
predicates) is the hard version of design TODO #2 and embeds the same identity.
Natural home: extend fx_passes/scatter_reduce_fusion.py. Safety conditions:
scatter base is full(0) (or add sum(base) separately); ALL consumers are full
reductions over scattered dims (slices → dest range tests, where(m,0,x) →
multiply by !m gathered at dest); sum/add semiring only.

## Status: IMPLEMENTED (2026-06-10, commit 5489b8c2bb9 on pr-184905)

### Result (official harness, B200, fresh cache, CUDAGraph + GPU lock)
- Before: oracle 204.6us, compile 736.3us, ratio 3.60x
- After:  oracle 203.8us, compile 91.8us,  ratio 0.45x — compile BEATS the
  hand-written Triton oracle by 2.2x.
- Kernels: 3 before (zeros-init poi / atomic-scatter poi / masked-reduce red,
  201MB traffic w/ 382MB-class scratch) -> 7 after but no scratch buffer and
  no atomics: 1 fused gather-predicate red kernel producing both [B,C]
  partial pairs + per-output {mask-count red, [B,C] combine red, [C] final
  per} (the 7 tiny reduction kernels total ~92us).

### What was implemented (commit 5489b8c2bb9)
- torch/_inductor/fx_passes/scatter_reduce_fusion.py:
  - New entry `scatter_add_gather_reduce_pass` (line ~365): default-ON pass
    running the existing Phase-1a scatter_add chain detection/rewrite under
    new config flag `scatter_add_reduce_elimination`
    (TORCHINDUCTOR_SCATTER_ADD_REDUCE_ELIMINATION, default 1). No-ops when
    the experimental `scatter_reduce_fusion` flag is on (avoids double
    application). Skip-connection/multiplier (UNet BN-backward) variants are
    excluded from the default-on path — measured net-negative (see below).
  - `_rewrite_scatter_add_reduce_chain`: switched final reduction to
    TWO-STAGE: sum spatial dims -> [B,C] partials, then sum batch -> [C].
    The old single-stage sum([0,2]) left only C=128 parallel reduction rows
    on a 148-SM B200 (306us); two stages give B*C=65536 rows (92us).
  - Safety guards added: plain sum only (keepdim=False/no dtype arg),
    scatter_dim==1, idx.shape==src.shape, mask shape == sliced view shape,
    0-dim fill scalar, src rows factor as [B,C_full], CUDA-only, device from
    graph meta (was hard-coded cuda).
- torch/_inductor/config.py: new flag `scatter_add_reduce_elimination`.
- torch/_inductor/fx_passes/post_grad.py: registration after
  scatter_reduce_fusion.

### Corpus impact (coord-descent us, flag OFF -> ON)
sum_sum_3219a09ab96a 736->92; sum_18262b26934c 1357->299;
sum_sum_8bcd6e12dcd4 793->158; sum_7ee057acd9bc 491->117;
sum_3ee4028cab37 676->159; sum_7df61c52c7f8 709->146;
sum_34d3dca078b3 1067->481; sum_14fe7b321763 278->89;
sum_8a66186d1ffe 147->50; sum_bc4a942a8c4f 159->52; sum_f4f71b921e96 37->22.
All 11 pass eager-vs-compiled at 1e-2. No regressions:
sum_sum_sum_04ab10ca59ee 0.83x unchanged, pointwise_27183a793fcd at floor.

### Deliberately excluded (measured net-negative, guard in pass)
UNet BN-backward skip/mult variants (sum_sum_sum_{612b509be4a7,
cad09d5cca79, 1627b1a3a6f6, 8e6cde42e572}): rewrite needs the f32
multiplier gathered at destinations + skip/fill terms re-reading the
output space; at B=8 this lost 1.7-2.2x vs baseline. Still reachable via
TORCHINDUCTOR_SCATTER_REDUCE_FUSION=1.

### Family-membership correction
sum_sum_sum_4d7a27f102ca (2.67x) and sum_sum_d73ef74614dc (2.50x) are NOT
this family: in both, the index_put(accumulate=True) result is a GRAPH
OUTPUT (Swin rel-pos bias table / Qwen vocab gradient), not consumed by
reductions — the reduction-over-scatter identity does not apply. They need
a different optimization (duplicate-index scatter materialization);
unchanged by this work, separate diagnosis required.
