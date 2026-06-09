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

## Status: DIAGNOSED_PENDING_IMPLEMENTATION (awaiting altitude decision +
free fx_passes agent slot; do not dispatch concurrently with the MT5 scatter
agent — same file)
