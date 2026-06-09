# Gap Diagnosis: sum_sum_mean_9af96955f8cc (rmsnorm_bwd)

## Classification: REDUCTION_CHAINING

## Measurements

| Config | Time (us) | Ratio vs Default |
|--------|-----------|------------------|
| Default | 88.9 | 1.00x |
| Combo Looped (CD) | 44.8 | 1.99x |
| Combo Persistent (CD) | 45.1 | 1.97x |
| Oracle (handwritten Triton) | ~36 | ~2.5x |

- **Bytes accessed**: 84.1 MB (logical)
- **Best compile SOL**: 2.44 GB/s effective BW ratio
- **Oracle SOL**: ~2.3 TB/s effective BW (near H100 peak)

## Pattern Description

Qwen3-30B-A3B RMSNorm backward: gather + scale + mask + sum-over-8-experts followed
by RMSNorm (add + x^2 + mean + rsqrt + weight * x).

## Root Cause

Default generates 4 kernels. The dominant gap is between K2 (gather+scale+mask+sum,
grid=4M, reduction=8) and K3 (RMSNorm persistent reduction over 2048 cols, grid=2048).
K2 writes an 8MB intermediate that K3 rereads.

Coordinate descent alone closes 50% of the gap (default 88.9us -> combo 44.8us) by
fixing tile sizes in the bottleneck kernel K2. The remaining gap to oracle (44.8 -> 36us)
requires fusing K2+K3 so the intermediate stays in registers.

## Oracle Strategy

Single persistent kernel with grid=[2048]. Each program:
1. Loads 8 rows from grouped_mm via inverse permutation (64MB read)
2. Scales/masks/sums experts in registers (no intermediate write)
3. Adds residual, computes RMSNorm in-place
4. Writes final output (8MB)

Eliminates 16MB intermediate round-trip.

## Inductor Fix Required

**REDUCTION_CHAINING**: When a reduction kernel (sum over experts, output [2048, 2048])
feeds directly into another reduction kernel (RMSNorm mean over 2048 cols, output [2048]),
and the intermediate's inner dim equals the second kernel's reduction dim, fuse into a
single persistent kernel. The grid matches the second kernel's outer dim (2048), and each
program tile-loops over the first reduction while accumulating both partial sums.

## Generalizability

HIGH - This pattern (reduction -> normalization with intermediate in registers) appears
in every MoE model with expert routing followed by RMSNorm. Applies to 2 repros in this
corpus (9af96955f8cc and 1dfbbe76c078) and likely many more MoE models.

## Remaining Gap After CD

~8-10us (44.8us best compile vs ~36us oracle). This is the fusion opportunity.
