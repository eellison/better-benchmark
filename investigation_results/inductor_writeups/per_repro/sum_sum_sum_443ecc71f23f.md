# sum_sum_sum_443ecc71f23f

## Compile: 69.7us, Oracle: 58.2us, Gap: 1.20x

## Diagnosis: SCATTER_REDUCE

## Root cause: Inductor emits separate generic batch reductions, permute/reshape copies, and two index_put scatter kernels for the three-output Swin relative-position-bias backward, rather than fusing the softmax-backward producer with duplicate-index scatter-reduce accumulation into [169,32] buckets while writing the required [4096,49,49] side output.

## Fix path: Add a structured relative-position scatter-reduce lowering that fuses batch reductions with indexed accumulation and emits the complete three-output tuple (two [169,32] scatter tables + [4096,49,49] side output) from the same producer.

## Status: design_todo

## Details

- Model: timm_swin_base_patch4_window7_224 (train, 1 shape)
- Pattern: sum+sum+sum reduction with dual index_put scatter
- Ops: fma, mul, neg, sum x3, squeeze x2, permute x2, reshape x6, index_put x2, full
- Shape: [128,32,49,49] softmax-backward -> two [169,32] scatter targets + [4096,49,49] side output
- 5 kernels generated; oracle fuses into fewer; gap is modest at 1.20x.
