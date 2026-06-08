# sum_ebebe1ce6cce

## Compile: 20.29us, Oracle: 24.22us, Gap: 0.838x

## Classification: BAD_ORACLE

## Root Cause

The oracle (bandwidth_row_fma) is slower than torch.compile. The compiled code at 20.29us outperforms the oracle at 24.22us, giving a ratio of 0.838x. The oracle implements a single Triton row-reduction kernel covering the [512,128,128] view semantics, fp32 product, last-dimension sum, FMA, sliced broadcast mask, scalar fill, and final contiguous output. Inductor already emits a similar fused persistent reduction that achieves better throughput via its autotuned tile configuration.

## Kernel Count
- Oracle: N/A (slower than compile)
- Inductor: baseline is already faster

## Config Exploration
No config exploration needed -- oracle is slower than compile.

## Fix Assessment: No action needed

The oracle is a BAD_ORACLE. Inductor already wins on this repro.

## Details
- Model: fused row reduction with FMA + broadcast mask
- Shape: [512, 128, 128] f32 input, [512, 128, 128] f32 output
- The oracle's single-kernel approach cannot beat Inductor's autotuned persistent reduction at this scale.
