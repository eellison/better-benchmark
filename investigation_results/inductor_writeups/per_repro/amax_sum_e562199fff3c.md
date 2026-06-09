# amax_sum_e562199fff3c

## Status: NO GAP (BAD_ORACLE)

## Measurement
- Oracle: 76.74 us
- Compiled: 64.86 us
- Ratio: 0.845x (oracle is ~1.18x SLOWER than compiled)

## Pattern
Identity mask softmax. Pattern hash `e562199fff3c`, from `hf_DebertaV2ForMaskedLM_infer_000`.
Shape: `[192, 512, 512]` (batch*heads=192, seq=512).
Ops: view, amax, sub, exp, sum, div, where (with full defaults).

## What the Oracle Does
Custom Triton kernel that fuses view, where (identity mask based on two `full.default` scalars), stable softmax (amax/sub/exp/sum/div), and final view into a single row kernel.

## Root Cause of BAD_ORACLE
Inductor's compiled output already handles this pattern efficiently. At [192, 512, 512] with a trivial identity mask (where selecting between two `full.default` constants), Inductor's persistent reduction with online softmax is faster than the oracle. The oracle's custom kernel does not provide additional savings over what Inductor already achieves.

## Classification: NO_GAP

No Inductor fix needed. Compile already outperforms oracle. No investigation required.
