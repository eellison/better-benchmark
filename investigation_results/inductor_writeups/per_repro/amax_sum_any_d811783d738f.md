# amax_sum_any_d811783d738f

## Status: NO GAP (BAD_ORACLE)

## Measurement
- Oracle: 100.29 us
- Compiled: 50.82 us
- Ratio: 0.507x (oracle is ~2x SLOWER than compiled)

## Pattern
Full BERT attention softmax + dropout. Pattern hash `d811783d738f`, from `torchbench_hf_Bert_train_000`.
Shape: `[48, 512, 512]` (batch=4, heads=12, seq=512).

## What the Oracle Does
Custom Triton kernel that fuses:
- Always-true iota/ge broadcast mask folding
- Row-wise amax, sub, exp, sum (online softmax)
- All-masked-row guard (any dim check)
- Inductor RNG dropout with scale
- Output transpose/view

## Root Cause of BAD_ORACLE
Inductor's compiled output already handles this pattern efficiently. The oracle's custom
Triton kernel for full attention softmax with dropout is actually slower than Inductor's
generated code for this shape. At [48, 512, 512], Inductor's persistent reduction +
pointwise fusion already performs well -- the oracle's single-kernel approach adds overhead
from loading the random seed tensor and doing per-row dropout inline that is not offset by
saved memory traffic at this shape.

## Classification
NEW_PATTERN (per oracle docstring), but no performance gap exists. Inductor already
matches or exceeds oracle performance. No fix needed.

## Kernel Count
Not investigated (no gap to close).
