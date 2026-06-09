# sum_sum_sum_1482b693d4f2
## Compile: 124.64us, Oracle: 124.58us, Gap: 1.00x (at floor)

**Previous**: Compile: 144us, Oracle: 128us, Gap: 1.13x

## Diagnosis: CLOSED by inline_recomputable_producers (f58d2545cd2)
## Root cause (historical): Inductor combo_persistent was within 13% of the oracle that fuses Swin MLP layernorm-backward, drop-path, window-reverse layout, and multiple column reductions; the remaining gap was largely scheduling/launch overhead from 4 kernels vs 1.
## Fix: The inline_recomputable_producers extension closed the remaining 13% gap. Compile 124.64us now matches the oracle 124.58us (1.00x ratio). Re-measured 2026-06-08.
## Status: closed
