# sum_sum_sum_1482b693d4f2
## Compile: 144us, Oracle: 128us, Gap: 1.13x
## Diagnosis: COOPERATIVE_SPLIT_K
## Root cause: Inductor combo_persistent is within 13% of the oracle that fuses Swin MLP layernorm-backward, drop-path, window-reverse layout, and multiple column reductions; the remaining gap is largely scheduling/launch overhead from 4 kernels vs 1.
## Fix path: Minimal gain (16us, 1.13x); cooperative split-K with window-reverse layout fusion would be complex for small benefit. Effectively at floor.
## Status: at_floor
