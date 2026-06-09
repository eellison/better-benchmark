# sum_sum_sum_512dcf4a167b
## Oracle: oracle_multi_output_reduction (BEiT layer-norm backward, defers to sum_sum_sum_64f701d26f0a)
## Compile: 177.7us, Oracle: 163.4us, Gap: 1.09x
## Diagnosis: BANDWIDTH_BOUND
## What the oracle does differently: Implements a two-phase dependent multi-output reduction (row dual-reduce + column reduce with side stores), sharing reads between row sums and column accumulators; identical schedule to sum_sum_sum_64f701d26f0a with only capture-time argument renaming.
## Inductor fix: At floor -- the 1.09x gap is within measurement noise; current Inductor fixes (scalar_acc, linear_reduction_elimination, coordinate_descent) have nearly closed this gap for the BEiT LN-backward shape (M=25216, D=768).
