# var_mean_mean_61303c3fa6f4

## Compile: 32.35us, Oracle: 30.11us, Gap: 1.074x

## Diagnosis: BANDWIDTH_BOUND

## Root cause: The oracle diagnoses itself as BANDWIDTH_BOUND. This is an EfficientNet training-BatchNorm plus SiLU spatial-mean scope. The oracle computes per-channel var_mean over [128,7,7], running mean/variance copy_ updates, affine normalization, SiLU activation, and the final [128,1152,1,1] spatial mean in a fused kernel. Inductor already lands in the same two-pass normalization plus nonlinear-reduction memory and launch envelope. The ~7% gap is within noise/launch overhead territory and multi_kernel configs don't help.

## Kernel count
- Inductor: standard BN-training + spatial-mean reduction schedule
- Oracle: fused BN-train + SiLU + spatial-mean

## Config exploration results
- multi_kernel=1 (default): 32.35us (ratio 1.074x)
- multi_kernel=2: 32.48us (ratio 1.065x) - no improvement
- multi_kernel=3: 32.42us (ratio 1.066x) - no improvement
- coordinate_descent_tuning=True, combo_kernels=True: already enabled

## Classification: BANDWIDTH_BOUND

The remaining gap is dominated by required activation reads for statistics and output reduction, exp latency for SiLU, running-stat stores, and output stores rather than avoidable intermediate traffic. At-floor case.

## Status: closed_at_floor
