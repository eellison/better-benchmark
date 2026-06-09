# var_mean_ef88cc4744f1

## Compile: 7.49us, Oracle: 7.10us, Gap: 1.054x

## Diagnosis: AT_FLOOR

## Root cause: The gap is barely above the 1.05x threshold (1.054x) and within noise. This is a Bart embedding + layernorm pattern where the oracle embeds token+position, adds, and normalizes in one kernel. Inductor already generates a single fused kernel. With multi_kernel=3, the ratio drops to 1.045x, confirming this is at floor.

## Kernel count
- Inductor: 1 kernel (fused embedding + layernorm)
- Oracle: 1 kernel (same approach)

## Config exploration results
- multi_kernel=1 (default): 7.49us (ratio 1.054x)
- multi_kernel=2: 7.46us (ratio 1.050x) - at floor
- multi_kernel=3: 7.39us (ratio 1.045x) - at floor
- coordinate_descent_tuning=True, combo_kernels=True: already enabled

## Status: closed_at_floor
