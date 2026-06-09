# var_mean_0854c7adb682

## Classification: SCHEDULER_FUSION

## Current Result

- Oracle path: `repros/canonical/var_mean_0854c7adb682/oracle_ghostnet_training_bn_cat.py`
- Correctness: PASS
- Oracle: `55.07 us`
- `torch.compile coordinate_descent_tuning=True`: `58.05 us` (60.3us in bench_compare)
- `torch.compile combo_kernels=True,multi_kernel=2`: `58.3 us`
- `torch.compile combo_kernels=True,multi_kernel=3`: `57.2 us`
- Best compile config: `multi_kernel=3` at `57.2 us`
- Ratio (best compile vs oracle): 1.039x (borderline, effectively closed)
- Status: `at_floor`

## Diagnosis

The oracle computes the GhostNet training-BN + cat pattern [512,960,7,7] with fused channel statistics and concatenation. With `multi_kernel=3`, Inductor reaches 57.2us vs oracle 55.07us -- a 3.9% gap that is within noise/measurement tolerance.

The gap is marginal. The oracle fuses:
1. Two separate BN branches (480 channels each)
2. Channel concatenation to [512,960,7,7]
3. Running stat updates

Inductor with multi_kernel=3 nearly matches by using looped reductions for the per-channel statistics.

## Kernel count
- Oracle: fused multi-branch BN + cat kernel
- Inductor: separate BN kernels + cat (with multi_kernel=3 reducing overhead)

## Config exploration results
- `coordinate_descent_tuning=True`: 60.3 us (baseline)
- `multi_kernel=2`: 58.3 us (1.03x faster)
- `multi_kernel=3`: 57.2 us (1.05x faster, best, nearly closes gap)

## Fix path: Gap is borderline (3.9%). With multi_kernel=3 this is effectively at floor. No dedicated fix needed.

## Status: at_floor
