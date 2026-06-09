# var_mean_13eb1aa0a32d

## Classification: PERSISTENT_THRESHOLD

## Current Result

- Oracle path: `repros/canonical/var_mean_13eb1aa0a32d/oracle_bn_leaky_relu_training.py`
- Correctness: PASS
- Oracle: `19.33 us`
- `torch.compile coordinate_descent_tuning=True`: `24.38 us` (24.2us in bench_compare)
- `torch.compile combo_kernels=True,multi_kernel=2`: `24.4 us`
- `torch.compile combo_kernels=True,multi_kernel=3`: `18.3 us`
- Best compile config: `multi_kernel=3` at `18.3 us`
- Ratio (best compile vs oracle): 0.947x (gap CLOSED, Inductor FASTER)
- Status: `closed_by_config`

## Diagnosis

The oracle computes the DCGAN training-BatchNorm leaky-ReLU [1024,512,4,4] in one per-channel kernel: var_mean over [1024,4,4], rsqrt, running-stat updates, and leaky_relu(0.2). With `multi_kernel=3` (force looped reduction), Inductor achieves 18.3us -- actually FASTER than the oracle's 19.33us.

The default config (coordinate_descent_tuning only) uses a non-optimal reduction strategy for elements_per_channel=16384 with 512 channels. multi_kernel=3 forces the looped persistent variant which is optimal for this shape.

## Kernel count
- Oracle: 1 kernel (per-channel BN + leaky_relu fused)
- Inductor (multi_kernel=3): comparable kernel structure

## Config exploration results
- `coordinate_descent_tuning=True`: 24.2 us (baseline, 1.26x slower than oracle)
- `multi_kernel=2`: 24.4 us (no improvement)
- `multi_kernel=3`: 18.3 us (1.32x faster than baseline, BEATS oracle)
- multi_kernel=3 is the winning config -- gap fully closed and exceeded

## Fix path: Gap fully closed by `triton.multi_kernel=3`. The default reduction strategy should choose the looped variant for this shape (ELEMS_PER_CHANNEL=16384, CHANNELS=512). This is a PERSISTENT_THRESHOLD issue where the default heuristic picks the wrong strategy.

## Status: closed_by_config
- File references: /tmp/pytorch-work/torch/_inductor/choices.py (reduction strategy selection)
