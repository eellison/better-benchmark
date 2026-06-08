# sum_sum_bc46e8b5be67

## Classification: COOPERATIVE_SPLIT_K

## Current Result

- Oracle path: `repros/canonical/sum_sum_bc46e8b5be67/oracle_mt5_residual_fused_backward.py`
- Correctness: PASS
- Oracle: `16.03 us`
- `torch.compile coordinate_descent_tuning=True`: `19.2 us` (19.7us in bench_compare)
- `torch.compile combo_kernels=True,multi_kernel=2`: `16.3 us`
- `torch.compile combo_kernels=True,multi_kernel=3`: `16.1 us`
- Best compile config: `multi_kernel=3` at `16.1 us`
- Ratio (best compile vs oracle): 1.004x (gap CLOSED)
- Status: `closed_by_config`

## Diagnosis

The oracle computes the MT5 residual dropout/layer-norm-backward return tuple by row-tiling the [4096,512] producer, sharing row-local reductions for the transposed [512,4096] output, adding the residual, and emitting partial column sums cooperatively. With `multi_kernel=3` (force looped reduction), Inductor achieves 16.1us vs oracle's 16.03us -- effectively at parity.

## Kernel count
- Oracle: 2 kernels (row-tiled producer + column-sum finalizer)
- Inductor (multi_kernel=3): comparable kernel structure via looped reduction

## Config exploration results
- `coordinate_descent_tuning=True`: 19.7 us (baseline)
- `multi_kernel=2`: 16.3 us (1.21x faster, nearly closes gap)
- `multi_kernel=3`: 16.1 us (1.22x faster, CLOSES gap to 1.004x vs oracle)
- multi_kernel=3 is the winning config for this pattern

## Fix path: Gap fully closed by `triton.multi_kernel=3`. No scheduler change needed.

## Status: closed_by_config (but see regression note below)

## Re-measurement (2026-06-08)

- Oracle: 15.9 us
- Compiled (default autotuning): 18.21 us
- Ratio: 1.145x

NOTE: The default compile path now shows a 1.145x gap. This may be because the aggressive split
threshold (8586e404cc8) changed which reduction strategy is auto-selected. The gap was previously
closed by explicitly setting multi_kernel=3. The fix here is that the autotuner should still
select the looped reduction mode (multi_kernel=3) for this pattern. If multi_kernel=3 is still
the winning config, this is an autotuning regression, not a codegen regression.
