# var_mean_7f13aadfc66c - Swin DropPath Residual LayerNorm (Training)

## Classification: AT_FLOOR

## Benchmark Results
- Oracle: 19.3 us
- Compile (baseline): 20.0 us
- Ratio: 1.036x (within noise, no meaningful gap)
- Status: AT_FLOOR

## Oracle
- Path: `repros/canonical/var_mean_7f13aadfc66c/oracle_swin_droppath_layernorm.py`
- Correctness: PASS (stochastic outputs auto-skipped)
- Model: timm_swin_base_patch4_window7_224 (training)
- Shape: [128, 7, 7, 1024] -> [6272, 1024]

## Diagnosis

Inductor already matches the oracle within noise (ratio < 1.05x). The oracle fuses
degenerate window reshape/permute, input-seeded drop-path, residual add, population
var_mean, affine epilogue, and final view into one row kernel. Inductor achieves
equivalent performance on this [6272, 1024] shape because the kernel is
compute-bound on the hidden-1024 reduction and the scheduling overhead
of the producer chain is amortized by the row count.

The SCHEDULER_FUSION classification in the oracle docstring describes the theoretical
opportunity, but on this specific shape, Inductor's default schedule is already at floor.

## Config Exploration
- No further config exploration needed (baseline already at oracle level).
