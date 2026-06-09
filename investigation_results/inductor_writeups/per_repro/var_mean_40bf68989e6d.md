# var_mean_40bf68989e6d


## Measured Timings
- Oracle: 31.52 us
- Compile (CDT): not available
- Ratio: N/A

## Classification: AT_FLOOR

## Current Result

- Oracle path: `repros/canonical/var_mean_40bf68989e6d/oracle_swin_droppath_patchmerge_layernorm.py`
- Oracle: measured
- Ratio: 1.036x
- Status: `at_floor`

## Diagnosis

Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete Swin seed-index-41 batch DropPath residual add, 2x2 patch-merge window clone into hidden-size-2048 rows, population var_mean LayerNorm, affine epilogue, final [6272, 2048] output, and `rsqrt / 2048` side output with a shape-specialized Triton normalization kernel, whereas Inductor lowers the seed lookup/random, residual pointwise work, window permute/clone, normalization, affine, and side-output division as generic sc

## Config exploration results

- Inductor already matches or beats the oracle (ratio <= 1.05x). No config exploration needed.
- Status: Small gap (1.036x) within noise threshold
