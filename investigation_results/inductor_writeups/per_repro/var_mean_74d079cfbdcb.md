# var_mean_74d079cfbdcb

## Classification: AT_FLOOR

## Current Result

- Oracle path: `repros/canonical/var_mean_74d079cfbdcb/oracle_swin_window_layernorm.py`
- Oracle: measured
- Ratio: 0.974x
- Status: `at_floor`

## Diagnosis

Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete fp32 hidden-size-512 population LayerNorm affine followed by the fixed Swin 7x7 window-partition clone by traversing final window rows and storing contiguously into the clone/view output, whereas Inductor fuses the same ATen scope but schedules source-contiguous rows and emits strided stores into the clone buffer; parent `bench_oracle()` measures the two schedules at the same floor, so the output-contiguous remap 

## Config exploration results

- Inductor already matches or beats the oracle (ratio <= 1.05x). No config exploration needed.
- Status: Inductor matches oracle within noise
