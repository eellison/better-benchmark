# sum_sum_sum_e51450a39c69

## Classification: MULTI_OUTPUT_SHARED_REDUCTION

## Current Result

- Family: `mobilebert_multi_output`
- Oracle path: `repros/canonical/sum_sum_sum_e51450a39c69/oracle_mobilebert_multi_output.py`
- Correctness: PASS (oracle correct)
- Oracle: 17.82 us
- Compiled (coordinate_descent_tuning=True): 19.87 us
- Ratio: 1.12x
- Status: FIXED (codegen bug resolved, residual performance gap)

## Diagnosis (Historical)

Previously, Inductor's compiled code had a codegen bug -- it generated a reference to `buf4` that was not defined in the output wrapper. Same class of bug as `sum_sum_sum_bedefd130db8`. This bug has been fixed.

The oracle computes MobileBERT multi-output reduction pattern over shape [128, 32768], producing 4 outputs: three [128] reduction vectors and one [128, 32768] elementwise output.

## Config exploration results
- coordinate_descent_tuning=True: 19.87 us (1.12x)

## Residual Gap (1.12x)

The remaining 1.12x gap is a kernel quality/tiling issue, not a codegen bug. The oracle achieves better throughput with its fused multi-output kernel.
