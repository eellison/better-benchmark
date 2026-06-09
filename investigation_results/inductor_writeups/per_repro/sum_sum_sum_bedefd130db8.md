# sum_sum_sum_bedefd130db8

## Classification: MULTI_OUTPUT_SHARED_REDUCTION

## Current Result

- Family: `gptneo_ln_backward_splitk`
- Oracle path: `repros/canonical/sum_sum_sum_bedefd130db8/oracle_gptneo_ln_backward_splitk.py`
- Correctness: PASS (oracle correct)
- Oracle: 44.8 us
- Compiled (coordinate_descent_tuning=True): 42.91 us
- Ratio: 0.96x (Inductor beats oracle)
- Status: AT_FLOOR (codegen bug fixed, Inductor now at or better than oracle)

## Diagnosis (Historical)

Previously, Inductor's compiled code had a codegen bug -- it generated a reference to `buf6` that was not defined in the output wrapper. This buffer naming/elimination bug has been fixed.

The oracle computes GPT-Neo LayerNorm backward with split-K reduction over shape [2048, 4096], producing 4 outputs: two [2048] gradient vectors, one [2048, 4096] grad_input, and one [2048] scale gradient.

## Config exploration results
- coordinate_descent_tuning=True: 42.91 us (0.96x, beats oracle)

## Resolution

The codegen bug has been fixed. Inductor now compiles and runs this repro successfully, and actually outperforms the oracle (0.96x ratio). No further optimization needed.
