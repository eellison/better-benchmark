# var_mean_bc6eb324bf67

## Classification: BAD_ORACLE

## Current Result

- Oracle path: `repros/canonical/var_mean_bc6eb324bf67/oracle_gelu_layernorm.py`
- Correctness: PASS
- Oracle: `20.61 us`
- `torch.compile coordinate_descent_tuning=True`: `17.22 us`
- Ratio: 0.835 (oracle slower than compile)
- Status: `bad_oracle`

## Diagnosis

The oracle for the gelu layernorm pattern is slower than Inductor's generated code on this hardware. Inductor already reaches or exceeds the oracle's performance level, so no gap exists.

## Config exploration results
- No configs needed -- oracle is slower than baseline compile.
