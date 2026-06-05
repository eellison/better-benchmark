# pointwise_0de8974bd5cb

## Classification: BAD_ORACLE

## Current Result

- Family: `silu_pointwise`
- Oracle path: `repros/canonical/pointwise_0de8974bd5cb/oracle_silu_pointwise.py`
- Correctness: PASS
- Oracle: `11.07 us`
- `torch.compile coordinate_descent_tuning=True`: `10.37 us`
- Ratio: 0.936 (oracle slower than compile)
- Status: `bad_oracle`

## Diagnosis

The oracle for the SiLU pointwise pattern is 6% slower than Inductor's generated code on this hardware. The [8192, 480] SiLU pointwise is a simple element-wise operation where Inductor's autotuned codegen already achieves optimal throughput.

## Config exploration results
- No configs needed -- oracle is slower than baseline compile.
