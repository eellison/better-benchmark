# pointwise_391b0009ff83

## Classification: BAD_ORACLE

## Current Result

- Family: `swiglu_pointwise`
- Oracle path: `repros/canonical/pointwise_391b0009ff83/oracle_swiglu_pointwise.py`
- Correctness: PASS
- Oracle: `12.74 us`
- `torch.compile coordinate_descent_tuning=True`: `11.55 us`
- Ratio: 0.907 (oracle slower than compile)
- Status: `bad_oracle`

## Diagnosis

The oracle for the SwiGLU pointwise pattern on shape [512, 11008] fp16 is 9% slower than Inductor's generated code on this hardware. The SwiGLU (silu * gate) pointwise is a simple element-wise fusion where Inductor's autotuned codegen already achieves optimal throughput.

## Config exploration results
- No configs needed -- oracle is slower than baseline compile.
