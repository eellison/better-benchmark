# pointwise_75d8fa236ed7

## Classification: BAD_ORACLE

## Current Result

- Family: `nfnet_gelu_scale`
- Oracle path: `repros/canonical/pointwise_75d8fa236ed7/oracle_nfnet_gelu_scale.py`
- Correctness: PASS
- Oracle: `10.94 us`
- `torch.compile coordinate_descent_tuning=True`: `9.95 us`
- Ratio: 0.909 (oracle slower than compile)
- Status: `bad_oracle`

## Diagnosis

The oracle for the NFNet GELU scale pattern on shape [128, 768, 6, 6] is 9% slower than Inductor's generated code on this hardware. The GELU + scale pointwise fusion is a simple element-wise operation where Inductor's autotuned codegen already achieves optimal throughput.

## Config exploration results
- No configs needed -- oracle is slower than baseline compile.
