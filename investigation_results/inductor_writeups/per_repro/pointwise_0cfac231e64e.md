# pointwise_0cfac231e64e

## Classification: BAD_ORACLE

## Current Result

- Family: `index_put_scatter_reduce`
- Oracle path: `repros/canonical/pointwise_0cfac231e64e/oracle_index_put_scatter_reduce.py`
- Correctness: PASS
- Oracle: `19.3 us`
- `torch.compile coordinate_descent_tuning=True`: `17.28 us`
- Ratio: 0.896 (oracle slower than compile)
- Status: `bad_oracle`

## Diagnosis

The oracle's fused index_put/scatter_reduce kernel is 10% slower than Inductor's generated code. Inductor's generic scatter lowering outperforms the hand-tuned oracle on this hardware.

## Config exploration results
- No configs needed -- oracle is slower than baseline compile.
