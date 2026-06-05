# pointwise_e0eff9609cf0

## Classification: BAD_ORACLE

## Current Result

- Oracle path: `repros/canonical/pointwise_e0eff9609cf0/oracle_broadcast_hoist.py`
- Correctness: PASS
- Oracle: `17.18 us`
- `torch.compile coordinate_descent_tuning=True`: `15.36 us`
- Ratio: 0.894 (oracle slower than compile)
- Status: `bad_oracle`

## Diagnosis

The oracle for the broadcast hoist pattern is slower than Inductor's generated code on this hardware. Inductor already reaches or exceeds the oracle's performance level, so no gap exists.

## Config exploration results
- No configs needed -- oracle is slower than baseline compile.
