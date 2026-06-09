# sum_sum_2b7bf2b45160

## Status

- Family: `cooperative_split_k`
- Closure status: `bad_oracle`
- Artifact: `repros/canonical/sum_sum_2b7bf2b45160/oracle_cooperative_split_k.py`
- Classification: `BAD_ORACLE`

## Full-Scope Contract

The oracle consumes the same inputs as `repro.py` and returns the same two outputs:
`float32[256, 960, 7, 7]` full tensor and `float32[960]` channel reduction.

## Timings

- Oracle: 59.14 us
- torch.compile (coordinate_descent_tuning=True, combo_kernels=True): 54.08 us
- Ratio: 0.915x (compiler is faster)

## Gap Diagnosis

The compiled code is already faster than the oracle (ratio 0.915x). The oracle
cooperative split-k approach does not provide a speedup over Inductor's current
strategy for this shape. No action needed.

## Validation

- `oracle_cooperative_split_k.py --check`: PASS
- `oracle_cooperative_split_k.py --bench`: ratio 0.915x, status BAD_ORACLE
