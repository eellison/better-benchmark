# pointwise_f1d4acfe63b0


## Measured Timings
- Oracle: 7.74 us
- Compile (CDT): 7.74 us
- Ratio: 1.00x

Full-scope oracle: `repros/canonical/pointwise_f1d4acfe63b0/oracle_index_put.py`.

Gap diagnosis (classification: BANDWIDTH_BOUND): the oracle implements an index_put operation producing `float32[128, 2560]`. Inductor already matches the oracle performance.

Measurements:
- `python repros/canonical/pointwise_f1d4acfe63b0/oracle_index_put.py --check`: PASS, output 0 shape `[128, 2560]`, dtype `torch.float32`, max diff `9.54e-07`.
- `python repros/canonical/pointwise_f1d4acfe63b0/oracle_index_put.py --bench`: oracle_us=7.74, compile_us=7.74, ratio=1.0, status=AT_FLOOR.

Conclusion: Inductor is at floor (ratio 1.0x). No performance gap exists.
