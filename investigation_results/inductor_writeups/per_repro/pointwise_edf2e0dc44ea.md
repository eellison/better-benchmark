# pointwise_edf2e0dc44ea


## Measured Timings
- Oracle: 7.42 us
- Compile (CDT): 6.43 us
- Ratio: 0.87x

Full-scope oracle: `repros/canonical/pointwise_edf2e0dc44ea/oracle_embedding.py`.

Gap diagnosis (classification: BANDWIDTH_BOUND): the oracle implements an embedding lookup producing `float32[128, 2560]`. Inductor's compiled output already matches or exceeds the oracle speed.

Measurements:
- `python repros/canonical/pointwise_edf2e0dc44ea/oracle_embedding.py --check`: PASS, output 0 shape `[128, 2560]`, dtype `torch.float32`, max diff `0.00e+00`.
- `python repros/canonical/pointwise_edf2e0dc44ea/oracle_embedding.py --bench`: oracle_us=7.424, compile_us=6.688, ratio=0.901, status=BAD_ORACLE.

Conclusion: The oracle is slower than torch.compile (ratio 0.901x). No Inductor regression exists for this repro. The compiled code with coordinate_descent_tuning already achieves optimal performance for this embedding operation.
