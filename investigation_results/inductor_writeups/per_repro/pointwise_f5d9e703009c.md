# pointwise_f5d9e703009c


## Measured Timings
- Oracle: 6.43 us
- Compile (CDT): 5.98 us
- Ratio: 0.93x

Full-scope oracle: `repros/canonical/pointwise_f5d9e703009c/oracle_attention_q_layout.py`.

Gap diagnosis (classification: BANDWIDTH_BOUND): the oracle implements an attention Q-projection layout transformation producing `float32[12, 512, 64]` with stride `(64, 768, 1)`. Inductor's compiled output already exceeds oracle speed.

Measurements:
- `python repros/canonical/pointwise_f5d9e703009c/oracle_attention_q_layout.py --check`: PASS, output 0 shape `[12, 512, 64]`, dtype `torch.float32`, max diff `0.00e+00`, stride `(64, 768, 1)`.
- `python repros/canonical/pointwise_f5d9e703009c/oracle_attention_q_layout.py --bench`: oracle_us=6.432, compile_us=5.888, ratio=0.915, status=BAD_ORACLE.

Conclusion: The oracle is slower than torch.compile (ratio 0.915x). No Inductor regression exists. Inductor with coordinate_descent_tuning already achieves optimal performance for this layout transformation.
