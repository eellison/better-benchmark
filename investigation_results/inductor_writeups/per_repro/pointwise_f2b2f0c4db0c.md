# pointwise_f2b2f0c4db0c


## Measured Timings
- Oracle: 12.03 us
- Compile (CDT): 11.17 us
- Ratio: 0.93x

Full-scope oracle: `repros/canonical/pointwise_f2b2f0c4db0c/oracle_bn_residual_relu.py`.

Gap diagnosis (classification: SCHEDULER_FUSION): the oracle fuses batch normalization + residual add + ReLU into a single kernel on `float16[32, 2048, 7, 7]`. However, Inductor's compiled output already matches or exceeds the oracle speed.

Measurements:
- `python repros/canonical/pointwise_f2b2f0c4db0c/oracle_bn_residual_relu.py --check`: PASS, output 0 shape `[32, 2048, 7, 7]`, dtype `torch.float16`.
- `python repros/canonical/pointwise_f2b2f0c4db0c/oracle_bn_residual_relu.py --bench`: oracle_us=12.03, compile_us=11.17, ratio=0.928, status=BAD_ORACLE.

Conclusion: The oracle is slower than torch.compile (ratio 0.928x). No Inductor regression exists. Inductor's existing fusion already handles BN+residual+ReLU efficiently for this shape.
