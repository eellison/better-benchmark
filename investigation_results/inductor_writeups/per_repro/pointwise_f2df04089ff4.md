# pointwise_f2df04089ff4


## Measured Timings
- Oracle: 57.12 us
- Compile (CDT): 56.32 us
- Ratio: 0.99x

Full-scope oracle: `repros/canonical/pointwise_f2df04089ff4/oracle_resnet_bn_relu_maxpool.py`.

Gap diagnosis (classification: SCHEDULER_FUSION): the oracle fuses BN + ReLU + maxpool into a single kernel on `float16[32, 64, 56, 56]` producing both pooled values and int8 offsets. Inductor is essentially at parity.

Measurements:
- `python repros/canonical/pointwise_f2df04089ff4/oracle_resnet_bn_relu_maxpool.py --check`: PASS, output 0 shape `[32, 64, 56, 56]`, dtype `torch.float16`; output 1 exact int8 match.
- `python repros/canonical/pointwise_f2df04089ff4/oracle_resnet_bn_relu_maxpool.py --bench`: oracle_us=57.12, compile_us=58.27, ratio=1.02, status=AT_FLOOR.

Conclusion: Inductor is at floor (ratio 1.02x). The 2% gap is within noise. No actionable performance improvement possible.
