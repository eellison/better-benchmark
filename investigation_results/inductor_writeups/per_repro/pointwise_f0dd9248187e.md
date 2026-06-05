# pointwise_f0dd9248187e

Full-scope oracle: `repros/canonical/pointwise_f0dd9248187e/oracle_stencil_fusion.py`.

Gap diagnosis (classification: SCHEDULER_FUSION): the oracle fuses a stencil computation into a single kernel on `float16[64, 512, 7, 7]`. However, Inductor's compiled output already matches or exceeds the oracle speed.

Measurements:
- `python repros/canonical/pointwise_f0dd9248187e/oracle_stencil_fusion.py --check`: PASS, output 0 shape `[64, 512, 7, 7]`, dtype `torch.float16`.
- `python repros/canonical/pointwise_f0dd9248187e/oracle_stencil_fusion.py --bench`: oracle_us=10.18, compile_us=9.54, ratio=0.937, status=BAD_ORACLE.

Conclusion: The oracle is slower than torch.compile (ratio 0.937x). No Inductor regression exists. The stencil fusion oracle does not improve over Inductor's existing codegen for this workload.
