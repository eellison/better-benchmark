# pointwise_f3fcc01a1b31

Full-scope oracle: `repros/canonical/pointwise_f3fcc01a1b31/oracle_gelu_dropout.py`.

Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): the oracle implements fused GELU + dropout. However, Inductor's compiled output already matches or exceeds the oracle speed.

Measurements:
- `python repros/canonical/pointwise_f3fcc01a1b31/oracle_gelu_dropout.py --check`: PASS (stochastic output 0 skipped).
- `python repros/canonical/pointwise_f3fcc01a1b31/oracle_gelu_dropout.py --bench`: oracle_us=30.37, compile_us=28.42, ratio=0.936, status=BAD_ORACLE.

Conclusion: The oracle is slower than torch.compile (ratio 0.936x). No Inductor regression exists. Inductor already handles GELU+dropout fusion efficiently.
