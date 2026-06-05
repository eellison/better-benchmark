# pointwise_ef9ca88fea46

Full-scope oracle: `repros/canonical/pointwise_ef9ca88fea46/oracle_relu_dropout.py`.

Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): the oracle implements a fused ReLU + dropout with boolean mask sibling return. The compiled Inductor output already matches or exceeds the oracle speed.

Measurements:
- `python repros/canonical/pointwise_ef9ca88fea46/oracle_relu_dropout.py --check`: PASS (stochastic output 0 skipped, output 1 exact bool pass).
- `python repros/canonical/pointwise_ef9ca88fea46/oracle_relu_dropout.py --bench`: oracle_us=8.1, compile_us=7.46, ratio=0.921, status=BAD_ORACLE.

Conclusion: The oracle is slower than torch.compile (ratio 0.921x). No Inductor regression exists. The compiled code already achieves optimal performance for ReLU+dropout fusion.
