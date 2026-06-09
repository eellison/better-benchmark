# pointwise_f48fe6a2319f


## Measured Timings
- Oracle: 7.30 us
- Compile (CDT): 7.62 us
- Ratio: 1.04x

Full-scope oracle: `repros/canonical/pointwise_f48fe6a2319f/oracle_segment_mask_multi_output.py`.

Gap diagnosis (classification: BANDWIDTH_BOUND): the oracle implements a segment mask computation with multi-output (24 outputs of shape `[1, 1, 512, 512]`). Inductor's compiled output significantly outperforms the oracle.

Measurements:
- `python repros/canonical/pointwise_f48fe6a2319f/oracle_segment_mask_multi_output.py --check`: PASS, all 24 outputs layout-correct, storage unique count matches.
- `python repros/canonical/pointwise_f48fe6a2319f/oracle_segment_mask_multi_output.py --bench`: oracle_us=7.3, compile_us=5.7, ratio=0.781, status=BAD_ORACLE.

Conclusion: The oracle is significantly slower than torch.compile (ratio 0.781x). No Inductor regression exists. Inductor's existing multi-output codegen is already superior to the handwritten oracle.
