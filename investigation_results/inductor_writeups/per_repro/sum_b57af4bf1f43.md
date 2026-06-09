# sum_b57af4bf1f43 - oracle_fused_product_sum

## Status: BAD_ORACLE (compile beats oracle)

- Oracle: 31.49 us
- Compile: 24.32 us
- Ratio: 0.772x

## Classification: NO_GAP

Oracle path: `repros/canonical/sum_b57af4bf1f43/oracle_fused_product_sum.py`

The compiled Inductor output outperforms the oracle by 23%. No Inductor improvement needed.

## Details
- Models: hf Blenderbot, M2M100, MBart family training (11 instances)
- Pattern: Masked f32 product with returned transposed view and sibling reduction (128-row then 32-partial ordering)
- Shapes: [2560, 4096] f32 (product layout) + [2560] f32 (sum)
- Classification in oracle: BANDWIDTH_BOUND -- Inductor already emits this two-stage fused product/reduction form efficiently
- Correctness: PASS
