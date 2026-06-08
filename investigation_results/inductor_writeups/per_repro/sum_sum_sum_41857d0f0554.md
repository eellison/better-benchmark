# sum_sum_sum_41857d0f0554

## Classification: BAD_ORACLE

## Oracle: oracle_vit_ln_backward_multi_output.py

## Measurements

- Compiled: 660.26 us
- Oracle: 848.93 us
- Ratio: 0.778x (oracle 22% slower than compile)
- Oracle correctness: PASS

## Diagnosis

The oracle (vit_ln_backward_multi_output) is significantly slower than torch.compile output. For the ViT layernorm backward with output shapes [768] vectors and [768, 175360] f32 matrix, the oracle's fused multi-output kernel cannot beat Inductor's autotuned persistent reduction. The extremely wide reduction dimension (175360) favors Inductor's split-reduction strategy with optimized tile sizes over the oracle's monolithic fusion approach.

## Status: closed_no_gap (BAD_ORACLE)
