# sum_sum_sum_ed6c74b5ef96

## Compile: 598.0us, Oracle: 821.2us, Gap: 0.73x

## Diagnosis: AT_FLOOR

## Root cause: Inverts the max-pool-backward scatter-add in Triton and feeds four BN/ReLU-backward multi-output reductions without materializing the [128, 288, 35, 35] scatter result.

## Fix path: Add scatter-reduce fusion support that can sink a structured scatter-add producer into sibling channel reductions and their dependent tensor epilogues.

## Status: closed

## Details

- Model: timm_adv_inception_v3_train_001 (4 shapes)
- Pattern: sum x8 reduction (172 ops)
- Oracle: oracle_multi_output_reduction.py
