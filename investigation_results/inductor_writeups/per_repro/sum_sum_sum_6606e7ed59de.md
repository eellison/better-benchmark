# sum_sum_sum_6606e7ed59de

## Compile: 29.5us, Oracle: 33.2us, Gap: 0.89x

## Diagnosis: COOPERATIVE_SPLIT_K

## Root cause: Inductor compile already outperforms the oracle for this Swin layer-norm-backward/drop-path pattern. The cooperative split-K oracle has overhead from its coordination mechanism that exceeds the benefit at this small problem size ([128,7,7,1024] row reductions).

## Fix path: No Inductor change needed -- compile is already faster than the oracle. Mark as at_floor.

## Status: at_floor

## Details

- Model: timm_swin_base_patch4_window7_224 (train, 3 shapes)
- Pattern: sum x5 reductions (layer-norm backward + drop-path)
- Ops: div, expand, mul x7, sub x2, sum x5, permute, unsqueeze x2, view x3, convert_element_type
- Shape: [128,7,7,1024] row reductions -> [1024] vectors + [6272,1024] side output
- 4 kernels generated; Inductor's default schedule (29.5us) beats the cooperative split-K oracle (33.2us) because the coordination overhead dominates at small tile sizes.
