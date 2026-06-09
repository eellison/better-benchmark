# sum_sum_b1697ad4a549

## Compile: 67.5us, Oracle: 67.4us, Gap: 1.00x

## Diagnosis: COOPERATIVE_SPLIT_K

## Root cause: Inductor with coordinate descent tuning already matches the oracle for this MobileViT SiLU-gradient plus batch-norm-backward pattern. The cooperative split-K oracle does not improve over the current Inductor schedule for this particular shape ([128,96,32,32] channel reductions).

## Fix path: No Inductor change needed -- current CD compile is at the oracle floor. Mark as at_floor.

## Status: at_floor

## Details

- Model: timm_mobilevit_s (train, 4 shapes)
- Pattern: sum+sum reduction (BN backward channel reductions)
- Ops: slice, sub, mul (many), neg, exp, reciprocal, add, unsqueeze, sum x2
- Shape: [128,96,32,32] channel reductions over N/H/W -> [96] vector outputs
- Oracle classification was COOPERATIVE_SPLIT_K but the measured gap is effectively zero (1.00x), indicating Inductor's combo_looped schedule is already optimal for this workload size.
