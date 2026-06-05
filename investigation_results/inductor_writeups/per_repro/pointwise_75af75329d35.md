# pointwise_75af75329d35

## Compile: 11.01us, Oracle: 10.98us, Gap: 1.003x

## Diagnosis: AT_FLOOR

## Root cause: The oracle (rowwise affine with aliases) matches torch.compile output within noise. The 0.3% gap is not meaningful.

## Status: closed_at_floor

## Details

- Model: rowwise affine with aliased outputs
- Pattern: pointwise affine transformation [32768, 128] fp32, producing 2 aliased outputs
- Shape: [32768, 128] fp32 with identical aliased outputs
- Inductor generates an efficient fused kernel matching oracle performance
- Both output layout and aliasing constraints verified correct (exact match, max_diff=0.0)
- No config exploration needed -- already at floor
