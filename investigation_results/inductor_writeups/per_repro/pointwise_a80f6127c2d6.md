# pointwise_a80f6127c2d6

## Compile: 33.66us, Oracle: 33.76us, Gap: 0.997x (BAD_ORACLE)

## Diagnosis: AT_FLOOR

## Root cause: The oracle (layout bias add) matches torch.compile output within noise. Compile is marginally faster (0.3%).

## Status: closed_at_floor

## Details

- Model: layout bias add
- Pattern: pointwise bias addition with specific output layout [1, 512, 50265] fp16, stride (25735680, 50265, 1)
- Shape: [1, 512, 50265] float16 (large vocabulary dimension)
- Inductor generates an efficient fused kernel matching oracle performance
- Output layout constraint verified correct (max_diff=0.0)
- No config exploration needed -- already at floor
