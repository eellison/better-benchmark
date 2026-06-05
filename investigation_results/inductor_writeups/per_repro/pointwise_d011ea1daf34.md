# pointwise_d011ea1daf34

## Compile: 10.24us, Oracle: 9.86us, Gap: 1.039x

## Classification: AT_FLOOR

## Root Cause

The oracle (layout shift cat) is within 3.9% of the compiled output. At this ~10us kernel launch floor timescale, this is within measurement noise / autotuning variance.

## Status: at_floor

## Details
- Model: layout shift cat
- Shape: [768, 64, 128] f16
- Pattern: layout shift + cat fusion
- The 3.9% gap is within measurement noise at the launch floor
- No Inductor change needed
