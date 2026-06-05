# pointwise_b55a3e33e658

## Compile: 8.32us, Oracle: 8.22us, Gap: 1.012x

## Classification: AT_FLOOR

## Root Cause

The oracle (scaled head layout) is within 1.2% of the compiled output, which is within measurement noise at this kernel launch floor timescale (~8us).

## Status: at_floor

## Details
- Model: scaled head layout
- Shape: [8, 6, 1500, 64] f16
- Pattern: scale + layout change
- No gap exists; the 1.2% difference is measurement noise at the launch floor
