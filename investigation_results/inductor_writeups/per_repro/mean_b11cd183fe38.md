# mean_b11cd183fe38

## Compile: 8.99us, Oracle: 8.99us, Gap: 1.0x

## Classification: AT_FLOOR

## Root Cause

The oracle computes embedding RMSNorm with aliased outputs in a single kernel. Inductor already matches the oracle perfectly at the measurement floor.

## Kernel Count
- Oracle: 1 kernel
- Inductor: already at parity

## Config Exploration
| Config | Result |
|--------|--------|
| combo_kernels + CDT | 8.99 us (1.0x) |

## Status: at_floor

No gap exists. Both oracle and compile produce equivalent performance at 8.99us. The measurement shows exact parity.

## Details
- Model: embedding + RMSNorm with aliases
- Shape: [4096, 512] f32
- Pattern: embedding -> RMSNorm -> 3 aliased outputs
- No Inductor change needed
