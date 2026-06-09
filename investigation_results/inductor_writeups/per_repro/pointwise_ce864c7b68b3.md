# pointwise_ce864c7b68b3

## Compile: 8.96us, Oracle: 9.12us, Gap: 0.982x

## Classification: AT_FLOOR (BAD_ORACLE at measurement noise)

## Root Cause

Compile is already faster than the oracle by 1.8%. Both are at the launch floor (~9us). No investigation needed.

## Status: at_floor

## Details
- Model: timm ViT layout materialize
- Shape: [6304, 384] f32
- Pattern: layout materialization
- Oracle and compile are both at the performance floor; compile is marginally faster
