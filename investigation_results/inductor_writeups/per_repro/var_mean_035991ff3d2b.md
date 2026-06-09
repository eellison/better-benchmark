# var_mean_035991ff3d2b

## Classification: NEW_PATTERN

## Oracle: oracle_weight_standardization.py

## Measurements

- Compiled: 9.79 us
- Oracle: 9.92 us
- Ratio: 0.987x (oracle ~1% slower than compile; effectively at floor)
- Oracle correctness: PASS

## Diagnosis

The oracle (weight_standardization) computes the full NFNet training weight-standardization scope in one fixed-shape two-warp Triton kernel, matching Inductor's Welford population `var_mean` over each `[1,3072,1536]` logical row, `libdevice.rsqrt(var + 1e-5)`, the generated multiply order `((x - mean) * invstd) * (gain * 0.02551551815399144)`, and the returned invstd, contiguous standardized weight, and mean view.

Inductor already fuses the scope into one generated reduction kernel and achieves essentially identical performance. The oracle's specialized low-warp Welford lowering provides no measurable improvement over Inductor's generic reduction scheduling for this shape. Both are at the hardware floor (~9.8-9.9us).

## Config Exploration

Not explored (no gap to close -- both oracle and compile at hardware floor).

## Status: closed_no_gap (AT_FLOOR)
