# pointwise_388aef7dede1 — AT_FLOOR (ratio 0.985x)

## Summary
Oracle: `oracle_full_layout_stencil.py` — fused add/div + Longformer sliding-window as_strided clone
Benchmark: oracle=10.91us, compile=10.75us, ratio=0.985x

## Classification: SCHEDULER_FUSION (already at floor)

## Root Cause
The oracle fuses add/div, Longformer sliding-window as_strided clone, and final layout
view into one Triton materialization kernel. The diagnosis claims Inductor treats the
pointwise producer and overlapping layout clone as separate scheduling regions.

However, measured performance shows Inductor actually outperforms the oracle (0.985x),
indicating Inductor's codegen is already optimal for this workload at this shape.

## Kernel Count
- Inductor: matches or exceeds oracle
- Oracle: 1 kernel

## Config Exploration
No config changes needed — Inductor already faster than oracle.

## Conclusion
No action needed. Oracle is slower than compiled code for this shape.
