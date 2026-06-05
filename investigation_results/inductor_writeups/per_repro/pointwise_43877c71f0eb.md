# pointwise_43877c71f0eb — AT_FLOOR (ratio 0.957x)

## Summary
Oracle: `oracle_bf16_div_fp8.py` — bf16 scalar-divide + float8_e4m3fn cast
Benchmark: oracle=8.93us, compile=8.54us, ratio=0.957x

## Classification: BANDWIDTH_BOUND (at floor)

## Root Cause
The oracle computes bf16 scalar-divide plus `float8_e4m3fn` cast in one flat Triton
pointwise kernel. Inductor already lowers the same two-op contiguous pointwise region
to equivalent work. The remaining cost is dominated by the bf16 read (6144*768*2 = 9.4MB)
and fp8 write (6144*768*1 = 4.7MB), totaling ~14MB of memory traffic.

Inductor actually outperforms the oracle here (0.957x), indicating its codegen and
autotuning for this simple pointwise pattern is better than the oracle's fixed config.

## Kernel Count
- Inductor: 1 kernel
- Oracle: 1 kernel

## Config Exploration
N/A — Inductor already faster than oracle.

## Conclusion
No action needed. Inductor matches/exceeds oracle at the bandwidth floor.
