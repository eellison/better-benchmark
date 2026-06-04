# pointwise_4e4da02d582f

Classification: SCHEDULER_FUSION. The oracle returns the first split column as the exact `[256, 1]` aliasing view of the input and computes `exp(6 * (tanh(second) + 1) - 10)` plus `1 - tanh(second)^2` for the second column in one Triton kernel, so it covers the full tuple without materializing the view output. Inductor is close on this tiny case, but it does not currently model tuple returns that mix aliasing views with separately materialized fused pointwise outputs as a single scheduler objective with explicit alias preservation; the required Inductor change is SCHEDULER_FUSION support for this mixed view/materialized tuple pattern.

Measured 2026-06-04 on the local GPU: oracle `--bench --warmup 10 --rep 50` reported 3.42 us versus harness compile 22.69 us. `scripts/bench_compare.py` reported coordinate-descent compile 3.94 us and combo compile 4.00 us, so this remains a true floor candidate with parent status `implemented_unmeasured`.
