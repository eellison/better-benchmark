# any_17b234fd7f4f


## Measured Timings
- Oracle: 4.38 us
- Compile (CDT): 5.70 us
- Ratio: 1.30x

## Queue Position

- Rank: 633
- Family: `multi_output_reduction_templates`
- Parent status: `active_subagent`
- Oracle status: `implemented`

## Current Gap

- Repro: `repros/canonical/any_17b234fd7f4f/repro.py`
- Oracle: `repros/canonical/any_17b234fd7f4f/oracle_any_reduce.py`
- Pattern: `gt(arg0, 0)` over contiguous f32 `[8, 1024]`, `view([8192])`, `any`, returning a bool scalar.

## Measurements

- Oracle command: `python repros/canonical/any_17b234fd7f4f/oracle_any_reduce.py --bench --warmup 10 --rep 50`
- Oracle result: `oracle_us=4.38`, `compile_us=5.82`, `ratio=1.328`, `status=GOOD`
- Interleaved compile comparison command: `python scripts/bench_compare.py repros/canonical/any_17b234fd7f4f/repro.py --config baseline --label default --config coordinate_descent_tuning=True --label coord_descent --config combo_kernels=True,combo_kernel_per_subkernel_blocks=True,coordinate_descent_tuning=True,benchmark_combo_kernel=True,triton.multi_kernel=2 --label combo_persistent_cd --config combo_kernels=True,combo_kernel_per_subkernel_blocks=True,coordinate_descent_tuning=True,benchmark_combo_kernel=True,triton.multi_kernel=3 --label combo_looped_cd --n-warmup 10 --n-rep 50 --rounds 5`
- Interleaved results: `default=4.35us`, `coord_descent=4.35us`, `combo_persistent_cd=4.32us`, `combo_looped_cd=4.42us`; fastest `combo_persistent_cd`.

## Classification

- Classification: `BANDWIDTH_BOUND`
- Rationale: the full-scope oracle is one Triton reduction kernel that reads the required 8192 f32 elements and stores one bool scalar. Interleaved compile configs are already at the same launch-scale floor, and combo/scheduler settings move runtime by only about 1%. This does not need a scheduler-fusion fix.

## Done Criteria

- Full-scope Triton oracle implemented.
- Correctness checked against eager.
- Oracle benchmark and required compile configs measured.
