# sum_sum_sum_1d10e6314ef6

Full-scope Triton oracle: `repros/canonical/sum_sum_sum_1d10e6314ef6/oracle_multi_output_reduction.py`

Gap diagnosis (classification: COOPERATIVE_SPLIT_K): the oracle covers the same scope as `repro.py`, including the `mm_134` view, residual add with `mul_290`, affine multiply by `arg5_1`, the two row-local layernorm-backward reductions, `arg298_1` scaling, boolean mask application from `arg103_1`, the two sibling `[768]` residual-input reductions, the masked-gradient `[768]` reduction, and the returned `[768, 8192]` transpose view. It differs from Inductor by computing each row tile's scalar reductions, masked gradient side-output store, and three column partials in one Triton pass, followed by a small finalizer over row-tile partials, instead of scheduling the row reductions, column sums, mask multiply, and transpose-producing pointwise region as separate template work. Inductor cannot do this today because the scheduler does not represent a dependent multi-output reduction with row-wise scalar intermediates plus a materialized transpose side output as one cooperatively split reduction; the fix is COOPERATIVE_SPLIT_K support for dependent multi-output reductions that accumulate compatible column partials while writing the side output.

Correctness with `--check`: PASS. Outputs match the repro shapes and strides; max abs errors were output0 `6.103516e-05`, output1 `7.629395e-05`, output2 `6.250000e-02`, and output3 `2.929688e-03`.

Benchmark with `--bench --warmup 10 --rep 50`: oracle full-scope Longformer reductions plus transpose `52.128 us`; `coordinate_descent_tuning=True` `104.160 us`; `combo_kernels=True,combo_kernel_per_subkernel_blocks=True,coordinate_descent_tuning=True,benchmark_combo_kernel=True,triton.multi_kernel=3` `129.248 us`.

Valid floor: yes. The timed oracle covers the full compiled repro scope and is faster than both required compile configurations in this run.
