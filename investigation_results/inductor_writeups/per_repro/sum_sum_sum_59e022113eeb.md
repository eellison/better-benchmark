# sum_sum_sum_59e022113eeb

Gap diagnosis (classification: COOPERATIVE_SPLIT_K): the full-scope Triton oracle covers the same scope as `repro.py`, including the `mm_66` view, residual add with `mul_152`, affine multiply by `arg8_1`, the two row-local reductions over the last dimension, `arg164_1` scaling, the two sibling `[768]` reductions over the residual input, the `[768]` reduction of the scaled gradient, and the returned `[768, 32768]` transpose view. It differs from Inductor by computing each row tile's scalar reductions, transpose backing store, and three compatible column partials in one Triton pass, followed by a small split-K finalizer for the three vector outputs, instead of scheduling row reductions, the dependent pointwise epilogue, column reductions, and transpose-producing view as separate template work. Inductor cannot do this today because the scheduler does not represent a dependent multi-output reduction with row-wise scalar intermediates plus a materialized transpose side output as one cooperatively split reduction; the fix is COOPERATIVE_SPLIT_K support for dependent multi-output reductions that accumulate compatible column partials while writing the side output.

Full-scope Triton oracle: `repros/canonical/sum_sum_sum_59e022113eeb/oracle_multi_output_reduction.py`

Correctness with `--check`: PASS. Outputs match the repro shapes and strides, including the returned transpose stride `(1, 768)`; max abs errors were output0 `2.021790e-04`, output1 `1.831055e-04`, output2 `3.906250e-03`, and output3 `2.500000e-01`.

Benchmark with `--bench --warmup 10 --rep 50`: oracle full-scope DistilBERT reductions plus transpose `149.984 us`; `coordinate_descent_tuning=True` `172.640 us`; `combo_kernels=True,combo_kernel_per_subkernel_blocks=True,coordinate_descent_tuning=True,benchmark_combo_kernel=True,triton.multi_kernel=3` `171.360 us`.

Valid floor: yes. The timed oracle covers the full compiled repro computation, outputs, dtypes, and output strides, and is faster than both required compile configurations in this run.
