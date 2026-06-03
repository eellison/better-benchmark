# sum_sum_f10bfbcf6729

Full-scope oracle: `repros/canonical/sum_sum_f10bfbcf6729/oracle_multi_output_reduction.py`

Gap diagnosis (classification: `BANDWIDTH_BOUND`): The oracle covers the same computation scope as `repro.py`: it replaces the `avg_pool2d_backward(view(mm), ...)` producer with the equivalent `mm / 16` broadcast, applies the `arg102 <= 0` mask, shares that masked producer across both channel reductions, writes the `[512]` side output, and computes the full `[1024,512,4,4]` BN-backward epilogue with matching strides. The tuned Inductor coordinate-descent path is already faster than this full-scope handwritten oracle and close to the byte-accounted floor for the graph, so this does not identify a missing scheduler fusion worth treating as an oracle floor. The corresponding Inductor action is `BANDWIDTH_BOUND`: no fix from this oracle; leave queue integration as diagnosis-only unless a future oracle beats both compile configs.

Correctness:
- `--check`: PASS
- output 0 `[1024,512,4,4]`: max_abs `1.192093e-07`, max_rel `1.675696e-02`, stride match true
- output 1 `[512]`: max_abs `1.525879e-05`, max_rel `2.460729e-04`, stride match true

Benchmark (`--bench --warmup 10 --rep 50`):
- Oracle full-scope masked dual reduction + epilogue: `61.888 us`
- `coordinate_descent_tuning=True`: `59.552 us`
- `combo_kernels=True,combo_kernel_per_subkernel_blocks=True,coordinate_descent_tuning=True,benchmark_combo_kernel=True,triton.multi_kernel=3`: `69.792 us`

Valid floor: no. The oracle is full-scope and useful as a diagnosis artifact, but it is slower than the coordinate-descent compile path.
