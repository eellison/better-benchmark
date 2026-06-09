# sum_7c50e5b80bfe

Full-scope oracle: `repros/canonical/sum_7c50e5b80bfe/oracle_multi_output_reduction.py`

Gap diagnosis: `SCHEDULER_FUSION`. The oracle uses one Triton kernel to compute both `[256, 1]` column producers, store the full permuted `[2, 256]` materialization with stride `(1, 2)`, and accumulate the `[2]` column sum while the values are still in registers; Inductor cannot do this today because the scheduler does not fuse the shared pointwise/cat producer into both a materialized permute side output and a sibling reduction output in one loop nest, so the required Inductor change is scheduler fusion for this shared producer plus small reduction.

Results: `--check` PASS with output 0 max diff `0.00e+00` and output 1 max diff `1.53e-05`; oracle bench `oracle_us=3.87`, tuned `compile_us=26.98`, ratio `6.967`, `GOOD`; interleaved `bench_compare` required configs measured `coordinate_descent_tuning=True` at `6.943999789655209 us` and combo/CD tuning at `7.104000076651573 us`. Parent should set `implemented_unmeasured`, not `not_true_floor`.
