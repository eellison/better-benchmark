# pointwise_6c03626e988e

Full-scope oracle: `repros/canonical/pointwise_6c03626e988e/oracle_layout.py`.

Gap diagnosis (classification: `BANDWIDTH_BOUND`): the repro has no inputs and
returns one fresh contiguous `float32[32, 1, 1, 512]` tensor filled with
`-0.0`. The oracle covers the full scope by allocating the exact eager layout
with stride `(512, 512, 512, 1)` and materializing the f32 negative-zero bit
pattern with one Triton fill kernel. Inductor already performs equivalent tiny
constant-fill work for the returned tensor, and cannot remove the allocation,
launch, or 64 KiB output materialization without changing the observable fresh
tensor contract, so the remaining difference is a launch/materialization floor
rather than scheduler fusion, scatter-reduce, split-K, algebraic elimination,
or recomputation fusion.

Measurements:

- `python repros/canonical/pointwise_6c03626e988e/oracle_layout.py --check`:
  PASS; output shape `[32, 1, 1, 512]`, dtype `torch.float32`, stride
  `(512, 512, 512, 1)`, `max_diff=0.00e+00`, and negative-zero bit pattern PASS.
- `python repros/canonical/pointwise_6c03626e988e/oracle_layout.py --bench --warmup 10 --rep 50`:
  `oracle_us=3.39`, harness `compile_us=3.36`, ratio `0.991`, status `AT_FLOOR`.
- `python scripts/bench_compare.py repros/canonical/pointwise_6c03626e988e/repro.py --config "coordinate_descent_tuning=True" --label cd --config "combo_kernels=True,combo_kernel_per_subkernel_blocks=True,coordinate_descent_tuning=True,benchmark_combo_kernel=True,triton.multi_kernel=3" --label combo --rounds 5 --n-warmup 10 --n-rep 50 --max-workers 1 --output /tmp/pointwise_6c03626e988e_bench_compare.json`:
  `cd=3.8079998921602964 us`, `combo=3.776000114157796 us`, `rep_per_round=500`,
  `total_bytes=65536`.
- `python -m py_compile repros/canonical/pointwise_6c03626e988e/repro.py repros/canonical/pointwise_6c03626e988e/oracle_layout.py`:
  PASS.
- `python scripts/validate_corpus_invariants.py`: PASS.

Parent status recommendation: `not_true_floor`, not `implemented_unmeasured`.
The oracle is exact and full-scope, but the standard oracle harness measured the
compiled parent slightly faster than the hand-written Triton materialization
kernel, so this is diagnosis-only launch/materialization-floor behavior. CSVs
were intentionally not edited.
