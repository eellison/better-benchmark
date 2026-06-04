# pointwise_edfab1af9465

Full-scope oracle: `repros/canonical/pointwise_edfab1af9465/oracle_layout.py`.

Gap diagnosis (classification: `BANDWIDTH_BOUND`): the repro computes a single
`aten.tanh.default` over the default contiguous `float32[1000, 16]` input and
returns a fresh contiguous `float32[1000, 16]` tensor with stride `(16, 1)`.
The oracle covers exactly that full scope with one Triton pointwise kernel that
loads the input, evaluates `tanh`, and stores the returned layout directly.
Inductor already lowers the same single pointwise op to one Triton kernel, so
there is no scheduler-fusion, scatter-reduce, cooperative split-K,
algebraic-elimination, recompute-fusion, or new-pattern optimization to claim;
the remaining cost is the required read, tanh math, output materialization,
allocation, and launch floor.

Measurements from 2026-06-04:

- `python repros/canonical/pointwise_edfab1af9465/oracle_layout.py --check`:
  PASS; output 0 `max_diff=0.00e+00`; layout PASS with shape `[1000, 16]`,
  stride `(16, 1)`, dtype `torch.float32`, storage offset `0`.
- `python repros/canonical/pointwise_edfab1af9465/oracle_layout.py --bench --warmup 10 --rep 50`:
  `oracle_us=4.03`, harness `compile_us=3.84`, ratio `0.952`, status
  `AT_FLOOR`.
- `python scripts/bench_compare.py repros/canonical/pointwise_edfab1af9465/repro.py --config "coordinate_descent_tuning=True" --label cd --config "combo_kernels=True,combo_kernel_per_subkernel_blocks=True,coordinate_descent_tuning=True,benchmark_combo_kernel=True,triton.multi_kernel=2" --label combo_persistent_cd --config "combo_kernels=True,combo_kernel_per_subkernel_blocks=True,coordinate_descent_tuning=True,benchmark_combo_kernel=True,triton.multi_kernel=3" --label combo_looped_cd --rounds 5 --n-warmup 10 --n-rep 50 --max-workers 1 --output /tmp/pointwise_edfab1af9465_bench_compare.json`:
  `cd=4.255999810993671 us`, `combo_persistent_cd=4.31999983265996 us`,
  `combo_looped_cd=4.352000076323748 us`, `total_bytes=128000`,
  `rep_per_round=500`; fastest required config was `cd`.
- `python -m py_compile repros/canonical/pointwise_edfab1af9465/repro.py repros/canonical/pointwise_edfab1af9465/oracle_layout.py`:
  PASS.
- `python scripts/validate_corpus_invariants.py`: PASS; all hard invariants
  satisfied.

Parent status recommendation: `not_true_floor` / already at floor. The
full-scope oracle is exact and the required interleaved compile configs are in
the same launch-level band, while the standard oracle harness measures the
local compiled parent slightly faster than the handwritten Triton kernel. The
queue CSVs were intentionally not edited.
