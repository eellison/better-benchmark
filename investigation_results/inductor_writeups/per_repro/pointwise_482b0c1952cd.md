# pointwise_482b0c1952cd


## Measured Timings
- Oracle: 4.06 us
- Compile (CDT): 5.63 us
- Ratio: 1.39x

Full-scope oracle: `repros/canonical/pointwise_482b0c1952cd/oracle_layout.py`.

Gap diagnosis (classification: `BANDWIDTH_BOUND`): the repro computes the
Lennard-Jones broadcast pointwise expression
`tanh(arg2[:, 0:1] * arg0[:, 0].unsqueeze(0) + arg1)` and returns the complete
contiguous `float32[1000, 16]` tensor. The oracle does the same full scope in
one Triton kernel by loading the `[1000, 1]` row vector and two `[16]` column
vectors directly, avoiding the layout-only permute view and the multiply-by-one
nodes. Inductor already emits one fused pointwise Triton kernel for the captured
permute/mul/mul/add/tanh graph, so the remaining cost is required elementwise
math plus output materialization rather than a missing compiler transformation.

Measurements:

- `python repros/canonical/pointwise_482b0c1952cd/oracle_layout.py --check`:
  PASS, output shape `[1000, 16]`, stride `(16, 1)`, dtype `torch.float32`,
  max diff `1.19e-07`.
- `python repros/canonical/pointwise_482b0c1952cd/oracle_layout.py --bench --warmup 10 --rep 50`:
  `oracle_us=4.06`, harness `compile_us=3.71`, ratio `0.913`, status
  `BAD_ORACLE`.
- `python scripts/bench_compare.py repros/canonical/pointwise_482b0c1952cd/repro.py --config "coordinate_descent_tuning=True" --label cd --config "combo_kernels=True,combo_kernel_per_subkernel_blocks=True,coordinate_descent_tuning=True,benchmark_combo_kernel=True,triton.multi_kernel=3" --label combo --rounds 5 --n-warmup 10 --n-rep 50 --max-workers 1 --output /tmp/pointwise_482b0c1952cd_bench_compare.json`:
  coordinate descent `4.224000032991171 us`, combo
  `4.255999810993671 us`, `total_bytes=68128`, `rep_per_round=500`.
- `python -m py_compile repros/canonical/pointwise_482b0c1952cd/repro.py repros/canonical/pointwise_482b0c1952cd/oracle_layout.py`:
  PASS.
- `python scripts/validate_corpus_invariants.py`: PASS, all hard invariants
  satisfied.

Parent status recommendation: `not_true_floor`, not `implemented_unmeasured`.
The oracle is exact and full-scope, but the standard oracle harness measures the
local compiled parent faster than the hand-written Triton kernel, and generated
kernel counting confirms Inductor is already at one fused pointwise kernel for
this graph. The queue CSV was intentionally not edited.
