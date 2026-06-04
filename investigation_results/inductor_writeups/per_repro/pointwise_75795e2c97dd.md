# pointwise_75795e2c97dd

Full-scope oracle: `repros/canonical/pointwise_75795e2c97dd/oracle_layout.py`.

Gap diagnosis (classification: `BANDWIDTH_BOUND`): the repro reads the default
contiguous `int64[8, 1024]` input, computes `ne(input, 1)`, converts the bool
predicate to `int32`, and returns a fresh contiguous `int32[8, 1024]` tensor
with stride `(1024, 1)`. The oracle covers the full scope with one Triton
pointwise kernel that loads the input and stores the final `int32` output
directly. Inductor already lowers this standalone predicate/conversion chain to
the same one-launch materialization shape of work, so the remaining cost is the
required input read, output write, allocation, and launch.

Measurements:

- `python repros/canonical/pointwise_75795e2c97dd/oracle_layout.py --check`:
  PASS; output 0 exact `torch.int32` match; layout PASS with shape `[8, 1024]`,
  stride `(1024, 1)`, and dtype `torch.int32`.
- `python repros/canonical/pointwise_75795e2c97dd/oracle_layout.py --bench --warmup 10 --rep 50`:
  `oracle_us=3.65`, harness `compile_us=3.68`, ratio `1.009`, status
  `AT_FLOOR`.
- `python scripts/bench_compare.py repros/canonical/pointwise_75795e2c97dd/repro.py --config "coordinate_descent_tuning=True" --label cd --config "combo_kernels=True,combo_kernel_per_subkernel_blocks=True,coordinate_descent_tuning=True,benchmark_combo_kernel=True,triton.multi_kernel=3" --label combo --rounds 5 --n-warmup 10 --n-rep 50 --max-workers 1 --output /tmp/pointwise_75795e2c97dd_bench_compare.json`:
  coordinate descent `4.1600000113248825 us`, combo looped coordinate descent
  `4.224000032991171 us`, `total_bytes=98304`, `rep_per_round=500`.
- `python -m py_compile repros/canonical/pointwise_75795e2c97dd/repro.py repros/canonical/pointwise_75795e2c97dd/oracle_layout.py`:
  PASS.
- `python scripts/validate_corpus_invariants.py`: PASS; all hard invariants
  satisfied.

Parent status recommendation: `not_true_floor` / already at floor. The oracle
is exact and full-scope, but it is within noise of the local compiled parent and
this is only a one-kernel 96 KiB materialization. The queue CSVs were
intentionally not edited.
