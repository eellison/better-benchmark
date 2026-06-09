# pointwise_d9f9b7733ca3


## Measured Timings
- Oracle: 3.23 us
- Compile (CDT): 2.88 us
- Ratio: 0.89x

Full-scope oracle: `repros/canonical/pointwise_d9f9b7733ca3/oracle_layout.py`.

Gap diagnosis (classification: `BANDWIDTH_BOUND`): the repro has no inputs and
returns one fresh contiguous `float32[16, 1, 1, 512]` tensor filled with
`-0.0`; the oracle covers the full scope by allocating the exact eager layout
with stride `(512, 512, 512, 1)` and materializing the f32 negative-zero bit
pattern with one Triton fill kernel. Inductor already performs equivalent tiny
constant-fill work for the returned tensor, and the remaining cost is the
allocation, launch, and 32 KiB output store floor rather than a missing fusion,
scatter, split-K, algebraic-elimination, or recomputation optimization.

Measurements:

- `python repros/canonical/pointwise_d9f9b7733ca3/oracle_layout.py --check`:
  PASS; output shape `[16, 1, 1, 512]`, dtype `torch.float32`, stride
  `(512, 512, 512, 1)`, `max_diff=0.00e+00`, and negative-zero bit pattern PASS.
- `python repros/canonical/pointwise_d9f9b7733ca3/oracle_layout.py --bench --warmup 10 --rep 50`:
  `oracle_us=3.23`, harness `compile_us=3.17`, ratio `0.980`, status `AT_FLOOR`.
- `python scripts/bench_compare.py repros/canonical/pointwise_d9f9b7733ca3/repro.py --config "coordinate_descent_tuning=True" --label cd --config "combo_kernels=True,combo_kernel_per_subkernel_blocks=True,coordinate_descent_tuning=True,benchmark_combo_kernel=True,triton.multi_kernel=3" --label combo --n-warmup 10 --n-rep 50 --rounds 5 --max-workers 1`:
  `cd=3.648000070825219 us`, `combo=3.616000059992075 us`, `rep_per_round=500`.
- `python -m py_compile repros/canonical/pointwise_d9f9b7733ca3/repro.py repros/canonical/pointwise_d9f9b7733ca3/oracle_layout.py`:
  PASS.
- `python scripts/validate_corpus_invariants.py`: PASS.

Parent status recommendation: `not_true_floor`, not `implemented_unmeasured`.
The oracle is exact and full-scope, but the standard oracle harness measured the
compiled parent slightly faster than the hand-written Triton materialization
kernel, so this is diagnosis-only launch/materialization-floor behavior. CSVs
were intentionally not edited.
