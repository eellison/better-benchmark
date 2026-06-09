# pointwise_5aafedcd8ca4


## Measured Timings
- Oracle: 3.36 us
- Compile (CDT): 5.18 us
- Ratio: 1.54x

Full-scope oracle: `repros/canonical/pointwise_5aafedcd8ca4/oracle_layout.py`.

Gap diagnosis (classification: `BANDWIDTH_BOUND`): the repro has no inputs and
returns one fresh contiguous `float16[1, 1, 256, 257]` tensor filled with
`-inf`. The oracle covers the full scope by allocating the exact eager layout
with stride `(65792, 65792, 257, 1)` and materializing the fp16 `-inf` bit
pattern with one packed Triton fill kernel. Inductor already performs the same
essential no-load constant-store work for the returned tensor, so the remaining
cost is fresh allocation, one launch, and the 131,584-byte output store rather
than a missing scheduler fusion, scatter-reduce, split-K, algebraic, or
recompute optimization.

Measurements:

- `python repros/canonical/pointwise_5aafedcd8ca4/oracle_layout.py --check`:
  PASS; output shape `[1, 1, 256, 257]`, dtype `torch.float16`, stride
  `(65792, 65792, 257, 1)`, and fp16 `-inf` bit pattern PASS.
- `python repros/canonical/pointwise_5aafedcd8ca4/oracle_layout.py --bench --warmup 10 --rep 50`:
  `oracle_us=3.36`, harness `compile_us=3.20`, ratio `0.952`, status `AT_FLOOR`.
- `python scripts/bench_compare.py repros/canonical/pointwise_5aafedcd8ca4/repro.py --config "coordinate_descent_tuning=True" --label cd --config "combo_kernels=True,combo_kernel_per_subkernel_blocks=True,coordinate_descent_tuning=True,benchmark_combo_kernel=True,triton.multi_kernel=3" --label combo --n-warmup 10 --n-rep 50 --rounds 5 --max-workers 1 --output /tmp/pointwise_5aafedcd8ca4_bench_compare.json`:
  `cd=3.7120000924915075 us`, `combo=3.6800000816583633 us`,
  `total_bytes=131584`, `rep_per_round=500`.
- `python -m py_compile repros/canonical/pointwise_5aafedcd8ca4/repro.py repros/canonical/pointwise_5aafedcd8ca4/oracle_layout.py`:
  PASS.
- `python scripts/validate_corpus_invariants.py`: PASS; all hard invariants
  satisfied.

Parent status recommendation: `not_true_floor`, not `implemented_unmeasured`.
The oracle is exact and full-scope, but the standard oracle harness measured
the compiled parent slightly faster than the hand-written Triton fill kernel.
Keep the artifact for diagnosis; the parent queue should treat this as
bandwidth-bound fresh tensor materialization. CSVs were intentionally not
edited.
