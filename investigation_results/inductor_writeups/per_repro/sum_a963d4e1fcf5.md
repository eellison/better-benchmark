Gap diagnosis (classification: `BANDWIDTH_BOUND`): the full-scope Triton oracle
computes the complete `Repro.forward` row L2 normalization in one row-reduction
kernel. It loads `arg0[:, -1, :]` directly from the original contiguous
`f32[64,50,256]` input, reduces the 256-wide sum of squares for each row,
applies `sqrt` and `clamp_min(1e-12)`, divides the selected row by the row
denominator, and returns the required contiguous `f32[64,256]` output. Inductor
is already at the practical one-launch floor for this tiny capture, so the
remaining difference is hand-written scheduling noise rather than a missing
scheduler-fusion, scatter-reduce, split-K, algebraic-elimination, or recompute
fusion opportunity.

Results:

- `python repros/canonical/sum_a963d4e1fcf5/oracle_row_norm.py --check`: PASS,
  output `[64,256]` f32, max diff `2.98e-08`, expected/oracle stride `(256, 1)`.
- `python repros/canonical/sum_a963d4e1fcf5/oracle_row_norm.py --bench --warmup 10 --rep 50`:
  oracle `4.13 us`, compile `4.06 us`, ratio `0.984`, status `AT_FLOOR`.
- `python scripts/bench_compare.py repros/canonical/sum_a963d4e1fcf5/repro.py --config "coordinate_descent_tuning=True" --label cd --config "combo_kernels=True,combo_kernel_per_subkernel_blocks=True,coordinate_descent_tuning=True,benchmark_combo_kernel=True,triton.multi_kernel=3" --label combo --rounds 5 --n-warmup 10 --n-rep 50 --max-workers 1 --output /tmp/sum_a963d4e1fcf5_bench_compare.json`:
  interleaved `cd=4.480000119656324 us`, `combo=4.544000141322613 us`,
  `rep_per_round=500`.

Parent disposition: recommended parent status is `not_true_floor` with
`classification=BANDWIDTH_BOUND`. The oracle is exact and full-scope, but the
standard oracle harness compiled path is already slightly faster than the
hand-written Triton row kernel.
