Gap diagnosis (classification: SCHEDULER_FUSION): the full-scope Triton oracle
computes the batchnorm-like affine, ReLU, `[2, 1]` low-memory max-pool with
int8 offsets, and the final `[1,32,512]` eager-layout f32 permute view in one
kernel. Inductor currently generates two kernels: the elementwise producer and
the max-pool-with-offsets consumer. The missing compiler capability is scheduler
fusion through this structured pooling op plus direct layout-aware multi-output
stores for both returned tensors.

Status: full-scope oracle implemented in
`repros/canonical/pointwise_7727cefe098e/oracle_layout.py`. This is not a
pointwise-only or maxpool-only subset; it returns both eager outputs:
`int8[1,512,1,32]` offsets and the non-contiguous `float32[1,32,512]` permuted
view with stride `(16384, 1, 32)`.

Measurements:

- `python repros/canonical/pointwise_7727cefe098e/oracle_layout.py --check`:
  PASS for both outputs. Output 0 is exact; output 1 matches with
  `max_diff=3.81e-06`, equal NaN mask, and eager stride `(16384, 1, 32)`.
- `python repros/canonical/pointwise_7727cefe098e/oracle_layout.py --bench --warmup 10 --rep 50`:
  oracle `4.10 us`, coordinate-descent compile `14.34 us`, ratio `3.50x`,
  status `GOOD`.
- `python scripts/bench_compare.py repros/canonical/pointwise_7727cefe098e/repro.py --config "coordinate_descent_tuning=True" --label cd --config "combo_kernels=True,combo_kernel_per_subkernel_blocks=True,coordinate_descent_tuning=True,benchmark_combo_kernel=True,triton.multi_kernel=2" --label combo_persistent_cd --config "combo_kernels=True,combo_kernel_per_subkernel_blocks=True,coordinate_descent_tuning=True,benchmark_combo_kernel=True,triton.multi_kernel=3" --label combo_looped_cd --rounds 5 --n-warmup 10 --n-rep 50 --max-workers 1 --output /tmp/pointwise_7727cefe098e_bench_compare.json`:
  `cd=5.888000130653381 us`, `combo_persistent_cd=6.016000173985958 us`,
  `combo_looped_cd=6.912000011652708 us`.

Recommended parent status: `implemented_unmeasured` with
`classification=SCHEDULER_FUSION` and `true_floor=yes`; do not mark
`not_true_floor`.
