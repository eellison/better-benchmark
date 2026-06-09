# max_448f6b3c8f1d

Full-scope oracle: `repros/canonical/max_448f6b3c8f1d/oracle_max_reduce.py`

Gap diagnosis: `BANDWIDTH_BOUND`. The oracle computes the exact `int64[1,4096]`
to `int64[]` max with a single Triton `tl.max` block reduction. The compiled
repro is already at the one-launch small-reduction floor under the required
tuned configs, so there is no distinct scheduler fusion, cooperative split-K,
scatter-reduce, algebraic elimination, or new-pattern opportunity here.

Results measured 2026-06-04:

- `--check`: PASS, exact `torch.int64` scalar match.
- `--bench --warmup 10 --rep 50`: oracle `4.54 us`, harness compile `4.26 us`,
  ratio `0.937`, status `BAD_ORACLE`.
- Required `bench_compare.py` configs, `--rounds 5 --n-warmup 10 --n-rep 50`:
  `coordinate_descent_tuning=True` `4.575999919325113 us`; combo looped CD
  config `4.6720001846551895 us`.

Parent status should be `not_true_floor`: the full-scope oracle is correct, but
it does not beat the local compile floor.
