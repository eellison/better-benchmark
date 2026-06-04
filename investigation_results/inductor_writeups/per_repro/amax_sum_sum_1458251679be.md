# amax_sum_sum_1458251679be

## Classification: `SCATTER_REDUCE`

## Oracle State

- Oracle path: `repros/canonical/amax_sum_sum_1458251679be/oracle_full_scope.py`
- Oracle status: `true_oracle_measured`
- Scope: full `Repro.forward` output, not a softmax-only or reduction-only floor.
- Output invariant: returns the final contiguous `float32[4, 8192]` `constant_pad_nd` result.

The oracle computes the scalar `div` and mask `where` producers, the row-local
softmax backward over `[8, 2]`, the `index_put(accumulate=True)` into the
logical `[8, 1024, 2]` scatter buffer, and the final `view -> permute ->
constant_pad_nd` layout in one Triton kernel that writes the padded transpose
directly.

## Measurements

- `python repros/canonical/amax_sum_sum_1458251679be/oracle_full_scope.py --check`: PASS, max diff `5.96e-08`.
- `python repros/canonical/amax_sum_sum_1458251679be/oracle_full_scope.py --bench --warmup 10 --rep 50`: oracle `5.47 us`, compile `20.51 us`, ratio `3.749x`, status `GOOD`.
- `bench_compare` (`--n-warmup 10 --n-rep 50 --rounds 3`): default `7.65 us`, CD `7.78 us`, combo+CD `6.94 us`; fastest compile config was combo+CD.

## Diagnosis

Inductor currently treats the decomposed softmax-gradient producer and
duplicate-index scatter as ordinary scheduled work around a materialized
logical scatter buffer and separate layout/pad epilogue. The missing capability
is a scatter-reduce lowering that keeps the tiny row-local softmax-backward
producer in registers and emits the final padded transpose directly.

Recommended parent status: `implemented_unmeasured` with `true_floor=yes`.
The CSV queues were intentionally not edited.
