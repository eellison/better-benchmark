# sum_sum_sum_f0377fc40fe2

Full-scope oracle for the DeiT tiny layernorm-backward tail. The oracle computes the same five outputs as `repro.py`: the two `[192]` reductions over the original `mm`/normalized producer, the `[1,197,192]` batch reduction after the residual add, the cls-token `[1,1,192]` reduction, and the patch-grid `[192]` reduction.

Gap diagnosis: the oracle differs from Inductor by computing the row-local layernorm backward scalars, sibling column partials, and token-sum atomics in one split-row Triton pass, then using small finalize kernels for the column, cls-token, and patch-grid outputs. Inductor cannot do this today because the graph mixes reductions over C, batch, token, and patch dimensions with a dependent row reduction; the scheduler therefore materializes/interleaves producer-consumer reductions instead of coordinating one multi-output reduction with row-tile partials and atomic/reduction finalization. Classification: `COOPERATIVE_SPLIT_K`.

Results from `--bench --warmup 10 --rep 50`:

| path | time |
| --- | ---: |
| oracle full-scope split-row/token-atomic reduction | 44.160 us |
| `coordinate_descent_tuning=True` | 73.984 us |
| combo-looped coordinate descent | 49.696 us |

Correctness from `--check`: PASS for all five outputs with matching shape and stride. This is a valid full-scope floor candidate.
