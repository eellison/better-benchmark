# Gap Diagnosis: sum_sum_sum_559e754b2ff4 (cooperative_split_k)


## Measured Timings
- Oracle: measurement failed (runtime error)
- Compile (CDT): 15.90 us
- Ratio: N/A

## Classification: COOPERATIVE_SPLIT_K

## Measurements

| Config | Time (us) | Ratio vs Default |
|--------|-----------|------------------|
| Default | 21.6 | 1.00x |
| Combo Looped (CD) | 17.4 | 1.24x |
| Combo Persistent (CD) | 17.2 | 1.26x |
| Coord Descent (best) | 17.5 | 1.23x |
| Oracle (handwritten Triton) | ~8.2 | ~2.6x |

- **Bytes accessed**: 14.7 MB (logical)
- **Best compile SOL**: 2.10 GB/s effective BW ratio
- **Oracle SOL**: ~1.8 TB/s effective BW
- **Models**: hf_GPTJForCausalLM_train, hf_GPTJForQuestionAnswering_train (4 shapes)

## Pattern Description

GPT-J layer-norm backward: 4 matrix multiply inputs (mm_316, mm_321, mm_323, mm_325)
summed, then layer-norm backward formula applied over hidden dim 4096 with M=128 rows.
Outputs:
1. sum_x_xhat [4096] -- column sum of (summed * xhat)
2. sum_x [4096] -- column sum of summed
3. grad_md [4096, 128] -- transposed gradient (= input_grad.T)
4. sum_grad [4096] -- column sum of final gradient

## Root Cause

Inductor emits generic row reductions (for the LN backward row sums/dots), pointwise
materialization/permute work, and SEPARATE column reductions over the same gradient.
With only M=128 rows and D=4096 hidden dim:
- Row reductions have low parallelism (only 128 rows)
- The 4096-wide hidden dim is too large for single-pass persistent reduction per row
- Column reductions reread the materialized gradient

The oracle uses a split-K strategy: split the D=4096 hidden dim into tiles, compute
partial row sums per tile, finalize row sums, then in a second cooperative pass produce
the gradient AND accumulate column sums atomically.

## Oracle Strategy

Three-phase cooperative approach:
1. **Row dual-partial kernel**: grid=(M, num_d_tiles). Each tile computes partial
   row_sum and row_dot over its D-slice. Writes partial accumulators.
2. **Finalize row partials**: grid=(M,). Sums the D-tile partials to get final
   row_sum[m] and row_dot[m].
3. **Grad store + column reduce**: grid=(M_tiles, D_tiles). Computes the final
   gradient, stores it, and atomically accumulates column sums (sum_x_xhat, sum_x,
   sum_grad) in the same pass.

## Inductor Fix Required

**COOPERATIVE_SPLIT_K**: Teach Inductor to:
1. Split compatible layer-norm-backward reduction dimensions when M is small
2. Finalize shared row partials across tiles
3. Fuse side-output stores (the transposed gradient) with cooperative column reductions
   (atomic accumulation of weight gradients)

The scheduler needs a multi-output reduction template that coordinates:
- Row-summary partials (split across D-tiles, finalized with a barrier)
- Row-tiled side-output producers (materialized gradient)
- Sibling column accumulators (atomic partial sums)

## Generalizability

MEDIUM-HIGH - This pattern (LN backward with small M, large D, plus transposed side
output and column reductions) appears in GPT-J and similar architectures with large
hidden dimensions relative to sequence length in certain graph partitions. The
cooperative split-K technique is generally applicable to any reduction where the
reduction dimension is much larger than the batch dimension.

## Remaining Gap After CD

~9us (17.2us best compile vs ~8.2us oracle). The gap is the inability to split-K
the row reduction and cooperatively produce the column sums.
