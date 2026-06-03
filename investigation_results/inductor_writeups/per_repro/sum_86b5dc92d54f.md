# sum_86b5dc92d54f

## Classification: SCATTER_REDUCE

## Oracle: oracle_longformer_index_put_sum_transpose.py

## Model: torchbench_hf_Longformer_train_005

## Measurements

- Compiled (cd): 31.5 us
- Oracle (triton): 93.7 us

## Diagnosis

Compile dramatically outperforms the oracle (3.0x faster). The oracle attempts to fuse the
Longformer duplicate-index `index_put(accumulate=True)` with the hidden-dimension sum and
transposed side-output into one structured scatter-reduce template. However, the oracle at 93.7us
is far worse than Inductor's 31.5us.

The oracle likely suffers from poor occupancy or suboptimal atomics for the duplicate-index
scatter pattern at this shape (768 hidden dim, 2048 sequence length). Inductor's generic lowering
with separate atomic index_put and downstream reduction/layout kernels is much more efficient here.

The oracle header diagnoses the pattern correctly (one-dimensional scatter-add with aliased views
feeding both a reduction and a layout-changing store), but the implementation is suboptimal.

## Status

AT_FLOOR -- compile outperforms oracle by 3.0x. Oracle needs rewrite but no Inductor work needed.

## Done Criteria

Closed. Compiled output already faster than oracle.
