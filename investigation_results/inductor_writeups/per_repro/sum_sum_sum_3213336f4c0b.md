# sum_sum_sum_3213336f4c0b
## Compile: 87us, Oracle: 52us, Gap: 1.67x
## Diagnosis: COOPERATIVE_SPLIT_K
## Root cause: Inductor materializes/schedules the triple-matmul add (permuted view), row-local layernorm reductions, mask multiply, transpose side output, and sibling column sums as separate kernels; the oracle row-tiles the [8,1024,768] producer, folds the three permuted matmul buffers, writes the transposed masked output, and cooperatively accumulates column sums.
## Fix path: Cooperative split-K with non-contiguous row-tile producers: tile compatible layernorm-backward producers across token rows, emit/finalize shared column partials, fuse layout-mapped producer plus masked transpose store.
## Status: design_todo
