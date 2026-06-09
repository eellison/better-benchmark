# sum_sum_sum_e2388f04f7c2
## Compile: 125us, Oracle: 72us, Gap: 1.72x
## Diagnosis: COOPERATIVE_SPLIT_K
## Root cause: Inductor schedules the selected-real [32,512,768] producer, row-local layernorm reductions, mask multiply, transpose side output, and sibling column sums as separate generic kernels; the oracle row-tiles the producer, writes the transposed side output directly, and cooperatively accumulates all three [768] column sums from the same tiles.
## Fix path: Cooperative split-K row-tiled producer: read selected-view inputs directly, write required side outputs while emitting column partials, finalize compatible layernorm-backward reductions together.
## Status: design_todo
