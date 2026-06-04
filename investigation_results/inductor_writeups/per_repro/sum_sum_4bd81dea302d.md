# sum_sum_4bd81dea302d
## Compile: 81us, Oracle: 62us, Gap: 1.29x
## Diagnosis: SCATTER_REDUCE
## Root cause: Inductor materializes two zero-filled [6291456] scatter buffers for the overlapping-window index_puts, then schedules transposes and reductions separately; the oracle decodes scatter indices to direct loads, writes both transposed side outputs, and accumulates both [768] sums in one tile.
## Fix path: Structured Longformer scatter-reduce lowering targeting final transposed layouts, fusing scale/reduction epilogues without scatter buffers; borderline gap (1.29x, 18us).
## Status: borderline
