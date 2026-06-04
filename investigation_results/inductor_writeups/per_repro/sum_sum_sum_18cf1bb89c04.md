# sum_sum_sum_18cf1bb89c04
## Compile: 166us, Oracle: 146us, Gap: 1.14x
## Diagnosis: SCATTER_REDUCE
## Root cause: Inductor combo_persistent is within 14% of the oracle that fuses layernorm-backward row reductions with three indexed accumulator outputs (position, segment, vocabulary); the remaining gap reflects the overhead of generic kernel launches vs a single fused producer.
## Fix path: Structured embedding scatter-reduce lowering with three indexed outputs; small gap (20us, 1.14x) means this is effectively at floor. Low priority.
## Status: at_floor
