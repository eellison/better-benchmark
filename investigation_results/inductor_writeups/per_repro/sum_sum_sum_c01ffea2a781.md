# sum_sum_sum_c01ffea2a781
## Compile: 196us, Oracle: 196us, Gap: 1.00x
## Diagnosis: SCATTER_REDUCE
## Root cause: The oracle embedding scatter-reduce (fusing layernorm-backward, dropout, hidden column reductions, and dual index_put accumulations) matches Inductor within measurement noise; Inductor generic kernels already reach the same throughput on this shape.
## Fix path: No fix needed; Inductor already at oracle parity. The scatter-reduce fusion provides no benefit over generic kernels at this sequence length and vocabulary size.
## Status: at_floor
