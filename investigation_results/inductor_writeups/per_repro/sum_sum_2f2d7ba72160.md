# sum_sum_2f2d7ba72160
## Compile: 68.9us, Oracle: 55.1us, Gap: 1.25x
## Diagnosis: cooperative_split_k
## Root cause: Inductor schedules the DenseNet batch-norm-backward sibling channel reductions (sum over N/H/W domain), the masked BN-backward producer, and the last-32-channel sliced add as separate generic reduction and pointwise kernels with materialized intermediates instead of cooperatively splitting the reduced spatial domain.
## Fix path: Cooperative split-K multi-output reduction template that splits compatible small-output channel reductions across the N/H/W domain, combines partial summaries, and fuses downstream vector and sliced tensor epilogues with the slice additions.
## Status: design_todo
