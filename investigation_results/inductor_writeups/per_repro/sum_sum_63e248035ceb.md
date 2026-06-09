# sum_sum_63e248035ceb
## Compile: 38.2us, Oracle: 25.3us, Gap: 1.51x
## Diagnosis: cooperative_split_k
## Root cause: Inductor schedules the GhostNet batch-norm-backward memory-format copy/slice producer, sibling sum([0,2,3]) channel reductions, and dependent BN-backward epilogues as separate generic reductions and pointwise kernels over materialized intermediates, without coordinating the reductions across the (N, H, W) domain or emitting channels-last output stores.
## Fix path: Cooperative split-K multi-output reduction template that splits compatible small-output channel reductions across the reduced (N, H, W) domain, combines partial summaries, and fuses the downstream tensor and vector epilogues with memory-format-aware stores.
## Status: design_todo
