# sum_sum_02744d87feff
## Compile: 145us, Oracle: 94us, Gap: 1.54x
## Diagnosis: COOPERATIVE_SPLIT_K
## Root cause: Inductor schedules the DenseNet BN-backward paired channel reductions and sliced residual-add epilogue as separate kernels; the oracle cooperatively splits the N/H/W domain, combines partial summaries, and fuses the scale-gradient vector + sliced residual output in one finalizer.
## Fix path: Cooperative split-K multi-output reduction template: split compatible BN channel reductions, combine partial summaries, fuse vector and slice epilogues with upstream residual additions.
## Status: design_todo
