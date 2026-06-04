# sum_sum_505c30a7fb41
## Compile: 188us, Oracle: 109us, Gap: 1.72x
## Diagnosis: COOPERATIVE_SPLIT_K
## Root cause: Inductor schedules DenseNet BN-backward paired channel reductions and the sliced residual-add epilogue as separate generic reduction and pointwise kernels; the oracle split-K-reduces over N/H/W, finalizes summaries, and writes the full return tuple including sliced residual additions in one epilogue.
## Fix path: Cooperative split-K multi-output reduction: coordinate compatible BN channel reductions over shared N/H/W domain, combine partials, fuse vector + slice epilogues with upstream residual slice inputs.
## Status: design_todo
