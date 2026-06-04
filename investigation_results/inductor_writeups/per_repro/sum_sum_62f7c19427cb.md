# sum_sum_62f7c19427cb
## Compile: 166us, Oracle: 102us, Gap: 1.62x
## Diagnosis: COOPERATIVE_SPLIT_K
## Root cause: Inductor schedules DenseNet BN-backward sibling channel reductions and three surrounding residual slice adds as separate kernels; the oracle cooperatively splits the N/H/W domain, finalizes [160] scale-gradient, and fuses dependent [64,32,56,56] sliced input-gradient stores with residual producers.
## Fix path: Cooperative split-K multi-output reduction template with residual-slice side-output epilogues: coordinate BN channel reductions, combine partials, fuse downstream vector and residual-slice stores.
## Status: design_todo
