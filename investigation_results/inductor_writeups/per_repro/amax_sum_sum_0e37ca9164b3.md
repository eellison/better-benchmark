# amax_sum_sum_0e37ca9164b3
## Compile: 41.8us, Oracle: 33.6us, Gap: 1.24x
## Diagnosis: new_pattern (ignore-index cross-entropy mean)
## Root cause: Inductor lowers the view/amax/sub/exp/sum/log/sub/gather/where/sum/count/div graph as separate generic row reductions, pointwise kernels, and scalar reductions that materialize the full log-softmax intermediate; it does not recognize the ignore-index cross-entropy mean pattern as a single fused row-reduction plus scalar epilogue.
## Fix path: Add an Inductor pattern/lowering for log_softmax + masked gather + loss/count mean that emits an online cross-entropy row kernel and a small final scalar reduction directly.
## Status: design_todo
