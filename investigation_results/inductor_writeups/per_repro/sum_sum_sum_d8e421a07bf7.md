# sum_sum_sum_d8e421a07bf7
## Compile: 39us, Oracle: 26us, Gap: 1.54x
## Diagnosis: ALGEBRAIC_ELIMINATION
## Root cause: Inductor preserves the literal dependent-reduction graph and does not prove the linear identity that collapses sum((where - mean - centered * coef) * scale) into sibling summaries; it schedules the first two DenseNet BN-backward channel reductions, broadcast arithmetic, slice add, and final channel reduction as separate generic pointwise/reduction work over large intermediates (N=128, C=32, H=32, W=32).
## Fix path: Add a guarded dependent-reduction rewrite for this BN-backward form that introduces the needed sum(centered) and sum(slice) summaries and lowers the full tuple as one multi-accumulator reduction plus a small vector epilogue, eliminating the extra pass over the large spatial domain.
## Status: design_todo
