# sum_sum_sum_7ebfef460635
## Compile: 780.0us, Oracle: 500.7us, Gap: 1.56x
## Diagnosis: scatter_reduce (structured relative-position scatter)
## Root cause: Inductor lowers each of 12 sum(dim=0) -> full/slice_scatter/constant_pad_nd -> squeeze/permute/reshape -> index_put(accumulate=True) branches as separate generic reduction, layout materialization, and scatter kernels, without recognizing the zero-fill slice_scatter plus negative-pad crop as a structured no-op or fusing the batch reduction with indexed accumulation across repeated return branches.
## Fix path: Add a structured scatter-reduce lowering that folds the slice_scatter/pad view chain, reduces source tiles over batch, and accumulates duplicate relative-position indices directly into output buckets for every branch in the full return tuple.
## Status: design_todo
