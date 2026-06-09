# sum_sum_sum_88c9224fa05a
## Oracle: oracle_fused_embedding_scatter (FNet layer-norm backward + embedding scatters)
## Compile: 166.0us, Oracle: 123.5us, Gap: 1.34x
## Diagnosis: SCATTER_REDUCE
## What the oracle does differently: Computes layer-norm-backward row reductions, two sibling column reductions, and three returned position/segment/vocabulary embedding-gradient scatter-add outputs from the same Triton producer, avoiding materialization of the full [32, 512, 768] layer-norm input-gradient tensor and re-reads through separate scatter paths.
## Inductor fix: Scheduler/codegen support for fused multi-output reduction/scatter templates -- the scheduler cannot currently represent one row-reduction producer that simultaneously feeds independent column reductions and multiple side-effect scatter-add stores (index_put) with different output layouts (position=[512,768], segment=[4,768], vocab=[32000,768]).
