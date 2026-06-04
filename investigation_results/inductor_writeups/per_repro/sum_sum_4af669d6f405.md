# sum_sum_4af669d6f405
## Compile: 145us, Oracle: 120us, Gap: 1.21x
## Diagnosis: COOPERATIVE_SPLIT_K
## Root cause: Inductor schedules the channel-shuffle (cat/view/permute/clone) producer and sibling BN-backward reductions separately; the oracle reconstructs the shuffled layout in-register and cooperatively reduces both summaries in one pass, saving the intermediate materialization.
## Fix path: Cooperative split-K multi-output reduction with layout-changing producer fusion; borderline 1.21x gap means this is low-priority.
## Status: borderline
