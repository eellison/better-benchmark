# sum_sum_sum_a970af617f4c

## Compile: 208.03us, Oracle: 169.66us, Gap: 1.23x

## Diagnosis: NEW_PATTERN

## Root cause: Inductor schedules the MobileViT backward scope as 4 separate Triton kernels: row reductions for the hidden-dimension summaries, column reductions for the [144] weight/bias gradients, a pointwise epilogue for the residual add, and a reshape/permute/clone layout pipeline for the returned f32[128,144,32,32] tensor. The oracle fuses all of these into a coordinated plan that maps flattened patch rows directly to NCHW stores while sharing the row summaries and channel-reduction partials, avoiding intermediate materializations.

## Fix path: Add a MobileViT patch-fold reduction template that fuses the row-local reduction, sibling column reductions, residual add, and the reshape+permute+clone layout restore into one planned schedule. This requires Inductor's scheduler to recognize the patch-unfold pattern and sink the layout transform into the reduction epilogue.

## Status: design_todo

## Details

- Model: timm_mobilevit_s (train)
- Pattern: sum+sum+sum reduction (layer-norm backward + patch layout restore)
- Inductor kernels: 4 unique Triton kernels
- Ops: reshape, mul, sum([2],keepdim), sum([0,1]), sub, div, add, permute, clone
- Shapes: [131072,144] -> [512,256,144] row reductions, [144] column reductions, output [128,144,32,32] via reshape+permute+clone
- The gap (1.23x) comes from: (1) materializing the [512,256,144] intermediate before the layout transform, (2) separate clone+permute kernel for the patch-fold output, (3) redundant reads for column reductions
- combo_kernels=True was not tested but the pattern requires cross-axis fusion unsupported by existing templates
- No existing Inductor config flag addresses this gap
