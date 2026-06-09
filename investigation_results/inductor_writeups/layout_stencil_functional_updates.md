# Inductor Writeup: Layout / Stencil Functional Updates

## Status
- Queue id: `layout_stencil_functional_updates`
- Priority: P1
- Oracle: `repros/canonical/pointwise_70c0751eb408/oracle_stencil_canonicalized.py`

## Targets
`pointwise_70c0751eb408`, `pointwise_531d72f1b34a`, `pointwise_a2382a85ee99`.

## Plan
Canonicalize `slice/select -> copy -> slice_scatter/select_scatter` chains into explicit offset-load/store stencil domains before scheduler fusion. Keep boundaries separate. For pure f64 expression cases, improve CSE/register-pressure-aware splitting rather than adding launches blindly.

## Hooks
FX/post-grad canonicalization for functional updates; scheduler cost model to penalize full-tensor reconstruction; codegen path for explicit-offset interior kernels.

## Validation
Use forced coordinate-descent configs from `investigation_results/inductor_target_configs.csv` and oracle plan script for `pointwise_70c0751eb408`.
