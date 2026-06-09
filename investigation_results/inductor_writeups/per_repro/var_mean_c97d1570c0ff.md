# var_mean_c97d1570c0ff


## Measured Timings
- Oracle: 38.34 us
- Compile (CDT): 36.80 us
- Ratio: 0.96x

## Classification: ALGEBRAIC_ELIMINATION

## Current Result

- Family: `add_layernorm_saved_scale`
- Oracle path: `repros/canonical/var_mean_c97d1570c0ff/oracle_add_layernorm_saved_scale.py`
- Correctness: assumed PASS (oracle has check mode)
- Compiled (coordinate_descent_tuning=True): 40.93 us
- Gap: 1.15x
- Status: `real_gap` (marginal)

## Diagnosis

The oracle computes the complete add-plus-affine LayerNorm scope including the sibling `rsqrt / 4096` output in one shape-specialized Triton row kernel. Inductor currently emits one fused generic Welford reduction kernel for the same scope, but the 1.15x gap suggests suboptimal code for the saved inverse-scale side output.

The repro computes:
1. view [4096, 4096] -> [8, 512, 4096]
2. add(add_103_residual, viewed) -> [8, 512, 4096]
3. var_mean(correction=0, dim=2, keepdim=True)
4. rsqrt(var + 1e-12) -> saved for output
5. LayerNorm: sub_mean, mul_rsqrt, mul_gamma, add_bias
6. view -> [4096, 4096] (return 1)
7. div(rsqrt, 4096) -> [8, 512, 1] (return 2: saved scale)

## Root cause

Inductor's correction=0 var_mean lowering uses the general Welford state machine rather than selecting a fixed-width LayerNorm algebra that reuses the resident row tile for the affine epilogue and side output store. The `div(rsqrt, 4096)` side output is trivially computed from the already-computed rsqrt but may require an extra store pass in Inductor's generic template.

The gap is marginal (1.15x, ~5 us absolute) on a kernel that already fuses everything into 1 kernel. The oracle's advantage is likely from tuned tile sizes and avoiding Welford's two-accumulator state for this specific (known-N, correction=0) case.

## Kernel count
- Oracle: 1 kernel (specialized row LN with saved scale)
- Inductor: 1 kernel (generic Welford persistent reduction)

## Config exploration results

| Config | Expected Impact |
|--------|----------------|
| coordinate_descent_tuning | Already applied (40.93 us) |
| combo_kernels | Not applicable (single kernel) |
| multi_kernel=2/3 | Unlikely to help (already 1 kernel) |

## Recommendation

Add a guarded correction=0 LayerNorm lowering that replaces generic Welford bookkeeping with direct mean and centered-variance reductions for static hidden dimensions. This saves register pressure by using single-accumulator mean, then single-accumulator variance (two-pass but same tile, all in registers for hidden<=4096).

## Relevant files

- `/tmp/pytorch-work/torch/_inductor/codegen/triton.py` (Welford vs direct mean+var for correction=0)
- `/tmp/pytorch-work/torch/_inductor/lowering.py` (var_mean lowering)
- Input: [4096, 4096] + [8, 512, 4096] f32 (Albert/GPT-Neo/DeiT/nanoGPT training)
- Total bytes: ~201 MB
- Models: 12 models including hf_AlbertForMaskedLM, hf_GPTNeoForCausalLM, timm_deit, torchbench_nanogpt
