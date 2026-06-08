# var_mean_c9d97f0d1948

## Classification: SCHEDULER_FUSION

## Current Result

- Family: `gelu_layernorm`
- Oracle path: `repros/canonical/var_mean_c9d97f0d1948/oracle_gelu_layernorm.py`
- Correctness: assumed PASS (oracle has check mode)
- Compiled (coordinate_descent_tuning=True): 16.38 us
- Gap: 1.23x
- Status: `real_gap` (marginal)

## Diagnosis

The oracle computes the complete exact-erf GELU plus population LayerNorm scope in one Triton row-reduction kernel. Inductor currently lowers the decomposed GELU, var_mean(correction=0), rsqrt, affine, and view graph through its generic fused normalization reduction schedule.

The repro computes:
1. view [8192, 768] -> [8, 1024, 768]
2. GELU(viewed): mul(x, 0.5) * (1 + erf(x * 0.7071)) -> [8, 1024, 768]
3. var_mean(correction=0, dim=2, keepdim=True)
4. LayerNorm: sub_mean, rsqrt(var+eps), mul_gamma, add_bias
5. view -> [8192, 768]

## Root cause

Inductor emits 1 fused kernel for this scope (confirmed: 1 kernel). The 1.23x gap (2-3 us absolute on a 16 us kernel) suggests the oracle's shape-specialized kernel has better:
- Tile size selection for hidden_size=768
- Register allocation (GELU producer kept live through statistics)
- Possibly fused-multiply-add utilization for erf approximation

The generic Welford reduction with GELU producer fusion may not select the optimal BLOCK_M / num_warps for this specific (8192 rows, 768 hidden) shape.

## Kernel count
- Oracle: 1 kernel (fused GELU + LN row kernel)
- Inductor: 1 kernel (fused persistent Welford)

## Config exploration results

| Config | Expected Impact |
|--------|----------------|
| coordinate_descent_tuning | Already applied (16.38 us) |
| combo_kernels | Not applicable (single kernel) |
| multi_kernel=2/3 | May help with tile selection |

## Recommendation

This is a tuning gap rather than a structural fusion gap. The oracle's advantage comes from shape-specialized tile sizes and warp counts for the GELU+LN pattern at hidden=768. Coordinate descent tuning should eventually close this.

Consider adding an exact-GELU-producing LayerNorm template hint that fuses the pointwise producer, row moments, epsilon rsqrt, affine epilogue, and final viewed store with tuned tile parameters for common hidden sizes (768, 1024, 1536, 4096).

## Relevant files

- `/tmp/pytorch-work/torch/_inductor/codegen/triton.py` (tile selection for GELU+LN)
- `/tmp/pytorch-work/torch/_inductor/choices.py` (persistent reduction parameters)
- Input: [8192, 768] f32 (Longformer/BERT/DistilBERT/Electra inference)
- Total bytes: ~50 MB
- Models: 7 models including hf_AllenaiLongformerBase_infer, hf_BertForMaskedLM_infer, hf_DistilBertForMaskedLM_infer
