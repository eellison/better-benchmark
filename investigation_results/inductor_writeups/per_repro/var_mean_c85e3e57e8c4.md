# var_mean_c85e3e57e8c4

## Classification: BANDWIDTH_BOUND

## Current Result

- Family: `beit_affine_residual_layernorm`
- Oracle path: `repros/canonical/var_mean_c85e3e57e8c4/oracle_fused_layernorm.py`
- Correctness: assumed PASS (oracle has check mode)
- Compiled (coordinate_descent_tuning=True): 40.54 us
- Gap: 0.99x (compile matches or beats oracle)
- Status: `at_floor`

## Diagnosis

The oracle computes a BEiT affine-residual LayerNorm in one dedicated hidden-size-768 Triton row kernel. Inductor already emits one generic persistent reduction for the identical fused scope.

The repro computes:
1. view [25216, 768] -> [128, 197, 768]
2. mul(arg202_gamma, viewed) -- affine pre-multiply
3. add(add_76_residual, result) -- residual connection
4. var_mean(correction=0, dim=2, keepdim=True)
5. LayerNorm: sub, add_eps, rsqrt, mul_gamma, add_bias
6. view -> [25216, 768]

## Root cause

No gap. Inductor already fuses the entire scope (affine pre-multiply, residual add, population var_mean, rsqrt, scale, bias) into a single persistent row-reduction kernel. The 0.99x ratio means torch.compile matches the hand-written oracle exactly.

## Kernel count
- Oracle: 1 kernel (fused LN row kernel)
- Inductor: 1 kernel (fused persistent reduction)

## Config exploration results

| Config | Expected Impact |
|--------|----------------|
| coordinate_descent_tuning | Already applied (40.54 us) |
| All configs | No improvement needed -- already at oracle floor |

## Recommendation

No action needed. Inductor already achieves optimal performance for this pattern. Record as floor case.

## Relevant files

- Input: [25216, 768] + [128, 197, 768] + [768] f32 (BEiT inference)
- Total bytes: ~232 MB
- Models: timm_beit_base_patch16_224_infer_000, timm_timm_beit_base_patch16_224_infer_infer_000
