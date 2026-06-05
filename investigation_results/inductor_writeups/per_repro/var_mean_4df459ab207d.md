# var_mean_4df459ab207d

## Classification: NCHW_CHANNEL_LAYERNORM_TILING

## Current Result

- Family: `channel_layernorm`
- Oracle path: `repros/canonical/var_mean_4df459ab207d/oracle_channel_layernorm.py`
- Correctness: PASS
- Oracle: `31.9 us`
- `torch.compile coordinate_descent_tuning=True`: `33.92 us`
- Ratio: 1.063
- Best config: `default (cd=True)`: `33.92 us`
- Status: `real_gap` (marginal)

## Diagnosis

The oracle computes the complete ConvNeXtV2 channel-LayerNorm scope in one Triton kernel on shape [128, 320, 14, 14]. It includes the NCHW input add, per-[N,H,W] population var_mean over 320 channels, eps=1e-6 affine epilogue, direct final NCHW output stores, and the live rsqrt/320 side output. The oracle tiles 16 adjacent spatial rows together so physically strided channel loads become coalesced row groups.

Inductor treats the permute-to-NHWC LayerNorm as a generic row reduction over a non-inner channel dimension, generating output in the same NCHW layout but without the spatial-row grouped tiling optimization.

## Config exploration results

| Config | Time (us) |
|--------|-----------|
| Default (cd=True) | 33.92 |
| combo+mk=2 | 35.94 |
| combo+mk=3 | 35.92 |
| Oracle | 31.90 |

No config closes the gap. Multi-kernel configs make it slightly worse.

## Root cause

The 6.3% gap comes from the oracle's spatial-row grouped tiling strategy for NCHW channel-LayerNorm. The oracle groups 16 adjacent (N,H,W) positions and loads C=320 channels for each group, achieving coalesced memory access. Inductor's generic norm template does not have this NCHW-specific tiling optimization.

## Kernel count
- Oracle: 1 kernel (channel-LN with spatial row grouping)
- Inductor: 1 kernel (generic row reduction)

## Recommendation

The gap is marginal (6.3%) and within the range where autotuning variance could explain it. However, a dedicated NCHW channel-LayerNorm tiling rule that groups neighboring spatial positions for coalesced channel access would close this gap for the ConvNeXtV2 family.

File references: `torch/_inductor/codegen/triton.py` (tiling heuristics), `torch/_inductor/lowering.py` (layer_norm lowering).
