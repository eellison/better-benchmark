# pointwise_4ac748523d2a

## Classification: STENCIL_PRODUCER_INLINE

## Current Result

- Family: `relu_maxpool_flatten_mask`
- Oracle path: `repros/canonical/pointwise_4ac748523d2a/oracle_relu_maxpool_flatten_mask.py`
- Correctness: PASS (3 outputs: int8 offsets, float32 flattened pool, bool mask)
- Oracle: `24.38 us`
- `torch.compile coordinate_descent_tuning=True`: `27.97 us`
- Ratio: 1.147
- Best config: `default (cd=True)`: `27.97 us`
- Status: `real_gap`

## Diagnosis

The oracle computes the complete ReLU -> 2x2 stride-2 maxpool-with-offsets -> flatten plus the full input-shaped ReLU<=0 mask in one Triton kernel, writing int8 pool offsets, flattened pooled values, and boolean mask directly. Shape: [128, 512, 14, 14] input -> [128, 512, 7, 7] pooled output -> [128, 25088] flattened.

Inductor generates a single fused kernel that handles this but with less optimal code (the stencil/maxpool with multiple outputs including the flat view and mask side-output is handled sub-optimally in one kernel).

## Config exploration results

| Config | Time (us) |
|--------|-----------|
| Default (cd=True) | 27.97 |
| combo+mk=2 | 29.73 |
| combo+mk=3 | 29.73 |
| Oracle | 24.38 |

No config closes the 1.15x gap. Multi-kernel configs make it slightly worse.

## Root cause

Although Inductor fuses into 1 kernel (same kernel count as oracle), the oracle's hand-tuned stencil code is more efficient. The oracle uses a specialized maxpool tiling that processes the 2x2 window inline with the ReLU, writes int8 offsets and boolean mask simultaneously, and sinks the flatten into the store plan. Inductor's generic codegen for the multi-output maxpool stencil pattern has suboptimal tiling for this particular shape combination (14x14 -> 7x7 with 3 heterogeneous output types).

## Kernel count
- Oracle: 1 kernel
- Inductor: 1 kernel (fused pointwise with stencil)

## Recommendation

The gap is in codegen quality for multi-output maxpool stencils. The scheduler correctly fuses everything into one kernel, but the generated code for the heterogeneous-output (int8 + float32 + bool) stencil pattern does not tile as efficiently as the hand-written oracle. Improving this requires better tiling heuristics in `torch/_inductor/codegen/triton.py` for maxpool-with-indices patterns that output multiple dtypes.

File references: `torch/_inductor/codegen/triton.py` (stencil codegen tiling), `torch/_inductor/lowering.py` (max_pool2d_with_indices lowering).
