# pointwise_d66b13ff6ca0

## Compile: 5.54us, Oracle: 5.7us, Gap: 0.97x (at floor, compile matches oracle)

**Previous**: Compile: 8.13us, Oracle: 6.05us, Gap: 1.34x

## Diagnosis: CLOSED by inline_recomputable_producers (f58d2545cd2)

## Root cause

The oracle fuses per-channel batch normalization (sub, mul by inv_std, affine scale+bias), ReLU, and 2x2 stride-2 max_pool_with_offsets into a single output-tiled Triton kernel that never materializes the intermediate [1,128,16,64] activated tensor. Instead it loads the four stencil positions from the original convolution output, computes norm+relu inline for each, and emits both the max value and the int8 offset in one pass.

Inductor generates 2 kernels:
1. `triton_poi_fused_add_mul_reciprocal_relu_sqrt_sub_unsqueeze_0`: Materializes the full [1,128,16,64] normalized+ReLU tensor into buf0.
2. `triton_poi_fused__low_memory_max_pool_with_offsets_...1`: Reads buf0 at 4 stencil positions per output element, computes max + offset.

The fusion does not happen because the `_low_memory_max_pool_with_offsets` lowering in `lowering.py:5625` creates a reduction node that accesses its input at shifted 2D stencil positions (`in_ptr0 + (2*x0 + 128*x1)`, `in_ptr0 + (1 + 2*x0 + 128*x1)`, etc.). The scheduler's fusion logic does not inline a pointwise producer into a stencil-consuming reduction when the consumer's index expressions are non-trivial affine functions of the output coordinates (they involve stride-2 offsets into a larger spatial grid).

## Kernel count

- Inductor: 2 kernels
- Oracle: 1 kernel (single fused norm+relu+maxpool)

## Config exploration

| Config | Time (us) |
|--------|--------:|
| coordinate_descent_tuning=True | 8.13 |
| combo_kernels=True, multi_kernel=3 | 18.24 |
| Default compile | ~8.1 |

Config exploration does not help because the gap is structural (fusion failure), not a tuning issue.

## Fix path

The scheduler enhancement needed: allow pointwise producers to be fused into stencil-pattern consumers (like max_pool) when the producer is element-wise along the same spatial dimensions. Specifically:

1. **File**: `/tmp/pytorch-work/torch/_inductor/scheduler.py` (can_fuse / score_fusion)
2. **Enhancement**: When a Pointwise node feeds a single Reduction/Pointwise consumer that accesses it at known stencil offsets (e.g., 2x2 pool with stride 2), allow fusion by inlining the producer computation at each access point.
3. **Constraint**: The producer must be a simple elementwise/broadcast op (no reduction, no multi-consumer). The stencil window must be small (e.g., <= 9 elements) to avoid register pressure blowup.
4. **Memory savings**: Eliminates the [1,128,16,64] = 512KB intermediate buffer round-trip (write 512KB + read 128KB for the 4 pool positions = 640KB saved).

Alternative approach: In `lowering.py:_low_memory_max_pool_with_offsets`, instead of creating a reduction that reads from `x` at stencil positions, inline the producer's `inner_fn` directly into the pool kernel's loop body (similar to how broadcast-dominated producers are handled).

## Models affected

- torchbench_doctr_reco_predictor_infer_000
- torchbench_pytorch_unet_infer_000

## Fix: inline_recomputable_producers extension

The extension to `inline_recomputable_producers` that handles cheap non-broadcast
producers closed this gap. Compile 5.54us now matches the oracle 5.7us (0.97x ratio).
The stencil-consumer fusion issue is now moot since the producer is inlined directly.
Re-measured 2026-06-08.

## Status: closed
