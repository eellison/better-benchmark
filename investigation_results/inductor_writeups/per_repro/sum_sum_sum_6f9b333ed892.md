# sum_sum_sum_6f9b333ed892

## Summary

- Model: NFNet-F0 (timm_dm_nfnet_f0_train)
- Oracle: `oracle_nfnet_gated_gelu_pool_multi_output.py`
- Classification: ALGEBRAIC_ELIMINATION
- Ratio: 1.941x (oracle 301.9us, compile 585.8us)

## Root Cause

The repro computes the NFNet backward fragment including avg_pool2d_backward expansion, broadcast sigmoid gate, exact GELU-derivative pointwise chain, scalar sum, sigmoid-gradient channel reduction, and sibling `mul_tensor_12` channel reduction over [128, 256, 48, 48] f32 activations.

The oracle streams all operations through one Triton producer kernel with three per-(N,C) accumulators (scalar accumulator, gate gradient accumulator, main gradient accumulator), then finalizes with two small reduction kernels. This avoids materializing large intermediates.

Inductor currently:
1. Materializes `mul_tensor_12` as a separate tensor
2. Separately materializes the spatial reduction feeding `sigmoid(arg188_1) * (1 - sigmoid(arg188_1))`
3. Launches separate reductions for the scalar sum and sibling channel sum

The 1.94x gap comes from:
- Materializing large [128, 256, 48, 48] intermediates that the oracle never writes
- Multiple kernel launches (each with its own activation reads) vs oracle's single-pass approach
- The algebraic structure `sum([2,3]) -> sigmoid-derivative multiply -> sum([0,2,3])` is a dependent reduction chain that Inductor does not flatten into one multi-accumulator kernel

## Config Exploration

| Config | Compile Time (us) |
|--------|-------------------|
| baseline (CDT) | 585.8 |
| multi_kernel=2 | CRASH (ND reduction assert) |
| multi_kernel=3 | CRASH (ND reduction assert) |

multi_kernel=2/3 trigger an Inductor bug: `AssertionError: Expected ND reduction size ({'r0_': 65536}) to have 131072 elements` in `_get_nd_reduction_numels`. This is a separate bug in persistent_reduction config generation for large spatial reductions.

## Fix Assessment

**Algebraic elimination** -- Teach Inductor to reassociate the dependent reduction pattern and emit one multi-accumulator channel-reduction template over the shared fused producer.

### What's needed:
1. Recognize that the GELU-derivative pointwise chain + avg_pool2d_backward can be computed on-the-fly during the reduction (no need to materialize)
2. Fuse the scalar sum, gate gradient channel sum, and main gradient channel sum into one multi-accumulator spatial reduction
3. Handle the dependent `sum([2,3]) -> sigmoid_derivative -> sum([0,2,3])` pattern by carrying the per-channel partial sum as an accumulator

### Difficulty: High
This requires both algebraic simplification (recognizing the dependent reduction chain) and multi-output reduction scheduling (three accumulators with different pre-processing). The oracle demonstrates this is achievable but requires a new pattern-matching pass.

## Additional Bug Found

multi_kernel=2/3 crashes with ND reduction assertion error for shapes involving large spatial dims. File separately.

## Status: open_gap

## Affected Repros

This NFNet gated-GELU-pool backward pattern is specific to NFNet variants. The ALGEBRAIC_ELIMINATION classification may apply to similar gated-activation backward patterns in other architectures.
