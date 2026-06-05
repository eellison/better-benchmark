# pointwise_2cb3b25427b0 — oracle_complex_rope_mul

## Status: CORRECTNESS FAIL (oracle fails check)

## Benchmark Results
- Not benchmarked (correctness check failed)

## Classification: NEW_PATTERN (complex dtype support gap)

## Root Cause

The oracle implements complex64 broadcast multiply (RoPE factor applied to two complex64 tensors) using a Triton kernel that decomposes complex multiplication into real/imag arithmetic. However, the oracle **fails correctness** — the check reports exact complex64 comparison failure for both outputs.

The repro performs:
```python
view_default = view(slice_1, [1, 32, 1, 32])  # complex64[32,32] -> [1,32,1,32]
mul_tensor = view_as_complex_14 * view_default   # complex64[32,32,8,32] * [1,32,1,32]
mul_tensor_1 = view_as_complex_15 * view_default # complex64[32,32,8,32] * [1,32,1,32]
```

This is a LLaMA-style complex RoPE broadcast multiply. The oracle's indexing formula for the broadcast factor (`d1 = (offsets // 256) % 32; d3 = offsets % 32; factor_offsets = (d1 * 32 + d3) * 2`) appears to have a mismatch with the actual broadcast semantics of `[1, 32, 1, 32]` viewed over `[32, 32, 8, 32]`.

## Inductor Status

Inductor currently warns that it does not support code generation for complex operators and may use fallback paths. This is a known limitation — complex dtype support in Inductor is incomplete.

## Design Doc

### Problem
Inductor lacks native complex-aware pointwise codegen. Complex operations like `complex64 * complex64` with broadcasting should be decomposed into real/imag loads, (a+bi)*(c+di) = (ac-bd) + (ad+bc)i arithmetic, and real/imag stores in a single fused kernel.

### Proposed Fix (NEW_PATTERN)
Add a native complex broadcast-multiply lowering in `torch/_inductor/lowering.py` or `fx_passes/` that:
1. Decomposes complex mul into view_as_real, arithmetic on real/imag pairs, view_as_complex
2. Allows fusion of sibling complex outputs (both mul_tensor and mul_tensor_1 share the same broadcast factor)

### Priority: MEDIUM
Complex RoPE multiply is common in LLaMA-family models. However, since the oracle itself has a correctness bug, the actual performance gap cannot be measured until the oracle is fixed.

### Files to Modify
- `/tmp/pytorch-work/torch/_inductor/lowering.py`: Add complex mul decomposition
- `/tmp/pytorch-work/torch/_inductor/fx_passes/post_grad.py`: Recognize sibling complex muls with shared factor
