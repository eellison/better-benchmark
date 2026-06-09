# pointwise_6a479e2e07d2 — Stochastic Complex View

## Summary
- **Ratio**: 1.387x (oracle: 53.18us, compile: 73.76us)
- **Classification**: SCHEDULER_FUSION
- **Oracle kernels**: 1 (fused random + mask + complex store)
- **Inductor kernels**: 1

## Root Cause

Both produce a single kernel, but with different codegen quality. The scope computes:
1. Generate random values via `inductor_random` (Philox RNG)
2. Compare `rand > 1e-30` (stochastic dropout mask)
3. Multiply input by mask (element-wise)
4. View `[16384, 768]` as `[32, 512, 768]`
5. Convert real f32 to complex64

The oracle's kernel (`_complex_view_tlrand_kernel`) writes the complex output directly as interleaved real/imag stores: `out[offset*2] = real; out[offset*2+1] = 0.0`. This avoids materializing an intermediate f32 buffer.

Inductor's kernel (`triton_poi_fused_gt_inductor_lookup_seed_inductor_random_mul_view_0`) writes the f32 result to an intermediate buffer (`buf3`), then a separate `torch.view_as_complex` or copy operation converts it to complex64. The Runner code shows:
- `buf3` is allocated as `f32[32, 512, 768]`
- `buf2` is allocated as `complex64[32, 512, 768]`
- The conversion from buf3 to buf2 happens after the Triton kernel

The 38.7% gap comes from:
1. Extra memory traffic: writing 48MB (f32 intermediate) + reading 48MB + writing 96MB (complex64) vs. writing 96MB directly
2. The extra copy/conversion kernel launch overhead

## Config Exploration

Standard configs active. This is fundamentally a codegen gap -- Inductor's complex number lowering does not fuse the real-to-complex conversion into the producing kernel's stores.

## Design Doc

**Why it cannot be fixed today**: Inductor's complex number support lowers `view_as_complex` as a separate buffer operation rather than fusing it into the producer kernel's store pattern. The codegen does not know how to emit interleaved real/imag stores from a pointwise kernel that produces real-valued results destined for a complex output.

**What enhancement is needed**: A codegen/lowering enhancement in `torch/_inductor/codegen/triton.py` and `torch/_inductor/lowering.py` that:
1. Detects when a pointwise kernel's output feeds directly into `view_as_complex`
2. Rewrites the store to use interleaved addressing: `store(out_ptr + offset*2, real_value)` and `store(out_ptr + offset*2 + 1, 0.0)`
3. Eliminates the intermediate f32 buffer entirely

**Affected files**:
- `torch/_inductor/lowering.py` (complex conversion fusion)
- `torch/_inductor/codegen/triton.py` (interleaved store emission)
- `torch/_inductor/ir.py` (complex layout handling)

**Affected repro count**: Any model using stochastic ops with complex outputs (speech/signal processing models), estimated 2-3 repros.
