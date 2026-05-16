# Prior Investigation Log (from separate session)

Cross-referenced with our FINDINGS.md. Key items that are PR-ready or have
verified patches in /data/users/eellison/pytorch:

## PR-Ready Patches (verified on H100)

1. **Small static topk decomposition** — k<=16, dim<=64, decompose instead of ATen fallback
2. **Blackwell low-register RBLOCK=2048** — same as our Issue 2/persistent threshold
3. **combo_kernels_auto_static_reductions** — narrow auto-enable for static short sum reductions with rnumel<=2048
4. **bf16 SiLU*other → x*sigmoid(x)*other** — uses tl.sigmoid instead of libdevice.exp decomposition
5. **H100 large low-register inner-reduction config** — XBLOCK=8, R0_BLOCK=1024, num_warps=2
6. **ND tiling for collapsed static pointwise** — fixes the repeat+reshape flat-copy with div/mod

## Key Insights Not In Our Findings

- sorted=False topk has exactness concerns (CUDA radix ordering must be preserved)
- Split-reduction finalizer is NOT ordinary producer/consumer fusion (Issue 45)
- Many stored B200 multi-kernel symptoms are STALE on current PyTorch (already fixed upstream)
- combo_kernels=True doesn't help BERT backward (Issue 36) — needs reduction reuse/finalizer
- CUDA graph replay of many small sums does NOT help (inter-kernel barriers)

## Repro Quality Issues They Also Hit
- Same bool/device/inf/randint issues we fixed
- prims.fma unavailable on some checkouts
- Symbolic size references (s77) in make_inputs
