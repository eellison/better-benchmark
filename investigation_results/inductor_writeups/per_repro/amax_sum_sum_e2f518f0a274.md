# amax_sum_sum_e2f518f0a274

## Compile: 157.3us, Oracle: 92.0us, Gap: 1.71x
## With multi_kernel=3: 129.9us, Gap: 1.41x

## Diagnosis: ONLINE_CROSS_ENTROPY (codegen template overhead in looped online softmax)

## Root cause

Inductor correctly identifies and fuses the cross-entropy pattern into:
- Kernel 0: Online softmax (streaming amax + sum_exp over [2048, 50257])
- Kernel 1: Epilogue (gather target_logit from logits, compute masked loss, reduce to scalar)

The oracle achieves 92us with 2 kernels:
- Kernel 0: Per-row online CE (1 row/CTA, block_n=4096, num_warps=8), gathers target inline during streaming
- Kernel 1: Final mean reduction from loss[2048] + valid[2048]

The 1.71x gap decomposes as:
1. **Kernel 0 overhead (main bottleneck)**: Inductor's 2D kernel template (XBLOCK=1, R0_BLOCK=4096, num_warps=16) is ~1.7x slower than oracle's 1D kernel (block_n=4096, num_warps=8) for the SAME computation. Root cause: broadcasting operations, 2D masking, and `triton_helpers.max2` with explicit dim argument add overhead vs oracle's clean 1D scalar accumulators.
2. **Epilogue re-reads logits**: Kernel 1 gathers target_logit via random access into the 400MB logits tensor (2048 scattered reads). The oracle computes this during the streaming pass.

With `multi_kernel=3` (force looped reduction), the gap reduces to 1.41x because the looped variant autotuner picks better configs. The remaining 1.41x gap is purely the 2D template overhead.

## Investigation: CrossEntropyOnlineReduction attempt

There is an existing `CrossEntropyOnlineReduction` class in ir.py (line 2613) with full codegen support for `online_softmax_cross_entropy` in triton.py. This was designed to fuse the target gather into the reduction kernel. However:
- The lowering at line 9175 was NOT using it (it used OnlineSoftmaxReduction + separate Pointwise)
- When wired up, the fused kernel correctly eliminates the logits re-read in kernel 1
- BUT the in-loop target capture code (broadcast + max2 + where/_loaded flag) adds per-iteration overhead
- The persistent codegen path has a shape mismatch bug (target_idx is [XBLOCK, R0_BLOCK] but store expects [XBLOCK, 1])
- Net result: CrossEntropyOnlineReduction gives 146us (vs 157us original) but not as good as multi_kernel=3

The real fix requires **hoisting loop-invariant loads** out of the reduction loop. The target_logit load depends only on row index (outer dim), not column index (reduction dim), so it should be emitted before the loop. This requires the codegen to:
1. Analyze which parts of inner_fn output are reduction-invariant
2. Emit those parts in the `self.body` section (pre-loop) rather than `self.compute` (in-loop)

## Config exploration

| Config | Time (us) | Ratio |
|--------|-----------|-------|
| Default (coord_descent + combo) | 157.3 | 1.71x |
| multi_kernel=3 (looped) | 129.9 | 1.41x |
| multi_kernel=2 (persistent) | 396.4 | 4.31x |
| CrossEntropyOnlineReduction (default) | 146.4 | 1.59x |

## Fix status: design_todo

The fix requires two changes:
1. **Codegen: loop-invariant hoisting** for `online_softmax_cross_entropy` — emit target_logit load before the reduction loop
2. **Codegen: persistent path fix** — reduce [XBLOCK, R0_BLOCK] target tensor to [XBLOCK, 1] before store

## Affected repros

- amax_sum_sum_e2f518f0a274 (1.71x, GPT-2 50257 vocab) - main affected
- amax_sum_sum_1bad0f362efd (1.12x, Qwen 262144 vocab) - modest gap
- amax_sum_sum_b9ac8700504c (1.04x, Blenderbot 8008 vocab) - at parity
- amax_sum_sum_d5ddd6e16182 (1.00x, Electra 30522 vocab) - at parity

## File references

- `/tmp/pytorch-work/torch/_inductor/lowering.py:9175` - cross_entropy_loss_online lowering
- `/tmp/pytorch-work/torch/_inductor/ir.py:2613` - CrossEntropyOnlineReduction class (unused)
- `/tmp/pytorch-work/torch/_inductor/codegen/triton.py:5047` - online_softmax_cross_entropy codegen (looped path)
- `/tmp/pytorch-work/torch/_inductor/codegen/triton.py:4789` - persistent path (buggy store shape)
- `/tmp/pytorch-work/torch/_inductor/fx_passes/post_grad.py:295` - _cross_entropy_gather_fusion_pass
