"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the full `arg0_1 > 0`, metadata-only flatten view, and scalar `aten.any` scope in one shape-specialized Triton program while also materializing the visible boolean predicate output, whereas Inductor lowers this small flattened any reduction through the generic reduction scheduler around an otherwise inlineable comparison; Inductor cannot do this today because its scheduler lacks a compact zero-dimensional any/all specialization that preserves the producer side output and scalar reduction from the same predicate without generic reduction-loop scaffolding; the fix is SCHEDULER_FUSION: teach the reduction scheduler to emit a single compact predicate-store plus scalar-any reduction for this pattern."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _gt_zero_mask_any_kernel(
    x_ptr,
    mask_ptr,
    any_ptr,
    BLOCK: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK)
    values = tl.load(x_ptr + offsets)
    keep = values > 0.0
    tl.store(mask_ptr + offsets, keep)
    any_value = tl.max(keep.to(tl.int32), axis=0) != 0
    tl.store(any_ptr, any_value)

# a7e138d5: (T([8,1024], bf16), S([8192]))
@oracle_impl(hardware="B200", point="a7e138d5", BLOCK=8192, num_warps=8, num_stages=1)
# 5002b631: (T([8,1024], f32), S([8192]))
@oracle_impl(hardware="B200", point="5002b631", BLOCK=8192, num_warps=8, num_stages=1)
def oracle_forward(inputs, *, BLOCK: int, num_warps: int, num_stages: int):
    arg0_1, _shape_param_0 = inputs
    gt = torch.empty_strided(
        arg0_1.shape,
        arg0_1.stride(),
        device=arg0_1.device,
        dtype=torch.bool,
    )
    any_1 = torch.empty((), device=arg0_1.device, dtype=torch.bool)
    _gt_zero_mask_any_kernel[(1,)](
        arg0_1,
        gt,
        any_1,
        BLOCK=BLOCK,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return gt, any_1
