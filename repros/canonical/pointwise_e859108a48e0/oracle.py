"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the full slice -> permute -> contiguous clone -> view -> view MobileViT layout materialization by directly copying the retained bf16 channels from contiguous [512, 4, 16, 64] input storage into the required fresh contiguous [8192, 240] output with a shape-specialized affine Triton kernel, whereas Inductor lowers the decomposed chain through an equivalent generic layout-copy pointwise kernel with scalar affine decoding; Inductor cannot do more than tie this today because its scheduler/codegen does not recognize this static padded-channel permute-contiguous-flatten idiom as a dedicated layout-copy pattern; the fix is NEW_PATTERN: add a specialized lowering for padded NHSD-to-NS/HD materializations that writes the final view storage directly."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _slice_permute_clone_view_linear_kernel(
    x,
    out,
    XBLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)
    channel = offsets % 60
    head = (offsets // 60) % 4
    token = (offsets // 240) % 16
    batch = offsets // 3840
    values = tl.load(x + channel + 64 * token + 1024 * head + 4096 * batch).to(tl.float32)
    tl.store(out + offsets, values)


@oracle_impl(hardware="B200", point="d449276a", XBLOCK=1024, num_warps=4, num_stages=3)
def oracle_forward(inputs, *, XBLOCK, num_warps, num_stages):
    x, _shape_param_0, _shape_param_1 = inputs
    out = torch.empty((8192, 240), device=x.device, dtype=x.dtype)
    grid = (triton.cdiv(1966080, XBLOCK),)
    _slice_permute_clone_view_linear_kernel[grid](
        x,
        out,
        XBLOCK=XBLOCK,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out
