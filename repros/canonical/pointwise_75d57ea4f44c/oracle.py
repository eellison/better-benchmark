"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete XLNet bf16 attention-bias pointwise layout scope in one storage-linear Triton kernel, preserving the two exact `bf16 + f32 -> f32 -> bf16` add/cast boundaries and returning both `[256, 512, 64]` outputs plus their permuted metadata aliases from the two backing buffers, whereas Inductor lowers the captured unsqueeze/view/permute/add/cast/view/permuted-alias graph through generic pointwise output planning; Inductor cannot do this today because its pointwise scheduler does not recognize sibling broadcast-add outputs with shared storage-linear indexing and alias-only returned permutes as one guarded template; the fix is NEW_PATTERN: add a B200-tuned dual-output broadcast-add pointwise layout specialization that coalesces the view algebra, emits one producer kernel, and returns the alias views without extra device work."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _dual_broadcast_add_bf16_kernel(
    x_ptr,
    bias0_ptr,
    bias1_ptr,
    out0_ptr,
    out1_ptr,
    n_elements: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    bias_offsets = offsets % 1024

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    bias0 = tl.load(bias0_ptr + bias_offsets, mask=mask, other=0.0)
    bias1 = tl.load(bias1_ptr + bias_offsets, mask=mask, other=0.0)

    tl.store(out0_ptr + offsets, x + bias0, mask=mask)
    tl.store(out1_ptr + offsets, x + bias1, mask=mask)


# 798a086d: (T([8192,1024], bf16), T([16,64], f32), T([16,64], f32), S([512,16,1,16,64]), S([512,16,16,64]), S([256,512,64]), S([256,512,64]))
@oracle_impl(hardware="B200", point="798a086d", BLOCK=1024, num_warps=4)
def oracle_forward(inputs, *, BLOCK, num_warps):
    arg0_1, arg1_1, arg2_1, _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3 = inputs
    del _shape_param_0, _shape_param_1

    view_2 = torch.empty_strided(
        tuple(_shape_param_2),
        (64, 16384, 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    view_3 = torch.empty_strided(
        tuple(_shape_param_3),
        (64, 16384, 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )

    n_elements = arg0_1.numel()
    grid = (triton.cdiv(n_elements, BLOCK),)
    _dual_broadcast_add_bf16_kernel[grid](
        arg0_1,
        arg1_1,
        arg2_1,
        view_2,
        view_3,
        n_elements,
        BLOCK=BLOCK,
        num_warps=num_warps,
    )

    return view_2, view_3, view_3.permute(0, 2, 1), view_2.permute(0, 2, 1)
