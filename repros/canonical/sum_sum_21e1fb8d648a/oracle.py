"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete bf16 ALBERT layer-norm-backward dual-row-sum scope in one row-resident Triton kernel, including the bf16-to-fp32 matmul view, returned fp32 add side output, the two hidden-dimension fp32 sums, returned fp32 dependent epilogue, explicit bf16 cast, final `[4096,4096]` view, and transpose alias, whereas Inductor lowers the visible add producer, dependent row reductions, fp32 side output, and bf16 alias outputs through its generic reduction/output scheduling; Inductor cannot do this today because the scheduler does not specialize this fixed-hidden multi-output row-reduction pattern while preserving all observable side stores and dtype boundaries from one row tile; the fix is SCHEDULER_FUSION: teach reduction scheduling to keep the visible producer, sibling row sums, dependent epilogue, bf16 materialization, and alias-only transpose return in one guarded ALBERT LayerNorm-backward schedule."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _albert_ln_backward_dual_sum_kernel(
    mm_bf16_ptr,
    residual_ptr,
    weight_ptr,
    normed_ptr,
    row_scale_ptr,
    add_out_ptr,
    epilogue_out_ptr,
    bf16_out_ptr,
    HIDDEN: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < HIDDEN
    offsets = row * HIDDEN + cols

    mm = tl.load(mm_bf16_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    add_val = residual + mm
    tl.store(add_out_ptr + offsets, add_val, mask=mask)

    weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    normed = tl.load(normed_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    scaled = add_val * weight

    sum_scaled = tl.sum(tl.where(mask, scaled, 0.0), axis=0)
    scaled_normed = scaled * normed
    sum_scaled_normed = tl.sum(tl.where(mask, scaled_normed, 0.0), axis=0)

    row_scale = tl.load(row_scale_ptr + row).to(tl.float32)
    term0 = scaled * HIDDEN
    term1 = term0 - sum_scaled
    term2 = normed * sum_scaled_normed
    epilogue = row_scale * (term1 - term2)

    tl.store(epilogue_out_ptr + offsets, epilogue, mask=mask)
    tl.store(bf16_out_ptr + offsets, epilogue.to(tl.bfloat16), mask=mask)


# 6542053c: (T([4096,4096], bf16), T([8,512,4096], f32), T([4096], f32), T([8,512,4096], f32), T([8,512,1], f32), S([8,512,4096]), S([4096,4096]))
@oracle_impl(hardware="B200", point="6542053c", BLOCK_N=4096, num_warps=8)
def oracle_forward(inputs, *, BLOCK_N: int, num_warps: int):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, _shape_param_0, _shape_param_1 = inputs
    del _shape_param_0

    out_add = torch.empty_strided(
        tuple(int(dim) for dim in arg1_1.shape),
        tuple(int(stride) for stride in arg1_1.stride()),
        device=arg1_1.device,
        dtype=torch.float32,
    )
    out_epilogue = torch.empty_strided(
        tuple(int(dim) for dim in arg3_1.shape),
        tuple(int(stride) for stride in arg3_1.stride()),
        device=arg3_1.device,
        dtype=torch.float32,
    )
    out_bf16 = torch.empty_strided(
        tuple(int(dim) for dim in _shape_param_1),
        (int(_shape_param_1[1]), 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )

    rows = int(arg1_1.numel() // arg1_1.shape[-1])
    hidden = int(arg1_1.shape[-1])
    _albert_ln_backward_dual_sum_kernel[(rows,)](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        out_add,
        out_epilogue,
        out_bf16,
        HIDDEN=hidden,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=3,
    )

    return out_add, out_epilogue, out_bf16, out_bf16.permute(1, 0)
