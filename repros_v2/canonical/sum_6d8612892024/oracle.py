"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete bf16 attention softmax-backward/dropout scope in one Triton row kernel, including the shape-param views, bf16 dropout-mask scaling by 1.1111111111111112, natural libdevice.exp and round-to-nearest f32 division with the captured fp32 row normalizers, explicit bf16 probability rounding and all-masked-row zeroing, the fp32 last-dimension reduction, fused multiply-add epilogue, final bf16 cast, and returned contiguous 3D view, whereas Inductor lowers this as a generic persistent reduction whose fused cast order and launch choices are not a reusable attention-backward template across the 128- and 512-wide points; Inductor cannot do this today as a consistently optimal floor because its scheduler does not expose this softmax-backward row reduction with preserved bf16 rounding boundaries and dropout-mask producer as one tuned full-scope template; the fix is SCHEDULER_FUSION: teach Inductor's attention-backward scheduler to fuse the dropout scaling, probability reconstruction, row reduction, FMA epilogue, and view-equivalent output store in one generated row kernel."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


@triton.jit
def _f32_mul(a, b):
    return tl.inline_asm_elementwise(
        "mul.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _f32_fma(a, b, c):
    return tl.inline_asm_elementwise(
        "fma.rn.f32 $0, $1, $2, $3;",
        constraints="=f,f,f,f",
        args=[a, b, c],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _f32_div(a, b):
    return tl.inline_asm_elementwise(
        "div.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _softmax_backward_dropout_kernel(
    grad_ptr,
    dropout_mask_ptr,
    logits_ptr,
    row_shift_ptr,
    row_denom_ptr,
    zero_row_ptr,
    out_ptr,
    n_rows: tl.constexpr,
    k_len: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    row_mask = rows < n_rows
    col_mask = cols < k_len
    mask = row_mask[:, None] & col_mask[None, :]
    offsets = rows[:, None] * k_len + cols[None, :]

    keep = tl.load(dropout_mask_ptr + offsets, mask=mask, other=0).to(tl.int1)
    dropout_scale = tl.full((BLOCK_M, BLOCK_N), 1.1111111111111112, tl.float32)
    scaled_mask = tl.where(keep, dropout_scale, 0.0).to(tl.bfloat16)

    grad = tl.load(grad_ptr + offsets, mask=mask, other=0.0).to(tl.bfloat16)
    scaled_grad = _f32_mul(grad.to(tl.float32), scaled_mask.to(tl.float32))
    scaled_grad = scaled_grad.to(tl.bfloat16).to(tl.float32)

    logits = tl.load(logits_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    row_shift = tl.load(row_shift_ptr + rows, mask=row_mask, other=0.0).to(tl.float32)
    row_denom = tl.load(row_denom_ptr + rows, mask=row_mask, other=1.0).to(tl.float32)
    numer = libdevice.exp(logits - row_shift[:, None])
    probs = _f32_div(numer, row_denom[:, None]).to(tl.bfloat16)

    zero_row = tl.load(zero_row_ptr + rows, mask=row_mask, other=1).to(tl.int1)
    probs = tl.where(zero_row[:, None], 0.0, probs).to(tl.bfloat16).to(tl.float32)

    prod = _f32_mul(scaled_grad, probs)
    row_sum = tl.sum(tl.where(mask, prod, 0.0), axis=1)
    out = _f32_fma(-probs, row_sum[:, None], prod).to(tl.bfloat16)
    tl.store(out_ptr + offsets, out, mask=mask)


def _launch(inputs, *, BLOCK_M: int, BLOCK_N: int, num_warps: int, num_stages: int):
    (
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        arg5_1,
        _shape_param_0,
        _shape_param_1,
        _shape_param_2,
        _shape_param_3,
    ) = inputs
    del _shape_param_0, _shape_param_1, _shape_param_2

    out_shape = tuple(int(dim) for dim in _shape_param_3)
    k_len = int(out_shape[-1])
    n_rows = arg0_1.numel() // k_len
    out = torch.empty_strided(
        out_shape,
        (out_shape[1] * out_shape[2], out_shape[2], 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )

    grid = (triton.cdiv(n_rows, BLOCK_M),)
    _softmax_backward_dropout_kernel[grid](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        arg5_1,
        out,
        n_rows=n_rows,
        k_len=k_len,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out


# 931c2d63: (T([3072,128,128], bf16), T([256,12,128,128], b8), ...)
# 79d7858b: (T([384,512,512], bf16), T([32,12,512,512], b8), ...)
# 4e534079: (T([256,512,512], bf16), T([64,4,512,512], b8), ...)
# 5d18732f: (T([1024,128,128], bf16), T([256,4,128,128], b8), ...)
@oracle_impl(hardware="B200", point="931c2d63", BLOCK_M=4, BLOCK_N=128, num_warps=4, num_stages=3)
@oracle_impl(hardware="B200", point="79d7858b", BLOCK_M=1, BLOCK_N=512, num_warps=8, num_stages=3)
@oracle_impl(hardware="B200", point="4e534079", BLOCK_M=1, BLOCK_N=512, num_warps=8, num_stages=3)
@oracle_impl(hardware="B200", point="5d18732f", BLOCK_M=4, BLOCK_N=128, num_warps=4, num_stages=3)
def oracle_forward(
    inputs,
    *,
    BLOCK_M: int,
    BLOCK_N: int,
    num_warps: int,
    num_stages: int,
):
    return _launch(
        inputs,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=num_stages,
    )
