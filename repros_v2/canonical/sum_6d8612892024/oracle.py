"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete bf16 attention softmax-backward/dropout scope with Triton kernels, including the shape-param views, bf16 dropout-mask scaling by 1.1111111111111112, natural libdevice.exp and round-to-nearest f32 division with the captured fp32 row normalizers, explicit bf16 probability rounding and all-masked-row zeroing, the fp32 last-dimension reduction, fused multiply-add epilogue, final bf16 cast, and returned contiguous 3D view, whereas Inductor lowers the dropout producer, exp/div probability reconstruction, mask selection, reduction, FMA epilogue, and layout-only view through generic scheduler fragments; Inductor cannot do this today because its scheduler does not recognize this softmax-backward row reduction with preserved bf16 rounding boundaries and dropout-mask producer as one full-scope fused pattern; the fix is SCHEDULER_FUSION: teach Inductor's attention backward scheduler to fuse the dropout scaling, probability reconstruction, row reduction, FMA epilogue, and view-equivalent output store in a generated row kernel."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


@triton.jit
def _f32_add(a, b):
    return tl.inline_asm_elementwise(
        "add.rn.f32 $0, $1, $2;",
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
def _product_and_prob(
    grad_ptr,
    dropout_mask_ptr,
    logits_ptr,
    row_shift_ptr,
    row_denom_ptr,
    zero_row_ptr,
    rows,
    cols,
    row_mask,
    mask,
    k_len: tl.constexpr,
):
    offsets = rows[:, None] * k_len + cols[None, :]

    grad = tl.load(grad_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    keep = tl.load(dropout_mask_ptr + offsets, mask=mask, other=0).to(tl.int1)
    keep_scale = (keep.to(tl.float32) * 1.1111111111111112).to(
        tl.bfloat16, fp_downcast_rounding="rtne"
    ).to(tl.float32)
    scaled_grad = (grad * keep_scale).to(
        tl.bfloat16, fp_downcast_rounding="rtne"
    ).to(tl.float32)

    logits = tl.load(logits_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    row_shift = tl.load(row_shift_ptr + rows, mask=row_mask, other=0.0).to(tl.float32)
    row_denom = tl.load(row_denom_ptr + rows, mask=row_mask, other=1.0).to(tl.float32)
    numer = libdevice.exp(logits - row_shift[:, None])
    probs = _f32_div(numer, row_denom[:, None]).to(
        tl.bfloat16, fp_downcast_rounding="rtne"
    )
    zero_row = tl.load(zero_row_ptr + rows, mask=row_mask, other=1).to(tl.int1)
    probs = tl.where(zero_row[:, None], 0.0, probs).to(
        tl.bfloat16, fp_downcast_rounding="rtne"
    ).to(tl.float32)

    return scaled_grad * probs, probs


@triton.jit
def _row_sum_fma_recompute_prob_kernel(
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
    CHUNK_N: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = rows < n_rows
    chunk_cols = tl.arange(0, CHUNK_N)

    lane_sum = tl.zeros((BLOCK_M, CHUNK_N), tl.float32)
    for start in tl.static_range(0, BLOCK_N, CHUNK_N):
        cols = start + chunk_cols
        mask = row_mask[:, None] & (cols[None, :] < k_len)
        product, _ = _product_and_prob(
            grad_ptr,
            dropout_mask_ptr,
            logits_ptr,
            row_shift_ptr,
            row_denom_ptr,
            zero_row_ptr,
            rows,
            cols,
            row_mask,
            mask,
            k_len,
        )
        lane_sum = _f32_add(lane_sum, tl.where(mask, product, 0.0))

    row_sum = tl.sum(lane_sum, axis=1)[:, None].to(tl.float32)

    for start in tl.static_range(0, BLOCK_N, CHUNK_N):
        cols = start + chunk_cols
        mask = row_mask[:, None] & (cols[None, :] < k_len)
        offsets = rows[:, None] * k_len + cols[None, :]
        product, probs = _product_and_prob(
            grad_ptr,
            dropout_mask_ptr,
            logits_ptr,
            row_shift_ptr,
            row_denom_ptr,
            zero_row_ptr,
            rows,
            cols,
            row_mask,
            mask,
            k_len,
        )
        out = _f32_fma(-probs, row_sum, product).to(
            tl.bfloat16, fp_downcast_rounding="rtne"
        )
        tl.store(out_ptr + offsets, out, mask=mask)


@triton.jit
def _row_sum_fma_single_pass_kernel(
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
    mask = row_mask[:, None] & (cols[None, :] < k_len)
    offsets = rows[:, None] * k_len + cols[None, :]

    product, probs = _product_and_prob(
        grad_ptr,
        dropout_mask_ptr,
        logits_ptr,
        row_shift_ptr,
        row_denom_ptr,
        zero_row_ptr,
        rows,
        cols,
        row_mask,
        mask,
        k_len,
    )
    row_sum = tl.sum(tl.where(mask, product, 0.0), axis=1)[:, None].to(tl.float32)
    out = _f32_fma(-probs, row_sum, product).to(
        tl.bfloat16, fp_downcast_rounding="rtne"
    )
    tl.store(out_ptr + offsets, out, mask=mask)


@triton.jit
def _probability_kernel(
    logits_ptr,
    row_shift_ptr,
    row_denom_ptr,
    zero_row_ptr,
    probs_ptr,
    n_rows: tl.constexpr,
    k_len: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    row_mask = rows < n_rows
    mask = row_mask[:, None] & (cols[None, :] < k_len)
    offsets = rows[:, None] * k_len + cols[None, :]

    logits = tl.load(logits_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    row_shift = tl.load(row_shift_ptr + rows, mask=row_mask, other=0.0).to(tl.float32)
    row_denom = tl.load(row_denom_ptr + rows, mask=row_mask, other=1.0).to(tl.float32)
    numer = libdevice.exp(logits - row_shift[:, None])
    probs = _f32_div(numer, row_denom[:, None]).to(
        tl.bfloat16, fp_downcast_rounding="rtne"
    )
    zero_row = tl.load(zero_row_ptr + rows, mask=row_mask, other=1).to(tl.int1)
    probs = tl.where(zero_row[:, None], 0.0, probs).to(
        tl.bfloat16, fp_downcast_rounding="rtne"
    )

    tl.store(probs_ptr + offsets, probs, mask=mask)


@triton.jit
def _row_sum_fma_bf16_prob_kernel(
    grad_ptr,
    dropout_mask_ptr,
    probs_ptr,
    out_ptr,
    n_rows: tl.constexpr,
    k_len: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CHUNK_N: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = rows < n_rows
    chunk_cols = tl.arange(0, CHUNK_N)

    lane_sum = tl.zeros((BLOCK_M, CHUNK_N), tl.float32)
    for start in tl.static_range(0, BLOCK_N, CHUNK_N):
        cols = start + chunk_cols
        mask = row_mask[:, None] & (cols[None, :] < k_len)
        offsets = rows[:, None] * k_len + cols[None, :]

        grad = tl.load(grad_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        keep = tl.load(dropout_mask_ptr + offsets, mask=mask, other=0).to(tl.int1)
        probs = tl.load(probs_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

        keep_scale = (keep.to(tl.float32) * 1.1111111111111112).to(
            tl.bfloat16, fp_downcast_rounding="rtne"
        ).to(tl.float32)
        scaled_grad = (grad * keep_scale).to(
            tl.bfloat16, fp_downcast_rounding="rtne"
        ).to(tl.float32)
        product = scaled_grad * probs
        lane_sum = _f32_add(lane_sum, tl.where(mask, product, 0.0))

    row_sum = tl.sum(lane_sum, axis=1)[:, None].to(tl.float32)

    for start in tl.static_range(0, BLOCK_N, CHUNK_N):
        cols = start + chunk_cols
        mask = row_mask[:, None] & (cols[None, :] < k_len)
        offsets = rows[:, None] * k_len + cols[None, :]

        grad = tl.load(grad_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        keep = tl.load(dropout_mask_ptr + offsets, mask=mask, other=0).to(tl.int1)
        probs = tl.load(probs_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

        keep_scale = (keep.to(tl.float32) * 1.1111111111111112).to(
            tl.bfloat16, fp_downcast_rounding="rtne"
        ).to(tl.float32)
        scaled_grad = (grad * keep_scale).to(
            tl.bfloat16, fp_downcast_rounding="rtne"
        ).to(tl.float32)
        product = scaled_grad * probs
        out = _f32_fma(-probs, row_sum, product).to(
            tl.bfloat16, fp_downcast_rounding="rtne"
        )
        tl.store(out_ptr + offsets, out, mask=mask)


def _launch(
    inputs,
    *,
    BLOCK_M: int,
    BLOCK_N: int,
    CHUNK_N: int,
    num_warps: int,
    num_stages: int,
    MATERIALIZE_PROBS: bool,
    SINGLE_PASS: bool,
):
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
    if SINGLE_PASS:
        _row_sum_fma_single_pass_kernel[grid](
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

    if not MATERIALIZE_PROBS:
        _row_sum_fma_recompute_prob_kernel[grid](
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
            CHUNK_N=CHUNK_N,
            num_warps=num_warps,
            num_stages=1,
        )
        return out

    probs = torch.empty_strided(
        out_shape,
        (out_shape[1] * out_shape[2], out_shape[2], 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    _probability_kernel[grid](
        arg2_1,
        arg3_1,
        arg4_1,
        arg5_1,
        probs,
        n_rows=n_rows,
        k_len=k_len,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    _row_sum_fma_bf16_prob_kernel[grid](
        arg0_1,
        arg1_1,
        probs,
        out,
        n_rows=n_rows,
        k_len=k_len,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        CHUNK_N=CHUNK_N,
        num_warps=num_warps,
        num_stages=1,
    )
    return out


# 931c2d63: (T([3072,128,128], bf16), T([256,12,128,128], b8), ...)
# 79d7858b: (T([384,512,512], bf16), T([32,12,512,512], b8), ...)
# 4e534079: (T([256,512,512], bf16), T([64,4,512,512], b8), ...)
# 5d18732f: (T([1024,128,128], bf16), T([256,4,128,128], b8), ...)
@oracle_impl(hardware="B200", point="931c2d63", BLOCK_M=4, BLOCK_N=128, CHUNK_N=64, num_warps=1, num_stages=3, MATERIALIZE_PROBS=False, SINGLE_PASS=True)
@oracle_impl(hardware="B200", point="79d7858b", BLOCK_M=1, BLOCK_N=512, CHUNK_N=256, num_warps=1, num_stages=3, MATERIALIZE_PROBS=True, SINGLE_PASS=False)
@oracle_impl(hardware="B200", point="4e534079", BLOCK_M=1, BLOCK_N=512, CHUNK_N=64, num_warps=4, num_stages=3, MATERIALIZE_PROBS=False, SINGLE_PASS=True)
@oracle_impl(hardware="B200", point="5d18732f", BLOCK_M=4, BLOCK_N=128, CHUNK_N=64, num_warps=4, num_stages=3, MATERIALIZE_PROBS=False, SINGLE_PASS=False)
def oracle_forward(
    inputs,
    *,
    BLOCK_M: int,
    BLOCK_N: int,
    CHUNK_N: int,
    num_warps: int,
    num_stages: int,
    MATERIALIZE_PROBS: bool,
    SINGLE_PASS: bool,
):
    return _launch(
        inputs,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        CHUNK_N=CHUNK_N,
        num_warps=num_warps,
        num_stages=num_stages,
        MATERIALIZE_PROBS=MATERIALIZE_PROBS,
        SINGLE_PASS=SINGLE_PASS,
    )
