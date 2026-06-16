"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete Longformer sliding-window softmax-backward and diagonal-layout scatter scope, including the returned zero/full scaffolding tensors, expanded bool masks, the bf16 dropout-scaled incoming gradient reconstruction, natural-exp probability reconstruction, f32 row product sum, fma.rn epilogue, bf16 cast boundary, and final `[288, 512, 512]` band-gradient layout. Inductor lowers the captured graph as generic slice/scatter/view/pad materialization around a generic row reduction, so it materializes many structured intermediates and cannot keep the Longformer diagonal mapping tied to the saved-softmax backward row update. The fix is NEW_PATTERN: add a Longformer sliding-window attention-backward lowering that recognizes the 513-wide diagonal band layout, emits required returned scaffolds, and scatters the row-reduction epilogue directly into the chunk-gradient layout."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


BATCH = 8
SEQ = 1024
HEADS = 12
CHUNK = 256
CHUNKS = 4
WINDOW = 513
HIDDEN = 768
GROUPS = BATCH * HEADS
ROWS = GROUPS * SEQ
FINAL_M = GROUPS * 3
FINAL_N = 512
FINAL_K = 512
DROPOUT_SCALE = 1.1111111111111112


@triton.jit
def _round_to_bf16_f32(x):
    return tl.inline_asm_elementwise(
        "{ .reg .b16 t; cvt.rn.bf16.f32 t, $1; cvt.f32.bf16 $0, t; }",
        constraints="=f,f",
        args=[x],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _fma_rn_f32(a, b, c):
    return tl.inline_asm_elementwise(
        "fma.rn.f32 $0, $1, $2, $3;",
        constraints="=f,f,f,f",
        args=[a, b, c],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _zero_many_bf16_kernel(
    out0,
    out1,
    out4,
    out5,
    out7,
    out8,
    out10,
    out11,
    out12,
    out13,
    out14,
    out15,
    out16,
    out17,
    out18,
    N0: tl.constexpr,
    N1: tl.constexpr,
    N4: tl.constexpr,
    N5: tl.constexpr,
    N7: tl.constexpr,
    N8: tl.constexpr,
    N10: tl.constexpr,
    N11: tl.constexpr,
    N12: tl.constexpr,
    N13: tl.constexpr,
    N14: tl.constexpr,
    N15: tl.constexpr,
    N16: tl.constexpr,
    N17: tl.constexpr,
    N18: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    z = tl.zeros((BLOCK,), dtype=tl.float32)
    tl.store(out0 + offs, z, mask=offs < N0)
    tl.store(out1 + offs, z, mask=offs < N1)
    tl.store(out4 + offs, z, mask=offs < N4)
    tl.store(out5 + offs, z, mask=offs < N5)
    tl.store(out7 + offs, z, mask=offs < N7)
    tl.store(out8 + offs, z, mask=offs < N8)
    tl.store(out10 + offs, z, mask=offs < N10)
    tl.store(out11 + offs, z, mask=offs < N11)
    tl.store(out12 + offs, z, mask=offs < N12)
    tl.store(out13 + offs, z, mask=offs < N13)
    tl.store(out14 + offs, z, mask=offs < N14)
    tl.store(out15 + offs, z, mask=offs < N15)
    tl.store(out16 + offs, z, mask=offs < N16)
    tl.store(out17 + offs, z, mask=offs < N17)
    tl.store(out18 + offs, z, mask=offs < N18)


@triton.jit
def _zero_f32_scalar_kernel(out):
    tl.store(out, tl.full((), 0.0, tl.float32))


@triton.jit
def _expand_bool_masks_kernel(
    mask0_ptr,
    mask1_ptr,
    out0_ptr,
    out1_ptr,
    TOTAL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    k = offs % 257
    tmp = offs // 257
    tmp = tmp // 12
    pos = tmp % 256
    valid = offs < TOTAL
    src = pos * 257 + k
    v0 = tl.load(mask0_ptr + src, mask=valid, other=0.0).to(tl.float32) != 0.0
    v1 = tl.load(mask1_ptr + src, mask=valid, other=0.0).to(tl.float32) != 0.0
    tl.store(out0_ptr + offs, v0, mask=valid)
    tl.store(out1_ptr + offs, v1, mask=valid)


@triton.jit
def _longformer_backward_scatter_kernel(
    grad_src_ptr,
    keep_ptr,
    query_mask_ptr,
    logits_ptr,
    row_shift_ptr,
    row_denom_ptr,
    right_edge_mask_ptr,
    left_edge_mask_ptr,
    final_ptr,
    KEEP_S0: tl.constexpr,
    KEEP_S1: tl.constexpr,
    KEEP_S2: tl.constexpr,
    KEEP_S3: tl.constexpr,
    LOGITS_S0: tl.constexpr,
    LOGITS_S1: tl.constexpr,
    LOGITS_S2: tl.constexpr,
    LOGITS_S3: tl.constexpr,
    SHIFT_S0: tl.constexpr,
    SHIFT_S1: tl.constexpr,
    SHIFT_S2: tl.constexpr,
    DENOM_S0: tl.constexpr,
    DENOM_S1: tl.constexpr,
    DENOM_S2: tl.constexpr,
    WINDOW_: tl.constexpr,
    SEQ_: tl.constexpr,
    HEADS_: tl.constexpr,
    CHUNK_: tl.constexpr,
    CHUNKS_: tl.constexpr,
    HIDDEN_: tl.constexpr,
    FINAL_N_: tl.constexpr,
    FINAL_K_: tl.constexpr,
    DROPOUT_SCALE_: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    col_mask = cols < WINDOW_

    group = row // SEQ_
    seq_pos = row - group * SEQ_
    chunk = seq_pos // CHUNK_
    pos = seq_pos - chunk * CHUNK_
    batch = group // HEADS_
    head = group - batch * HEADS_

    grad_row = group * CHUNKS_ + chunk
    grad_feature = pos + cols
    grad = tl.load(
        grad_src_ptr + (grad_row * CHUNK_ + pos) * HIDDEN_ + grad_feature,
        mask=col_mask,
        other=0.0,
    ).to(tl.float32)

    keep = tl.load(
        keep_ptr
        + batch * KEEP_S0
        + seq_pos * KEEP_S1
        + head * KEEP_S2
        + cols * KEEP_S3,
        mask=col_mask,
        other=0,
    ).to(tl.float32)
    keep_scaled = _round_to_bf16_f32(keep * DROPOUT_SCALE_)
    scaled_grad = _round_to_bf16_f32(grad * keep_scaled)

    masked_query = tl.load(query_mask_ptr + batch * SEQ_ + seq_pos)
    grad_term = tl.where(masked_query, 0.0, scaled_grad)

    logits = tl.load(
        logits_ptr
        + batch * LOGITS_S0
        + seq_pos * LOGITS_S1
        + head * LOGITS_S2
        + cols * LOGITS_S3,
        mask=col_mask,
        other=0.0,
    ).to(tl.float32)
    shift = tl.load(
        row_shift_ptr + batch * SHIFT_S0 + seq_pos * SHIFT_S1 + head * SHIFT_S2
    ).to(tl.float32)
    denom = tl.load(
        row_denom_ptr + batch * DENOM_S0 + seq_pos * DENOM_S1 + head * DENOM_S2
    ).to(tl.float32)
    probs = libdevice.exp(logits - shift) / denom
    product = grad_term * probs
    row_sum = tl.sum(tl.where(col_mask, product, 0.0), axis=0)
    fma = _fma_rn_f32(-probs, row_sum, product)

    start_mask = tl.load(
        left_edge_mask_ptr + pos * 257 + cols,
        mask=col_mask & (cols <= 256),
        other=0.0,
    ).to(tl.float32) != 0.0
    end_mask = tl.load(
        right_edge_mask_ptr + pos * 257 + (cols - 256),
        mask=col_mask & (cols >= 256),
        other=0.0,
    ).to(tl.float32) != 0.0
    fma = tl.where((chunk == 0) & (cols <= 256) & start_mask, 0.0, fma)
    fma = tl.where((chunk == (CHUNKS_ - 1)) & (cols >= 256) & end_mask, 0.0, fma)
    result = fma.to(tl.bfloat16, fp_downcast_rounding="rtne")

    cols_i64 = cols.to(tl.int64)
    out_plane = FINAL_N_ * FINAL_K_

    upper_linear = pos * 513 + (cols_i64 - 256)
    upper_offset = (group * 3 + chunk) * out_plane + upper_linear
    tl.store(
        final_ptr + upper_offset,
        result,
        mask=(chunk <= (CHUNKS_ - 2)) & (cols >= 256) & col_mask,
    )

    left_linear = (pos - 1) * 513 + (cols_i64 + 257)
    left_offset = group * 3 * out_plane + left_linear
    tl.store(
        final_ptr + left_offset,
        result,
        mask=(chunk == 0) & (pos >= 1) & (cols >= 1) & (cols <= 255),
    )

    lower_linear = (255 + pos) * 513 + (cols_i64 + 257)
    lower_offset = (group * 3 + chunk - 1) * out_plane + lower_linear
    tl.store(
        final_ptr + lower_offset,
        result,
        mask=(chunk >= 1) & (cols <= 255),
    )

    right_linear = (256 + pos) * 513 + (cols_i64 - 256)
    right_offset = (group * 3 + CHUNKS_ - 2) * out_plane + right_linear
    tl.store(
        final_ptr + right_offset,
        result,
        mask=(chunk == (CHUNKS_ - 1)) & (cols >= 256) & col_mask & (right_linear < out_plane),
    )


def _empty_bf16(shape, device):
    return torch.empty_strided(shape, tuple(_contiguous_stride(shape)), device=device, dtype=torch.bfloat16)


def _contiguous_stride(shape):
    stride = []
    running = 1
    for size in reversed(shape):
        stride.append(running)
        running *= int(size)
    return reversed(stride)


# AllenaiLongformerBase train, sliding-window softmax backward scaffold + final [288,512,512].
@oracle_impl(hardware="B200", point="39dafa96", BLOCK_N=1024, ZERO_BLOCK=1024, BOOL_BLOCK=1024, num_warps=4)
def oracle_forward(
    inputs,
    *,
    BLOCK_N: int,
    ZERO_BLOCK: int,
    BOOL_BLOCK: int,
    num_warps: int,
):
    (
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        arg5_1,
        arg6_1,
        arg7_1,
        *_shape_params,
    ) = inputs
    del _shape_params

    device = arg0_1.device
    full = _empty_bf16((96, 4, 256, 769), device)
    full_1 = _empty_bf16((96, 4, 197120), device)
    full_2 = torch.empty_strided((), (), device=device, dtype=torch.float32)
    full_3 = _empty_bf16((8, 256, 12, 257), device)
    full_4 = torch.empty_strided((), (), device=device, dtype=torch.bfloat16)
    mask_0 = torch.empty_strided((8, 256, 12, 257), (789504, 3084, 257, 1), device=device, dtype=torch.bool)
    full_5 = _empty_bf16((8, 256, 12, 513), device)
    full_6 = _empty_bf16((8, 1024, 12, 513), device)
    mask_1 = torch.empty_strided((8, 256, 12, 257), (789504, 3084, 257, 1), device=device, dtype=torch.bool)
    full_7 = _empty_bf16((96, 255, 255), device)
    full_8 = _empty_bf16((96, 255, 513), device)
    full_9 = _empty_bf16((96, 512, 513), device)
    full_10 = _empty_bf16((96, 3, 512, 513), device)
    full_11 = _empty_bf16((96, 3, 256, 256), device)
    full_12 = _empty_bf16((96, 3, 256, 513), device)
    full_13 = _empty_bf16((96, 256, 257), device)
    full_14 = _empty_bf16((96, 256, 513), device)
    final = torch.empty_strided(
        (FINAL_M, FINAL_N, FINAL_K),
        (FINAL_N * FINAL_K, FINAL_K, 1),
        device=device,
        dtype=torch.bfloat16,
    )

    max_zero = max(
        full.numel(),
        full_1.numel(),
        full_3.numel(),
        full_5.numel(),
        full_6.numel(),
        full_7.numel(),
        full_8.numel(),
        full_9.numel(),
        full_10.numel(),
        full_11.numel(),
        full_12.numel(),
        full_13.numel(),
        full_14.numel(),
        final.numel(),
    )
    _zero_many_bf16_kernel[(triton.cdiv(max_zero, ZERO_BLOCK),)](
        full,
        full_1,
        full_3,
        full_4,
        full_5,
        full_6,
        full_7,
        full_8,
        full_9,
        full_10,
        full_11,
        full_12,
        full_13,
        full_14,
        final,
        N0=full.numel(),
        N1=full_1.numel(),
        N4=full_3.numel(),
        N5=1,
        N7=full_5.numel(),
        N8=full_6.numel(),
        N10=full_7.numel(),
        N11=full_8.numel(),
        N12=full_9.numel(),
        N13=full_10.numel(),
        N14=full_11.numel(),
        N15=full_12.numel(),
        N16=full_13.numel(),
        N17=full_14.numel(),
        N18=final.numel(),
        BLOCK=ZERO_BLOCK,
        num_warps=8,
        num_stages=3,
    )
    _zero_f32_scalar_kernel[(1,)](full_2)
    _expand_bool_masks_kernel[(triton.cdiv(mask_0.numel(), BOOL_BLOCK),)](
        arg6_1,
        arg7_1,
        mask_0,
        mask_1,
        TOTAL=mask_0.numel(),
        BLOCK=BOOL_BLOCK,
        num_warps=4,
        num_stages=3,
    )
    _longformer_backward_scatter_kernel[(ROWS,)](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        arg5_1,
        arg6_1,
        arg7_1,
        final,
        KEEP_S0=arg1_1.stride(0),
        KEEP_S1=arg1_1.stride(1),
        KEEP_S2=arg1_1.stride(2),
        KEEP_S3=arg1_1.stride(3),
        LOGITS_S0=arg3_1.stride(0),
        LOGITS_S1=arg3_1.stride(1),
        LOGITS_S2=arg3_1.stride(2),
        LOGITS_S3=arg3_1.stride(3),
        SHIFT_S0=arg4_1.stride(0),
        SHIFT_S1=arg4_1.stride(1),
        SHIFT_S2=arg4_1.stride(2),
        DENOM_S0=arg5_1.stride(0),
        DENOM_S1=arg5_1.stride(1),
        DENOM_S2=arg5_1.stride(2),
        WINDOW_=WINDOW,
        SEQ_=SEQ,
        HEADS_=HEADS,
        CHUNK_=CHUNK,
        CHUNKS_=CHUNKS,
        HIDDEN_=HIDDEN,
        FINAL_N_=FINAL_N,
        FINAL_K_=FINAL_K,
        DROPOUT_SCALE_=DROPOUT_SCALE,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=3,
    )

    return (
        full,
        full_1,
        arg2_1.view(8, 1024, 1, 1),
        full_2,
        full_3,
        full_4,
        mask_0,
        full_5,
        full_6,
        mask_1,
        full_7,
        full_8,
        full_9,
        full_10,
        full_11,
        full_12,
        full_13,
        full_14,
        final,
    )
