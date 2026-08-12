"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete MobileBERT bf16 embedding-add-affine alias scope in one shape-specialized Triton row/hidden writer, including the sliced word embedding, the all-zero token-type embedding folded to direct row-0 loads, the per-hidden scale and bias, the eager-visible bf16 rounding after each add/mul boundary, and the returned `[256,128,512]` base tensor plus three `[32768,512]` metadata views aliasing the same buffer, whereas Inductor lowers the slice/full/embedding/broadcast/add/mul/add/view chain through generic embedding and pointwise/layout scheduling; Inductor cannot express this exact full returned-output envelope today because its scheduler does not fuse embedding gathers, constant-token-type broadcast, affine epilogue, and repeated alias-only returns into one guarded plan while preserving bf16 roundings; the fix is SCHEDULER_FUSION: teach pointwise scheduling to sink this MobileBERT embedding-add producer into an alias-aware affine writer."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


BATCH = 256
SEQ = 128
HIDDEN = 512
ROWS = BATCH * SEQ


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
def _mobilebert_embedding_affine_alias_kernel(
    activation_ptr,
    token_ids_ptr,
    word_table_ptr,
    token_type_table_ptr,
    scale_ptr,
    bias_ptr,
    out_ptr,
    N_ROWS: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    HIDDEN_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    row_mask = rows < N_ROWS
    col_mask = cols < HIDDEN_SIZE
    mask = row_mask[:, None] & col_mask[None, :]

    seq = rows % SEQ_LEN
    token_ids = tl.load(token_ids_ptr + seq, mask=row_mask, other=0)
    offsets = rows[:, None] * HIDDEN_SIZE + cols[None, :]

    activation = tl.load(activation_ptr + offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    word = tl.load(
        word_table_ptr + token_ids[:, None] * HIDDEN_SIZE + cols[None, :],
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    token_type = tl.load(
        token_type_table_ptr + cols,
        mask=col_mask,
        other=0.0,
    ).to(tl.float32)
    scale = tl.load(scale_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)

    add0 = _f32_add(activation, word).to(tl.bfloat16).to(tl.float32)
    add1 = _f32_add(add0, token_type[None, :]).to(tl.bfloat16).to(tl.float32)
    mul = _f32_mul(add1, scale[None, :]).to(tl.bfloat16).to(tl.float32)
    out = _f32_add(mul, bias[None, :]).to(tl.bfloat16)

    tl.store(out_ptr + offsets, out, mask=mask)


def _contiguous_stride(shape):
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= int(dim)
    return tuple(reversed(stride))


def _shape_tuple(shape):
    return tuple(int(dim) for dim in shape)


@oracle_impl(hardware="B200", point="0c4f2b31", BLOCK_M=8, BLOCK_N=512, num_warps=8, num_stages=3)
def oracle_forward(
    inputs,
    *,
    BLOCK_M: int,
    BLOCK_N: int,
    num_warps: int,
    num_stages: int,
):
    (
        activation,
        token_ids,
        word_table,
        token_type_table,
        scale,
        bias,
        base_shape,
        _full_shape,
        view_shape_0,
        view_shape_1,
        view_shape_2,
    ) = inputs
    del _full_shape

    base_shape = _shape_tuple(base_shape)
    view_shape_0 = _shape_tuple(view_shape_0)
    view_shape_1 = _shape_tuple(view_shape_1)
    view_shape_2 = _shape_tuple(view_shape_2)
    out = torch.empty_strided(
        base_shape,
        _contiguous_stride(base_shape),
        device=activation.device,
        dtype=activation.dtype,
    )

    _mobilebert_embedding_affine_alias_kernel[(triton.cdiv(ROWS, BLOCK_M),)](
        activation,
        token_ids,
        word_table,
        token_type_table,
        scale,
        bias,
        out,
        N_ROWS=ROWS,
        SEQ_LEN=SEQ,
        HIDDEN_SIZE=HIDDEN,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return (
        out,
        out.view(view_shape_0),
        out.view(view_shape_1),
        out.view(view_shape_2),
    )
