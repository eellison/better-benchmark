"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete MobileBERT bf16 embedding-add-affine staging scope in one shape-specialized Triton row/hidden writer, including the returned zero int64 `[256,128]` token-type ids, the sliced token-id word embedding `[1,128,512]`, the returned all-zero-token-type embedding `[256,128,512]`, the f32 add/add/mul/add affine result, and the final bf16 `[32768,512]` view. Inductor lowers the captured slice/full/embedding/broadcast/add/mul/add/cast/view graph through generic embedding and pointwise/layout scheduling even though the token-type embedding is statically row zero and all consumers share the same row/hidden loop nest. Inductor cannot express this exact full returned-output envelope today because its scheduler does not fuse embedding gathers, constant-token-type broadcast, affine epilogue, and the visible bf16 cast while also emitting the side outputs. The fix is SCHEDULER_FUSION: keep the embedding rows and affine operands virtual in one guarded row template and write every escaping tensor from the same pass."""

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
def _mobilebert_embedding_affine_kernel(
    token_ids_ptr,
    activations_ptr,
    word_table_ptr,
    token_type_table_ptr,
    scale_ptr,
    bias_ptr,
    full_ptr,
    word_embedding_ptr,
    token_embedding_ptr,
    affine_ptr,
    bf16_view_ptr,
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

    activation = tl.load(activations_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    word = tl.load(
        word_table_ptr + token_ids[:, None] * HIDDEN_SIZE + cols[None, :],
        mask=mask,
        other=0.0,
    )
    token_type = tl.load(token_type_table_ptr + cols, mask=col_mask, other=0.0)
    scale = tl.load(scale_ptr + cols, mask=col_mask, other=0.0)
    bias = tl.load(bias_ptr + cols, mask=col_mask, other=0.0)

    add0 = _f32_add(activation, word)
    add1 = _f32_add(add0, token_type[None, :])
    scaled = _f32_mul(add1, scale[None, :])
    affine = _f32_add(scaled, bias[None, :])

    tl.store(full_ptr + rows, tl.zeros((BLOCK_M,), dtype=tl.int64), mask=row_mask)
    tl.store(
        word_embedding_ptr + seq[:, None] * HIDDEN_SIZE + cols[None, :],
        word,
        mask=(rows[:, None] < SEQ_LEN) & col_mask[None, :],
    )
    tl.store(token_embedding_ptr + offsets, token_type[None, :], mask=mask)
    tl.store(affine_ptr + offsets, affine, mask=mask)
    tl.store(bf16_view_ptr + offsets, affine, mask=mask)


# (T([1,512], i64), T([32768,512], bf16), T([512,512], f32), T([2,512], f32), T([512], f32), T([512], f32), S([256,128]), S([256,128,512]), S([32768,512]))
@oracle_impl(hardware="B200", point="c404fad8", BLOCK_M=1, BLOCK_N=512, num_warps=8, num_stages=3)
def oracle_forward(inputs, *, BLOCK_M: int, BLOCK_N: int, num_warps: int, num_stages: int):
    token_ids, activations, word_table, token_type_table, scale, bias, _full_shape, _view_shape, _flat_shape = inputs
    full = torch.empty_strided(
        (BATCH, SEQ),
        (SEQ, 1),
        device=token_ids.device,
        dtype=torch.int64,
    )
    word_embedding = torch.empty_strided(
        (1, SEQ, HIDDEN),
        (SEQ * HIDDEN, HIDDEN, 1),
        device=word_table.device,
        dtype=torch.float32,
    )
    token_embedding = torch.empty_strided(
        (BATCH, SEQ, HIDDEN),
        (SEQ * HIDDEN, HIDDEN, 1),
        device=token_type_table.device,
        dtype=torch.float32,
    )
    affine = torch.empty_strided(
        (BATCH, SEQ, HIDDEN),
        (SEQ * HIDDEN, HIDDEN, 1),
        device=activations.device,
        dtype=torch.float32,
    )
    bf16_view = torch.empty_strided(
        (ROWS, HIDDEN),
        (HIDDEN, 1),
        device=activations.device,
        dtype=activations.dtype,
    )

    _mobilebert_embedding_affine_kernel[(triton.cdiv(ROWS, BLOCK_M),)](
        token_ids,
        activations,
        word_table,
        token_type_table,
        scale,
        bias,
        full,
        word_embedding,
        token_embedding,
        affine,
        bf16_view,
        N_ROWS=ROWS,
        SEQ_LEN=SEQ,
        HIDDEN_SIZE=HIDDEN,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return full, word_embedding, token_embedding, affine, bf16_view
