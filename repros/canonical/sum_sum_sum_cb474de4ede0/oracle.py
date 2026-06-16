"""Gap diagnosis (classification: SCATTER_REDUCE): this oracle computes the complete MobileBERT bf16/fp32 backward tuple by sharing the row-token producer across the two column reductions, the bf16-rounded dense side output and transpose alias, the bf16-side reduction, and both duplicate-preserving masked index accumulations, whereas Inductor materializes the shared `[256,128,512]` intermediates and lowers the reductions, `_unsafe_masked_index_put_accumulate` calls, bf16 conversion, and transpose as separate generic scheduled regions; Inductor cannot do this today because scheduler/codegen does not recognize this embedding-gradient scatter-reduce family as one shared producer with sibling reductions, explicit bf16 cast boundaries, and multiple accumulator destinations; the fix is SCATTER_REDUCE: add a lowering that fuses rowwise producers into direct duplicate-safe scatter-reduce epilogues and sibling reductions without materializing avoidable intermediates."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


BATCH = 256
TOKENS = 128
VOCAB_ROWS = 512
BUCKET_ROWS = 2


def _ceil_pow2(value):
    return 1 << (int(value) - 1).bit_length()


@triton.jit
def _add_rn(a, b):
    return tl.inline_asm_elementwise(
        "add.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _mul_rn(a, b):
    return tl.inline_asm_elementwise(
        "mul.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _zero_kernel(out_ptr, total: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    tl.store(out_ptr + offsets, tl.zeros((BLOCK,), dtype=tl.float32), mask=offsets < total)


@triton.jit
def _producer_partials_kernel(
    arg0_ptr,
    arg1_ptr,
    arg2_ptr,
    arg3_ptr,
    arg4_ptr,
    arg5_ptr,
    arg6_ptr,
    scale_ptr,
    bucket_index_ptr,
    side_f32_ptr,
    side_bf16_ptr,
    partial_sum_ptr,
    partial_weighted_ptr,
    partial_side_sum_ptr,
    partial_bucket0_ptr,
    partial_bucket1_ptr,
    hidden: tl.constexpr,
    xblock: tl.constexpr,
    rblock: tl.constexpr,
):
    xindex = tl.program_id(0) * xblock + tl.arange(0, xblock)[:, None]
    token_offsets = tl.arange(0, rblock)[None, :]
    hidden_index = xindex % hidden
    batch_index = xindex // hidden
    flat_row = batch_index * rblock + token_offsets
    offsets = flat_row * hidden + hidden_index

    x = _add_rn(
        tl.load(arg1_ptr + offsets).to(tl.float32),
        tl.load(arg0_ptr + offsets).to(tl.float32),
    )
    x = _add_rn(x, tl.load(arg2_ptr + offsets).to(tl.float32))
    x = _add_rn(x, tl.load(arg3_ptr + offsets).to(tl.float32))

    weighted_rhs = _add_rn(
        tl.load(arg4_ptr + offsets).to(tl.float32),
        tl.load(arg5_ptr + token_offsets * hidden + hidden_index).to(tl.float32),
    )
    weighted_rhs = _add_rn(weighted_rhs, tl.load(arg6_ptr + offsets).to(tl.float32))

    scale = tl.load(scale_ptr + hidden_index).to(tl.float32)
    weighted = _mul_rn(x, weighted_rhs)
    side_f32 = _mul_rn(x, scale)
    side_bf16 = side_f32.to(tl.bfloat16)

    tl.store(side_f32_ptr + offsets, side_f32)
    tl.store(side_bf16_ptr + offsets, side_bf16)

    bucket = tl.load(bucket_index_ptr + flat_row).to(tl.int64)
    bucket0 = bucket == 0
    partial_offsets = batch_index * hidden + hidden_index
    tl.store(partial_sum_ptr + partial_offsets, tl.sum(x, axis=1)[:, None])
    tl.store(partial_weighted_ptr + partial_offsets, tl.sum(weighted, axis=1)[:, None])
    tl.store(
        partial_side_sum_ptr + partial_offsets,
        tl.sum(side_bf16.to(tl.float32), axis=1)[:, None],
    )
    tl.store(
        partial_bucket0_ptr + partial_offsets,
        tl.sum(tl.where(bucket0, side_f32, 0.0), axis=1)[:, None],
    )
    tl.store(
        partial_bucket1_ptr + partial_offsets,
        tl.sum(tl.where(bucket0, 0.0, side_f32), axis=1)[:, None],
    )


@triton.jit
def _finalize_partials_kernel(
    partial_sum_ptr,
    partial_weighted_ptr,
    partial_side_sum_ptr,
    partial_bucket0_ptr,
    partial_bucket1_ptr,
    out_sum_ptr,
    out_weighted_ptr,
    out_bucket_ptr,
    out_side_sum_ptr,
    hidden: tl.constexpr,
    block_batch: tl.constexpr,
):
    col = tl.program_id(0)
    batches = tl.arange(0, block_batch)
    offsets = batches * hidden + col
    sum_acc = tl.load(partial_sum_ptr + offsets).to(tl.float32)
    weighted_acc = tl.load(partial_weighted_ptr + offsets).to(tl.float32)
    side_acc = tl.load(partial_side_sum_ptr + offsets).to(tl.float32)
    bucket0_acc = tl.load(partial_bucket0_ptr + offsets).to(tl.float32)
    bucket1_acc = tl.load(partial_bucket1_ptr + offsets).to(tl.float32)

    tl.store(out_sum_ptr + col, tl.sum(sum_acc, axis=0))
    tl.store(out_weighted_ptr + col, tl.sum(weighted_acc, axis=0))
    tl.store(out_side_sum_ptr + col, tl.sum(side_acc, axis=0).to(tl.bfloat16).to(tl.float32))
    tl.store(out_bucket_ptr + col, tl.sum(bucket0_acc, axis=0))
    tl.store(out_bucket_ptr + hidden + col, tl.sum(bucket1_acc, axis=0))


@triton.jit
def _token_scatter_kernel(
    side_f32_ptr,
    token_index_ptr,
    out_token_ptr,
    hidden: tl.constexpr,
    tokens: tl.constexpr,
    vocab_rows: tl.constexpr,
    block_batch: tl.constexpr,
    block_c: tl.constexpr,
):
    token = tl.program_id(0)
    c_block = tl.program_id(1)
    batches = tl.arange(0, block_batch)[:, None]
    cols = c_block * block_c + tl.arange(0, block_c)[None, :]
    offsets = (batches * tokens + token) * hidden + cols
    values = tl.load(side_f32_ptr + offsets, mask=cols < hidden, other=0.0).to(tl.float32)
    token_sum = tl.sum(values, axis=0)

    raw_index = tl.load(token_index_ptr + token).to(tl.int64)
    valid = (raw_index >= 0) & (raw_index < vocab_rows) & (raw_index != -1)
    tl.atomic_add(
        out_token_ptr + raw_index * hidden + cols,
        token_sum[None, :],
        sem="relaxed",
        mask=(cols < hidden) & valid,
    )


@oracle_impl(
    hardware="B200",
    point="4e4a9284",
    xblock=32,
    token_block_c=16,
    num_warps=4,
    token_warps=4,
    final_warps=4,
)
def oracle_forward(inputs, *, xblock, token_block_c, num_warps, token_warps, final_warps):
    (
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        arg5_1,
        arg6_1,
        arg7_1,
        arg8_1,
        arg9_1,
        *_,
    ) = inputs
    hidden = int(arg7_1.shape[0])
    device = arg1_1.device

    out_sum = torch.empty((hidden,), device=device, dtype=torch.float32)
    out_weighted = torch.empty((hidden,), device=device, dtype=torch.float32)
    out_bucket = torch.empty((BUCKET_ROWS, hidden), device=device, dtype=torch.float32)
    out_token = torch.empty((VOCAB_ROWS, hidden), device=device, dtype=torch.float32)
    side_f32 = torch.empty((BATCH, TOKENS, hidden), device=device, dtype=torch.float32)
    side_bf16 = torch.empty((BATCH * TOKENS, hidden), device=device, dtype=torch.bfloat16)
    out_side_sum = torch.empty((hidden,), device=device, dtype=torch.float32)

    partial_shape = (BATCH, hidden)
    partial_sum = torch.empty(partial_shape, device=device, dtype=torch.float32)
    partial_weighted = torch.empty(partial_shape, device=device, dtype=torch.float32)
    partial_side_sum = torch.empty(partial_shape, device=device, dtype=torch.float32)
    partial_bucket0 = torch.empty(partial_shape, device=device, dtype=torch.float32)
    partial_bucket1 = torch.empty(partial_shape, device=device, dtype=torch.float32)

    xnumel = BATCH * hidden
    _producer_partials_kernel[(triton.cdiv(xnumel, xblock),)](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        arg5_1,
        arg6_1,
        arg7_1,
        arg8_1,
        side_f32,
        side_bf16,
        partial_sum,
        partial_weighted,
        partial_side_sum,
        partial_bucket0,
        partial_bucket1,
        hidden=hidden,
        xblock=xblock,
        rblock=TOKENS,
        num_warps=num_warps,
    )
    _finalize_partials_kernel[(hidden,)](
        partial_sum,
        partial_weighted,
        partial_side_sum,
        partial_bucket0,
        partial_bucket1,
        out_sum,
        out_weighted,
        out_bucket,
        out_side_sum,
        hidden=hidden,
        block_batch=_ceil_pow2(BATCH),
        num_warps=final_warps,
    )
    _zero_kernel[(triton.cdiv(VOCAB_ROWS * hidden, 1024),)](
        out_token,
        total=VOCAB_ROWS * hidden,
        BLOCK=1024,
        num_warps=4,
    )
    _token_scatter_kernel[(TOKENS, triton.cdiv(hidden, token_block_c))](
        side_f32,
        arg9_1,
        out_token,
        hidden=hidden,
        tokens=TOKENS,
        vocab_rows=VOCAB_ROWS,
        block_batch=_ceil_pow2(BATCH),
        block_c=token_block_c,
        num_warps=token_warps,
    )
    return (
        out_sum,
        out_weighted,
        out_bucket,
        out_token,
        side_bf16,
        side_bf16.permute(1, 0),
        out_side_sum,
    )
