"""Gap diagnosis (classification: SCATTER_REDUCE): this oracle computes the complete ConvBERT attention-gradient scatter/reduction tuple by reconstructing the width-9 natural-exp probabilities once in Triton, returning the original zero `full` tensor, scatter-adding the bf16-rounded value producer directly into the live cropped `[16384,384]` result, emitting the sibling bf16 softmax-backward `[16384,54]` result, and finalizing both returned bf16-rounded column sums from those materialized outputs, whereas Inductor lowers the zero full, exp/div producer, bf16 cast, `index_put(accumulate=True)`, crop/clone/view fanout, row fma epilogue, and sibling reductions as generic scheduled kernels over intermediates; Inductor cannot do this today because its scheduler/codegen cannot represent an indexed scatter-reduce with shared probability reconstruction, explicit bf16 rounding boundaries, view-only alias returns, and dependent column reductions as one full-scope plan; the fix is SCATTER_REDUCE: add a guarded lowering that fuses the probability producer into the scatter and row-fma consumers while reducing the returned bf16 outputs without rematerializing the layout chain."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


BATCH = 32
TOKENS = 512
GROUPS = 6
PROB_K = 9
PROB_ROWS = 98304
OUT0_HIDDEN = 384
OUT3_HIDDEN = 54
OUT0_ROWS = BATCH * TOKENS
PADDED_TOKENS = 520
CROP = 4
ARG4_CHANNELS = 64
FULL_NUMEL = BATCH * OUT0_HIDDEN * PADDED_TOKENS


@triton.jit
def _add_rn_f32(a, b):
    return tl.inline_asm_elementwise(
        "add.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _sub_rn_f32(a, b):
    return tl.inline_asm_elementwise(
        "sub.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _mul_rn_f32(a, b):
    return tl.inline_asm_elementwise(
        "mul.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _div_rn_f32(a, b):
    return tl.inline_asm_elementwise(
        "div.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
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
def _probabilities(
    bias_ptr,
    logits_ptr,
    shift_ptr,
    denom_ptr,
    rows,
    ks,
    row_mask,
    k_mask,
):
    token = rows // 6
    group = rows - token * 6
    channel = group[:, None] * 9 + ks[None, :]
    mask = row_mask[:, None] & k_mask[None, :]
    bias = tl.load(bias_ptr + channel, mask=mask, other=0.0).to(tl.bfloat16)
    bias_f32 = bias.to(tl.float32)
    logits = tl.load(logits_ptr + token[:, None] * 54 + channel, mask=mask, other=0.0).to(tl.float32)
    rounded_scores = _add_rn_f32(logits, bias_f32).to(tl.bfloat16).to(tl.float32)
    shifted = _sub_rn_f32(rounded_scores, tl.load(shift_ptr + rows, mask=row_mask, other=0.0).to(tl.float32)[:, None])
    numer = libdevice.exp(shifted)
    denom = tl.load(denom_ptr + rows, mask=row_mask, other=1.0).to(tl.float32)
    return _div_rn_f32(numer, denom[:, None])


@triton.jit
def _zero_full_and_temp_kernel(
    full_ptr,
    temp_ptr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(full_ptr + offsets, tl.zeros((BLOCK,), dtype=tl.float32), mask=offsets < 6389760)
    tl.store(temp_ptr + offsets, tl.zeros((BLOCK,), dtype=tl.float32), mask=offsets < 6291456)


@triton.jit
def _scatter_probabilities_kernel(
    bias_ptr,
    logits_ptr,
    shift_ptr,
    denom_ptr,
    arg4_ptr,
    index_ptr,
    temp_ptr,
    BLOCK_C: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0)
    c_block = tl.program_id(1)
    cs = c_block * BLOCK_C + tl.arange(0, BLOCK_C)
    ks = tl.arange(0, BLOCK_K)
    row_vec = tl.full((1,), row, tl.int64)
    probs = _probabilities(
        bias_ptr,
        logits_ptr,
        shift_ptr,
        denom_ptr,
        row_vec,
        ks,
        tl.full((1,), True, tl.int1),
        ks < 9,
    )
    probs = tl.reshape(probs, (BLOCK_K,))

    token = row // 6
    group = row - token * 6
    batch = token // 512
    seq = token - batch * 512
    idx = tl.load(index_ptr + ks * 512 + seq, mask=ks < 9, other=-1).to(tl.int64)
    live = (ks < 9) & (idx >= 4) & (idx < 516)
    out_row = batch * 512 + idx - 4

    c_mask = cs < 64
    arg4 = tl.load(arg4_ptr + row * 64 + cs, mask=c_mask, other=0.0).to(tl.float32)
    scatter_probs = probs.to(tl.bfloat16).to(tl.float32)
    values = _mul_rn_f32(arg4[:, None], scatter_probs[None, :]).to(tl.bfloat16).to(tl.float32)
    out_channel = group * 64 + cs
    out_offsets = out_row[None, :] * 384 + out_channel[:, None]
    tl.atomic_add(temp_ptr + out_offsets, values, sem="relaxed", mask=c_mask[:, None] & live[None, :])


@triton.jit
def _store_out3_kernel(
    bias_ptr,
    logits_ptr,
    shift_ptr,
    denom_ptr,
    arg5_ptr,
    out3_ptr,
    BLOCK_R: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
    ks = tl.arange(0, BLOCK_K)
    row_mask = rows < 98304
    k_mask = ks < 9
    probs = _probabilities(
        bias_ptr,
        logits_ptr,
        shift_ptr,
        denom_ptr,
        rows,
        ks,
        row_mask,
        k_mask,
    )
    mask = row_mask[:, None] & k_mask[None, :]
    products = _mul_rn_f32(
        tl.load(arg5_ptr + rows[:, None] * 9 + ks[None, :], mask=mask, other=0.0).to(tl.float32),
        probs,
    )
    row_sum = tl.sum(tl.where(mask, products, 0.0), axis=1)
    out = _fma_rn_f32(-probs, row_sum[:, None], products).to(tl.bfloat16)

    token = rows // 6
    group = rows - token * 6
    channel = group[:, None] * 9 + ks[None, :]
    tl.store(out3_ptr + token[:, None] * 54 + channel, out, mask=mask)


@triton.jit
def _finalize_out0_partials_kernel(
    temp_ptr,
    out0_ptr,
    partials_ptr,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    row_group = tl.program_id(0)
    col_group = tl.program_id(1)
    rows = row_group * BLOCK_R + tl.arange(0, BLOCK_R)
    cols = col_group * BLOCK_C + tl.arange(0, BLOCK_C)
    mask = (rows[:, None] < 16384) & (cols[None, :] < 384)
    offsets = rows[:, None] * 384 + cols[None, :]
    rounded = tl.load(temp_ptr + offsets, mask=mask, other=0.0).to(tl.float32).to(tl.bfloat16)
    tl.store(out0_ptr + offsets, rounded, mask=mask)
    sums = tl.sum(tl.where(mask, rounded.to(tl.float32), 0.0), axis=0)
    tl.store(partials_ptr + row_group * 384 + cols, sums, mask=cols < 384)


@triton.jit
def _reduce_bf16_columns_kernel(
    source_ptr,
    partials_ptr,
    ROWS: tl.constexpr,
    HIDDEN: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    row_group = tl.program_id(0)
    col_group = tl.program_id(1)
    rows = row_group * BLOCK_R + tl.arange(0, BLOCK_R)
    cols = col_group * BLOCK_C + tl.arange(0, BLOCK_C)
    mask = (rows[:, None] < ROWS) & (cols[None, :] < HIDDEN)
    vals = tl.load(source_ptr + rows[:, None] * HIDDEN + cols[None, :], mask=mask, other=0.0).to(tl.float32)
    sums = tl.sum(tl.where(mask, vals, 0.0), axis=0)
    tl.store(partials_ptr + row_group * HIDDEN + cols, sums, mask=cols < HIDDEN)


@triton.jit
def _finalize_bf16_sum_kernel(
    partials_ptr,
    out_ptr,
    NUM_GROUPS: tl.constexpr,
    HIDDEN: tl.constexpr,
    GROUP_BLOCK: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    cols = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    groups = tl.arange(0, GROUP_BLOCK)
    mask = (groups[:, None] < NUM_GROUPS) & (cols[None, :] < HIDDEN)
    values = tl.load(partials_ptr + groups[:, None] * HIDDEN + cols[None, :], mask=mask, other=0.0).to(tl.float32)
    rounded_sum = tl.sum(values, axis=0).to(tl.bfloat16).to(tl.float32)
    tl.store(out_ptr + cols, rounded_sum, mask=cols < HIDDEN)


@oracle_impl(hardware="B200", point="00333af8")
def oracle_forward(inputs):
    (
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        arg5_1,
        arg6_1,
        arg7_1,
        _shape_param_0,
        _shape_param_1,
        _shape_param_2,
        _shape_param_3,
        _shape_param_4,
        _shape_param_5,
        _shape_param_6,
        _shape_param_7,
        _shape_param_8,
        _shape_param_9,
        _shape_param_10,
        _shape_param_11,
        _shape_param_12,
    ) = inputs

    full = torch.empty((BATCH, OUT0_HIDDEN, PADDED_TOKENS, 1), device=arg1_1.device, dtype=torch.float32)
    temp0 = torch.empty((OUT0_ROWS, OUT0_HIDDEN), device=arg1_1.device, dtype=torch.float32)
    out0 = torch.empty((OUT0_ROWS, OUT0_HIDDEN), device=arg1_1.device, dtype=torch.bfloat16)
    out3 = torch.empty((OUT0_ROWS, OUT3_HIDDEN), device=arg1_1.device, dtype=torch.bfloat16)
    out2 = torch.empty((OUT0_HIDDEN,), device=arg1_1.device, dtype=torch.float32)
    out5 = torch.empty((OUT3_HIDDEN,), device=arg1_1.device, dtype=torch.float32)

    zero_block = 256
    _zero_full_and_temp_kernel[(triton.cdiv(max(FULL_NUMEL, OUT0_ROWS * OUT0_HIDDEN), zero_block),)](
        full,
        temp0,
        BLOCK=zero_block,
        num_warps=4,
    )

    _scatter_probabilities_kernel[(PROB_ROWS, triton.cdiv(ARG4_CHANNELS, 32))](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        arg6_1,
        temp0,
        BLOCK_C=32,
        BLOCK_K=16,
        num_warps=8,
    )

    reduce_block_r = 16
    reduce_block_c = 64
    out0_groups = triton.cdiv(OUT0_ROWS, reduce_block_r)
    out0_partials = torch.empty((out0_groups, OUT0_HIDDEN), device=arg1_1.device, dtype=torch.float32)
    _finalize_out0_partials_kernel[(out0_groups, triton.cdiv(OUT0_HIDDEN, reduce_block_c))](
        temp0,
        out0,
        out0_partials,
        BLOCK_R=reduce_block_r,
        BLOCK_C=reduce_block_c,
        num_warps=4,
    )
    _finalize_bf16_sum_kernel[(triton.cdiv(OUT0_HIDDEN, 16),)](
        out0_partials,
        out2,
        NUM_GROUPS=out0_groups,
        HIDDEN=OUT0_HIDDEN,
        GROUP_BLOCK=1 << (out0_groups - 1).bit_length(),
        BLOCK_C=16,
        num_warps=8,
    )

    _store_out3_kernel[(triton.cdiv(PROB_ROWS, 16),)](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg5_1,
        out3,
        BLOCK_R=16,
        BLOCK_K=16,
        num_warps=8,
    )

    out3_groups = triton.cdiv(OUT0_ROWS, reduce_block_r)
    out3_partials = torch.empty((out3_groups, OUT3_HIDDEN), device=arg1_1.device, dtype=torch.float32)
    _reduce_bf16_columns_kernel[(out3_groups, triton.cdiv(OUT3_HIDDEN, reduce_block_c))](
        out3,
        out3_partials,
        ROWS=OUT0_ROWS,
        HIDDEN=OUT3_HIDDEN,
        BLOCK_R=reduce_block_r,
        BLOCK_C=reduce_block_c,
        num_warps=4,
    )
    _finalize_bf16_sum_kernel[(triton.cdiv(OUT3_HIDDEN, 16),)](
        out3_partials,
        out5,
        NUM_GROUPS=out3_groups,
        HIDDEN=OUT3_HIDDEN,
        GROUP_BLOCK=1 << (out3_groups - 1).bit_length(),
        BLOCK_C=16,
        num_warps=8,
    )

    return full, out0, out0.permute(1, 0), out2, out3, out3.permute(1, 0), out5
