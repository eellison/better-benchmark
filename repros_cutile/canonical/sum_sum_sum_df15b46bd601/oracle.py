"""cuTile port of sum_sum_sum_df15b46bd601: GPT-J QA one-hot correction.

Single-block cuTile kernel that consumes the two-branch [1,128] logits
sequence, computes the one-hot correction per branch (bf16 round-trip on
the centered/exp/mul boundary), materializes the [128,2] compact and
[128,8] zero-padded outputs, and emits the bf16-rounded [2] column sum.
Torch precomputes scalars (count reductions, half_grad) to keep the
kernel entirely per-column tile operations.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


N_COLS = 128


@ct.kernel
def _one_hot_correction_kernel(
    scale0_ptr,      # f32 [1]  half_grad / count0
    valid0_ptr,      # b8  [1]
    idx0_ptr,        # i64 [1]  clamped index
    logits0_ptr,     # bf16 [128]
    shift0_ptr,      # f32 [1]
    shift1_ptr,      # f32 [1]
    base0_ptr,       # bf16 [128] (arg6_1)
    scale1_ptr,      # f32 [1]
    valid1_ptr,      # b8  [1]
    idx1_ptr,        # i64 [1]
    logits1_ptr,     # bf16 [128]
    shift2_ptr,      # f32 [1]
    shift3_ptr,      # f32 [1]
    base1_ptr,       # bf16 [128] (arg12_1)
    padded_ptr,      # bf16 [128, 8]
    compact_ptr,     # bf16 [128, 2]
    sum_ptr,         # f32 [2]
    BLOCK_N: ct.Constant[int],
):
    cols_i32 = ct.arange(BLOCK_N, dtype=ct.int32)
    cols_i64 = ct.astype(cols_i32, ct.int64)

    # Load scalars broadcast to (BLOCK_N,).
    scale0_1 = ct.load(scale0_ptr, index=(0,), shape=(1,))
    valid0_1 = ct.load(valid0_ptr, index=(0,), shape=(1,))
    idx0_1 = ct.load(idx0_ptr, index=(0,), shape=(1,))
    shift0_1 = ct.load(shift0_ptr, index=(0,), shape=(1,))
    shift1_1 = ct.load(shift1_ptr, index=(0,), shape=(1,))
    scale1_1 = ct.load(scale1_ptr, index=(0,), shape=(1,))
    valid1_1 = ct.load(valid1_ptr, index=(0,), shape=(1,))
    idx1_1 = ct.load(idx1_ptr, index=(0,), shape=(1,))
    shift2_1 = ct.load(shift2_ptr, index=(0,), shape=(1,))
    shift3_1 = ct.load(shift3_ptr, index=(0,), shape=(1,))

    scale0 = ct.broadcast_to(scale0_1, (BLOCK_N,))
    scale1 = ct.broadcast_to(scale1_1, (BLOCK_N,))
    valid0 = ct.broadcast_to(valid0_1, (BLOCK_N,))
    valid1 = ct.broadcast_to(valid1_1, (BLOCK_N,))
    idx0_b = ct.broadcast_to(idx0_1, (BLOCK_N,))
    idx1_b = ct.broadcast_to(idx1_1, (BLOCK_N,))
    shift0 = ct.broadcast_to(shift0_1, (BLOCK_N,))
    shift1 = ct.broadcast_to(shift1_1, (BLOCK_N,))
    shift2 = ct.broadcast_to(shift2_1, (BLOCK_N,))
    shift3 = ct.broadcast_to(shift3_1, (BLOCK_N,))

    # Branch 0
    selected0 = idx0_b == cols_i64  # (BLOCK_N,) b8
    neg_ones = ct.full((BLOCK_N,), -1.0, dtype=ct.float32)
    zeros_f = ct.full((BLOCK_N,), 0.0, dtype=ct.float32)
    onehot0 = ct.where(selected0, neg_ones, zeros_f)
    where2_0 = ct.where(valid0, scale0, zeros_f)
    indexed0_f = onehot0 * where2_0
    indexed0 = ct.astype(ct.astype(indexed0_f, ct.bfloat16), ct.float32)

    logits0_bf = ct.load(logits0_ptr, index=(0,), shape=(BLOCK_N,))
    centered0 = ct.astype(logits0_bf, ct.float32) - shift0
    centered0 = ct.astype(ct.astype(centered0 - shift1, ct.bfloat16), ct.float32)
    exp0 = ct.exp(centered0)
    sum0_scalar = ct.sum(indexed0)
    sum0_b = ct.broadcast_to(ct.reshape(sum0_scalar, (1,)), (BLOCK_N,))
    update0_bf = ct.astype(indexed0 - exp0 * sum0_b, ct.bfloat16)

    base0_bf = ct.load(base0_ptr, index=(0,), shape=(BLOCK_N,))
    row1 = ct.astype(
        ct.astype(base0_bf, ct.float32) + ct.astype(update0_bf, ct.float32),
        ct.bfloat16,
    )

    # Branch 1
    selected1 = idx1_b == cols_i64
    onehot1 = ct.where(selected1, neg_ones, zeros_f)
    where2_1 = ct.where(valid1, scale1, zeros_f)
    indexed1_f = onehot1 * where2_1
    indexed1 = ct.astype(ct.astype(indexed1_f, ct.bfloat16), ct.float32)

    logits1_bf = ct.load(logits1_ptr, index=(0,), shape=(BLOCK_N,))
    centered1 = ct.astype(logits1_bf, ct.float32) - shift2
    centered1 = ct.astype(ct.astype(centered1 - shift3, ct.bfloat16), ct.float32)
    exp1 = ct.exp(centered1)
    sum1_scalar = ct.sum(indexed1)
    sum1_b = ct.broadcast_to(ct.reshape(sum1_scalar, (1,)), (BLOCK_N,))
    update1_bf = ct.astype(indexed1 - exp1 * sum1_b, ct.bfloat16)

    base1_bf = ct.load(base1_ptr, index=(0,), shape=(BLOCK_N,))
    row0 = ct.astype(
        ct.astype(base1_bf, ct.float32) + ct.astype(update1_bf, ct.float32),
        ct.bfloat16,
    )

    # Compact [128, 2]: col 0 = row0, col 1 = row1.
    row0_2d = ct.expand_dims(row0, axis=1)  # (128, 1)
    row1_2d = ct.expand_dims(row1, axis=1)  # (128, 1)
    compact_tile = ct.cat((row0_2d, row1_2d), axis=1)  # (128, 2)
    ct.store(compact_ptr, index=(0, 0), tile=compact_tile)

    # Padded [128, 8]: cols 0/1 = row0/row1, cols 2..7 = 0.
    zero_bf_128_2 = ct.full((128, 2), 0.0, dtype=ct.bfloat16)
    ct.store(padded_ptr, index=(0, 0), tile=compact_tile)
    ct.store(padded_ptr, index=(0, 1), tile=zero_bf_128_2)
    ct.store(padded_ptr, index=(0, 2), tile=zero_bf_128_2)
    ct.store(padded_ptr, index=(0, 3), tile=zero_bf_128_2)

    # Column sums: bf16 round-trip.
    sum_row0_f = ct.sum(ct.astype(row0, ct.float32))
    sum_row1_f = ct.sum(ct.astype(row1, ct.float32))
    sum_row0_bf = ct.astype(sum_row0_f, ct.bfloat16)
    sum_row1_bf = ct.astype(sum_row1_f, ct.bfloat16)
    sum_row0_out = ct.astype(sum_row0_bf, ct.float32)
    sum_row1_out = ct.astype(sum_row1_bf, ct.float32)
    sum_tile_1a = ct.reshape(sum_row0_out, (1,))
    sum_tile_1b = ct.reshape(sum_row1_out, (1,))
    sum_tile_2 = ct.cat((sum_tile_1a, sum_tile_1b), axis=0)  # (2,)
    ct.store(sum_ptr, index=(0,), tile=sum_tile_2)


@oracle_impl(hardware="B200", point="3e32ddcf", BLOCK_N=128)
def oracle_forward(inputs, *, BLOCK_N: int):
    (
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1,
        arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1,
        *_shape,
    ) = inputs
    device = arg0_1.device

    # Precompute scalars (all tiny/scalar operations).
    half_grad = arg0_1 / 2.0
    count0 = arg1_1.sum().to(torch.float32)
    count1 = arg7_1.sum().to(torch.float32)
    scale0 = (half_grad / count0).contiguous().view(1)
    scale1 = (half_grad / count1).contiguous().view(1)
    idx0_clamped = torch.clamp(arg2_1, 0, 128).view(1)
    idx1_clamped = torch.clamp(arg8_1, 0, 128).view(1)
    valid0 = (idx0_clamped != 128).view(1)
    valid1 = (idx1_clamped != 128).view(1)

    logits0 = arg3_1.contiguous().view(N_COLS)
    logits1 = arg9_1.contiguous().view(N_COLS)
    shift0 = arg4_1.contiguous().view(1)
    shift1 = arg5_1.contiguous().view(1)
    shift2 = arg10_1.contiguous().view(1)
    shift3 = arg11_1.contiguous().view(1)
    base0 = arg6_1.contiguous().view(N_COLS)  # produces row1 upstream? no, row1 = arg6 + update0
    base1 = arg12_1.contiguous().view(N_COLS)

    padded = torch.empty((N_COLS, 8), device=device, dtype=torch.bfloat16)
    compact = torch.empty((N_COLS, 2), device=device, dtype=torch.bfloat16)
    out_sum = torch.empty((2,), device=device, dtype=torch.float32)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (1, 1, 1),
        _one_hot_correction_kernel,
        (scale0, valid0, idx0_clamped, logits0, shift0, shift1, base0,
         scale1, valid1, idx1_clamped, logits1, shift2, shift3, base1,
         padded, compact, out_sum,
         BLOCK_N),
    )

    return padded, compact.permute(1, 0), out_sum
