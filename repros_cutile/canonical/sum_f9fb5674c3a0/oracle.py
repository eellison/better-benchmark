"""cuTile port of sum_f9fb5674c3a0: Demucs GLU + channel sum.

Computes:
  add = bf16(arg0 + arg1) (fp32 intermediate)
  out[b, c, t]     = bf16(sigmoid(arg2[b, c+HALF, t]) * add_f32[b, c, t])  for c in 0..HALF-1
  out[b, c+HALF,t] = bf16((1-sig)*sig*arg2[b, c, t]*add_f32)                for c in 0..HALF-1
  sum = sum(out, dim=[0,2]) in bf16 -> f32
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 4
HALF_C = 256
OUT_C = 512
T = 5979
BLOCK_C = 32
BLOCK_T = 256


@ct.kernel
def _producer_kernel(
    add_lhs_ptr,    # (BATCH, HALF_C, T) bf16
    add_rhs_ptr,    # (BATCH, HALF_C, T) bf16
    gate_src_ptr,   # (BATCH, OUT_C, T) bf16
    out_ptr,        # (BATCH, OUT_C, T) bf16
    partial_ptr,    # (BATCH * n_time_tiles, OUT_C) fp32
    N_TIME_TILES: ct.Constant[int],
    BLOCK_C_: ct.Constant[int],
    BLOCK_T_: ct.Constant[int],
):
    batch = ct.bid(0)
    c_block = ct.bid(1)
    t_tile = ct.bid(2)
    lhs = ct.load(
        add_lhs_ptr, index=(batch, c_block, t_tile),
        shape=(1, BLOCK_C_, BLOCK_T_), padding_mode=ct.PaddingMode.ZERO,
    )
    rhs = ct.load(
        add_rhs_ptr, index=(batch, c_block, t_tile),
        shape=(1, BLOCK_C_, BLOCK_T_), padding_mode=ct.PaddingMode.ZERO,
    )
    # Match reference: bf16 add first, then cast to f32.
    added_bf16 = lhs + rhs
    added = ct.astype(added_bf16, ct.float32)
    # gate_src has channels 0..OUT_C-1. x_value = gate_src[..., 0..HALF-1] = c_block
    # gate = gate_src[..., HALF..OUT-1] = c_block + HALF_C/BLOCK_C_
    x_value = ct.astype(
        ct.load(gate_src_ptr, index=(batch, c_block, t_tile),
                shape=(1, BLOCK_C_, BLOCK_T_), padding_mode=ct.PaddingMode.ZERO),
        ct.float32,
    )
    gate = ct.astype(
        ct.load(gate_src_ptr, index=(batch, c_block + HALF_C // BLOCK_C_, t_tile),
                shape=(1, BLOCK_C_, BLOCK_T_), padding_mode=ct.PaddingMode.ZERO),
        ct.float32,
    )
    sigmoid = 1.0 / (ct.exp(-gate) + 1.0)
    first_half = sigmoid * added
    second_half = (1.0 - sigmoid) * sigmoid * x_value * added
    first_bf16 = ct.astype(first_half, ct.bfloat16)
    second_bf16 = ct.astype(second_half, ct.bfloat16)

    t_idx = ct.arange(BLOCK_T_, dtype=ct.int32) + t_tile * BLOCK_T_
    valid_t = ct.reshape(t_idx < T, (1, 1, BLOCK_T_))
    zero_bf16 = ct.zeros((1, BLOCK_C_, BLOCK_T_), dtype=ct.bfloat16)
    first_bf16 = ct.where(valid_t, first_bf16, zero_bf16)
    second_bf16 = ct.where(valid_t, second_bf16, zero_bf16)

    ct.store(out_ptr, index=(batch, c_block, t_tile), tile=first_bf16)
    ct.store(out_ptr, index=(batch, c_block + HALF_C // BLOCK_C_, t_tile), tile=second_bf16)

    # Partial sums over time axis.
    first_sum = ct.sum(ct.astype(first_bf16, ct.float32), axis=2)  # (1, BLOCK_C_)
    second_sum = ct.sum(ct.astype(second_bf16, ct.float32), axis=2)

    # Store partials at index (batch*N_TIME_TILES + t_tile, c_block/c_block+HALF).
    partial_index = batch * N_TIME_TILES + t_tile
    ct.store(partial_ptr, index=(partial_index, c_block), tile=first_sum)
    ct.store(partial_ptr, index=(partial_index, c_block + HALF_C // BLOCK_C_), tile=second_sum)


@ct.kernel
def _finalize_kernel(
    partial_ptr,   # (n_partials, OUT_C) f32
    sum_ptr,       # (OUT_C,) f32
    N_PARTIALS: ct.Constant[int],
    BLOCK_P: ct.Constant[int],
    BLOCK_C_: ct.Constant[int],
):
    c_block = ct.bid(0)
    total = ct.zeros((BLOCK_C_,), dtype=ct.float32)
    n_p_tiles = ct.cdiv(N_PARTIALS, BLOCK_P)
    for p_block in range(n_p_tiles):
        chunk = ct.load(
            partial_ptr,
            index=(p_block, c_block),
            shape=(BLOCK_P, BLOCK_C_),
            padding_mode=ct.PaddingMode.ZERO,
        )
        p_idx = ct.arange(BLOCK_P, dtype=ct.int32) + p_block * BLOCK_P
        valid = ct.reshape(p_idx < N_PARTIALS, (BLOCK_P, 1))
        zero_2d = ct.zeros((BLOCK_P, BLOCK_C_), dtype=ct.float32)
        chunk = ct.where(valid, chunk, zero_2d)
        total = total + ct.reshape(ct.sum(chunk, axis=0), (BLOCK_C_,))
    # Reference sums in bf16, then casts to f32.
    total_bf = ct.astype(ct.astype(total, ct.bfloat16), ct.float32)
    ct.store(sum_ptr, index=(c_block,), tile=total_bf)


@oracle_impl(hardware="B200", point="0c6e91db")
def oracle_forward(inputs):
    arg0_1, arg1_1, arg2_1 = inputs
    device = arg0_1.device
    n_time_tiles = (T + BLOCK_T - 1) // BLOCK_T
    n_partials = BATCH * n_time_tiles

    out = torch.empty_strided(
        (BATCH, OUT_C, T),
        (OUT_C * T, T, 1),
        device=device,
        dtype=torch.bfloat16,
    )
    partial = torch.empty(
        (n_partials, OUT_C),
        device=device,
        dtype=torch.float32,
    )
    out_sum = torch.empty((OUT_C,), device=device, dtype=torch.float32)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (BATCH, HALF_C // BLOCK_C, n_time_tiles),
        _producer_kernel,
        (arg0_1, arg1_1, arg2_1, out, partial, n_time_tiles, BLOCK_C, BLOCK_T),
    )
    # 4 * 24 = 96 partials, so pow2 = 128
    BLOCK_P = 128
    ct.launch(
        stream,
        (OUT_C // BLOCK_C, 1, 1),
        _finalize_kernel,
        (partial, out_sum, n_partials, BLOCK_P, BLOCK_C),
    )
    return out, out_sum
