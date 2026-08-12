"""cuTile port of sum_aa70ea7e1d52: Demucs GLU-derivative + channel sum.

Produces:
  out = [4, 256, 23210] bf16 = cat([sigmoid*grad, (1-sigmoid)*sigmoid*first_half*grad])
  sum = fp32 [256] sum over dims [0, 2] of out (rounded from bf16 sum).
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 4
C_HALF = 128
C_TOTAL = 256
TIME = 23210
BLOCK_C = 32
BLOCK_T = 256


@ct.kernel
def _produce_kernel(
    grad_ptr,       # (BATCH, C_HALF, TIME) bf16
    gate_pair_ptr,  # (BATCH, C_TOTAL, TIME) bf16
    out_ptr,        # (BATCH, C_TOTAL, TIME) bf16
    partials_ptr,   # (2, BATCH * n_time_tiles, C_HALF) fp32
    N_TIME_TILES: ct.Constant[int],
    BLOCK_C_: ct.Constant[int],
    BLOCK_T_: ct.Constant[int],
):
    batch = ct.bid(0)
    c_block = ct.bid(1)
    t_tile = ct.bid(2)

    # Load first_half from gate_pair[batch, c, t] (channels 0..127).
    # Load gate from gate_pair[batch, c + 128, t] (channels 128..255).
    grad = ct.astype(
        ct.load(
            grad_ptr,
            index=(batch, c_block, t_tile),
            shape=(1, BLOCK_C_, BLOCK_T_),
            padding_mode=ct.PaddingMode.ZERO,
        ),
        ct.float32,
    )
    first_half = ct.astype(
        ct.load(
            gate_pair_ptr,
            index=(batch, c_block, t_tile),
            shape=(1, BLOCK_C_, BLOCK_T_),
            padding_mode=ct.PaddingMode.ZERO,
        ),
        ct.float32,
    )
    # gate lives at channel c_block + (C_HALF / BLOCK_C_).
    gate = ct.astype(
        ct.load(
            gate_pair_ptr,
            index=(batch, c_block + C_HALF // BLOCK_C_, t_tile),
            shape=(1, BLOCK_C_, BLOCK_T_),
            padding_mode=ct.PaddingMode.ZERO,
        ),
        ct.float32,
    )

    sigmoid = 1.0 / (ct.exp(-gate) + 1.0)
    first = sigmoid * grad
    second = (1.0 - sigmoid) * sigmoid * first_half * grad
    first_bf16 = ct.astype(first, ct.bfloat16)
    second_bf16 = ct.astype(second, ct.bfloat16)

    # Mask out invalid time positions
    t_idx = ct.arange(BLOCK_T_, dtype=ct.int32) + t_tile * BLOCK_T_
    valid_t = ct.reshape(t_idx < TIME, (1, 1, BLOCK_T_))
    zero_bf16 = ct.zeros((1, BLOCK_C_, BLOCK_T_), dtype=ct.bfloat16)
    first_bf16 = ct.where(valid_t, first_bf16, zero_bf16)
    second_bf16 = ct.where(valid_t, second_bf16, zero_bf16)

    # Store first_bf16 into out[batch, c_block (first half), t_tile].
    ct.store(out_ptr, index=(batch, c_block, t_tile), tile=first_bf16)
    # Store second_bf16 into out[batch, c_block + C_HALF/BLOCK_C_, t_tile].
    ct.store(out_ptr, index=(batch, c_block + C_HALF // BLOCK_C_, t_tile), tile=second_bf16)

    # Compute partials: sum over time axis (axis=2) then valid channel.
    first_sum = ct.sum(ct.astype(first_bf16, ct.float32), axis=2)  # (1, BLOCK_C_)
    second_sum = ct.sum(ct.astype(second_bf16, ct.float32), axis=2)
    # Reshape to match the (2, N_PARTIALS, C_HALF) 3D array — need a (1, 1, BLOCK_C_) tile
    # since we're storing at index (0/1, partial_index, c_block).
    first_sum_3d = ct.reshape(first_sum, (1, 1, BLOCK_C_))
    second_sum_3d = ct.reshape(second_sum, (1, 1, BLOCK_C_))
    partial_index = batch * N_TIME_TILES + t_tile
    ct.store(partials_ptr, index=(0, partial_index, c_block), tile=first_sum_3d)
    ct.store(partials_ptr, index=(1, partial_index, c_block), tile=second_sum_3d)


@ct.kernel
def _finalize_kernel(
    partials_ptr,  # (2, batch * n_time_tiles, C_HALF) f32
    sum_ptr,       # (C_TOTAL,) f32
    N_PARTIALS: ct.Constant[int],
    BLOCK_P: ct.Constant[int],
    BLOCK_C_: ct.Constant[int],
):
    c_block = ct.bid(0)
    # Sum all partials.
    total_first = ct.zeros((BLOCK_C_,), dtype=ct.float32)
    total_second = ct.zeros((BLOCK_C_,), dtype=ct.float32)
    n_p_tiles = ct.cdiv(N_PARTIALS, BLOCK_P)
    for p_block in range(n_p_tiles):
        first = ct.load(
            partials_ptr,
            index=(0, p_block, c_block),
            shape=(1, BLOCK_P, BLOCK_C_),
            padding_mode=ct.PaddingMode.ZERO,
        )
        second = ct.load(
            partials_ptr,
            index=(1, p_block, c_block),
            shape=(1, BLOCK_P, BLOCK_C_),
            padding_mode=ct.PaddingMode.ZERO,
        )
        p_idx = ct.arange(BLOCK_P, dtype=ct.int32) + p_block * BLOCK_P
        valid = ct.reshape(p_idx < N_PARTIALS, (1, BLOCK_P, 1))
        zero_3d = ct.zeros((1, BLOCK_P, BLOCK_C_), dtype=ct.float32)
        first = ct.where(valid, first, zero_3d)
        second = ct.where(valid, second, zero_3d)
        total_first = total_first + ct.reshape(ct.sum(first, axis=1), (BLOCK_C_,))
        total_second = total_second + ct.reshape(ct.sum(second, axis=1), (BLOCK_C_,))

    # Reference does: sum in bf16, then cast to fp32.
    total_first_bf = ct.astype(ct.astype(total_first, ct.bfloat16), ct.float32)
    total_second_bf = ct.astype(ct.astype(total_second, ct.bfloat16), ct.float32)
    # Store first_bf at [c_block] and second_bf at [c_block + C_HALF/BLOCK_C_].
    ct.store(sum_ptr, index=(c_block,), tile=total_first_bf)
    ct.store(sum_ptr, index=(c_block + C_HALF // BLOCK_C_,), tile=total_second_bf)


@oracle_impl(hardware="B200", point="5bd69c19")
def oracle_forward(inputs):
    grad, gate_pair = inputs
    device = grad.device
    n_time_tiles = (TIME + BLOCK_T - 1) // BLOCK_T
    n_partials = BATCH * n_time_tiles

    out = torch.empty_strided(
        (BATCH, C_TOTAL, TIME),
        (C_TOTAL * TIME, TIME, 1),
        device=device,
        dtype=torch.bfloat16,
    )
    partials = torch.empty(
        (2, n_partials, C_HALF),
        device=device,
        dtype=torch.float32,
    )
    sum_out = torch.empty((C_TOTAL,), device=device, dtype=torch.float32)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (BATCH, C_HALF // BLOCK_C, n_time_tiles),
        _produce_kernel,
        (grad, gate_pair, out, partials, n_time_tiles, BLOCK_C, BLOCK_T),
    )
    # Finalize: 91 partials per channel_half (fits in a single 128-wide block).
    BLOCK_P = 128
    ct.launch(
        stream,
        (C_HALF // BLOCK_C, 1, 1),
        _finalize_kernel,
        (partials, sum_out, n_partials, BLOCK_P, BLOCK_C),
    )
    return out, sum_out
