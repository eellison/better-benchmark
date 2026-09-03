"""cuTile port of sum_39ea00f83135: Demucs GLU-derivative cat + channel sum.

Two kernels matching Triton's produce_partials + finalize structure.
Kernel 1: emit bf16 cat output halves and per-tile partial sums along axis T.
Kernel 2: reduce partials along tile axis, round to bf16, store as f32.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 4
C_HALF = 256
C_TOTAL = 512
TIME = 5802
BLOCK_C = 16
BLOCK_T = 256


def _cdiv(a, b):
    return (a + b - 1) // b


def _next_pow2(v):
    return 1 << (v - 1).bit_length()


TILES_T = _cdiv(TIME, BLOCK_T)
NUM_PARTIALS = BATCH * TILES_T
FINAL_BLOCK = _next_pow2(NUM_PARTIALS)


@ct.kernel
def _produce_partials_kernel(
    grad_ptr,        # bf16 [BATCH, C_HALF, TIME]
    gate_pair_ptr,   # bf16 [BATCH, C_TOTAL, TIME]
    out_first_ptr,   # bf16 [BATCH, C_HALF, TIME]  (view of out[:, :C_HALF, :])
    out_second_ptr,  # bf16 [BATCH, C_HALF, TIME]  (view of out[:, C_HALF:, :])
    partials_first_ptr,   # f32 [NUM_PARTIALS, C_HALF]
    partials_second_ptr,  # f32 [NUM_PARTIALS, C_HALF]
    C_HALF_: ct.Constant[int],
    TIME_: ct.Constant[int],
    TILES_T_: ct.Constant[int],
    BLOCK_C_: ct.Constant[int],
    BLOCK_T_: ct.Constant[int],
):
    batch = ct.bid(0)
    c_block = ct.bid(1)
    t_tile = ct.bid(2)

    # Load a (BLOCK_C, BLOCK_T) tile
    grad = ct.load(
        grad_ptr, index=(batch, c_block, t_tile),
        shape=(1, BLOCK_C_, BLOCK_T_),
        padding_mode=ct.PaddingMode.ZERO,
    )
    grad_f = ct.astype(ct.reshape(grad, (BLOCK_C_, BLOCK_T_)), ct.float32)

    first_half = ct.load(
        gate_pair_ptr, index=(batch, c_block, t_tile),
        shape=(1, BLOCK_C_, BLOCK_T_),
        padding_mode=ct.PaddingMode.ZERO,
    )
    first_half_f = ct.astype(ct.reshape(first_half, (BLOCK_C_, BLOCK_T_)), ct.float32)

    # gate is at c + C_HALF; since gate_pair is (B, C_TOTAL, T), tile-space
    # index (batch, c_block + C_HALF//BLOCK_C, t_tile) picks the gate half
    # (BLOCK_C must divide C_HALF, which it does: 16 | 256).
    gate_c_offset = C_HALF_ // BLOCK_C_
    gate = ct.load(
        gate_pair_ptr, index=(batch, c_block + gate_c_offset, t_tile),
        shape=(1, BLOCK_C_, BLOCK_T_),
        padding_mode=ct.PaddingMode.ZERO,
    )
    gate_f = ct.astype(ct.reshape(gate, (BLOCK_C_, BLOCK_T_)), ct.float32)

    sigmoid = 1.0 / (ct.exp(-gate_f) + 1.0)
    first = sigmoid * grad_f
    second = (1.0 - sigmoid) * sigmoid * first_half_f * grad_f
    first_bf16 = ct.astype(first, ct.bfloat16)
    second_bf16 = ct.astype(second, ct.bfloat16)

    # Store the two halves of the cat output
    ct.store(out_first_ptr, index=(batch, c_block, t_tile),
             tile=ct.reshape(first_bf16, (1, BLOCK_C_, BLOCK_T_)))
    ct.store(out_second_ptr, index=(batch, c_block, t_tile),
             tile=ct.reshape(second_bf16, (1, BLOCK_C_, BLOCK_T_)))

    # Mask & partial reductions along axis=T
    channels = c_block * BLOCK_C_ + ct.arange(BLOCK_C_, dtype=ct.int32)
    times = t_tile * BLOCK_T_ + ct.arange(BLOCK_T_, dtype=ct.int32)
    c_valid = ct.reshape(channels < C_HALF_, (BLOCK_C_, 1))
    t_valid = ct.reshape(times < TIME_, (1, BLOCK_T_))
    valid = c_valid & t_valid

    first_f = ct.astype(first_bf16, ct.float32)
    second_f = ct.astype(second_bf16, ct.float32)
    first_masked = ct.where(valid, first_f, 0.0)
    second_masked = ct.where(valid, second_f, 0.0)
    first_partial = ct.sum(first_masked, axis=1)   # shape (BLOCK_C,)
    second_partial = ct.sum(second_masked, axis=1)

    # Store partial to (NUM_PARTIALS, C_HALF): row = batch*TILES_T + t_tile, col = c_block*BLOCK_C..
    partial_row = batch * TILES_T_ + t_tile
    ct.store(partials_first_ptr, index=(partial_row, c_block),
             tile=ct.reshape(first_partial, (1, BLOCK_C_)))
    ct.store(partials_second_ptr, index=(partial_row, c_block),
             tile=ct.reshape(second_partial, (1, BLOCK_C_)))


@ct.kernel
def _finalize_sum_kernel(
    partials_first_ptr,   # f32 [NUM_PARTIALS, C_HALF]
    partials_second_ptr,  # f32 [NUM_PARTIALS, C_HALF]
    sum_first_ptr,        # f32 [C_HALF]
    sum_second_ptr,       # f32 [C_HALF]
    C_HALF_: ct.Constant[int],
    NUM_PARTIALS_: ct.Constant[int],
    BLOCK_C_: ct.Constant[int],
    BLOCK_P_: ct.Constant[int],
):
    c_block = ct.bid(0)

    first = ct.load(
        partials_first_ptr, index=(0, c_block),
        shape=(BLOCK_P_, BLOCK_C_),
        padding_mode=ct.PaddingMode.ZERO,
    )
    second = ct.load(
        partials_second_ptr, index=(0, c_block),
        shape=(BLOCK_P_, BLOCK_C_),
        padding_mode=ct.PaddingMode.ZERO,
    )
    first_f = ct.astype(first, ct.float32)
    second_f = ct.astype(second, ct.float32)

    partials = ct.arange(BLOCK_P_, dtype=ct.int32)
    channels = c_block * BLOCK_C_ + ct.arange(BLOCK_C_, dtype=ct.int32)
    p_valid = ct.reshape(partials < NUM_PARTIALS_, (BLOCK_P_, 1))
    c_valid = ct.reshape(channels < C_HALF_, (1, BLOCK_C_))
    valid = p_valid & c_valid

    first_masked = ct.where(valid, first_f, 0.0)
    second_masked = ct.where(valid, second_f, 0.0)
    first_sum = ct.sum(first_masked, axis=0)
    second_sum = ct.sum(second_masked, axis=0)

    # Round to bf16 then back to f32 (matches Triton's final rounding)
    first_rounded = ct.astype(ct.astype(first_sum, ct.bfloat16), ct.float32)
    second_rounded = ct.astype(ct.astype(second_sum, ct.bfloat16), ct.float32)

    ct.store(sum_first_ptr, index=(c_block,), tile=first_rounded)
    ct.store(sum_second_ptr, index=(c_block,), tile=second_rounded)


@oracle_impl(hardware="B200", point="d44b8225", BLOCK_C=BLOCK_C, BLOCK_T=BLOCK_T)
def oracle_forward(inputs, *, BLOCK_C: int, BLOCK_T: int):
    grad, gate_pair = inputs
    device = grad.device

    out = torch.empty_strided(
        (BATCH, C_TOTAL, TIME),
        (C_TOTAL * TIME, TIME, 1),
        device=device,
        dtype=torch.bfloat16,
    )
    # Views for storing each half via cuTile's 3D indexing
    out_first = out[:, :C_HALF, :]   # bf16 [B, C_HALF, T], strided (C_TOTAL*T, T, 1)
    out_second = out[:, C_HALF:, :]  # same

    # Partials: 2 arrays, each f32 [NUM_PARTIALS, C_HALF]
    partials_first = torch.empty((NUM_PARTIALS, C_HALF), device=device, dtype=torch.float32)
    partials_second = torch.empty((NUM_PARTIALS, C_HALF), device=device, dtype=torch.float32)

    sum_out = torch.empty_strided((C_TOTAL,), (1,), device=device, dtype=torch.float32)
    sum_first = sum_out[:C_HALF]
    sum_second = sum_out[C_HALF:]

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (BATCH, C_HALF // BLOCK_C, TILES_T),
        _produce_partials_kernel,
        (grad, gate_pair, out_first, out_second, partials_first, partials_second,
         C_HALF, TIME, TILES_T, BLOCK_C, BLOCK_T),
    )
    ct.launch(
        stream,
        (C_HALF // BLOCK_C, 1, 1),
        _finalize_sum_kernel,
        (partials_first, partials_second, sum_first, sum_second,
         C_HALF, NUM_PARTIALS, BLOCK_C, FINAL_BLOCK),
    )
    return out, sum_out
