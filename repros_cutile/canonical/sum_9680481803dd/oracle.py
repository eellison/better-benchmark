"""cuTile port of sum_9680481803dd: Demucs GLU add + cat + channel sum.

Two kernels matching Triton's produce_partials + finalize structure.
Kernel 1: compute f32 add + bf16-rounded add (eager), sigmoid GLU derivative,
emit bf16 cat output halves, per-tile partial sums using the bf16-rounded add
values (matching the Inductor "eager add boundary" for the reduction).
Kernel 2: reduce partials, round to bf16, store as f32.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 4
C_HALF = 64
C_TOTAL = 128
TIME = 95696
BLOCK_C = 16
BLOCK_T = 256
FINAL_BLOCK_C = 16


def _cdiv(a, b):
    return (a + b - 1) // b


def _next_pow2(v):
    return 1 << (v - 1).bit_length()


TILES_T = _cdiv(TIME, BLOCK_T)
NUM_PARTIALS = BATCH * TILES_T
FINAL_BLOCK = _next_pow2(NUM_PARTIALS)


@ct.kernel
def _produce_partials_kernel(
    add_lhs_ptr,     # bf16 [BATCH, C_HALF, TIME]
    add_rhs_ptr,     # bf16 [BATCH, C_HALF, TIME]
    gate_pair_ptr,   # bf16 [BATCH, C_TOTAL, TIME]
    out_first_ptr,   # bf16 [BATCH, C_HALF, TIME]  (view)
    out_second_ptr,  # bf16 [BATCH, C_HALF, TIME]  (view)
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

    lhs = ct.load(
        add_lhs_ptr, index=(batch, c_block, t_tile),
        shape=(1, BLOCK_C_, BLOCK_T_),
        padding_mode=ct.PaddingMode.ZERO,
    )
    rhs = ct.load(
        add_rhs_ptr, index=(batch, c_block, t_tile),
        shape=(1, BLOCK_C_, BLOCK_T_),
        padding_mode=ct.PaddingMode.ZERO,
    )
    lhs_f = ct.astype(ct.reshape(lhs, (BLOCK_C_, BLOCK_T_)), ct.float32)
    rhs_f = ct.astype(ct.reshape(rhs, (BLOCK_C_, BLOCK_T_)), ct.float32)
    add_compiled = lhs_f + rhs_f
    add_eager = ct.astype(ct.astype(add_compiled, ct.bfloat16), ct.float32)

    first_input = ct.load(
        gate_pair_ptr, index=(batch, c_block, t_tile),
        shape=(1, BLOCK_C_, BLOCK_T_),
        padding_mode=ct.PaddingMode.ZERO,
    )
    first_input_f = ct.astype(ct.reshape(first_input, (BLOCK_C_, BLOCK_T_)), ct.float32)

    gate_c_offset = C_HALF_ // BLOCK_C_
    gate = ct.load(
        gate_pair_ptr, index=(batch, c_block + gate_c_offset, t_tile),
        shape=(1, BLOCK_C_, BLOCK_T_),
        padding_mode=ct.PaddingMode.ZERO,
    )
    gate_f = ct.astype(ct.reshape(gate, (BLOCK_C_, BLOCK_T_)), ct.float32)

    sigmoid = 1.0 / (ct.exp(-gate_f) + 1.0)
    gate_deriv = (1.0 - sigmoid) * sigmoid * first_input_f

    first = sigmoid * add_compiled
    second = gate_deriv * add_compiled
    first_bf16 = ct.astype(first, ct.bfloat16)
    second_bf16 = ct.astype(second, ct.bfloat16)

    ct.store(out_first_ptr, index=(batch, c_block, t_tile),
             tile=ct.reshape(first_bf16, (1, BLOCK_C_, BLOCK_T_)))
    ct.store(out_second_ptr, index=(batch, c_block, t_tile),
             tile=ct.reshape(second_bf16, (1, BLOCK_C_, BLOCK_T_)))

    first_for_sum = ct.astype(sigmoid * add_eager, ct.bfloat16)
    second_for_sum = ct.astype(gate_deriv * add_eager, ct.bfloat16)

    channels = c_block * BLOCK_C_ + ct.arange(BLOCK_C_, dtype=ct.int32)
    times = t_tile * BLOCK_T_ + ct.arange(BLOCK_T_, dtype=ct.int32)
    c_valid = ct.reshape(channels < C_HALF_, (BLOCK_C_, 1))
    t_valid = ct.reshape(times < TIME_, (1, BLOCK_T_))
    valid = c_valid & t_valid

    first_sum_f = ct.astype(first_for_sum, ct.float32)
    second_sum_f = ct.astype(second_for_sum, ct.float32)
    first_masked = ct.where(valid, first_sum_f, 0.0)
    second_masked = ct.where(valid, second_sum_f, 0.0)
    first_partial = ct.sum(first_masked, axis=1)
    second_partial = ct.sum(second_masked, axis=1)

    partial_row = batch * TILES_T_ + t_tile
    ct.store(partials_first_ptr, index=(partial_row, c_block),
             tile=ct.reshape(first_partial, (1, BLOCK_C_)))
    ct.store(partials_second_ptr, index=(partial_row, c_block),
             tile=ct.reshape(second_partial, (1, BLOCK_C_)))


@ct.kernel
def _finalize_sum_kernel(
    partials_first_ptr,
    partials_second_ptr,
    sum_first_ptr,
    sum_second_ptr,
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

    first_rounded = ct.astype(ct.astype(first_sum, ct.bfloat16), ct.float32)
    second_rounded = ct.astype(ct.astype(second_sum, ct.bfloat16), ct.float32)

    ct.store(sum_first_ptr, index=(c_block,), tile=first_rounded)
    ct.store(sum_second_ptr, index=(c_block,), tile=second_rounded)


@oracle_impl(hardware="B200", point="b453da2f", BLOCK_C=BLOCK_C, BLOCK_T=BLOCK_T,
             FINAL_BLOCK_C=FINAL_BLOCK_C)
def oracle_forward(inputs, *, BLOCK_C: int, BLOCK_T: int, FINAL_BLOCK_C: int):
    add_lhs, add_rhs, gate_pair = inputs
    device = add_lhs.device

    out = torch.empty_strided(
        (BATCH, C_TOTAL, TIME),
        (C_TOTAL * TIME, TIME, 1),
        device=device,
        dtype=torch.bfloat16,
    )
    out_first = out[:, :C_HALF, :]
    out_second = out[:, C_HALF:, :]

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
        (add_lhs, add_rhs, gate_pair, out_first, out_second,
         partials_first, partials_second,
         C_HALF, TIME, TILES_T, BLOCK_C, BLOCK_T),
    )
    ct.launch(
        stream,
        (C_HALF // FINAL_BLOCK_C, 1, 1),
        _finalize_sum_kernel,
        (partials_first, partials_second, sum_first, sum_second,
         C_HALF, NUM_PARTIALS, FINAL_BLOCK_C, FINAL_BLOCK),
    )
    return out, sum_out
