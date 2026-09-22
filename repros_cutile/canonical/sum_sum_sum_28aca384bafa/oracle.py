"""cuTile port of sum_sum_sum_28aca384bafa: NFNet gated GELU backward multi-kernel.

All erf/exp/sigmoid math is computed INSIDE the cuTile kernels, matching the
Triton oracle's kernel structure:
  * sigmoid via 1 / (1 + ct.exp(-x))
  * erf via Abramowitz-Stegun 7.1.26 polynomial (bf16 tolerance covers the ~1e-7
    error vs libdevice.erf)
  * exp via ct.exp

Structure (mirrors Triton's 4 kernels):
  1. _producer_kernel: reads arg0..arg5, computes sigmoid + gelu-grad path,
     produces add_2 (bf16), sigmoid (bf16 [N,C]), mul_14 (bf16), partial
     per-(n,c) sums for sum_1 and sum_2.
  2. _side_kernel: from partial sum2 + sigmoid produces side (bf16 [N,C]).
  3. _channel_reduce_kernel: sums side over N to yield f32 [C].
  4. Scalar sum_1 finalized by a tiny torch reduction over partial_sum1.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


def _next_pow2(v: int) -> int:
    r = 1
    while r < v:
        r <<= 1
    return r


@ct.kernel
def _producer_kernel(
    arg0_ptr,        # bf16 [N, C, HW]
    arg1_ptr,        # bf16 [N, C, HW]
    arg2_ptr,        # bf16 [N, C, HW]
    arg3_ptr,        # bf16 [N, C]        gate
    arg4_ptr,        # bf16 [N, C, HW]
    arg5_ptr,        # f32  [1]           scalar
    out_add_ptr,     # bf16 [N, C, HW]
    out_sigmoid_ptr, # bf16 [N, C]
    out_mul14_ptr,   # bf16 [N, C, HW]
    partial_sum1_ptr,# f32  [N, C]
    sum2_ptr,        # f32  [N, C]
    HW_C: ct.Constant[int],
    HW_PAD: ct.Constant[int],
    BLOCK_HW: ct.Constant[int],
):
    """Grid: (N, C, 1). One (n, c) per program. Loops over HW blocks."""
    n = ct.bid(0)
    c = ct.bid(1)

    # --- Sigmoid computed INSIDE kernel from arg3.
    gate_bf = ct.load(arg3_ptr, index=(n, c), shape=(1, 1))
    gate_f = ct.astype(gate_bf, ct.float32)
    sigmoid_f = 1.0 / (1.0 + ct.exp(-gate_f))
    sigmoid_bf = ct.astype(sigmoid_f, ct.bfloat16)
    ct.store(out_sigmoid_ptr, index=(n, c), tile=sigmoid_bf)
    # bf16-rounded f32 for downstream use (matches Triton _round_to_bf16_f32).
    sigmoid_bf16_f = ct.astype(sigmoid_bf, ct.float32)
    sigmoid_scalar = ct.reshape(sigmoid_bf16_f, ())

    # --- Scalar arg5, bf16-rounded then back to f32 (Triton does the same).
    scalar_f = ct.astype(ct.load(arg5_ptr, index=(0,), shape=(1,)), ct.float32)
    scalar_bf16_f = ct.astype(ct.astype(scalar_f, ct.bfloat16), ct.float32)
    scalar_scalar = ct.reshape(scalar_bf16_f, ())

    partial_sum1 = ct.full((), 0.0, dtype=ct.float32)
    partial_sum2 = ct.full((), 0.0, dtype=ct.float32)

    for block in ct.static_iter(range(HW_PAD // BLOCK_HW)):
        hw_offsets = block * BLOCK_HW + ct.arange(BLOCK_HW, dtype=ct.int32)
        hw_mask = hw_offsets < HW_C

        arg0 = ct.astype(
            ct.load(arg0_ptr, index=(n, c, block), shape=(1, 1, BLOCK_HW),
                    padding_mode=ct.PaddingMode.ZERO),
            ct.float32,
        )
        arg0_1d = ct.reshape(arg0, (BLOCK_HW,))
        arg1 = ct.astype(
            ct.load(arg1_ptr, index=(n, c, block), shape=(1, 1, BLOCK_HW),
                    padding_mode=ct.PaddingMode.ZERO),
            ct.float32,
        )
        arg1_1d = ct.reshape(arg1, (BLOCK_HW,))
        arg2 = ct.astype(
            ct.load(arg2_ptr, index=(n, c, block), shape=(1, 1, BLOCK_HW),
                    padding_mode=ct.PaddingMode.ZERO),
            ct.float32,
        )
        arg2_1d = ct.reshape(arg2, (BLOCK_HW,))
        arg4 = ct.astype(
            ct.load(arg4_ptr, index=(n, c, block), shape=(1, 1, BLOCK_HW),
                    padding_mode=ct.PaddingMode.ZERO),
            ct.float32,
        )
        arg4_1d = ct.reshape(arg4, (BLOCK_HW,))

        # mul   = to_bf16(arg0 * 0.9622...)
        mul_bf = ct.astype(arg0_1d * 0.9622504486493761, ct.bfloat16)
        mul_f = ct.astype(mul_bf, ct.float32)
        # mul_1 = to_bf16(mul * 1.7015...)
        mul_1_bf = ct.astype(mul_f * 1.7015043497085571, ct.bfloat16)
        mul_1_f = ct.astype(mul_1_bf, ct.float32)

        # erf(arg1 * 0.7071...) via Abramowitz-Stegun 7.1.26 polynomial.
        x = arg1_1d * 0.7071067811865476
        ax = ct.where(x < 0.0, -x, x)
        t = 1.0 / (1.0 + 0.3275911 * ax)
        poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))))
        erf_val_pos = 1.0 - poly * ct.exp(-x * x)
        erf_signed = ct.where(x < 0.0, -erf_val_pos, erf_val_pos)

        # add   = erf + 1
        # mul_3 = add * 0.5
        # mul_4 = arg1 * arg1
        # mul_5 = mul_4 * -0.5
        # exp   = exp(mul_5)
        # mul_6 = exp * 0.3989...
        # mul_7 = arg1 * mul_6
        # add_1 = mul_3 + mul_7
        add_val = erf_signed + 1.0
        mul_3 = add_val * 0.5
        mul_4 = arg1_1d * arg1_1d
        mul_5 = mul_4 * -0.5
        exp_val = ct.exp(mul_5)
        mul_6 = exp_val * 0.3989422804014327
        mul_7 = arg1_1d * mul_6
        add_1 = mul_3 + mul_7

        # convert_2 = to_bf16(mul_1 * add_1)
        convert_2_bf = ct.astype(mul_1_f * add_1, ct.bfloat16)
        convert_2_f = ct.astype(convert_2_bf, ct.float32)

        # add_2 = to_bf16(arg2 + convert_2)
        add_2 = ct.astype(arg2_1d + convert_2_f, ct.bfloat16)
        add_2_f = ct.astype(add_2, ct.float32)

        # mul_9  = to_bf16(add_2 * 0.2)
        mul_9 = ct.astype(add_2_f * 0.2, ct.bfloat16)
        mul_9_f = ct.astype(mul_9, ct.float32)
        # mul_10 = to_bf16(arg4 * sigmoid)
        mul_10 = ct.astype(arg4_1d * sigmoid_scalar, ct.bfloat16)
        mul_10_f = ct.astype(mul_10, ct.float32)
        # mul_11 = to_bf16(mul_10 * 2)
        mul_11 = ct.astype(mul_10_f * 2.0, ct.bfloat16)
        mul_11_f = ct.astype(mul_11, ct.float32)
        # mul_12 = to_bf16(mul_9 * mul_11)
        mul_12 = ct.astype(mul_9_f * mul_11_f, ct.bfloat16)
        mul_12_f = ct.astype(mul_12, ct.float32)

        # mul_13 = to_bf16(mul_9 * scalar)
        mul_13 = ct.astype(mul_9_f * scalar_scalar, ct.bfloat16)
        mul_13_f = ct.astype(mul_13, ct.float32)
        # mul_14 = to_bf16(mul_13 * 2)
        mul_14 = ct.astype(mul_13_f * 2.0, ct.bfloat16)
        mul_14_f = ct.astype(mul_14, ct.float32)
        # mul_15 = to_bf16(mul_14 * arg4)
        mul_15 = ct.astype(mul_14_f * arg4_1d, ct.bfloat16)
        mul_15_f = ct.astype(mul_15, ct.float32)

        # Mask OOB.
        mul_12_masked = ct.where(hw_mask, mul_12_f, 0.0)
        mul_15_masked = ct.where(hw_mask, mul_15_f, 0.0)

        partial_sum1 = partial_sum1 + ct.sum(mul_12_masked)
        partial_sum2 = partial_sum2 + ct.sum(mul_15_masked)

        add_2_3d = ct.reshape(add_2, (1, 1, BLOCK_HW))
        mul_14_3d = ct.reshape(mul_14, (1, 1, BLOCK_HW))
        ct.store(out_add_ptr, index=(n, c, block), tile=add_2_3d)
        ct.store(out_mul14_ptr, index=(n, c, block), tile=mul_14_3d)

    ct.store(partial_sum1_ptr, index=(n, c), tile=ct.reshape(partial_sum1, (1, 1)))
    ct.store(sum2_ptr, index=(n, c), tile=ct.reshape(partial_sum2, (1, 1)))


@ct.kernel
def _side_kernel(
    sum2_ptr,        # f32 [N, C]
    sigmoid_ptr,     # bf16 [N, C]
    side_out_ptr,    # bf16 [N, C]  (mul_17_bf16)
    N_C: ct.Constant[int],
    C_C: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    """Grid: (N, C // BLOCK_C, 1)."""
    n = ct.bid(0)
    c_block = ct.bid(1)
    sum2 = ct.load(sum2_ptr, index=(n, c_block), shape=(1, BLOCK_C))
    sum2_bf = ct.astype(sum2, ct.bfloat16)
    sum2_bf_f = ct.astype(sum2_bf, ct.float32)
    sig = ct.astype(ct.load(sigmoid_ptr, index=(n, c_block), shape=(1, BLOCK_C)), ct.float32)
    grad = sig * (1.0 - sig)
    side = ct.astype(sum2_bf_f * grad, ct.bfloat16)
    ct.store(side_out_ptr, index=(n, c_block), tile=side)


@ct.kernel
def _channel_reduce_kernel(
    side_ptr,        # bf16 [N, C]
    channel_ptr,     # f32 [C]
    N_C: ct.Constant[int],
    C_C: ct.Constant[int],
    N_PAD: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    """Grid: (C // BLOCK_C, 1, 1). Reduces over N (padded to N_PAD) per BLOCK_C channels."""
    c_block = ct.bid(0)
    n_offsets = ct.arange(N_PAD, dtype=ct.int32)
    n_mask = n_offsets < N_C

    vals = ct.load(
        side_ptr, index=(0, c_block),
        shape=(N_PAD, BLOCK_C),
        padding_mode=ct.PaddingMode.ZERO,
    )
    vals_f = ct.astype(vals, ct.float32)
    n_mask_2d = ct.reshape(n_mask, (N_PAD, 1))
    vals_masked = ct.where(n_mask_2d, vals_f, 0.0)
    reduced = ct.sum(vals_masked, axis=0)  # (BLOCK_C,)
    ct.store(channel_ptr, index=(c_block,), tile=reduced)


@oracle_impl(hardware="B200", point="bfd27ee1", H=12, BLOCK_C=64, BLOCK_HW=32,
             SIDE_BLOCK=256, FINAL_BLOCK_C=16)
@oracle_impl(hardware="B200", point="9d70a1e6", H=6, BLOCK_C=128, BLOCK_HW=64,
             SIDE_BLOCK=256, FINAL_BLOCK_C=16)
def oracle_forward(inputs, *, H, BLOCK_C, BLOCK_HW, SIDE_BLOCK, FINAL_BLOCK_C):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1 = inputs
    n = int(arg0_1.shape[0])
    c = int(arg0_1.shape[1])
    hw = H * H
    device = arg0_1.device

    # Force NCHW-contiguous scratches for cuTile (n, c, hw) tile access. The
    # inputs are channels-last strided [N, C, H, H]; contiguous().view(n, c, hw)
    # gives the elements in the flat memory order the kernel expects.
    arg0_nchw = arg0_1.contiguous().view(n, c, hw)
    arg1_nchw = arg1_1.contiguous().view(n, c, hw)
    arg2_nchw = arg2_1.contiguous().view(n, c, hw)
    arg4_nchw = arg4_1.contiguous().view(n, c, hw)
    # arg3 is bf16 [N, C, 1, 1] with contiguous default -> view as [N, C].
    arg3_2d = arg3_1.contiguous().view(n, c)
    arg5_1d = arg5_1.view(1)

    # Outputs / scratch.
    add_2_nchw = torch.empty((n, c, hw), device=device, dtype=torch.bfloat16)
    mul_14_nchw = torch.empty((n, c, hw), device=device, dtype=torch.bfloat16)
    sigmoid_nc = torch.empty((n, c), device=device, dtype=torch.bfloat16)
    partial_sum1 = torch.empty((n, c), device=device, dtype=torch.float32)
    sum_2_nc = torch.empty((n, c), device=device, dtype=torch.float32)

    # Pick BLOCK_HW s.t. HW % BLOCK_HW == 0.
    if hw == 144:
        BLOCK_HW = 16
    elif hw == 36:
        BLOCK_HW = 4
    else:
        BLOCK_HW = 1
        cand = 1
        while cand <= hw and hw % cand == 0:
            BLOCK_HW = cand
            cand *= 2

    hw_pad = hw  # since HW % BLOCK_HW == 0

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (n, c, 1),
        _producer_kernel,
        (
            arg0_nchw, arg1_nchw, arg2_nchw, arg3_2d, arg4_nchw, arg5_1d,
            add_2_nchw, sigmoid_nc, mul_14_nchw, partial_sum1, sum_2_nc,
            hw, hw_pad, BLOCK_HW,
        ),
    )

    # Copy NCHW scratch -> NHWC-strided outputs.
    out_add = torch.empty_strided(
        (n, c, H, H), (c * hw, 1, H * c, c),
        device=device, dtype=torch.bfloat16,
    )
    out_add.copy_(add_2_nchw.view(n, c, H, H))
    out_mul14 = torch.empty_strided(
        (n, c, H, H), (c * hw, 1, H * c, c),
        device=device, dtype=torch.bfloat16,
    )
    out_mul14.copy_(mul_14_nchw.view(n, c, H, H))

    # Scalar sum_1 = total of partial_sum1 across (n, c).
    out_scalar = torch.empty_strided((), (), device=device, dtype=torch.float32)
    out_scalar.copy_(partial_sum1.sum())

    # Sigmoid output: bf16 [N, C, 1, 1] with (C, 1, 1, 1) stride.
    out_sigmoid = torch.empty_strided(
        (n, c, 1, 1), (c, 1, 1, 1),
        device=device, dtype=torch.bfloat16,
    )
    out_sigmoid.copy_(sigmoid_nc.view(n, c, 1, 1))

    # Side output: bf16 [N, C, 1, 1] = to_bf16(to_bf16(sum2_nc) * sig * (1 - sig)).
    side_nc = torch.empty((n, c), device=device, dtype=torch.bfloat16)
    if c % BLOCK_C != 0:
        raise NotImplementedError(f"C={c} not divisible by BLOCK_C={BLOCK_C}")
    ct.launch(
        stream,
        (n, c // BLOCK_C, 1),
        _side_kernel,
        (sum_2_nc, sigmoid_nc, side_nc, n, c, BLOCK_C),
    )
    out_side = torch.empty_strided(
        (n, c, 1, 1), (c, 1, 1, 1),
        device=device, dtype=torch.bfloat16,
    )
    out_side.copy_(side_nc.view(n, c, 1, 1))

    # Channel reduce: f32 sum over N of the side (bf16) values.
    n_pad = _next_pow2(n)
    out_channel = torch.empty((c,), device=device, dtype=torch.float32)
    if c % FINAL_BLOCK_C != 0:
        raise NotImplementedError(f"C={c} not divisible by FINAL_BLOCK_C={FINAL_BLOCK_C}")
    ct.launch(
        stream,
        (c // FINAL_BLOCK_C, 1, 1),
        _channel_reduce_kernel,
        (side_nc, out_channel, n, c, n_pad, FINAL_BLOCK_C),
    )

    return out_add, out_sigmoid, out_scalar, out_mul14, out_side, out_channel
