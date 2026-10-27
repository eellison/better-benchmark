"""cuTile port of sum_sum_1a7b8d26a98e: NFNet bf16 SiLU-gradient + spatial sum + channel sum.

Inputs are channels-last [N, C, H, W]. We view them as [N, H*W, C] (NHWC)
which is contiguous — no copy required. Per (n, c) program aggregates over
HW using channels-last flat address arithmetic via `torch.as_strided`.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


SILU_SCALE = 0.9622504486493761
ADD_SCALE = 0.2
MUL5_SCALE = 2.0


def _next_pow2(v):
    r = 1
    while r < v:
        r <<= 1
    return r


@ct.kernel
def _silu_grad_reduce_kernel(
    arg0_arr,      # bf16 [N, HW, C]  (NHWC contiguous view)
    arg1_arr,      # bf16 [N, HW, C]
    arg2_arr,      # bf16 [N, HW, C]
    arg3_arr,      # bf16 [N, HW, C]
    arg4_arr,      # bf16 [N, C]
    add1_out_arr,  # bf16 [N, HW, C]
    mul5_out_arr,  # bf16 [N, HW, C]
    sigmoid1_out_arr, # bf16 [N, C]
    row_out_arr,   # bf16 [N, C]
    HW: ct.Constant[int],
    HW_PAD: ct.Constant[int],
    C: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    n = ct.bid(0)
    c_block = ct.bid(1)

    # Load tile of shape (HW_PAD, BLOCK_C) from NHWC layout.
    a0 = ct.load(arg0_arr, index=(n, 0, c_block),
                 shape=(1, HW_PAD, BLOCK_C),
                 padding_mode=ct.PaddingMode.ZERO)
    a1 = ct.load(arg1_arr, index=(n, 0, c_block),
                 shape=(1, HW_PAD, BLOCK_C),
                 padding_mode=ct.PaddingMode.ZERO)
    a2 = ct.load(arg2_arr, index=(n, 0, c_block),
                 shape=(1, HW_PAD, BLOCK_C),
                 padding_mode=ct.PaddingMode.ZERO)
    a3 = ct.load(arg3_arr, index=(n, 0, c_block),
                 shape=(1, HW_PAD, BLOCK_C),
                 padding_mode=ct.PaddingMode.ZERO)

    a0_f = ct.astype(a0, ct.float32)
    a1_f = ct.astype(a1, ct.float32)
    a2_f = ct.astype(a2, ct.float32)
    a3_f = ct.astype(a3, ct.float32)

    scaled_bf16 = ct.astype(a0_f * SILU_SCALE, ct.bfloat16)
    scaled = ct.astype(scaled_bf16, ct.float32)

    sig = 1.0 / (1.0 + ct.exp(-a1_f))
    silu_grad = scaled * sig * (a1_f * (1.0 - sig) + 1.0)
    silu_grad_bf16 = ct.astype(silu_grad, ct.bfloat16)

    add_bf16 = ct.astype(a2_f + ct.astype(silu_grad_bf16, ct.float32), ct.bfloat16)
    add_f = ct.astype(add_bf16, ct.float32)
    mul4_bf16 = ct.astype(add_f * ADD_SCALE, ct.bfloat16)
    mul5_bf16 = ct.astype(ct.astype(mul4_bf16, ct.float32) * MUL5_SCALE, ct.bfloat16)
    mul6_bf16 = ct.astype(ct.astype(mul5_bf16, ct.float32) * a3_f, ct.bfloat16)

    ct.store(add1_out_arr, index=(n, 0, c_block), tile=add_bf16)
    ct.store(mul5_out_arr, index=(n, 0, c_block), tile=mul5_bf16)

    # Spatial sum over HW axis (mask hw >= HW).
    hw_idx = ct.arange(HW_PAD, dtype=ct.int32)
    hw_valid = hw_idx < HW
    hw_valid_3d = ct.broadcast_to(
        ct.reshape(hw_valid, (1, HW_PAD, 1)), (1, HW_PAD, BLOCK_C))
    zero = ct.zeros((1, HW_PAD, BLOCK_C), dtype=ct.float32)
    mul6_f = ct.astype(mul6_bf16, ct.float32)
    mul6_masked = ct.where(hw_valid_3d, mul6_f, zero)
    spatial_sum = ct.sum(mul6_masked, axis=1)  # (1, BLOCK_C)

    gate = ct.load(arg4_arr, index=(n, c_block), shape=(1, BLOCK_C))
    gate_f = ct.astype(gate, ct.float32)
    gate_sig = 1.0 / (1.0 + ct.exp(-gate_f))
    gate_sig_bf16 = ct.astype(gate_sig, ct.bfloat16)
    ct.store(sigmoid1_out_arr, index=(n, c_block), tile=gate_sig_bf16)

    gate_sig_f32 = ct.astype(gate_sig_bf16, ct.float32)
    gate_deriv = gate_sig_f32 * (1.0 - gate_sig_f32)

    sum_bf16_f32 = ct.astype(ct.astype(spatial_sum, ct.bfloat16), ct.float32)
    row_bf16 = ct.astype(sum_bf16_f32 * gate_deriv, ct.bfloat16)
    ct.store(row_out_arr, index=(n, c_block), tile=row_bf16)


@ct.kernel
def _channel_finalizer_kernel(
    row_out_arr,     # bf16 [N, C]
    final_out_arr,   # f32  [C]
    N: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    c_block = ct.bid(0)
    vals = ct.load(row_out_arr, index=(0, c_block), shape=(N, BLOCK_C))
    vals_f = ct.astype(vals, ct.float32)
    total = ct.sum(vals_f, axis=0)  # (BLOCK_C,)
    total_bf16_f32 = ct.astype(ct.astype(total, ct.bfloat16), ct.float32)
    ct.store(final_out_arr, index=(c_block,), tile=total_bf16_f32)


@oracle_impl(hardware="B200", point="3e2b5914", BLOCK_C=32)
@oracle_impl(hardware="B200", point="31d48416", BLOCK_C=32)
def oracle_forward(inputs, *, BLOCK_C: int):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1 = inputs
    n = int(arg0_1.shape[0])
    c = int(arg0_1.shape[1])
    h = int(arg0_1.shape[2])
    w = int(arg0_1.shape[3])
    hw = h * w
    device = arg0_1.device

    # Channels-last inputs [N,C,H,W] with stride (C*H*W, 1, W*C, C). Permute
    # (0, 2, 3, 1) -> NHWC contiguous. Reshape to (N, HW, C).
    a0_3d = arg0_1.permute(0, 2, 3, 1).reshape(n, hw, c)
    a1_3d = arg1_1.permute(0, 2, 3, 1).reshape(n, hw, c)
    a2_3d = arg2_1.permute(0, 2, 3, 1).reshape(n, hw, c)
    a3_3d = arg3_1.permute(0, 2, 3, 1).reshape(n, hw, c)
    a4_2d = arg4_1.view(n, c)

    add1_out_cl = torch.empty_strided(
        tuple(arg2_1.shape), tuple(arg2_1.stride()),
        device=device, dtype=torch.bfloat16,
    )
    mul5_out_cl = torch.empty_strided(
        tuple(arg2_1.shape), tuple(arg2_1.stride()),
        device=device, dtype=torch.bfloat16,
    )
    add1_view = add1_out_cl.permute(0, 2, 3, 1).reshape(n, hw, c)
    mul5_view = mul5_out_cl.permute(0, 2, 3, 1).reshape(n, hw, c)

    sigmoid1 = torch.empty_strided(
        tuple(arg4_1.shape), tuple(arg4_1.stride()),
        device=device, dtype=torch.bfloat16,
    )
    row_out = torch.empty_strided(
        tuple(arg4_1.shape), tuple(arg4_1.stride()),
        device=device, dtype=torch.bfloat16,
    )
    sig_2d = sigmoid1.view(n, c)
    row_2d = row_out.view(n, c)

    HW_PAD = _next_pow2(hw)

    stream = torch.cuda.current_stream()
    grid = (n, ct.cdiv(c, BLOCK_C), 1)
    ct.launch(
        stream,
        grid,
        _silu_grad_reduce_kernel,
        (a0_3d, a1_3d, a2_3d, a3_3d, a4_2d,
         add1_view, mul5_view, sig_2d, row_2d,
         hw, HW_PAD, c, BLOCK_C),
    )

    channel_out = torch.empty((c,), device=device, dtype=torch.float32)
    ct.launch(
        stream,
        (ct.cdiv(c, BLOCK_C), 1, 1),
        _channel_finalizer_kernel,
        (row_2d, channel_out, n, BLOCK_C),
    )
    return add1_out_cl, mul5_out_cl, sigmoid1, row_out, channel_out
