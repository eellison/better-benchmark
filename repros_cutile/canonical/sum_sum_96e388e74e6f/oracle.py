"""cuTile port of sum_sum_96e388e74e6f: GhostNet BN + gate reduction.

Mirrors Triton's 2-kernel plan:
1. `_row_sum_kernel` (grid = (N*C,)): row-wise sum over HW via ct.sum,
   matching Triton's `_spatial_row_gate_kernel` spatial reduction.
2. `_channel_sum_kernel` (grid = (C,)): per-channel sum over N of the
   gated bf16 output via ct.sum, matching Triton's
   `_channel_sum_finalize_kernel`.

Torch does the BN forward, gate, and where. cuTile does both reductions
in-kernel via ct.sum so the compiler-vs-compiler comparison is fair.

Handles both point 0e39883f (C=72, HW=28*28=784, BLOCK_HW=1024) and
3bcfd222 (C=672, HW=7*7=49, BLOCK_HW=64).
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


N = 512


@ct.kernel
def _row_sum_kernel(
    prod_ptr,      # bf16 [N*C, HW]
    sum_ptr,       # f32 [N*C]
    HW: ct.Constant[int],
    BLOCK_HW: ct.Constant[int],
):
    row = ct.bid(0)
    x = ct.load(
        prod_ptr, index=(row, 0), shape=(1, BLOCK_HW),
        padding_mode=ct.PaddingMode.ZERO,
    )
    x_f = ct.astype(x, ct.float32)
    col_idx = ct.arange(BLOCK_HW, dtype=ct.int32)
    col_mask = ct.reshape(col_idx < HW, (1, BLOCK_HW))
    x_masked = ct.where(col_mask, x_f, 0.0)
    row_sum = ct.sum(x_masked)
    ct.store(sum_ptr, index=(row,), tile=ct.reshape(row_sum, (1,)))


@ct.kernel
def _channel_sum_kernel(
    gated_ptr,      # bf16 [N*C] flat
    channel_ptr,    # f32 [C]
    C_: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
):
    c = ct.bid(0)
    rows = ct.arange(BLOCK_N, dtype=ct.int32)
    active = rows < N
    offsets = rows * C_ + c
    values = ct.gather(gated_ptr, offsets)
    values_f = ct.astype(values, ct.float32)
    values_f = ct.where(active, values_f, 0.0)
    total = ct.sum(values_f)
    # bf16 round-trip to match Triton's `.to(tl.bfloat16).to(tl.float32)`
    total_bf = ct.astype(total, ct.bfloat16)
    total_f = ct.astype(total_bf, ct.float32)
    ct.store(channel_ptr, index=(c,), tile=ct.reshape(total_f, (1,)))


def _next_pow2(v):
    return 1 << (int(v) - 1).bit_length()


@oracle_impl(hardware="B200", point="0e39883f", BLOCK_HW=1024)
@oracle_impl(hardware="B200", point="3bcfd222", BLOCK_HW=64)
def oracle_forward(inputs, *, BLOCK_HW: int):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1 = inputs
    device = arg0_1.device
    batch = int(arg0_1.shape[0])
    c = int(arg0_1.shape[1])
    h = int(arg0_1.shape[2])
    w = int(arg0_1.shape[3])
    hw = h * w

    # BN forward in torch (matches current port's producer creation).
    sub = arg0_1 - arg1_1
    mul = sub * arg2_1
    mul_1 = mul * arg3_1.view(1, c, 1, 1)
    add = mul_1 + arg4_1.view(1, c, 1, 1)
    conv = add.to(torch.bfloat16)
    mul_2 = arg5_1 * conv
    conv_1 = arg6_1.to(torch.float32)

    # Spatial sum over HW via cuTile.
    mul_2_contig = mul_2.contiguous()
    mul_2_2d = mul_2_contig.view(batch * c, hw)
    sum_1_1d = torch.empty(batch * c, device=device, dtype=torch.float32)
    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (batch * c, 1, 1), _row_sum_kernel,
        (mul_2_2d, sum_1_1d, hw, BLOCK_HW),
    )
    sum_1 = sum_1_1d.view(batch, c, 1, 1)

    # Gate in torch (matches BN + hardsigmoid + where).
    conv_2 = sum_1.to(torch.bfloat16)
    conv_3 = conv_2.to(torch.float32)
    gt = conv_1 > -3.0
    lt = conv_1 < 3.0
    bitwise_and = gt & lt
    mul_3 = conv_3 * 0.16666666666666666
    where = torch.where(bitwise_and, mul_3, arg7_1)
    conv_4 = where.to(torch.bfloat16)

    # Channel sum over N via cuTile (matches Triton's finalize kernel).
    conv_4_flat = conv_4.contiguous().view(-1)
    conv_5 = torch.empty((c,), device=device, dtype=torch.float32)
    block_n = _next_pow2(batch)
    ct.launch(
        stream, (c, 1, 1), _channel_sum_kernel,
        (conv_4_flat, conv_5, c, block_n),
    )

    return conv_1, conv_4, conv_5
