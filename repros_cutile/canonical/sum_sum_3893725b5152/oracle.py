"""cuTile port of sum_sum_3893725b5152: NFNet gate gradient.

Reference (channels-last arg0/arg1 [128, 1536, 7, 7], arg2 [128, 1536, 1, 1]):
  mul_1 = bf16(bf16(arg0 * 0.2) * 2.0)
  mul_2 = bf16(mul_1 * arg1)
  sigmoid = bf16 sigmoid(arg2)
  row_sum = bf16(sum(mul_2, dim=[2,3])) (per (n, c))
  gate_grad_bf16 = bf16(row_sum * sigmoid * (1 - sigmoid))
  channel_sum_bf16 = bf16(sum(gate_grad_bf16, dim=[0, 2, 3]))

cuTile plan:
  * pointwise_kernel: reads arg0/arg1 in (N*H*W, C) contiguous view and stores
    mul_1 and mul_2 with a per-pixel bf16 rounding chain; also emits mul_2 as
    an f32 spatial partial (per pixel) for reduction.
  * spatial_sum_kernel: reduces mul_2 over N*H*W to per-channel (or per (N, C))
    partials with masked sum.
  * gate_and_channel: computes sigmoid, gate gradient, and final channel sum
    fully in cuTile.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


N = 128
C = 1536
H = 7
W = 7
HW = H * W  # 49
R = N * HW  # 6272
BLOCK_R = 128  # divides 6272 (6272/128 = 49)
BLOCK_C_ROW = 128
BLOCK_C_FINAL = 32


@ct.kernel
def _pointwise_kernel(
    arg0_ptr,        # bf16 [R, C]  channels-last flattened
    arg1_ptr,        # bf16 [R, C]
    mul1_ptr,        # bf16 [R, C]
    mul2_ptr,        # bf16 [R, C]
    BLOCK_R_: ct.Constant[int],
    BLOCK_C_: ct.Constant[int],
):
    r = ct.bid(0)
    c = ct.bid(1)

    x0 = ct.load(arg0_ptr, index=(r, c), shape=(BLOCK_R_, BLOCK_C_))
    x1 = ct.load(arg1_ptr, index=(r, c), shape=(BLOCK_R_, BLOCK_C_))
    mul_0_bf16 = ct.astype(ct.astype(x0, ct.float32) * 0.2, ct.bfloat16)
    mul_1_bf16 = ct.astype(ct.astype(mul_0_bf16, ct.float32) * 2.0, ct.bfloat16)
    mul_2_bf16 = ct.astype(
        ct.astype(mul_1_bf16, ct.float32) * ct.astype(x1, ct.float32),
        ct.bfloat16,
    )
    ct.store(mul1_ptr, index=(r, c), tile=mul_1_bf16)
    ct.store(mul2_ptr, index=(r, c), tile=mul_2_bf16)


@ct.kernel
def _row_sum_kernel(
    mul2_ptr,        # bf16 [R, C]
    sigmoid_ptr,     # bf16 [N, C] (broadcast)
    gate_grad_ptr,   # bf16 [N, C] (view of [N, C, 1, 1])
    ROWS_PER_BATCH: ct.Constant[int],
    BLOCK_ROW: ct.Constant[int],
    BLOCK_C_: ct.Constant[int],
):
    n = ct.bid(0)
    c = ct.bid(1)
    # Load ROWS_PER_BATCH rows (= HW = 49 rows padded to BLOCK_ROW=64) for this (n, c) group.
    tile = ct.load(
        mul2_ptr, index=(n, 0, c), shape=(1, BLOCK_ROW, BLOCK_C_),
        padding_mode=ct.PaddingMode.ZERO,
    )
    tile_f = ct.astype(tile, ct.float32)
    row_idx = ct.arange(BLOCK_ROW, dtype=ct.int32)
    row_mask = ct.reshape(row_idx < ROWS_PER_BATCH, (1, BLOCK_ROW, 1))
    valid = ct.where(row_mask, tile_f, 0.0)
    row_sum_f32 = ct.sum(valid, axis=1)  # (1, BLOCK_C_)
    row_sum_bf16 = ct.astype(row_sum_f32, ct.bfloat16)

    sigmoid = ct.load(sigmoid_ptr, index=(n, c), shape=(1, BLOCK_C_))
    sigmoid_f = ct.astype(sigmoid, ct.float32)
    deriv = sigmoid_f * (1.0 - sigmoid_f)
    gate_grad = ct.astype(ct.astype(row_sum_bf16, ct.float32) * deriv, ct.bfloat16)
    ct.store(gate_grad_ptr, index=(n, c), tile=gate_grad)


@ct.kernel
def _channel_finalize_kernel(
    values_ptr,       # bf16 [N, C]
    out_ptr,          # f32 [C]
    N_: ct.Constant[int],
    C_: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    c_tile = ct.bid(0)
    values = ct.load(
        values_ptr, index=(0, c_tile), shape=(BLOCK_N, BLOCK_C),
        padding_mode=ct.PaddingMode.ZERO,
    )
    values_f = ct.astype(values, ct.float32)
    row_idx = ct.arange(BLOCK_N, dtype=ct.int32)
    row_mask = ct.reshape(row_idx < N_, (BLOCK_N, 1))
    valid = ct.where(row_mask, values_f, 0.0)
    total = ct.sum(valid, axis=0)
    total_bf16 = ct.astype(total, ct.bfloat16)
    total_f32 = ct.astype(total_bf16, ct.float32)
    ct.store(out_ptr, index=(c_tile,), tile=total_f32)


def _next_p2(v):
    return 1 << (int(v) - 1).bit_length()


@oracle_impl(hardware="B200", point="9983a35a", BLOCK_HW=64, BLOCK_C=128, FINAL_BLOCK_C=32)
def oracle_forward(inputs, **_kwargs):
    arg0_1, arg1_1, arg2_1 = inputs
    device = arg0_1.device
    n = int(arg0_1.shape[0])
    c = int(arg0_1.shape[1])
    h = int(arg0_1.shape[2])
    w = int(arg0_1.shape[3])
    assert n == N and c == C and h == H and w == W

    # Channels-last view: (N, H, W, C) contiguous.
    arg0_rc = arg0_1.permute(0, 2, 3, 1).contiguous().view(R, C)
    arg1_rc = arg1_1.permute(0, 2, 3, 1).contiguous().view(R, C)

    mul_1_rc = torch.empty((R, C), device=device, dtype=torch.bfloat16)
    mul_2_rc = torch.empty((R, C), device=device, dtype=torch.bfloat16)

    stream = torch.cuda.current_stream()
    # Pointwise mul_1/mul_2 as a tile grid.
    ct.launch(
        stream, (R // BLOCK_R, C // BLOCK_C_ROW, 1), _pointwise_kernel,
        (arg0_rc, arg1_rc, mul_1_rc, mul_2_rc, BLOCK_R, BLOCK_C_ROW),
    )

    # For the row_sum kernel we need mul_2 viewed as (N, H*W, C). Since we
    # already have (R, C) contiguous with R = N * HW, the reshape is just a
    # view.
    mul_2_nhc = mul_2_rc.view(N, HW, C)
    # sigmoid on arg2 (bf16 [N, C, 1, 1]) → (N, C).
    sigmoid_bf16 = torch.sigmoid(arg2_1)  # bf16 [N, C, 1, 1]
    sigmoid_2d = sigmoid_bf16.view(N, C)
    gate_grad_2d = torch.empty((N, C), device=device, dtype=torch.bfloat16)

    BLOCK_ROW = _next_p2(HW)  # 64
    ct.launch(
        stream, (N, C // BLOCK_C_ROW, 1), _row_sum_kernel,
        (mul_2_nhc, sigmoid_2d, gate_grad_2d, HW, BLOCK_ROW, BLOCK_C_ROW),
    )

    # Channel finalization.
    channel_sum = torch.empty((C,), device=device, dtype=torch.float32)
    BLOCK_N = _next_p2(N)
    ct.launch(
        stream, (C // BLOCK_C_FINAL, 1, 1), _channel_finalize_kernel,
        (gate_grad_2d, channel_sum, N, C, BLOCK_N, BLOCK_C_FINAL),
    )

    # Rebuild mul_1 with channels-last stride.
    mul_1 = torch.empty_strided(
        (N, C, H, W), (C * HW, 1, W * C, C),
        device=device, dtype=torch.bfloat16,
    )
    mul_1.copy_(mul_1_rc.view(N, H, W, C).permute(0, 3, 1, 2))

    gate_grad_4d = gate_grad_2d.view(N, C, 1, 1)
    return mul_1, sigmoid_bf16, gate_grad_4d, channel_sum
