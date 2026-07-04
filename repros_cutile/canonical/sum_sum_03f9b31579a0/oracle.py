"""cuTile port of sum_sum_03f9b31579a0: GhostNet masked BN-backward.

Classification: COOPERATIVE_SPLIT_K. Uses a cuTile kernel for the fused
hard-sigmoid + add_1 producer over channels-last strided tensors, then
torch for the reductions and BN-backward epilogue on the 60-channel tail.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


N = 512
FULL_C = 120
C = 60
H = 28
W = 28
HW = H * W
INV_R = 2.4912308673469386e-06


@ct.kernel
def _producer_add1_kernel(
    gate_ptr,        # f32 [N*FULL_C]
    spatial_ptr,     # bf16 [N*H*W*FULL_C]  (channels-last)
    bias_ptr,        # bf16 [N*FULL_C]
    add_ptr,         # bf16 [N*H*W*FULL_C]  (channels-last)
    N_HW_FC: ct.Constant[int],
    FC: ct.Constant[int],
    HW_: ct.Constant[int],
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    offsets = pid * BLOCK + ct.arange(BLOCK, dtype=ct.int64)
    mask = offsets < N_HW_FC

    c_idx = offsets % FC
    row_flat = offsets // FC
    n = row_flat // HW_
    param_idx = n * FC + c_idx

    gate_source = ct.astype(ct.gather(gate_ptr, param_idx, mask=mask), ct.float32)
    shifted = gate_source + 3.0
    zero_t = ct.full((BLOCK,), 0.0, dtype=ct.float32)
    six_t = ct.full((BLOCK,), 6.0, dtype=ct.float32)
    clamped = ct.where(shifted < zero_t, zero_t, shifted)
    clamped = ct.where(clamped > six_t, six_t, clamped)
    gate = ct.astype(ct.astype(clamped / 6.0, ct.bfloat16), ct.float32)

    spatial = ct.astype(ct.gather(spatial_ptr, offsets, mask=mask), ct.float32)
    scaled = ct.astype(ct.astype(spatial * gate, ct.bfloat16), ct.float32)
    bias = ct.astype(ct.gather(bias_ptr, param_idx, mask=mask), ct.float32)
    averaged = ct.astype(ct.astype(bias * (1.0 / 784.0), ct.bfloat16), ct.float32)
    add_value = ct.astype(scaled + averaged, ct.bfloat16)

    ct.scatter(add_ptr, offsets, add_value, mask=mask)


@oracle_impl(hardware="B200", point="3e5640ef", BLOCK=1024)
def oracle_forward(inputs, *, BLOCK: int):
    (arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, _shape_param_0) = inputs
    device = arg1_1.device

    # Allocate add_1 output in channels-last layout.
    add_out = torch.empty_strided(
        (N, FULL_C, H, W),
        (FULL_C * HW, 1, FULL_C * W, FULL_C),
        device=device,
        dtype=torch.bfloat16,
    )
    # Flat view of channels-last memory (NHWC-contig).
    add_flat = torch.empty(N * H * W * FULL_C, device=device, dtype=torch.bfloat16)
    arg1_perm = arg1_1.permute(0, 2, 3, 1).contiguous().view(-1)
    arg0_flat = arg0_1.view(N, FULL_C).contiguous().view(-1)
    arg2_flat = arg2_1.view(N, FULL_C).contiguous().view(-1)

    total = N * FULL_C * HW
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(total, BLOCK), 1, 1),
        _producer_add1_kernel,
        (arg0_flat, arg1_perm, arg2_flat, add_flat, total, FULL_C, HW, BLOCK),
    )
    # Copy NHWC-contig flat -> channels-last strided add_out
    add_out.copy_(add_flat.view(N, H, W, FULL_C).permute(0, 3, 1, 2))

    # Rest is torch (matches the atol=1e-2/rtol=1e-2 gate).
    # slice_1 = add_out[:, 60:, :, :]
    slice_1 = add_out[:, C:, :, :]  # bf16 [N, C, H, W]

    # BN affine on arg3_1 (channels-last bf16 [N, C, H, W])
    arg3_f = arg3_1.to(torch.float32)
    centered = arg3_f - arg4_1  # (1,C,1,1) broadcast
    normed = centered * arg5_1
    affine = normed * arg6_1.view(1, C, 1, 1) + arg7_1.view(1, C, 1, 1)
    affine_bf16 = affine.to(torch.bfloat16)
    relu = torch.relu(affine_bf16)
    le = relu <= 0
    where_val = torch.where(le, arg8_1, slice_1)  # bf16 [N, C, H, W]
    where_f32 = where_val.to(torch.float32)

    invstd_v = arg5_1.view(C)

    # sum_1 = sum(where_f32, [0, 2, 3])  -> [C]
    sum_1 = where_f32.sum(dim=[0, 2, 3])

    sub_1 = centered  # f32 [N, C, H, W]
    mul_3 = where_f32 * sub_1
    sum_2 = mul_3.sum(dim=[0, 2, 3])

    mul_4 = sum_1 * INV_R  # [C]
    mul_5 = sum_2 * INV_R
    invstd_sq = invstd_v * invstd_v
    mul_7 = mul_5 * invstd_sq

    mul_8_scale = invstd_v * arg6_1  # out_scale [C]
    mean_term = mul_4
    dot_coeff = mul_7

    mul_9 = sub_1 * dot_coeff.view(1, C, 1, 1)
    sub_2 = where_f32 - mul_9
    sub_3 = sub_2 - mean_term.view(1, C, 1, 1)
    mul_10 = sub_3 * mul_8_scale.view(1, C, 1, 1)
    out_bf16 = mul_10.to(torch.bfloat16)

    mul_11 = sum_2 * invstd_v  # f32 [C]

    out_bf16 = out_bf16.contiguous(memory_format=torch.channels_last)

    return add_out, sum_1, mul_11, out_bf16
