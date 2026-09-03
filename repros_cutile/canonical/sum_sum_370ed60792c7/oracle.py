"""cuTile port of sum_sum_370ed60792c7: DenseNet BN backward tail with residual chain.

Matches Triton's 1 kernel structure: one program per channel processes
BLOCK_K positions, computes the two channel sums, applies the BN-backward
epilogue, and (for c in slice range) sums 23 residual inputs + dense_bf16
into add_out.

All reductions and bf16 residual-chain adds happen inside the cuTile kernel.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


N = 4
C = 288
H = 14
W = 14
HW = H * W  # 196
K_TOTAL = N * HW  # 784
SLICE_START = 256
SLICE_C = 32
SCALE = 0.0012755102040816326

# Residual channel counts (matches shapes.json arg0..arg22 second dim).
R_CHANNELS = (
    1024, 992, 960, 928, 896, 864, 832, 800, 768, 736,
    704, 672, 640, 608, 576, 544, 512, 480, 448, 416,
    384, 352, 320,
)


@ct.kernel
def _bn_tail_kernel(
    r0_ptr, r1_ptr, r2_ptr, r3_ptr, r4_ptr, r5_ptr, r6_ptr, r7_ptr, r8_ptr,
    r9_ptr, r10_ptr, r11_ptr, r12_ptr, r13_ptr, r14_ptr, r15_ptr, r16_ptr,
    r17_ptr, r18_ptr, r19_ptr, r20_ptr, r21_ptr, r22_ptr,
    mask_ptr,             # bf16 (N*C*HW,) — NCHW contiguous flat
    fill_ptr,             # bf16 (1,)
    source_ptr,           # bf16 (N*C*HW,)
    centered_source_ptr,  # bf16 (N*C*HW,)
    mean_ptr,             # f32 (C,)
    invstd_ptr,           # f32 (C,)
    weight_ptr,           # f32 (C,)
    sum_out_ptr,          # f32 (C,)
    scale_grad_ptr,       # f32 (C,)
    dense_out_ptr,        # bf16 (N*C*HW,)
    add_out_ptr,          # bf16 (N*SLICE_C*HW,)
    C_: ct.Constant[int],
    HW_: ct.Constant[int],
    K_TOTAL_: ct.Constant[int],
    SLICE_START_: ct.Constant[int],
    SLICE_C_: ct.Constant[int],
    SCALE_: ct.Constant[float],
    BLOCK_K: ct.Constant[int],
    R0_C: ct.Constant[int], R1_C: ct.Constant[int], R2_C: ct.Constant[int],
    R3_C: ct.Constant[int], R4_C: ct.Constant[int], R5_C: ct.Constant[int],
    R6_C: ct.Constant[int], R7_C: ct.Constant[int], R8_C: ct.Constant[int],
    R9_C: ct.Constant[int], R10_C: ct.Constant[int], R11_C: ct.Constant[int],
    R12_C: ct.Constant[int], R13_C: ct.Constant[int], R14_C: ct.Constant[int],
    R15_C: ct.Constant[int], R16_C: ct.Constant[int], R17_C: ct.Constant[int],
    R18_C: ct.Constant[int], R19_C: ct.Constant[int], R20_C: ct.Constant[int],
    R21_C: ct.Constant[int], R22_C: ct.Constant[int],
):
    c = ct.bid(0)
    k = ct.arange(BLOCK_K, dtype=ct.int32)
    active = k < K_TOTAL_
    n = k // HW_
    spatial = k - n * HW_
    offsets = n * (C_ * HW_) + c * HW_ + spatial

    mask_value = ct.gather(mask_ptr, offsets, mask=active)
    fill_1 = ct.load(fill_ptr, index=(0,), shape=(1,))
    fill_bc = fill_1 + ct.zeros((BLOCK_K,), dtype=ct.bfloat16)  # (BLOCK_K,)
    source = ct.gather(source_ptr, offsets, mask=active)
    zero_bf = ct.astype(0.0, ct.bfloat16)
    where_bf16 = ct.where(mask_value <= zero_bf, fill_bc, source)
    where_f32 = ct.astype(where_bf16, ct.float32)
    where_f32 = ct.where(active, where_f32, 0.0)

    centered_source = ct.astype(ct.gather(centered_source_ptr, offsets, mask=active), ct.float32)
    mean_1 = ct.load(mean_ptr, index=(c,), shape=(1,))
    centered = centered_source - mean_1  # broadcasts (1,) to (BLOCK_K,)
    centered = ct.where(active, centered, 0.0)

    product = where_f32 * centered
    sum_value = ct.sum(where_f32, axis=0)   # scalar
    dot_value = ct.sum(product, axis=0)

    invstd_1 = ct.load(invstd_ptr, index=(c,), shape=(1,))
    weight_1 = ct.load(weight_ptr, index=(c,), shape=(1,))
    dot_scaled = dot_value * SCALE_
    invstd_sq = invstd_1 * invstd_1
    variance_term = dot_scaled * invstd_sq

    corrected = where_f32 - centered * variance_term
    mean_term = sum_value * SCALE_
    centered_grad = corrected - mean_term
    output_scale = invstd_1 * weight_1
    dense_bf16 = ct.astype(centered_grad * output_scale, ct.bfloat16)

    ct.store(sum_out_ptr, index=(c,), tile=ct.reshape(sum_value, (1,)))
    ct.store(scale_grad_ptr, index=(c,), tile=ct.reshape(dot_value * invstd_1, (1,)))
    ct.scatter(dense_out_ptr, offsets, dense_bf16, mask=active)

    # Residual add path: only active for c in [SLICE_START, SLICE_START+SLICE_C).
    in_slice_scalar = c >= SLICE_START_
    slice_c = c - SLICE_START_
    add_mask = active & in_slice_scalar
    add_offsets = n * (SLICE_C_ * HW_) + slice_c * HW_ + spatial

    # Load residual channels using per-residual Cr strides.
    off_0 = n * (R0_C * HW_) + c * HW_ + spatial
    off_1 = n * (R1_C * HW_) + c * HW_ + spatial
    off_2 = n * (R2_C * HW_) + c * HW_ + spatial
    off_3 = n * (R3_C * HW_) + c * HW_ + spatial
    off_4 = n * (R4_C * HW_) + c * HW_ + spatial
    off_5 = n * (R5_C * HW_) + c * HW_ + spatial
    off_6 = n * (R6_C * HW_) + c * HW_ + spatial
    off_7 = n * (R7_C * HW_) + c * HW_ + spatial
    off_8 = n * (R8_C * HW_) + c * HW_ + spatial
    off_9 = n * (R9_C * HW_) + c * HW_ + spatial
    off_10 = n * (R10_C * HW_) + c * HW_ + spatial
    off_11 = n * (R11_C * HW_) + c * HW_ + spatial
    off_12 = n * (R12_C * HW_) + c * HW_ + spatial
    off_13 = n * (R13_C * HW_) + c * HW_ + spatial
    off_14 = n * (R14_C * HW_) + c * HW_ + spatial
    off_15 = n * (R15_C * HW_) + c * HW_ + spatial
    off_16 = n * (R16_C * HW_) + c * HW_ + spatial
    off_17 = n * (R17_C * HW_) + c * HW_ + spatial
    off_18 = n * (R18_C * HW_) + c * HW_ + spatial
    off_19 = n * (R19_C * HW_) + c * HW_ + spatial
    off_20 = n * (R20_C * HW_) + c * HW_ + spatial
    off_21 = n * (R21_C * HW_) + c * HW_ + spatial
    off_22 = n * (R22_C * HW_) + c * HW_ + spatial

    v0 = ct.gather(r0_ptr, off_0, mask=add_mask)
    v1 = ct.gather(r1_ptr, off_1, mask=add_mask)
    v2 = ct.gather(r2_ptr, off_2, mask=add_mask)
    v3 = ct.gather(r3_ptr, off_3, mask=add_mask)
    v4 = ct.gather(r4_ptr, off_4, mask=add_mask)
    v5 = ct.gather(r5_ptr, off_5, mask=add_mask)
    v6 = ct.gather(r6_ptr, off_6, mask=add_mask)
    v7 = ct.gather(r7_ptr, off_7, mask=add_mask)
    v8 = ct.gather(r8_ptr, off_8, mask=add_mask)
    v9 = ct.gather(r9_ptr, off_9, mask=add_mask)
    v10 = ct.gather(r10_ptr, off_10, mask=add_mask)
    v11 = ct.gather(r11_ptr, off_11, mask=add_mask)
    v12 = ct.gather(r12_ptr, off_12, mask=add_mask)
    v13 = ct.gather(r13_ptr, off_13, mask=add_mask)
    v14 = ct.gather(r14_ptr, off_14, mask=add_mask)
    v15 = ct.gather(r15_ptr, off_15, mask=add_mask)
    v16 = ct.gather(r16_ptr, off_16, mask=add_mask)
    v17 = ct.gather(r17_ptr, off_17, mask=add_mask)
    v18 = ct.gather(r18_ptr, off_18, mask=add_mask)
    v19 = ct.gather(r19_ptr, off_19, mask=add_mask)
    v20 = ct.gather(r20_ptr, off_20, mask=add_mask)
    v21 = ct.gather(r21_ptr, off_21, mask=add_mask)
    v22 = ct.gather(r22_ptr, off_22, mask=add_mask)

    # bf16-rounded sequential add chain.
    residual = ct.astype(ct.astype(v0, ct.float32) + ct.astype(v1, ct.float32), ct.bfloat16)
    residual = ct.astype(ct.astype(residual, ct.float32) + ct.astype(v2, ct.float32), ct.bfloat16)
    residual = ct.astype(ct.astype(residual, ct.float32) + ct.astype(v3, ct.float32), ct.bfloat16)
    residual = ct.astype(ct.astype(residual, ct.float32) + ct.astype(v4, ct.float32), ct.bfloat16)
    residual = ct.astype(ct.astype(residual, ct.float32) + ct.astype(v5, ct.float32), ct.bfloat16)
    residual = ct.astype(ct.astype(residual, ct.float32) + ct.astype(v6, ct.float32), ct.bfloat16)
    residual = ct.astype(ct.astype(residual, ct.float32) + ct.astype(v7, ct.float32), ct.bfloat16)
    residual = ct.astype(ct.astype(residual, ct.float32) + ct.astype(v8, ct.float32), ct.bfloat16)
    residual = ct.astype(ct.astype(residual, ct.float32) + ct.astype(v9, ct.float32), ct.bfloat16)
    residual = ct.astype(ct.astype(residual, ct.float32) + ct.astype(v10, ct.float32), ct.bfloat16)
    residual = ct.astype(ct.astype(residual, ct.float32) + ct.astype(v11, ct.float32), ct.bfloat16)
    residual = ct.astype(ct.astype(residual, ct.float32) + ct.astype(v12, ct.float32), ct.bfloat16)
    residual = ct.astype(ct.astype(residual, ct.float32) + ct.astype(v13, ct.float32), ct.bfloat16)
    residual = ct.astype(ct.astype(residual, ct.float32) + ct.astype(v14, ct.float32), ct.bfloat16)
    residual = ct.astype(ct.astype(residual, ct.float32) + ct.astype(v15, ct.float32), ct.bfloat16)
    residual = ct.astype(ct.astype(residual, ct.float32) + ct.astype(v16, ct.float32), ct.bfloat16)
    residual = ct.astype(ct.astype(residual, ct.float32) + ct.astype(v17, ct.float32), ct.bfloat16)
    residual = ct.astype(ct.astype(residual, ct.float32) + ct.astype(v18, ct.float32), ct.bfloat16)
    residual = ct.astype(ct.astype(residual, ct.float32) + ct.astype(v19, ct.float32), ct.bfloat16)
    residual = ct.astype(ct.astype(residual, ct.float32) + ct.astype(v20, ct.float32), ct.bfloat16)
    residual = ct.astype(ct.astype(residual, ct.float32) + ct.astype(v21, ct.float32), ct.bfloat16)
    residual = ct.astype(ct.astype(residual, ct.float32) + ct.astype(v22, ct.float32), ct.bfloat16)
    add_value = ct.astype(ct.astype(residual, ct.float32) + ct.astype(dense_bf16, ct.float32), ct.bfloat16)

    ct.scatter(add_out_ptr, add_offsets, add_value, mask=add_mask)


@oracle_impl(hardware="B200", point="86b8300f", BLOCK_K=1024)
def oracle_forward(inputs, *, BLOCK_K: int):
    (
        arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9,
        arg10, arg11, arg12, arg13, arg14, arg15, arg16, arg17, arg18, arg19,
        arg20, arg21, arg22, arg23, arg24, arg25, arg26, arg27, arg28, arg29,
    ) = inputs
    device = arg23.device

    residuals = [arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9,
                 arg10, arg11, arg12, arg13, arg14, arg15, arg16, arg17, arg18,
                 arg19, arg20, arg21, arg22]
    r_flats = [r.contiguous().view(-1) for r in residuals]

    mask_flat = arg23.contiguous().view(-1)
    fill_flat = arg24.view(1)
    source_flat = arg25.contiguous().view(-1)
    centered_source_flat = arg26.contiguous().view(-1)
    mean_flat = arg27.view(C)
    invstd_flat = arg28.view(C)
    weight_flat = arg29.view(C)

    n_flat = N * C * HW
    add_n_flat = N * SLICE_C * HW
    dense_out_flat = torch.empty(n_flat, device=device, dtype=torch.bfloat16)
    add_out_flat = torch.empty(add_n_flat, device=device, dtype=torch.bfloat16)
    sum_out = torch.empty((C,), device=device, dtype=torch.float32)
    scale_grad = torch.empty((C,), device=device, dtype=torch.float32)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (C, 1, 1), _bn_tail_kernel,
        (
            *r_flats,
            mask_flat, fill_flat, source_flat, centered_source_flat,
            mean_flat, invstd_flat, weight_flat,
            sum_out, scale_grad, dense_out_flat, add_out_flat,
            C, HW, K_TOTAL, SLICE_START, SLICE_C, SCALE, BLOCK_K,
            *R_CHANNELS,
        ),
    )
    dense_out = dense_out_flat.view(N, C, H, W)
    add_out = add_out_flat.view(N, SLICE_C, H, W)
    return sum_out, scale_grad, dense_out, add_out
