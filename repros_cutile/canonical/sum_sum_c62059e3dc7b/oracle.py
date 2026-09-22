"""cuTile port of sum_sum_c62059e3dc7b: DenseNet transition BN backward + pool bwd.

Single-kernel design mirrors Triton's structure: grid = (C,); each program
handles one channel and all 4*7*7=196 spatial positions. It performs:
  - the sliced NCHW load of the 16 residual inputs
  - the mask-and-where masked producer
  - both f32 channel reductions (sum_where, sum_centered) using ct.sum
  - the BN-backward dense value
  - the residual add + 2x2 avg_pool2d_backward store (spread each value to
    a 2x2 block in the output).

Pool backward stores the value at 4 positions (h*2, h*2+1), (w*2, w*2+1) with
value * 0.25. Uses scatter for irregular writes.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


N = 4
C = 512
H = 7
W = 7
HW = H * W
R = N * HW  # 196
OUT_H = 14
OUT_W = 14
OUT_HW = OUT_H * OUT_W
SCALE = 0.00510204081632653


def _next_pow2(v):
    return 1 << (int(v) - 1).bit_length()


@ct.kernel
def _densenet_pool_kernel(
    r0_ptr, r1_ptr, r2_ptr, r3_ptr, r4_ptr, r5_ptr, r6_ptr, r7_ptr,
    r8_ptr, r9_ptr, r10_ptr, r11_ptr, r12_ptr, r13_ptr, r14_ptr, r15_ptr,
    mask_ptr,           # bf16 (N, C, H, W)
    fill_ptr,           # bf16 (1,)
    source_ptr,         # bf16 (N, C, H, W)
    centered_source_ptr,# bf16 (N, C, H, W)
    mean_ptr,           # f32 (C,)
    invstd_ptr,         # f32 (C,)
    weight_ptr,         # f32 (C,)
    sum_out_ptr,        # f32 (C,)
    scale_grad_ptr,     # f32 (C,)
    pool_out_ptr,       # bf16 flat (N*C*OUT_HW,)
    BLOCK_R: ct.Constant[int],
):
    c = ct.bid(0)
    rows = ct.arange(BLOCK_R, dtype=ct.int32)
    active = rows < R

    n = rows // HW
    spatial = rows - n * HW
    # dense_offsets = n * (C * HW) + c * HW + spatial (mask-in-vocab-space)
    dense_offsets = n * (C * HW) + c * HW + spatial

    zero_int = ct.zeros((BLOCK_R,), dtype=ct.int32)
    zero_f = ct.zeros((BLOCK_R,), dtype=ct.float32)
    zero_bf = ct.zeros((BLOCK_R,), dtype=ct.bfloat16)
    safe_off = ct.where(active, dense_offsets, zero_int)

    mask_value = ct.gather(mask_ptr, safe_off)
    fill_value_1 = ct.load(fill_ptr, index=(0,), shape=(1,))
    fill_value = ct.reshape(fill_value_1, (1,)) + zero_bf  # broadcast to (BLOCK_R,)
    source_value = ct.gather(source_ptr, safe_off)
    mask_f = ct.astype(mask_value, ct.float32)
    zero_scalar = ct.zeros((BLOCK_R,), dtype=ct.float32)
    le_zero = mask_f <= zero_scalar
    where_bf16 = ct.where(le_zero, fill_value, source_value)
    where_f32 = ct.where(active, ct.astype(where_bf16, ct.float32), zero_f)

    centered_source = ct.astype(ct.gather(centered_source_ptr, safe_off), ct.float32)
    mean_1 = ct.load(mean_ptr, index=(c,), shape=(1,))
    mean_bcast = ct.reshape(mean_1, (1,)) + zero_f  # broadcast
    centered = ct.where(active, centered_source - mean_bcast, zero_f)

    product = where_f32 * centered
    sum_where = ct.sum(where_f32, axis=0)      # scalar
    sum_centered = ct.sum(product, axis=0)      # scalar

    invstd_1 = ct.load(invstd_ptr, index=(c,), shape=(1,))
    weight_1 = ct.load(weight_ptr, index=(c,), shape=(1,))
    invstd = ct.reshape(invstd_1, (1,))
    weight = ct.reshape(weight_1, (1,))
    mean_term = sum_where * SCALE
    prod_scaled = sum_centered * SCALE
    invstd_sq = invstd * invstd
    variance_term = prod_scaled * invstd_sq
    output_scale = invstd * weight

    # Broadcast scalars to (BLOCK_R,)
    variance_term_b = variance_term + zero_f
    mean_term_b = mean_term + zero_f
    output_scale_b = output_scale + zero_f

    corrected = where_f32 - centered * variance_term_b
    centered_grad = corrected - mean_term_b
    bn_bf16 = ct.astype(centered_grad * output_scale_b, ct.bfloat16)

    # Sixteen residual loads with per-arg C dim (arg{i} has slightly different C).
    # local_offset = c * HW + spatial, base = n * (r_c * HW) + local_offset
    local = c * HW + spatial

    def _load(rptr, r_c):
        off = n * (r_c * HW) + local
        safe = ct.where(active, off, zero_int)
        return ct.gather(rptr, safe)

    def _bf_add(a, b):
        a_f = ct.astype(a, ct.float32)
        b_f = ct.astype(b, ct.float32)
        return ct.astype(a_f + b_f, ct.bfloat16)

    v0 = _load(r0_ptr, 1024)
    v1 = _load(r1_ptr, 992)
    residual = _bf_add(v0, v1)
    v2 = _load(r2_ptr, 960);   residual = _bf_add(residual, v2)
    v3 = _load(r3_ptr, 928);   residual = _bf_add(residual, v3)
    v4 = _load(r4_ptr, 896);   residual = _bf_add(residual, v4)
    v5 = _load(r5_ptr, 864);   residual = _bf_add(residual, v5)
    v6 = _load(r6_ptr, 832);   residual = _bf_add(residual, v6)
    v7 = _load(r7_ptr, 800);   residual = _bf_add(residual, v7)
    v8 = _load(r8_ptr, 768);   residual = _bf_add(residual, v8)
    v9 = _load(r9_ptr, 736);   residual = _bf_add(residual, v9)
    v10 = _load(r10_ptr, 704); residual = _bf_add(residual, v10)
    v11 = _load(r11_ptr, 672); residual = _bf_add(residual, v11)
    v12 = _load(r12_ptr, 640); residual = _bf_add(residual, v12)
    v13 = _load(r13_ptr, 608); residual = _bf_add(residual, v13)
    v14 = _load(r14_ptr, 576); residual = _bf_add(residual, v14)
    v15 = _load(r15_ptr, 544); residual = _bf_add(residual, v15)
    pool_parent = _bf_add(residual, bn_bf16)
    pool_value = ct.astype(ct.astype(pool_parent, ct.float32) * 0.25, ct.bfloat16)

    h_rows = spatial // W
    w_rows = spatial - h_rows * W
    # out_base = n * (C * OUT_HW) + c * OUT_HW + h*2*OUT_W + w*2
    out_base = n * (C * OUT_HW) + c * OUT_HW + h_rows * (2 * OUT_W) + w_rows * 2

    ct.store(sum_out_ptr, index=(c,), tile=ct.reshape(sum_where, (1,)))
    ct.store(scale_grad_ptr, index=(c,), tile=ct.reshape(sum_centered * invstd, (1,)))
    # Pool backward: write pool_value at (out_base), (out_base+1), (out_base+OUT_W), (out_base+OUT_W+1)
    ct.scatter(pool_out_ptr, (ct.where(active, out_base, zero_int),), pool_value, mask=active)
    ct.scatter(pool_out_ptr, (ct.where(active, out_base + 1, zero_int),), pool_value, mask=active)
    ct.scatter(pool_out_ptr, (ct.where(active, out_base + OUT_W, zero_int),), pool_value, mask=active)
    ct.scatter(pool_out_ptr, (ct.where(active, out_base + OUT_W + 1, zero_int),), pool_value, mask=active)


@oracle_impl(hardware="B200", point="6f10577d", BLOCK_R=256)
def oracle_forward(inputs, *, BLOCK_R: int):
    (
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1,
        arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1,
        arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1,
    ) = inputs
    device = arg16_1.device

    sum_out = torch.empty((C,), device=device, dtype=torch.float32)
    scale_grad = torch.empty((C,), device=device, dtype=torch.float32)
    pool_out = torch.empty((N, C, OUT_H, OUT_W), device=device, dtype=torch.bfloat16)

    fill_1d = arg17_1.view(1)
    mean_1d = arg20_1.view(C)

    stream = torch.cuda.current_stream()
    # Flatten input tensors (all NCHW contiguous) for gather-based access.
    ct.launch(
        stream,
        (C, 1, 1),
        _densenet_pool_kernel,
        (arg0_1.contiguous().view(-1), arg1_1.contiguous().view(-1),
         arg2_1.contiguous().view(-1), arg3_1.contiguous().view(-1),
         arg4_1.contiguous().view(-1), arg5_1.contiguous().view(-1),
         arg6_1.contiguous().view(-1), arg7_1.contiguous().view(-1),
         arg8_1.contiguous().view(-1), arg9_1.contiguous().view(-1),
         arg10_1.contiguous().view(-1), arg11_1.contiguous().view(-1),
         arg12_1.contiguous().view(-1), arg13_1.contiguous().view(-1),
         arg14_1.contiguous().view(-1), arg15_1.contiguous().view(-1),
         arg16_1.contiguous().view(-1), fill_1d,
         arg18_1.contiguous().view(-1), arg19_1.contiguous().view(-1),
         mean_1d, arg21_1, arg22_1,
         sum_out, scale_grad, pool_out.view(-1),
         BLOCK_R),
    )
    return sum_out, scale_grad, pool_out
