"""cuTile port of sum_sum_cc8aafa6ae2d: DenseNet BN-backward + 11-input residual add.

Single-kernel design mirrors Triton: grid = (C,); each program handles one
channel and all N*HW=3136 spatial positions. It performs the mask/where +
per-channel reductions + BN backward + 11-input residual slice add.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


N = 4
C = 160
H = 28
W = 28
HW = H * W
R = N * HW  # 3136
SLICE_START = 128
SLICE_C = 32
SCALE = 0.00031887755102040814


@ct.kernel
def _tail_kernel(
    r0_ptr, r1_ptr, r2_ptr, r3_ptr, r4_ptr, r5_ptr, r6_ptr, r7_ptr,
    r8_ptr, r9_ptr, r10_ptr,
    mask_input_ptr,       # bf16 (N, C, H, W) flat
    fill_ptr,             # bf16 (1,)
    source_ptr,           # bf16 (N, C, H, W) flat
    centered_source_ptr,  # bf16 (N, C, H, W) flat
    mean_ptr,             # f32 (C,)
    invstd_ptr,           # f32 (C,)
    weight_ptr,           # f32 (C,)
    sum_out_ptr,          # f32 (C,)
    scale_grad_ptr,       # f32 (C,)
    dense_out_ptr,        # bf16 (N, C, H, W) flat
    add_out_ptr,          # bf16 (N, SLICE_C, H, W) flat
    BLOCK_R: ct.Constant[int],
):
    c = ct.bid(0)
    rows = ct.arange(BLOCK_R, dtype=ct.int32)
    active = rows < R

    n = rows // HW
    spatial = rows - n * HW
    zero_int = ct.zeros((BLOCK_R,), dtype=ct.int32)
    zero_f = ct.zeros((BLOCK_R,), dtype=ct.float32)
    zero_bf = ct.zeros((BLOCK_R,), dtype=ct.bfloat16)

    dense_offsets = n * (C * HW) + c * HW + spatial
    safe_off = ct.where(active, dense_offsets, zero_int)

    mask_value = ct.gather(mask_input_ptr, safe_off)
    fill_v_1 = ct.load(fill_ptr, index=(0,), shape=(1,))
    fill_v = ct.reshape(fill_v_1, (1,)) + zero_bf
    source_value = ct.gather(source_ptr, safe_off)
    mask_f = ct.astype(mask_value, ct.float32)
    le_zero = mask_f <= zero_f
    where_bf16 = ct.where(le_zero, fill_v, source_value)
    where_f32 = ct.where(active, ct.astype(where_bf16, ct.float32), zero_f)

    centered_source = ct.astype(ct.gather(centered_source_ptr, safe_off), ct.float32)
    mean_1 = ct.load(mean_ptr, index=(c,), shape=(1,))
    mean_b = ct.reshape(mean_1, (1,)) + zero_f
    centered = ct.where(active, centered_source - mean_b, zero_f)

    product = where_f32 * centered
    sum_where = ct.sum(where_f32, axis=0)     # scalar
    sum_centered = ct.sum(product, axis=0)     # scalar

    invstd_1 = ct.load(invstd_ptr, index=(c,), shape=(1,))
    weight_1 = ct.load(weight_ptr, index=(c,), shape=(1,))
    invstd = ct.reshape(invstd_1, (1,))
    weight = ct.reshape(weight_1, (1,))
    mean_term = sum_where * SCALE
    prod_scaled = sum_centered * SCALE
    invstd_sq = invstd * invstd
    variance_term = prod_scaled * invstd_sq
    output_scale = invstd * weight

    variance_term_b = variance_term + zero_f
    mean_term_b = mean_term + zero_f
    output_scale_b = output_scale + zero_f

    corrected = where_f32 - centered * variance_term_b
    centered_grad = corrected - mean_term_b
    dense_bf16 = ct.astype(centered_grad * output_scale_b, ct.bfloat16)

    ct.store(sum_out_ptr, index=(c,), tile=ct.reshape(sum_where, (1,)))
    ct.store(scale_grad_ptr, index=(c,), tile=ct.reshape(sum_centered * invstd, (1,)))
    ct.scatter(dense_out_ptr, (safe_off,), dense_bf16, mask=active)

    # Residual add: only when c >= SLICE_START
    in_slice = c >= SLICE_START
    if in_slice:
        slice_c = c - SLICE_START
        add_offsets = n * (SLICE_C * HW) + slice_c * HW + spatial
        safe_add = ct.where(active, add_offsets, zero_int)
        local = c * HW + spatial

        def _load(rptr, r_c):
            off = n * (r_c * HW) + local
            safe = ct.where(active, off, zero_int)
            return ct.gather(rptr, safe)

        def _bf_add(a, b):
            a_f = ct.astype(a, ct.float32)
            b_f = ct.astype(b, ct.float32)
            return ct.astype(a_f + b_f, ct.bfloat16)

        v0 = _load(r0_ptr, 512)
        v1 = _load(r1_ptr, 480)
        residual = _bf_add(v0, v1)
        v2 = _load(r2_ptr, 448);  residual = _bf_add(residual, v2)
        v3 = _load(r3_ptr, 416);  residual = _bf_add(residual, v3)
        v4 = _load(r4_ptr, 384);  residual = _bf_add(residual, v4)
        v5 = _load(r5_ptr, 352);  residual = _bf_add(residual, v5)
        v6 = _load(r6_ptr, 320);  residual = _bf_add(residual, v6)
        v7 = _load(r7_ptr, 288);  residual = _bf_add(residual, v7)
        v8 = _load(r8_ptr, 256);  residual = _bf_add(residual, v8)
        v9 = _load(r9_ptr, 224);  residual = _bf_add(residual, v9)
        v10 = _load(r10_ptr, 192);residual = _bf_add(residual, v10)
        add_value = _bf_add(residual, dense_bf16)
        ct.scatter(add_out_ptr, (safe_add,), add_value, mask=active)


@oracle_impl(hardware="B200", point="ad8a4af8", BLOCK_R=4096)
def oracle_forward(inputs, *, BLOCK_R: int):
    (
        a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10,
        a11, a12, a13, a14, a15, a16, a17,
    ) = inputs
    device = a13.device

    dense_out = torch.empty((N, C, H, W), device=device, dtype=torch.bfloat16)
    add_out = torch.empty((N, SLICE_C, H, W), device=device, dtype=torch.bfloat16)
    sum_out = torch.empty((C,), device=device, dtype=torch.float32)
    scale_grad = torch.empty((C,), device=device, dtype=torch.float32)

    fill_1d = a12.view(1)
    mean_1d = a15.view(C)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (C, 1, 1),
        _tail_kernel,
        (a0.contiguous().view(-1), a1.contiguous().view(-1),
         a2.contiguous().view(-1), a3.contiguous().view(-1),
         a4.contiguous().view(-1), a5.contiguous().view(-1),
         a6.contiguous().view(-1), a7.contiguous().view(-1),
         a8.contiguous().view(-1), a9.contiguous().view(-1),
         a10.contiguous().view(-1),
         a11.contiguous().view(-1), fill_1d,
         a13.contiguous().view(-1), a14.contiguous().view(-1),
         mean_1d, a16, a17,
         sum_out, scale_grad,
         dense_out.view(-1), add_out.view(-1),
         BLOCK_R),
    )
    return sum_out, scale_grad, dense_out, add_out
