"""cuTile port of sum_sum_0fbee3d7ac79: DenseNet BN backward tail.

Mirrors the Triton oracle: a single kernel with one program per output
channel that folds the masked producer, sibling reductions, BN-backward
epilogue, and residual slice-add for channels [288:320] in one shot.

Because cuTile has no runtime branch on `c`, we dispatch two launches:
one for channels [0, SLICE_START) that only writes dense_out, and one
for channels [SLICE_START, C) that also writes add_out.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


N = 4
C = 320
H = 28
W = 28
HW = H * W  # 784
K_TOTAL = N * HW  # 3136
SLICE_START = 288
SLICE_C = 32
SCALE = 0.00031887755102040814
HW_PAD = 1024  # next pow2 >= HW=784


def _bf16_add(a, b):
    return ct.astype(ct.astype(a, ct.float32) + ct.astype(b, ct.float32), ct.bfloat16)


@ct.kernel
def _bn_tail_kernel_no_add(
    mask_ptr,
    fill_ptr,
    source_ptr,
    centered_ptr,
    mean_ptr,
    invstd_ptr,
    weight_ptr,
    sum_out_ptr,
    scale_grad_ptr,
    dense_out_ptr,
    N_: ct.Constant[int],
    HW_PAD_: ct.Constant[int],
    HW_VALID_: ct.Constant[int],
    SCALE_: ct.Constant[float],
):
    c = ct.bid(0)

    fill_tile = ct.load(fill_ptr, index=(0,), shape=(1,))
    fill_bf_3d = ct.reshape(fill_tile, (1, 1, 1))

    mask_bf = ct.load(mask_ptr, index=(0, c, 0), shape=(N_, 1, HW_PAD_), padding_mode=ct.PaddingMode.ZERO)
    source_bf = ct.load(source_ptr, index=(0, c, 0), shape=(N_, 1, HW_PAD_), padding_mode=ct.PaddingMode.ZERO)
    centered_bf = ct.load(centered_ptr, index=(0, c, 0), shape=(N_, 1, HW_PAD_), padding_mode=ct.PaddingMode.ZERO)

    zero_bf = ct.zeros((N_, 1, HW_PAD_), dtype=ct.bfloat16)
    where_bf = ct.where(mask_bf <= zero_bf, fill_bf_3d, source_bf)
    where_f = ct.astype(where_bf, ct.float32)

    mean_1 = ct.load(mean_ptr, index=(c,), shape=(1,))
    mean_bcast = ct.reshape(mean_1, (1, 1, 1))
    centered_f = ct.astype(centered_bf, ct.float32) - mean_bcast

    hw_idx = ct.arange(HW_PAD_, dtype=ct.int32)
    active_1d = hw_idx < HW_VALID_
    active_3d = ct.reshape(active_1d, (1, 1, HW_PAD_))
    zero_f = ct.zeros((N_, 1, HW_PAD_), dtype=ct.float32)
    where_active = ct.where(active_3d, where_f, zero_f)
    centered_active = ct.where(active_3d, centered_f, zero_f)
    dot = where_active * centered_active

    sum_value = ct.sum(where_active)
    dot_value = ct.sum(dot)

    invstd_1 = ct.load(invstd_ptr, index=(c,), shape=(1,))
    weight_1 = ct.load(weight_ptr, index=(c,), shape=(1,))
    sum_tile = ct.full((1,), sum_value, dtype=ct.float32)
    dot_tile = ct.full((1,), dot_value, dtype=ct.float32)
    ct.store(sum_out_ptr, index=(c,), tile=sum_tile)
    ct.store(scale_grad_ptr, index=(c,), tile=dot_tile * invstd_1)

    dot_scaled = dot_tile * SCALE_
    invstd_sq = invstd_1 * invstd_1
    variance_term_bc = ct.reshape(dot_scaled * invstd_sq, (1, 1, 1))
    mean_term_bc = ct.reshape(sum_tile * SCALE_, (1, 1, 1))
    output_scale_bc = ct.reshape(invstd_1 * weight_1, (1, 1, 1))

    corrected = where_active - centered_active * variance_term_bc
    centered_grad = corrected - mean_term_bc
    dense_f = centered_grad * output_scale_bc
    dense_bf = ct.astype(dense_f, ct.bfloat16)
    dense_bf_masked = ct.where(active_3d, dense_bf, ct.zeros((N_, 1, HW_PAD_), dtype=ct.bfloat16))
    ct.store(dense_out_ptr, index=(0, c, 0), tile=dense_bf_masked)


@ct.kernel
def _bn_tail_kernel_slice(
    r0_ptr,
    r1_ptr,
    r2_ptr,
    r3_ptr,
    r4_ptr,
    r5_ptr,
    mask_ptr,
    fill_ptr,
    source_ptr,
    centered_ptr,
    mean_ptr,
    invstd_ptr,
    weight_ptr,
    sum_out_ptr,
    scale_grad_ptr,
    dense_out_ptr,
    add_out_ptr,
    SLICE_START_: ct.Constant[int],
    N_: ct.Constant[int],
    HW_PAD_: ct.Constant[int],
    HW_VALID_: ct.Constant[int],
    SCALE_: ct.Constant[float],
):
    slice_c = ct.bid(0)
    c = slice_c + SLICE_START_

    fill_tile = ct.load(fill_ptr, index=(0,), shape=(1,))
    fill_bf_3d = ct.reshape(fill_tile, (1, 1, 1))

    mask_bf = ct.load(mask_ptr, index=(0, c, 0), shape=(N_, 1, HW_PAD_), padding_mode=ct.PaddingMode.ZERO)
    source_bf = ct.load(source_ptr, index=(0, c, 0), shape=(N_, 1, HW_PAD_), padding_mode=ct.PaddingMode.ZERO)
    centered_bf = ct.load(centered_ptr, index=(0, c, 0), shape=(N_, 1, HW_PAD_), padding_mode=ct.PaddingMode.ZERO)

    zero_bf = ct.zeros((N_, 1, HW_PAD_), dtype=ct.bfloat16)
    where_bf = ct.where(mask_bf <= zero_bf, fill_bf_3d, source_bf)
    where_f = ct.astype(where_bf, ct.float32)

    mean_1 = ct.load(mean_ptr, index=(c,), shape=(1,))
    mean_bcast = ct.reshape(mean_1, (1, 1, 1))
    centered_f = ct.astype(centered_bf, ct.float32) - mean_bcast

    hw_idx = ct.arange(HW_PAD_, dtype=ct.int32)
    active_1d = hw_idx < HW_VALID_
    active_3d = ct.reshape(active_1d, (1, 1, HW_PAD_))
    zero_f = ct.zeros((N_, 1, HW_PAD_), dtype=ct.float32)
    where_active = ct.where(active_3d, where_f, zero_f)
    centered_active = ct.where(active_3d, centered_f, zero_f)
    dot = where_active * centered_active

    sum_value = ct.sum(where_active)
    dot_value = ct.sum(dot)

    invstd_1 = ct.load(invstd_ptr, index=(c,), shape=(1,))
    weight_1 = ct.load(weight_ptr, index=(c,), shape=(1,))
    sum_tile = ct.full((1,), sum_value, dtype=ct.float32)
    dot_tile = ct.full((1,), dot_value, dtype=ct.float32)
    ct.store(sum_out_ptr, index=(c,), tile=sum_tile)
    ct.store(scale_grad_ptr, index=(c,), tile=dot_tile * invstd_1)

    dot_scaled = dot_tile * SCALE_
    invstd_sq = invstd_1 * invstd_1
    variance_term_bc = ct.reshape(dot_scaled * invstd_sq, (1, 1, 1))
    mean_term_bc = ct.reshape(sum_tile * SCALE_, (1, 1, 1))
    output_scale_bc = ct.reshape(invstd_1 * weight_1, (1, 1, 1))

    corrected = where_active - centered_active * variance_term_bc
    centered_grad = corrected - mean_term_bc
    dense_f = centered_grad * output_scale_bc
    dense_bf = ct.astype(dense_f, ct.bfloat16)
    dense_bf_masked = ct.where(active_3d, dense_bf, ct.zeros((N_, 1, HW_PAD_), dtype=ct.bfloat16))
    ct.store(dense_out_ptr, index=(0, c, 0), tile=dense_bf_masked)

    r0_v = ct.load(r0_ptr, index=(0, c, 0), shape=(N_, 1, HW_PAD_), padding_mode=ct.PaddingMode.ZERO)
    r1_v = ct.load(r1_ptr, index=(0, c, 0), shape=(N_, 1, HW_PAD_), padding_mode=ct.PaddingMode.ZERO)
    r2_v = ct.load(r2_ptr, index=(0, c, 0), shape=(N_, 1, HW_PAD_), padding_mode=ct.PaddingMode.ZERO)
    r3_v = ct.load(r3_ptr, index=(0, c, 0), shape=(N_, 1, HW_PAD_), padding_mode=ct.PaddingMode.ZERO)
    r4_v = ct.load(r4_ptr, index=(0, c, 0), shape=(N_, 1, HW_PAD_), padding_mode=ct.PaddingMode.ZERO)
    r5_v = ct.load(r5_ptr, index=(0, c, 0), shape=(N_, 1, HW_PAD_), padding_mode=ct.PaddingMode.ZERO)

    residual = _bf16_add(r0_v, r1_v)
    residual = _bf16_add(residual, r2_v)
    residual = _bf16_add(residual, r3_v)
    residual = _bf16_add(residual, r4_v)
    residual = _bf16_add(residual, r5_v)
    add_value = _bf16_add(residual, dense_bf)
    add_value_masked = ct.where(active_3d, add_value, ct.zeros((N_, 1, HW_PAD_), dtype=ct.bfloat16))
    ct.store(add_out_ptr, index=(0, slice_c, 0), tile=add_value_masked)


@oracle_impl(hardware="B200", point="08ef2101", BLOCK_K=4096)
def oracle_forward(inputs, *, BLOCK_K: int):
    (
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1,
        arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1,
    ) = inputs
    del BLOCK_K
    device = arg8_1.device

    sum_out = torch.empty_strided((C,), (1,), device=device, dtype=torch.float32)
    scale_grad = torch.empty_strided((C,), (1,), device=device, dtype=torch.float32)
    dense_out = torch.empty_strided(
        (N, C, H, W), (C * HW, HW, W, 1),
        device=device, dtype=torch.bfloat16,
    )
    add_out = torch.empty_strided(
        (N, SLICE_C, H, W), (SLICE_C * HW, HW, W, 1),
        device=device, dtype=torch.bfloat16,
    )

    arg0_v = arg0_1.view(N, 512, HW)
    arg1_v = arg1_1.view(N, 480, HW)
    arg2_v = arg2_1.view(N, 448, HW)
    arg3_v = arg3_1.view(N, 416, HW)
    arg4_v = arg4_1.view(N, 384, HW)
    arg5_v = arg5_1.view(N, 352, HW)
    mask_v = arg6_1.view(N, C, HW)
    source_v = arg8_1.view(N, C, HW)
    centered_v = arg9_1.view(N, C, HW)
    dense_out_v = dense_out.view(N, C, HW)
    add_out_v = add_out.view(N, SLICE_C, HW)
    fill_1d = arg7_1.view(1)
    mean_1d = arg10_1.view(C)
    invstd_1d = arg11_1.view(C)
    weight_1d = arg12_1.view(C)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (SLICE_START, 1, 1), _bn_tail_kernel_no_add,
        (mask_v, fill_1d, source_v, centered_v,
         mean_1d, invstd_1d, weight_1d,
         sum_out, scale_grad, dense_out_v,
         N, HW_PAD, HW, SCALE),
    )
    ct.launch(
        stream, (C - SLICE_START, 1, 1), _bn_tail_kernel_slice,
        (arg0_v, arg1_v, arg2_v, arg3_v, arg4_v, arg5_v,
         mask_v, fill_1d, source_v, centered_v,
         mean_1d, invstd_1d, weight_1d,
         sum_out, scale_grad, dense_out_v, add_out_v,
         SLICE_START, N, HW_PAD, HW, SCALE),
    )
    return sum_out, scale_grad, dense_out, add_out
