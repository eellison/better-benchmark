"""cuTile port of sum_sum_4efc39f71987: DenseNet BN-backward + slice residual add.

Similar to sum_sum_936e8304ff14 but with C=672, and 11 residual inputs.
Two shape points:
  e99e1f22: H=W=14 (HW=196), N=4, TOTAL_SPATIAL=784
  0c2cfaff: H=W=7  (HW=49),  N=4, TOTAL_SPATIAL=196
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


C = 672
SLICE_START = 640
SLICE_C = 32


@ct.kernel
def _bn_reduce_and_dense_kernel(
    mask_ptr,         # bf16 (C, N*HW)
    fill_ptr,         # bf16 (1,)
    source_ptr,       # bf16 (C, N*HW)
    centered_src_ptr, # bf16 (C, N*HW)
    mean_ptr,         # f32 (C,)
    invstd_ptr,       # f32 (C,)
    weight_ptr,       # f32 (C,)
    sum_where_ptr,    # f32 (C,)
    mul8_ptr,         # f32 (C,)
    dense_ptr,        # bf16 (C, N*HW)
    TOTAL_C: ct.Constant[int],
    SCALE_C: ct.Constant[float],
    BLOCK_R_C: ct.Constant[int],
):
    c = ct.bid(0)

    mask_val = ct.load(
        mask_ptr, index=(c, 0), shape=(1, BLOCK_R_C),
        padding_mode=ct.PaddingMode.ZERO,
    )
    source_val = ct.load(
        source_ptr, index=(c, 0), shape=(1, BLOCK_R_C),
        padding_mode=ct.PaddingMode.ZERO,
    )
    fill = ct.load(fill_ptr, index=(0,), shape=(1,))
    fill_bc = ct.reshape(fill, (1, 1))
    zero_bf16_tile = ct.zeros((1, BLOCK_R_C), dtype=ct.bfloat16)
    where_bf16 = ct.where(mask_val <= zero_bf16_tile, fill_bc, source_val)
    where_f = ct.astype(where_bf16, ct.float32)
    # Mask OOB
    col_idx = ct.arange(BLOCK_R_C, dtype=ct.int32)
    valid = ct.reshape(col_idx < TOTAL_C, (1, BLOCK_R_C))
    zero_2d = ct.zeros((1, BLOCK_R_C), dtype=ct.float32)
    where_f = ct.where(valid, where_f, zero_2d)

    centered_src = ct.astype(
        ct.load(
            centered_src_ptr, index=(c, 0), shape=(1, BLOCK_R_C),
            padding_mode=ct.PaddingMode.ZERO,
        ),
        ct.float32,
    )
    mean_1d = ct.load(mean_ptr, index=(c,), shape=(1,))
    mean_bc = ct.reshape(mean_1d, (1, 1))
    centered = ct.where(valid, centered_src - mean_bc, zero_2d)

    prod = where_f * centered
    sum_where = ct.sum(where_f, axis=1, keepdims=True)
    sum_mul = ct.sum(prod, axis=1, keepdims=True)

    invstd_1d = ct.load(invstd_ptr, index=(c,), shape=(1,))
    weight_1d = ct.load(weight_ptr, index=(c,), shape=(1,))
    invstd = ct.reshape(invstd_1d, (1, 1))
    weight = ct.reshape(weight_1d, (1, 1))
    mean_term = sum_where * SCALE_C
    variance_term = (sum_mul * SCALE_C) * (invstd * invstd)
    out_weight = invstd * weight

    grad = (where_f - centered * variance_term - mean_term) * out_weight

    sum_where_1d = ct.reshape(sum_where, (1,))
    mul8 = ct.reshape(sum_mul * invstd, (1,))
    ct.store(sum_where_ptr, index=(c,), tile=sum_where_1d)
    ct.store(mul8_ptr, index=(c,), tile=mul8)
    ct.store(dense_ptr, index=(c, 0), tile=ct.astype(grad, ct.bfloat16))


def _to_c_nhw(t, N, H, W, HW):
    return t.permute(1, 0, 2, 3).contiguous().view(-1, N * HW)


def _launch(inputs, *, H, W, BLOCK_R, SCALE):
    (
        arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9,
        arg10, arg11, arg12, arg13, arg14, arg15, arg16, arg17,
    ) = inputs
    device = arg13.device
    N = 4
    HW = H * W
    TOTAL_SPATIAL = N * HW

    # arg11 = mask (bf16 4,672,H,W), arg12 = fill (bf16 []), arg13 = source,
    # arg14 = centered_source, arg15 = mean, arg16 = invstd, arg17 = weight.
    mask_c = _to_c_nhw(arg11, N, H, W, HW)
    source_c = _to_c_nhw(arg13, N, H, W, HW)
    centered_src_c = _to_c_nhw(arg14, N, H, W, HW)

    dense_c = torch.empty((C, TOTAL_SPATIAL), device=device, dtype=torch.bfloat16)
    mean_1d = arg15.view(C)
    sum_where = torch.empty((C,), device=device, dtype=torch.float32)
    mul8 = torch.empty((C,), device=device, dtype=torch.float32)
    fill_view = arg12.view(1)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (C, 1, 1),
        _bn_reduce_and_dense_kernel,
        (
            mask_c, fill_view, source_c, centered_src_c,
            mean_1d, arg16, arg17,
            sum_where, mul8, dense_c,
            TOTAL_SPATIAL, SCALE, BLOCK_R,
        ),
    )

    dense_out = dense_c.view(C, N, H, W).permute(1, 0, 2, 3).contiguous()
    dense_slice = dense_out[:, SLICE_START:C]

    # 11 residual inputs: arg0..arg10, each with different C but same H, W, N.
    residual = arg0[:, SLICE_START:SLICE_START + SLICE_C]
    for arg in (arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10):
        residual = residual + arg[:, SLICE_START:SLICE_START + SLICE_C]
    add_out = residual + dense_slice

    return sum_where, mul8, dense_out, add_out


# e99e1f22: H=W=14 (HW=196, TOTAL=784)
@oracle_impl(hardware="B200", point="e99e1f22", H=14, W=14, BLOCK_R=1024)
# 0c2cfaff: H=W=7 (HW=49, TOTAL=196)
@oracle_impl(hardware="B200", point="0c2cfaff", H=7, W=7, BLOCK_R=256)
def oracle_forward(inputs, *, H: int, W: int, BLOCK_R: int):
    return _launch(inputs, H=H, W=W, BLOCK_R=BLOCK_R, SCALE=0.0012755102040816326)
