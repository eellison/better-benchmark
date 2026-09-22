"""cuTile port of sum_sum_936e8304ff14: DenseNet BN-backward + slice residual add.

Two-step:
  1. BN-backward kernel: per-channel (C=192) reduce over N*H*W=3136 rows,
     compute grad_out (dense bf16 output).
  2. Slice add kernel: for slice channels 160:192, add 10 residual slices +
     dense's slice.

C=192 is not power of 2 (round to 256), HW=784 not pow-2 (round to 1024),
so we mask. Shapes: N=4, C=192, H=W=28, HW=784, TOTAL_SPATIAL=3136.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


N = 4
C = 192
H = 28
W = 28
HW = H * W  # 784
TOTAL_SPATIAL = N * HW  # 3136
BLOCK_R = 4096  # >= 3136, pow-2
SLICE_START = 160
SLICE_C = 32
SCALE = 0.00031887755102040814


@ct.kernel
def _bn_reduce_and_dense_kernel(
    mask_ptr,         # bf16 (C, N*HW) i.e. (192, 3136) — flattened NCHW
    fill_ptr,         # bf16 () scalar
    source_ptr,       # bf16 (C, N*HW)
    centered_src_ptr, # bf16 (C, N*HW)
    mean_ptr,         # f32 (C,)
    invstd_ptr,       # f32 (C,)
    weight_ptr,       # f32 (C,)
    sum_where_ptr,    # f32 (C,)
    mul8_ptr,         # f32 (C,)
    dense_ptr,        # bf16 (C, N*HW)
    C_C: ct.Constant[int],
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
    # where_bf16 = mask <= 0 ? fill : source (both bf16)
    where_bf16 = ct.where(mask_val <= ct.astype(ct.zeros((1, BLOCK_R_C), dtype=ct.bfloat16), ct.bfloat16), fill_bc, source_val)
    where_f = ct.astype(where_bf16, ct.float32)
    # Mask OOB positions to 0 before reduction.
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
    sum_where = ct.sum(where_f, axis=1, keepdims=True)  # (1,1)
    sum_mul = ct.sum(prod, axis=1, keepdims=True)       # (1,1)

    invstd_1d = ct.load(invstd_ptr, index=(c,), shape=(1,))
    weight_1d = ct.load(weight_ptr, index=(c,), shape=(1,))
    invstd = ct.reshape(invstd_1d, (1, 1))
    weight = ct.reshape(weight_1d, (1, 1))
    mean_term = sum_where * SCALE_C
    variance_term = (sum_mul * SCALE_C) * (invstd * invstd)
    out_weight = invstd * weight

    grad = (where_f - centered * variance_term - mean_term) * out_weight

    # Store sum_where, mul8=sum_mul*invstd, and grad (masked).
    # sum_where and mul8 are scalars — reshape (1,1) → (1,) → store 1 element.
    sum_where_1d = ct.reshape(sum_where, (1,))
    mul8 = ct.reshape(sum_mul * invstd, (1,))
    ct.store(sum_where_ptr, index=(c,), tile=sum_where_1d)
    ct.store(mul8_ptr, index=(c,), tile=mul8)
    ct.store(dense_ptr, index=(c, 0), tile=ct.astype(grad, ct.bfloat16))


@oracle_impl(hardware="B200", point="817f488a", BLOCK_R=BLOCK_R)
def oracle_forward(inputs, *, BLOCK_R: int):
    (
        arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9,
        arg10, arg11, arg12, arg13, arg14, arg15, arg16,
    ) = inputs
    device = arg10.device

    # arg10..arg13 are (N, C, H, W) with layout (C * HW, HW, W, 1) so C outer.
    # We want to reshape as (C, N*HW) for column-major reductions per channel.
    # Approach: permute (N, C, H, W) -> (C, N, H, W), then flatten to (C, N*HW).
    # Since memory is contiguous, this needs a permute which changes strides.
    # cuTile handles strided tensors, so we can permute without copy.
    def _to_c_nhw(t):
        # (N, C, H, W) contiguous — natural stride (C*H*W, H*W, W, 1).
        # We reshape by permuting: (C, N, H, W) with strides (H*W, C*H*W, W, 1),
        # then flatten last three: (C, N*H*W) with strides (H*W, ???). Actually
        # (C, N, H, W) permuted view is not contiguous in the last three dims.
        # We need a separate contiguous view.
        return t.permute(1, 0, 2, 3).contiguous().view(C, N * HW)

    mask_c = _to_c_nhw(arg10)
    source_c = _to_c_nhw(arg12)
    centered_src_c = _to_c_nhw(arg13)

    # Padded output buffer for dense (per-channel).
    dense_c = torch.empty((C, N * HW), device=device, dtype=torch.bfloat16)

    # Squeeze arg14 (1,C,1,1) to (C,)
    mean_1d = arg14.view(C)

    sum_where = torch.empty((C,), device=device, dtype=torch.float32)
    mul8 = torch.empty((C,), device=device, dtype=torch.float32)

    fill_view = arg11.view(1)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (C, 1, 1),
        _bn_reduce_and_dense_kernel,
        (
            mask_c, fill_view, source_c, centered_src_c,
            mean_1d, arg15, arg16,
            sum_where, mul8, dense_c,
            C, TOTAL_SPATIAL, SCALE, BLOCK_R,
        ),
    )

    # Reshape dense_c (C, N*HW) → (N, C, H, W) with contiguous stride
    # (C*HW, HW, W, 1). Permute back: (C, N, H, W) → (N, C, H, W).
    dense_out = dense_c.view(C, N, H, W).permute(1, 0, 2, 3).contiguous()

    # slice_add: channels 160:192 of dense_out, added with 10 residual slices.
    # This is a small tensor: (N=4, SLICE_C=32, H=28, W=28).
    dense_slice = dense_out[:, SLICE_START:C]

    # 10 residual inputs: arg0..arg9, each with different C but same H, W, N.
    # Each: (N, C_r, H, W) sliced to [:, 160:192].
    residual = arg0[:, SLICE_START:SLICE_START + SLICE_C]
    for arg in (arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9):
        residual = residual + arg[:, SLICE_START:SLICE_START + SLICE_C]

    add_out = residual + dense_slice

    return sum_where, mul8, dense_out, add_out
