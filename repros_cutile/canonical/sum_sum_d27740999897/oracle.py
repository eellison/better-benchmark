"""cuTile port of sum_sum_d27740999897: DenseNet BN backward.

Matches Triton: per-channel program iterates R spatial elements from
NCHW-contiguous arg0/arg2/arg3 via gather (no permute/pad copies).
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


N = 4
C = 512
H = 28
W = 28
HW = H * W
R = N * HW  # 3136
BLOCK_R = 4096
SCALE = 0.00031887755102040814
SLICE_START = 480


@ct.kernel
def _dense_bn_backward_kernel(
    mask_ptr,        # bf16 flat NCHW view (numel,)
    fill_ptr,        # bf16 [1]  (scalar)
    source_ptr,      # bf16 flat NCHW view
    centered_source_ptr,  # bf16 flat NCHW view
    mean_ptr,        # f32 [C]
    invstd_ptr,      # f32 [C]
    weight_ptr,      # f32 [C]
    sum_out_ptr,     # f32 [C]
    scale_grad_ptr,  # f32 [C]
    dense_out_ptr,   # bf16 flat NCHW view
    BLOCK_R: ct.Constant[int],
    R_: ct.Constant[int],
    HW_: ct.Constant[int],
    C_STRIDE: ct.Constant[int],
    SCALE_: ct.Constant[float],
):
    c = ct.bid(0)
    rows = ct.arange(BLOCK_R, dtype=ct.int32)
    active = rows < R_
    n = rows // HW_
    spatial = rows - n * HW_
    offsets = n * (C_STRIDE) + c * HW_ + spatial

    zero_bf = ct.zeros((BLOCK_R,), dtype=ct.bfloat16)
    mask_value = ct.gather(mask_ptr, (offsets,), mask=active)
    # For inactive elements, gather returns 0 (default). fill scalar and source:
    fill_scalar = ct.load(fill_ptr, index=(0,), shape=(1,))
    fill_bc = ct.full((BLOCK_R,), 0.0, dtype=ct.bfloat16)  # placeholder
    # Build fill_bc by broadcasting the loaded scalar via reshape+broadcast
    fill_2d = ct.reshape(fill_scalar, (1,))
    # Use where: le triggers fill. Load source at same offsets.
    source_value = ct.gather(source_ptr, (offsets,), mask=active)
    le = mask_value <= zero_bf
    # broadcast fill scalar to shape (BLOCK_R,)
    fill_broadcast = ct.broadcast_to(fill_2d, (BLOCK_R,))
    where_bf16 = ct.where(le, fill_broadcast, source_value)
    where_f32 = ct.astype(where_bf16, ct.float32)
    where_f32 = ct.where(active, where_f32, ct.zeros((BLOCK_R,), dtype=ct.float32))

    centered_source_bf = ct.gather(centered_source_ptr, (offsets,), mask=active)
    centered_source = ct.astype(centered_source_bf, ct.float32)
    mean_v = ct.load(mean_ptr, index=(c,), shape=(1,))
    mean_bc = ct.broadcast_to(mean_v, (BLOCK_R,))
    centered = centered_source - mean_bc
    centered = ct.where(active, centered, ct.zeros((BLOCK_R,), dtype=ct.float32))

    product = where_f32 * centered
    sum_where = ct.sum(where_f32)
    sum_centered = ct.sum(product)

    invstd_v = ct.load(invstd_ptr, index=(c,), shape=(1,))
    weight_v = ct.load(weight_ptr, index=(c,), shape=(1,))
    mean_term = sum_where * SCALE_
    dot_scaled = sum_centered * SCALE_
    invstd_scalar = ct.reshape(invstd_v, (1,))
    weight_scalar = ct.reshape(weight_v, (1,))
    variance_term = dot_scaled * (invstd_scalar * invstd_scalar)
    output_scale = invstd_scalar * weight_scalar

    variance_term_bc = ct.broadcast_to(variance_term, (BLOCK_R,))
    mean_term_bc = ct.broadcast_to(ct.reshape(mean_term, (1,)), (BLOCK_R,))
    output_scale_bc = ct.broadcast_to(output_scale, (BLOCK_R,))
    after_variance = where_f32 - centered * variance_term_bc
    after_mean = after_variance - mean_term_bc
    dense_bf16 = ct.astype(after_mean * output_scale_bc, ct.bfloat16)

    ct.store(sum_out_ptr, index=(c,), tile=ct.reshape(sum_where, (1,)))
    ct.store(scale_grad_ptr, index=(c,), tile=ct.reshape(sum_centered * invstd_scalar, (1,)))
    ct.scatter(dense_out_ptr, (offsets,), dense_bf16, mask=active)


@oracle_impl(hardware="B200", point="8cf827b9")
def oracle_forward(inputs):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1 = inputs
    device = arg0_1.device

    sum_out = torch.empty_strided((C,), (1,), device=device, dtype=torch.float32)
    scale_grad = torch.empty_strided((C,), (1,), device=device, dtype=torch.float32)
    dense_out = torch.empty_strided(
        (N, C, H, W),
        (C * HW, HW, W, 1),
        device=device,
        dtype=torch.bfloat16,
    )
    numel = N * C * HW
    # Get flat contiguous views (arg0/arg2/arg3 are NCHW contiguous)
    arg0_flat = arg0_1.view(-1)
    arg2_flat = arg2_1.view(-1)
    arg3_flat = arg3_1.view(-1)
    dense_flat = dense_out.view(-1)
    fill_view = arg1_1.view(1)
    mean_1d = arg4_1.view(C)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (C, 1, 1),
        _dense_bn_backward_kernel,
        (arg0_flat, fill_view, arg2_flat, arg3_flat, mean_1d, arg5_1, arg6_1,
         sum_out, scale_grad, dense_flat,
         BLOCK_R, R, HW, C * HW, SCALE),
    )

    return sum_out, scale_grad, dense_out, dense_out[:, SLICE_START:C, :, :]
