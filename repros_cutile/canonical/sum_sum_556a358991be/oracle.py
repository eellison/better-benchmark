"""cuTile port of sum_sum_556a358991be: DenseNet BN-backward tail.

Reference (standard NCHW layout, contiguous):
  where       = where(arg1 <= 0, arg2, arg3)            bf16 [N, C, H, W]
  sum_1       = sum(f32(where), [0, 2, 3])              f32 [C]
  sub         = f32(arg4) - arg5                         (arg5 broadcasts to [N, C, H, W])
  sum_2       = sum(f32(where) * sub, [0, 2, 3])       f32 [C]
  mean_term   = sum_1 * SCALE                            f32 [C]
  variance    = sum_2 * SCALE * arg6^2                   f32 [C]
  output_scale= arg6 * arg7                              f32 [C]
  grad_bf16   = bf16((f32(where) - sub*variance - mean_term) * output_scale)
                                                          bf16 [N, C, H, W]
  slice_1     = arg0[:, 448:480, :, :]                    bf16 [N, 32, H, W]
  slice_2     = grad_bf16[:, 448:480, :, :]
  add         = slice_1 + slice_2                         bf16 [N, 32, H, W]
  Returns (sum_1, sum_2 * arg6, grad_bf16, add).

Port strategy: NCHW-contiguous makes each channel a contiguous HW-slab per
batch. Tile [C, N*HW] with a per-channel program (grid=C=480). One kernel
does everything: dual reduction across N*HW=3136 in shared memory, then
writes the epilogue. C fits comfortably in a per-channel grid.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


N = 4
C = 480
H = 28
W = 28
HW = H * W
R = N * HW                # 3136 rows per channel
SLICE_START = 448
SLICE_C = C - SLICE_START  # 32
RESIDUAL_C = 512
SCALE = 0.00031887755102040814


@ct.kernel
def _bn_tail_kernel(
    residual_ptr,   # bf16 [N, RESIDUAL_C, H, W]
    mask_ptr,       # bf16 [N, C, H, W]
    fill_ptr,       # bf16 [1]
    source_ptr,     # bf16 [N, C, H, W]
    bn_input_ptr,   # bf16 [N, C, H, W]
    mean_ptr,       # f32  [C]
    invstd_ptr,     # f32  [C]
    weight_ptr,     # f32  [C]
    sum_out_ptr,    # f32  [C]
    scale_grad_ptr, # f32  [C]
    dense_out_ptr,  # bf16 [N, C, H, W]
    add_out_ptr,    # bf16 [N, SLICE_C, H, W]
    C_: ct.Constant[int],
    HW_: ct.Constant[int],
    R_: ct.Constant[int],
    N_: ct.Constant[int],
    SLICE_START_: ct.Constant[int],
    SLICE_C_: ct.Constant[int],
    RESIDUAL_C_: ct.Constant[int],
    SCALE_: ct.Constant[float],
    BLOCK_R: ct.Constant[int],
):
    c = ct.bid(0)
    # Load full column [R_] flat by loading tiles of BLOCK_R and iterating.
    # For R=3136, we tile in a single program with BLOCK_R=4096 (>R, padded).
    # Use ct.arange for the row index and mask.
    rows = ct.arange(BLOCK_R, dtype=ct.int32)
    active = rows < R_
    n = rows // HW_
    spatial = rows - n * HW_

    # Load per-channel scalars.
    mean = ct.load(mean_ptr, index=(c,), shape=(1,))
    invstd = ct.load(invstd_ptr, index=(c,), shape=(1,))
    weight = ct.load(weight_ptr, index=(c,), shape=(1,))
    fill = ct.load(fill_ptr, index=(0,), shape=(1,))

    # Gather values via ct.gather on 4D indices.
    c_t = ct.full((BLOCK_R,), c, dtype=ct.int32)
    h_t = spatial // W
    w_t = spatial - h_t * W

    mask_v = ct.gather(mask_ptr, (n, c_t, h_t, w_t), mask=active,
                       padding_value=ct.bfloat16(0.0))
    source_v = ct.gather(source_ptr, (n, c_t, h_t, w_t), mask=active,
                         padding_value=ct.bfloat16(0.0))
    bn_v = ct.gather(bn_input_ptr, (n, c_t, h_t, w_t), mask=active,
                     padding_value=ct.bfloat16(0.0))

    fill_bc = ct.broadcast_to(ct.reshape(fill, (1,)), (BLOCK_R,))
    zero_bf = ct.astype(0.0, ct.bfloat16)
    where_bf = ct.where(mask_v <= zero_bf, fill_bc, source_v)
    where_f = ct.astype(where_bf, ct.float32)
    where_f = ct.where(active, where_f, 0.0)

    bn_f = ct.astype(bn_v, ct.float32)
    mean_bc = ct.broadcast_to(ct.astype(mean, ct.float32), (BLOCK_R,))
    centered = bn_f - mean_bc
    centered = ct.where(active, centered, 0.0)

    sum_where = ct.sum(where_f)
    sum_centered = ct.sum(where_f * centered)

    invstd_f = ct.astype(invstd, ct.float32)
    weight_f = ct.astype(weight, ct.float32)
    mean_term = sum_where * SCALE_
    variance = sum_centered * SCALE_ * (invstd_f * invstd_f)
    output_scale = invstd_f * weight_f

    variance_bc = ct.broadcast_to(ct.reshape(variance, (1,)), (BLOCK_R,))
    mean_term_bc = ct.broadcast_to(ct.reshape(mean_term, (1,)), (BLOCK_R,))
    out_scale_bc = ct.broadcast_to(ct.reshape(output_scale, (1,)), (BLOCK_R,))
    after_variance = where_f - centered * variance_bc
    after_mean = after_variance - mean_term_bc
    dense_bf = ct.astype(after_mean * out_scale_bc, ct.bfloat16)

    ct.scatter(dense_out_ptr, (n, c_t, h_t, w_t), dense_bf, mask=active)

    # Write scalars (sum_where, sum_centered * invstd)
    ct.store(sum_out_ptr, index=(c,), tile=ct.reshape(sum_where, (1,)))
    scale_grad = sum_centered * invstd_f
    ct.store(scale_grad_ptr, index=(c,), tile=ct.reshape(scale_grad, (1,)))

    # Residual add for the slice [SLICE_START:SLICE_START+SLICE_C]
    # If c is in that slice range, load and add.
    in_slice = c >= SLICE_START_
    if in_slice:
        slice_c = c - SLICE_START_
        c_res_t = ct.full((BLOCK_R,), c, dtype=ct.int32)
        c_slice_t = ct.full((BLOCK_R,), slice_c, dtype=ct.int32)
        residual = ct.gather(
            residual_ptr, (n, c_res_t, h_t, w_t), mask=active,
            padding_value=ct.bfloat16(0.0),
        )
        add_val = ct.astype(
            ct.astype(residual, ct.float32) + ct.astype(dense_bf, ct.float32),
            ct.bfloat16,
        )
        ct.scatter(add_out_ptr, (n, c_slice_t, h_t, w_t), add_val, mask=active)


@oracle_impl(hardware="B200", point="fdf0143e", BLOCK_R=4096)
def oracle_forward(inputs, *, BLOCK_R):
    residual, mask_input, fill, source, bn_input, mean, invstd, weight = inputs
    device = source.device

    # arg5 is [1, C, 1, 1]; the kernel wants a plain [C] per-channel mean.
    mean_1d = mean.view(C)
    fill_1d = fill.view(1)

    sum_out = torch.empty((C,), device=device, dtype=torch.float32)
    scale_grad = torch.empty((C,), device=device, dtype=torch.float32)
    dense_out = torch.empty_strided(
        (N, C, H, W),
        (C * HW, HW, W, 1),
        device=device, dtype=torch.bfloat16,
    )
    add_out = torch.empty_strided(
        (N, SLICE_C, H, W),
        (SLICE_C * HW, HW, W, 1),
        device=device, dtype=torch.bfloat16,
    )

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (C, 1, 1), _bn_tail_kernel,
        (residual, mask_input, fill_1d, source, bn_input,
         mean_1d, invstd, weight,
         sum_out, scale_grad, dense_out, add_out,
         C, HW, R, N, SLICE_START, SLICE_C, RESIDUAL_C, SCALE, BLOCK_R),
    )
    return sum_out, scale_grad, dense_out, add_out
