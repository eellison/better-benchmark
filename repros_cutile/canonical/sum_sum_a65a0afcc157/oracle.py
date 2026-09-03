"""cuTile port of sum_sum_a65a0afcc157: phlippe DenseNet BN-backward.

Reference (standard NCHW):
  where       = where(mask <= 0, fill, source)             bf16 [N, C, H, W]
  sum_1       = sum(f32(where), [0, 2, 3])                  f32 [C]
  sub         = f32(bn_input) - mean                        broadcast [N, C, H, W]
  sum_2       = sum(f32(where) * sub, [0, 2, 3])          f32 [C]
  mean_term   = sum_1 * SCALE
  variance    = sum_2 * SCALE * invstd^2
  output_scale= invstd * weight
  grad_bf16   = bf16((f32(where) - sub*variance - mean_term) * output_scale)
                                                            bf16 [N, C, H, W]
  slice_1     = arg0[:, SLICE_START:SLICE_START+C, :, :]  bf16 [N, C, H, W]
  add         = slice_1 + grad_bf16                         bf16 [N, C, H, W]
  slice_2     = add[:, :SLICE_C, :, :]                       bf16 [N, SLICE_C, H, W]
  Returns (sum_1, sum_2 * invstd, add, slice_2).

Multi-kernel plan:
  * Kernel 1: per-channel dual reduction over R = N*HW = 128*1024 = 131072.
    Split reduction across N — one program per (n_block, c) chunk to keep
    register pressure low; use ct.sum in-tile then a torch finalize.
  * Kernel 2: BN-backward epilogue + residual add. Write channels-first
    contiguous NCHW.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


N = 128
C = 48
INPUT_C = 64
SLICE_START = 16
SLICE_C = 16
H = 32
W = 32
HW = H * W       # 1024
R = N * HW       # 131072
BLOCK_HW = 1024  # process one full HW slab per program
NUM_BLOCKS = N   # 128 partial rows per channel
SCALE = 7.62939453125e-06


@ct.kernel
def _partial_reduce_kernel(
    mask_ptr,        # bf16 [N, C, H, W]
    fill_ptr,        # bf16 [1]
    source_ptr,      # bf16 [N, C, H, W]
    bn_input_ptr,    # bf16 [N, C, H, W]
    mean_ptr,        # f32  [C]
    partial_sum_ptr, # f32  [N, C]
    partial_dot_ptr, # f32  [N, C]
    C_: ct.Constant[int],
    HW_: ct.Constant[int],
    N_: ct.Constant[int],
    BLOCK_HW_: ct.Constant[int],
):
    n = ct.bid(0)
    c = ct.bid(1)
    fill = ct.load(fill_ptr, index=(0,), shape=(1,))
    mean = ct.load(mean_ptr, index=(c,), shape=(1,))

    # Load the full HW slab as a 1D tile.
    hw_idx = ct.arange(BLOCK_HW_, dtype=ct.int32)
    active = hw_idx < HW_
    h = hw_idx // W
    w = hw_idx - h * W
    n_t = ct.full((BLOCK_HW_,), n, dtype=ct.int32)
    c_t = ct.full((BLOCK_HW_,), c, dtype=ct.int32)

    mask_v = ct.gather(mask_ptr, (n_t, c_t, h, w), mask=active,
                       padding_value=ct.bfloat16(0.0))
    source_v = ct.gather(source_ptr, (n_t, c_t, h, w), mask=active,
                         padding_value=ct.bfloat16(0.0))
    bn_v = ct.gather(bn_input_ptr, (n_t, c_t, h, w), mask=active,
                     padding_value=ct.bfloat16(0.0))

    fill_bc = ct.broadcast_to(ct.reshape(fill, (1,)), (BLOCK_HW_,))
    zero_bf = ct.astype(0.0, ct.bfloat16)
    where_bf = ct.where(mask_v <= zero_bf, fill_bc, source_v)
    where_f = ct.astype(where_bf, ct.float32)
    where_f = ct.where(active, where_f, 0.0)

    bn_f = ct.astype(bn_v, ct.float32)
    mean_bc = ct.broadcast_to(ct.astype(mean, ct.float32), (BLOCK_HW_,))
    centered = ct.where(active, bn_f - mean_bc, 0.0)

    partial_sum = ct.sum(where_f)
    partial_dot = ct.sum(where_f * centered)
    ct.store(partial_sum_ptr, index=(n, c), tile=ct.reshape(partial_sum, (1, 1)))
    ct.store(partial_dot_ptr, index=(n, c), tile=ct.reshape(partial_dot, (1, 1)))


@ct.kernel
def _finalize_kernel(
    partial_sum_ptr, # f32 [N, C]
    partial_dot_ptr, # f32 [N, C]
    invstd_ptr,      # f32 [C]
    weight_ptr,      # f32 [C]
    sum_out_ptr,     # f32 [C]
    scaled_dot_ptr,  # f32 [C]
    mean_term_ptr,   # f32 [C]
    variance_ptr,    # f32 [C]
    output_scale_ptr,# f32 [C]
    C_: ct.Constant[int],
    N_: ct.Constant[int],
    BLOCK_C_: ct.Constant[int],
    BLOCK_N_: ct.Constant[int],
    SCALE_C: ct.Constant[float],
):
    """Grid: (C // BLOCK_C_,). Reduce partials along N inside kernel."""
    c_block = ct.bid(0)
    partial_sum = ct.astype(
        ct.load(partial_sum_ptr, index=(0, c_block), shape=(BLOCK_N_, BLOCK_C_)),
        ct.float32,
    )
    partial_dot = ct.astype(
        ct.load(partial_dot_ptr, index=(0, c_block), shape=(BLOCK_N_, BLOCK_C_)),
        ct.float32,
    )
    sum_value = ct.sum(partial_sum, axis=0)
    dot_value = ct.sum(partial_dot, axis=0)
    invstd = ct.astype(ct.load(invstd_ptr, index=(c_block,), shape=(BLOCK_C_,)), ct.float32)
    weight = ct.astype(ct.load(weight_ptr, index=(c_block,), shape=(BLOCK_C_,)), ct.float32)

    scaled_dot = dot_value * invstd
    mean_term = sum_value * SCALE_C
    variance = (dot_value * SCALE_C) * (invstd * invstd)
    output_scale = invstd * weight

    ct.store(sum_out_ptr, index=(c_block,), tile=sum_value)
    ct.store(scaled_dot_ptr, index=(c_block,), tile=scaled_dot)
    ct.store(mean_term_ptr, index=(c_block,), tile=mean_term)
    ct.store(variance_ptr, index=(c_block,), tile=variance)
    ct.store(output_scale_ptr, index=(c_block,), tile=output_scale)


@ct.kernel
def _epilogue_kernel(
    residual_ptr,    # bf16 [N, INPUT_C, H, W]
    mask_ptr,        # bf16 [N, C, H, W]
    fill_ptr,        # bf16 [1]
    source_ptr,      # bf16 [N, C, H, W]
    bn_input_ptr,    # bf16 [N, C, H, W]
    mean_ptr,        # f32  [C]
    mean_term_ptr,   # f32  [C]
    variance_ptr,    # f32  [C]
    output_scale_ptr,# f32  [C]
    add_out_ptr,     # bf16 [N, C, H, W]
    C_: ct.Constant[int],
    INPUT_C_: ct.Constant[int],
    SLICE_START_: ct.Constant[int],
    HW_: ct.Constant[int],
    BLOCK_HW_: ct.Constant[int],
):
    n = ct.bid(0)
    c = ct.bid(1)
    fill = ct.load(fill_ptr, index=(0,), shape=(1,))
    mean = ct.load(mean_ptr, index=(c,), shape=(1,))
    mean_term = ct.load(mean_term_ptr, index=(c,), shape=(1,))
    variance = ct.load(variance_ptr, index=(c,), shape=(1,))
    output_scale = ct.load(output_scale_ptr, index=(c,), shape=(1,))

    hw_idx = ct.arange(BLOCK_HW_, dtype=ct.int32)
    active = hw_idx < HW_
    h = hw_idx // W
    w = hw_idx - h * W
    n_t = ct.full((BLOCK_HW_,), n, dtype=ct.int32)
    c_t = ct.full((BLOCK_HW_,), c, dtype=ct.int32)
    c_res_t = ct.full((BLOCK_HW_,), c + SLICE_START_, dtype=ct.int32)

    mask_v = ct.gather(mask_ptr, (n_t, c_t, h, w), mask=active,
                       padding_value=ct.bfloat16(0.0))
    source_v = ct.gather(source_ptr, (n_t, c_t, h, w), mask=active,
                         padding_value=ct.bfloat16(0.0))
    bn_v = ct.gather(bn_input_ptr, (n_t, c_t, h, w), mask=active,
                     padding_value=ct.bfloat16(0.0))
    residual = ct.gather(residual_ptr, (n_t, c_res_t, h, w), mask=active,
                         padding_value=ct.bfloat16(0.0))

    fill_bc = ct.broadcast_to(ct.reshape(fill, (1,)), (BLOCK_HW_,))
    zero_bf = ct.astype(0.0, ct.bfloat16)
    where_bf = ct.where(mask_v <= zero_bf, fill_bc, source_v)
    where_f = ct.astype(where_bf, ct.float32)

    bn_f = ct.astype(bn_v, ct.float32)
    mean_bc = ct.broadcast_to(ct.astype(mean, ct.float32), (BLOCK_HW_,))
    centered = bn_f - mean_bc

    mean_term_bc = ct.broadcast_to(ct.astype(mean_term, ct.float32), (BLOCK_HW_,))
    variance_bc = ct.broadcast_to(ct.astype(variance, ct.float32), (BLOCK_HW_,))
    out_scale_bc = ct.broadcast_to(ct.astype(output_scale, ct.float32), (BLOCK_HW_,))
    after_variance = where_f - centered * variance_bc
    after_mean = after_variance - mean_term_bc
    grad_bf = ct.astype(after_mean * out_scale_bc, ct.bfloat16)

    add_val = ct.astype(
        ct.astype(residual, ct.float32) + ct.astype(grad_bf, ct.float32),
        ct.bfloat16,
    )
    ct.scatter(add_out_ptr, (n_t, c_t, h, w), add_val, mask=active)


@oracle_impl(hardware="B200", point="a7975a32")
def oracle_forward(inputs):
    residual, mask, fill, source, bn_input, mean, invstd, weight = inputs
    device = mask.device

    mean_1d = mean.view(C)
    fill_1d = fill.view(1)

    partial_sum = torch.empty((N, C), device=device, dtype=torch.float32)
    partial_dot = torch.empty((N, C), device=device, dtype=torch.float32)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (N, C, 1), _partial_reduce_kernel,
        (mask, fill_1d, source, bn_input, mean_1d,
         partial_sum, partial_dot,
         C, HW, N, BLOCK_HW),
    )

    # Finalize per-channel summaries in-kernel — match Triton's _finalize_kernel.
    sum_out = torch.empty((C,), device=device, dtype=torch.float32)
    scale_grad = torch.empty((C,), device=device, dtype=torch.float32)
    mean_term = torch.empty((C,), device=device, dtype=torch.float32)
    variance = torch.empty((C,), device=device, dtype=torch.float32)
    output_scale = torch.empty((C,), device=device, dtype=torch.float32)
    invstd_1d = invstd.view(C).contiguous()
    weight_1d = weight.view(C).contiguous()
    BLOCK_C = 8
    BLOCK_N_POW2 = 128  # N=128 is already a power of 2
    assert C % BLOCK_C == 0
    ct.launch(
        stream, (C // BLOCK_C, 1, 1), _finalize_kernel,
        (partial_sum, partial_dot, invstd_1d, weight_1d,
         sum_out, scale_grad, mean_term, variance, output_scale,
         C, N, BLOCK_C, BLOCK_N_POW2, SCALE),
    )

    add_out = torch.empty_strided(
        (N, C, H, W),
        (C * HW, HW, W, 1),
        device=device, dtype=torch.bfloat16,
    )

    ct.launch(
        stream, (N, C, 1), _epilogue_kernel,
        (residual, mask, fill_1d, source, bn_input, mean_1d,
         mean_term, variance, output_scale, add_out,
         C, INPUT_C, SLICE_START, HW, BLOCK_HW),
    )

    slice_out = torch.as_strided(
        add_out, (N, SLICE_C, H, W), (C * HW, HW, W, 1),
    )
    return sum_out, scale_grad, add_out, slice_out
