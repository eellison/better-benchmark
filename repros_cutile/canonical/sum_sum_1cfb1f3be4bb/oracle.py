"""cuTile port of sum_sum_1cfb1f3be4bb: GhostNet BN-forward+ReLU-backward
+ BN-backward tail.

Three-kernel structure mirrors the Triton reference:
1. `_partial_reduce_kernel` — split-K partial sums of `selected` and
   `selected * centered`.
2. `_finalize_kernel` — reduces chunks, computes scale_grad / mean_term /
   variance_term / output_scale in-kernel.
3. `_epilogue_kernel` — per-element BN-backward writeback.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


N = 512
C_IN = 16
C = 8
H = 112
W = 112
HW = H * W
K_TOTAL = N * HW              # 6,422,528
TOTAL = N * C * HW            # 51,380,224
SCALE = 1.5570192920918366e-07

BLOCK_ROW = 1024
NUM_BLOCKS = K_TOTAL // BLOCK_ROW  # 6272


def _next_pow2(v: int) -> int:
    r = 1
    while r < v:
        r <<= 1
    return r


NUM_BLOCKS_PAD = _next_pow2(NUM_BLOCKS)  # 8192


@ct.kernel
def _partial_reduce_kernel(
    slice_ptr,       # bf16 [K_TOTAL, C]  (arg0[:, 8:16] materialized nhwc-flat)
    bn_input_ptr,    # bf16 [K_TOTAL, C]
    mean_ptr,        # f32  [C]
    invstd_ptr,      # f32  [C]
    weight_ptr,      # f32  [C]
    bias_ptr,        # f32  [C]
    fill_ptr,        # bf16 [1]
    partial_sum_ptr, # f32  [NUM_BLOCKS, C]
    partial_dot_ptr, # f32  [NUM_BLOCKS, C]
    C_: ct.Constant[int],
    BLOCK_ROW_: ct.Constant[int],
):
    pid = ct.bid(0)
    slice_bf = ct.load(slice_ptr, index=(pid, 0), shape=(BLOCK_ROW_, C_))
    bn_bf = ct.load(bn_input_ptr, index=(pid, 0), shape=(BLOCK_ROW_, C_))
    mean = ct.load(mean_ptr, index=(0,), shape=(C_,))
    invstd = ct.load(invstd_ptr, index=(0,), shape=(C_,))
    weight = ct.load(weight_ptr, index=(0,), shape=(C_,))
    bias = ct.load(bias_ptr, index=(0,), shape=(C_,))
    fill = ct.load(fill_ptr, index=(0,), shape=(1,))

    x_f = ct.astype(bn_bf, ct.float32)
    mean_2d = ct.reshape(mean, (1, C_))
    invstd_2d = ct.reshape(invstd, (1, C_))
    weight_2d = ct.reshape(weight, (1, C_))
    bias_2d = ct.reshape(bias, (1, C_))
    centered = x_f - mean_2d
    normalized = centered * invstd_2d
    affine = normalized * weight_2d + bias_2d
    affine_bf16 = ct.astype(affine, ct.bfloat16)

    zero_bf = ct.astype(0.0, ct.bfloat16)
    fill_bc = ct.broadcast_to(ct.reshape(fill, (1, 1)), (BLOCK_ROW_, C_))
    selected_bf = ct.where(affine_bf16 <= zero_bf, fill_bc, slice_bf)
    selected_f = ct.astype(selected_bf, ct.float32)
    dot = selected_f * centered

    p_sum = ct.sum(selected_f, axis=0)
    p_dot = ct.sum(dot, axis=0)
    ct.store(
        partial_sum_ptr, index=(pid, 0),
        tile=ct.reshape(p_sum, (1, C_)),
    )
    ct.store(
        partial_dot_ptr, index=(pid, 0),
        tile=ct.reshape(p_dot, (1, C_)),
    )


@ct.kernel
def _finalize_kernel(
    partial_sum_ptr,   # f32 [NUM_BLOCKS, C]
    partial_dot_ptr,   # f32 [NUM_BLOCKS, C]
    invstd_ptr,        # f32 [C]
    weight_ptr,        # f32 [C]
    sum_out_ptr,       # f32 [C]
    scale_grad_ptr,    # f32 [C]  (dot * invstd)
    mean_term_ptr,     # f32 [C]  (sum * SCALE)
    variance_term_ptr, # f32 [C]  (dot * SCALE * invstd^2)
    output_scale_ptr,  # f32 [C]  (invstd * weight)
    C_: ct.Constant[int],
    BLOCK_TILES: ct.Constant[int],
    SCALE_: ct.Constant[float],
):
    p_sum = ct.load(
        partial_sum_ptr, index=(0, 0), shape=(BLOCK_TILES, C_),
        padding_mode=ct.PaddingMode.ZERO,
    )
    p_dot = ct.load(
        partial_dot_ptr, index=(0, 0), shape=(BLOCK_TILES, C_),
        padding_mode=ct.PaddingMode.ZERO,
    )
    sum_val = ct.sum(p_sum, axis=0)
    dot_val = ct.sum(p_dot, axis=0)

    invstd = ct.load(invstd_ptr, index=(0,), shape=(C_,))
    weight = ct.load(weight_ptr, index=(0,), shape=(C_,))

    scale_grad = dot_val * invstd
    mean_term = sum_val * SCALE_
    dot_scaled = dot_val * SCALE_
    invstd_sq = invstd * invstd
    variance_term = dot_scaled * invstd_sq
    output_scale = invstd * weight

    ct.store(sum_out_ptr, index=(0,), tile=sum_val)
    ct.store(scale_grad_ptr, index=(0,), tile=scale_grad)
    ct.store(mean_term_ptr, index=(0,), tile=mean_term)
    ct.store(variance_term_ptr, index=(0,), tile=variance_term)
    ct.store(output_scale_ptr, index=(0,), tile=output_scale)


@ct.kernel
def _epilogue_kernel(
    slice_ptr,           # bf16 [K_TOTAL, C]
    bn_input_ptr,        # bf16 [K_TOTAL, C]
    mean_ptr,            # f32  [C]
    invstd_ptr,          # f32  [C]
    weight_ptr,          # f32  [C]
    bias_ptr,            # f32  [C]
    fill_ptr,            # bf16 [1]
    mean_term_ptr,       # f32  [C]
    variance_term_ptr,   # f32  [C]
    output_scale_ptr,    # f32  [C]
    out_ptr,             # bf16 [K_TOTAL, C]
    C_: ct.Constant[int],
    BLOCK_ROW_: ct.Constant[int],
):
    pid = ct.bid(0)
    slice_bf = ct.load(slice_ptr, index=(pid, 0), shape=(BLOCK_ROW_, C_))
    bn_bf = ct.load(bn_input_ptr, index=(pid, 0), shape=(BLOCK_ROW_, C_))
    mean = ct.load(mean_ptr, index=(0,), shape=(C_,))
    invstd = ct.load(invstd_ptr, index=(0,), shape=(C_,))
    weight = ct.load(weight_ptr, index=(0,), shape=(C_,))
    bias = ct.load(bias_ptr, index=(0,), shape=(C_,))
    fill = ct.load(fill_ptr, index=(0,), shape=(1,))
    mean_term = ct.load(mean_term_ptr, index=(0,), shape=(C_,))
    variance_term = ct.load(variance_term_ptr, index=(0,), shape=(C_,))
    output_scale = ct.load(output_scale_ptr, index=(0,), shape=(C_,))

    x_f = ct.astype(bn_bf, ct.float32)
    mean_2d = ct.reshape(mean, (1, C_))
    invstd_2d = ct.reshape(invstd, (1, C_))
    weight_2d = ct.reshape(weight, (1, C_))
    bias_2d = ct.reshape(bias, (1, C_))
    centered = x_f - mean_2d
    normalized = centered * invstd_2d
    affine = normalized * weight_2d + bias_2d
    affine_bf16 = ct.astype(affine, ct.bfloat16)

    zero_bf = ct.astype(0.0, ct.bfloat16)
    fill_bc = ct.broadcast_to(ct.reshape(fill, (1, 1)), (BLOCK_ROW_, C_))
    selected = ct.astype(
        ct.where(affine_bf16 <= zero_bf, fill_bc, slice_bf), ct.float32,
    )
    mean_term_2d = ct.reshape(mean_term, (1, C_))
    variance_term_2d = ct.reshape(variance_term, (1, C_))
    output_scale_2d = ct.reshape(output_scale, (1, C_))
    after_variance = selected - centered * variance_term_2d
    after_mean = after_variance - mean_term_2d
    out_f = after_mean * output_scale_2d
    ct.store(out_ptr, index=(pid, 0), tile=ct.astype(out_f, ct.bfloat16))


@oracle_impl(hardware="B200", point="a67efa72")
def oracle_forward(inputs):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1 = inputs
    device = arg1_1.device

    # NHWC-flat views for narrow tensors (channels-innermost, contiguous).
    def _nhwc_flat(t, c_dim):
        return t.permute(0, 2, 3, 1).contiguous().view(K_TOTAL, c_dim)

    slice_wide = _nhwc_flat(arg0_1, C_IN)                   # (K_TOTAL, 16)
    slice_narrow = slice_wide[:, C:C + C].contiguous()      # arg0[:, 8:16]
    bn_input = _nhwc_flat(arg1_1, C)                        # (K_TOTAL, 8)

    mean_1d = arg2_1.view(C).contiguous()
    invstd_1d = arg3_1.view(C).contiguous()
    weight_1d = arg4_1.contiguous()
    bias_1d = arg5_1.contiguous()
    fill_1d = arg6_1.reshape(1)

    partial_sum = torch.empty((NUM_BLOCKS, C), device=device, dtype=torch.float32)
    partial_dot = torch.empty((NUM_BLOCKS, C), device=device, dtype=torch.float32)
    sum_out = torch.empty((C,), device=device, dtype=torch.float32)
    scale_grad = torch.empty((C,), device=device, dtype=torch.float32)
    mean_term = torch.empty((C,), device=device, dtype=torch.float32)
    variance_term = torch.empty((C,), device=device, dtype=torch.float32)
    output_scale = torch.empty((C,), device=device, dtype=torch.float32)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (NUM_BLOCKS, 1, 1), _partial_reduce_kernel,
        (slice_narrow, bn_input, mean_1d, invstd_1d, weight_1d, bias_1d,
         fill_1d, partial_sum, partial_dot, C, BLOCK_ROW),
    )
    ct.launch(
        stream, (1, 1, 1), _finalize_kernel,
        (partial_sum, partial_dot, invstd_1d, weight_1d,
         sum_out, scale_grad, mean_term, variance_term, output_scale,
         C, NUM_BLOCKS_PAD, SCALE),
    )

    out_flat = torch.empty((K_TOTAL, C), device=device, dtype=torch.bfloat16)
    ct.launch(
        stream, (NUM_BLOCKS, 1, 1), _epilogue_kernel,
        (slice_narrow, bn_input, mean_1d, invstd_1d, weight_1d, bias_1d,
         fill_1d, mean_term, variance_term, output_scale, out_flat,
         C, BLOCK_ROW),
    )

    out_nhwc = out_flat.view(N, H, W, C)
    dense_out = torch.empty_strided(
        (N, C, H, W), (C * HW, 1, W * C, C),
        device=device, dtype=torch.bfloat16,
    )
    dense_out.copy_(out_nhwc.permute(0, 3, 1, 2))
    return sum_out, scale_grad, dense_out
