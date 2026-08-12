"""cuTile port of sum_sum_18bae59d9cc2: GhostNet BN-backward, channels-last.

Kernel structure mirrors the Triton reference (3 kernels):
1. Partial reduce: per-chunk sums of `selected` and `selected * centered`.
2. Finalize: reduces per-chunk partials and computes derived channel scalars
   (scale_grad, mean_term, variance_term, output_scale) in-kernel.
3. Epilogue: per-element BN-backward writeback using the finalized scalars.
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
TOTAL_ROWS = N * HW              # 6,422,528; each row has C=8 elements
BLOCK_ROW = 1024
NUM_BLOCKS = TOTAL_ROWS // BLOCK_ROW  # 6272


def _next_pow2(v: int) -> int:
    r = 1
    while r < v:
        r <<= 1
    return r


NUM_BLOCKS_PAD = _next_pow2(NUM_BLOCKS)  # 8192

SCALE = 1.5570192920918366e-07


@ct.kernel
def _partial_reduce_kernel(
    add_ptr,           # bf16 [TOTAL_ROWS, C]
    mask_ptr,          # bf16 [TOTAL_ROWS, C]
    fill_ptr,          # bf16 [1]
    x_ptr,             # bf16 [TOTAL_ROWS, C]  (arg4, centered source)
    mean_ptr,          # f32  [C]
    partial_sum_ptr,   # f32  [NUM_BLOCKS, C]
    partial_dot_ptr,   # f32  [NUM_BLOCKS, C]
    C_: ct.Constant[int],
    BLOCK_ROW_: ct.Constant[int],
):
    pid = ct.bid(0)
    add_v = ct.load(
        add_ptr, index=(pid, 0), shape=(BLOCK_ROW_, C_),
    )
    mask_v = ct.load(
        mask_ptr, index=(pid, 0), shape=(BLOCK_ROW_, C_),
    )
    x = ct.load(x_ptr, index=(pid, 0), shape=(BLOCK_ROW_, C_))
    mean = ct.load(mean_ptr, index=(0,), shape=(C_,))

    fill = ct.load(fill_ptr, index=(0,), shape=(1,))
    fill_bc = ct.broadcast_to(ct.reshape(fill, (1, 1)), (BLOCK_ROW_, C_))
    zero_bf = ct.astype(0.0, ct.bfloat16)
    selected_bf = ct.where(mask_v <= zero_bf, fill_bc, add_v)
    selected_f = ct.astype(selected_bf, ct.float32)

    x_f = ct.astype(x, ct.float32)
    mean_2d = ct.reshape(mean, (1, C_))
    centered = x_f - mean_2d
    product = selected_f * centered

    partial_sum = ct.sum(selected_f, axis=0)
    partial_dot = ct.sum(product, axis=0)
    ct.store(
        partial_sum_ptr, index=(pid, 0),
        tile=ct.reshape(partial_sum, (1, C_)),
    )
    ct.store(
        partial_dot_ptr, index=(pid, 0),
        tile=ct.reshape(partial_dot, (1, C_)),
    )


@ct.kernel
def _finalize_kernel(
    partial_sum_ptr,      # f32 [NUM_BLOCKS, C]
    partial_dot_ptr,      # f32 [NUM_BLOCKS, C]
    invstd_ptr,           # f32 [C]
    weight_ptr,           # f32 [C]
    sum_out_ptr,          # f32 [C]
    scale_grad_ptr,       # f32 [C]
    mean_term_ptr,        # f32 [C]
    variance_term_ptr,    # f32 [C]
    output_scale_ptr,     # f32 [C]
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
    sum_val = ct.sum(p_sum, axis=0)   # (C,)
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
    add_ptr,           # bf16 [TOTAL_ROWS, C]
    mask_ptr,          # bf16 [TOTAL_ROWS, C]
    fill_ptr,          # bf16 [1]
    x_ptr,             # bf16 [TOTAL_ROWS, C]
    mean_ptr,          # f32  [C]
    mean_term_ptr,     # f32 [C]
    variance_term_ptr, # f32 [C]
    output_scale_ptr,  # f32 [C]
    out_ptr,           # bf16 [TOTAL_ROWS, C]
    C_: ct.Constant[int],
    BLOCK_ROW_: ct.Constant[int],
):
    pid = ct.bid(0)
    add_v = ct.load(
        add_ptr, index=(pid, 0), shape=(BLOCK_ROW_, C_),
    )
    mask_v = ct.load(
        mask_ptr, index=(pid, 0), shape=(BLOCK_ROW_, C_),
    )
    x = ct.load(x_ptr, index=(pid, 0), shape=(BLOCK_ROW_, C_))
    mean = ct.load(mean_ptr, index=(0,), shape=(C_,))

    fill = ct.load(fill_ptr, index=(0,), shape=(1,))
    fill_bc = ct.broadcast_to(ct.reshape(fill, (1, 1)), (BLOCK_ROW_, C_))
    zero_bf = ct.astype(0.0, ct.bfloat16)
    selected_bf = ct.where(mask_v <= zero_bf, fill_bc, add_v)
    selected_f = ct.astype(selected_bf, ct.float32)

    x_f = ct.astype(x, ct.float32)
    mean_2d = ct.reshape(mean, (1, C_))
    centered = x_f - mean_2d

    mean_term = ct.load(mean_term_ptr, index=(0,), shape=(C_,))
    variance_term = ct.load(variance_term_ptr, index=(0,), shape=(C_,))
    output_scale = ct.load(output_scale_ptr, index=(0,), shape=(C_,))

    mean_bc = ct.reshape(mean_term, (1, C_))
    var_bc = ct.reshape(variance_term, (1, C_))
    out_scale_bc = ct.reshape(output_scale, (1, C_))

    after_variance = selected_f - centered * var_bc
    after_mean = after_variance - mean_bc
    out_f = after_mean * out_scale_bc
    ct.store(out_ptr, index=(pid, 0), tile=ct.astype(out_f, ct.bfloat16))


@oracle_impl(hardware="B200", point="1cc77f2b")
def oracle_forward(inputs):
    (
        arg0_1,           # bf16 [N, C_IN, H, W]   channels-last
        arg1_1,           # bf16 [N, C, H, W]      channels-last
        arg2_1,           # bf16 [N, C, H, W]      channels-last (mask)
        arg3_1,           # bf16 []                scalar fill
        arg4_1,           # bf16 [N, C, H, W]      channels-last (bn source)
        arg5_1,           # f32  [1, C, 1, 1]      per-channel mean
        arg6_1,           # f32  [C]               invstd
        arg7_1,           # f32  [C]               weight
    ) = inputs
    device = arg1_1.device

    # NHWC (channels innermost) contiguous views.
    arg0_nhwc = arg0_1.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C_IN]
    slice1_nhwc = arg0_nhwc[:, :, :, :C].contiguous()    # [N, H, W, C]
    arg1_nhwc = arg1_1.permute(0, 2, 3, 1).contiguous()
    add_nhwc = (slice1_nhwc + arg1_nhwc).contiguous()    # bf16
    mask_nhwc = arg2_1.permute(0, 2, 3, 1).contiguous()  # bf16
    x_nhwc = arg4_1.permute(0, 2, 3, 1).contiguous()     # bf16

    add_flat = add_nhwc.view(TOTAL_ROWS, C)
    mask_flat = mask_nhwc.view(TOTAL_ROWS, C)
    x_flat = x_nhwc.view(TOTAL_ROWS, C)
    mean_1d = arg5_1.view(C)
    fill_flat = arg3_1.view(1)

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
        (add_flat, mask_flat, fill_flat, x_flat, mean_1d,
         partial_sum, partial_dot, C, BLOCK_ROW),
    )
    ct.launch(
        stream, (1, 1, 1), _finalize_kernel,
        (partial_sum, partial_dot, arg6_1, arg7_1,
         sum_out, scale_grad, mean_term, variance_term, output_scale,
         C, NUM_BLOCKS_PAD, SCALE),
    )

    out_nhwc = torch.empty((N, H, W, C), device=device, dtype=torch.bfloat16)
    out_flat = out_nhwc.view(TOTAL_ROWS, C)

    ct.launch(
        stream, (NUM_BLOCKS, 1, 1), _epilogue_kernel,
        (add_flat, mask_flat, fill_flat, x_flat, mean_1d,
         mean_term, variance_term, output_scale, out_flat,
         C, BLOCK_ROW),
    )

    out_cl = torch.empty_strided(
        (N, C, H, W),
        (C * HW, 1, W * C, C),
        device=device, dtype=torch.bfloat16,
    )
    out_cl.copy_(out_nhwc.permute(0, 3, 1, 2))

    return sum_out, scale_grad, out_cl
