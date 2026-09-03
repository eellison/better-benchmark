"""cuTile port of sum_sum_86b0392acb3d: LearningToPaint pool BN-backward.

Pipeline (mirrors the Triton oracle structurally):
  * Kernel 1 (_pool_where_kernel): full-tensor NCHW flat. Reads grad from arg0
    (viewed as [N, C]), applies POOL_SCALE=1/16 and bf16-rounds, then produces
    where(mask<=0, 0.0, pooled) as bf16 into `where_out`.
  * Kernel 2 (_reduce_kernel): one program per channel, gathers N*H*W rows
    from where_out (NCHW contig) and centered_source (converted to NCHW), and
    computes f32 per-channel sum(where) and sum(where*centered).
  * Kernel 3 (_epilogue_kernel): full-tensor NCHW flat. Reads per-channel sums
    and produces the dense bf16 gradient.

Note: `centered_source` (arg2) is NHWC (channels-last). We convert to
contiguous NCHW ahead of time (same numeric content, just different layout)
for simple index math in the kernels.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


N = 96
C = 512
H = 4
W = 4
HW = H * W
TOTAL = N * C * HW
TOTAL_SPATIAL = N * HW  # 96*16 = 1536
POOL_SCALE = 0.0625
REDUCE_SCALE = 0.0006510416666666666


@ct.kernel
def _pool_where_kernel(
    grad_ptr,           # bf16 [N, C]
    mask_ptr,           # bf16 [N*C*HW] NCHW contig
    where_ptr,          # bf16 [N*C*HW] NCHW contig
    C_: ct.Constant[int],
    HW_: ct.Constant[int],
    BLOCK: ct.Constant[int],
    TOTAL_: ct.Constant[int],
):
    pid = ct.bid(0)
    mask = ct.load(
        mask_ptr, index=(pid,), shape=(BLOCK,),
        padding_mode=ct.PaddingMode.ZERO,
    )
    idxs = ct.arange(BLOCK, dtype=ct.int32) + pid * BLOCK
    # NCHW: flatten idx / HW = n*C + c
    nc = idxs // HW_
    # gather grad[nc]
    grad_bf = ct.gather(grad_ptr, nc)

    grad_f = ct.astype(grad_bf, ct.float32)
    pooled_f = grad_f * 0.0625
    pooled_bf = ct.astype(pooled_f, ct.bfloat16)
    zero_bf = ct.astype(ct.zeros((BLOCK,), dtype=ct.float32), ct.bfloat16)
    mask_f = ct.astype(mask, ct.float32)
    where_bf = ct.where(mask_f <= 0.0, zero_bf, pooled_bf)
    ct.store(where_ptr, index=(pid,), tile=where_bf)


@ct.kernel
def _reduce_kernel(
    where_ptr,          # bf16 [N*C*HW] NCHW contig
    centered_ptr,       # f32  [N*C*HW] NCHW contig  (arg2.f32 - arg3)
    sum_out_ptr,        # f32  [C]
    dot_out_ptr,        # f32  [C]
    C_: ct.Constant[int],
    HW_: ct.Constant[int],
    BLOCK_R: ct.Constant[int],
    R_: ct.Constant[int],
):
    c = ct.bid(0)
    rows = ct.arange(BLOCK_R, dtype=ct.int32)
    active = rows < R_
    n = rows // HW_
    spatial = rows - n * HW_
    offsets = n * (C_ * HW_) + c * HW_ + spatial

    zero_bf = ct.astype(0.0, ct.bfloat16)
    zero_f = ct.astype(0.0, ct.float32)

    where_bf = ct.gather(where_ptr, offsets, mask=active, padding_value=zero_bf)
    where_f = ct.astype(where_bf, ct.float32)
    centered = ct.gather(centered_ptr, offsets, mask=active, padding_value=zero_f)

    w_masked = ct.where(active, where_f, zero_f)
    dot_masked = ct.where(active, where_f * centered, zero_f)

    sum_val = ct.sum(w_masked)
    dot_val = ct.sum(dot_masked)

    c_idx = ct.full((1,), c, dtype=ct.int32)
    sum_1 = ct.reshape(sum_val, (1,))
    dot_1 = ct.reshape(dot_val, (1,))
    ct.atomic_add(sum_out_ptr, (c_idx,), sum_1)
    ct.atomic_add(dot_out_ptr, (c_idx,), dot_1)


@ct.kernel
def _epilogue_kernel(
    where_ptr,
    centered_ptr,
    sum_ptr,
    dot_ptr,
    invstd_ptr,
    weight_ptr,
    out_ptr,
    C_: ct.Constant[int],
    HW_: ct.Constant[int],
    BLOCK: ct.Constant[int],
    TOTAL_: ct.Constant[int],
    SCALE_: ct.Constant[float],
):
    pid = ct.bid(0)
    where_val = ct.load(
        where_ptr, index=(pid,), shape=(BLOCK,),
        padding_mode=ct.PaddingMode.ZERO,
    )
    centered = ct.load(
        centered_ptr, index=(pid,), shape=(BLOCK,),
        padding_mode=ct.PaddingMode.ZERO,
    )
    idxs = ct.arange(BLOCK, dtype=ct.int32) + pid * BLOCK
    c_full = idxs // HW_
    channel = c_full - (c_full // C_) * C_
    sum_v = ct.gather(sum_ptr, channel)
    dot_v = ct.gather(dot_ptr, channel)
    invstd = ct.gather(invstd_ptr, channel)
    weight = ct.gather(weight_ptr, channel)

    where_f = ct.astype(where_val, ct.float32)
    mean_term = sum_v * SCALE_
    invstd_sq = invstd * invstd
    variance_term = dot_v * SCALE_ * invstd_sq
    affine = invstd * weight
    grad = ((where_f - centered * variance_term) - mean_term) * affine
    ct.store(out_ptr, index=(pid,), tile=ct.astype(grad, ct.bfloat16))


@oracle_impl(hardware="B200", point="ab4c6849")
def oracle_forward(inputs):
    arg0, arg1, arg2, arg3, arg4, arg5, _shape_param_0 = inputs
    device = arg0.device

    # arg2 is channels-last. Move to contiguous NCHW for simple index math.
    arg2_nchw = arg2.contiguous()
    centered = arg2_nchw.to(torch.float32) - arg3  # (1,C,1,1) broadcasts
    centered_flat = centered.view(-1)

    mask_flat = arg1.contiguous().view(-1)
    grad_flat = arg0.contiguous().view(-1)  # [N*C]

    where_out = torch.empty(TOTAL, device=device, dtype=torch.bfloat16)
    sum_out = torch.zeros(C, device=device, dtype=torch.float32)
    dot_out = torch.zeros(C, device=device, dtype=torch.float32)

    stream = torch.cuda.current_stream()

    # Kernel 1: pool + where -> bf16
    BLOCK = 256
    ct.launch(
        stream, (ct.cdiv(TOTAL, BLOCK), 1, 1), _pool_where_kernel,
        (grad_flat, mask_flat, where_out, C, HW, BLOCK, TOTAL),
    )
    where_out_4d = where_out.view(N, C, H, W)

    # Kernel 2: per-channel reduce
    BLOCK_R = 2048  # >= TOTAL_SPATIAL=1536, pow2
    ct.launch(
        stream, (C, 1, 1), _reduce_kernel,
        (where_out, centered_flat, sum_out, dot_out,
         C, HW, BLOCK_R, TOTAL_SPATIAL),
    )

    scale_grad = dot_out * arg4  # arg4 is invstd (f32[C])

    # Kernel 3: BN backward dense
    dense_flat = torch.empty(TOTAL, device=device, dtype=torch.bfloat16)
    ct.launch(
        stream, (ct.cdiv(TOTAL, BLOCK), 1, 1), _epilogue_kernel,
        (where_out, centered_flat, sum_out, dot_out,
         arg4, arg5, dense_flat,
         C, HW, BLOCK, TOTAL, REDUCE_SCALE),
    )
    dense_out = dense_flat.view(N, C, H, W)

    full = torch.zeros((), device=device, dtype=torch.bfloat16)

    return full, where_out_4d, sum_out, scale_grad, dense_out
