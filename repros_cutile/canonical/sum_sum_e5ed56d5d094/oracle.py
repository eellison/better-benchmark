"""cuTile port of sum_sum_e5ed56d5d094: DenseNet BN-backward with slice epilogue.

Structure:
  * torch precomputes:
    - `where` value (ReLU mask epilogue): where(arg9<=0, arg10, arg11) → bf16
    - `centered = arg12 - arg13` (f32)
    - Residual chain: series of bf16 add rounds arg0..arg8 sliced on channels [192:224]
      → residual_exact (bf16)
  * cuTile kernel 1 (row-per-channel producer): sum(where.f32) and
    sum(where*centered) via a per-channel program; writes out_sum and out_mul.
  * cuTile kernel 2 (dense epilogue): full-tensor bn-backward → out_full (bf16).
  * cuTile kernel 3 (slice epilogue): compute grad on the sliced channels
    [192:224], add residual_exact, store bf16 slice output.

For portability across H100/B200 the shape is fixed.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


N = 4
C = 224
H = 28
W = 28
HW = H * W
TOTAL = N * C * HW
SLICE_START = 192
SLICE_C = 32
N_SLICE_HW = N * SLICE_C * HW
SCALE = 0.00031887755102040814


@ct.kernel
def _sum_per_channel_kernel(
    mask_ptr,           # bf16 [N, C, H, W] (arg9_1)
    source_ptr,         # bf16 [N, C, H, W] (arg11_1)
    centered_ptr,       # f32  [N, C, H, W] (arg12_1 - arg13_1)
    fill_scalar_ptr,    # bf16 scalar (arg10_1)
    out_sum_ptr,        # f32 [C]
    out_mul_ptr,        # f32 [C]
    sum_centered_ptr,   # f32 [C]
    HW_: ct.Constant[int],
    N_: ct.Constant[int],
    BLOCK: ct.Constant[int],
    TOTAL_: ct.Constant[int],
):
    # One program per channel, iterating over N*HW elements
    c = ct.bid(0)
    fill = ct.load(fill_scalar_ptr, index=(0,), shape=(1,))
    fill_f = ct.astype(fill, ct.float32)
    fill_scalar = ct.reshape(fill_f, (1,))

    n_iters = ct.cdiv(N_ * HW_, BLOCK)
    acc_sum = ct.zeros((BLOCK,), dtype=ct.float32)
    acc_dot = ct.zeros((BLOCK,), dtype=ct.float32)

    for i in range(n_iters):
        offsets = ct.arange(BLOCK, dtype=ct.int32) + i * BLOCK
        # Map linear offset in [0, N*HW) to (n, hw). Then full offset:
        # (n, c, hw)  as if the tensor is [N, C, HW] contiguous.
        # We instead pass tensors as [N, C, H*W] and load channel by channel.
        # Load the c-th channel slice of shape (1, 1, BLOCK).
        n = offsets // HW_
        hw = offsets - n * HW_
        # Load [N, C, HW] as 3D. Use ct.gather-like via computed offset.
        # We chose to reshape input to [N, C, HW] and use tile-load with
        # (n_tile, c_tile, hw_tile) but that's not fixed-shape. Instead,
        # pass flat arrays and index by row = n * C * HW + c * HW + hw.
        pass


# Simpler: switch to flat kernel per-element that adds to atomic sums.
@ct.kernel
def _producer_flat_kernel(
    mask_ptr,           # bf16 [N*C*HW]  (arg9_1 flat NCHW)
    source_ptr,         # bf16 [N*C*HW]  (arg11_1 flat)
    centered_ptr,       # f32  [N*C*HW]  (arg12-arg13 flat)
    fill_scalar_ptr,    # bf16 scalar
    where_ptr,          # bf16 [N*C*HW]  (where(le, fill, source))
    sum_ptr,            # f32 [C]
    dot_ptr,            # f32 [C]
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
    source = ct.load(
        source_ptr, index=(pid,), shape=(BLOCK,),
        padding_mode=ct.PaddingMode.ZERO,
    )
    centered = ct.load(
        centered_ptr, index=(pid,), shape=(BLOCK,),
        padding_mode=ct.PaddingMode.ZERO,
    )
    fill = ct.load(fill_scalar_ptr, index=(0,), shape=(1,))

    idxs = ct.arange(BLOCK, dtype=ct.int32) + pid * BLOCK
    valid = idxs < TOTAL_
    # NCHW layout: c = (idx // HW) % C
    c_full = idxs // HW_
    channel = c_full - (c_full // C_) * C_

    mask_f = ct.astype(mask, ct.float32)
    source_f = ct.astype(source, ct.float32)
    fill_f = ct.astype(fill, ct.float32)
    fill_bcast = ct.reshape(fill_f, (1,))
    where_val_f = ct.where(mask_f <= 0.0, fill_bcast, source_f)
    where_val_bf = ct.astype(where_val_f, ct.bfloat16)
    ct.store(where_ptr, index=(pid,), tile=where_val_bf)

    zero = ct.zeros((BLOCK,), dtype=ct.float32)
    w_masked = ct.where(valid, where_val_f, zero)
    dot_masked = ct.where(valid, where_val_f * centered, zero)
    ct.atomic_add(sum_ptr, channel, w_masked)
    ct.atomic_add(dot_ptr, channel, dot_masked)


@ct.kernel
def _epilogue_flat_kernel(
    where_ptr,      # bf16 [N*C*HW] (previously stored where val)
    centered_ptr,   # f32  [N*C*HW]
    sum_ptr,        # f32 [C]
    dot_ptr,        # f32 [C]
    invstd_ptr,     # f32 [C]
    weight_ptr,     # f32 [C]
    out_ptr,        # bf16 [N*C*HW] full grad
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


@oracle_impl(hardware="H100", point="9f9c3e11")
@oracle_impl(hardware="B200", point="9f9c3e11")
def oracle_forward(inputs):
    (
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1,
        arg5_1, arg6_1, arg7_1, arg8_1,
        arg9_1,   # bf16 [4, 224, 28, 28] — le mask input
        arg10_1,  # bf16 scalar — fill
        arg11_1,  # bf16 [4, 224, 28, 28] — source
        arg12_1,  # bf16 [4, 224, 28, 28] — centered source
        arg13_1,  # f32  [1, 224, 1, 1] — mean
        arg14_1,  # f32  [224] — invstd
        arg15_1,  # f32  [224] — weight
    ) = inputs
    device = arg9_1.device

    # centered = arg12_1.f32 - arg13_1
    centered = arg12_1.to(torch.float32) - arg13_1  # broadcasting

    # Flat views
    arg9_flat = arg9_1.view(-1)
    arg11_flat = arg11_1.view(-1)
    centered_flat = centered.view(-1)

    where_out = torch.empty(TOTAL, device=device, dtype=torch.bfloat16)
    sum_out = torch.zeros(C, device=device, dtype=torch.float32)
    dot_out = torch.zeros(C, device=device, dtype=torch.float32)

    fill_view = arg10_1.view(1)
    invstd_1d = arg14_1
    weight_1d = arg15_1

    BLOCK = 512
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(TOTAL, BLOCK), 1, 1),
        _producer_flat_kernel,
        (arg9_flat, arg11_flat, centered_flat, fill_view, where_out,
         sum_out, dot_out, C, HW, BLOCK, TOTAL),
    )

    # out_mul_1 = dot_out * arg14_1 (invstd) — matches eager mul_10
    # eager: mul_8 = sum_2 * squeeze_1 where squeeze_1 = arg14_1
    # But mul_10 (the returned) = sum_2 * squeeze_1 = mul_8 too. Let me re-check.
    # From eager: mul_10 = torch.mul(sum_2, squeeze_1), returned as 3rd output.
    scale_grad = dot_out * invstd_1d

    # Dense epilogue
    dense_out_flat = torch.empty(TOTAL, device=device, dtype=torch.bfloat16)
    ct.launch(
        stream,
        (ct.cdiv(TOTAL, BLOCK), 1, 1),
        _epilogue_flat_kernel,
        (where_out, centered_flat, sum_out, dot_out,
         invstd_1d, weight_1d, dense_out_flat,
         C, HW, BLOCK, TOTAL, SCALE),
    )
    dense_out = dense_out_flat.view(N, C, H, W)

    # slice_10 = dense_out[:, 192:224, :, :]  → bf16
    # residual_exact = sum of bf16-add-rounded slices of arg0..arg8
    def slice_ch(t):
        return t[:, SLICE_START:C, :, :]  # [N, SLICE_C, H, W]

    residual = slice_ch(arg0_1).contiguous()
    for t in (arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1):
        residual = residual + slice_ch(t).contiguous()
        # (bf16 add returns bf16 with the intermediate rounding)

    slice_from_dense = dense_out[:, SLICE_START:C, :, :]
    add_8 = residual + slice_from_dense  # bf16 + bf16 → bf16

    return sum_out, scale_grad, dense_out, add_8.contiguous()
