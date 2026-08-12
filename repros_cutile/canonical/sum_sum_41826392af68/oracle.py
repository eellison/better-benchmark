"""cuTile port of sum_sum_41826392af68: ShuffleNet channel-shuffle + BN-backward.

Strategy:
  * torch: build the shuffled tensor `view_1` = clone(permute(view(arg0+arg1),
    [0, 2, 1, 3, 4])). Extract slice_1 (upper half of shuffle).
  * torch: BN forward to produce the ReLU mask (bf16 relu -> le).
  * cuTile Kernel 1 (row producer): compute per-channel `sum_where` and
    `sum_centered` via atomic_add. Writes an intermediate "selected"
    tensor of bf16.
  * cuTile Kernel 2 (dense epilogue): computes the full BN-backward
    tensor in bf16.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


N = 128
C_IN = 116
C = 58
H = 28
W = 28
HW = H * W
TOTAL = N * C * HW
SCALE = 9.964923469387754e-06


@ct.kernel
def _producer_kernel(
    selected_ptr,   # bf16 [N*HW*C]  (channels-last-permuted flat, N,H,W,C -> flat)
    centered_ptr,   # f32  [N*HW*C]  (same layout)
    sum_ptr,        # f32  [C]       (per-channel sum_where)
    dot_ptr,        # f32  [C]       (per-channel sum(centered*selected))
    C_: ct.Constant[int],
    HWC: ct.Constant[int],
    BLOCK: ct.Constant[int],
    TOTAL_: ct.Constant[int],
):
    pid = ct.bid(0)
    selected = ct.load(
        selected_ptr, index=(pid,), shape=(BLOCK,),
        padding_mode=ct.PaddingMode.ZERO,
    )
    centered = ct.load(
        centered_ptr, index=(pid,), shape=(BLOCK,),
        padding_mode=ct.PaddingMode.ZERO,
    )
    idxs = ct.arange(BLOCK, dtype=ct.int32) + pid * BLOCK
    valid = idxs < TOTAL_
    channel = idxs - (idxs // C_) * C_  # idxs % C

    sel_f = ct.astype(selected, ct.float32)
    zero = ct.zeros((BLOCK,), dtype=ct.float32)
    sel_masked = ct.where(valid, sel_f, zero)
    product = ct.where(valid, sel_f * centered, zero)

    ct.atomic_add(sum_ptr, channel, sel_masked)
    ct.atomic_add(dot_ptr, channel, product)


@ct.kernel
def _epilogue_kernel(
    selected_ptr,   # bf16 [total] channels-last flat
    centered_ptr,   # f32  [total] channels-last flat
    sum_ptr,        # f32  [C]
    dot_ptr,        # f32  [C]
    invstd_ptr,     # f32  [C]
    weight_ptr,     # f32  [C]
    dense_out_ptr,  # bf16 [total] channels-last flat
    C_: ct.Constant[int],
    HWC: ct.Constant[int],
    BLOCK: ct.Constant[int],
    TOTAL_: ct.Constant[int],
    SCALE_: ct.Constant[float],
):
    pid = ct.bid(0)
    selected = ct.load(
        selected_ptr, index=(pid,), shape=(BLOCK,),
        padding_mode=ct.PaddingMode.ZERO,
    )
    centered = ct.load(
        centered_ptr, index=(pid,), shape=(BLOCK,),
        padding_mode=ct.PaddingMode.ZERO,
    )
    idxs = ct.arange(BLOCK, dtype=ct.int32) + pid * BLOCK
    channel = idxs - (idxs // C_) * C_
    sum_val = ct.gather(sum_ptr, channel)
    dot_val = ct.gather(dot_ptr, channel)
    invstd = ct.gather(invstd_ptr, channel)
    weight = ct.gather(weight_ptr, channel)

    sel_f = ct.astype(selected, ct.float32)
    mean_term = sum_val * SCALE_
    invstd_sq = invstd * invstd
    variance_term = dot_val * SCALE_ * invstd_sq
    output_scale = invstd * weight
    after_variance = sel_f - centered * variance_term
    after_mean = after_variance - mean_term
    out = after_mean * output_scale
    ct.store(dense_out_ptr, index=(pid,), tile=ct.astype(out, ct.bfloat16))


@oracle_impl(hardware="B200", point="aa052f71")
def oracle_forward(inputs):
    lhs, rhs, bn_input, mean_v, invstd_v, weight, bias, fill, _shape0, _shape1 = inputs
    device = lhs.device

    # Step 1: shuffle output (matches eager view_1)
    added = (lhs + rhs)  # bf16 [128, 116, 28, 28] channels-last
    # Ensure operations follow the channels-last layout the eager Repro implies.
    shuffle_source = added.view(N, C, 2, H, W).permute(0, 2, 1, 3, 4).contiguous()
    shuffle_out = shuffle_source.view(N, C_IN, H, W)
    # The output stride in the returned view_1 was inferred to match
    # the source's channels-last layout — but the eager captures did `clone` (contig),
    # then reshape to NCHW, then slicing works elementwise on that contiguous.
    # For strict layout: torch.clone default is `torch.contiguous_format`
    # so the shuffle_out is standard NCHW contiguous.

    # slice_1 = shuffle_out[:, 58:116, :, :]  → [N, C, H, W]
    slice_1 = shuffle_out[:, C:, :, :].contiguous()

    # Step 2: BN forward to reproduce ReLU mask (bf16 relu → le)
    # sub = bn_input.f32 - arg3  (arg3 is mean shape [1,C,1,1])
    # mul = sub * invstd
    # mul_1 = mul * unsqueeze(weight, [-1, -1])
    # add_1 = mul_1 + unsqueeze(bias, [-1, -1])
    # bf16 = add_1.bf16
    # relu = bf16.relu
    # le = relu <= 0
    bn_input_f = bn_input.to(torch.float32)  # channels-last
    sub_v = bn_input_f - mean_v  # broadcasting [1,C,1,1]
    mul_v = sub_v * invstd_v
    mul_1 = mul_v * weight.view(1, C, 1, 1)
    add_v = mul_1 + bias.view(1, C, 1, 1)
    affine_bf = add_v.to(torch.bfloat16)
    relu = torch.relu(affine_bf)  # bf16
    le = relu <= 0  # bool

    # selected = where(le, fill (bf16 scalar), slice_1) → bf16
    selected = torch.where(le, fill, slice_1)

    # For sum reductions along [0, H, W] we need a channels-last-friendly
    # flat layout. The eager sum_1 = sum(f32(selected), [0, 2, 3]) works
    # equivalently to a channel reduction. We'll do this with a cuTile
    # atomic_add producer over N*H*W*C flat indices with channel = idx%C.
    #
    # To match cuTile's default flat load semantics, we pass channels-last
    # flat tensors of shape [N*H*W*C] where channel is the innermost axis.
    # selected is NCHW contiguous (from where(le, fill, slice_1) where slice_1
    # is contig NCHW), so permute -> NHWC needs contiguous(). sub_v inherits
    # channels-last strides from bn_input, so permute->NHWC is metadata-only
    # (already contiguous) — use reshape.
    selected_perm = selected.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
    selected_flat = selected_perm.view(-1)

    centered_flat = sub_v.permute(0, 2, 3, 1).reshape(-1)

    sum_out = torch.zeros(C, device=device, dtype=torch.float32)
    dot_out = torch.zeros(C, device=device, dtype=torch.float32)

    BLOCK = 512
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(TOTAL, BLOCK), 1, 1),
        _producer_kernel,
        (selected_flat, centered_flat, sum_out, dot_out,
         C, HW * C, BLOCK, TOTAL),
    )

    # Compute scale_grad = sum_2 * invstd_v_squeezed
    # sum_2 = dot_out here (since selected * centered)
    invstd_squeezed = invstd_v.view(C)
    scale_grad = dot_out * invstd_squeezed  # matches mul_10 = sum_2 * squeeze_1

    # Dense epilogue via cuTile kernel
    dense_out_perm = torch.empty((N, H, W, C), device=device, dtype=torch.bfloat16)
    dense_out_flat = dense_out_perm.view(-1)

    ct.launch(
        stream,
        (ct.cdiv(TOTAL, BLOCK), 1, 1),
        _epilogue_kernel,
        (selected_flat, centered_flat, sum_out, dot_out,
         invstd_squeezed, weight, dense_out_flat,
         C, HW * C, BLOCK, TOTAL, SCALE),
    )
    # Reshape to channels-last strided [N, C, H, W]
    dense_out = torch.empty_strided(
        (N, C, H, W),
        (C * HW, 1, W * C, C),
        device=device,
        dtype=torch.bfloat16,
    )
    dense_out.copy_(dense_out_perm.permute(0, 3, 1, 2))

    return shuffle_out, sum_out, scale_grad, dense_out
