"""cuTile port of pointwise_00475df23925: ShuffleNet BN + ReLU + channel-shuffle cat.

Computes BN+ReLU via a cuTile kernel over (N*C, HW) rows, then torch is
used for the cat+channel-shuffle layout dance (produces the final split
views).
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


N = 64
C = 232
H = 7
W = 7
HW = H * W
OUT_C = 2 * C
EPS = 1.0e-5


@ct.kernel
def _bn_relu_pointwise_kernel(
    x_ptr,          # bf16 [N * C, HW_PAD]
    mean_ptr,       # bf16 [N * C, HW_PAD] broadcast
    invstd_ptr,     # f32 [N * C, HW_PAD] broadcast
    weight_ptr,     # bf16 [N * C, HW_PAD] broadcast
    bias_ptr,       # bf16 [N * C, HW_PAD] broadcast
    relu_ptr,       # bf16 [N * C, HW_PAD] output
    BLOCK_HW: ct.Constant[int],
):
    row = ct.bid(0)
    x_bf = ct.load(x_ptr, index=(row, 0), shape=(1, BLOCK_HW))
    mean_bf = ct.load(mean_ptr, index=(row, 0), shape=(1, BLOCK_HW))
    invstd = ct.load(invstd_ptr, index=(row, 0), shape=(1, BLOCK_HW))
    weight_bf = ct.load(weight_ptr, index=(row, 0), shape=(1, BLOCK_HW))
    bias_bf = ct.load(bias_ptr, index=(row, 0), shape=(1, BLOCK_HW))

    x_f = ct.astype(x_bf, ct.float32)
    mean_f = ct.astype(mean_bf, ct.float32)
    weight_f = ct.astype(weight_bf, ct.float32)
    bias_f = ct.astype(bias_bf, ct.float32)

    centered = x_f - mean_f
    normalized = centered * invstd
    affine = normalized * weight_f + bias_f
    rounded_bf = ct.astype(affine, ct.bfloat16)

    zero_bf = ct.full((1, BLOCK_HW), 0.0, dtype=ct.bfloat16)
    # ReLU preserves NaN: pass through when x > 0 OR x != x (NaN check).
    is_positive = rounded_bf > zero_bf
    is_nan = rounded_bf != rounded_bf
    keep = is_positive | is_nan
    relu = ct.where(keep, rounded_bf, zero_bf)
    ct.store(relu_ptr, index=(row, 0), tile=relu)


@oracle_impl(hardware="B200", point="46084af7", BLOCK_HW=64)
def oracle_forward(inputs, *, BLOCK_HW: int):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, _s0, _s1 = inputs
    device = arg1_1.device

    # Compute invstd in torch, matching the reference exact op sequence
    # (add + sqrt + reciprocal) which propagates NaN for negative var+eps.
    var_f = arg2_1.to(torch.float32) + EPS
    invstd = torch.reciprocal(torch.sqrt(var_f))

    # Pad HW=49 -> 64 for cuTile power-of-2 shape.
    # We create a padded backing tensor and slice/reslice.
    n_rows = N * C
    x_pad = torch.zeros((n_rows, BLOCK_HW), device=device, dtype=torch.bfloat16)
    x_flat = arg1_1.contiguous().view(n_rows, HW)
    x_pad[:, :HW].copy_(x_flat)

    mean_pad = torch.zeros((n_rows, BLOCK_HW), device=device, dtype=torch.bfloat16)
    invstd_pad = torch.zeros((n_rows, BLOCK_HW), device=device, dtype=torch.float32)
    weight_pad = torch.zeros((n_rows, BLOCK_HW), device=device, dtype=torch.bfloat16)
    bias_pad = torch.zeros((n_rows, BLOCK_HW), device=device, dtype=torch.bfloat16)

    mean_pad[:, :HW] = arg0_1.unsqueeze(0).expand(N, C).reshape(n_rows, 1).expand(n_rows, HW)
    invstd_pad[:, :HW] = invstd.unsqueeze(0).expand(N, C).reshape(n_rows, 1).expand(n_rows, HW)
    weight_pad[:, :HW] = arg3_1.unsqueeze(0).expand(N, C).reshape(n_rows, 1).expand(n_rows, HW)
    bias_pad[:, :HW] = arg4_1.unsqueeze(0).expand(N, C).reshape(n_rows, 1).expand(n_rows, HW)

    relu_pad = torch.empty((n_rows, BLOCK_HW), device=device, dtype=torch.bfloat16)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (n_rows, 1, 1), _bn_relu_pointwise_kernel,
        (x_pad, mean_pad, invstd_pad, weight_pad, bias_pad, relu_pad, BLOCK_HW),
    )
    relu_view = relu_pad[:, :HW].contiguous().view(N, C, H, W)

    # Cat with arg5_1 to form (N, 2C, H, W) then channel-shuffle.
    cat_out = torch.empty((N, OUT_C, H, W), device=device, dtype=torch.bfloat16)
    cat_out[:, :C, :, :] = arg5_1
    cat_out[:, C:, :, :] = relu_view

    shuffled = (
        cat_out.view(N, 2, C, H, W)
        .permute(0, 2, 1, 3, 4)
        .contiguous()
        .view(N, OUT_C, H, W)
    )
    return shuffled[:, :C, :, :], shuffled[:, C:, :, :]
