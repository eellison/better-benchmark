"""cuTile port of pointwise_7308e0024674: ShuffleNet BN + ReLU + channel-shuffle split.

Ports Triton `_shuffle_bn_relu_kernel` — for each of two `[64, 58, 28, 28]`
branches computes BN(x) then bf16 rounding then relu, then interleaves the
two branches into a `[64, 116, 28, 28]` tensor and returns
`(shuffled[:, :58], shuffled[:, 58:])`.

Channel=58 (non-pow2), so we run a flat NCHW kernel with `flat_idx // HW % C`
gather of the per-channel params, then use torch to interleave and split.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 64
CHANNELS = 58
HEIGHT = 28
WIDTH = 28
HW = HEIGHT * WIDTH
BRANCH_NUMEL = BATCH * CHANNELS * HW  # 2910208
EPS = 1.0e-5


@ct.kernel
def _bn_relu_flat_kernel(
    mean_ptr,     # bf16[CHANNELS]
    x_ptr,        # bf16[BRANCH_NUMEL] (NCHW contiguous flat)
    var_ptr,      # bf16[CHANNELS]
    weight_ptr,   # bf16[CHANNELS]
    bias_ptr,     # bf16[CHANNELS]
    out_ptr,      # bf16[BRANCH_NUMEL]
    C: ct.Constant[int],
    HW_: ct.Constant[int],
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    idx = pid * BLOCK + ct.arange(BLOCK, dtype=ct.int32)
    c_idx = (idx // HW_) % C

    x = ct.load(x_ptr, index=(pid,), shape=(BLOCK,))
    mean = ct.astype(ct.gather(mean_ptr, c_idx), ct.float32)
    var = ct.astype(ct.gather(var_ptr, c_idx), ct.float32)
    weight = ct.astype(ct.gather(weight_ptr, c_idx), ct.float32)
    bias = ct.astype(ct.gather(bias_ptr, c_idx), ct.float32)

    inv = 1.0 / ct.sqrt(var + EPS)
    normalized_f32 = (ct.astype(x, ct.float32) - mean) * inv * weight + bias
    normalized_bf16 = ct.astype(normalized_f32, ct.bfloat16)
    zero = ct.astype(ct.zeros((BLOCK,), dtype=ct.bfloat16), ct.bfloat16)
    # NaN-preserving relu: where(y > 0 | y != y, y, 0)
    relu = ct.where(ct.isnan(normalized_bf16) | (normalized_bf16 > zero),
                    normalized_bf16, zero)
    ct.store(out_ptr, index=(pid,), tile=relu)


@oracle_impl(hardware="B200", point="cff026d0", BLOCK=1024)
def oracle_forward(inputs, *, BLOCK):
    (
        mean_a, conv_a, var_a, weight_a, bias_a,
        mean_b, conv_b, var_b, weight_b, bias_b,
        _shape0, _shape1,
    ) = inputs
    del _shape0, _shape1

    branch_a = torch.empty((BRANCH_NUMEL,), device=conv_a.device, dtype=torch.bfloat16)
    branch_b = torch.empty((BRANCH_NUMEL,), device=conv_a.device, dtype=torch.bfloat16)
    conv_a_flat = conv_a.contiguous().view(BRANCH_NUMEL)
    conv_b_flat = conv_b.contiguous().view(BRANCH_NUMEL)

    stream = torch.cuda.current_stream()
    grid = (ct.cdiv(BRANCH_NUMEL, BLOCK), 1, 1)
    ct.launch(stream, grid, _bn_relu_flat_kernel,
              (mean_a, conv_a_flat, var_a, weight_a, bias_a, branch_a,
               CHANNELS, HW, BLOCK))
    ct.launch(stream, grid, _bn_relu_flat_kernel,
              (mean_b, conv_b_flat, var_b, weight_b, bias_b, branch_b,
               CHANNELS, HW, BLOCK))

    # Reshape to (B, C, H, W)
    branch_a_view = branch_a.view(BATCH, CHANNELS, HEIGHT, WIDTH)
    branch_b_view = branch_b.view(BATCH, CHANNELS, HEIGHT, WIDTH)
    # Interleave along channel dim: shuffled[:, 2c] = branch_a[:, c],
    # shuffled[:, 2c+1] = branch_b[:, c]. stack at dim=2 gives (B, C, 2, H, W);
    # reshape to (B, 2C, H, W) collapses (C, 2) into an interleaved 2C dim.
    shuffled = torch.stack([branch_a_view, branch_b_view], dim=2).reshape(
        BATCH, 2 * CHANNELS, HEIGHT, WIDTH
    )
    return shuffled[:, :CHANNELS, :, :], shuffled[:, CHANNELS:, :, :]
