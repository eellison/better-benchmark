"""cuTile port of mean_dc61a7b3b5a3: RepVGG dual BN-inference + ReLU + 7x7 spatial mean.

Two BN-inference branches on channels-last bf16 [N, C, 7, 7] tensors,
per-channel affine, bf16 rounding at each stage, elementwise add of the two
bf16 branches (also rounded), a ReLU, and a spatial mean over (H=7, W=7) →
bf16 [N, C]. Non-power-of-2 HW=49; padded to 8x8 tile with zero-loads for
the masked reduction.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


EPS = 1.0e-5
H = 7
W = 7
H_PADDED = 8
W_PADDED = 8


@ct.kernel
def _dual_bn_relu_mean_kernel(
    x0,       # bf16 (N, C, H, W) channels-last
    x1,       # bf16 (N, C, H, W) channels-last
    mean0,    # bf16 (C,)
    var0,     # bf16 (C,)
    weight0,  # bf16 (C,)
    bias0,    # bf16 (C,)
    mean1,
    var1,
    weight1,
    bias1,
    out,      # bf16 (N, C)
):
    n = ct.bid(0)
    c = ct.bid(1)

    v0 = ct.load(
        x0, index=(n, c, 0, 0), shape=(1, 1, H_PADDED, W_PADDED),
        padding_mode=ct.PaddingMode.ZERO,
    )
    v1 = ct.load(
        x1, index=(n, c, 0, 0), shape=(1, 1, H_PADDED, W_PADDED),
        padding_mode=ct.PaddingMode.ZERO,
    )
    v0_f = ct.astype(v0, ct.float32)
    v1_f = ct.astype(v1, ct.float32)

    m0 = ct.astype(ct.load(mean0, index=(c,), shape=(1,)), ct.float32)
    va0 = ct.astype(ct.load(var0, index=(c,), shape=(1,)), ct.float32)
    w0 = ct.astype(ct.load(weight0, index=(c,), shape=(1,)), ct.float32)
    b0 = ct.astype(ct.load(bias0, index=(c,), shape=(1,)), ct.float32)
    m1 = ct.astype(ct.load(mean1, index=(c,), shape=(1,)), ct.float32)
    va1 = ct.astype(ct.load(var1, index=(c,), shape=(1,)), ct.float32)
    w1 = ct.astype(ct.load(weight1, index=(c,), shape=(1,)), ct.float32)
    b1 = ct.astype(ct.load(bias1, index=(c,), shape=(1,)), ct.float32)

    invstd0 = 1.0 / ct.sqrt(va0 + EPS)
    invstd1 = 1.0 / ct.sqrt(va1 + EPS)

    # Broadcast scalars (1,) to (1,1,H_PADDED,W_PADDED)
    m0_4 = ct.reshape(m0, (1, 1, 1, 1))
    invstd0_4 = ct.reshape(invstd0, (1, 1, 1, 1))
    w0_4 = ct.reshape(w0, (1, 1, 1, 1))
    b0_4 = ct.reshape(b0, (1, 1, 1, 1))
    m1_4 = ct.reshape(m1, (1, 1, 1, 1))
    invstd1_4 = ct.reshape(invstd1, (1, 1, 1, 1))
    w1_4 = ct.reshape(w1, (1, 1, 1, 1))
    b1_4 = ct.reshape(b1, (1, 1, 1, 1))

    branch0 = (v0_f - m0_4) * invstd0_4 * w0_4 + b0_4
    branch1 = (v1_f - m1_4) * invstd1_4 * w1_4 + b1_4

    # Materialize each branch as bf16 (roundtrip through cast) — mirrors
    # Triton's cvt.rn.bf16.f32 boundary.
    branch0_bf = ct.astype(branch0, ct.bfloat16)
    branch1_bf = ct.astype(branch1, ct.bfloat16)
    branch0_f = ct.astype(branch0_bf, ct.float32)
    branch1_f = ct.astype(branch1_bf, ct.float32)
    summed_bf = ct.astype(branch0_f + branch1_f, ct.bfloat16)
    summed_f = ct.astype(summed_bf, ct.float32)

    # ReLU (preserves NaN: (x > 0) | (x != x) selects x, else 0)
    relu = ct.where((summed_f > 0.0) | (summed_f != summed_f), summed_f, 0.0)

    # Mask out the (H_PADDED, W_PADDED) region beyond H, W before summing
    h_idx = ct.arange(H_PADDED, dtype=ct.int32)
    w_idx = ct.arange(W_PADDED, dtype=ct.int32)
    h_mask = h_idx < H
    w_mask = w_idx < W
    h_mask_4 = ct.reshape(h_mask, (1, 1, H_PADDED, 1))
    w_mask_4 = ct.reshape(w_mask, (1, 1, 1, W_PADDED))
    valid = h_mask_4 & w_mask_4
    relu_masked = ct.where(valid, relu, 0.0)
    reduced = ct.sum(relu_masked) * (1.0 / (H * W))
    reduced_bf = ct.astype(reduced, ct.bfloat16)
    # Store single element at [n, c]
    reduced_2d = ct.reshape(reduced_bf, (1, 1))
    ct.store(out, index=(n, c), tile=reduced_2d)


@oracle_impl(hardware="B200", point="84ddc6ab")
def oracle_forward(inputs):
    mean0, x0, var0, weight0, bias0, mean1, x1, var1, weight1, bias1, *_ = inputs
    n = int(x0.shape[0])
    c = int(x0.shape[1])
    out = torch.empty_strided(
        (n, c), (c, 1), device=x0.device, dtype=torch.bfloat16
    )
    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (n, c, 1),
        _dual_bn_relu_mean_kernel,
        (x0, x1, mean0, var0, weight0, bias0,
         mean1, var1, weight1, bias1, out),
    )
    return out
