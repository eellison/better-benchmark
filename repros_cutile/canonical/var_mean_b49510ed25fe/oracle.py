"""cuTile port of var_mean_b49510ed25fe: pytorch_unet BN + affine + ReLU + maxpool2x2.

Four @ct.kernel bodies mirroring Triton's 4-kernel path:
  1. _partial_stats_kernel: per (channel, chunk) partial sum/sqsum tiles.
  2. _finalize_stats_kernel: per-channel combine + running-stat update.
  3. _affine_relu_kernel: per-element BN affine + bf16-cast + ReLU.
  4. _maxpool2x2_kernel: 2x2 maxpool with offsets.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


EPS = 1.0e-5
MOMENTUM = 0.1
RUNNING_VAR_CORRECTION = 1.0001050530517912


@ct.kernel
def _partial_stats_kernel(
    x_ptr,          # bf16 [C, H, W] flat (contiguous, N=1)
    partial_sum_ptr,   # f32 [C, NUM_CHUNKS]
    partial_sq_ptr,    # f32 [C, NUM_CHUNKS]
    HW: ct.Constant[int],
    NUM_CHUNKS: ct.Constant[int],
    CHUNK: ct.Constant[int],
    BLOCK_R: ct.Constant[int],
):
    channel = ct.bid(0)
    chunk = ct.bid(1)
    r_off = ct.arange(BLOCK_R, dtype=ct.int32)
    base_off = chunk * CHUNK + r_off
    active = (r_off < ct.full((BLOCK_R,), CHUNK, dtype=ct.int32)) & (base_off < ct.full((BLOCK_R,), HW, dtype=ct.int32))

    off = ct.where(active, channel * HW + base_off, ct.zeros((BLOCK_R,), dtype=ct.int32))
    x_bf = ct.gather(x_ptr, (off,))
    x_f = ct.astype(x_bf, ct.float32)
    zeros_f = ct.zeros((BLOCK_R,), dtype=ct.float32)
    ones_f = ct.full((BLOCK_R,), 1.0, dtype=ct.float32)
    valid_f = ct.where(active, ones_f, zeros_f)
    x_masked = x_f * valid_f
    sum_x = ct.sum(x_masked)
    sum_x2 = ct.sum(x_masked * x_f)
    ct.store(partial_sum_ptr, index=(channel, chunk), tile=ct.reshape(sum_x, (1, 1)))
    ct.store(partial_sq_ptr, index=(channel, chunk), tile=ct.reshape(sum_x2, (1, 1)))


@ct.kernel
def _finalize_stats_kernel(
    partial_sum_ptr,    # f32 [C, NUM_CHUNKS]
    partial_sq_ptr,     # f32 [C, NUM_CHUNKS]
    running_mean_ptr,   # f32 [C]
    running_var_ptr,    # f32 [C]
    mean_ptr,           # f32 [C]
    invstd_ptr,         # f32 [C]
    HW: ct.Constant[int],
    NUM_CHUNKS: ct.Constant[int],
    BLOCK_N: ct.Constant[int],  # power of 2 >= NUM_CHUNKS
):
    channel = ct.bid(0)
    n_off = ct.arange(BLOCK_N, dtype=ct.int32)
    n_lim = ct.full((BLOCK_N,), NUM_CHUNKS, dtype=ct.int32)
    active = n_off < n_lim
    # Row-load from partial_..._ptr[channel, 0:BLOCK_N] via ct.load(index=(channel,0), shape=(1,BLOCK_N)).
    sums = ct.reshape(ct.load(partial_sum_ptr, index=(channel, 0), shape=(1, BLOCK_N),
                              padding_mode=ct.PaddingMode.ZERO), (BLOCK_N,))
    sqs = ct.reshape(ct.load(partial_sq_ptr, index=(channel, 0), shape=(1, BLOCK_N),
                              padding_mode=ct.PaddingMode.ZERO), (BLOCK_N,))
    zeros_f = ct.zeros((BLOCK_N,), dtype=ct.float32)
    ones_f = ct.full((BLOCK_N,), 1.0, dtype=ct.float32)
    valid_f = ct.where(active, ones_f, zeros_f)
    total_sum = ct.sum(sums * valid_f)
    total_sq = ct.sum(sqs * valid_f)
    inv_hw = 1.0 / HW
    mean = total_sum * inv_hw
    var = total_sq * inv_hw - mean * mean
    var_pos = ct.maximum(var, 0.0)
    invstd = ct.rsqrt(var_pos + EPS)

    old_mean = ct.reshape(ct.load(running_mean_ptr, index=(channel,), shape=(1,)), ())
    old_var = ct.reshape(ct.load(running_var_ptr, index=(channel,), shape=(1,)), ())
    new_mean = old_mean * (1.0 - MOMENTUM) + mean * MOMENTUM
    new_var = old_var * (1.0 - MOMENTUM) + var_pos * RUNNING_VAR_CORRECTION * MOMENTUM

    ct.store(mean_ptr, index=(channel,), tile=ct.reshape(mean, (1,)))
    ct.store(invstd_ptr, index=(channel,), tile=ct.reshape(invstd, (1,)))
    ct.store(running_mean_ptr, index=(channel,), tile=ct.reshape(new_mean, (1,)))
    ct.store(running_var_ptr, index=(channel,), tile=ct.reshape(new_var, (1,)))


@ct.kernel
def _affine_relu_kernel(
    x_ptr,          # bf16 [C, H, W] flat
    mean_ptr,       # f32 [C]
    invstd_ptr,     # f32 [C]
    weight_ptr,     # f32 [C]
    bias_ptr,       # f32 [C]
    relu_ptr,       # bf16 [C, H, W] flat
    HW: ct.Constant[int],
    C: ct.Constant[int],
    TOTAL: ct.Constant[int],  # C * HW
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    off = pid * BLOCK + ct.arange(BLOCK, dtype=ct.int32)
    active = off < ct.full((BLOCK,), TOTAL, dtype=ct.int32)
    off_masked = ct.where(active, off, ct.zeros((BLOCK,), dtype=ct.int32))
    channel = off_masked // HW  # per-element channel

    x_bf = ct.gather(x_ptr, (off_masked,))
    x_f = ct.astype(x_bf, ct.float32)
    mean = ct.gather(mean_ptr, (channel,))
    invstd = ct.gather(invstd_ptr, (channel,))
    weight = ct.gather(weight_ptr, (channel,))
    bias = ct.gather(bias_ptr, (channel,))
    y_f = (x_f - mean) * invstd * weight + bias
    y_bf = ct.astype(y_f, ct.bfloat16)
    y_bf_f = ct.astype(y_bf, ct.float32)
    zeros_f = ct.zeros((BLOCK,), dtype=ct.float32)
    relu_f = ct.where(y_bf_f > 0.0, y_bf_f, zeros_f)
    relu_bf = ct.astype(relu_f, ct.bfloat16)
    zero_bf = ct.full((BLOCK,), 0.0, dtype=ct.bfloat16)
    ct.scatter(relu_ptr, (off_masked,), ct.where(active, relu_bf, zero_bf), mask=active)


@ct.kernel
def _maxpool2x2_kernel(
    relu_ptr,     # bf16 [C, H, W]
    values_ptr,   # bf16 [C, OUT_H, OUT_W]
    offsets_ptr,  # int8 [C, OUT_H, OUT_W]
    OUT_H_: ct.Constant[int],
    OUT_W_: ct.Constant[int],
    OUT_H_PAD_: ct.Constant[int],
    OUT_W_PAD_: ct.Constant[int],
):
    channel = ct.bid(0)
    out_h_blk = ct.bid(1)
    out_w_blk = ct.bid(2)
    x = ct.load(
        relu_ptr, index=(channel, out_h_blk, out_w_blk), shape=(1, 2, 2),
        padding_mode=ct.PaddingMode.ZERO,
    )
    x_2d = ct.reshape(x, (2, 2))
    x_flat = ct.reshape(ct.astype(x_2d, ct.float32), (4,))
    argmax = ct.argmax(x_flat, axis=0)
    max_val = ct.max(x_flat)
    ct.store(
        values_ptr,
        index=(channel, out_h_blk, out_w_blk),
        tile=ct.reshape(ct.astype(max_val, ct.bfloat16), (1, 1, 1)),
    )
    ct.store(
        offsets_ptr,
        index=(channel, out_h_blk, out_w_blk),
        tile=ct.reshape(ct.astype(argmax, ct.int8), (1, 1, 1)),
    )


def _next_pow2(v: int) -> int:
    return 1 << (int(v) - 1).bit_length()


@oracle_impl(hardware="B200", point="6750c5ad")
@oracle_impl(hardware="B200", point="8e0806dd")
@oracle_impl(hardware="B200", point="538399db")
@oracle_impl(hardware="B200", point="37819cd3")
def oracle_forward(inputs, **_kwargs):
    x, running_mean, running_var, weight, bias, _, _ = inputs
    n = int(x.shape[0])
    c = int(x.shape[1])
    h = int(x.shape[2])
    w = int(x.shape[3])
    hw = h * w
    total = c * hw
    out_h = (h - 2) // 2 + 1
    out_w = (w - 2) // 2 + 1
    device = x.device
    assert n == 1  # simplifies channel offset arithmetic

    x_flat = x.contiguous().view(-1)
    relu = torch.empty((1, c, h, w), device=device, dtype=torch.bfloat16)
    relu_flat = relu.view(-1)
    values = torch.empty((1, c, out_h, out_w), device=device, dtype=torch.bfloat16)
    offsets = torch.empty((1, c, out_h, out_w), device=device, dtype=torch.int8)
    mean = torch.empty((c,), device=device, dtype=torch.float32)
    invstd = torch.empty((c,), device=device, dtype=torch.float32)

    # Stats: partial + finalize.
    CHUNK = 8192
    BLOCK_R = 8192
    num_chunks = (hw + CHUNK - 1) // CHUNK
    partial_sum = torch.empty((c, num_chunks), device=device, dtype=torch.float32)
    partial_sq = torch.empty((c, num_chunks), device=device, dtype=torch.float32)
    BLOCK_N = _next_pow2(num_chunks)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (c, num_chunks, 1), _partial_stats_kernel,
        (x_flat, partial_sum, partial_sq, hw, num_chunks, CHUNK, BLOCK_R),
    )
    ct.launch(
        stream, (c, 1, 1), _finalize_stats_kernel,
        (partial_sum, partial_sq, running_mean, running_var, mean, invstd,
         hw, num_chunks, BLOCK_N),
    )

    # Affine + ReLU.
    BLOCK_AR = 1024
    ct.launch(
        stream, (ct.cdiv(total, BLOCK_AR), 1, 1), _affine_relu_kernel,
        (x_flat, mean, invstd, weight, bias, relu_flat,
         hw, c, total, BLOCK_AR),
    )

    # Maxpool.
    relu_3d = relu.view(c, h, w)
    values_3d = values.view(c, out_h, out_w)
    offsets_3d = offsets.view(c, out_h, out_w)
    ct.launch(
        stream, (c, out_h, out_w), _maxpool2x2_kernel,
        (relu_3d, values_3d, offsets_3d, out_h, out_w, out_h, out_w),
    )

    mean_out = mean.view(1, c, 1, 1)
    invstd_out = invstd.view(1, c, 1, 1)
    return mean_out, invstd_out, relu, values, offsets, running_mean, running_var
