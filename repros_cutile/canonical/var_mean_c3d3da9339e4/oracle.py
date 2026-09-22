"""cuTile port of var_mean_c3d3da9339e4: DenseNet channel-concat + BN train + ReLU.

Two cuTile kernels per shape:
  1) Per-channel partial stats: chunked (sum, sum_sq) partial reductions.
  2) Finalize: reduce partials -> mean/var/invstd; update running_mean/var
     inputs; also compute new running stats to copy_ back on the torch side.
  3) BN affine + ReLU epilogue: written by torch (small compute) or a small
     cuTile kernel using the finalized mean/invstd.

The concat is torch.cat + permute+contiguous to (channels, N*H*W) layout.
Running-stat mutation happens on the torch side via torch.ops.aten.copy_
which works fine under CUDA graph capture.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


EPS = 1.0e-5
MOMENTUM = 0.1
MOMENTUM_OLD = 0.9
RUNNING_VAR_CORRECTION = 1.005128205128205


def _next_pow2(v: int) -> int:
    p = 1
    while p < v:
        p *= 2
    return p


@ct.kernel
def _partial_stats_kernel(
    x_ptr,        # bf16 (channels, elements_per_channel_padded)
    ps_ptr,       # f32 (channels, num_chunks)  partial sum
    psq_ptr,      # f32 (channels, num_chunks)  partial sum_sq
    E: ct.Constant[int],
    NUM_CHUNKS: ct.Constant[int],
    BLOCK_E: ct.Constant[int],
):
    channel = ct.bid(0)
    for chunk_idx in ct.static_iter(range(NUM_CHUNKS)):
        cols = chunk_idx * BLOCK_E + ct.arange(BLOCK_E, dtype=ct.int32)
        valid = cols < E
        x_bf = ct.load(
            x_ptr, index=(channel, chunk_idx),
            shape=(1, BLOCK_E),
            padding_mode=ct.PaddingMode.ZERO,
        )
        x_f = ct.astype(x_bf, ct.float32)
        # Mask OOB entries (padded storage may hold garbage from previous
        # buffer reuse, so zero them explicitly).
        valid_2d = ct.reshape(valid, (1, BLOCK_E))
        zero_f = ct.zeros((1, BLOCK_E), dtype=ct.float32)
        x_masked = ct.where(valid_2d, x_f, zero_f)
        # Sum keeping rank-2 so we can store into rank-2 tensor view.
        s = ct.sum(x_masked, axis=1, keepdims=True)   # (1, 1)
        sq = ct.sum(x_masked * x_masked, axis=1, keepdims=True)  # (1, 1)
        ct.store(ps_ptr, index=(channel, chunk_idx), tile=s)
        ct.store(psq_ptr, index=(channel, chunk_idx), tile=sq)


@ct.kernel
def _finalize_stats_kernel(
    ps_ptr,       # f32 (channels, num_chunks)
    psq_ptr,      # f32 (channels, num_chunks)
    mean_out_ptr, # f32 (channels,)
    var_out_ptr,  # f32 (channels,)
    invstd_ptr,   # f32 (channels,)
    NUM_CHUNKS: ct.Constant[int],
    BLOCK_CHUNKS: ct.Constant[int],
    INV_E: ct.Constant[float],
):
    channel = ct.bid(0)
    ps = ct.load(
        ps_ptr, index=(channel, 0),
        shape=(1, BLOCK_CHUNKS),
        padding_mode=ct.PaddingMode.ZERO,
    )
    psq = ct.load(
        psq_ptr, index=(channel, 0),
        shape=(1, BLOCK_CHUNKS),
        padding_mode=ct.PaddingMode.ZERO,
    )
    chunks = ct.arange(BLOCK_CHUNKS, dtype=ct.int32)
    chunk_valid = ct.reshape(chunks < NUM_CHUNKS, (1, BLOCK_CHUNKS))
    zero_f = ct.zeros((1, BLOCK_CHUNKS), dtype=ct.float32)
    ps_masked = ct.where(chunk_valid, ps, zero_f)
    psq_masked = ct.where(chunk_valid, psq, zero_f)
    total_s = ct.sum(ps_masked)
    total_sq = ct.sum(psq_masked)
    mean = total_s * INV_E
    var = total_sq * INV_E - mean * mean
    zero_scalar = ct.zeros((1,), dtype=ct.float32)
    mean_1d = ct.reshape(mean, (1,))
    var_1d = ct.reshape(var, (1,))
    var_clamped = ct.where(var_1d > zero_scalar, var_1d, zero_scalar)
    invstd_1d = ct.rsqrt(var_clamped + EPS)
    ct.store(mean_out_ptr, index=(channel,), tile=mean_1d)
    ct.store(var_out_ptr, index=(channel,), tile=var_clamped)
    ct.store(invstd_ptr, index=(channel,), tile=invstd_1d)


@ct.kernel
def _bn_relu_epilogue_kernel(
    x_ptr,         # bf16 (channels, elements_per_channel_padded)
    mean_ptr,      # f32 (channels,)
    invstd_ptr,    # f32 (channels,)
    weight_ptr,    # f32 (channels,)
    bias_ptr,      # f32 (channels,)
    relu_out_ptr,  # bf16 (channels, elements_per_channel_padded)
    E: ct.Constant[int],
    BLOCK_E: ct.Constant[int],
):
    channel = ct.bid(0)
    chunk_idx = ct.bid(1)
    cols = chunk_idx * BLOCK_E + ct.arange(BLOCK_E, dtype=ct.int32)
    valid = cols < E

    x_bf = ct.load(
        x_ptr, index=(channel, chunk_idx),
        shape=(1, BLOCK_E),
        padding_mode=ct.PaddingMode.ZERO,
    )
    x_f = ct.astype(x_bf, ct.float32)
    mean = ct.load(mean_ptr, index=(channel,), shape=(1,))
    invstd = ct.load(invstd_ptr, index=(channel,), shape=(1,))
    weight = ct.load(weight_ptr, index=(channel,), shape=(1,))
    bias = ct.load(bias_ptr, index=(channel,), shape=(1,))
    mean_2d = ct.reshape(mean, (1, 1))
    invstd_2d = ct.reshape(invstd, (1, 1))
    weight_2d = ct.reshape(weight, (1, 1))
    bias_2d = ct.reshape(bias, (1, 1))

    y = (x_f - mean_2d) * invstd_2d
    y = y * weight_2d + bias_2d
    y_bf = ct.astype(y, ct.bfloat16)
    zero_bf = ct.zeros((1, BLOCK_E), dtype=ct.bfloat16)
    is_nan = y_bf != y_bf
    relu = ct.where(is_nan, y_bf, ct.where(y_bf > zero_bf, y_bf, zero_bf))
    valid_2d = ct.reshape(valid, (1, BLOCK_E))
    # Store: only meaningful for valid positions, but we always store — the
    # padded storage is scratch anyway. Zero padded positions to make the
    # narrow safe.
    out_tile = ct.where(valid_2d, relu, zero_bf)
    ct.store(relu_out_ptr, index=(channel, chunk_idx), tile=out_tile)


def _run(inputs):
    x0, x1, running_mean, running_var, weight, bias = inputs
    n = int(x0.shape[0])
    c0 = int(x0.shape[1])
    c1 = int(x1.shape[1])
    height = int(x0.shape[2])
    width = int(x0.shape[3])
    channels = c0 + c1
    hw = height * width
    elements_per_channel = n * hw
    device = x0.device

    # 1) concat -> (n, C, H, W) bf16 contiguous
    cat = torch.cat([x0, x1], dim=1).contiguous()
    # For the reduction and epilogue kernels we want (C, N*H*W) layout.
    cat_ce = cat.permute(1, 0, 2, 3).contiguous().view(channels, elements_per_channel)

    # Pick BLOCK_E and chunk count. Cap BLOCK_E at 4096 to keep tiles small.
    if elements_per_channel <= 256:
        block_e = 256
    elif elements_per_channel <= 1024:
        block_e = 1024
    elif elements_per_channel <= 4096:
        block_e = _next_pow2(elements_per_channel)
    else:
        block_e = 4096
    num_chunks = (elements_per_channel + block_e - 1) // block_e
    block_chunks = _next_pow2(num_chunks)
    padded_e = num_chunks * block_e

    # Pad cat_ce to (C, padded_e) so we can safely do (1, BLOCK_E) tile loads
    # at (channel, chunk_idx) for chunk_idx in [0, num_chunks).
    if padded_e != elements_per_channel:
        cat_padded = torch.zeros((channels, padded_e),
                                 device=device, dtype=torch.bfloat16)
        cat_padded[:, :elements_per_channel] = cat_ce
    else:
        cat_padded = cat_ce

    # Allocate outputs.
    mean_out = torch.empty((channels,), device=device, dtype=torch.float32)
    var_out = torch.empty((channels,), device=device, dtype=torch.float32)
    invstd = torch.empty_strided((channels,), (1,),
                                 device=device, dtype=torch.float32)
    ps = torch.empty((channels, num_chunks), device=device, dtype=torch.float32)
    psq = torch.empty((channels, num_chunks), device=device, dtype=torch.float32)

    # relu output must be (n, channels, H, W) bf16 (contiguous strides).
    # For the kernel we write to (channels, padded_e) then narrow+copy_ back.
    relu_padded = torch.empty((channels, padded_e),
                              device=device, dtype=torch.bfloat16)

    stream = torch.cuda.current_stream()

    # 2) partial stats kernel
    ct.launch(
        stream, (channels, 1, 1), _partial_stats_kernel,
        (cat_padded, ps, psq,
         elements_per_channel, num_chunks, block_e),
    )
    # 3) finalize kernel
    inv_e = 1.0 / float(elements_per_channel)
    ct.launch(
        stream, (channels, 1, 1), _finalize_stats_kernel,
        (ps, psq, mean_out, var_out, invstd,
         num_chunks, block_chunks, inv_e),
    )
    # 4) BN affine + ReLU
    ct.launch(
        stream, (channels, num_chunks, 1), _bn_relu_epilogue_kernel,
        (cat_padded, mean_out, invstd, weight, bias, relu_padded,
         elements_per_channel, block_e),
    )

    # Reshape relu back to (n, channels, H, W).
    out_shape = (n, channels, height, width)
    out_stride = (channels * hw, hw, width, 1)
    relu = torch.empty_strided(out_shape, out_stride,
                               device=device, dtype=torch.bfloat16)
    relu_ce = relu_padded.narrow(1, 0, elements_per_channel).view(
        channels, n, height, width).permute(1, 0, 2, 3).contiguous()
    relu.copy_(relu_ce)

    # 5) running stats mutation via torch (graph-capture safe).
    new_running_mean = running_mean * MOMENTUM_OLD + mean_out * MOMENTUM
    new_running_var = (
        running_var * MOMENTUM_OLD
        + var_out * RUNNING_VAR_CORRECTION * MOMENTUM
    )
    torch.ops.aten.copy_.default(running_mean, new_running_mean)
    torch.ops.aten.copy_.default(running_var, new_running_var)

    mean_saved = mean_out.view(1, channels, 1, 1)
    return cat, invstd, relu, mean_saved, running_mean, running_var


@oracle_impl(hardware="B200", point="baa3bff3")
@oracle_impl(hardware="B200", point="61221734")
@oracle_impl(hardware="B200", point="8cd7a902")
@oracle_impl(hardware="B200", point="730264ea")
@oracle_impl(hardware="B200", point="44272dd4")
@oracle_impl(hardware="B200", point="b5c4b927")
@oracle_impl(hardware="B200", point="2880e8b1")
@oracle_impl(hardware="B200", point="d8527cc2")
@oracle_impl(hardware="B200", point="8c03465a")
@oracle_impl(hardware="B200", point="c9d7e6de")
@oracle_impl(hardware="B200", point="64e8ebcd")
@oracle_impl(hardware="B200", point="51b1a42a")
@oracle_impl(hardware="B200", point="f5eed02d")
@oracle_impl(hardware="B200", point="f827b58f")
@oracle_impl(hardware="B200", point="6ae814af")
@oracle_impl(hardware="B200", point="a3f6dafb")
@oracle_impl(hardware="B200", point="c6af141d")
@oracle_impl(hardware="B200", point="1001e061")
@oracle_impl(hardware="B200", point="3d0ffd00")
@oracle_impl(hardware="B200", point="ad23f619")
@oracle_impl(hardware="B200", point="da960947")
@oracle_impl(hardware="B200", point="cca336d9")
@oracle_impl(hardware="B200", point="be40c2c6")
@oracle_impl(hardware="B200", point="d85763d2")
@oracle_impl(hardware="B200", point="5e0f066d")
@oracle_impl(hardware="B200", point="91486e42")
@oracle_impl(hardware="B200", point="308447fe")
def oracle_forward(inputs):
    return _run(inputs)
