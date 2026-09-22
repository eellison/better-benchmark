"""cuTile port of var_mean_522de516d7b7: MobileNetV3 training-BN + hardswish.

Three @ct.kernel bodies mirroring Triton's 3 kernels:
  1. _partial_stats_kernel: per (channel, chunk) partial sum + sum-of-squares.
  2. _finalize_stats_kernel: per-channel combine + running-stat update.
  3. _hardswish_kernel: BN affine + bf16-cast + hardswish activation.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


EPS = 1.0e-5
MOMENTUM = 0.1
RUNNING_VAR_CORRECTION = 1.0000398612827361


@ct.kernel
def _partial_stats_kernel(
    x_ptr,             # bf16 flat view of x (any strides)
    partial_sum_ptr,   # f32 [C, NUM_CHUNKS]
    partial_sq_ptr,    # f32 [C, NUM_CHUNKS]
    C: ct.Constant[int],
    HW: ct.Constant[int],
    HW_ELEMS: ct.Constant[int],   # N*H*W per channel
    C_STRIDE: ct.Constant[int],   # element-stride of channel dim
    N_STRIDE: ct.Constant[int],   # element-stride of batch dim
    H: ct.Constant[int],
    W: ct.Constant[int],
    H_STRIDE: ct.Constant[int],
    W_STRIDE: ct.Constant[int],
    NUM_CHUNKS: ct.Constant[int],
    CHUNK: ct.Constant[int],      # BLOCK_R
    BLOCK_R: ct.Constant[int],
):
    channel = ct.bid(0)
    chunk = ct.bid(1)
    r_off = ct.arange(BLOCK_R, dtype=ct.int32)
    base_off = chunk * CHUNK + r_off
    active = (r_off < ct.full((BLOCK_R,), CHUNK, dtype=ct.int32)) & (base_off < ct.full((BLOCK_R,), HW_ELEMS, dtype=ct.int32))

    # Decode (n, hw) -> (n, h, w)
    n = base_off // HW
    hw = base_off - n * HW
    h_idx = hw // W
    w_idx = hw - h_idx * W
    off = n * N_STRIDE + channel * C_STRIDE + h_idx * H_STRIDE + w_idx * W_STRIDE
    off_masked = ct.where(active, off, ct.zeros((BLOCK_R,), dtype=ct.int32))

    x_bf = ct.gather(x_ptr, (off_masked,))
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
    HW_ELEMS: ct.Constant[int],
    NUM_CHUNKS: ct.Constant[int],
    BLOCK_N: ct.Constant[int],   # power-of-2 >= NUM_CHUNKS
):
    channel = ct.bid(0)
    n_off = ct.arange(BLOCK_N, dtype=ct.int32)
    n_lim = ct.full((BLOCK_N,), NUM_CHUNKS, dtype=ct.int32)
    active = n_off < n_lim

    sums = ct.reshape(ct.load(partial_sum_ptr, index=(channel, 0), shape=(1, BLOCK_N),
                              padding_mode=ct.PaddingMode.ZERO), (BLOCK_N,))
    sqs = ct.reshape(ct.load(partial_sq_ptr, index=(channel, 0), shape=(1, BLOCK_N),
                              padding_mode=ct.PaddingMode.ZERO), (BLOCK_N,))
    zeros_f = ct.zeros((BLOCK_N,), dtype=ct.float32)
    ones_f = ct.full((BLOCK_N,), 1.0, dtype=ct.float32)
    valid_f = ct.where(active, ones_f, zeros_f)
    total_sum = ct.sum(sums * valid_f)
    total_sq = ct.sum(sqs * valid_f)
    inv_e = 1.0 / HW_ELEMS
    mean = total_sum * inv_e
    var = total_sq * inv_e - mean * mean
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
def _hardswish_kernel(
    x_ptr,          # bf16 flat
    mean_ptr,       # f32 [C]
    invstd_ptr,     # f32 [C]
    weight_ptr,     # f32 [C]
    bias_ptr,       # f32 [C]
    out_ptr,        # bf16 flat, same layout as x
    TOTAL: ct.Constant[int],   # total elements
    C: ct.Constant[int],
    HW: ct.Constant[int],
    IS_CHANNELS_LAST: ct.Constant[int],  # 1 if x is NHWC-strided, else 0
    BLOCK_E: ct.Constant[int],
):
    pid = ct.bid(0)
    off = pid * BLOCK_E + ct.arange(BLOCK_E, dtype=ct.int32)
    active = off < ct.full((BLOCK_E,), TOTAL, dtype=ct.int32)
    off_m = ct.where(active, off, ct.zeros((BLOCK_E,), dtype=ct.int32))

    # channel index: for channels-last (NHWC), off % C; for NCHW, (off // HW) % C.
    if IS_CHANNELS_LAST == 1:
        channel = off_m % C
    else:
        channel = (off_m // HW) % C

    x_bf = ct.gather(x_ptr, (off_m,))
    x_f = ct.astype(x_bf, ct.float32)
    mean = ct.gather(mean_ptr, (channel,))
    invstd = ct.gather(invstd_ptr, (channel,))
    weight = ct.gather(weight_ptr, (channel,))
    bias = ct.gather(bias_ptr, (channel,))

    norm = (x_f - mean) * invstd
    affine = norm * weight + bias
    rounded_bf = ct.astype(affine, ct.bfloat16)
    rounded = ct.astype(rounded_bf, ct.float32)

    zeros_f = ct.zeros((BLOCK_E,), dtype=ct.float32)
    six_f = ct.full((BLOCK_E,), 6.0, dtype=ct.float32)
    three_f = ct.full((BLOCK_E,), 3.0, dtype=ct.float32)
    relu6 = rounded + three_f
    relu6 = ct.where(relu6 < zeros_f, zeros_f, relu6)
    relu6 = ct.where(relu6 > six_f, six_f, relu6)
    hswish = rounded * relu6 * (1.0 / 6.0)
    hswish_bf = ct.astype(hswish, ct.bfloat16)
    zero_bf = ct.full((BLOCK_E,), 0.0, dtype=ct.bfloat16)
    ct.scatter(out_ptr, (off_m,), ct.where(active, hswish_bf, zero_bf), mask=active)


def _next_pow2(v: int) -> int:
    return 1 << (int(v) - 1).bit_length()


@oracle_impl(hardware="B200", point="d6a317bc", BLOCK_E=1024)
@oracle_impl(hardware="B200", point="055f7aad", BLOCK_E=1024)
@oracle_impl(hardware="B200", point="bb4cfa64", BLOCK_E=1024)
@oracle_impl(hardware="B200", point="c0a7cc9a", BLOCK_E=1024)
@oracle_impl(hardware="B200", point="03d2a306", BLOCK_E=1024)
@oracle_impl(hardware="B200", point="df1d49cf", BLOCK_E=1024)
@oracle_impl(hardware="B200", point="66871354", BLOCK_E=1024)
@oracle_impl(hardware="B200", point="a3c80480", BLOCK_E=1024)
@oracle_impl(hardware="B200", point="e8a9d13a", BLOCK_E=1024)
@oracle_impl(hardware="B200", point="8aafc432", BLOCK_E=1024)
@oracle_impl(hardware="B200", point="ba044ce1", BLOCK_E=1024)
@oracle_impl(hardware="B200", point="531cd9bd", BLOCK_E=1024)
@oracle_impl(hardware="B200", point="76491bca", BLOCK_E=1024)
@oracle_impl(hardware="B200", point="4e453d00", BLOCK_E=1024)
@oracle_impl(hardware="B200", point="85aaacf2", BLOCK_E=1024)
@oracle_impl(hardware="B200", point="5e2cd32b", BLOCK_E=1024)
def oracle_forward(inputs, *, BLOCK_E):
    x, running_mean, running_var, weight, bias = inputs
    device = x.device
    n = int(x.shape[0])
    c = int(x.shape[1])
    h = int(x.shape[2])
    w = int(x.shape[3])
    hw = h * w
    hw_elems = n * hw
    total = n * c * hw

    is_channels_last = int(x.stride(1) == 1)

    # Kernel indexes flat memory with strides. We pass strides as-is.
    n_stride = int(x.stride(0))
    c_stride = int(x.stride(1))
    h_stride = int(x.stride(2))
    w_stride = int(x.stride(3))

    # Use the raw tensor as 1D flat (contiguous storage view via .as_strided or -1).
    # For strided tensors, we compute offsets in the kernel using strides.
    x_1d = torch.as_strided(x, (x.numel(),), (1,))  # flat 1D view of storage

    out = torch.empty_strided(
        tuple(x.shape), tuple(x.stride()),
        device=device, dtype=torch.bfloat16,
    )
    out_1d = torch.as_strided(out, (out.numel(),), (1,))

    mean = torch.empty((c,), device=device, dtype=torch.float32)
    invstd = torch.empty((c,), device=device, dtype=torch.float32)

    # Chunked stats.
    CHUNK = 4096
    BLOCK_R = 4096
    num_chunks = (hw_elems + CHUNK - 1) // CHUNK
    BLOCK_N = _next_pow2(num_chunks)
    partial_sum = torch.empty((c, num_chunks), device=device, dtype=torch.float32)
    partial_sq = torch.empty((c, num_chunks), device=device, dtype=torch.float32)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (c, num_chunks, 1), _partial_stats_kernel,
        (x_1d, partial_sum, partial_sq,
         c, hw, hw_elems, c_stride, n_stride, h, w, h_stride, w_stride,
         num_chunks, CHUNK, BLOCK_R),
    )
    ct.launch(
        stream, (c, 1, 1), _finalize_stats_kernel,
        (partial_sum, partial_sq, running_mean, running_var, mean, invstd,
         hw_elems, num_chunks, BLOCK_N),
    )
    # Activation over flat storage using strides.
    # For hardswish kernel, we iterate over LOGICAL element index in NCHW or NHWC order.
    # Do this by using the contiguous / channels-last flat mapping.
    # Since the tensor is either NCHW-contiguous or channels-last-contiguous, iterating
    # over 0..total in flat storage order matches iterating over logical elements in
    # some order. We just need the channel index for each flat position.
    if is_channels_last:
        # Storage is NHWC: order n*h*w*c + h*w*c + w*c + c. flat_off % c = channel.
        ct.launch(
            stream, (ct.cdiv(total, BLOCK_E), 1, 1), _hardswish_kernel,
            (x_1d, mean, invstd, weight, bias, out_1d,
             total, c, hw, 1, BLOCK_E),
        )
    else:
        # Storage is NCHW: order n*c*h*w. flat_off // hw % c = channel.
        ct.launch(
            stream, (ct.cdiv(total, BLOCK_E), 1, 1), _hardswish_kernel,
            (x_1d, mean, invstd, weight, bias, out_1d,
             total, c, hw, 0, BLOCK_E),
        )

    mean_r = mean.view(1, c, 1, 1)
    invstd_r = invstd.view(1, c, 1, 1)
    return mean_r, invstd_r, out, running_mean, running_var
