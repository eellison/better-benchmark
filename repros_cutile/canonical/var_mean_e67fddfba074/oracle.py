"""cuTile port of var_mean_e67fddfba074: ShuffleNet training BN + channel
shuffle. Non-pow2 channel (116). Two kernels:
  1) BN stats: per-channel reduce (K = N*H*W = 25088) → mean, invstd.
     Also mutates running_mean/running_var in place.
  2) Shuffle output: apply BN affine + ReLU + interleave with skip branch
     into shuffled output.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


N = 128
C = 116
H = 14
W = 14
HW = H * W          # 196
K = N * HW          # 25088
OUT_C = C * 2       # 232
EPS = 1.0e-5
MOMENTUM = 0.1
OLD_WEIGHT = 0.9
RUNNING_VAR_CORRECTION = 1.0000398612827361


@ct.kernel
def _bn_stats_kernel(
    x_ptr,             # bf16 [N, C, H, W] flat -> access strided per channel
    running_mean_ptr,  # f32 [C]
    running_var_ptr,   # f32 [C]
    mean_ptr,          # f32 [C]
    invstd_ptr,        # f32 [C]
    channels: ct.Constant[int],
    hw_size: ct.Constant[int],
    k_size: ct.Constant[int],
    eps_: ct.Constant[float],
    momentum_: ct.Constant[float],
    old_weight_: ct.Constant[float],
    running_var_correction_: ct.Constant[float],
    BLOCK: ct.Constant[int],
):
    channel = ct.bid(0)
    offsets = ct.arange(BLOCK, dtype=ct.int32)
    active = offsets < k_size
    n_idx = offsets // hw_size
    hw_idx = offsets - n_idx * hw_size
    x_offsets = n_idx * (channels * hw_size) + channel * hw_size + hw_idx

    x = ct.astype(ct.gather(x_ptr, (x_offsets,), mask=active, padding_value=ct.bfloat16(0.0)), ct.float32)
    zero_f = ct.zeros((BLOCK,), dtype=ct.float32)
    x_masked = ct.where(active, x, zero_f)
    mean = ct.sum(x_masked) / float(k_size)
    centered = ct.where(active, x - mean, zero_f)
    var_val = ct.sum(centered * centered) / float(k_size)
    var_val = ct.where(var_val > 0.0, var_val, 0.0)
    invstd = ct.rsqrt(var_val + eps_)

    ct.store(mean_ptr, index=(channel,), tile=ct.reshape(ct.full((1,), 0.0, dtype=ct.float32) + mean, (1,)))
    ct.store(invstd_ptr, index=(channel,), tile=ct.reshape(ct.full((1,), 0.0, dtype=ct.float32) + invstd, (1,)))

    old_mean = ct.astype(ct.load(running_mean_ptr, index=(channel,), shape=(1,)), ct.float32)
    old_var = ct.astype(ct.load(running_var_ptr, index=(channel,), shape=(1,)), ct.float32)
    new_running_mean = old_mean * old_weight_ + mean * momentum_
    new_running_var = old_var * old_weight_ + var_val * running_var_correction_ * momentum_
    ct.store(running_mean_ptr, index=(channel,), tile=ct.reshape(ct.full((1,), 0.0, dtype=ct.float32) + new_running_mean, (1,)))
    ct.store(running_var_ptr, index=(channel,), tile=ct.reshape(ct.full((1,), 0.0, dtype=ct.float32) + new_running_var, (1,)))


@ct.kernel
def _shuffle_output_kernel(
    x_ptr,           # bf16 [N * C * HW] flat
    weight_ptr,      # f32 [C]
    bias_ptr,        # f32 [C]
    skip_ptr,        # bf16 strided [N, C, H, W]
    mean_ptr,        # f32 [C]
    invstd_ptr,      # f32 [C]
    out_ptr,         # bf16 [N * OUT_C * HW]  interleaved output
    total: ct.Constant[int],
    channels: ct.Constant[int],
    out_channels: ct.Constant[int],
    hw_size: ct.Constant[int],
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    offsets = pid * BLOCK + ct.arange(BLOCK, dtype=ct.int32)
    mask = offsets < total
    hw_idx = offsets - (offsets // hw_size) * hw_size
    ch_and_hw = offsets // hw_size
    channel = ch_and_hw - (ch_and_hw // channels) * channels
    n_idx = offsets // (channels * hw_size)

    x = ct.astype(ct.gather(x_ptr, (offsets,), mask=mask, padding_value=ct.bfloat16(0.0)), ct.float32)
    mean = ct.gather(mean_ptr, (channel,), mask=mask, padding_value=ct.float32(0.0))
    invstd = ct.gather(invstd_ptr, (channel,), mask=mask, padding_value=ct.float32(0.0))
    weight = ct.gather(weight_ptr, (channel,), mask=mask, padding_value=ct.float32(0.0))
    bias = ct.gather(bias_ptr, (channel,), mask=mask, padding_value=ct.float32(0.0))

    y = (x - mean) * invstd * weight + bias
    y_rounded = ct.astype(ct.astype(y, ct.bfloat16), ct.float32)
    # ReLU (NaN-preserving)
    zero_f = ct.zeros((BLOCK,), dtype=ct.float32)
    y_relu_f = ct.where(y_rounded != y_rounded, y_rounded, ct.where(y_rounded > 0.0, y_rounded, zero_f))
    y_relu_bf = ct.astype(y_relu_f, ct.bfloat16)

    skip_offsets = n_idx * (out_channels * hw_size) + channel * hw_size + hw_idx
    skip = ct.gather(skip_ptr, (skip_offsets,), mask=mask, padding_value=ct.bfloat16(0.0))

    out_base = n_idx * (out_channels * hw_size) + channel * 2 * hw_size + hw_idx
    ct.scatter(out_ptr, (out_base,), skip, mask=mask)
    ct.scatter(out_ptr, (out_base + hw_size,), y_relu_bf, mask=mask)


@oracle_impl(hardware="B200", point="00e34431", BLOCK_K=32768, BLOCK_OUT=256)
def oracle_forward(inputs, *, BLOCK_K: int, BLOCK_OUT: int):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, _shape_param_0, _shape_param_1 = inputs

    mean = torch.empty_strided((1, C, 1, 1), (C, 1, 1, 1), device=arg0_1.device, dtype=torch.float32)
    invstd = torch.empty_strided((1, C, 1, 1), (C, 1, 1, 1), device=arg0_1.device, dtype=torch.float32)
    shuffled = torch.empty_strided((N, OUT_C, H, W), (OUT_C * HW, HW, W, 1),
                                    device=arg0_1.device, dtype=torch.bfloat16)

    # x is contiguous [N, C, H, W] with default strides.
    x_flat = arg0_1.reshape(-1)

    # skip (arg5_1) is strided [N, C, H, W] with stride (45472, 196, 14, 1).
    # 45472 = 232 * 196 = OUT_C * HW. So it's really a slice of an OUT_C tensor.
    # We can access it via gather: skip_offsets = n * OUT_C * HW + c * HW + hw. But the
    # actual storage has n_stride=45472, c_stride=196, so a flat view respecting the
    # original OUT_C-strided storage requires we look at the underlying storage.
    # arg5_1 has offset 0 into that storage (per the strides). The underlying storage
    # can be reached as arg5_1.storage() but simpler: allocate a full [N, OUT_C, H, W]
    # temp and copy skip into it. But that changes semantics. Actually just gather the
    # right elements with the correct arithmetic based on N-stride.
    # Since c_stride=196=HW, the c dim is contiguous per channel. n_stride=45472. So
    # from position (n, c, hw), the flat offset in the underlying storage is
    # n*45472 + c*196 + hw. This is exactly what we want if we treat skip as a 1D
    # tensor via .as_strided(size=(N*OUT_C*HW,), stride=(1,)).
    # But we can compute flat offset in kernel: skip_offsets = n * OUT_C * HW + c * HW + hw
    # which matches n*45472 + c*196 + hw. So a 1D "logical" pointer that begins at the same
    # storage byte as arg5_1 will work IF the tensor covers those bytes.
    # arg5_1 is [N, C, H, W] with strides (45472, 196, 14, 1). Its element count is
    # N * OUT_C * HW when we treat it as OUT_C sized, but we only have C sized. The
    # underlying storage may not span N*OUT_C*HW elements.
    # Safer approach: use arg5_1's underlying storage via as_strided.
    storage_numel = arg5_1.untyped_storage().nbytes() // arg5_1.element_size()
    skip_flat = arg5_1.as_strided((storage_numel,), (1,))

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (C, 1, 1),
        _bn_stats_kernel,
        (x_flat, arg1_1, arg2_1, mean.view(-1), invstd.view(-1),
         C, HW, K, EPS, MOMENTUM, OLD_WEIGHT, RUNNING_VAR_CORRECTION, BLOCK_K),
    )
    total = N * C * HW
    ct.launch(
        stream,
        ((total + BLOCK_OUT - 1) // BLOCK_OUT, 1, 1),
        _shuffle_output_kernel,
        (x_flat, arg3_1, arg4_1, skip_flat, mean.view(-1), invstd.view(-1),
         shuffled.view(-1), total, C, OUT_C, HW, BLOCK_OUT),
    )

    return (
        mean,
        invstd,
        shuffled[:, :C, :, :],
        shuffled[:, C:, :, :],
        arg1_1,
        arg2_1,
    )
