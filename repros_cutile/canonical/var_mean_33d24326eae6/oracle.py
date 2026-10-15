"""cuTile port of var_mean_33d24326eae6: ShuffleNet BN training + channel-shuffle.

Per-channel population var_mean over (N, H, W) for a (N, C, H, W) bf16 input:
  - Compute mean, variance (correction=0), rsqrt(var + 1e-5).
  - Update running_mean = mean * 0.1 + running_mean * 0.9.
  - Update running_var = variance * 1.0001594642002871 * 0.1 + running_var * 0.9.
  - Apply affine (weight, bias) + bf16 cast + ReLU (preserving NaN).
  - Concatenate ReLU output with skip tensor (strided view) into a (N, 2C, H, W)
    channel-shuffled contiguous output where channel c_in maps to
    output channel 2*c_in for skip and 2*c_in + 1 for BN+ReLU.

Three cuTile kernels: partial stats, finalize (mean+invstd), apply-BN-ReLU-store.
Uses `.permute(1, 0, 2, 3).contiguous()` to lay the data channel-major so each
kernel program handles one channel's contiguous elements.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


EPS = 1.0e-5
MOMENTUM = 0.1
RUNNING_VAR_CORRECTION = 1.0001594642002871


@ct.kernel
def _bn_partial_stats_kernel(
    x_ptr,             # bf16 (C, N*HW)
    partial_sum_ptr,   # f32 (num_chunks, C)
    partial_sumsq_ptr, # f32 (num_chunks, C)
    EPC_: ct.Constant[int],
    BLOCK_E_: ct.Constant[int],
):
    c = ct.bid(0)
    chunk = ct.bid(1)
    x_bf = ct.load(
        x_ptr, index=(c, chunk), shape=(1, BLOCK_E_),
        padding_mode=ct.PaddingMode.ZERO,
    )
    x_f = ct.astype(x_bf, ct.float32)
    e_idx = chunk * BLOCK_E_ + ct.arange(BLOCK_E_, dtype=ct.int32)
    valid = ct.reshape(e_idx < EPC_, (1, BLOCK_E_))
    zero_f = ct.zeros((1, BLOCK_E_), dtype=ct.float32)
    x_masked = ct.where(valid, x_f, zero_f)
    s = ct.sum(x_masked)
    ss = ct.sum(x_masked * x_masked)
    ct.store(partial_sum_ptr, index=(chunk, c), tile=ct.reshape(s, (1, 1)))
    ct.store(partial_sumsq_ptr, index=(chunk, c), tile=ct.reshape(ss, (1, 1)))


@ct.kernel
def _bn_finalize_stats_kernel(
    partial_sum_ptr,    # f32 (num_chunks, C)
    partial_sumsq_ptr,  # f32 (num_chunks, C)
    mean_ptr,           # f32 (C,)
    variance_ptr,       # f32 (C,)
    invstd_ptr,         # f32 (C,)
    EPC_: ct.Constant[int],
    NUM_CHUNKS_: ct.Constant[int],
    BLOCK_CHUNKS_: ct.Constant[int],
):
    c = ct.bid(0)
    ch_idx = ct.arange(BLOCK_CHUNKS_, dtype=ct.int32)
    valid_chunk = ct.reshape(ch_idx < NUM_CHUNKS_, (BLOCK_CHUNKS_, 1))
    zero_f = ct.zeros((BLOCK_CHUNKS_, 1), dtype=ct.float32)
    s = ct.load(
        partial_sum_ptr, index=(0, c), shape=(BLOCK_CHUNKS_, 1),
        padding_mode=ct.PaddingMode.ZERO,
    )
    ss = ct.load(
        partial_sumsq_ptr, index=(0, c), shape=(BLOCK_CHUNKS_, 1),
        padding_mode=ct.PaddingMode.ZERO,
    )
    s_masked = ct.where(valid_chunk, s, zero_f)
    ss_masked = ct.where(valid_chunk, ss, zero_f)
    total = ct.sum(s_masked)
    total_sq = ct.sum(ss_masked)
    mean = total * (1.0 / EPC_)
    variance = total_sq * (1.0 / EPC_) - mean * mean
    var_pos = ct.where(variance > 0.0, variance, 0.0)
    invstd = ct.rsqrt(var_pos + EPS)
    ct.store(mean_ptr, index=(c,), tile=ct.reshape(mean, (1,)))
    ct.store(variance_ptr, index=(c,), tile=ct.reshape(var_pos, (1,)))
    ct.store(invstd_ptr, index=(c,), tile=ct.reshape(invstd, (1,)))


@ct.kernel
def _bn_apply_kernel(
    x_ptr,        # bf16 (C, N*HW)
    weight_ptr,   # f32 (C,)
    bias_ptr,     # f32 (C,)
    mean_ptr,     # f32 (C,)
    invstd_ptr,   # f32 (C,)
    out_ptr,      # bf16 (C, N*HW)
    EPC_: ct.Constant[int],
    BLOCK_E_: ct.Constant[int],
):
    c = ct.bid(0)
    chunk = ct.bid(1)
    x_bf = ct.load(
        x_ptr, index=(c, chunk), shape=(1, BLOCK_E_),
        padding_mode=ct.PaddingMode.ZERO,
    )
    mean_t = ct.load(mean_ptr, index=(c,), shape=(1,))
    invstd_t = ct.load(invstd_ptr, index=(c,), shape=(1,))
    weight_t = ct.load(weight_ptr, index=(c,), shape=(1,))
    bias_t = ct.load(bias_ptr, index=(c,), shape=(1,))

    x_f = ct.astype(x_bf, ct.float32)
    mean_v = ct.reshape(mean_t, (1, 1))
    invstd_v = ct.reshape(invstd_t, (1, 1))
    weight_v = ct.reshape(weight_t, (1, 1))
    bias_v = ct.reshape(bias_t, (1, 1))

    centered = x_f - mean_v
    normalized = centered * invstd_v
    affine = normalized * weight_v + bias_v
    affine_bf = ct.astype(affine, ct.bfloat16)
    # ReLU preserving NaN: compare in f32 to catch NaNs
    affine_f = ct.astype(affine_bf, ct.float32)
    zero_bf = ct.zeros((1, BLOCK_E_), dtype=ct.bfloat16)
    keep = (affine_f > 0.0) | (affine_f != affine_f)
    relu_bf = ct.where(keep, affine_bf, zero_bf)
    # Mask out padded positions
    e_idx = chunk * BLOCK_E_ + ct.arange(BLOCK_E_, dtype=ct.int32)
    valid = ct.reshape(e_idx < EPC_, (1, BLOCK_E_))
    relu_masked = ct.where(valid, relu_bf, zero_bf)
    ct.store(out_ptr, index=(c, chunk), tile=relu_masked)


def _next_power_of_2(value):
    return 1 << (int(value) - 1).bit_length()


@oracle_impl(hardware="B200", point="b0183d97", BLOCK_E=1024)
@oracle_impl(hardware="B200", point="00e34431", BLOCK_E=1024)
@oracle_impl(hardware="B200", point="99b3b05e", BLOCK_E=1024)
def oracle_forward(inputs, *, BLOCK_E: int):
    (arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1,
     _shape_param_0, _shape_param_1) = inputs
    device = arg0_1.device

    N = int(arg0_1.shape[0])
    C = int(arg0_1.shape[1])
    H = int(arg0_1.shape[2])
    W = int(arg0_1.shape[3])
    hw = H * W
    epc = N * hw  # elements per channel
    out_C = 2 * C

    # Permute input to channel-major and make contiguous.
    x_perm = arg0_1.permute(1, 0, 2, 3).contiguous().view(C, epc)

    num_chunks = (epc + BLOCK_E - 1) // BLOCK_E
    block_chunks = _next_power_of_2(num_chunks)

    partial_sum = torch.empty((num_chunks, C), device=device, dtype=torch.float32)
    partial_sumsq = torch.empty((num_chunks, C), device=device, dtype=torch.float32)
    mean_1d = torch.empty((C,), device=device, dtype=torch.float32)
    variance_1d = torch.empty((C,), device=device, dtype=torch.float32)
    invstd_1d = torch.empty((C,), device=device, dtype=torch.float32)
    bn_relu_perm = torch.empty((C, epc), device=device, dtype=torch.bfloat16)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (C, num_chunks, 1), _bn_partial_stats_kernel,
        (x_perm, partial_sum, partial_sumsq, epc, BLOCK_E),
    )
    ct.launch(
        stream, (C, 1, 1), _bn_finalize_stats_kernel,
        (partial_sum, partial_sumsq, mean_1d, variance_1d, invstd_1d,
         epc, num_chunks, block_chunks),
    )
    ct.launch(
        stream, (C, num_chunks, 1), _bn_apply_kernel,
        (x_perm, arg3_1, arg4_1, mean_1d, invstd_1d, bn_relu_perm,
         epc, BLOCK_E),
    )

    # Reshape BN-relu output back to (N, C, H, W).
    bn_relu = bn_relu_perm.view(C, N, H, W).permute(1, 0, 2, 3).contiguous()

    # Update running stats in place via copy_.
    new_running_mean = mean_1d * MOMENTUM + arg1_1 * (1.0 - MOMENTUM)
    new_running_var = (variance_1d * RUNNING_VAR_CORRECTION * MOMENTUM
                       + arg2_1 * (1.0 - MOMENTUM))
    arg1_1.copy_(new_running_mean)
    arg2_1.copy_(new_running_var)

    # Build shuffled output (N, 2C, H, W) contiguous.
    shuffled = torch.empty_strided(
        (N, out_C, H, W),
        (out_C * hw, hw, W, 1),
        device=device, dtype=torch.bfloat16,
    )
    # View (N, C, 2, H, W): slot 0 = skip, slot 1 = BN+ReLU.
    shuf_view = shuffled.view(N, C, 2, H, W)
    shuf_view[:, :, 0, :, :].copy_(arg5_1)
    shuf_view[:, :, 1, :, :].copy_(bn_relu)

    # Return outputs: (mean, rsqrt/invstd, shuffled, running_mean, running_var).
    mean_out = mean_1d.view(1, C, 1, 1).clone()
    rsqrt_out = invstd_1d.view(1, C, 1, 1).clone()
    return mean_out, rsqrt_out, shuffled, arg1_1, arg2_1
