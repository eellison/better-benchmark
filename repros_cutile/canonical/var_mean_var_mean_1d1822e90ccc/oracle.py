"""cuTile port of var_mean_var_mean_1d1822e90ccc: Inception dual BN-train + pools.

Uses cuTile for the two per-channel var_mean reductions; BN affine + ReLU,
maxpool-with-offsets, cat, avg_pool2d, and running-stat updates run in torch.

For N_ELEMS large enough that a single-tile pow2 blows shared memory
(point 505c1a16: N_ELEMS = 36992, next_pow2 = 65536), we use a chunked
partial-reduction + finalize kernel pair.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


EPS = 0.001
VAR_CORRECTION = 1.0001220852154804
MOMENTUM_NEW = 0.1
MOMENTUM_OLD = 0.9


def _next_pow2(v):
    r = 1
    while r < v:
        r <<= 1
    return r


@ct.kernel
def _channel_partial_stats_kernel(
    x_ptr,          # bf16 [C, N_ELEMS_PAD]  (zero-padded to power of 2)
    psum_ptr,       # f32  [num_chunks, C]
    psq_ptr,        # f32  [num_chunks, C]
    N_ELEMS: ct.Constant[int],
    BLOCK_R: ct.Constant[int],
):
    channel = ct.bid(0)
    chunk = ct.bid(1)
    x = ct.load(
        x_ptr, index=(channel, chunk), shape=(1, BLOCK_R),
        padding_mode=ct.PaddingMode.ZERO,
    )
    x_f = ct.astype(x, ct.float32)

    e_arange = ct.arange(BLOCK_R, dtype=ct.int32)
    e_global = chunk * BLOCK_R + e_arange
    e_valid = e_global < N_ELEMS
    e_valid_2d = ct.reshape(e_valid, (1, BLOCK_R))
    zero = ct.full((1, BLOCK_R), 0.0, dtype=ct.float32)
    x_masked = ct.where(e_valid_2d, x_f, zero)

    psum = ct.sum(x_masked)
    psq = ct.sum(x_masked * x_masked)
    ct.store(psum_ptr, index=(chunk, channel), tile=ct.reshape(psum, (1, 1)))
    ct.store(psq_ptr, index=(chunk, channel), tile=ct.reshape(psq, (1, 1)))


@ct.kernel
def _channel_finalize_stats_kernel(
    psum_ptr,       # f32 [num_chunks, C]
    psq_ptr,        # f32 [num_chunks, C]
    mean_ptr,       # f32 [C]
    var_ptr,        # f32 [C]
    NUM_CHUNKS: ct.Constant[int],
    BLOCK_CHUNKS: ct.Constant[int],
    INV_N: ct.Constant[float],
):
    channel = ct.bid(0)
    psum = ct.load(psum_ptr, index=(0, channel), shape=(BLOCK_CHUNKS, 1),
                   padding_mode=ct.PaddingMode.ZERO)
    psq = ct.load(psq_ptr, index=(0, channel), shape=(BLOCK_CHUNKS, 1),
                  padding_mode=ct.PaddingMode.ZERO)
    chunks = ct.arange(BLOCK_CHUNKS, dtype=ct.int32)
    chunk_mask = chunks < NUM_CHUNKS
    chunk_mask_2d = ct.reshape(chunk_mask, (BLOCK_CHUNKS, 1))
    zero = ct.full((BLOCK_CHUNKS, 1), 0.0, dtype=ct.float32)
    total = ct.sum(ct.where(chunk_mask_2d, psum, zero))
    total_sq = ct.sum(ct.where(chunk_mask_2d, psq, zero))
    mean = total * INV_N
    var = total_sq * INV_N - mean * mean
    ct.store(mean_ptr, index=(channel,), tile=ct.reshape(mean, (1,)))
    ct.store(var_ptr, index=(channel,), tile=ct.reshape(var, (1,)))


def _bn_branch_var_mean(x, device):
    """Compute per-channel mean and var using cuTile. x is [N, C, H, W]."""
    n = int(x.shape[0])
    c = int(x.shape[1])
    h = int(x.shape[2])
    w = int(x.shape[3])
    n_elems = n * h * w

    # Chunk size: pick BLOCK_R = 1024 (works for all shapes); num_chunks = cdiv.
    BLOCK_R = 1024
    num_chunks = (n_elems + BLOCK_R - 1) // BLOCK_R
    block_chunks = _next_pow2(num_chunks)

    # Permute channels-last to [C, N*H*W] contiguous.
    x_2d = x.permute(1, 0, 2, 3).contiguous().reshape(c, n_elems)

    psum = torch.empty((num_chunks, c), device=device, dtype=torch.float32)
    psq = torch.empty((num_chunks, c), device=device, dtype=torch.float32)
    mean_1d = torch.empty((c,), device=device, dtype=torch.float32)
    var_1d = torch.empty((c,), device=device, dtype=torch.float32)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (c, num_chunks, 1),
        _channel_partial_stats_kernel,
        (x_2d, psum, psq, n_elems, BLOCK_R),
    )
    ct.launch(
        stream, (c, 1, 1),
        _channel_finalize_stats_kernel,
        (psum, psq, mean_1d, var_1d, num_chunks, block_chunks,
         1.0 / float(n_elems)),
    )
    var_1d = torch.clamp_min(var_1d, 0.0)
    return mean_1d, var_1d


def _bn_apply(x, mean_1d, var_1d, weight, bias, running_mean, running_var, c):
    mean = mean_1d.view(1, c, 1, 1)
    var = var_1d.view(1, c, 1, 1)
    rsqrt = torch.rsqrt(var + EPS)
    normalized = (x.to(torch.float32) - mean) * rsqrt
    affine = normalized * weight.view(1, c, 1, 1) + bias.view(1, c, 1, 1)
    relu = torch.relu(affine.to(torch.bfloat16))
    # Running-stat update
    new_running_mean = mean_1d * MOMENTUM_NEW + running_mean * MOMENTUM_OLD
    new_running_var = (var_1d * VAR_CORRECTION) * MOMENTUM_NEW + running_var * MOMENTUM_OLD
    running_mean.copy_(new_running_mean)
    running_var.copy_(new_running_var)
    # Return as [1, C, 1, 1] shaped f32 mean and rsqrt (matching Repro getitem_1 and rsqrt)
    return mean, rsqrt, relu


@oracle_impl(hardware="B200", point="5b79c9f7")
@oracle_impl(hardware="B200", point="505c1a16")
def oracle_forward(inputs):
    (arg0, arg1, arg2, arg3, arg4,
     arg5, arg6, arg7, arg8, arg9,
     arg10, kernel_size, stride) = inputs
    device = arg0.device
    c0 = int(arg0.shape[1])
    c1 = int(arg5.shape[1])

    mean0_1d, var0_1d = _bn_branch_var_mean(arg0, device)
    mean1_1d, var1_1d = _bn_branch_var_mean(arg5, device)

    getitem_1, rsqrt0, relu0 = _bn_apply(
        arg0, mean0_1d, var0_1d, arg3, arg4, arg1, arg2, c0)
    getitem_3, rsqrt1, relu1 = _bn_apply(
        arg5, mean1_1d, var1_1d, arg8, arg9, arg6, arg7, c1)

    getitem_4, getitem_5 = torch.ops.prims._low_memory_max_pool_with_offsets.default(
        arg10, list(kernel_size), list(stride), [0, 0], [1, 1], False)

    cat = torch.cat([relu0, relu1, getitem_4], dim=1)
    avg_pool2d = torch.nn.functional.avg_pool2d(cat, [3, 3], [1, 1], [1, 1])

    return (getitem_1, rsqrt0, getitem_3, rsqrt1, getitem_5, cat, avg_pool2d,
            arg1, arg2, arg6, arg7)
