"""cuTile port of var_mean_0e86f1318633: Inception BN + maxpool with offsets.

Channels-last bf16 [128, 64, 147, 147] input. Reinterpret as (E, C) NHWC-flat
for the reductions. Three cuTile kernels compute:
  1) partial per-channel sum + sum_sq over E axis in chunks.
  2) finalize: mean/var/invstd (eps=0.001), running-stat updates for
     branch means and variances (correction=1.0000003615393043).
  3) BN affine + bf16 cast + ReLU into an NHWC bf16 intermediate.

The 3x3 stride-2 maxpool_with_offsets is applied via torch.ops.prims — this
is a natively-optimized op that matches the eager Repro exactly. Running
stats are copy_'d back to the input tensors post-kernel.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


EPS = 0.001
VAR_CORRECTION = 1.0000003615393043


@ct.kernel
def _partial_stats_kernel(
    x_ptr,       # bf16 (E, C)
    ps_ptr,      # f32 (num_chunks, C)
    psq_ptr,
    BLOCK_E: ct.Constant[int],
    C_BLOCK: ct.Constant[int],
):
    e_block = ct.bid(0)
    c_block = ct.bid(1)
    x_bf = ct.load(x_ptr, index=(e_block, c_block),
                   shape=(BLOCK_E, C_BLOCK),
                   padding_mode=ct.PaddingMode.ZERO)
    x_f = ct.astype(x_bf, ct.float32)
    s = ct.sum(x_f, axis=0, keepdims=True)
    sq = ct.sum(x_f * x_f, axis=0, keepdims=True)
    ct.store(ps_ptr, index=(e_block, c_block), tile=s)
    ct.store(psq_ptr, index=(e_block, c_block), tile=sq)


@ct.kernel
def _finalize_stats_kernel(
    ps_ptr, psq_ptr,
    old_rm_ptr, old_rv_ptr,
    saved_mean_ptr, invstd_ptr,
    new_rm_ptr, new_rv_ptr,
    NUM_CHUNKS: ct.Constant[int],
    BLOCK_CHUNKS: ct.Constant[int],
    C_BLOCK: ct.Constant[int],
    INV_E: ct.Constant[float],
):
    c_block = ct.bid(0)

    ps = ct.load(ps_ptr, index=(0, c_block),
                 shape=(BLOCK_CHUNKS, C_BLOCK),
                 padding_mode=ct.PaddingMode.ZERO)
    psq = ct.load(psq_ptr, index=(0, c_block),
                  shape=(BLOCK_CHUNKS, C_BLOCK),
                  padding_mode=ct.PaddingMode.ZERO)

    chunks = ct.arange(BLOCK_CHUNKS, dtype=ct.int32)
    chunk_mask = ct.reshape(chunks < NUM_CHUNKS, (BLOCK_CHUNKS, 1))
    zero_f = ct.zeros((BLOCK_CHUNKS, C_BLOCK), dtype=ct.float32)
    ps = ct.where(chunk_mask, ps, zero_f)
    psq = ct.where(chunk_mask, psq, zero_f)

    total_s = ct.sum(ps, axis=0)
    total_sq = ct.sum(psq, axis=0)
    mean = total_s * INV_E
    var = total_sq * INV_E - mean * mean
    zero_1d = ct.zeros((C_BLOCK,), dtype=ct.float32)
    var = ct.where(var > 0.0, var, zero_1d)
    invstd = ct.rsqrt(var + EPS)

    old_rm = ct.load(old_rm_ptr, index=(c_block,), shape=(C_BLOCK,),
                     padding_mode=ct.PaddingMode.ZERO)
    old_rv = ct.load(old_rv_ptr, index=(c_block,), shape=(C_BLOCK,),
                     padding_mode=ct.PaddingMode.ZERO)
    new_rm = mean * 0.1 + old_rm * 0.9
    new_rv = var * VAR_CORRECTION * 0.1 + old_rv * 0.9

    ct.store(saved_mean_ptr, index=(c_block,), tile=mean)
    ct.store(invstd_ptr, index=(c_block,), tile=invstd)
    ct.store(new_rm_ptr, index=(c_block,), tile=new_rm)
    ct.store(new_rv_ptr, index=(c_block,), tile=new_rv)


@ct.kernel
def _bn_relu_kernel(
    x_ptr,           # bf16 (E, C)
    mean_ptr,        # f32 (C,)
    invstd_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,         # bf16 (E, C)
    BLOCK_E: ct.Constant[int],
    C_BLOCK: ct.Constant[int],
):
    e_block = ct.bid(0)
    c_block = ct.bid(1)

    x_bf = ct.load(x_ptr, index=(e_block, c_block),
                   shape=(BLOCK_E, C_BLOCK),
                   padding_mode=ct.PaddingMode.ZERO)
    x_f = ct.astype(x_bf, ct.float32)
    mean = ct.load(mean_ptr, index=(c_block,), shape=(C_BLOCK,),
                   padding_mode=ct.PaddingMode.ZERO)
    invstd = ct.load(invstd_ptr, index=(c_block,), shape=(C_BLOCK,),
                     padding_mode=ct.PaddingMode.ZERO)
    weight = ct.load(weight_ptr, index=(c_block,), shape=(C_BLOCK,),
                     padding_mode=ct.PaddingMode.ZERO)
    bias = ct.load(bias_ptr, index=(c_block,), shape=(C_BLOCK,),
                   padding_mode=ct.PaddingMode.ZERO)
    mean_2d = ct.reshape(mean, (1, C_BLOCK))
    invstd_2d = ct.reshape(invstd, (1, C_BLOCK))
    weight_2d = ct.reshape(weight, (1, C_BLOCK))
    bias_2d = ct.reshape(bias, (1, C_BLOCK))

    normalized = (x_f - mean_2d) * invstd_2d
    affine = normalized * weight_2d + bias_2d
    affine_bf = ct.astype(affine, ct.bfloat16)
    zero_bf = ct.zeros((BLOCK_E, C_BLOCK), dtype=ct.bfloat16)
    relu = ct.where(affine_bf > zero_bf, affine_bf, zero_bf)
    ct.store(out_ptr, index=(e_block, c_block), tile=relu)


def _next_pow2(x):
    return 1 << (int(x) - 1).bit_length()


@oracle_impl(hardware="B200", point="22c184d8")
def oracle_forward(inputs):
    x, rm, rv, weight, bias, kernel_size, stride = inputs
    n = int(x.shape[0])
    c = int(x.shape[1])
    h = int(x.shape[2])
    w = int(x.shape[3])
    e = n * h * w
    device = x.device

    # Channels-last -> NHWC contiguous view, then flatten to (E, C).
    x_nhwc = x.permute(0, 2, 3, 1).contiguous()
    x_flat = x_nhwc.view(e, c)

    block_e = 4096
    c_block = 8
    num_chunks = (e + block_e - 1) // block_e
    block_chunks = _next_pow2(num_chunks)

    ps = torch.empty((num_chunks, c), device=device, dtype=torch.float32)
    psq = torch.empty((num_chunks, c), device=device, dtype=torch.float32)
    saved_mean = torch.empty_strided((1, c, 1, 1), (c, 1, 1, 1),
                                     device=device, dtype=torch.float32)
    invstd = torch.empty_strided((1, c, 1, 1), (c, 1, 1, 1),
                                 device=device, dtype=torch.float32)
    new_rm = torch.empty((c,), device=device, dtype=torch.float32)
    new_rv = torch.empty((c,), device=device, dtype=torch.float32)

    # BN+ReLU output in NHWC contiguous bf16, then create channels-last view.
    bn_relu_nhwc = torch.empty((n, h, w, c), device=device, dtype=torch.bfloat16)
    bn_relu_flat = bn_relu_nhwc.view(e, c)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (num_chunks, (c + c_block - 1) // c_block, 1),
        _partial_stats_kernel,
        (x_flat, ps, psq, block_e, c_block),
    )
    ct.launch(
        stream,
        ((c + c_block - 1) // c_block, 1, 1),
        _finalize_stats_kernel,
        (ps, psq, rm, rv,
         saved_mean.view(c), invstd.view(c),
         new_rm, new_rv,
         num_chunks, block_chunks, c_block, 1.0 / e),
    )
    ct.launch(
        stream,
        (num_chunks, (c + c_block - 1) // c_block, 1),
        _bn_relu_kernel,
        (x_flat,
         saved_mean.view(c), invstd.view(c),
         weight, bias,
         bn_relu_flat,
         block_e, c_block),
    )

    # BN+ReLU as channels-last (N, C, H, W) view of NHWC storage.
    bn_relu = bn_relu_nhwc.permute(0, 3, 1, 2)
    # 3x3 stride-2 low-memory maxpool with offsets — native torch op that
    # matches eager numerics bit-exact.
    ks = list(kernel_size) if not isinstance(kernel_size, list) else kernel_size
    st = list(stride) if not isinstance(stride, list) else stride
    values, offsets = torch.ops.prims._low_memory_max_pool_with_offsets.default(
        bn_relu, ks, st, [0, 0], [1, 1], False,
    )

    # Running-stat mutation
    torch.ops.aten.copy_(rm, new_rm)
    torch.ops.aten.copy_(rv, new_rv)

    return saved_mean, invstd, values, offsets, rm, rv
