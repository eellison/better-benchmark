"""cuTile port of var_mean_88858c55c3b4: MobileViT/EfficientNet BN training + SiLU epilogue.

The inputs are channels-last bf16 [N, C, H, W]. We reinterpret as (E, C)
contiguous NHWC and run three cuTile kernels:
  1) per-channel partial sum + sum_sq over E axis (in chunks).
  2) finalize: total reduction, mean/var/invstd, running-stat update.
  3) SiLU affine epilogue: BN affine (fp32) -> bf16 -> SiLU (using exp) -> bf16.

Running-stats are copy_'d back to the input tensors post-kernel.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


EPS = 1.0e-5
VAR_CORRECTION = 1.0001220852154804


@ct.kernel
def _partial_stats_kernel(
    x_ptr,       # bf16 (E, C)
    ps_ptr,      # f32 (num_chunks, C)  partial sum
    psq_ptr,     # f32 (num_chunks, C)  partial sum_sq
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
    ps_ptr,          # f32 (num_chunks, C)
    psq_ptr,
    old_rm_ptr,      # f32 (C,)
    old_rv_ptr,
    saved_mean_ptr,
    invstd_ptr,
    new_rm_ptr,
    new_rv_ptr,
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
def _silu_epilogue_kernel(
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
    affine_f = ct.astype(affine_bf, ct.float32)
    neg = ct.zeros((BLOCK_E, C_BLOCK), dtype=ct.float32) - affine_f
    denom = ct.exp(neg) + 1.0
    silu = affine_f / denom
    ct.store(out_ptr, index=(e_block, c_block), tile=ct.astype(silu, ct.bfloat16))


def _next_pow2(x):
    return 1 << (int(x) - 1).bit_length()


def _run(inputs):
    x, rm, rv, weight, bias = inputs
    n = int(x.shape[0])
    c = int(x.shape[1])
    h = int(x.shape[2])
    w = int(x.shape[3])
    hw = h * w
    e = n * hw
    device = x.device

    # channels-last -> (N, HW, C) contiguous
    x_nhwc = x.permute(0, 2, 3, 1).contiguous().view(n, hw, c)
    x_flat = x_nhwc.view(e, c)

    block_e = 256
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
    out = torch.empty_strided((n, c, h, w),
                              (c * hw, 1, w * c, c),
                              device=device, dtype=torch.bfloat16)

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

    # Epilogue writes NHWC layout in a temp then copies via as-strided view.
    out_nhwc = torch.empty((n, hw, c), device=device, dtype=torch.bfloat16)
    out_flat = out_nhwc.view(e, c)
    ct.launch(
        stream,
        (num_chunks, (c + c_block - 1) // c_block, 1),
        _silu_epilogue_kernel,
        (x_flat,
         saved_mean.view(c), invstd.view(c),
         weight, bias,
         out_flat,
         block_e, c_block),
    )
    # Reshape NHWC -> NCHW channels-last
    out_reshaped = out_nhwc.view(n, h, w, c).permute(0, 3, 1, 2)
    out.copy_(out_reshaped)

    torch.ops.aten.copy_(rm, new_rm)
    torch.ops.aten.copy_(rv, new_rv)

    return saved_mean, invstd, out, rm, rv


@oracle_impl(hardware="B200", point="000da78a")
@oracle_impl(hardware="B200", point="2e27daef")
@oracle_impl(hardware="B200", point="ecabde94")
@oracle_impl(hardware="B200", point="3a7a99d6")
@oracle_impl(hardware="B200", point="85dd3e54")
@oracle_impl(hardware="B200", point="ceebf62c")
@oracle_impl(hardware="B200", point="ddcb5e92")
@oracle_impl(hardware="B200", point="d22d384a")
@oracle_impl(hardware="B200", point="42a503b0")
@oracle_impl(hardware="B200", point="2814ad41")
@oracle_impl(hardware="B200", point="e9256d98")
@oracle_impl(hardware="B200", point="0f9a93bc")
@oracle_impl(hardware="B200", point="837ab064")
@oracle_impl(hardware="B200", point="c2791544")
@oracle_impl(hardware="B200", point="e41460a6")
@oracle_impl(hardware="B200", point="886f27c9")
@oracle_impl(hardware="B200", point="cdca2f80")
@oracle_impl(hardware="B200", point="46238279")
@oracle_impl(hardware="B200", point="bf1cc8fb")
def oracle_forward(inputs):
    return _run(inputs)
