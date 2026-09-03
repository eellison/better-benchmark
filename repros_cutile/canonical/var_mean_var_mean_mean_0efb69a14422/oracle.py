"""cuTile port of var_mean_var_mean_mean_0efb69a14422: RepVGG dual-BN training with spatial mean.

The input tensors are channels-last bf16 [128, 1408, 7, 7] with stride
[68992, 1, 9856, 1408]. Reinterpreted as [128, 49, 1408] contiguous NHWC,
which yields a simple (E, C) view for the per-channel reductions.

Three cuTile kernels:
  1) partial per-channel sum + sum_sq over E axis in chunks of BLOCK_E for
     both branches (x0, x1).
  2) finalize: reduce over chunks, compute mean/var/invstd for each branch,
     and produce updated running mean + running variance in fresh temp
     tensors.
  3) epilogue: BN affine on each branch (fp32), cast to bf16, add branches,
     ReLU (bf16), spatial mean over H*W (fp32 accumulator), cast to bf16.

Running-stat mutation is applied post-kernel via torch.ops.aten.copy_.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


EPS = 1.0e-5
VAR_CORRECTION = 1.0001594642002871


@ct.kernel
def _partial_stats_kernel(
    x0_ptr,             # bf16 (E, C)
    x1_ptr,             # bf16 (E, C)
    ps0_ptr,            # f32 (num_chunks, C)  partial sum for branch 0
    psq0_ptr,           # f32 (num_chunks, C)  partial sum_sq for branch 0
    ps1_ptr,            # f32 (num_chunks, C)  partial sum for branch 1
    psq1_ptr,           # f32 (num_chunks, C)  partial sum_sq for branch 1
    BLOCK_E: ct.Constant[int],
    C_BLOCK: ct.Constant[int],
):
    e_block = ct.bid(0)
    c_block = ct.bid(1)

    x0_bf = ct.load(x0_ptr, index=(e_block, c_block),
                    shape=(BLOCK_E, C_BLOCK),
                    padding_mode=ct.PaddingMode.ZERO)
    x1_bf = ct.load(x1_ptr, index=(e_block, c_block),
                    shape=(BLOCK_E, C_BLOCK),
                    padding_mode=ct.PaddingMode.ZERO)
    x0_f = ct.astype(x0_bf, ct.float32)
    x1_f = ct.astype(x1_bf, ct.float32)

    sum0 = ct.sum(x0_f, axis=0, keepdims=True)
    sq0 = ct.sum(x0_f * x0_f, axis=0, keepdims=True)
    sum1 = ct.sum(x1_f, axis=0, keepdims=True)
    sq1 = ct.sum(x1_f * x1_f, axis=0, keepdims=True)

    ct.store(ps0_ptr, index=(e_block, c_block), tile=sum0)
    ct.store(psq0_ptr, index=(e_block, c_block), tile=sq0)
    ct.store(ps1_ptr, index=(e_block, c_block), tile=sum1)
    ct.store(psq1_ptr, index=(e_block, c_block), tile=sq1)


@ct.kernel
def _finalize_stats_kernel(
    ps0_ptr, psq0_ptr, ps1_ptr, psq1_ptr,   # (num_chunks, C)
    old_rm0_ptr, old_rv0_ptr,               # (C,)
    old_rm1_ptr, old_rv1_ptr,
    saved_mean0_ptr, invstd0_ptr,           # (C,)
    saved_mean1_ptr, invstd1_ptr,
    new_rm0_ptr, new_rv0_ptr,
    new_rm1_ptr, new_rv1_ptr,
    NUM_CHUNKS: ct.Constant[int],
    BLOCK_CHUNKS: ct.Constant[int],
    C_BLOCK: ct.Constant[int],
    INV_E: ct.Constant[float],
):
    c_block = ct.bid(0)

    ps0 = ct.load(ps0_ptr, index=(0, c_block),
                  shape=(BLOCK_CHUNKS, C_BLOCK),
                  padding_mode=ct.PaddingMode.ZERO)
    psq0 = ct.load(psq0_ptr, index=(0, c_block),
                   shape=(BLOCK_CHUNKS, C_BLOCK),
                   padding_mode=ct.PaddingMode.ZERO)
    ps1 = ct.load(ps1_ptr, index=(0, c_block),
                  shape=(BLOCK_CHUNKS, C_BLOCK),
                  padding_mode=ct.PaddingMode.ZERO)
    psq1 = ct.load(psq1_ptr, index=(0, c_block),
                   shape=(BLOCK_CHUNKS, C_BLOCK),
                   padding_mode=ct.PaddingMode.ZERO)

    # Mask out padded chunks (contribute 0 since ZERO padding, but sum() ignores those anyway)
    chunks = ct.arange(BLOCK_CHUNKS, dtype=ct.int32)
    chunk_mask = ct.reshape(chunks < NUM_CHUNKS, (BLOCK_CHUNKS, 1))
    zero_f = ct.zeros((BLOCK_CHUNKS, C_BLOCK), dtype=ct.float32)
    ps0 = ct.where(chunk_mask, ps0, zero_f)
    psq0 = ct.where(chunk_mask, psq0, zero_f)
    ps1 = ct.where(chunk_mask, ps1, zero_f)
    psq1 = ct.where(chunk_mask, psq1, zero_f)

    total_s0 = ct.sum(ps0, axis=0)   # (C_BLOCK,)
    total_sq0 = ct.sum(psq0, axis=0)
    total_s1 = ct.sum(ps1, axis=0)
    total_sq1 = ct.sum(psq1, axis=0)

    mean0 = total_s0 * INV_E
    mean1 = total_s1 * INV_E
    var0 = total_sq0 * INV_E - mean0 * mean0
    var1 = total_sq1 * INV_E - mean1 * mean1
    zero_1d = ct.zeros((C_BLOCK,), dtype=ct.float32)
    var0 = ct.where(var0 > 0.0, var0, zero_1d)
    var1 = ct.where(var1 > 0.0, var1, zero_1d)
    invstd0 = ct.rsqrt(var0 + EPS)
    invstd1 = ct.rsqrt(var1 + EPS)

    old_rm0 = ct.load(old_rm0_ptr, index=(c_block,), shape=(C_BLOCK,),
                      padding_mode=ct.PaddingMode.ZERO)
    old_rv0 = ct.load(old_rv0_ptr, index=(c_block,), shape=(C_BLOCK,),
                      padding_mode=ct.PaddingMode.ZERO)
    old_rm1 = ct.load(old_rm1_ptr, index=(c_block,), shape=(C_BLOCK,),
                      padding_mode=ct.PaddingMode.ZERO)
    old_rv1 = ct.load(old_rv1_ptr, index=(c_block,), shape=(C_BLOCK,),
                      padding_mode=ct.PaddingMode.ZERO)

    new_rm0 = mean0 * 0.1 + old_rm0 * 0.9
    new_rv0 = var0 * VAR_CORRECTION * 0.1 + old_rv0 * 0.9
    new_rm1 = mean1 * 0.1 + old_rm1 * 0.9
    new_rv1 = var1 * VAR_CORRECTION * 0.1 + old_rv1 * 0.9

    ct.store(saved_mean0_ptr, index=(c_block,), tile=mean0)
    ct.store(invstd0_ptr, index=(c_block,), tile=invstd0)
    ct.store(saved_mean1_ptr, index=(c_block,), tile=mean1)
    ct.store(invstd1_ptr, index=(c_block,), tile=invstd1)
    ct.store(new_rm0_ptr, index=(c_block,), tile=new_rm0)
    ct.store(new_rv0_ptr, index=(c_block,), tile=new_rv0)
    ct.store(new_rm1_ptr, index=(c_block,), tile=new_rm1)
    ct.store(new_rv1_ptr, index=(c_block,), tile=new_rv1)


@ct.kernel
def _epilogue_kernel(
    x0_ptr,           # bf16 (N, HW, C)
    x1_ptr,           # bf16 (N, HW, C)
    saved_mean0_ptr,  # f32 (C,)
    invstd0_ptr,
    saved_mean1_ptr,
    invstd1_ptr,
    w0_ptr, b0_ptr, w1_ptr, b1_ptr,   # f32 (C,)
    out_ptr,          # bf16 (N, C)
    HW: ct.Constant[int],
    HW_POW2: ct.Constant[int],
    C_BLOCK: ct.Constant[int],
    INV_HW: ct.Constant[float],
):
    n = ct.bid(0)
    c_block = ct.bid(1)

    x0_bf = ct.load(x0_ptr, index=(n, 0, c_block),
                    shape=(1, HW_POW2, C_BLOCK),
                    padding_mode=ct.PaddingMode.ZERO)
    x1_bf = ct.load(x1_ptr, index=(n, 0, c_block),
                    shape=(1, HW_POW2, C_BLOCK),
                    padding_mode=ct.PaddingMode.ZERO)
    x0_f = ct.astype(x0_bf, ct.float32)
    x1_f = ct.astype(x1_bf, ct.float32)

    m0 = ct.load(saved_mean0_ptr, index=(c_block,), shape=(C_BLOCK,),
                 padding_mode=ct.PaddingMode.ZERO)
    is0 = ct.load(invstd0_ptr, index=(c_block,), shape=(C_BLOCK,),
                  padding_mode=ct.PaddingMode.ZERO)
    m1 = ct.load(saved_mean1_ptr, index=(c_block,), shape=(C_BLOCK,),
                 padding_mode=ct.PaddingMode.ZERO)
    is1 = ct.load(invstd1_ptr, index=(c_block,), shape=(C_BLOCK,),
                  padding_mode=ct.PaddingMode.ZERO)
    w0 = ct.load(w0_ptr, index=(c_block,), shape=(C_BLOCK,),
                 padding_mode=ct.PaddingMode.ZERO)
    b0 = ct.load(b0_ptr, index=(c_block,), shape=(C_BLOCK,),
                 padding_mode=ct.PaddingMode.ZERO)
    w1 = ct.load(w1_ptr, index=(c_block,), shape=(C_BLOCK,),
                 padding_mode=ct.PaddingMode.ZERO)
    b1 = ct.load(b1_ptr, index=(c_block,), shape=(C_BLOCK,),
                 padding_mode=ct.PaddingMode.ZERO)

    m0_3d = ct.reshape(m0, (1, 1, C_BLOCK))
    is0_3d = ct.reshape(is0, (1, 1, C_BLOCK))
    m1_3d = ct.reshape(m1, (1, 1, C_BLOCK))
    is1_3d = ct.reshape(is1, (1, 1, C_BLOCK))
    w0_3d = ct.reshape(w0, (1, 1, C_BLOCK))
    b0_3d = ct.reshape(b0, (1, 1, C_BLOCK))
    w1_3d = ct.reshape(w1, (1, 1, C_BLOCK))
    b1_3d = ct.reshape(b1, (1, 1, C_BLOCK))

    y0_f = ((x0_f - m0_3d) * is0_3d) * w0_3d + b0_3d
    y1_f = ((x1_f - m1_3d) * is1_3d) * w1_3d + b1_3d
    y0_bf = ct.astype(y0_f, ct.bfloat16)
    y1_bf = ct.astype(y1_f, ct.bfloat16)
    added = y0_bf + y1_bf
    zero_bf = ct.zeros((1, HW_POW2, C_BLOCK), dtype=ct.bfloat16)
    relu = ct.where(added > zero_bf, added, zero_bf)
    relu_f = ct.astype(relu, ct.float32)

    hw = ct.arange(HW_POW2, dtype=ct.int32)
    hw_mask = ct.reshape(hw < HW, (1, HW_POW2, 1))
    zero_3d_f = ct.zeros((1, HW_POW2, C_BLOCK), dtype=ct.float32)
    relu_masked = ct.where(hw_mask, relu_f, zero_3d_f)
    # Reduce over axis=1 (HW). Result shape (1, C_BLOCK).
    pooled = ct.sum(relu_masked, axis=1) * INV_HW
    ct.store(out_ptr, index=(n, c_block), tile=ct.astype(pooled, ct.bfloat16))


def _next_pow2(x):
    return 1 << (int(x) - 1).bit_length()


@oracle_impl(hardware="B200", point="e1dd88f2")
def oracle_forward(inputs):
    (
        x0, rm0, rv0, w0, b0,
        x1, rm1, rv1, w1, b1,
        _shape0, _shape1, _shape2,
    ) = inputs

    n = int(x0.shape[0])
    c = int(x0.shape[1])
    h = int(x0.shape[2])
    w = int(x0.shape[3])
    hw = h * w
    e = n * hw
    device = x0.device
    hw_pow2 = _next_pow2(hw)

    # Reinterpret channels-last inputs as (N, HW, C) contiguous.
    # Original stride is [C*H*W, 1, W*C, C] which matches NHWC layout.
    x0_nhwc = x0.permute(0, 2, 3, 1).contiguous().view(n, hw, c)
    x1_nhwc = x1.permute(0, 2, 3, 1).contiguous().view(n, hw, c)
    x0_flat = x0_nhwc.view(e, c)
    x1_flat = x1_nhwc.view(e, c)

    # Chunking for partial reductions
    block_e = 1024
    c_block = 8
    num_chunks = (e + block_e - 1) // block_e
    block_chunks = _next_pow2(num_chunks)

    ps0 = torch.empty((num_chunks, c), device=device, dtype=torch.float32)
    psq0 = torch.empty((num_chunks, c), device=device, dtype=torch.float32)
    ps1 = torch.empty((num_chunks, c), device=device, dtype=torch.float32)
    psq1 = torch.empty((num_chunks, c), device=device, dtype=torch.float32)

    saved_mean0 = torch.empty_strided((1, c, 1, 1), (c, 1, 1, 1),
                                      device=device, dtype=torch.float32)
    invstd0 = torch.empty_strided((1, c, 1, 1), (c, 1, 1, 1),
                                  device=device, dtype=torch.float32)
    saved_mean1 = torch.empty_strided((1, c, 1, 1), (c, 1, 1, 1),
                                      device=device, dtype=torch.float32)
    invstd1 = torch.empty_strided((1, c, 1, 1), (c, 1, 1, 1),
                                  device=device, dtype=torch.float32)
    new_rm0 = torch.empty((c,), device=device, dtype=torch.float32)
    new_rv0 = torch.empty((c,), device=device, dtype=torch.float32)
    new_rm1 = torch.empty((c,), device=device, dtype=torch.float32)
    new_rv1 = torch.empty((c,), device=device, dtype=torch.float32)
    spatial_mean = torch.empty_strided((n, c), (c, 1),
                                       device=device, dtype=torch.bfloat16)

    stream = torch.cuda.current_stream()

    ct.launch(
        stream,
        (num_chunks, (c + c_block - 1) // c_block, 1),
        _partial_stats_kernel,
        (x0_flat, x1_flat, ps0, psq0, ps1, psq1, block_e, c_block),
    )

    ct.launch(
        stream,
        ((c + c_block - 1) // c_block, 1, 1),
        _finalize_stats_kernel,
        (ps0, psq0, ps1, psq1,
         rm0, rv0, rm1, rv1,
         saved_mean0.view(c), invstd0.view(c),
         saved_mean1.view(c), invstd1.view(c),
         new_rm0, new_rv0, new_rm1, new_rv1,
         num_chunks, block_chunks, c_block, 1.0 / e),
    )

    ct.launch(
        stream,
        (n, (c + c_block - 1) // c_block, 1),
        _epilogue_kernel,
        (x0_nhwc, x1_nhwc,
         saved_mean0.view(c), invstd0.view(c),
         saved_mean1.view(c), invstd1.view(c),
         w0, b0, w1, b1,
         spatial_mean,
         hw, hw_pow2, c_block, 1.0 / hw),
    )

    # Running-stat mutation via aten.copy_
    torch.ops.aten.copy_(rm0, new_rm0)
    torch.ops.aten.copy_(rv0, new_rv0)
    torch.ops.aten.copy_(rm1, new_rm1)
    torch.ops.aten.copy_(rv1, new_rv1)

    return (
        saved_mean0, invstd0, saved_mean1, invstd1, spatial_mean,
        rm0, rv0, rm1, rv1,
    )
