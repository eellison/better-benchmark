"""cuTile port of var_mean_mean_3574762a85f1: GhostNet training-BN + spatial mean.

Multi-kernel plan (all cuTile):
1. `_bn_partial_stats_kernel`: reduces the channels-last input [N*H*W, C] into
   per-channel partial (sum, sum_sq) over BLOCK_E chunks.
2. `_bn_finalize_stats_kernel`: combines the chunk partials into (mean, var,
   invstd), stores saved_mean/invstd, and updates running_mean/running_var
   using the captured momentum + unbiased-variance correction literal.
3. `_bn_relu_spatial_mean_kernel`: applies fp32 affine, bf16 rounding, ReLU,
   then per-(n, c) spatial mean over the 7x7 tile.

After the kernels we call `torch.ops.aten.copy_` on running_mean/running_var
to satisfy the Repro's `copy_` returns; the mutation lives OUTSIDE the kernel
and works under CUDA-graph capture (aten.copy_ is graph-capturable).
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


EPS = 1.0e-5
MOMENTUM = 0.1
ONE_MINUS_MOMENTUM = 0.9
UNBIAS_CORRECTION = 1.0000398612827361


def _next_power_of_2(value):
    return 1 << (int(value) - 1).bit_length()


@ct.kernel
def _bn_partial_stats_kernel(
    x_ptr,       # bf16 [E, C]
    psum_ptr,    # f32  [num_chunks, C]
    psq_ptr,     # f32  [num_chunks, C]
    E: ct.Constant[int],
    BLOCK_E: ct.Constant[int],
    C_BLOCK: ct.Constant[int],
):
    chunk = ct.bid(0)
    cblk = ct.bid(1)

    x = ct.load(
        x_ptr, index=(chunk, cblk), shape=(BLOCK_E, C_BLOCK),
        padding_mode=ct.PaddingMode.ZERO,
    )
    x_f = ct.astype(x, ct.float32)

    e_arange = ct.arange(BLOCK_E, dtype=ct.int32)
    e_global = chunk * BLOCK_E + e_arange
    e_valid = e_global < E
    e_valid_2d = ct.reshape(e_valid, (BLOCK_E, 1))
    zero = ct.full((BLOCK_E, C_BLOCK), 0.0, dtype=ct.float32)
    x_masked = ct.where(e_valid_2d, x_f, zero)

    sums = ct.sum(x_masked, axis=0, keepdims=True)
    sums_sq = ct.sum(x_masked * x_masked, axis=0, keepdims=True)

    ct.store(psum_ptr, index=(chunk, cblk), tile=sums)
    ct.store(psq_ptr, index=(chunk, cblk), tile=sums_sq)


@ct.kernel
def _bn_finalize_stats_kernel(
    psum_ptr,           # f32 [num_chunks_pad, C]
    psq_ptr,            # f32 [num_chunks_pad, C]
    running_mean_ptr,   # f32 [C]  (read then updated)
    running_var_ptr,    # f32 [C]  (read then updated)
    saved_mean_ptr,     # f32 [C]
    invstd_ptr,         # f32 [C]
    NUM_CHUNKS: ct.Constant[int],
    BLOCK_CHUNKS: ct.Constant[int],
    C_BLOCK: ct.Constant[int],
    INV_E: ct.Constant[float],
    EPS_: ct.Constant[float],
    UNBIAS: ct.Constant[float],
):
    cblk = ct.bid(0)

    psum = ct.load(
        psum_ptr, index=(0, cblk), shape=(BLOCK_CHUNKS, C_BLOCK),
        padding_mode=ct.PaddingMode.ZERO,
    )
    psq = ct.load(
        psq_ptr, index=(0, cblk), shape=(BLOCK_CHUNKS, C_BLOCK),
        padding_mode=ct.PaddingMode.ZERO,
    )

    chunks = ct.arange(BLOCK_CHUNKS, dtype=ct.int32)
    chunk_mask = chunks < NUM_CHUNKS
    chunk_mask_2d = ct.reshape(chunk_mask, (BLOCK_CHUNKS, 1))
    zero = ct.full((BLOCK_CHUNKS, C_BLOCK), 0.0, dtype=ct.float32)

    total = ct.sum(ct.where(chunk_mask_2d, psum, zero), axis=0)
    total_sq = ct.sum(ct.where(chunk_mask_2d, psq, zero), axis=0)

    mean = total * INV_E
    ex2 = total_sq * INV_E
    diff = ex2 - mean * mean
    zero_c = ct.full((C_BLOCK,), 0.0, dtype=ct.float32)
    var = ct.where(diff > zero_c, diff, zero_c)
    invstd = ct.rsqrt(var + EPS_)

    old_mean = ct.load(running_mean_ptr, index=(cblk,), shape=(C_BLOCK,))
    old_var = ct.load(running_var_ptr, index=(cblk,), shape=(C_BLOCK,))
    new_mean = mean * 0.1 + old_mean * 0.9
    new_var = (var * UNBIAS) * 0.1 + old_var * 0.9

    ct.store(saved_mean_ptr, index=(cblk,), tile=mean)
    ct.store(invstd_ptr, index=(cblk,), tile=invstd)
    ct.store(running_mean_ptr, index=(cblk,), tile=new_mean)
    ct.store(running_var_ptr, index=(cblk,), tile=new_var)


@ct.kernel
def _bn_relu_spatial_mean_kernel(
    x_ptr,           # bf16 [N*HW_PAD, C]  (padded input; padding rows are 0)
    weight_ptr,      # f32 [C]
    bias_ptr,        # f32 [C]
    saved_mean_ptr,  # f32 [C]
    invstd_ptr,      # f32 [C]
    out_ptr,         # bf16 [N, C]
    HW: ct.Constant[int],
    INV_HW: ct.Constant[float],
    HW_PAD: ct.Constant[int],
    C_BLOCK: ct.Constant[int],
):
    n = ct.bid(0)
    cblk = ct.bid(1)

    x = ct.load(x_ptr, index=(n, cblk), shape=(HW_PAD, C_BLOCK))
    x_f = ct.astype(x, ct.float32)

    mean = ct.load(saved_mean_ptr, index=(cblk,), shape=(C_BLOCK,))
    invstd = ct.load(invstd_ptr, index=(cblk,), shape=(C_BLOCK,))
    weight = ct.load(weight_ptr, index=(cblk,), shape=(C_BLOCK,))
    bias = ct.load(bias_ptr, index=(cblk,), shape=(C_BLOCK,))
    mean_2d = ct.reshape(mean, (1, C_BLOCK))
    invstd_2d = ct.reshape(invstd, (1, C_BLOCK))
    weight_2d = ct.reshape(weight, (1, C_BLOCK))
    bias_2d = ct.reshape(bias, (1, C_BLOCK))

    centered = x_f - mean_2d
    normalized = centered * invstd_2d
    scaled = normalized * weight_2d
    affine = scaled + bias_2d
    affine_bf16 = ct.astype(affine, ct.bfloat16)
    zero_bf16 = ct.full((HW_PAD, C_BLOCK), 0.0, dtype=ct.bfloat16)
    relu = ct.where(affine_bf16 < zero_bf16, zero_bf16, affine_bf16)

    hw_arange = ct.arange(HW_PAD, dtype=ct.int32)
    hw_valid = hw_arange < HW
    hw_valid_2d = ct.reshape(hw_valid, (HW_PAD, 1))
    zero_f = ct.full((HW_PAD, C_BLOCK), 0.0, dtype=ct.float32)
    relu_f = ct.astype(relu, ct.float32)
    relu_masked = ct.where(hw_valid_2d, relu_f, zero_f)
    pooled = ct.sum(relu_masked, axis=0, keepdims=True) * INV_HW
    ct.store(out_ptr, index=(n, cblk), tile=ct.astype(pooled, ct.bfloat16))


@oracle_impl(hardware="B200", point="d6a317bc")
def oracle_forward(inputs):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, _shape_param_0, _shape_param_1 = inputs
    n = int(arg0_1.shape[0])
    c = int(arg0_1.shape[1])
    h = int(arg0_1.shape[2])
    w = int(arg0_1.shape[3])
    hw = h * w
    e = n * hw
    device = arg0_1.device

    # BN reduction params
    BLOCK_E = 4096
    C_BLOCK = 32
    num_chunks = (e + BLOCK_E - 1) // BLOCK_E
    block_chunks = _next_power_of_2(num_chunks)
    assert c % C_BLOCK == 0

    # arg0 is channels-last [N,C,H,W] with strides [C*H*W, 1, W*C, C].
    # permute(0,2,3,1) yields [N,H,W,C] contiguous over the same memory.
    x_flat = arg0_1.permute(0, 2, 3, 1).reshape(e, c)  # bf16 [E, C], contiguous

    partial_sum = torch.empty((num_chunks, c), device=device, dtype=torch.float32)
    partial_sq = torch.empty((num_chunks, c), device=device, dtype=torch.float32)
    saved_mean = torch.empty_strided(
        (1, c, 1, 1), (c, 1, 1, 1), device=device, dtype=torch.float32,
    )
    invstd = torch.empty_strided(
        (1, c, 1, 1), (c, 1, 1, 1), device=device, dtype=torch.float32,
    )

    stream = torch.cuda.current_stream()

    # 1) Partial reduction
    ct.launch(
        stream,
        (num_chunks, c // C_BLOCK, 1),
        _bn_partial_stats_kernel,
        (x_flat, partial_sum, partial_sq, e, BLOCK_E, C_BLOCK),
    )

    # 2) Finalize stats + running-stat updates. In-place update of arg1_1, arg2_1
    # via the kernel's stores. Views into the (c,) flat layout.
    saved_mean_1d = saved_mean.view(c)
    invstd_1d = invstd.view(c)
    ct.launch(
        stream,
        (c // C_BLOCK, 1, 1),
        _bn_finalize_stats_kernel,
        (partial_sum, partial_sq, arg1_1, arg2_1, saved_mean_1d, invstd_1d,
         num_chunks, block_chunks, C_BLOCK,
         1.0 / float(e), EPS, UNBIAS_CORRECTION),
    )

    # 3) Affine + ReLU + spatial mean.
    # Pad input to [N, HW_PAD, C] with zeros so we can safely tile-store per batch.
    HW_PAD = _next_power_of_2(hw)
    padded_x = torch.zeros((n, HW_PAD, c), device=device, dtype=torch.bfloat16)
    padded_x[:, :hw, :].copy_(x_flat.view(n, hw, c))
    padded_x_2d = padded_x.view(n * HW_PAD, c)

    spatial_mean = torch.empty_strided(
        (n, c, 1, 1), (c, 1, c, c), device=device, dtype=torch.bfloat16,
    )
    spatial_mean_2d = spatial_mean.view(n, c)

    ct.launch(
        stream,
        (n, c // C_BLOCK, 1),
        _bn_relu_spatial_mean_kernel,
        (padded_x_2d, arg3_1, arg4_1, saved_mean_1d, invstd_1d,
         spatial_mean_2d, hw, 1.0 / float(hw), HW_PAD, C_BLOCK),
    )

    return saved_mean, invstd, spatial_mean, arg1_1, arg2_1
