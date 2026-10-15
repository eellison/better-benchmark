"""cuTile port of sum_sum_sum_7ea3cfe27698: Adv-Inception avg-pool + 2 BN backward branches.

The avg_pool2d_backward part is done in torch (it's just a fixed 3x3 window with
count_include_pad=True). Then the 4 adds and 2 BN backward branches are done in cuTile.

Structure:
  - Kernel 1: pool + 3 bf16 adds → add_2 (channels-last strided [N, 1280, H, W]).
  - Kernel 2 (twice with different SLICE_OFFSETs/channels): BN backward branch.
    Uses split-K + finalize + epilogue.

We use torch for avg_pool2d_backward for simplicity (matches bf16 output exactly).
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


N = 128
C_TOTAL = 1280
H = 8
W = 8
HW = H * W
TOTAL_SPATIAL = N * HW
REDUCE_SCALE = 0.0001220703125


def _next_pow2(v: int) -> int:
    r = 1
    while r < v:
        r <<= 1
    return r


@ct.kernel
def _pool_add_kernel(
    pool_ptr,     # bf16 [N, C_TOTAL, HW]  (already-computed avg_pool2d_backward output)
    add0_ptr,     # bf16 [N, C_TOTAL, HW]  (arg2_1)
    add1_ptr,     # bf16 [N, C_TOTAL, HW]  (arg3_1)
    add2_ptr,     # bf16 [N, C_TOTAL, HW]  (arg4_1)
    out_ptr,      # bf16 [N, C_TOTAL, HW]
    HW_C: ct.Constant[int],
    BLOCK_HW: ct.Constant[int],
):
    """Grid: (N, C_TOTAL, HW // BLOCK_HW)."""
    n = ct.bid(0)
    c = ct.bid(1)
    hw_b = ct.bid(2)
    p = ct.load(pool_ptr, index=(n, c, hw_b), shape=(1, 1, BLOCK_HW))
    a0 = ct.load(add0_ptr, index=(n, c, hw_b), shape=(1, 1, BLOCK_HW))
    a1 = ct.load(add1_ptr, index=(n, c, hw_b), shape=(1, 1, BLOCK_HW))
    a2 = ct.load(add2_ptr, index=(n, c, hw_b), shape=(1, 1, BLOCK_HW))
    y = ct.astype(ct.astype(p, ct.float32) + ct.astype(a0, ct.float32), ct.bfloat16)
    y = ct.astype(ct.astype(y, ct.float32) + ct.astype(a1, ct.float32), ct.bfloat16)
    y = ct.astype(ct.astype(y, ct.float32) + ct.astype(a2, ct.float32), ct.bfloat16)
    ct.store(out_ptr, index=(n, c, hw_b), tile=y)


@ct.kernel
def _branch_partial_kernel(
    add2_ptr,        # bf16 [N, C_TOTAL, HW]
    activation_ptr,  # bf16 [N, C_BRANCH, HW]
    mean_ptr,        # f32 [C_BRANCH]
    invstd_ptr,      # f32 [C_BRANCH]
    gamma_ptr,       # f32 [C_BRANCH]
    beta_ptr,        # f32 [C_BRANCH]
    fill_ptr,        # bf16 [1]  (arg10_1)
    partial_sum_ptr, # f32 [num_tiles, C_BRANCH]
    partial_xhat_ptr,# f32 [num_tiles, C_BRANCH]
    SOURCE_OFFSET: ct.Constant[int],  # channel offset into add2 (0 for 320-branch, 320 for 192-branch)
    C_BRANCH: ct.Constant[int],
    C_TOTAL_C: ct.Constant[int],
    HW_C: ct.Constant[int],
    K_TOTAL: ct.Constant[int],
    BLOCK_K: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    """Grid: (C_BRANCH // BLOCK_C, num_tiles, 1). K = N*HW."""
    c_block = ct.bid(0)
    k_block = ct.bid(1)

    k_offsets = k_block * BLOCK_K + ct.arange(BLOCK_K, dtype=ct.int32)
    n_offsets = k_offsets // HW_C
    hw_offsets = k_offsets - n_offsets * HW_C
    k_mask = k_offsets < K_TOTAL

    c_offsets = c_block * BLOCK_C + ct.arange(BLOCK_C, dtype=ct.int32)
    c_offsets_full = c_offsets + SOURCE_OFFSET  # into add2's C_TOTAL space

    n_2d = ct.reshape(n_offsets, (BLOCK_K, 1))
    hw_2d = ct.reshape(hw_offsets, (BLOCK_K, 1))
    c_2d = ct.reshape(c_offsets, (1, BLOCK_C))
    c_full_2d = ct.reshape(c_offsets_full, (1, BLOCK_C))

    n_bc = ct.broadcast_to(n_2d, (BLOCK_K, BLOCK_C))
    hw_bc = ct.broadcast_to(hw_2d, (BLOCK_K, BLOCK_C))
    c_bc = ct.broadcast_to(c_2d, (BLOCK_K, BLOCK_C))
    c_full_bc = ct.broadcast_to(c_full_2d, (BLOCK_K, BLOCK_C))

    x = ct.astype(ct.gather(activation_ptr, (n_bc, c_bc, hw_bc)), ct.float32)
    mean = ct.astype(ct.load(mean_ptr, index=(c_block,), shape=(BLOCK_C,)), ct.float32)
    invstd = ct.astype(ct.load(invstd_ptr, index=(c_block,), shape=(BLOCK_C,)), ct.float32)
    gamma = ct.astype(ct.load(gamma_ptr, index=(c_block,), shape=(BLOCK_C,)), ct.float32)
    beta = ct.astype(ct.load(beta_ptr, index=(c_block,), shape=(BLOCK_C,)), ct.float32)
    mean_bc = ct.reshape(mean, (1, BLOCK_C))
    invstd_bc = ct.reshape(invstd, (1, BLOCK_C))
    gamma_bc = ct.reshape(gamma, (1, BLOCK_C))
    beta_bc = ct.reshape(beta, (1, BLOCK_C))

    centered = x - mean_bc
    affine = centered * invstd_bc * gamma_bc + beta_bc
    affine_bf16_f = ct.astype(ct.astype(affine, ct.bfloat16), ct.float32)
    relu_mask = affine_bf16_f <= 0.0

    src = ct.astype(ct.gather(add2_ptr, (n_bc, c_full_bc, hw_bc)), ct.float32)
    fill = ct.astype(ct.load(fill_ptr, index=(0,), shape=(1,)), ct.float32)
    fill_s = ct.reshape(fill, ())
    where_val = ct.where(relu_mask, fill_s, src)
    k_mask_2d = ct.reshape(k_mask, (BLOCK_K, 1))
    where_masked = ct.where(k_mask_2d, where_val, 0.0)

    partial_sum = ct.sum(where_masked, axis=0)
    partial_xhat = ct.sum(where_masked * centered, axis=0)

    ct.store(partial_sum_ptr, index=(k_block, c_block),
             tile=ct.reshape(partial_sum, (1, BLOCK_C)))
    ct.store(partial_xhat_ptr, index=(k_block, c_block),
             tile=ct.reshape(partial_xhat, (1, BLOCK_C)))


@ct.kernel
def _branch_epilogue_kernel(
    add2_ptr,        # bf16 [N, C_TOTAL, HW]
    activation_ptr,  # bf16 [N, C_BRANCH, HW]
    mean_ptr,        # f32 [C_BRANCH]
    invstd_ptr,      # f32 [C_BRANCH]
    gamma_ptr,       # f32 [C_BRANCH]
    beta_ptr,        # f32 [C_BRANCH]
    fill_ptr,        # bf16 [1]
    sum_ptr,         # f32 [C_BRANCH]
    xhat_sum_ptr,    # f32 [C_BRANCH]
    out_ptr,         # bf16 [N, C_BRANCH, HW]
    SOURCE_OFFSET: ct.Constant[int],
    C_BRANCH: ct.Constant[int],
    C_TOTAL_C: ct.Constant[int],
    HW_C: ct.Constant[int],
    REDUCE_SCALE_C: ct.Constant[float],
    BLOCK_C: ct.Constant[int],
    BLOCK_HW: ct.Constant[int],
):
    """Grid: (N, C_BRANCH // BLOCK_C, HW // BLOCK_HW)."""
    n = ct.bid(0)
    c_block = ct.bid(1)
    hw_block = ct.bid(2)

    # Load per-channel scalars.
    mean = ct.astype(ct.load(mean_ptr, index=(c_block,), shape=(BLOCK_C,)), ct.float32)
    invstd = ct.astype(ct.load(invstd_ptr, index=(c_block,), shape=(BLOCK_C,)), ct.float32)
    gamma = ct.astype(ct.load(gamma_ptr, index=(c_block,), shape=(BLOCK_C,)), ct.float32)
    beta = ct.astype(ct.load(beta_ptr, index=(c_block,), shape=(BLOCK_C,)), ct.float32)
    sumv = ct.astype(ct.load(sum_ptr, index=(c_block,), shape=(BLOCK_C,)), ct.float32)
    xhat_sumv = ct.astype(ct.load(xhat_sum_ptr, index=(c_block,), shape=(BLOCK_C,)), ct.float32)

    mean_bc = ct.reshape(mean, (1, BLOCK_C, 1))
    invstd_bc = ct.reshape(invstd, (1, BLOCK_C, 1))
    gamma_bc = ct.reshape(gamma, (1, BLOCK_C, 1))
    beta_bc = ct.reshape(beta, (1, BLOCK_C, 1))

    # Load full 3D tile.
    x = ct.astype(
        ct.load(activation_ptr, index=(n, c_block, hw_block), shape=(1, BLOCK_C, BLOCK_HW)),
        ct.float32,
    )
    centered = x - mean_bc
    affine = centered * invstd_bc * gamma_bc + beta_bc
    affine_bf16_f = ct.astype(ct.astype(affine, ct.bfloat16), ct.float32)
    relu_mask = affine_bf16_f <= 0.0

    # For add2 we need to load at channel offset SOURCE_OFFSET + c_block*BLOCK_C.
    # If SOURCE_OFFSET is divisible by BLOCK_C, we can shift the tile index.
    src_c_block = c_block + SOURCE_OFFSET // BLOCK_C
    src = ct.astype(
        ct.load(add2_ptr, index=(n, src_c_block, hw_block), shape=(1, BLOCK_C, BLOCK_HW)),
        ct.float32,
    )
    fill = ct.astype(ct.load(fill_ptr, index=(0,), shape=(1,)), ct.float32)
    fill_s = ct.reshape(fill, ())
    where_val = ct.where(relu_mask, fill_s, src)

    mean_term = ct.reshape(sumv * REDUCE_SCALE_C, (1, BLOCK_C, 1))
    invstd_sq = ct.reshape(invstd * invstd, (1, BLOCK_C, 1))
    variance_term = (ct.reshape(xhat_sumv * REDUCE_SCALE_C, (1, BLOCK_C, 1))) * invstd_sq
    affine_scale = ct.reshape(invstd * gamma, (1, BLOCK_C, 1))

    out = (where_val - centered * variance_term - mean_term) * affine_scale
    ct.store(out_ptr, index=(n, c_block, hw_block), tile=ct.astype(out, ct.bfloat16))


def _run_branch(add2_ncf, activation, mean, invstd, gamma, beta, full, source_offset,
                channels, device, *, BLOCK_C, BLOCK_K, BLOCK_HW_EP):
    """Run partial-reduce + finalize (torch) + epilogue for a branch."""
    k_total = N * HW
    num_tiles = (k_total + BLOCK_K - 1) // BLOCK_K

    partial_sum = torch.empty((num_tiles, channels), device=device, dtype=torch.float32)
    partial_xhat = torch.empty((num_tiles, channels), device=device, dtype=torch.float32)

    mean_1d = mean.view(channels).contiguous()
    invstd_1d = invstd.view(channels).contiguous()
    gamma_1d = gamma.view(channels).contiguous()
    beta_1d = beta.view(channels).contiguous()

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (channels // BLOCK_C, num_tiles, 1),
        _branch_partial_kernel,
        (
            add2_ncf, activation, mean_1d, invstd_1d, gamma_1d, beta_1d, full.view(1),
            partial_sum, partial_xhat,
            source_offset, channels, C_TOTAL, HW, k_total, BLOCK_K, BLOCK_C,
        ),
    )

    sum_out = partial_sum.sum(dim=0, dtype=torch.float32)
    xhat_sum = partial_xhat.sum(dim=0, dtype=torch.float32)
    vector_out = xhat_sum * invstd_1d  # mul_10 or mul_21

    tensor_out = torch.empty_strided(
        (N, channels, H, W), (channels * HW, 1, W * channels, channels),
        device=device, dtype=torch.bfloat16,
    )
    tensor_out_3d = torch.as_strided(tensor_out, (N, channels, HW), (channels * HW, 1, channels))

    if HW % BLOCK_HW_EP != 0:
        raise NotImplementedError(f"HW={HW} not divisible by BLOCK_HW_EP={BLOCK_HW_EP}")

    ct.launch(
        stream,
        (N, channels // BLOCK_C, HW // BLOCK_HW_EP),
        _branch_epilogue_kernel,
        (
            add2_ncf, activation, mean_1d, invstd_1d, gamma_1d, beta_1d, full.view(1),
            sum_out, xhat_sum, tensor_out_3d,
            source_offset, channels, C_TOTAL, HW, REDUCE_SCALE, BLOCK_C, BLOCK_HW_EP,
        ),
    )
    return sum_out, vector_out, tensor_out


@oracle_impl(hardware="B200", point="71ddf97f", ADD_BLOCK=256, C_BLOCK=16, K_BLOCK=512, ELEM_BLOCK=256)
def oracle_forward(inputs, **_kwargs):
    (
        arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10,
        arg11, arg12, arg13, arg14, arg15,
    ) = inputs
    device = arg0.device

    # avg_pool2d_backward: use torch (matches bf16 exactly for count_include_pad=True).
    pool_bwd = torch.ops.aten.avg_pool2d_backward.default(
        arg0, arg1, [3, 3], [1, 1], [1, 1], False, True, None
    )
    # pool_bwd, arg2, arg3, arg4 are all bf16 NHWC-strided [N, C_TOTAL, H, W].
    # Force NCHW-contiguous for the cuTile add kernel.
    pool_nchw = pool_bwd.contiguous().view(N, C_TOTAL, HW)
    arg2_nchw = arg2.contiguous().view(N, C_TOTAL, HW)
    arg3_nchw = arg3.contiguous().view(N, C_TOTAL, HW)
    arg4_nchw = arg4.contiguous().view(N, C_TOTAL, HW)

    add2_nchw = torch.empty((N, C_TOTAL, HW), device=device, dtype=torch.bfloat16)

    # HW=64 is power-of-2, use full BLOCK_HW.
    BLOCK_HW_ADD = 32
    if HW % BLOCK_HW_ADD != 0:
        BLOCK_HW_ADD = HW
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (N, C_TOTAL, HW // BLOCK_HW_ADD),
        _pool_add_kernel,
        (pool_nchw, arg2_nchw, arg3_nchw, arg4_nchw, add2_nchw, HW, BLOCK_HW_ADD),
    )

    # add_2 is NHWC-strided [N, 1280, H, W].
    add2 = torch.empty_strided(
        (N, C_TOTAL, H, W), (C_TOTAL * HW, 1, W * C_TOTAL, C_TOTAL),
        device=device, dtype=torch.bfloat16,
    )
    add2.copy_(add2_nchw.view(N, C_TOTAL, H, W))
    add2_3d_ncf = add2_nchw  # NCHW contiguous 3D view

    # 192-branch: SOURCE_OFFSET=320 (channels 320..511 of add2).
    # Activation is arg5 bf16 [N, 192, H, W] NHWC-strided.
    arg5_nchw = arg5.contiguous().view(N, 192, HW)
    sum_1, mul_10, out_192 = _run_branch(
        add2_3d_ncf, arg5_nchw, arg6, arg7, arg8, arg9, arg10, 320, 192, device,
        BLOCK_C=16, BLOCK_K=512, BLOCK_HW_EP=32,
    )

    # 320-branch: SOURCE_OFFSET=0 (channels 0..319 of add2).
    arg11_nchw = arg11.contiguous().view(N, 320, HW)
    sum_3, mul_21, out_320 = _run_branch(
        add2_3d_ncf, arg11_nchw, arg12, arg13, arg14, arg15, arg10, 0, 320, device,
        BLOCK_C=16, BLOCK_K=512, BLOCK_HW_EP=32,
    )

    return add2, sum_1, mul_10, out_192, sum_3, mul_21, out_320
