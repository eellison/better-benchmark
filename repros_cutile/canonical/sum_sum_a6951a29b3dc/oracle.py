"""cuTile port of sum_sum_a6951a29b3dc: GhostNet BN backward (add + slice + BN).

Matches Triton's 4-kernel structure: add-clone-copy, partial reduce with
split along N*HW, per-channel finalize, and the BN-backward epilogue.
All math ops (add, reductions, BN scale/mean-term compute) live inside
cuTile kernels.
"""

from __future__ import annotations

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


N = 512
C_IN = 40
C = 20
SLICE_START = 20
H = 28
W = 28
HW = H * W
TOTAL_SPATIAL = N * HW
OUT_NUMEL = N * C * HW
REDUCE_SCALE = 2.4912308673469386e-06
REDUCE_R = 1024
REDUCE_PARTIALS = TOTAL_SPATIAL // REDUCE_R  # 392


@ct.kernel
def _add_clone_copy_kernel(
    arg0_ptr,       # bf16 [N, C_IN, H, W] channels-last
    arg1_ptr,       # bf16 [N, C_IN, H, W] channels-last
    clone_ptr,      # bf16 [N, C_IN, H, W] contiguous NCHW
    copy_ptr,       # bf16 [N, C_IN, H, W] channels-last
    C_IN_C: ct.Constant[int],
    HW_C: ct.Constant[int],
    N_C: ct.Constant[int],
    BLOCK_HW: ct.Constant[int],
):
    """Grid: (N, C_IN, HW // BLOCK_HW). Produces clone (NCHW-contig) and
    copy (channels-last)."""
    n = ct.bid(0)
    c = ct.bid(1)
    hw_b = ct.bid(2)
    a0 = ct.astype(
        ct.load(arg0_ptr, index=(n, c, hw_b), shape=(1, 1, BLOCK_HW)),
        ct.float32,
    )
    a1 = ct.astype(
        ct.load(arg1_ptr, index=(n, c, hw_b), shape=(1, 1, BLOCK_HW)),
        ct.float32,
    )
    value_bf = ct.astype(a0 + a1, ct.bfloat16)
    ct.store(clone_ptr, index=(n, c, hw_b), tile=value_bf)
    ct.store(copy_ptr, index=(n, c, hw_b), tile=value_bf)


@ct.kernel
def _partial_reduce_kernel(
    copy_ptr,         # bf16 [N, C_IN, HW] channels-last  (via as_strided)
    activation_ptr,   # bf16 [N, C, HW]    channels-last
    mean_ptr,         # f32  [C]
    partial_prod_ptr, # f32  [REDUCE_PARTIALS, C]
    partial_sum_ptr,  # f32  [REDUCE_PARTIALS, C]
    C_IN_C: ct.Constant[int],
    C_C: ct.Constant[int],
    SLICE_START_C: ct.Constant[int],
    HW_C: ct.Constant[int],
    REDUCE_R_C: ct.Constant[int],
):
    """Grid: (C, REDUCE_PARTIALS, 1). Each program reduces REDUCE_R spatial
    steps for one (c, partial)."""
    c = ct.bid(0)
    partial = ct.bid(1)
    r_idx = ct.arange(REDUCE_R_C, dtype=ct.int32)
    r_global = partial * REDUCE_R_C + r_idx  # spatial index over N*HW
    n = r_global // HW_C
    spatial = r_global - n * HW_C

    c_copy = ct.full((REDUCE_R_C,), c + SLICE_START_C, dtype=ct.int32)
    c_act = ct.full((REDUCE_R_C,), c, dtype=ct.int32)

    x = ct.astype(
        ct.gather(copy_ptr, (n, c_copy, spatial)),
        ct.float32,
    )
    a = ct.astype(
        ct.gather(activation_ptr, (n, c_act, spatial)),
        ct.float32,
    )
    mean = ct.astype(ct.load(mean_ptr, index=(c,), shape=(1,)), ct.float32)
    centered = a - ct.broadcast_to(mean, (REDUCE_R_C,))
    prod = x * centered

    partial_sum = ct.sum(x)
    partial_prod = ct.sum(prod)
    ct.store(partial_sum_ptr, index=(partial, c),
             tile=ct.reshape(partial_sum, (1, 1)))
    ct.store(partial_prod_ptr, index=(partial, c),
             tile=ct.reshape(partial_prod, (1, 1)))


@ct.kernel
def _finalize_kernel(
    partial_prod_ptr, # f32 [REDUCE_PARTIALS, C]
    partial_sum_ptr,  # f32 [REDUCE_PARTIALS, C]
    invstd_ptr,       # f32 [C]
    weight_ptr,       # f32 [C]
    sum_out_ptr,      # f32 [C]
    vector_out_ptr,   # f32 [C]
    mean_term_ptr,    # f32 [C]
    prod_coeff_ptr,   # f32 [C]
    output_scale_ptr, # f32 [C]
    C_C: ct.Constant[int],
    PARTIALS_C: ct.Constant[int],
    BLOCK_C_: ct.Constant[int],
    BLOCK_R_: ct.Constant[int],
    SCALE_C: ct.Constant[float],
):
    """Grid: (C // BLOCK_C_,). Aggregates partials along partial-axis."""
    c_block = ct.bid(0)
    part_prod = ct.astype(
        ct.load(partial_prod_ptr, index=(0, c_block),
                shape=(BLOCK_R_, BLOCK_C_),
                padding_mode=ct.PaddingMode.ZERO),
        ct.float32,
    )
    part_sum = ct.astype(
        ct.load(partial_sum_ptr, index=(0, c_block),
                shape=(BLOCK_R_, BLOCK_C_),
                padding_mode=ct.PaddingMode.ZERO),
        ct.float32,
    )
    sum_prod = ct.sum(part_prod, axis=0)
    sum_x = ct.sum(part_sum, axis=0)

    invstd = ct.astype(ct.load(invstd_ptr, index=(c_block,), shape=(BLOCK_C_,)),
                       ct.float32)
    weight = ct.astype(ct.load(weight_ptr, index=(c_block,), shape=(BLOCK_C_,)),
                       ct.float32)
    mean_term = sum_x * SCALE_C
    prod_coeff = (sum_prod * SCALE_C) * (invstd * invstd)
    output_scale = invstd * weight

    ct.store(sum_out_ptr, index=(c_block,), tile=sum_x)
    ct.store(vector_out_ptr, index=(c_block,), tile=sum_prod * invstd)
    ct.store(mean_term_ptr, index=(c_block,), tile=mean_term)
    ct.store(prod_coeff_ptr, index=(c_block,), tile=prod_coeff)
    ct.store(output_scale_ptr, index=(c_block,), tile=output_scale)


@ct.kernel
def _epilogue_kernel(
    copy_ptr,          # bf16 [N, C_IN, HW] channels-last (via as_strided)
    activation_ptr,    # bf16 [N, C, HW]    channels-last (via as_strided)
    mean_ptr,          # f32  [C]
    mean_term_ptr,     # f32  [C]
    prod_coeff_ptr,    # f32  [C]
    output_scale_ptr,  # f32  [C]
    out_ptr,           # bf16 [N, C, HW]    channels-last (via as_strided)
    C_IN_C: ct.Constant[int],
    C_C: ct.Constant[int],
    SLICE_START_C: ct.Constant[int],
    HW_C: ct.Constant[int],
    NUMEL_C: ct.Constant[int],
    BLOCK_E: ct.Constant[int],
):
    """Grid: (NUMEL // BLOCK_E,). Iterate over the compact NCHW linear
    index; math preserves Triton semantics."""
    pid = ct.bid(0)
    idx = ct.arange(BLOCK_E, dtype=ct.int32)
    linear = pid * BLOCK_E + idx
    active = linear < NUMEL_C

    c_lin = linear - (linear // C_C) * C_C  # linear % C_C
    nhw = linear // C_C
    hw = nhw - (nhw // HW_C) * HW_C
    n_lin = nhw // HW_C

    c_full = c_lin + SLICE_START_C
    x = ct.astype(
        ct.gather(copy_ptr, (n_lin, c_full, hw), mask=active,
                  padding_value=ct.bfloat16(0.0)),
        ct.float32,
    )
    a = ct.astype(
        ct.gather(activation_ptr, (n_lin, c_lin, hw), mask=active,
                  padding_value=ct.bfloat16(0.0)),
        ct.float32,
    )
    mean = ct.astype(
        ct.gather(mean_ptr, c_lin, mask=active, padding_value=0.0),
        ct.float32,
    )
    mean_term = ct.astype(
        ct.gather(mean_term_ptr, c_lin, mask=active, padding_value=0.0),
        ct.float32,
    )
    prod_coeff = ct.astype(
        ct.gather(prod_coeff_ptr, c_lin, mask=active, padding_value=0.0),
        ct.float32,
    )
    output_scale = ct.astype(
        ct.gather(output_scale_ptr, c_lin, mask=active, padding_value=0.0),
        ct.float32,
    )
    centered = a - mean
    corrected = x - centered * prod_coeff - mean_term
    out = corrected * output_scale
    ct.scatter(out_ptr, (n_lin, c_lin, hw),
               ct.astype(out, ct.bfloat16), mask=active)


@oracle_impl(hardware="B200", point="05e71cc7")
def oracle_forward(inputs):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1 = inputs
    device = arg0_1.device

    # Kernel 1: add + clone + copy.
    clone = torch.empty_strided(
        (N, C_IN, H, W), (C_IN * HW, HW, W, 1),
        device=device, dtype=torch.bfloat16,
    )
    copy = torch.empty_strided(
        (N, C_IN, H, W), (C_IN * HW, 1, W * C_IN, C_IN),
        device=device, dtype=torch.bfloat16,
    )
    # HW=784 = 2^4 * 7^2; pick BLOCK_HW=16.
    BLOCK_HW = 16
    assert HW % BLOCK_HW == 0

    # Present arg0/arg1 as 3D (N, C_IN, HW) via as_strided, preserving
    # their channels-last per-element stride.
    arg0_3d = torch.as_strided(
        arg0_1, (N, C_IN, HW), (C_IN * HW, 1, C_IN),
    )
    arg1_3d = torch.as_strided(
        arg1_1, (N, C_IN, HW), (C_IN * HW, 1, C_IN),
    )
    clone_3d = clone.view(N, C_IN, HW)
    copy_3d = torch.as_strided(
        copy, (N, C_IN, HW), (C_IN * HW, 1, C_IN),
    )
    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (N, C_IN, HW // BLOCK_HW), _add_clone_copy_kernel,
        (arg0_3d, arg1_3d, clone_3d, copy_3d, C_IN, HW, N, BLOCK_HW),
    )

    partial_prod = torch.empty((REDUCE_PARTIALS, C), device=device,
                               dtype=torch.float32)
    partial_sum = torch.empty((REDUCE_PARTIALS, C), device=device,
                              dtype=torch.float32)

    mean_1d = arg3_1.view(C).contiguous()
    invstd_1d = arg4_1.view(C).contiguous()
    weight_1d = arg5_1.view(C).contiguous()

    # activation is channels-last strided; use as_strided to a 3D view.
    activation_3d = torch.as_strided(
        arg2_1, (N, C, HW), (C * HW, 1, C),
    )
    ct.launch(
        stream, (C, REDUCE_PARTIALS, 1), _partial_reduce_kernel,
        (copy_3d, activation_3d, mean_1d, partial_prod, partial_sum,
         C_IN, C, SLICE_START, HW, REDUCE_R),
    )

    sum_out = torch.empty((C,), device=device, dtype=torch.float32)
    vector_out = torch.empty((C,), device=device, dtype=torch.float32)
    mean_term = torch.empty((C,), device=device, dtype=torch.float32)
    prod_coeff = torch.empty((C,), device=device, dtype=torch.float32)
    output_scale = torch.empty((C,), device=device, dtype=torch.float32)
    BLOCK_C = 4  # C=20 -> divides
    assert C % BLOCK_C == 0
    partials_pow2 = 1 << (REDUCE_PARTIALS - 1).bit_length()
    ct.launch(
        stream, (C // BLOCK_C, 1, 1), _finalize_kernel,
        (partial_prod, partial_sum, invstd_1d, weight_1d,
         sum_out, vector_out, mean_term, prod_coeff, output_scale,
         C, REDUCE_PARTIALS, BLOCK_C, partials_pow2, REDUCE_SCALE),
    )

    out = torch.empty_strided(
        (N, C, H, W), (C * HW, 1, W * C, C),
        device=device, dtype=torch.bfloat16,
    )
    out_3d = torch.as_strided(
        out, (N, C, HW), (C * HW, 1, C),
    )
    BLOCK_E = 2048
    grid_e = (OUT_NUMEL + BLOCK_E - 1) // BLOCK_E
    ct.launch(
        stream, (grid_e, 1, 1), _epilogue_kernel,
        (copy_3d, activation_3d, mean_1d, mean_term, prod_coeff, output_scale, out_3d,
         C_IN, C, SLICE_START, HW, OUT_NUMEL, BLOCK_E),
    )
    return clone, copy, sum_out, vector_out, out
