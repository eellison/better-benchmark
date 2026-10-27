"""cuTile port of sum_sum_4289edc98a8d: GhostNet BN backward tail.

Mirrors Triton's 4-kernel structure with `ct.sum` in-kernel:
  - `_add_clone_copy_kernel`: add arg0+arg1, dual-store contig (clone) and
    channels-last (copy).
  - `_partial_reduce_kernel`: reduce the upper-40-channel slice of `copy`
    against `arg2-arg3` (per-channel), producing (C, REDUCE_PARTIALS) partials
    via `ct.sum(..., axis=1)`.
  - `_finalize_partials_kernel`: reduce partials across the partial dimension
    via `ct.sum`, then compute BN scale factors.
  - `_epilogue_kernel`: apply BN backward to produce channels-last bf16 out.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


N = 512
C_IN = 80
C = 40
SLICE_START = 40
H = 14
W = 14
HW = H * W  # 196
TOTAL_SPATIAL = N * HW  # 100352
OUT_NUMEL = N * C * HW
REDUCE_SCALE = 9.964923469387754e-06
REDUCE_R = 1024
REDUCE_PARTIALS = TOTAL_SPATIAL // REDUCE_R  # 98


def _next_pow2(v):
    return 1 << (int(v) - 1).bit_length()


@ct.kernel
def _add_clone_copy_kernel(
    arg0_ptr,       # bf16 [N, C_IN, H, W] channels-last flat (numel=N*C_IN*HW)
    arg1_ptr,       # bf16 [N, C_IN, H, W] channels-last flat
    clone_ptr,      # bf16 flat contiguous storage
    copy_ptr,       # bf16 flat channels-last storage
    YBLOCK: ct.Constant[int],
    XBLOCK: ct.Constant[int],
):
    y_tile = ct.bid(1)
    x_tile = ct.bid(0)
    y = y_tile * YBLOCK + ct.arange(YBLOCK, dtype=ct.int32)  # (YBLOCK,)
    x = x_tile * XBLOCK + ct.arange(XBLOCK, dtype=ct.int32)  # (XBLOCK,)
    y2 = ct.reshape(y, (YBLOCK, 1))
    x2 = ct.reshape(x, (1, XBLOCK))
    y_active = ct.reshape(y < (N * C_IN), (YBLOCK, 1))
    x_active = ct.reshape(x < HW, (1, XBLOCK))
    active = y_active & x_active

    c = y2 - (y2 // C_IN) * C_IN
    n = y2 // C_IN
    channels_last_offset = c + C_IN * x2 + (C_IN * HW) * n
    contiguous_offset = x2 + HW * y2

    value = ct.astype(ct.gather(arg0_ptr, channels_last_offset, mask=active), ct.float32) + \
            ct.astype(ct.gather(arg1_ptr, channels_last_offset, mask=active), ct.float32)
    value_bf = ct.astype(value, ct.bfloat16)
    ct.scatter(clone_ptr, contiguous_offset, value_bf, mask=active)
    ct.scatter(copy_ptr, channels_last_offset, value_bf, mask=active)


@ct.kernel
def _partial_reduce_kernel(
    copy_ptr,          # bf16 flat channels-last storage
    activation_ptr,    # bf16 [N, C, H, W] channels-last flat
    mean_ptr,          # f32 (C,)
    partial_prod_ptr,  # f32 (C * REDUCE_PARTIALS,) stride (1, C)
    partial_sum_ptr,   # f32 (C * REDUCE_PARTIALS,) stride (1, C)
    XBLOCK: ct.Constant[int],
    RBLOCK: ct.Constant[int],
):
    x_tile = ct.bid(0)
    xindex_1d = x_tile * XBLOCK + ct.arange(XBLOCK, dtype=ct.int32)
    r_1d = ct.arange(RBLOCK, dtype=ct.int32)
    xindex = ct.reshape(xindex_1d, (XBLOCK, 1))
    r = ct.reshape(r_1d, (1, RBLOCK))
    xmask_1d = xindex_1d < (C * REDUCE_PARTIALS)
    xmask = ct.reshape(xmask_1d, (XBLOCK, 1))
    rmask = r < REDUCE_R
    active = xmask & rmask

    c = xindex - (xindex // C) * C
    partial = xindex // C

    # Offsets for `copy` (channels-last, C_IN channels stride 1):
    #   offset = SLICE_START + c + C_IN * r + (C_IN * REDUCE_R) * partial
    copy_offsets = SLICE_START + c + C_IN * r + (C_IN * REDUCE_R) * partial
    # For `activation` (channels-last, C channels stride 1):
    activation_offsets = c + C * r + (C * REDUCE_R) * partial

    x = ct.astype(ct.gather(copy_ptr, copy_offsets), ct.float32)
    a = ct.astype(ct.gather(activation_ptr, activation_offsets), ct.float32)
    c_1d = xindex_1d - (xindex_1d // C) * C
    mean = ct.gather(mean_ptr, c_1d)  # (XBLOCK,)
    mean_2d = ct.reshape(mean, (XBLOCK, 1))
    centered = a - mean_2d
    prod = x * centered

    # Zero masked lanes so ct.sum(axis=1) is correct.
    x_masked = ct.where(active, x, 0.0)
    prod_masked = ct.where(active, prod, 0.0)

    sum_x = ct.sum(x_masked, axis=1)     # (XBLOCK,)
    sum_prod = ct.sum(prod_masked, axis=1)  # (XBLOCK,)

    ct.scatter(partial_sum_ptr, xindex_1d, sum_x, mask=xmask_1d)
    ct.scatter(partial_prod_ptr, xindex_1d, sum_prod, mask=xmask_1d)


@ct.kernel
def _finalize_partials_kernel(
    partial_prod_ptr,   # f32 stride (1, C) — flat (C * REDUCE_PARTIALS,)
    partial_sum_ptr,    # f32 stride (1, C)
    invstd_ptr,         # f32 (C,)
    affine_weight_ptr,  # f32 (C,)
    sum_out_ptr,        # f32 (C,)
    vector_out_ptr,     # f32 (C,)
    mean_term_ptr,      # f32 (C,)
    prod_coeff_ptr,     # f32 (C,)
    output_scale_ptr,   # f32 (C,)
    XBLOCK: ct.Constant[int],
    RBLOCK: ct.Constant[int],
):
    c_tile = ct.bid(0)
    c_1d = c_tile * XBLOCK + ct.arange(XBLOCK, dtype=ct.int32)
    r_1d = ct.arange(RBLOCK, dtype=ct.int32)
    c = ct.reshape(c_1d, (XBLOCK, 1))
    r = ct.reshape(r_1d, (1, RBLOCK))
    active = (c < C) & (r < REDUCE_PARTIALS)
    offsets = c + C * r

    prod_values = ct.gather(partial_prod_ptr, offsets)
    sum_values = ct.gather(partial_sum_ptr, offsets)
    pv = ct.where(active, prod_values, 0.0)
    sv = ct.where(active, sum_values, 0.0)

    sum_prod = ct.sum(pv, axis=1)  # (XBLOCK,)
    sum_x = ct.sum(sv, axis=1)     # (XBLOCK,)

    cmask = c_1d < C
    # For c_tile=0 and XBLOCK=8, c_1d = 0..7 or 8..15 etc. Load with index=(c_tile,).
    invstd = ct.load(invstd_ptr, index=(c_tile,), shape=(XBLOCK,),
                     padding_mode=ct.PaddingMode.ZERO)
    affine_weight = ct.load(affine_weight_ptr, index=(c_tile,), shape=(XBLOCK,),
                            padding_mode=ct.PaddingMode.ZERO)

    mean_term = sum_x * REDUCE_SCALE
    prod_coeff = (sum_prod * REDUCE_SCALE) * (invstd * invstd)
    output_scale = invstd * affine_weight

    ct.scatter(sum_out_ptr, c_1d, sum_x, mask=cmask)
    ct.scatter(vector_out_ptr, c_1d, sum_prod * invstd, mask=cmask)
    ct.scatter(mean_term_ptr, c_1d, mean_term, mask=cmask)
    ct.scatter(prod_coeff_ptr, c_1d, prod_coeff, mask=cmask)
    ct.scatter(output_scale_ptr, c_1d, output_scale, mask=cmask)


@ct.kernel
def _epilogue_kernel(
    copy_ptr,           # bf16 flat channels-last storage (C_IN=80 channels)
    activation_ptr,     # bf16 [N, C, H, W] channels-last flat
    mean_ptr,           # f32 (C,)
    mean_term_ptr,      # f32 (C,)
    prod_coeff_ptr,     # f32 (C,)
    output_scale_ptr,   # f32 (C,)
    out_ptr,            # bf16 flat channels-last (C=40 channels)
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    linear_1d = pid * BLOCK + ct.arange(BLOCK, dtype=ct.int32)
    active = linear_1d < OUT_NUMEL
    c_1d = linear_1d - (linear_1d // C) * C
    nhw = linear_1d // C
    hw = nhw - (nhw // HW) * HW
    n = nhw // HW
    wide_offsets = n * (C_IN * HW) + hw * C_IN + SLICE_START + c_1d

    x = ct.astype(ct.gather(copy_ptr, wide_offsets), ct.float32)
    centered = ct.astype(ct.gather(activation_ptr, linear_1d), ct.float32) - \
               ct.gather(mean_ptr, c_1d)
    prod_coeff = ct.gather(prod_coeff_ptr, c_1d)
    mean_term = ct.gather(mean_term_ptr, c_1d)
    output_scale = ct.gather(output_scale_ptr, c_1d)

    out = (x - centered * prod_coeff - mean_term) * output_scale
    out_bf = ct.astype(out, ct.bfloat16)
    ct.scatter(out_ptr, linear_1d, out_bf, mask=active)


@oracle_impl(hardware="B200", point="ad0f8f2b")
def oracle_forward(inputs):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1 = inputs
    device = arg0_1.device

    clone = torch.empty_strided(
        (N, C_IN, H, W), (C_IN * HW, HW, W, 1),
        device=device, dtype=torch.bfloat16,
    )
    copy = torch.empty_strided(
        (N, C_IN, H, W), (C_IN * HW, 1, W * C_IN, C_IN),
        device=device, dtype=torch.bfloat16,
    )
    sum_out = torch.empty((C,), device=device, dtype=torch.float32)
    vector_out = torch.empty((C,), device=device, dtype=torch.float32)
    mean_term = torch.empty((C,), device=device, dtype=torch.float32)
    prod_coeff = torch.empty((C,), device=device, dtype=torch.float32)
    output_scale = torch.empty((C,), device=device, dtype=torch.float32)
    partial_prod = torch.empty_strided(
        (C, REDUCE_PARTIALS), (1, C), device=device, dtype=torch.float32,
    )
    partial_sum = torch.empty_strided(
        (C, REDUCE_PARTIALS), (1, C), device=device, dtype=torch.float32,
    )
    out = torch.empty_strided(
        (N, C, H, W), (C * HW, 1, W * C, C),
        device=device, dtype=torch.bfloat16,
    )

    # Physical flat storage views (raw memory, agnostic of stride).
    def _phys_flat(t):
        return t.as_strided((t.numel(),), (1,))

    arg0_flat = _phys_flat(arg0_1)
    arg1_flat = _phys_flat(arg1_1)
    clone_flat = _phys_flat(clone)
    copy_flat = _phys_flat(copy)
    arg2_flat = _phys_flat(arg2_1)
    out_flat = _phys_flat(out)

    ADD_YBLOCK = 16
    ADD_XBLOCK = 256
    PARTIAL_XBLOCK = 16
    FINAL_XBLOCK = 8
    EPILOGUE_BLOCK = 512

    def _cdiv(a, b): return (a + b - 1) // b

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (_cdiv(HW, ADD_XBLOCK), _cdiv(N * C_IN, ADD_YBLOCK), 1),
        _add_clone_copy_kernel,
        (arg0_flat, arg1_flat, clone_flat, copy_flat, ADD_YBLOCK, ADD_XBLOCK),
    )

    partial_prod_flat = _phys_flat(partial_prod)
    partial_sum_flat = _phys_flat(partial_sum)

    ct.launch(
        stream, (_cdiv(C * REDUCE_PARTIALS, PARTIAL_XBLOCK), 1, 1),
        _partial_reduce_kernel,
        (copy_flat, arg2_flat, arg3_1.view(C).contiguous(),
         partial_prod_flat, partial_sum_flat,
         PARTIAL_XBLOCK, REDUCE_R),
    )
    ct.launch(
        stream, (_cdiv(C, FINAL_XBLOCK), 1, 1),
        _finalize_partials_kernel,
        (partial_prod_flat, partial_sum_flat,
         arg4_1.contiguous(), arg5_1.contiguous(),
         sum_out, vector_out, mean_term, prod_coeff, output_scale,
         FINAL_XBLOCK, _next_pow2(REDUCE_PARTIALS)),
    )
    ct.launch(
        stream, (_cdiv(OUT_NUMEL, EPILOGUE_BLOCK), 1, 1),
        _epilogue_kernel,
        (copy_flat, arg2_flat, arg3_1.view(C).contiguous(),
         mean_term, prod_coeff, output_scale,
         out_flat, EPILOGUE_BLOCK),
    )

    return clone, copy, sum_out, vector_out, out
