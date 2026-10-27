"""cuTile port of sum_sum_e2c16e4d1008 (COOPERATIVE_SPLIT_K): Visformer BN
backward epilogue.

Strategy: mirror the Triton reference's 3-kernel structure — a per-tile
partial channel-reduction, a finalize kernel that reduces tile partials into
`sum_x`/`sum_xgrad` per channel (plus the `sum_xgrad * gamma` output vector),
and a dense elementwise BN-backward epilogue that emits fp32 `add_out` and
bf16 `bf16_out`.

Inputs arg0 (bf16) and arg1 (f32) are always channels-last (NHWC contiguous
in memory), so we permute/reshape them to a `(N*H*W, C)` contiguous 2D view
for the reductions. The epilogue keeps its stride-general gather/scatter so
it also handles the point where `residual` is not channels-last.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


INV_REDUCE = 9.964923469387754e-06


@ct.kernel
def _partial_reduce_kernel(
    x_ptr,             # bf16 view (R_TOTAL, C) contiguous
    grad_ptr,          # f32  view (R_TOTAL, C) contiguous
    partial_x_ptr,     # f32 [num_r_tiles, C]
    partial_xgrad_ptr, # f32 [num_r_tiles, C]
    R_TOTAL: ct.Constant[int],
    C_: ct.Constant[int],
    RBLOCK: ct.Constant[int],
    CBLOCK: ct.Constant[int],
):
    c_block = ct.bid(0)
    r_block = ct.bid(1)

    x_bf = ct.load(
        x_ptr, index=(r_block, c_block),
        shape=(RBLOCK, CBLOCK),
        padding_mode=ct.PaddingMode.ZERO,
    )
    x_f = ct.astype(x_bf, ct.float32)
    grad = ct.load(
        grad_ptr, index=(r_block, c_block),
        shape=(RBLOCK, CBLOCK),
        padding_mode=ct.PaddingMode.ZERO,
    )
    partial_x = ct.sum(x_f, axis=0)              # (CBLOCK,)
    partial_xgrad = ct.sum(x_f * grad, axis=0)   # (CBLOCK,)
    ct.store(partial_x_ptr, index=(r_block, c_block),
             tile=ct.reshape(partial_x, (1, CBLOCK)))
    ct.store(partial_xgrad_ptr, index=(r_block, c_block),
             tile=ct.reshape(partial_xgrad, (1, CBLOCK)))


@ct.kernel
def _finalize_kernel(
    partial_x_ptr,          # f32 [num_r_tiles, C]
    partial_xgrad_ptr,      # f32 [num_r_tiles, C]
    gamma_ptr,              # f32 [C]
    sum_x_out_ptr,          # f32 [C]  (raw sum_x)
    sum_xgrad_out_ptr,      # f32 [C]  (raw sum_xgrad, consumed by epilogue)
    sum_xgrad_gamma_out_ptr,# f32 [C]  (sum_xgrad * gamma, returned)
    NUM_TILES: ct.Constant[int],
    C_: ct.Constant[int],
    TBLOCK: ct.Constant[int],
    CBLOCK: ct.Constant[int],
):
    c_block = ct.bid(0)

    ps_x = ct.load(partial_x_ptr, index=(0, c_block),
                   shape=(TBLOCK, CBLOCK),
                   padding_mode=ct.PaddingMode.ZERO)
    ps_xgrad = ct.load(partial_xgrad_ptr, index=(0, c_block),
                       shape=(TBLOCK, CBLOCK),
                       padding_mode=ct.PaddingMode.ZERO)

    tile_local = ct.arange(TBLOCK, dtype=ct.int32)
    tile_valid = tile_local < NUM_TILES
    c_local = ct.arange(CBLOCK, dtype=ct.int32)
    c_valid = (c_block * CBLOCK + c_local) < C_
    mask = ct.reshape(tile_valid, (TBLOCK, 1)) & ct.reshape(c_valid, (1, CBLOCK))
    zero_2d = ct.zeros((TBLOCK, CBLOCK), dtype=ct.float32)
    ps_x_m = ct.where(mask, ps_x, zero_2d)
    ps_xgrad_m = ct.where(mask, ps_xgrad, zero_2d)
    sum_x = ct.sum(ps_x_m, axis=0)         # (CBLOCK,)
    sum_xgrad = ct.sum(ps_xgrad_m, axis=0) # (CBLOCK,)

    gamma_1d = ct.load(gamma_ptr, index=(c_block,),
                       shape=(CBLOCK,),
                       padding_mode=ct.PaddingMode.ZERO)

    ct.store(sum_x_out_ptr, index=(c_block,), tile=sum_x)
    ct.store(sum_xgrad_out_ptr, index=(c_block,), tile=sum_xgrad)
    ct.store(sum_xgrad_gamma_out_ptr, index=(c_block,), tile=sum_xgrad * gamma_1d)


@ct.kernel
def _epilogue_kernel(
    x_ptr,          # bf16 storage
    grad_ptr,       # f32 storage
    gamma_ptr,      # f32 [C]
    beta_ptr,       # f32 [C]
    residual_ptr,   # f32 storage
    sum_x_ptr,      # f32 [C]
    sum_xgrad_ptr,  # f32 [C]
    add_out_ptr,    # f32 storage
    bf16_out_ptr,   # bf16 storage
    TOTAL: ct.Constant[int],
    C: ct.Constant[int],
    XS0: ct.Constant[int],
    XS1: ct.Constant[int],
    XS2: ct.Constant[int],
    XS3: ct.Constant[int],
    OS0: ct.Constant[int],
    OS1: ct.Constant[int],
    OS2: ct.Constant[int],
    OS3: ct.Constant[int],
    H: ct.Constant[int],
    W: ct.Constant[int],
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    linear = pid * BLOCK + ct.arange(BLOCK, dtype=ct.int64)
    mask = linear < TOTAL
    zero64 = ct.zeros((BLOCK,), dtype=ct.int64)
    safe_lin = ct.where(mask, linear, zero64)

    c = safe_lin - (safe_lin // C) * C
    tmp = safe_lin // C
    w_ = tmp - (tmp // W) * W
    tmp = tmp // W
    h_ = tmp - (tmp // H) * H
    n_ = tmp // H
    x_off = n_ * XS0 + c * XS1 + h_ * XS2 + w_ * XS3
    o_off = n_ * OS0 + c * OS1 + h_ * OS2 + w_ * OS3
    safe_x = ct.where(mask, x_off, zero64)
    safe_o = ct.where(mask, o_off, zero64)

    x_bf = ct.gather(x_ptr, safe_x)
    x = ct.astype(x_bf, ct.float32)
    grad = ct.gather(grad_ptr, safe_x)
    residual = ct.gather(residual_ptr, safe_o)
    gamma = ct.gather(gamma_ptr, c)
    beta = ct.gather(beta_ptr, c)
    sum_x = ct.gather(sum_x_ptr, c)
    sum_xgrad = ct.gather(sum_xgrad_ptr, c)

    mean_x = sum_x * INV_REDUCE
    mean_xgrad = sum_xgrad * INV_REDUCE
    corrected = x - grad * (mean_xgrad * gamma * gamma)
    corrected = corrected - mean_x
    add_value = residual + corrected * (gamma * beta)

    ct.scatter(add_out_ptr, safe_o, add_value, mask=mask)
    ct.scatter(bf16_out_ptr, safe_o, ct.astype(add_value, ct.bfloat16), mask=mask)


@oracle_impl(
    hardware="B200", point="11133330",
    RBLOCK=128, CBLOCK=64,
    FINAL_TBLOCK=1024, FINAL_CBLOCK=16,
    EPILOGUE_BLOCK=1024,
)
@oracle_impl(
    hardware="B200", point="aa0e459f",
    RBLOCK=128, CBLOCK=64,
    FINAL_TBLOCK=256, FINAL_CBLOCK=16,
    EPILOGUE_BLOCK=1024,
)
@oracle_impl(
    hardware="B200", point="2e5336a2",
    RBLOCK=128, CBLOCK=64,
    FINAL_TBLOCK=64, FINAL_CBLOCK=16,
    EPILOGUE_BLOCK=1024,
)
def oracle_forward(
    inputs, *,
    RBLOCK, CBLOCK, FINAL_TBLOCK, FINAL_CBLOCK, EPILOGUE_BLOCK,
):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1 = inputs
    n = int(arg0_1.shape[0])
    c = int(arg0_1.shape[1])
    h = int(arg0_1.shape[2])
    w = int(arg0_1.shape[3])
    total = n * c * h * w
    r_total = n * h * w
    num_r_tiles = (r_total + RBLOCK - 1) // RBLOCK
    num_c_tiles = (c + CBLOCK - 1) // CBLOCK
    num_c_tiles_final = (c + FINAL_CBLOCK - 1) // FINAL_CBLOCK

    device = arg0_1.device

    # arg0/arg1 have NHWC-contiguous storage (channels-last strides), so
    # permute-then-reshape yields a contiguous (R_TOTAL, C) view for both.
    x_2d = arg0_1.permute(0, 2, 3, 1).reshape(r_total, c)
    grad_2d = arg1_1.permute(0, 2, 3, 1).reshape(r_total, c)

    partial_x = torch.empty((num_r_tiles, c), device=device, dtype=torch.float32)
    partial_xgrad = torch.empty((num_r_tiles, c), device=device, dtype=torch.float32)
    sum_x = torch.empty((c,), device=device, dtype=torch.float32)
    sum_xgrad = torch.empty((c,), device=device, dtype=torch.float32)
    out_sum_xgrad_gamma = torch.empty((c,), device=device, dtype=torch.float32)

    add_out = torch.empty_strided(
        tuple(arg4_1.shape), tuple(arg4_1.stride()),
        device=device, dtype=torch.float32,
    )
    bf16_out = torch.empty_strided(
        tuple(arg4_1.shape), tuple(arg4_1.stride()),
        device=device, dtype=torch.bfloat16,
    )

    stream = torch.cuda.current_stream()

    # 1) Partial per-tile channel reductions.
    ct.launch(
        stream, (num_c_tiles, num_r_tiles, 1), _partial_reduce_kernel,
        (x_2d, grad_2d, partial_x, partial_xgrad,
         r_total, c, RBLOCK, CBLOCK),
    )

    # 2) Finalize: sum tile partials into sum_x, sum_xgrad, sum_xgrad*gamma.
    ct.launch(
        stream, (num_c_tiles_final, 1, 1), _finalize_kernel,
        (partial_x, partial_xgrad, arg2_1,
         sum_x, sum_xgrad, out_sum_xgrad_gamma,
         num_r_tiles, c, FINAL_TBLOCK, FINAL_CBLOCK),
    )

    # 3) Dense elementwise epilogue over strided storage.
    def _flat_view(t):
        size = int(t.untyped_storage().nbytes() // t.element_size())
        return t.as_strided((size,), (1,))

    x_flat = _flat_view(arg0_1)
    grad_flat = _flat_view(arg1_1)
    residual_flat = _flat_view(arg4_1)
    add_flat = _flat_view(add_out)
    bf16_flat = _flat_view(bf16_out)

    ct.launch(
        stream,
        ((total + EPILOGUE_BLOCK - 1) // EPILOGUE_BLOCK, 1, 1),
        _epilogue_kernel,
        (x_flat, grad_flat, arg2_1, arg3_1, residual_flat,
         sum_x, sum_xgrad, add_flat, bf16_flat,
         total, c,
         int(arg0_1.stride(0)), int(arg0_1.stride(1)),
         int(arg0_1.stride(2)), int(arg0_1.stride(3)),
         int(arg4_1.stride(0)), int(arg4_1.stride(1)),
         int(arg4_1.stride(2)), int(arg4_1.stride(3)),
         h, w, EPILOGUE_BLOCK),
    )
    return sum_x, out_sum_xgrad_gamma, add_out, bf16_out
