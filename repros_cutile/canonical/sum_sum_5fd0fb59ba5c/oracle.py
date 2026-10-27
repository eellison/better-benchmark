"""cuTile port of sum_sum_5fd0fb59ba5c: GhostNet BN-backward masked slice-add.

Three-stage pipeline:
- per-channel per-chunk reductions of (a) where(arg2<=0, arg3, arg0[:,:240]+arg1)
  and (b) that * (arg4-mean).
- finalize per-channel sums into (sum_1, mul_8, and intermediates for epilogue).
- pointwise epilogue producing the bf16 [BATCH,CHANNELS,H,W] output.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 512
IN_CHANNELS = 480
CHANNELS = 240
HEIGHT = 14
WIDTH = 14
HW = HEIGHT * WIDTH
ELEMENTS_PER_CHANNEL = BATCH * HW  # rows per channel = 512*196 = 100352
REDUCE_SCALE = 9.964923469387754e-06


@ct.kernel
def _partial_reduce_kernel(
    a_add_ptr,       # bf16 [rows, CHANNELS]
    a_center_ptr,    # f32 [rows, CHANNELS]
    partial_sum_ptr, # f32 [num_chunks, CHANNELS]
    partial_prod_ptr, # f32 [num_chunks, CHANNELS]
    ROWS: ct.Constant[int],
    CHANNELS_: ct.Constant[int],
    BLOCK_R: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    c_block = ct.bid(0)
    r_block = ct.bid(1)

    a = ct.load(a_add_ptr, index=(r_block, c_block), shape=(BLOCK_R, BLOCK_C),
                padding_mode=ct.PaddingMode.ZERO)
    b = ct.load(a_center_ptr, index=(r_block, c_block), shape=(BLOCK_R, BLOCK_C),
                padding_mode=ct.PaddingMode.ZERO)
    a_f = ct.astype(a, ct.float32)
    prod = a_f * b

    s_sum = ct.sum(a_f, axis=0)  # (BLOCK_C,)
    s_prod = ct.sum(prod, axis=0)
    ct.store(partial_sum_ptr, index=(r_block, c_block),
             tile=ct.reshape(s_sum, (1, BLOCK_C)))
    ct.store(partial_prod_ptr, index=(r_block, c_block),
             tile=ct.reshape(s_prod, (1, BLOCK_C)))


@ct.kernel
def _finalize_kernel(
    partial_sum_ptr,
    partial_prod_ptr,
    invstd_ptr,
    affine_ptr,
    sum_out_ptr,
    mean_term_ptr,
    prod_coeff_ptr,
    output_scale_ptr,
    scale_grad_ptr,
    NUM_CHUNKS: ct.Constant[int],
    BLOCK_CHUNKS: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
    REDUCE_SCALE_: ct.Constant[float],
    CHANNELS_: ct.Constant[int],
):
    c_block = ct.bid(0)
    p_sum = ct.load(partial_sum_ptr, index=(0, c_block),
                    shape=(BLOCK_CHUNKS, BLOCK_C),
                    padding_mode=ct.PaddingMode.ZERO)
    p_prod = ct.load(partial_prod_ptr, index=(0, c_block),
                     shape=(BLOCK_CHUNKS, BLOCK_C),
                     padding_mode=ct.PaddingMode.ZERO)
    offs = ct.arange(BLOCK_CHUNKS, dtype=ct.int32)
    valid_1d = offs < NUM_CHUNKS
    valid = ct.reshape(valid_1d, (BLOCK_CHUNKS, 1))
    zero_f = ct.full((BLOCK_CHUNKS, BLOCK_C), 0.0, dtype=ct.float32)
    p_sum = ct.where(valid, p_sum, zero_f)
    p_prod = ct.where(valid, p_prod, zero_f)

    sum_v = ct.sum(p_sum, axis=0)
    prod_v = ct.sum(p_prod, axis=0)

    invstd = ct.load(invstd_ptr, index=(c_block,), shape=(BLOCK_C,),
                     padding_mode=ct.PaddingMode.ZERO)
    affine = ct.load(affine_ptr, index=(c_block,), shape=(BLOCK_C,),
                     padding_mode=ct.PaddingMode.ZERO)
    mean_term = sum_v * REDUCE_SCALE_
    prod_scaled = prod_v * REDUCE_SCALE_
    invstd_sq = invstd * invstd
    prod_coeff = prod_scaled * invstd_sq
    output_scale = invstd * affine
    scale_grad = prod_v * invstd

    ct.store(sum_out_ptr, index=(c_block,), tile=sum_v)
    ct.store(mean_term_ptr, index=(c_block,), tile=mean_term)
    ct.store(prod_coeff_ptr, index=(c_block,), tile=prod_coeff)
    ct.store(output_scale_ptr, index=(c_block,), tile=output_scale)
    ct.store(scale_grad_ptr, index=(c_block,), tile=scale_grad)


@ct.kernel
def _epilogue_kernel(
    a_add_ptr,       # bf16 [rows, CHANNELS]
    a_center_ptr,    # f32 [rows, CHANNELS]
    mean_term_ptr,
    prod_coeff_ptr,
    output_scale_ptr,
    out_ptr,         # bf16 [rows, CHANNELS]
    BLOCK_R: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    c_block = ct.bid(0)
    r_block = ct.bid(1)

    a = ct.load(a_add_ptr, index=(r_block, c_block), shape=(BLOCK_R, BLOCK_C),
                padding_mode=ct.PaddingMode.ZERO)
    centered = ct.load(a_center_ptr, index=(r_block, c_block),
                       shape=(BLOCK_R, BLOCK_C),
                       padding_mode=ct.PaddingMode.ZERO)
    a_f = ct.astype(a, ct.float32)

    mean_term = ct.load(mean_term_ptr, index=(c_block,), shape=(BLOCK_C,),
                        padding_mode=ct.PaddingMode.ZERO)
    prod_coeff = ct.load(prod_coeff_ptr, index=(c_block,), shape=(BLOCK_C,),
                         padding_mode=ct.PaddingMode.ZERO)
    output_scale = ct.load(output_scale_ptr, index=(c_block,), shape=(BLOCK_C,),
                           padding_mode=ct.PaddingMode.ZERO)
    mt_2d = ct.reshape(mean_term, (1, BLOCK_C))
    pc_2d = ct.reshape(prod_coeff, (1, BLOCK_C))
    os_2d = ct.reshape(output_scale, (1, BLOCK_C))

    correction = centered * pc_2d
    residual = a_f - correction
    residual = residual - mt_2d
    out = residual * os_2d
    ct.store(out_ptr, index=(r_block, c_block), tile=ct.astype(out, ct.bfloat16))


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p *= 2
    return p


@oracle_impl(hardware="B200", point="9c8c8e5a", BLOCK_R=512, BLOCK_C=16)
def oracle_forward(inputs, *, BLOCK_R: int, BLOCK_C: int):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1 = inputs
    device = arg0_1.device

    rows = ELEMENTS_PER_CHANNEL
    num_chunks = (rows + BLOCK_R - 1) // BLOCK_R
    block_chunks = _next_pow2(num_chunks)

    # Precompute where(arg2<=0, arg3, arg0[:,:240]+arg1) with channels-last layout.
    slice_1 = torch.ops.aten.slice.Tensor(arg0_1, 1, 0, 240)
    add_ = slice_1 + arg1_1  # bf16
    le_ = arg2_1 <= 0
    where_ = torch.where(le_, arg3_1, add_)
    # Channels-last stride means [n,c,h,w] with stride (C*H*W, 1, C*W, C);
    # reshape to (rows, CHANNELS) via bhwc permute.
    a_add = where_.permute(0, 2, 3, 1).contiguous().view(rows, CHANNELS)

    # a_center = arg4 - arg5 (broadcast) in fp32
    sub_ = arg4_1.to(torch.float32) - arg5_1
    a_center = sub_.permute(0, 2, 3, 1).contiguous().view(rows, CHANNELS)

    partial_sum = torch.empty((num_chunks, CHANNELS), device=device, dtype=torch.float32)
    partial_prod = torch.empty((num_chunks, CHANNELS), device=device, dtype=torch.float32)

    sum_out = torch.empty((CHANNELS,), device=device, dtype=torch.float32)
    mean_term = torch.empty((CHANNELS,), device=device, dtype=torch.float32)
    prod_coeff = torch.empty((CHANNELS,), device=device, dtype=torch.float32)
    output_scale = torch.empty((CHANNELS,), device=device, dtype=torch.float32)
    scale_grad = torch.empty((CHANNELS,), device=device, dtype=torch.float32)

    stream = torch.cuda.current_stream()
    grid_c = (CHANNELS + BLOCK_C - 1) // BLOCK_C
    ct.launch(
        stream,
        (grid_c, num_chunks, 1),
        _partial_reduce_kernel,
        (a_add, a_center, partial_sum, partial_prod,
         rows, CHANNELS, BLOCK_R, BLOCK_C),
    )
    ct.launch(
        stream,
        (grid_c, 1, 1),
        _finalize_kernel,
        (partial_sum, partial_prod, arg6_1, arg7_1,
         sum_out, mean_term, prod_coeff, output_scale, scale_grad,
         num_chunks, block_chunks, BLOCK_C, REDUCE_SCALE, CHANNELS),
    )

    out_flat = torch.empty((rows, CHANNELS), device=device, dtype=torch.bfloat16)
    ct.launch(
        stream,
        (grid_c, num_chunks, 1),
        _epilogue_kernel,
        (a_add, a_center, mean_term, prod_coeff, output_scale, out_flat,
         BLOCK_R, BLOCK_C),
    )

    out_bhwc = out_flat.view(BATCH, HEIGHT, WIDTH, CHANNELS)
    out = out_bhwc.permute(0, 3, 1, 2).contiguous(memory_format=torch.channels_last)
    return sum_out, scale_grad, out
