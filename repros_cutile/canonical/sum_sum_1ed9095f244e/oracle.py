"""cuTile port of sum_sum_1ed9095f244e: MobileNetV2 adaptive-pool + BN-backward.

Uses cuTile kernels for the per-channel reduction (a stat sum) and the
per-element epilogue (BN backward output).
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 128
CHANNELS = 1280
HEIGHT = 7
WIDTH = 7
HW = 49
ELEMENTS_PER_CHANNEL = BATCH * HW
REDUCE_SCALE = 0.00015943877551020407  # 1 / (128*7*7)


@ct.kernel
def _partial_reduce_kernel(
    a_add_ptr,       # f32 [BATCH, HW, CHANNELS] = [rows, CHANNELS]  where(mask, 0, div)
    a_center_ptr,    # f32 [rows, CHANNELS]  arg1 - arg2
    partial_sum_ptr, # f32 [num_chunks, CHANNELS]
    partial_prod_ptr,
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
    prod = a * b
    s_sum = ct.sum(a, axis=0)
    s_prod = ct.sum(prod, axis=0)
    ct.store(partial_sum_ptr, index=(r_block, c_block),
             tile=ct.reshape(s_sum, (1, BLOCK_C)))
    ct.store(partial_prod_ptr, index=(r_block, c_block),
             tile=ct.reshape(s_prod, (1, BLOCK_C)))


@ct.kernel
def _finalize_kernel(
    partial_sum_ptr,
    partial_prod_ptr,
    invstd_ptr,      # f32 [C]  (from squeeze_3 = arg3_1 [1,C,1,1] squeezed)
    affine_ptr,      # f32 [C]  arg4_1
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
    prod_coeff = prod_scaled * invstd * invstd
    output_scale = invstd * affine
    scale_grad = prod_v * invstd  # this is mul_10
    ct.store(sum_out_ptr, index=(c_block,), tile=sum_v)
    ct.store(mean_term_ptr, index=(c_block,), tile=mean_term)
    ct.store(prod_coeff_ptr, index=(c_block,), tile=prod_coeff)
    ct.store(output_scale_ptr, index=(c_block,), tile=output_scale)
    ct.store(scale_grad_ptr, index=(c_block,), tile=scale_grad)


@ct.kernel
def _epilogue_kernel(
    a_add_ptr,
    a_center_ptr,
    mean_term_ptr,
    prod_coeff_ptr,
    output_scale_ptr,
    out_ptr,
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
    mt = ct.load(mean_term_ptr, index=(c_block,), shape=(BLOCK_C,),
                 padding_mode=ct.PaddingMode.ZERO)
    pc = ct.load(prod_coeff_ptr, index=(c_block,), shape=(BLOCK_C,),
                 padding_mode=ct.PaddingMode.ZERO)
    os = ct.load(output_scale_ptr, index=(c_block,), shape=(BLOCK_C,),
                 padding_mode=ct.PaddingMode.ZERO)
    mt_2d = ct.reshape(mt, (1, BLOCK_C))
    pc_2d = ct.reshape(pc, (1, BLOCK_C))
    os_2d = ct.reshape(os, (1, BLOCK_C))
    correction = centered * pc_2d
    residual = a - correction - mt_2d
    out = residual * os_2d
    ct.store(out_ptr, index=(r_block, c_block), tile=ct.astype(out, ct.bfloat16))


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p *= 2
    return p


@oracle_impl(hardware="B200", point="a44186d7", BLOCK_R=512, BLOCK_C=16)
def oracle_forward(inputs, *, BLOCK_R: int, BLOCK_C: int):
    (arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1,
     shape0, _shape1, shape2, _shape3, _shape4, _shape5, _shape6) = inputs
    device = arg0_1.device
    rows = ELEMENTS_PER_CHANNEL
    num_chunks = (rows + BLOCK_R - 1) // BLOCK_R
    block_chunks = _next_pow2(num_chunks)

    # Step 1: build `div = expand(arg0.view(N,C,1,1), (N,C,H,W)) / 49` bf16
    div = (arg0_1.view(BATCH, CHANNELS, 1, 1)
           .expand(BATCH, CHANNELS, HEIGHT, WIDTH)
           .contiguous()) / 49
    # Actually: aten.div.Scalar does exact bf16 division. Use scalar div.
    div = arg0_1.view(BATCH, CHANNELS, 1, 1).expand(
        BATCH, CHANNELS, HEIGHT, WIDTH).contiguous()
    div = torch.ops.aten.div.Scalar(div, 49)

    # Step 2: mask = (bf16(f32_op) <= 0) | (>= 6)
    # (arg1 - arg2) * arg3 * unsqueeze_1(arg4) + unsqueeze_3(arg5) bf16
    sub_0 = arg1_1.to(torch.float32) - arg2_1
    mul_step = sub_0 * arg3_1
    # unsqueeze arg4[1280] -> [1280,1,1]
    arg4_re = arg4_1.view(CHANNELS, 1, 1)
    mul_1 = mul_step * arg4_re
    arg5_re = arg5_1.view(CHANNELS, 1, 1)
    add_val = mul_1 + arg5_re
    cet = add_val.to(torch.bfloat16)
    le_ = cet <= 0.0
    ge_ = cet >= 6.0
    mask = le_ | ge_
    zero_bf = torch.zeros((), dtype=torch.bfloat16, device=device)
    where_ = torch.where(mask, zero_bf, div)  # bf16 [128, 1280, 7, 7]

    # a_add (f32): where_
    a_add = where_.to(torch.float32)
    # a_center: arg1 - arg2 (already sub_0)
    a_center = sub_0

    # Reshape both to [rows, CHANNELS] via NHWC layout.
    a_add_2d = a_add.permute(0, 2, 3, 1).contiguous().view(rows, CHANNELS)
    a_center_2d = a_center.permute(0, 2, 3, 1).contiguous().view(rows, CHANNELS)

    partial_sum = torch.empty((num_chunks, CHANNELS), device=device, dtype=torch.float32)
    partial_prod = torch.empty((num_chunks, CHANNELS), device=device, dtype=torch.float32)
    sum_out = torch.empty((CHANNELS,), device=device, dtype=torch.float32)
    mean_term = torch.empty((CHANNELS,), device=device, dtype=torch.float32)
    prod_coeff = torch.empty((CHANNELS,), device=device, dtype=torch.float32)
    output_scale = torch.empty((CHANNELS,), device=device, dtype=torch.float32)
    scale_grad = torch.empty((CHANNELS,), device=device, dtype=torch.float32)

    invstd = arg3_1.view(CHANNELS).contiguous()  # squeeze_3
    stream = torch.cuda.current_stream()
    grid_c = (CHANNELS + BLOCK_C - 1) // BLOCK_C
    ct.launch(
        stream,
        (grid_c, num_chunks, 1),
        _partial_reduce_kernel,
        (a_add_2d, a_center_2d, partial_sum, partial_prod,
         rows, CHANNELS, BLOCK_R, BLOCK_C),
    )
    ct.launch(
        stream,
        (grid_c, 1, 1),
        _finalize_kernel,
        (partial_sum, partial_prod, invstd, arg4_1,
         sum_out, mean_term, prod_coeff, output_scale, scale_grad,
         num_chunks, block_chunks, BLOCK_C, REDUCE_SCALE, CHANNELS),
    )
    out_flat = torch.empty((rows, CHANNELS), device=device, dtype=torch.bfloat16)
    ct.launch(
        stream,
        (grid_c, num_chunks, 1),
        _epilogue_kernel,
        (a_add_2d, a_center_2d, mean_term, prod_coeff, output_scale, out_flat,
         BLOCK_R, BLOCK_C),
    )
    out_bhwc = out_flat.view(BATCH, HEIGHT, WIDTH, CHANNELS)
    out = out_bhwc.permute(0, 3, 1, 2).contiguous(memory_format=torch.channels_last)

    full_scalar = torch.zeros((), device=device, dtype=torch.bfloat16)
    return full_scalar, sum_out, scale_grad, out
