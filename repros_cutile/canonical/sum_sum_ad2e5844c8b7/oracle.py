"""cuTile port of sum_sum_ad2e5844c8b7: DenseNet BN-backward.

Similar pipeline to sum_sum_5fd0fb59ba5c but contiguous-strided and with an
extra output = (arg0+arg1+arg2)[:, 896:928] + out[:, 896:928].
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _partial_reduce_kernel(
    a_add_ptr,       # f32 [BATCH, CHANNELS, H*W]  where(mask, full, arg5)
    a_center_ptr,    # f32 [BATCH, CHANNELS, H*W]  arg6 - mean
    partial_sum_ptr, # f32 [num_chunks_n, CHANNELS]
    partial_prod_ptr, # f32 [num_chunks_n, CHANNELS]
    BATCH_: ct.Constant[int],
    HW_: ct.Constant[int],
    BLOCK_HW: ct.Constant[int],
):
    channel = ct.bid(0)
    r_block = ct.bid(1)

    # Load [BATCH, BLOCK_HW] for this channel
    a = ct.load(a_add_ptr, index=(0, channel, r_block), shape=(BATCH_, 1, BLOCK_HW),
                padding_mode=ct.PaddingMode.ZERO)
    b = ct.load(a_center_ptr, index=(0, channel, r_block), shape=(BATCH_, 1, BLOCK_HW),
                padding_mode=ct.PaddingMode.ZERO)
    # Mask beyond hw
    hw_offs = ct.arange(BLOCK_HW, dtype=ct.int32) + r_block * BLOCK_HW
    valid_hw_1d = hw_offs < HW_
    valid_hw = ct.reshape(valid_hw_1d, (1, 1, BLOCK_HW))
    zero_a = ct.full((BATCH_, 1, BLOCK_HW), 0.0, dtype=ct.float32)
    a = ct.where(valid_hw, a, zero_a)
    b = ct.where(valid_hw, b, zero_a)
    prod = a * b

    s_sum = ct.sum(a)
    s_prod = ct.sum(prod)
    s_sum_1 = ct.reshape(s_sum, (1, 1))
    s_prod_1 = ct.reshape(s_prod, (1, 1))
    ct.store(partial_sum_ptr, index=(r_block, channel), tile=s_sum_1)
    ct.store(partial_prod_ptr, index=(r_block, channel), tile=s_prod_1)


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
    CHANNELS_: ct.Constant[int],
    REDUCE_SCALE_: ct.Constant[float],
):
    channel = ct.bid(0)
    p_sum = ct.load(partial_sum_ptr, index=(0, channel), shape=(BLOCK_CHUNKS, 1),
                    padding_mode=ct.PaddingMode.ZERO)
    p_prod = ct.load(partial_prod_ptr, index=(0, channel), shape=(BLOCK_CHUNKS, 1),
                     padding_mode=ct.PaddingMode.ZERO)
    offs = ct.arange(BLOCK_CHUNKS, dtype=ct.int32)
    valid_1d = offs < NUM_CHUNKS
    valid = ct.reshape(valid_1d, (BLOCK_CHUNKS, 1))
    zero_f = ct.full((BLOCK_CHUNKS, 1), 0.0, dtype=ct.float32)
    p_sum = ct.where(valid, p_sum, zero_f)
    p_prod = ct.where(valid, p_prod, zero_f)

    sum_v = ct.sum(p_sum)
    prod_v = ct.sum(p_prod)
    invstd = ct.load(invstd_ptr, index=(channel,), shape=(1,))
    affine = ct.load(affine_ptr, index=(channel,), shape=(1,))

    sum_1 = ct.reshape(sum_v, (1,))
    prod_1 = ct.reshape(prod_v, (1,))
    mean_term = sum_1 * REDUCE_SCALE_
    prod_scaled = prod_1 * REDUCE_SCALE_
    invstd_sq = invstd * invstd
    prod_coeff = prod_scaled * invstd_sq
    output_scale = invstd * affine
    scale_grad = prod_1 * invstd

    ct.store(sum_out_ptr, index=(channel,), tile=sum_1)
    ct.store(mean_term_ptr, index=(channel,), tile=mean_term)
    ct.store(prod_coeff_ptr, index=(channel,), tile=prod_coeff)
    ct.store(output_scale_ptr, index=(channel,), tile=output_scale)
    ct.store(scale_grad_ptr, index=(channel,), tile=scale_grad)


@ct.kernel
def _epilogue_kernel(
    a_add_ptr,       # f32 [BATCH, CHANNELS, HW]
    a_center_ptr,    # f32 [BATCH, CHANNELS, HW]
    mean_term_ptr,
    prod_coeff_ptr,
    output_scale_ptr,
    out_ptr,         # bf16 [BATCH, CHANNELS, HW]
    HW_: ct.Constant[int],
    BLOCK_HW: ct.Constant[int],
):
    channel = ct.bid(0)
    n = ct.bid(1)
    r_block = ct.bid(2)

    a = ct.load(a_add_ptr, index=(n, channel, r_block), shape=(1, 1, BLOCK_HW),
                padding_mode=ct.PaddingMode.ZERO)
    centered = ct.load(a_center_ptr, index=(n, channel, r_block), shape=(1, 1, BLOCK_HW),
                       padding_mode=ct.PaddingMode.ZERO)
    mt = ct.load(mean_term_ptr, index=(channel,), shape=(1,))
    pc = ct.load(prod_coeff_ptr, index=(channel,), shape=(1,))
    os = ct.load(output_scale_ptr, index=(channel,), shape=(1,))
    mt3 = ct.reshape(mt, (1, 1, 1))
    pc3 = ct.reshape(pc, (1, 1, 1))
    os3 = ct.reshape(os, (1, 1, 1))
    correction = centered * pc3
    residual = a - correction - mt3
    out = residual * os3
    ct.store(out_ptr, index=(n, channel, r_block),
             tile=ct.astype(out, ct.bfloat16))


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p *= 2
    return p


@oracle_impl(hardware="B200", point="b28a5714", HW=196, BLOCK_HW=256)
@oracle_impl(hardware="B200", point="702b6b63", HW=49, BLOCK_HW=64)
def oracle_forward(inputs, *, HW: int, BLOCK_HW: int):
    (arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1,
     arg8_1, arg9_1) = inputs
    device = arg0_1.device

    batch = int(arg3_1.shape[0])
    channels = int(arg3_1.shape[1])
    h = int(arg3_1.shape[2])
    w = int(arg3_1.shape[3])
    hw = h * w
    # The repro hard-codes the reduce scale from the CAPTURE-time shape;
    # it does not recompute it from the actual runtime shape. Use the
    # captured constant.
    reduce_scale = 0.0012755102040816326

    num_chunks_hw = (hw + BLOCK_HW - 1) // BLOCK_HW
    hw_pad = num_chunks_hw * BLOCK_HW  # padded HW = multiple of BLOCK_HW

    # Precompute where(arg3<=0, arg4, arg5) → f32, pad HW dim to hw_pad.
    le_ = arg3_1 <= 0
    where_ = torch.where(le_, arg4_1, arg5_1)
    a_add_raw = where_.to(torch.float32).contiguous().view(batch, channels, hw)
    a_add = torch.zeros((batch, channels, hw_pad), device=device, dtype=torch.float32)
    a_add[:, :, :hw].copy_(a_add_raw)

    # a_center = arg6 - arg7 in fp32
    sub_ = arg6_1.to(torch.float32) - arg7_1
    a_center_raw = sub_.contiguous().view(batch, channels, hw)
    a_center = torch.zeros((batch, channels, hw_pad), device=device, dtype=torch.float32)
    a_center[:, :, :hw].copy_(a_center_raw)

    # partial_sum: [num_chunks_hw, CHANNELS] — each block reduces over BATCH*BLOCK_HW.
    partial_sum = torch.empty((num_chunks_hw, channels), device=device, dtype=torch.float32)
    partial_prod = torch.empty((num_chunks_hw, channels), device=device, dtype=torch.float32)
    block_chunks = _next_pow2(num_chunks_hw)

    sum_out = torch.empty((channels,), device=device, dtype=torch.float32)
    mean_term = torch.empty((channels,), device=device, dtype=torch.float32)
    prod_coeff = torch.empty((channels,), device=device, dtype=torch.float32)
    output_scale = torch.empty((channels,), device=device, dtype=torch.float32)
    scale_grad = torch.empty((channels,), device=device, dtype=torch.float32)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (channels, num_chunks_hw, 1),
        _partial_reduce_kernel,
        (a_add, a_center, partial_sum, partial_prod,
         batch, hw, BLOCK_HW),
    )
    ct.launch(
        stream,
        (channels, 1, 1),
        _finalize_kernel,
        (partial_sum, partial_prod, arg8_1, arg9_1,
         sum_out, mean_term, prod_coeff, output_scale, scale_grad,
         num_chunks_hw, block_chunks, channels, reduce_scale),
    )

    out_pad = torch.empty((batch, channels, hw_pad), device=device, dtype=torch.bfloat16)
    ct.launch(
        stream,
        (channels, batch, num_chunks_hw),
        _epilogue_kernel,
        (a_add, a_center, mean_term, prod_coeff, output_scale, out_pad,
         hw, BLOCK_HW),
    )
    out = out_pad[:, :, :hw].contiguous()
    out_4d = out.view(batch, channels, h, w)

    # Compute the extra "add_2" output:
    # add_2 = arg0[:,896:928] + arg1[:,896:928] + arg2[:,896:928] + out[:,896:928]
    tail = out_4d[:, 896:928]
    ex = (arg0_1[:, 896:928] + arg1_1[:, 896:928] +
          arg2_1[:, 896:928] + tail)

    return sum_out, scale_grad, out_4d, ex
