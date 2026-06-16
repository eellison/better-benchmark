"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the complete Visformer bf16 BN-backward-style scope by reducing the channel-last activation and gradient once into shared `sum(x)` and `sum(x * grad)` channel summaries, using those summaries for both returned vectors, the full fp32 residual epilogue, and the final bf16 cast with the same output layout as the residual input, whereas Inductor schedules the reductions, channel-vector epilogue, full-tensor epilogue, and bf16 conversion through generic reduction and pointwise kernels over the captured strided tensors; Inductor cannot do this today because its scheduler/codegen has no cooperative split-K multi-output reduction template that treats NHWC physical reductions, dependent channel summaries, and required materialized side outputs as one coordinated plan; the fix is COOPERATIVE_SPLIT_K: add a channel-reduction template that emits shared partial summaries and fuses all dependent vector and tensor epilogues under the point's stride guards."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


INV_REDUCE = 9.964923469387754e-06


@triton.jit
def _partial_channel_sums_kernel(
    x_ptr,
    grad_ptr,
    partial_ptr,
    N_: tl.constexpr,
    C_: tl.constexpr,
    H_: tl.constexpr,
    W_: tl.constexpr,
    XS0: tl.constexpr,
    XS1: tl.constexpr,
    XS2: tl.constexpr,
    XS3: tl.constexpr,
    GS0: tl.constexpr,
    GS1: tl.constexpr,
    GS2: tl.constexpr,
    GS3: tl.constexpr,
    REDUCE_ELEMS: tl.constexpr,
    NUM_TILES: tl.constexpr,
    RBLOCK: tl.constexpr,
    CBLOCK: tl.constexpr,
):
    c = tl.program_id(0) * CBLOCK + tl.arange(0, CBLOCK)
    r = tl.program_id(1) * RBLOCK + tl.arange(0, RBLOCK)
    channel_mask = c < C_
    reduce_mask = r < REDUCE_ELEMS

    hw = r % (H_ * W_)
    n = r // (H_ * W_)
    h = hw // W_
    w = hw % W_

    x_offsets = (
        n[:, None] * XS0
        + c[None, :] * XS1
        + h[:, None] * XS2
        + w[:, None] * XS3
    )
    grad_offsets = (
        n[:, None] * GS0
        + c[None, :] * GS1
        + h[:, None] * GS2
        + w[:, None] * GS3
    )
    mask = reduce_mask[:, None] & channel_mask[None, :]

    x = tl.load(x_ptr + x_offsets, mask=mask, other=0.0).to(tl.float32)
    grad = tl.load(grad_ptr + grad_offsets, mask=mask, other=0.0).to(tl.float32)

    tile = tl.program_id(1)
    partial_offsets = tile * C_ + c
    plane_stride = NUM_TILES * C_
    tl.store(
        partial_ptr + partial_offsets,
        tl.sum(tl.where(mask, x, 0.0), axis=0),
        mask=channel_mask,
    )
    tl.store(
        partial_ptr + plane_stride + partial_offsets,
        tl.sum(tl.where(mask, x * grad, 0.0), axis=0),
        mask=channel_mask,
    )


@triton.jit
def _finalize_channel_sums_kernel(
    partial_ptr,
    gamma_ptr,
    stats_ptr,
    out_sum_x_ptr,
    out_sum_xgrad_gamma_ptr,
    C_: tl.constexpr,
    NUM_TILES: tl.constexpr,
    TBLOCK: tl.constexpr,
    CBLOCK: tl.constexpr,
):
    c = tl.program_id(0) * CBLOCK + tl.arange(0, CBLOCK)
    tiles = tl.arange(0, TBLOCK)
    channel_mask = c < C_
    tile_mask = tiles < NUM_TILES
    mask = tile_mask[:, None] & channel_mask[None, :]
    offsets = tiles[:, None] * C_ + c[None, :]
    plane_stride = NUM_TILES * C_

    sum_x = tl.sum(
        tl.load(partial_ptr + offsets, mask=mask, other=0.0).to(tl.float32),
        axis=0,
    )
    sum_xgrad = tl.sum(
        tl.load(partial_ptr + plane_stride + offsets, mask=mask, other=0.0).to(
            tl.float32
        ),
        axis=0,
    )
    gamma = tl.load(gamma_ptr + c, mask=channel_mask, other=0.0).to(tl.float32)

    tl.store(stats_ptr + c, sum_x, mask=channel_mask)
    tl.store(stats_ptr + C_ + c, sum_xgrad, mask=channel_mask)
    tl.store(out_sum_x_ptr + c, sum_x, mask=channel_mask)
    tl.store(out_sum_xgrad_gamma_ptr + c, sum_xgrad * gamma, mask=channel_mask)


@triton.jit
def _epilogue_kernel(
    x_ptr,
    grad_ptr,
    gamma_ptr,
    beta_ptr,
    residual_ptr,
    stats_ptr,
    add_out_ptr,
    bf16_out_ptr,
    TOTAL: tl.constexpr,
    C_: tl.constexpr,
    H_: tl.constexpr,
    W_: tl.constexpr,
    XS0: tl.constexpr,
    XS1: tl.constexpr,
    XS2: tl.constexpr,
    XS3: tl.constexpr,
    GS0: tl.constexpr,
    GS1: tl.constexpr,
    GS2: tl.constexpr,
    GS3: tl.constexpr,
    OS0: tl.constexpr,
    OS1: tl.constexpr,
    OS2: tl.constexpr,
    OS3: tl.constexpr,
    INV_REDUCE_: tl.constexpr,
    BLOCK: tl.constexpr,
):
    linear = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = linear < TOTAL

    c = linear % C_
    tmp = linear // C_
    w = tmp % W_
    tmp = tmp // W_
    h = tmp % H_
    n = tmp // H_

    x_offsets = n * XS0 + c * XS1 + h * XS2 + w * XS3
    grad_offsets = n * GS0 + c * GS1 + h * GS2 + w * GS3
    out_offsets = n * OS0 + c * OS1 + h * OS2 + w * OS3

    x = tl.load(x_ptr + x_offsets, mask=mask, other=0.0).to(tl.float32)
    grad = tl.load(grad_ptr + grad_offsets, mask=mask, other=0.0).to(tl.float32)
    residual = tl.load(residual_ptr + out_offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    gamma = tl.load(gamma_ptr + c, mask=mask, other=0.0).to(tl.float32)
    beta = tl.load(beta_ptr + c, mask=mask, other=0.0).to(tl.float32)
    sum_x = tl.load(stats_ptr + c, mask=mask, other=0.0).to(tl.float32)
    sum_xgrad = tl.load(stats_ptr + C_ + c, mask=mask, other=0.0).to(tl.float32)

    mean_x = sum_x * INV_REDUCE_
    mean_xgrad = sum_xgrad * INV_REDUCE_
    corrected = x - grad * (mean_xgrad * gamma * gamma)
    corrected = corrected - mean_x
    add_value = residual + corrected * (gamma * beta)

    tl.store(add_out_ptr + out_offsets, add_value, mask=mask)
    tl.store(bf16_out_ptr + out_offsets, add_value.to(tl.bfloat16), mask=mask)


@triton.jit
def _epilogue_nchw_kernel(
    x_ptr,
    grad_ptr,
    gamma_ptr,
    beta_ptr,
    residual_ptr,
    stats_ptr,
    add_out_ptr,
    bf16_out_ptr,
    TOTAL: tl.constexpr,
    C_: tl.constexpr,
    H_: tl.constexpr,
    W_: tl.constexpr,
    XS0: tl.constexpr,
    XS1: tl.constexpr,
    XS2: tl.constexpr,
    XS3: tl.constexpr,
    GS0: tl.constexpr,
    GS1: tl.constexpr,
    GS2: tl.constexpr,
    GS3: tl.constexpr,
    OS0: tl.constexpr,
    OS1: tl.constexpr,
    OS2: tl.constexpr,
    OS3: tl.constexpr,
    INV_REDUCE_: tl.constexpr,
    BLOCK: tl.constexpr,
):
    linear = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = linear < TOTAL

    w = linear % W_
    tmp = linear // W_
    h = tmp % H_
    tmp = tmp // H_
    c = tmp % C_
    n = tmp // C_

    x_offsets = n * XS0 + c * XS1 + h * XS2 + w * XS3
    grad_offsets = n * GS0 + c * GS1 + h * GS2 + w * GS3
    out_offsets = n * OS0 + c * OS1 + h * OS2 + w * OS3

    x = tl.load(x_ptr + x_offsets, mask=mask, other=0.0).to(tl.float32)
    grad = tl.load(grad_ptr + grad_offsets, mask=mask, other=0.0).to(tl.float32)
    residual = tl.load(residual_ptr + out_offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    gamma = tl.load(gamma_ptr + c, mask=mask, other=0.0).to(tl.float32)
    beta = tl.load(beta_ptr + c, mask=mask, other=0.0).to(tl.float32)
    sum_x = tl.load(stats_ptr + c, mask=mask, other=0.0).to(tl.float32)
    sum_xgrad = tl.load(stats_ptr + C_ + c, mask=mask, other=0.0).to(tl.float32)

    mean_x = sum_x * INV_REDUCE_
    mean_xgrad = sum_xgrad * INV_REDUCE_
    corrected = x - grad * (mean_xgrad * gamma * gamma)
    corrected = corrected - mean_x
    add_value = residual + corrected * (gamma * beta)

    tl.store(add_out_ptr + out_offsets, add_value, mask=mask)
    tl.store(bf16_out_ptr + out_offsets, add_value.to(tl.bfloat16), mask=mask)


@oracle_impl(
    hardware="B200",
    point="11133330",
    RBLOCK=128,
    CBLOCK=64,
    FINAL_TBLOCK=1024,
    FINAL_CBLOCK=16,
    EPILOGUE_BLOCK=1024,
    reduce_warps=8,
    final_warps=8,
    epilogue_warps=4,
    nchw_epilogue=False,
)
@oracle_impl(
    hardware="B200",
    point="aa0e459f",
    RBLOCK=128,
    CBLOCK=64,
    FINAL_TBLOCK=256,
    FINAL_CBLOCK=16,
    EPILOGUE_BLOCK=1024,
    reduce_warps=8,
    final_warps=8,
    epilogue_warps=4,
    nchw_epilogue=False,
)
@oracle_impl(
    hardware="B200",
    point="2e5336a2",
    RBLOCK=128,
    CBLOCK=64,
    FINAL_TBLOCK=64,
    FINAL_CBLOCK=16,
    EPILOGUE_BLOCK=1024,
    reduce_warps=8,
    final_warps=4,
    epilogue_warps=4,
    nchw_epilogue=True,
)
def oracle_forward(
    inputs,
    *,
    RBLOCK,
    CBLOCK,
    FINAL_TBLOCK,
    FINAL_CBLOCK,
    EPILOGUE_BLOCK,
    reduce_warps,
    final_warps,
    epilogue_warps,
    nchw_epilogue,
):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1 = inputs
    n = int(arg0_1.shape[0])
    c = int(arg0_1.shape[1])
    h = int(arg0_1.shape[2])
    w = int(arg0_1.shape[3])
    reduce_elems = n * h * w
    total = n * c * h * w
    num_tiles = triton.cdiv(reduce_elems, RBLOCK)

    out_sum_x = torch.empty((c,), device=arg0_1.device, dtype=torch.float32)
    out_sum_xgrad_gamma = torch.empty((c,), device=arg0_1.device, dtype=torch.float32)
    stats = torch.empty((2, c), device=arg0_1.device, dtype=torch.float32)
    partials = torch.empty((2, num_tiles, c), device=arg0_1.device, dtype=torch.float32)
    add_out = torch.empty_strided(
        tuple(arg4_1.shape),
        tuple(arg4_1.stride()),
        device=arg4_1.device,
        dtype=torch.float32,
    )
    bf16_out = torch.empty_strided(
        tuple(arg4_1.shape),
        tuple(arg4_1.stride()),
        device=arg4_1.device,
        dtype=torch.bfloat16,
    )

    _partial_channel_sums_kernel[(triton.cdiv(c, CBLOCK), num_tiles)](
        arg0_1,
        arg1_1,
        partials,
        N_=n,
        C_=c,
        H_=h,
        W_=w,
        XS0=int(arg0_1.stride(0)),
        XS1=int(arg0_1.stride(1)),
        XS2=int(arg0_1.stride(2)),
        XS3=int(arg0_1.stride(3)),
        GS0=int(arg1_1.stride(0)),
        GS1=int(arg1_1.stride(1)),
        GS2=int(arg1_1.stride(2)),
        GS3=int(arg1_1.stride(3)),
        REDUCE_ELEMS=reduce_elems,
        NUM_TILES=num_tiles,
        RBLOCK=RBLOCK,
        CBLOCK=CBLOCK,
        num_warps=reduce_warps,
        num_stages=1,
    )
    _finalize_channel_sums_kernel[(triton.cdiv(c, FINAL_CBLOCK),)](
        partials,
        arg2_1,
        stats,
        out_sum_x,
        out_sum_xgrad_gamma,
        C_=c,
        NUM_TILES=num_tiles,
        TBLOCK=FINAL_TBLOCK,
        CBLOCK=FINAL_CBLOCK,
        num_warps=final_warps,
        num_stages=1,
    )
    epilogue = _epilogue_nchw_kernel if nchw_epilogue else _epilogue_kernel
    epilogue[(triton.cdiv(total, EPILOGUE_BLOCK),)](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        stats,
        add_out,
        bf16_out,
        TOTAL=total,
        C_=c,
        H_=h,
        W_=w,
        XS0=int(arg0_1.stride(0)),
        XS1=int(arg0_1.stride(1)),
        XS2=int(arg0_1.stride(2)),
        XS3=int(arg0_1.stride(3)),
        GS0=int(arg1_1.stride(0)),
        GS1=int(arg1_1.stride(1)),
        GS2=int(arg1_1.stride(2)),
        GS3=int(arg1_1.stride(3)),
        OS0=int(arg4_1.stride(0)),
        OS1=int(arg4_1.stride(1)),
        OS2=int(arg4_1.stride(2)),
        OS3=int(arg4_1.stride(3)),
        INV_REDUCE_=INV_REDUCE,
        BLOCK=EPILOGUE_BLOCK,
        num_warps=epilogue_warps,
        num_stages=1,
    )
    return out_sum_x, out_sum_xgrad_gamma, add_out, bf16_out
