"""cuTile port of sum_sum_sum_c2ae956520f9: NFNet exact GELU-grad + pool-back
+ 4-kernel plan for output tuple. Uses precomputed torch.special.erf(gelu_x *
0.7071...) since cuTile lacks ct.erf.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


def _ceil_pow2(v: int) -> int:
    return 1 << (v - 1).bit_length()


@ct.kernel
def _pointwise_kernel(
    low_ptr,          # bf16 NHWC flat pool grad [N*C*PH*PW]
    add_ptr,          # bf16 NHWC flat [N*C*H*W]
    gelu_ptr,         # bf16 NHWC flat [N*C*H*W]
    gate_ptr,         # bf16 [N*C]
    rhs_ptr,          # bf16 NHWC flat [N*C*H*W]
    scalar_ptr,       # f32 scalar
    erf_ptr,          # f32 flat [N*C*H*W]  - precomputed erf(gelu * 0.7071)
    out_grad_ptr,     # bf16 NHWC flat
    out_scaled_ptr,   # bf16 NHWC flat
    scalar_partials_ptr,  # f32 [num_blocks]
    C_: ct.Constant[int],
    H_: ct.Constant[int],
    W_: ct.Constant[int],
    PH_: ct.Constant[int],
    PW_: ct.Constant[int],
    TOTAL_: ct.Constant[int],
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    offsets = pid * BLOCK + ct.arange(BLOCK, dtype=ct.int32)
    mask = offsets < TOTAL_
    c = offsets - (offsets // C_) * C_
    w = (offsets // C_) - ((offsets // C_) // W_) * W_
    h = (offsets // (C_ * W_)) - ((offsets // (C_ * W_)) // H_) * H_
    n = offsets // (C_ * H_ * W_)
    # low offsets in NHWC pool-down layout
    low_offsets = n * (C_ * PH_ * PW_) + ((h // 2) * PW_ + (w // 2)) * C_ + c

    pool_grad = ct.astype(ct.gather(low_ptr, (low_offsets,), mask=mask, padding_value=ct.bfloat16(0.0)), ct.float32)
    add_in = ct.astype(ct.gather(add_ptr, (offsets,), mask=mask, padding_value=ct.bfloat16(0.0)), ct.float32)
    gelu_x = ct.astype(ct.gather(gelu_ptr, (offsets,), mask=mask, padding_value=ct.bfloat16(0.0)), ct.float32)
    rhs = ct.astype(ct.gather(rhs_ptr, (offsets,), mask=mask, padding_value=ct.bfloat16(0.0)), ct.float32)
    gate_off = n * C_ + c
    gate = ct.astype(ct.gather(gate_ptr, (gate_off,), mask=mask, padding_value=ct.bfloat16(0.0)), ct.float32)
    scalar = ct.load(scalar_ptr, index=(0,), shape=(1,))
    # bf16 round scalar: to bf16 then f32
    scalar_bf = ct.astype(ct.astype(scalar, ct.bfloat16), ct.float32)

    pool_back_compiled = pool_grad / 4.0
    base_compiled = (add_in + pool_back_compiled) * 0.9622504486493761
    base_compiled = base_compiled * 1.7015043497085571

    add_bf = ct.astype(ct.astype(add_in + pool_back_compiled, ct.bfloat16), ct.float32)
    mul_bf = ct.astype(ct.astype(add_bf * 0.9622504486493761, ct.bfloat16), ct.float32)
    base_bf = ct.astype(ct.astype(mul_bf * 1.7015043497085571, ct.bfloat16), ct.float32)

    erf_val = ct.gather(erf_ptr, (offsets,), mask=mask, padding_value=ct.float32(0.0))
    cdf = (erf_val + 1.0) * 0.5
    pdf = ct.exp((gelu_x * gelu_x) * -0.5) * 0.3989422804014327
    gelu_grad = cdf + gelu_x * pdf

    grad_for_scalar = ct.astype(ct.astype(base_compiled * gelu_grad, ct.bfloat16), ct.float32)
    grad = ct.astype(ct.astype(base_bf * gelu_grad, ct.bfloat16), ct.float32)
    grad_bf16 = ct.astype(grad, ct.bfloat16)
    ct.scatter(out_grad_ptr, (offsets,), grad_bf16, mask=mask)

    mul9 = ct.astype(ct.astype(grad * 0.2, ct.bfloat16), ct.float32)
    scalar_bcast = ct.zeros((BLOCK,), dtype=ct.float32) + ct.reshape(scalar_bf, (1,))
    scaled = ct.astype(ct.astype(mul9 * scalar_bcast, ct.bfloat16), ct.float32)
    scaled = ct.astype(ct.astype(scaled * 2.0, ct.bfloat16), ct.float32)
    ct.scatter(out_scaled_ptr, (offsets,), ct.astype(scaled, ct.bfloat16), mask=mask)

    # scalar_terms = (grad_for_scalar * 0.2) * ((rhs * sigmoid) * 2.0)
    one = ct.full((BLOCK,), 1.0, dtype=ct.float32)
    sigmoid_f = 1.0 / (one + ct.exp(-gate))
    sigmoid_bf = ct.astype(ct.astype(sigmoid_f, ct.bfloat16), ct.float32)
    scalar_terms = (grad_for_scalar * 0.2) * ((rhs * sigmoid_bf) * 2.0)
    zero_f = ct.zeros((BLOCK,), dtype=ct.float32)
    partial = ct.sum(ct.where(mask, scalar_terms, zero_f))
    ct.store(scalar_partials_ptr, index=(pid,),
              tile=ct.reshape(ct.full((1,), 0.0, dtype=ct.float32) + partial, (1,)))


@ct.kernel
def _side_reduce_kernel(
    out_scaled_ptr,   # bf16 NHWC flat
    rhs_ptr,          # bf16 NHWC flat
    gate_ptr,         # bf16 [N*C]
    out_sigmoid_ptr,  # bf16 [N*C]
    out_side_ptr,     # bf16 [N*C]
    side_nc_ptr,      # f32 [N, C]
    C_: ct.Constant[int],
    HW_: ct.Constant[int],
    N_: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
    BLOCK_HW: ct.Constant[int],
):
    n = ct.bid(0)
    c_block = ct.bid(1)
    cs = c_block * BLOCK_C + ct.arange(BLOCK_C, dtype=ct.int32)
    c_mask = cs < C_

    nc_offsets = n * C_ + cs
    gate_val = ct.astype(ct.gather(gate_ptr, (nc_offsets,), mask=c_mask, padding_value=ct.bfloat16(0.0)), ct.float32)
    one_bc = ct.full((BLOCK_C,), 1.0, dtype=ct.float32)
    sigmoid_f = 1.0 / (one_bc + ct.exp(-gate_val))
    sigmoid_bf = ct.astype(sigmoid_f, ct.bfloat16)
    sigmoid = ct.astype(sigmoid_bf, ct.float32)
    ct.scatter(out_sigmoid_ptr, (nc_offsets,), sigmoid_bf, mask=c_mask)

    sum2 = ct.zeros((BLOCK_C,), dtype=ct.float32)

    # NHWC layout: offset(n, hw, c) = n * HW * C + hw * C + c
    zero_i32_2d = ct.zeros((BLOCK_C, BLOCK_HW), dtype=ct.int32)
    zero_bool_2d = ct.full((BLOCK_C, BLOCK_HW), False, dtype=ct.bool_)
    cs_2d = ct.reshape(cs, (BLOCK_C, 1)) + zero_i32_2d
    c_mask_2d = ct.reshape(c_mask, (BLOCK_C, 1)) | zero_bool_2d

    for hw_base in range(0, HW_, BLOCK_HW):
        hw = hw_base + ct.arange(BLOCK_HW, dtype=ct.int32)
        hw_mask = hw < HW_
        hw_mask_2d = ct.reshape(hw_mask, (1, BLOCK_HW)) | zero_bool_2d
        mask = c_mask_2d & hw_mask_2d
        hw_2d = ct.reshape(hw, (1, BLOCK_HW)) + zero_i32_2d
        offsets = n * (HW_ * C_) + hw_2d * C_ + cs_2d
        scaled = ct.astype(ct.gather(out_scaled_ptr, (offsets,), mask=mask, padding_value=ct.bfloat16(0.0)), ct.float32)
        rhs = ct.astype(ct.gather(rhs_ptr, (offsets,), mask=mask, padding_value=ct.bfloat16(0.0)), ct.float32)
        prod_bf = ct.astype(ct.astype(scaled * rhs, ct.bfloat16), ct.float32)
        zero_2d = ct.zeros((BLOCK_C, BLOCK_HW), dtype=ct.float32)
        sum2 = sum2 + ct.sum(ct.where(mask, prod_bf, zero_2d), axis=1)

    side = ct.astype(ct.astype(sum2, ct.bfloat16), ct.float32) * (sigmoid * (1.0 - sigmoid))
    side_bf16 = ct.astype(side, ct.bfloat16)
    ct.scatter(out_side_ptr, (nc_offsets,), side_bf16, mask=c_mask)
    ct.scatter(side_nc_ptr, (nc_offsets,), ct.astype(side_bf16, ct.float32), mask=c_mask)


@ct.kernel
def _finalize_channels_kernel(
    side_nc_ptr,  # f32 [N * C]
    out_channel_ptr,  # f32 [C]
    C_: ct.Constant[int],
    N_: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
):
    c_block = ct.bid(0)
    cs = c_block * BLOCK_C + ct.arange(BLOCK_C, dtype=ct.int32)
    c_mask = cs < C_

    ns = ct.arange(BLOCK_N, dtype=ct.int32)
    n_mask = ns < N_

    zero_i32_2d = ct.zeros((BLOCK_N, BLOCK_C), dtype=ct.int32)
    zero_bool_2d = ct.full((BLOCK_N, BLOCK_C), False, dtype=ct.bool_)
    n_mask_2d = ct.reshape(n_mask, (BLOCK_N, 1)) | zero_bool_2d
    c_mask_2d = ct.reshape(c_mask, (1, BLOCK_C)) | zero_bool_2d
    mask = n_mask_2d & c_mask_2d
    ns_2d = ct.reshape(ns, (BLOCK_N, 1)) + zero_i32_2d
    cs_2d = ct.reshape(cs, (1, BLOCK_C)) + zero_i32_2d
    offsets = ns_2d * C_ + cs_2d
    tile = ct.gather(side_nc_ptr, (offsets,), mask=mask, padding_value=ct.float32(0.0))
    channel = ct.astype(ct.astype(ct.sum(tile, axis=0), ct.bfloat16), ct.float32)
    ct.scatter(out_channel_ptr, (cs,), channel, mask=c_mask)


@ct.kernel
def _finalize_scalar_kernel(
    scalar_blocks_ptr,
    out_scalar_ptr,
    NUM_BLOCKS: ct.Constant[int],
    BLOCKS: ct.Constant[int],
):
    offsets = ct.arange(BLOCKS, dtype=ct.int32)
    mask = offsets < NUM_BLOCKS
    vals = ct.gather(scalar_blocks_ptr, (offsets,), mask=mask, padding_value=ct.float32(0.0))
    ct.store(out_scalar_ptr, index=(0,),
              tile=ct.reshape(ct.full((1,), 0.0, dtype=ct.float32) + ct.sum(vals), (1,)))


def _launch(
    inputs,
    *,
    C: int,
    H: int,
    W: int,
    PH: int,
    PW: int,
    BLOCK_C: int,
    BLOCK_HW: int,
    POINT_BLOCK: int,
    FINAL_BLOCK_C: int,
):
    arg0_1, _arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1 = inputs
    n = int(arg0_1.shape[0])
    hw = H * W
    device = arg0_1.device

    dense_shape = (n, C, H, W)
    dense_stride = (C * hw, 1, W * C, C)
    gate_shape = (n, C, 1, 1)
    gate_stride = (C, 1, 1, 1)

    # Allocate NHWC contiguous for outputs; permute for the strided output.
    out_grad_nhwc = torch.empty((n, H, W, C), device=device, dtype=torch.bfloat16)
    out_grad = out_grad_nhwc.permute(0, 3, 1, 2)
    out_sigmoid = torch.empty_strided(gate_shape, gate_stride, device=device, dtype=torch.bfloat16)
    out_scalar = torch.empty_strided((), (), device=device, dtype=torch.float32)
    out_scaled_nhwc = torch.empty((n, H, W, C), device=device, dtype=torch.bfloat16)
    out_scaled = out_scaled_nhwc.permute(0, 3, 1, 2)
    out_side = torch.empty_strided(gate_shape, gate_stride, device=device, dtype=torch.bfloat16)
    out_channel = torch.empty_strided((C,), (1,), device=device, dtype=torch.float32)
    side_nc = torch.empty_strided((n, C), (C, 1), device=device, dtype=torch.float32)
    total = n * C * hw
    scalar_blocks_count = (total + POINT_BLOCK - 1) // POINT_BLOCK
    scalar_blocks = torch.empty_strided(
        (scalar_blocks_count,), (1,), device=device, dtype=torch.float32
    )

    # Get NHWC views of the strided input tensors
    def nhwc_view(t):
        # t is (N, C, H, W) with strides (C*H*W, 1, W*C, C) → channels last
        return t.permute(0, 2, 3, 1).reshape(-1)

    arg0_flat = nhwc_view(arg0_1) if arg0_1.dim() == 4 else arg0_1.reshape(-1)
    arg2_flat = nhwc_view(arg2_1)
    arg3_flat = nhwc_view(arg3_1)
    arg5_flat = nhwc_view(arg5_1)

    # Precompute erf(gelu * 0.7071)
    erf_input = arg3_flat.to(torch.float32) * 0.7071067811865476
    erf_val = torch.special.erf(erf_input)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (scalar_blocks_count, 1, 1),
        _pointwise_kernel,
        (
            arg0_flat, arg2_flat, arg3_flat,
            arg4_1.view(-1), arg5_flat,
            arg6_1.view(1),
            erf_val,
            out_grad_nhwc.view(-1),
            out_scaled_nhwc.view(-1),
            scalar_blocks,
            C, H, W, PH, PW,
            total, POINT_BLOCK,
        ),
    )
    ct.launch(
        stream,
        (n, (C + BLOCK_C - 1) // BLOCK_C, 1),
        _side_reduce_kernel,
        (
            out_scaled_nhwc.view(-1),
            arg5_flat,
            arg4_1.view(-1),
            out_sigmoid.view(-1),
            out_side.view(-1),
            side_nc.view(-1),
            C, hw, n, BLOCK_C, BLOCK_HW,
        ),
    )
    final_blocks = (C + FINAL_BLOCK_C - 1) // FINAL_BLOCK_C
    ct.launch(
        stream,
        (final_blocks, 1, 1),
        _finalize_channels_kernel,
        (
            side_nc.view(-1),
            out_channel,
            C, n, FINAL_BLOCK_C, _ceil_pow2(n),
        ),
    )
    ct.launch(
        stream,
        (1, 1, 1),
        _finalize_scalar_kernel,
        (
            scalar_blocks,
            out_scalar.view(1),
            scalar_blocks_count, _ceil_pow2(scalar_blocks_count),
        ),
    )
    return out_grad, out_sigmoid, out_scalar, out_scaled, out_side, out_channel


@oracle_impl(hardware="B200", point="1a7f832d", C=512, H=24, W=24, PH=12, PW=12, BLOCK_C=16, BLOCK_HW=256, POINT_BLOCK=4096, FINAL_BLOCK_C=16)
@oracle_impl(hardware="B200", point="cf12eab2", C=1536, H=12, W=12, PH=6, PW=6, BLOCK_C=64, BLOCK_HW=32, POINT_BLOCK=512, FINAL_BLOCK_C=16)
def oracle_forward(inputs, *, C, H, W, PH, PW, BLOCK_C, BLOCK_HW, POINT_BLOCK, FINAL_BLOCK_C):
    return _launch(
        inputs,
        C=C, H=H, W=W, PH=PH, PW=PW,
        BLOCK_C=BLOCK_C, BLOCK_HW=BLOCK_HW,
        POINT_BLOCK=POINT_BLOCK,
        FINAL_BLOCK_C=FINAL_BLOCK_C,
    )
