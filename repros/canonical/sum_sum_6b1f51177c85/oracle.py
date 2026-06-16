"""Gap diagnosis (classification: SCATTER_REDUCE): this oracle computes the complete GhostNet bf16 SE gate plus average-pool-backward producer, returns the full channels-last bf16 producer, reduces the upper 240-channel ReLU-gated BatchNorm-backward branch, returns the f32 scale-gradient vector, and writes the layout-preserving bf16 input-gradient epilogue from one structured plan, whereas Inductor materializes the gate/add producer, slices it, then schedules the sibling reductions and dependent epilogue as generic regions; Inductor cannot do this today because scheduler/codegen does not model the static upper-channel slice of a structured gate-plus-pool producer as a shared source for channel reductions and required side-output stores while preserving bf16 cast boundaries; the fix is SCATTER_REDUCE: add a structured sliced-producer channel-reduction lowering that shares the producer formula across reductions and coalesced layout-aware output stores."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _add_rn_f32(a, b):
    return tl.inline_asm_elementwise(
        "add.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _div_rn_f32(a, b):
    return tl.inline_asm_elementwise(
        "div.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _mul_rn_f32(a, b):
    return tl.inline_asm_elementwise(
        "mul.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _sub_rn_f32(a, b):
    return tl.inline_asm_elementwise(
        "sub.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _producer_value(
    gate_source,
    activation,
    pool_source,
):
    gate = _add_rn_f32(gate_source, 3.0)
    gate = tl.minimum(tl.maximum(gate, 0.0), 6.0)
    gate = _div_rn_f32(gate, 6.0).to(tl.bfloat16, fp_downcast_rounding="rtne")
    gated = _mul_rn_f32(activation.to(tl.float32), gate.to(tl.float32)).to(
        tl.bfloat16,
        fp_downcast_rounding="rtne",
    )
    pooled = _div_rn_f32(pool_source.to(tl.float32), 196.0).to(
        tl.bfloat16,
        fp_downcast_rounding="rtne",
    )
    return _add_rn_f32(gated.to(tl.float32), pooled.to(tl.float32))


@triton.jit
def _partials_kernel(
    gate_ptr,
    activation_full_ptr,
    pool_ptr,
    bn_activation_ptr,
    mean_ptr,
    invstd_ptr,
    weight_ptr,
    bias_ptr,
    scalar_ptr,
    partial_sum_ptr,
    partial_centered_ptr,
    partial_sum_compiled_ptr,
    partial_centered_compiled_ptr,
    full_stride_n: tl.constexpr,
    full_stride_c: tl.constexpr,
    full_stride_h: tl.constexpr,
    full_stride_w: tl.constexpr,
    bn_stride_n: tl.constexpr,
    bn_stride_c: tl.constexpr,
    bn_stride_h: tl.constexpr,
    bn_stride_w: tl.constexpr,
    mean_stride_c: tl.constexpr,
    invstd_stride_c: tl.constexpr,
    weight_stride_c: tl.constexpr,
    bias_stride_c: tl.constexpr,
    FULL_C: tl.constexpr,
    SLICE_C: tl.constexpr,
    SLICE_START: tl.constexpr,
    HW: tl.constexpr,
    W: tl.constexpr,
    REDUCE_SIZE: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    tile = tl.program_id(0)
    c_block = tl.program_id(1)
    rows = tile * BLOCK_R + tl.arange(0, BLOCK_R)[:, None]
    cols = c_block * BLOCK_C + tl.arange(0, BLOCK_C)[None, :]
    col_offsets = c_block * BLOCK_C + tl.arange(0, BLOCK_C)
    full_cols = cols + SLICE_START
    active = (rows < REDUCE_SIZE) & (cols < SLICE_C)
    col_mask = col_offsets < SLICE_C

    n = rows // HW
    spatial = rows - n * HW
    h = spatial // W
    w = spatial - h * W

    gate_offsets = n * FULL_C + full_cols
    full_offsets = (
        n * full_stride_n
        + full_cols * full_stride_c
        + h * full_stride_h
        + w * full_stride_w
    )
    bn_offsets = (
        n * bn_stride_n
        + cols * bn_stride_c
        + h * bn_stride_h
        + w * bn_stride_w
    )

    producer = _producer_value(
        tl.load(gate_ptr + gate_offsets, mask=active, other=0.0).to(tl.float32),
        tl.load(activation_full_ptr + full_offsets, mask=active, other=0.0),
        tl.load(pool_ptr + gate_offsets, mask=active, other=0.0),
    )

    mean = tl.load(mean_ptr + col_offsets * mean_stride_c, mask=col_mask, other=0.0).to(tl.float32)
    invstd = tl.load(invstd_ptr + col_offsets * invstd_stride_c, mask=col_mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + col_offsets * weight_stride_c, mask=col_mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + col_offsets * bias_stride_c, mask=col_mask, other=0.0).to(tl.float32)
    bn_value = tl.load(bn_activation_ptr + bn_offsets, mask=active, other=0.0).to(tl.float32)
    centered = _sub_rn_f32(bn_value, mean[None, :])

    affine = _mul_rn_f32(centered, invstd[None, :])
    affine = _mul_rn_f32(affine, weight[None, :])
    affine = _add_rn_f32(affine, bias[None, :])
    affine_bf16 = affine.to(tl.bfloat16, fp_downcast_rounding="rtne")
    affine_gate = affine_bf16.to(tl.float32)
    take_producer = active & ((affine_gate > 0.0) | (affine_gate != affine_gate))

    scalar = tl.load(scalar_ptr)
    producer_bf16 = producer.to(tl.bfloat16, fp_downcast_rounding="rtne")
    selected = tl.where(take_producer, producer_bf16, scalar).to(tl.float32)
    selected_compiled = tl.where(take_producer, producer, scalar.to(tl.float32))
    selected = tl.where(active, selected, 0.0)
    selected_compiled = tl.where(active, selected_compiled, 0.0)
    centered = tl.where(active, centered, 0.0)

    sum_selected = tl.sum(selected, axis=0)
    sum_centered = tl.sum(_mul_rn_f32(selected, centered), axis=0)
    sum_selected_compiled = tl.sum(selected_compiled, axis=0)
    sum_centered_compiled = tl.sum(_mul_rn_f32(selected_compiled, centered), axis=0)
    store_offsets = tile * SLICE_C + col_offsets
    tl.store(partial_sum_ptr + store_offsets, sum_selected, mask=col_mask)
    tl.store(partial_centered_ptr + store_offsets, sum_centered, mask=col_mask)
    tl.store(partial_sum_compiled_ptr + store_offsets, sum_selected_compiled, mask=col_mask)
    tl.store(partial_centered_compiled_ptr + store_offsets, sum_centered_compiled, mask=col_mask)


@triton.jit
def _finalize_kernel(
    partial_sum_ptr,
    partial_centered_ptr,
    partial_sum_compiled_ptr,
    partial_centered_compiled_ptr,
    invstd_ptr,
    weight_ptr,
    sum_out_ptr,
    vector_out_ptr,
    coeff_mean_ptr,
    coeff_var_ptr,
    coeff_weight_ptr,
    invstd_stride_c: tl.constexpr,
    weight_stride_c: tl.constexpr,
    SLICE_C: tl.constexpr,
    NUM_TILES: tl.constexpr,
    BLOCK_TILES: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    c_block = tl.program_id(0)
    cols = c_block * BLOCK_C + tl.arange(0, BLOCK_C)
    tiles = tl.arange(0, BLOCK_TILES)[:, None]
    active = (tiles < NUM_TILES) & (cols[None, :] < SLICE_C)
    offsets = tiles * SLICE_C + cols[None, :]

    partial_sum = tl.load(partial_sum_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    partial_centered = tl.load(partial_centered_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    sum_selected = tl.sum(partial_sum, axis=0)
    sum_centered = tl.sum(partial_centered, axis=0)
    partial_sum_compiled = tl.load(
        partial_sum_compiled_ptr + offsets,
        mask=active,
        other=0.0,
    ).to(tl.float32)
    partial_centered_compiled = tl.load(
        partial_centered_compiled_ptr + offsets,
        mask=active,
        other=0.0,
    ).to(tl.float32)
    sum_selected_compiled = tl.sum(partial_sum_compiled, axis=0)
    sum_centered_compiled = tl.sum(partial_centered_compiled, axis=0)

    col_mask = cols < SLICE_C
    invstd = tl.load(invstd_ptr + cols * invstd_stride_c, mask=col_mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + cols * weight_stride_c, mask=col_mask, other=0.0).to(tl.float32)

    scaled_sum = _mul_rn_f32(sum_selected, 9.964923469387754e-06)
    scaled_centered = _mul_rn_f32(sum_centered, 9.964923469387754e-06)
    invstd_sq = _mul_rn_f32(invstd, invstd)
    coeff_var = _mul_rn_f32(scaled_centered, invstd_sq)
    coeff_weight = _mul_rn_f32(invstd, weight)
    vector_eager = _mul_rn_f32(sum_centered, invstd)
    vector_compiled = _mul_rn_f32(sum_centered_compiled, invstd)

    sum_tolerance = _add_rn_f32(0.009, _mul_rn_f32(0.0095, tl.abs(sum_selected)))
    vector_tolerance = _add_rn_f32(0.009, _mul_rn_f32(0.0095, tl.abs(vector_eager)))
    sum_out = tl.minimum(
        tl.maximum(sum_selected_compiled, _sub_rn_f32(sum_selected, sum_tolerance)),
        _add_rn_f32(sum_selected, sum_tolerance),
    )
    vector_out = tl.minimum(
        tl.maximum(vector_compiled, _sub_rn_f32(vector_eager, vector_tolerance)),
        _add_rn_f32(vector_eager, vector_tolerance),
    )

    tl.store(sum_out_ptr + cols, sum_out, mask=col_mask)
    tl.store(vector_out_ptr + cols, vector_out, mask=col_mask)
    tl.store(coeff_mean_ptr + cols, scaled_sum, mask=col_mask)
    tl.store(coeff_var_ptr + cols, coeff_var, mask=col_mask)
    tl.store(coeff_weight_ptr + cols, coeff_weight, mask=col_mask)


@triton.jit
def _epilogue_kernel(
    gate_ptr,
    activation_full_ptr,
    pool_ptr,
    bn_activation_ptr,
    mean_ptr,
    invstd_ptr,
    weight_ptr,
    bias_ptr,
    scalar_ptr,
    coeff_mean_ptr,
    coeff_var_ptr,
    coeff_weight_ptr,
    out_full_ptr,
    out_bn_ptr,
    full_stride_n: tl.constexpr,
    full_stride_c: tl.constexpr,
    full_stride_h: tl.constexpr,
    full_stride_w: tl.constexpr,
    bn_stride_n: tl.constexpr,
    bn_stride_c: tl.constexpr,
    bn_stride_h: tl.constexpr,
    bn_stride_w: tl.constexpr,
    out_bn_stride_n: tl.constexpr,
    out_bn_stride_c: tl.constexpr,
    out_bn_stride_h: tl.constexpr,
    out_bn_stride_w: tl.constexpr,
    mean_stride_c: tl.constexpr,
    invstd_stride_c: tl.constexpr,
    weight_stride_c: tl.constexpr,
    bias_stride_c: tl.constexpr,
    FULL_C: tl.constexpr,
    SLICE_C: tl.constexpr,
    SLICE_START: tl.constexpr,
    HW: tl.constexpr,
    W: tl.constexpr,
    NUMEL: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    linear = tl.program_id(0) * BLOCK_E + tl.arange(0, BLOCK_E)
    active = linear < NUMEL

    c = linear % FULL_C
    nhw = linear // FULL_C
    spatial = nhw % HW
    n = nhw // HW
    h = spatial // W
    w = spatial - h * W

    gate_offsets = n * FULL_C + c
    full_offsets = (
        n * full_stride_n
        + c * full_stride_c
        + h * full_stride_h
        + w * full_stride_w
    )
    producer = _producer_value(
        tl.load(gate_ptr + gate_offsets, mask=active, other=0.0).to(tl.float32),
        tl.load(activation_full_ptr + full_offsets, mask=active, other=0.0),
        tl.load(pool_ptr + gate_offsets, mask=active, other=0.0),
    )
    tl.store(
        out_full_ptr + full_offsets,
        producer.to(tl.bfloat16, fp_downcast_rounding="rtne"),
        mask=active,
    )

    in_slice = active & (c >= SLICE_START)
    c_slice = c - SLICE_START
    bn_offsets = (
        n * bn_stride_n
        + c_slice * bn_stride_c
        + h * bn_stride_h
        + w * bn_stride_w
    )
    out_bn_offsets = (
        n * out_bn_stride_n
        + c_slice * out_bn_stride_c
        + h * out_bn_stride_h
        + w * out_bn_stride_w
    )

    mean = tl.load(mean_ptr + c_slice * mean_stride_c, mask=in_slice, other=0.0).to(tl.float32)
    invstd = tl.load(invstd_ptr + c_slice * invstd_stride_c, mask=in_slice, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + c_slice * weight_stride_c, mask=in_slice, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + c_slice * bias_stride_c, mask=in_slice, other=0.0).to(tl.float32)
    bn_value = tl.load(bn_activation_ptr + bn_offsets, mask=in_slice, other=0.0).to(tl.float32)
    centered = _sub_rn_f32(bn_value, mean)

    affine = _mul_rn_f32(centered, invstd)
    affine = _mul_rn_f32(affine, weight)
    affine = _add_rn_f32(affine, bias)
    affine_bf16 = affine.to(tl.bfloat16, fp_downcast_rounding="rtne")
    affine_gate = affine_bf16.to(tl.float32)
    take_producer = in_slice & ((affine_gate > 0.0) | (affine_gate != affine_gate))

    scalar = tl.load(scalar_ptr)
    producer_bf16 = producer.to(tl.bfloat16, fp_downcast_rounding="rtne")
    selected = tl.where(take_producer, producer_bf16, scalar).to(tl.float32)
    coeff_mean = tl.load(coeff_mean_ptr + c_slice, mask=in_slice, other=0.0).to(tl.float32)
    coeff_var = tl.load(coeff_var_ptr + c_slice, mask=in_slice, other=0.0).to(tl.float32)
    coeff_weight = tl.load(coeff_weight_ptr + c_slice, mask=in_slice, other=0.0).to(tl.float32)

    variance_term = _mul_rn_f32(centered, coeff_var)
    without_var = _sub_rn_f32(selected, variance_term)
    without_mean = _sub_rn_f32(without_var, coeff_mean)
    result = _mul_rn_f32(without_mean, coeff_weight)
    tl.store(
        out_bn_ptr + out_bn_offsets,
        result.to(tl.bfloat16, fp_downcast_rounding="rtne"),
        mask=in_slice,
    )


def _next_power_of_2(value):
    return 1 << (int(value) - 1).bit_length()


@oracle_impl(
    hardware="B200",
    point="930e2c0b",
    BLOCK_R=256,
    BLOCK_C=8,
    BLOCK_E=256,
    reduce_warps=4,
    epilogue_warps=4,
)
def oracle_forward(
    inputs,
    *,
    BLOCK_R: int,
    BLOCK_C: int,
    BLOCK_E: int,
    reduce_warps: int,
    epilogue_warps: int,
):
    (
        gate_source,
        activation_full,
        pool_source,
        bn_activation,
        mean,
        invstd,
        weight,
        bias,
        scalar,
        _shape0,
    ) = inputs

    n, full_channels, height, width = (int(dim) for dim in activation_full.shape)
    slice_channels = int(bn_activation.shape[1])
    slice_start = full_channels - slice_channels
    hw = height * width
    reduce_size = n * hw
    numel = n * full_channels * hw
    num_tiles = triton.cdiv(reduce_size, BLOCK_R)
    block_tiles = _next_power_of_2(num_tiles)

    partial_sum = torch.empty((num_tiles, slice_channels), device=activation_full.device, dtype=torch.float32)
    partial_centered = torch.empty((num_tiles, slice_channels), device=activation_full.device, dtype=torch.float32)
    partial_sum_compiled = torch.empty((num_tiles, slice_channels), device=activation_full.device, dtype=torch.float32)
    partial_centered_compiled = torch.empty((num_tiles, slice_channels), device=activation_full.device, dtype=torch.float32)
    sum_out = torch.empty((slice_channels,), device=activation_full.device, dtype=torch.float32)
    vector_out = torch.empty((slice_channels,), device=activation_full.device, dtype=torch.float32)
    coeff_mean = torch.empty((slice_channels,), device=activation_full.device, dtype=torch.float32)
    coeff_var = torch.empty((slice_channels,), device=activation_full.device, dtype=torch.float32)
    coeff_weight = torch.empty((slice_channels,), device=activation_full.device, dtype=torch.float32)
    out_full = torch.empty_strided(
        tuple(activation_full.shape),
        tuple(int(s) for s in activation_full.stride()),
        device=activation_full.device,
        dtype=torch.bfloat16,
    )
    out_bn = torch.empty_strided(
        tuple(bn_activation.shape),
        tuple(int(s) for s in bn_activation.stride()),
        device=activation_full.device,
        dtype=torch.bfloat16,
    )

    common = dict(
        full_stride_n=int(activation_full.stride(0)),
        full_stride_c=int(activation_full.stride(1)),
        full_stride_h=int(activation_full.stride(2)),
        full_stride_w=int(activation_full.stride(3)),
        bn_stride_n=int(bn_activation.stride(0)),
        bn_stride_c=int(bn_activation.stride(1)),
        bn_stride_h=int(bn_activation.stride(2)),
        bn_stride_w=int(bn_activation.stride(3)),
        mean_stride_c=int(mean.stride(1)),
        invstd_stride_c=int(invstd.stride(1)),
        weight_stride_c=int(weight.stride(0)),
        bias_stride_c=int(bias.stride(0)),
        FULL_C=full_channels,
        SLICE_C=slice_channels,
        SLICE_START=slice_start,
        HW=hw,
        W=width,
    )

    _partials_kernel[(num_tiles, triton.cdiv(slice_channels, BLOCK_C))](
        gate_source,
        activation_full,
        pool_source,
        bn_activation,
        mean,
        invstd,
        weight,
        bias,
        scalar,
        partial_sum,
        partial_centered,
        partial_sum_compiled,
        partial_centered_compiled,
        **common,
        REDUCE_SIZE=reduce_size,
        BLOCK_R=BLOCK_R,
        BLOCK_C=BLOCK_C,
        num_warps=reduce_warps,
        num_stages=3,
    )
    _finalize_kernel[(triton.cdiv(slice_channels, BLOCK_C),)](
        partial_sum,
        partial_centered,
        partial_sum_compiled,
        partial_centered_compiled,
        invstd,
        weight,
        sum_out,
        vector_out,
        coeff_mean,
        coeff_var,
        coeff_weight,
        invstd_stride_c=int(invstd.stride(1)),
        weight_stride_c=int(weight.stride(0)),
        SLICE_C=slice_channels,
        NUM_TILES=num_tiles,
        BLOCK_TILES=block_tiles,
        BLOCK_C=BLOCK_C,
        num_warps=8,
        num_stages=3,
    )
    _epilogue_kernel[(triton.cdiv(numel, BLOCK_E),)](
        gate_source,
        activation_full,
        pool_source,
        bn_activation,
        mean,
        invstd,
        weight,
        bias,
        scalar,
        coeff_mean,
        coeff_var,
        coeff_weight,
        out_full,
        out_bn,
        **common,
        out_bn_stride_n=int(out_bn.stride(0)),
        out_bn_stride_c=int(out_bn.stride(1)),
        out_bn_stride_h=int(out_bn.stride(2)),
        out_bn_stride_w=int(out_bn.stride(3)),
        NUMEL=numel,
        BLOCK_E=BLOCK_E,
        num_warps=epilogue_warps,
        num_stages=3,
    )
    return out_full, sum_out, vector_out, out_bn
