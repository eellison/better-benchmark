"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete bf16 ReLU, low-memory maxpool-with-offsets values and int8 offsets, and the full input-shaped ReLU <= 0 mask by sinking ReLU into shape-specialized Triton maxpool kernels and writing the mask directly from the original input, whereas Inductor currently schedules the stencil maxpool consumer and the full-layout boolean side output as separate work from the shared ReLU producer; Inductor cannot do this today because its scheduler does not fuse one pointwise producer into both a low-memory maxpool-with-offsets stencil/index consumer and a full-layout sibling materialization across these shape-specialized NCHW loop nests; the fix is SCHEDULER_FUSION: teach Inductor to keep the ReLU virtual across low-memory maxpool-with-offsets and emit the sibling mask stores from the same fused schedule."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _relu_preserve_nan(x):
    return tl.where(x != x, x, tl.maximum(x, 0.0))


@triton.jit
def _pool2_mask_kernel(
    input_ptr,
    values_ptr,
    offsets_ptr,
    mask_ptr,
    total_outputs: tl.constexpr,
    h_in: tl.constexpr,
    w_in: tl.constexpr,
    h_out: tl.constexpr,
    w_out: tl.constexpr,
    BLOCK: tl.constexpr,
):
    linear = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active = linear < total_outputs

    ow = linear % w_out
    tmp = linear // w_out
    oh = tmp % h_out
    plane = tmp // h_out
    input_base = plane * (h_in * w_in) + oh * (2 * w_in) + ow * 2

    x00 = tl.load(input_ptr + input_base, mask=active, other=-float("inf"))
    x01 = tl.load(input_ptr + input_base + 1, mask=active, other=-float("inf"))
    x10 = tl.load(input_ptr + input_base + w_in, mask=active, other=-float("inf"))
    x11 = tl.load(input_ptr + input_base + w_in + 1, mask=active, other=-float("inf"))

    best = _relu_preserve_nan(x00)
    best_offset = tl.zeros((BLOCK,), dtype=tl.int32)

    candidate = _relu_preserve_nan(x01)
    take = (candidate > best) | (candidate != candidate)
    best = tl.where(take, candidate, best)
    best_offset = tl.where(take, 1, best_offset)

    candidate = _relu_preserve_nan(x10)
    take = (candidate > best) | (candidate != candidate)
    best = tl.where(take, candidate, best)
    best_offset = tl.where(take, 2, best_offset)

    candidate = _relu_preserve_nan(x11)
    take = (candidate > best) | (candidate != candidate)
    best = tl.where(take, candidate, best)
    best_offset = tl.where(take, 3, best_offset)

    tl.store(values_ptr + linear, best, mask=active)
    tl.store(offsets_ptr + linear, best_offset.to(tl.int8), mask=active)
    tl.store(mask_ptr + input_base, x00 <= 0.0, mask=active)
    tl.store(mask_ptr + input_base + 1, x01 <= 0.0, mask=active)
    tl.store(mask_ptr + input_base + w_in, x10 <= 0.0, mask=active)
    tl.store(mask_ptr + input_base + w_in + 1, x11 <= 0.0, mask=active)


@triton.jit
def _pool3_kernel(
    input_ptr,
    values_ptr,
    offsets_ptr,
    total_outputs: tl.constexpr,
    h_in: tl.constexpr,
    w_in: tl.constexpr,
    h_out: tl.constexpr,
    w_out: tl.constexpr,
    BLOCK: tl.constexpr,
):
    linear = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active = linear < total_outputs

    ow = linear % w_out
    tmp = linear // w_out
    oh = tmp % h_out
    plane = tmp // h_out
    input_base = plane * (h_in * w_in) + oh * (2 * w_in) + ow * 2

    best = tl.full((BLOCK,), -float("inf"), dtype=tl.float32)
    best_offset = tl.zeros((BLOCK,), dtype=tl.int32)

    for kh in tl.static_range(0, 3):
        for kw in tl.static_range(0, 3):
            raw = tl.load(
                input_ptr + input_base + kh * w_in + kw,
                mask=active,
                other=-float("inf"),
            )
            candidate = _relu_preserve_nan(raw)
            take = (candidate > best) | (candidate != candidate)
            best = tl.where(take, candidate, best)
            best_offset = tl.where(take, kh * 3 + kw, best_offset)

    tl.store(values_ptr + linear, best, mask=active)
    tl.store(offsets_ptr + linear, best_offset.to(tl.int8), mask=active)


@triton.jit
def _mask_kernel(
    input_ptr,
    mask_ptr,
    total_inputs: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active = offsets < total_inputs
    raw = tl.load(input_ptr + offsets, mask=active, other=1.0)
    tl.store(mask_ptr + offsets, raw <= 0.0, mask=active)


@oracle_impl(hardware="B200", point="27e7b058", POOL_BLOCK=256, MASK_BLOCK=1024, pool_warps=4, mask_warps=4)
@oracle_impl(hardware="B200", point="9594e3d9", POOL_BLOCK=256, MASK_BLOCK=1024, pool_warps=4, mask_warps=4)
@oracle_impl(hardware="B200", point="2ff9ef59", POOL_BLOCK=256, MASK_BLOCK=1024, pool_warps=4, mask_warps=4)
@oracle_impl(hardware="B200", point="00f3245f", POOL_BLOCK=256, MASK_BLOCK=1024, pool_warps=4, mask_warps=4)
@oracle_impl(hardware="B200", point="8d6fda54", POOL_BLOCK=256, MASK_BLOCK=1024, pool_warps=4, mask_warps=4)
@oracle_impl(hardware="B200", point="371add6b", POOL_BLOCK=256, MASK_BLOCK=1024, pool_warps=4, mask_warps=4)
def oracle_forward(inputs, *, POOL_BLOCK, MASK_BLOCK, pool_warps, mask_warps):
    arg0_1, _shape_param_0, _shape_param_1 = inputs
    kernel_h = int(_shape_param_0[0])
    kernel_w = int(_shape_param_0[1])
    stride_h = int(_shape_param_1[0])
    stride_w = int(_shape_param_1[1])

    batch, channels, h_in, w_in = arg0_1.shape
    h_out = (h_in - kernel_h) // stride_h + 1
    w_out = (w_in - kernel_w) // stride_w + 1
    values_shape = (batch, channels, h_out, w_out)
    values_stride = (channels * h_out * w_out, h_out * w_out, w_out, 1)

    values = torch.empty_strided(
        values_shape,
        values_stride,
        device=arg0_1.device,
        dtype=arg0_1.dtype,
    )
    offsets = torch.empty_strided(
        values_shape,
        values_stride,
        device=arg0_1.device,
        dtype=torch.int8,
    )
    le_mask = torch.empty_strided(
        tuple(arg0_1.shape),
        tuple(arg0_1.stride()),
        device=arg0_1.device,
        dtype=torch.bool,
    )

    total_outputs = values.numel()
    if kernel_h == 2 and kernel_w == 2:
        _pool2_mask_kernel[(triton.cdiv(total_outputs, POOL_BLOCK),)](
            arg0_1,
            values,
            offsets,
            le_mask,
            total_outputs,
            h_in,
            w_in,
            h_out,
            w_out,
            BLOCK=POOL_BLOCK,
            num_warps=pool_warps,
        )
    else:
        _pool3_kernel[(triton.cdiv(total_outputs, POOL_BLOCK),)](
            arg0_1,
            values,
            offsets,
            total_outputs,
            h_in,
            w_in,
            h_out,
            w_out,
            BLOCK=POOL_BLOCK,
            num_warps=pool_warps,
        )
        _mask_kernel[(triton.cdiv(arg0_1.numel(), MASK_BLOCK),)](
            arg0_1,
            le_mask,
            arg0_1.numel(),
            BLOCK=MASK_BLOCK,
            num_warps=mask_warps,
        )

    return values, offsets, le_mask
