"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete SqueezeNet bf16 two-input ReLU, virtual channel-cat, low-memory 3x3 stride-2 ceil-mode maxpool-with-offsets, and both full-layout ReLU mask outputs in one Triton stencil kernel, whereas Inductor lowers the ReLU/cat/stencil/mask graph through generic producer-consumer regions with avoidable activation and concatenation traffic; Inductor cannot do this today because its scheduler does not keep a fixed channel concat as a virtual multi-source layout feeding a low-memory maxpool stencil while also routing the shared ReLU predicates to sibling mask outputs; the fix is SCHEDULER_FUSION: teach pointwise/stencil scheduling to inline static cat producers into maxpool-with-offsets and emit deterministic sibling mask stores from the same plan while preserving bf16 ReLU NaN semantics, first-tie offsets, and output strides."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _relu_preserve_nan(x):
    return tl.where(x != x, x, tl.maximum(x, 0.0))


@triton.jit
def _virtual_cat_relu_pool_masks_kernel(
    x0_ptr,
    x1_ptr,
    values_ptr,
    offsets_ptr,
    mask_x1_ptr,
    mask_x0_ptr,
    total_outputs: tl.constexpr,
    channels: tl.constexpr,
    height: tl.constexpr,
    width: tl.constexpr,
    out_height: tl.constexpr,
    out_width: tl.constexpr,
    BLOCK: tl.constexpr,
):
    linear = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active = linear < total_outputs

    ow = linear % out_width
    tmp = linear // out_width
    oh = tmp % out_height
    tmp = tmp // out_height
    cat_channel = tmp % (channels * 2)
    batch = tmp // (channels * 2)

    use_x1 = cat_channel >= channels
    source_channel = cat_channel - tl.where(use_x1, channels, 0)
    source_base = (batch * channels + source_channel) * (height * width)

    best = tl.full((BLOCK,), -float("inf"), dtype=tl.float32)
    best_offset = tl.zeros((BLOCK,), dtype=tl.int32)

    for kh in tl.static_range(0, 3):
        in_h = oh * 2 + kh
        own_h = (kh != 2) | (oh == out_height - 1)
        for kw in tl.static_range(0, 3):
            in_w = ow * 2 + kw
            input_offset = source_base + in_h * width + in_w
            raw0 = tl.load(
                x0_ptr + input_offset,
                mask=active & ~use_x1,
                other=-float("inf"),
            )
            raw1 = tl.load(
                x1_ptr + input_offset,
                mask=active & use_x1,
                other=-float("inf"),
            )
            raw = tl.where(use_x1, raw1, raw0)

            own_mask = active & own_h & ((kw != 2) | (ow == out_width - 1))
            le_zero = raw <= 0.0
            tl.store(mask_x1_ptr + input_offset, le_zero, mask=own_mask & use_x1)
            tl.store(mask_x0_ptr + input_offset, le_zero, mask=own_mask & ~use_x1)

            relu = _relu_preserve_nan(raw)
            take = active & ((relu > best) | (relu != relu))
            best = tl.where(take, relu, best)
            best_offset = tl.where(take, kh * 3 + kw, best_offset)

    tl.store(values_ptr + linear, best, mask=active)
    tl.store(offsets_ptr + linear, best_offset.to(tl.int8), mask=active)


@oracle_impl(hardware="B200", point="e52f606e", BLOCK=256, num_warps=4)
@oracle_impl(hardware="B200", point="cb616840", BLOCK=256, num_warps=4)
def oracle_forward(inputs, *, BLOCK: int, num_warps: int):
    arg0_1, arg1_1, _shape_param_0, _shape_param_1 = inputs
    batch, channels, height, width = arg0_1.shape
    out_height = (height - 3) // 2 + 1
    out_width = (width - 3) // 2 + 1

    values_shape = (batch, channels * 2, out_height, out_width)
    values_stride = (channels * 2 * out_height * out_width, out_height * out_width, out_width, 1)
    mask_shape = tuple(arg0_1.shape)
    mask_stride = tuple(arg0_1.stride())

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
    mask_x1 = torch.empty_strided(
        mask_shape,
        mask_stride,
        device=arg0_1.device,
        dtype=torch.bool,
    )
    mask_x0 = torch.empty_strided(
        mask_shape,
        mask_stride,
        device=arg0_1.device,
        dtype=torch.bool,
    )

    total_outputs = values.numel()
    _virtual_cat_relu_pool_masks_kernel[(triton.cdiv(total_outputs, BLOCK),)](
        arg0_1,
        arg1_1,
        values,
        offsets,
        mask_x1,
        mask_x0,
        total_outputs=total_outputs,
        channels=channels,
        height=height,
        width=width,
        out_height=out_height,
        out_width=out_width,
        BLOCK=BLOCK,
        num_warps=num_warps,
        num_stages=4,
    )
    return values, offsets, mask_x1, mask_x0
