"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete SqueezeNet bf16 ReLU, 3x3 stride-2 low-memory maxpool values and int8 offsets, and full input-shaped ReLU `<= 0` mask in one output-tiled Triton stencil, sinking ReLU into each pooled load and storing every mask element exactly once from the same producer, whereas Inductor lowers the shared ReLU producer, maxpool tuple stencil, and sibling full-layout mask materialization through generic pointwise/stencil scheduling; Inductor cannot do this today because its scheduler does not fuse a pointwise producer into both a low-memory maxpool-with-offsets consumer and a layout-preserving side-output mask without splitting the region; the fix is SCHEDULER_FUSION: teach pointwise/stencil scheduling to keep ReLU virtual across maxpool-with-offsets while assigning owned side-output mask stores in the same tiled loop nest."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


BATCH = 32
CHANNELS = 64
HEIGHT = 111
WIDTH = 111
OUT_HEIGHT = 55
OUT_WIDTH = 55
TOTAL_OUT = BATCH * CHANNELS * OUT_HEIGHT * OUT_WIDTH


@triton.jit
def _relu_preserve_nan(x):
    return tl.where((x > 0.0) | (x != x), x, 0.0)


@triton.jit
def _relu_maxpool3_offsets_mask_kernel(
    input_ptr,
    values_ptr,
    offsets_ptr,
    mask_ptr,
    total_outputs: tl.constexpr,
    height: tl.constexpr,
    width: tl.constexpr,
    out_height: tl.constexpr,
    out_width: tl.constexpr,
    BLOCK: tl.constexpr,
):
    out_offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active = out_offsets < total_outputs

    ow = out_offsets % out_width
    tmp = out_offsets // out_width
    oh = tmp % out_height
    plane = tmp // out_height
    input_base = plane * height * width + oh * (2 * width) + ow * 2

    best = tl.full((BLOCK,), -float("inf"), dtype=tl.float32)
    best_idx = tl.zeros((BLOCK,), dtype=tl.int32)

    for kh in tl.static_range(0, 3):
        for kw in tl.static_range(0, 3):
            input_offset = input_base + kh * width + kw
            raw = tl.load(
                input_ptr + input_offset,
                mask=active,
                other=-float("inf"),
                eviction_policy="evict_last",
            ).to(tl.float32)
            relu = _relu_preserve_nan(raw)

            owns_mask = active
            if kh == 2:
                owns_mask = owns_mask & (oh == (out_height - 1))
            if kw == 2:
                owns_mask = owns_mask & (ow == (out_width - 1))
            tl.store(mask_ptr + input_offset, raw <= 0.0, mask=owns_mask)

            idx = kh * 3 + kw
            take = (relu > best) | ((relu != relu) & (best == best))
            best = tl.where(take, relu, best)
            best_idx = tl.where(take, idx, best_idx)

    tl.store(values_ptr + out_offsets, best.to(tl.bfloat16), mask=active)
    tl.store(offsets_ptr + out_offsets, best_idx.to(tl.int8), mask=active)


# 04c86358: bf16[32,64,111,111] -> pool bf16/int8 [32,64,55,55] and bool mask.
@oracle_impl(hardware="B200", point="04c86358", BLOCK=256, num_warps=4)
def oracle_forward(inputs, *, BLOCK: int, num_warps: int):
    x, _kernel_size, _stride = inputs
    values = torch.empty_strided(
        (BATCH, CHANNELS, OUT_HEIGHT, OUT_WIDTH),
        (CHANNELS * OUT_HEIGHT * OUT_WIDTH, OUT_HEIGHT * OUT_WIDTH, OUT_WIDTH, 1),
        device=x.device,
        dtype=torch.bfloat16,
    )
    offsets = torch.empty_strided(
        (BATCH, CHANNELS, OUT_HEIGHT, OUT_WIDTH),
        (CHANNELS * OUT_HEIGHT * OUT_WIDTH, OUT_HEIGHT * OUT_WIDTH, OUT_WIDTH, 1),
        device=x.device,
        dtype=torch.int8,
    )
    le_mask = torch.empty_strided(
        (BATCH, CHANNELS, HEIGHT, WIDTH),
        (CHANNELS * HEIGHT * WIDTH, HEIGHT * WIDTH, WIDTH, 1),
        device=x.device,
        dtype=torch.bool,
    )

    _relu_maxpool3_offsets_mask_kernel[(triton.cdiv(TOTAL_OUT, BLOCK),)](
        x,
        values,
        offsets,
        le_mask,
        total_outputs=TOTAL_OUT,
        height=HEIGHT,
        width=WIDTH,
        out_height=OUT_HEIGHT,
        out_width=OUT_WIDTH,
        BLOCK=BLOCK,
        num_warps=num_warps,
        num_stages=3,
    )
    return values, offsets, le_mask
