"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete bf16 VGG16 ReLU -> 2x2 stride-2 low-memory maxpool-with-offsets -> shape-identical adaptive_avg_pool2d/view plus the full input-shaped `relu <= 0` mask in one Triton stencil, writing the returned pooled values, int8 offsets, separate flattened pooled copy, and bool mask directly, whereas Inductor schedules the maxpool stencil, identity/layout materialization, and mask side-output as generic work around the shared ReLU producer; Inductor cannot do this today because its scheduler does not fuse a multi-output low-memory maxpool stencil with both a layout-only consumer and a full-layout pointwise side consumer while preserving first-index tie and NaN offset semantics; the fix is SCHEDULER_FUSION: teach maxpool lowering/scheduling to absorb the ReLU producer and emit all escaping values, offsets, layout views, and side masks from the same loop nest."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


BATCH = 64
CHANNELS = 512
H_IN = 14
W_IN = 14
H_OUT = 7
W_OUT = 7
INPUT_PLANE = H_IN * W_IN
OUTPUT_PLANE = H_OUT * W_OUT
TOTAL_OUTPUTS = BATCH * CHANNELS * OUTPUT_PLANE


@triton.jit
def _relu_preserve_nan(x):
    return tl.where((x > 0.0) | (x != x), x, 0.0)


@triton.jit
def _take_candidate(candidate, candidate_offset, best, best_offset):
    take = (candidate > best) | ((best == best) & (candidate != candidate))
    best = tl.where(take, candidate, best)
    best_offset = tl.where(take, candidate_offset, best_offset)
    return best, best_offset


@triton.jit
def _relu_maxpool_values_offsets_flat_mask_kernel(
    input_ptr,
    values_ptr,
    offsets_ptr,
    flat_values_ptr,
    mask_ptr,
    total_outputs: tl.constexpr,
    BLOCK: tl.constexpr,
):
    linear = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active = linear < total_outputs

    out_w = linear % 7
    tmp = linear // 7
    out_h = tmp % 7
    plane = tmp // 7

    input_base = plane * 196 + out_h * 28 + out_w * 2
    x00 = tl.load(input_ptr + input_base, mask=active, other=0.0).to(tl.float32)
    x01 = tl.load(input_ptr + input_base + 1, mask=active, other=0.0).to(tl.float32)
    x10 = tl.load(input_ptr + input_base + 14, mask=active, other=0.0).to(tl.float32)
    x11 = tl.load(input_ptr + input_base + 15, mask=active, other=0.0).to(tl.float32)

    best = _relu_preserve_nan(x00)
    best_offset = tl.zeros((BLOCK,), dtype=tl.int32)

    candidate = _relu_preserve_nan(x01)
    best, best_offset = _take_candidate(candidate, 1, best, best_offset)
    candidate = _relu_preserve_nan(x10)
    best, best_offset = _take_candidate(candidate, 2, best, best_offset)
    candidate = _relu_preserve_nan(x11)
    best, best_offset = _take_candidate(candidate, 3, best, best_offset)

    tl.store(values_ptr + linear, best, mask=active)
    tl.store(offsets_ptr + linear, best_offset.to(tl.int8), mask=active)
    tl.store(flat_values_ptr + linear, best, mask=active)
    tl.store(mask_ptr + input_base, x00 <= 0.0, mask=active)
    tl.store(mask_ptr + input_base + 1, x01 <= 0.0, mask=active)
    tl.store(mask_ptr + input_base + 14, x10 <= 0.0, mask=active)
    tl.store(mask_ptr + input_base + 15, x11 <= 0.0, mask=active)


# 2c0e267a: VGG16 bf16 [64,512,14,14] relu/maxpool/flatten/mask.
@oracle_impl(hardware="B200", point="2c0e267a", BLOCK=128, num_warps=4, num_stages=3)
def oracle_forward(inputs, *, BLOCK: int, num_warps: int, num_stages: int):
    arg0_1 = inputs[0]
    values = torch.empty_strided(
        (BATCH, CHANNELS, H_OUT, W_OUT),
        (CHANNELS * OUTPUT_PLANE, OUTPUT_PLANE, W_OUT, 1),
        device=arg0_1.device,
        dtype=arg0_1.dtype,
    )
    offsets = torch.empty_strided(
        (BATCH, CHANNELS, H_OUT, W_OUT),
        (CHANNELS * OUTPUT_PLANE, OUTPUT_PLANE, W_OUT, 1),
        device=arg0_1.device,
        dtype=torch.int8,
    )
    flat_values = torch.empty_strided(
        (BATCH, CHANNELS * OUTPUT_PLANE),
        (CHANNELS * OUTPUT_PLANE, 1),
        device=arg0_1.device,
        dtype=arg0_1.dtype,
    )
    le_mask = torch.empty_strided(
        (BATCH, CHANNELS, H_IN, W_IN),
        (CHANNELS * INPUT_PLANE, INPUT_PLANE, W_IN, 1),
        device=arg0_1.device,
        dtype=torch.bool,
    )

    _relu_maxpool_values_offsets_flat_mask_kernel[(triton.cdiv(TOTAL_OUTPUTS, BLOCK),)](
        arg0_1,
        values,
        offsets,
        flat_values,
        le_mask,
        total_outputs=TOTAL_OUTPUTS,
        BLOCK=BLOCK,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return values, offsets, flat_values, le_mask
