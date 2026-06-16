"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete visible bf16 `relu` plus 3x3 stride-2 low-memory maxpool value scope in one fixed-shape Triton stencil, sinking ReLU into the nine input loads and returning only the fresh contiguous `[16,64,55,55]` pooled tensor, whereas Inductor's generic low-memory maxpool path must preserve the full stencil/indexing abstraction around a pointwise producer even though the offset tensor is dead in this repro; Inductor cannot express this returned-value-only ReLU/maxpool specialization today as a simple fixed-window output materialization; the fix is SCHEDULER_FUSION: add a guarded low-memory maxpool value lowering that keeps pointwise producers virtual and skips dead offset work."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


BATCH = 16
CHANNELS = 64
HEIGHT = 111
WIDTH = 111
OUT_HEIGHT = 55
OUT_WIDTH = 55
TOTAL = BATCH * CHANNELS * OUT_HEIGHT * OUT_WIDTH


@triton.jit
def _relu_preserve_nan(x):
    return tl.where((x > 0.0) | (x != x), x, 0.0)


@triton.jit
def _take_candidate(candidate, best):
    take = (candidate > best) | ((candidate != candidate) & (best == best))
    return tl.where(take, candidate, best)


@triton.jit
def _relu_maxpool3_values_kernel(
    input_ptr,
    output_ptr,
    TOTAL_OUT: tl.constexpr,
    HEIGHT_: tl.constexpr,
    WIDTH_: tl.constexpr,
    OUT_HEIGHT_: tl.constexpr,
    OUT_WIDTH_: tl.constexpr,
    BLOCK: tl.constexpr,
):
    out_offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active = out_offsets < TOTAL_OUT

    out_w = out_offsets % OUT_WIDTH_
    tmp = out_offsets // OUT_WIDTH_
    out_h = tmp % OUT_HEIGHT_
    plane = tmp // OUT_HEIGHT_
    input_base = plane * HEIGHT_ * WIDTH_ + out_h * (2 * WIDTH_) + out_w * 2

    v0 = _relu_preserve_nan(tl.load(input_ptr + input_base, mask=active, other=-float("inf")).to(tl.float32))
    v1 = _relu_preserve_nan(tl.load(input_ptr + input_base + 1, mask=active, other=-float("inf")).to(tl.float32))
    v2 = _relu_preserve_nan(tl.load(input_ptr + input_base + 2, mask=active, other=-float("inf")).to(tl.float32))
    v3 = _relu_preserve_nan(tl.load(input_ptr + input_base + WIDTH_, mask=active, other=-float("inf")).to(tl.float32))
    v4 = _relu_preserve_nan(tl.load(input_ptr + input_base + WIDTH_ + 1, mask=active, other=-float("inf")).to(tl.float32))
    v5 = _relu_preserve_nan(tl.load(input_ptr + input_base + WIDTH_ + 2, mask=active, other=-float("inf")).to(tl.float32))
    v6 = _relu_preserve_nan(tl.load(input_ptr + input_base + 2 * WIDTH_, mask=active, other=-float("inf")).to(tl.float32))
    v7 = _relu_preserve_nan(tl.load(input_ptr + input_base + 2 * WIDTH_ + 1, mask=active, other=-float("inf")).to(tl.float32))
    v8 = _relu_preserve_nan(tl.load(input_ptr + input_base + 2 * WIDTH_ + 2, mask=active, other=-float("inf")).to(tl.float32))

    best = v0
    best = _take_candidate(v1, best)
    best = _take_candidate(v2, best)
    best = _take_candidate(v3, best)
    best = _take_candidate(v4, best)
    best = _take_candidate(v5, best)
    best = _take_candidate(v6, best)
    best = _take_candidate(v7, best)
    best = _take_candidate(v8, best)

    tl.store(output_ptr + out_offsets, best, mask=active)


# (T([16,64,111,111], bf16), S([3,3]), S([2,2]))
@oracle_impl(hardware="B200", point="63e4540f", BLOCK=256, num_warps=4)
def oracle_forward(inputs, *, BLOCK: int, num_warps: int):
    x, _kernel_size, _stride = inputs
    out = torch.empty_strided(
        (BATCH, CHANNELS, OUT_HEIGHT, OUT_WIDTH),
        (CHANNELS * OUT_HEIGHT * OUT_WIDTH, OUT_HEIGHT * OUT_WIDTH, OUT_WIDTH, 1),
        device=x.device,
        dtype=torch.bfloat16,
    )
    _relu_maxpool3_values_kernel[(triton.cdiv(TOTAL, BLOCK),)](
        x,
        out,
        TOTAL_OUT=TOTAL,
        HEIGHT_=HEIGHT,
        WIDTH_=WIDTH,
        OUT_HEIGHT_=OUT_HEIGHT,
        OUT_WIDTH_=OUT_WIDTH,
        BLOCK=BLOCK,
        num_warps=num_warps,
        num_stages=3,
    )
    return out
