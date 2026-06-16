"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete visible bf16 VGG ReLU -> 2x2 stride-2 low-memory maxpool value -> shape-identical adaptive_avg_pool2d -> `[4,25088]` flatten scope in one fixed-shape Triton stencil, sinking ReLU into the four maxpool input loads and writing the final flattened contiguous output directly, whereas Inductor's generic low-memory maxpool path keeps the stencil/indexing abstraction and the identity/layout consumers as separate scheduled work even though the int8 offset tensor is dead in this repro. Inductor cannot express this returned-value-only specialization today because the scheduler does not fuse pointwise producers and layout-only consumers through a low-memory maxpool while dropping unobserved offsets. The fix is SCHEDULER_FUSION: add a guarded maxpool-value lowering that keeps the ReLU producer virtual, skips dead offsets, and emits the flattened consumer layout from the same loop nest."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


BATCH = 4
CHANNELS = 512
HEIGHT = 14
WIDTH = 14
OUT_HEIGHT = 7
OUT_WIDTH = 7
FLAT_COLS = CHANNELS * OUT_HEIGHT * OUT_WIDTH
TOTAL = BATCH * FLAT_COLS


@triton.jit
def _relu_preserve_nan(x):
    return tl.where((x > 0.0) | (x != x), x, 0.0)


@triton.jit
def _take_candidate(candidate, best):
    take = (candidate > best) | ((candidate != candidate) & (best == best))
    return tl.where(take, candidate, best)


@triton.jit
def _relu_maxpool2_flatten_kernel(
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
    v2 = _relu_preserve_nan(tl.load(input_ptr + input_base + WIDTH_, mask=active, other=-float("inf")).to(tl.float32))
    v3 = _relu_preserve_nan(tl.load(input_ptr + input_base + WIDTH_ + 1, mask=active, other=-float("inf")).to(tl.float32))

    best = v0
    best = _take_candidate(v1, best)
    best = _take_candidate(v2, best)
    best = _take_candidate(v3, best)

    tl.store(output_ptr + out_offsets, best, mask=active)


# (T([4,512,14,14], bf16), S([2,2]), S([2,2]), S([4,25088]))
@oracle_impl(hardware="B200", point="ba170fb6", BLOCK=256, num_warps=4)
def oracle_forward(inputs, *, BLOCK: int, num_warps: int):
    x, _kernel_size, _stride, _shape = inputs
    out = torch.empty_strided(
        (BATCH, FLAT_COLS),
        (FLAT_COLS, 1),
        device=x.device,
        dtype=x.dtype,
    )
    _relu_maxpool2_flatten_kernel[(triton.cdiv(TOTAL, BLOCK),)](
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
