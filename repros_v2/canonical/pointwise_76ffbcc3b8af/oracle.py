"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete AlexNet bf16 ReLU -> 3x3 stride-2 low-memory maxpool values -> shape-identical adaptive_avg_pool2d -> final `[128,9216]` view by writing the flattened pooled tensor directly from one Triton stencil, whereas Inductor lowers the captured ReLU producer, tuple-producing maxpool stencil, identity adaptive pool, and layout-only view through generic scheduling around an unused int8 offsets output; Inductor cannot do this today because its scheduler/codegen does not split `_low_memory_max_pool_with_offsets` into a returned-value-only lowering while sinking the pointwise ReLU and identity/layout consumers into the same store schedule; the fix is SCHEDULER_FUSION: teach the maxpool scheduler to DCE dead offsets and emit a single ReLU/maxpool/value-view stencil for static 3x3 stride-2 cases."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


BATCH = 128
CHANNELS = 256
H_IN = 13
W_IN = 13
H_OUT = 6
W_OUT = 6
FLAT = CHANNELS * H_OUT * W_OUT
TOTAL = BATCH * FLAT


@triton.jit
def _relu_preserve_nan(x):
    return tl.where(x <= 0.0, 0.0, x)


@triton.jit
def _take_value(candidate, best):
    take = (candidate > best) | ((candidate != candidate) & (best == best))
    return tl.where(take, candidate, best)


@triton.jit
def _relu_maxpool3_flat_kernel(
    input_ptr,
    out_ptr,
    total: tl.constexpr,
    channels: tl.constexpr,
    h_in: tl.constexpr,
    w_in: tl.constexpr,
    h_out: tl.constexpr,
    w_out: tl.constexpr,
    BLOCK: tl.constexpr,
):
    linear = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active = linear < total

    out_w = linear % w_out
    tmp = linear // w_out
    out_h = tmp % h_out
    tmp = tmp // h_out
    channel = tmp % channels
    batch = tmp // channels
    input_base = (batch * channels + channel) * (h_in * w_in)
    window_base = input_base + out_h * (2 * w_in) + out_w * 2

    x00 = _relu_preserve_nan(tl.load(input_ptr + window_base, mask=active, other=-float("inf")).to(tl.float32))
    x01 = _relu_preserve_nan(tl.load(input_ptr + window_base + 1, mask=active, other=-float("inf")).to(tl.float32))
    x02 = _relu_preserve_nan(tl.load(input_ptr + window_base + 2, mask=active, other=-float("inf")).to(tl.float32))
    x10 = _relu_preserve_nan(tl.load(input_ptr + window_base + w_in, mask=active, other=-float("inf")).to(tl.float32))
    x11 = _relu_preserve_nan(tl.load(input_ptr + window_base + w_in + 1, mask=active, other=-float("inf")).to(tl.float32))
    x12 = _relu_preserve_nan(tl.load(input_ptr + window_base + w_in + 2, mask=active, other=-float("inf")).to(tl.float32))
    x20 = _relu_preserve_nan(tl.load(input_ptr + window_base + 2 * w_in, mask=active, other=-float("inf")).to(tl.float32))
    x21 = _relu_preserve_nan(tl.load(input_ptr + window_base + 2 * w_in + 1, mask=active, other=-float("inf")).to(tl.float32))
    x22 = _relu_preserve_nan(tl.load(input_ptr + window_base + 2 * w_in + 2, mask=active, other=-float("inf")).to(tl.float32))

    best = x00
    best = _take_value(x01, best)
    best = _take_value(x02, best)
    best = _take_value(x10, best)
    best = _take_value(x11, best)
    best = _take_value(x12, best)
    best = _take_value(x20, best)
    best = _take_value(x21, best)
    best = _take_value(x22, best)

    tl.store(out_ptr + linear, best, mask=active)


@oracle_impl(hardware="B200", point="a8ee30c6", BLOCK=128, num_warps=4)
def oracle_forward(inputs, *, BLOCK: int, num_warps: int):
    x, _kernel_size, _stride, flat_shape = inputs
    out = torch.empty_strided(
        tuple(int(dim) for dim in flat_shape),
        (FLAT, 1),
        device=x.device,
        dtype=torch.bfloat16,
    )
    _relu_maxpool3_flat_kernel[(triton.cdiv(TOTAL, BLOCK),)](
        x,
        out,
        total=TOTAL,
        channels=CHANNELS,
        h_in=H_IN,
        w_in=W_IN,
        h_out=H_OUT,
        w_out=W_OUT,
        BLOCK=BLOCK,
        num_warps=num_warps,
        num_stages=4,
    )
    return out
