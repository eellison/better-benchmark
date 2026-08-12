"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete Swin singleton-window bf16 LayerNorm scope by returning the captured input alias view and using one fixed-hidden Triton row kernel for fp32 population var_mean, eps-before-rsqrt affine, bf16 cast, and the metadata-only view/permute/view output, whereas Inductor lowers the same normalization through the generic normalization schedule with residual bookkeeping for the singleton window path; Inductor cannot do this today because its pattern library has no guarded singleton-window LayerNorm lowering that erases the no-op layout metadata and selects the compact row schedule while preserving the input-view alias; the fix is NEW_PATTERN: add a shape/layout-specialized singleton-window LayerNorm lowering for this fixed hidden-size path."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


ROWS = 6272
HIDDEN = 1024
EPS = 1.0e-5


@triton.autotune(
    configs=[
        triton.Config({"ROW_BLOCK": 1}, num_warps=4, num_stages=3),
        triton.Config({"ROW_BLOCK": 2}, num_warps=4, num_stages=3),
        triton.Config({"ROW_BLOCK": 4}, num_warps=8, num_stages=3),
        triton.Config({"ROW_BLOCK": 8}, num_warps=8, num_stages=3),
    ],
    key=["n_rows"],
)
@triton.jit
def _swin_singleton_layernorm_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    n_rows: tl.constexpr,
    ROW_BLOCK: tl.constexpr,
):
    rows = tl.program_id(0) * ROW_BLOCK + tl.arange(0, ROW_BLOCK)[:, None]
    cols = tl.arange(0, 1024)[None, :]
    offsets = rows * 1024 + cols

    x = tl.load(x_ptr + offsets, eviction_policy="evict_first").to(tl.float32)
    mean = tl.sum(x, axis=1)[:, None] * 0.0009765625
    centered = x - mean
    variance = tl.sum(centered * centered, axis=1)[:, None] * 0.0009765625
    invstd = libdevice.rsqrt(variance + 1.0e-5)

    weight = tl.load(weight_ptr + cols, eviction_policy="evict_last").to(tl.float32)
    bias = tl.load(bias_ptr + cols, eviction_policy="evict_last").to(tl.float32)
    y = centered * invstd * weight + bias
    tl.store(out_ptr + offsets, y)


@oracle_impl(hardware="B200", point="8b70fd76")
def oracle_forward(inputs):
    x, weight, bias = inputs[:3]
    out = torch.empty_strided(
        (ROWS, HIDDEN),
        (HIDDEN, 1),
        device=x.device,
        dtype=torch.bfloat16,
    )
    grid = lambda meta: (triton.cdiv(ROWS, meta["ROW_BLOCK"]),)
    _swin_singleton_layernorm_kernel[grid](
        x,
        weight,
        bias,
        out,
        n_rows=ROWS,
    )
    return x.view(128, 7, 7, 1024), out
