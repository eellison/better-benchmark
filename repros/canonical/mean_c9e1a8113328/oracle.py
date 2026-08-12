"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete bf16 NFNet channels-last spatial mean scope in one shape-specialized Triton row-reduction kernel, reading each `[H,W]` tile once from the captured strided `[N,C,H,W]` input and writing the final keepdim bf16 `[N,C,1,1]` output directly, whereas Inductor lowers standalone `aten.mean.dim(..., [2, 3], keepdim=True)` through its generic reduction scheduler for each captured spatial size; Inductor cannot do this today because its reduction scheduler does not select a tuned `(N,C)` row reduction template for fixed channels-last small/medium spatial means with direct keepdim output stores; the fix is NEW_PATTERN: add a guarded channels-last spatial-mean lowering that autotunes row grouping by spatial size and falls back to the generic schedule when it is already at floor."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _spatial_mean_cl_kernel(
    x_ptr,
    out_ptr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    n = tl.program_id(0)
    c_offsets = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    valid_c = c_offsets < C
    hw_offsets = tl.arange(0, BLOCK_HW)
    acc = tl.zeros((BLOCK_C,), tl.float32)

    for hw_base in range(0, H * W, BLOCK_HW):
        hw = hw_base + hw_offsets
        h_offsets = hw // W
        w_offsets = hw - h_offsets * W
        valid_hw = hw < (H * W)
        offsets = (
            n * (C * H * W)
            + h_offsets[:, None] * (W * C)
            + w_offsets[:, None] * C
            + c_offsets[None, :]
        )
        values = tl.load(x_ptr + offsets, mask=valid_hw[:, None] & valid_c[None, :], other=0.0).to(tl.float32)
        acc += tl.sum(values, axis=0)

    means = acc * (1.0 / (H * W))
    tl.store(out_ptr + n * C + c_offsets, means, mask=valid_c)


@triton.jit
def _spatial_mean_row_kernel(
    x_ptr,
    out_ptr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    n = rows // C
    c = rows - n * C
    valid_rows = rows < (128 * C)
    hw_offsets = tl.arange(0, BLOCK_HW)
    acc = tl.zeros((BLOCK_ROWS,), tl.float32)

    for hw_base in range(0, H * W, BLOCK_HW):
        hw = hw_base + hw_offsets
        h = hw // W
        w = hw - h * W
        valid_hw = hw < (H * W)
        offsets = n[:, None] * (C * H * W) + h[None, :] * (W * C) + w[None, :] * C + c[:, None]
        values = tl.load(
            x_ptr + offsets,
            mask=valid_rows[:, None] & valid_hw[None, :],
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        acc += tl.sum(values, axis=1)

    means = acc * (1.0 / (H * W))
    tl.store(out_ptr + rows, means, mask=valid_rows)


@oracle_impl(hardware="B200", point="69db7514", BLOCK_C=256, BLOCK_HW=64, num_warps=1, num_stages=1, MODE="row")
@oracle_impl(hardware="B200", point="280fa7d2", BLOCK_C=64, BLOCK_HW=256, num_warps=1, num_stages=1, MODE="row")
@oracle_impl(hardware="B200", point="2645a53e", BLOCK_C=64, BLOCK_HW=256, num_warps=1, num_stages=1, MODE="row")
@oracle_impl(hardware="B200", point="4ca56f88", BLOCK_C=64, BLOCK_HW=512, num_warps=8)
@oracle_impl(hardware="B200", point="d1686c77", BLOCK_C=256, BLOCK_HW=64, num_warps=4)
@oracle_impl(hardware="B200", point="333b11d0", BLOCK_C=128, BLOCK_HW=64, num_warps=4)
@oracle_impl(hardware="B200", point="547c8164", BLOCK_C=64, BLOCK_HW=256, num_warps=1, num_stages=1, MODE="row")
@oracle_impl(hardware="B200", point="f0a667e3", BLOCK_C=32, BLOCK_HW=2048, num_warps=8)
@oracle_impl(hardware="B200", point="44f1e75f", BLOCK_C=128, BLOCK_HW=128, num_warps=4)
@oracle_impl(hardware="B200", point="4784273d", BLOCK_C=128, BLOCK_HW=64, num_warps=4)
@oracle_impl(hardware="B200", point="3cf84b69", BLOCK_C=64, BLOCK_HW=128, num_warps=4)
@oracle_impl(hardware="B200", point="2150fa2c", BLOCK_C=64, BLOCK_HW=512, num_warps=8)
@oracle_impl(hardware="B200", point="6b68d184", BLOCK_C=256, BLOCK_HW=64, num_warps=4)
@oracle_impl(hardware="B200", point="d3381898", BLOCK_C=128, BLOCK_HW=64, num_warps=4)
@oracle_impl(hardware="B200", point="473b7062", BLOCK_C=64, BLOCK_HW=128, num_warps=4)
@oracle_impl(hardware="B200", point="630d0797", BLOCK_C=64, BLOCK_HW=512, num_warps=8)
def oracle_forward(inputs, *, BLOCK_C, BLOCK_HW, num_warps, num_stages: int = 3, MODE: str = "cl"):
    (x,) = inputs
    n, c, h, w = (int(dim) for dim in x.shape)
    out = torch.empty_strided((n, c, 1, 1), (c, 1, 1, 1), device=x.device, dtype=torch.bfloat16)
    if MODE == "row":
        _spatial_mean_row_kernel[(triton.cdiv(n * c, BLOCK_C),)](
            x,
            out,
            C=c,
            H=h,
            W=w,
            BLOCK_ROWS=BLOCK_C,
            BLOCK_HW=BLOCK_HW,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    else:
        _spatial_mean_cl_kernel[(n, triton.cdiv(c, BLOCK_C))](
            x,
            out,
            C=c,
            H=h,
            W=w,
            BLOCK_C=BLOCK_C,
            BLOCK_HW=BLOCK_HW,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    return out
