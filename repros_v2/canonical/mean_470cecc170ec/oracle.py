"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete GhostNet bf16 BN-inference affine, explicit bf16 cast, NaN-preserving ReLU, 7x7 spatial mean, and returned keepdim as_strided output in one Triton reduction, whereas Inductor lowers the broadcast normalization producer and spatial mean through generic normalization reduction code; Inductor cannot do this today because the scheduler does not sink the fixed BN-affine/ReLU producer into the small spatial mean while preserving the bf16 cast boundary and output stride; the fix is SCHEDULER_FUSION: add a guarded BN-affine activation spatial-mean lowering that emits the final keepdim as_strided output directly."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


BATCH = 512
CHANNELS = 960
HEIGHT = 7
WIDTH = 7
HW = 49
ROWS = BATCH * CHANNELS


@triton.jit
def _f32_add(a, b):
    return tl.inline_asm_elementwise(
        "add.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _f32_sub(a, b):
    return tl.inline_asm_elementwise(
        "sub.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _f32_mul(a, b):
    return tl.inline_asm_elementwise(
        "mul.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _bn_relu_mean_kernel(
    running_mean_ptr,
    x_ptr,
    running_var_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    TOTAL_ROWS: tl.constexpr,
    C: tl.constexpr,
    W: tl.constexpr,
    HW_SIZE: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    hw_offsets = tl.arange(0, BLOCK_HW)
    row_mask = rows < TOTAL_ROWS
    hw_mask = hw_offsets < HW_SIZE

    n_idx = rows // C
    channel = rows - n_idx * C
    h_idx = hw_offsets // W
    w_idx = hw_offsets - h_idx * W
    x_offsets = (
        n_idx[:, None] * (C * HW_SIZE)
        + channel[:, None]
        + h_idx[None, :] * (W * C)
        + w_idx[None, :] * C
    )
    mask = row_mask[:, None] & hw_mask[None, :]

    x = tl.load(x_ptr + x_offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(running_mean_ptr + channel, mask=row_mask, other=0.0).to(tl.float32)
    var = tl.load(running_var_ptr + channel, mask=row_mask, other=1.0).to(tl.float32)
    weight = tl.load(weight_ptr + channel, mask=row_mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + channel, mask=row_mask, other=0.0).to(tl.float32)

    sqrt_var = libdevice.sqrt(_f32_add(var, 1.0e-5))
    invstd = 1.0 / sqrt_var
    centered = _f32_sub(x, mean[:, None])
    normalized = _f32_mul(centered, invstd[:, None])
    scaled = _f32_mul(normalized, weight[:, None])
    affine_bf16 = _f32_add(scaled, bias[:, None]).to(tl.bfloat16)
    zero = tl.full((BLOCK_ROWS, BLOCK_HW), 0.0, tl.bfloat16)
    relu = tl.where(affine_bf16 < zero, zero, affine_bf16)
    summed = tl.sum(tl.where(mask, relu.to(tl.float32), 0.0), axis=1)
    pooled = summed * (1.0 / 49.0)
    tl.store(out_ptr + rows, pooled.to(tl.bfloat16), mask=row_mask)


@oracle_impl(hardware="B200", point="3e244c1d", BLOCK_ROWS=16, BLOCK_HW=64, num_warps=4)
def oracle_forward(inputs, *, BLOCK_ROWS, BLOCK_HW, num_warps):
    running_mean, x, running_var, weight, bias, _shape, _stride = inputs
    out = torch.empty_strided(
        (BATCH, CHANNELS, 1, 1),
        (CHANNELS, 1, CHANNELS, CHANNELS),
        device=x.device,
        dtype=torch.bfloat16,
    )
    _bn_relu_mean_kernel[(triton.cdiv(ROWS, BLOCK_ROWS),)](
        running_mean,
        x,
        running_var,
        weight,
        bias,
        out,
        TOTAL_ROWS=ROWS,
        C=CHANNELS,
        W=WIDTH,
        HW_SIZE=HW,
        BLOCK_ROWS=BLOCK_ROWS,
        BLOCK_HW=BLOCK_HW,
        num_warps=num_warps,
        num_stages=3,
    )
    return out
