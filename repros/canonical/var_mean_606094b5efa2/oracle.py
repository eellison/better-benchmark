"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete BEiT bf16 residual-scale LayerNorm training scope in one Triton row-reduction kernel, including the bf16 `[25216,768] -> [128,197,768]` view, fp32 hidden-vector scale, returned fp32 residual add, population `var_mean(..., dim=2, correction=0, keepdim=True)`, eps=1e-6 `libdevice.rsqrt`, returned mean and rsqrt tensors, fp32 affine epilogue, final bf16 cast, and flattened output view, whereas Inductor lowers the returned add, row statistics, saved-stat side outputs, affine cast, and final view through its generic normalization schedule; Inductor cannot do this today because the scheduler/codegen LayerNorm template does not keep an observable bf16 producer, multiple fp32 saved-stat outputs, and direct bf16 view-compatible stores in one B200-tuned fixed-hidden plan; the fix is SCHEDULER_FUSION: teach LayerNorm scheduling to fuse broadcast scale/residual producers while emitting returned add, mean, inverse-std, and final cast/view outputs from one guarded row plan."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


EPS = 1.0e-6
ROWS = 25216
HIDDEN = 768
BATCH = 128
TOKENS = 197


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
def _f32_sub(a, b):
    return tl.inline_asm_elementwise(
        "sub.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.autotune(
    configs=[
        triton.Config({"ROW_BLOCK": 1}, num_warps=4, num_stages=3),
        triton.Config({"ROW_BLOCK": 1}, num_warps=8, num_stages=3),
        triton.Config({"ROW_BLOCK": 2}, num_warps=4, num_stages=3),
        triton.Config({"ROW_BLOCK": 4}, num_warps=4, num_stages=3),
    ],
    key=["HIDDEN_SIZE"],
)
@triton.jit
def _beit_layernorm_multiout_kernel(
    flat_bf16_ptr,
    gamma_ptr,
    residual_ptr,
    weight_ptr,
    bias_ptr,
    add_ptr,
    mean_ptr,
    rsqrt_ptr,
    final_bf16_ptr,
    TOTAL_ROWS: tl.constexpr,
    HIDDEN_SIZE: tl.constexpr,
    EPSILON: tl.constexpr,
    BLOCK_H: tl.constexpr,
    ROW_BLOCK: tl.constexpr,
):
    row_ids = tl.program_id(0) * ROW_BLOCK + tl.arange(0, ROW_BLOCK)
    rows = row_ids[:, None]
    cols = tl.arange(0, BLOCK_H)[None, :]
    row_mask = row_ids[:, None] < TOTAL_ROWS
    col_mask = cols < HIDDEN_SIZE
    mask = row_mask & col_mask
    offsets = rows * HIDDEN_SIZE + cols

    flat = tl.load(
        flat_bf16_ptr + offsets,
        mask=mask,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)
    gamma = tl.load(
        gamma_ptr + cols,
        mask=col_mask,
        other=0.0,
        eviction_policy="evict_last",
    ).to(tl.float32)
    residual = tl.load(
        residual_ptr + offsets,
        mask=mask,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)

    scaled = _f32_mul(gamma, flat)
    x = _f32_add(residual, scaled)
    tl.store(add_ptr + offsets, x, mask=mask)

    x_for_sum = tl.where(mask, x, 0.0)
    mean_1d = tl.sum(x_for_sum, axis=1) / HIDDEN_SIZE
    mean = mean_1d[:, None]
    centered = _f32_sub(x, mean)
    variance_1d = (
        tl.sum(tl.where(mask, _f32_mul(centered, centered), 0.0), axis=1)
        / HIDDEN_SIZE
    )
    invstd_1d = libdevice.rsqrt(_f32_add(variance_1d, EPSILON))
    invstd = invstd_1d[:, None]

    weight = tl.load(
        weight_ptr + cols,
        mask=col_mask,
        other=0.0,
        eviction_policy="evict_last",
    ).to(tl.float32)
    bias = tl.load(
        bias_ptr + cols,
        mask=col_mask,
        other=0.0,
        eviction_policy="evict_last",
    ).to(tl.float32)
    normalized = _f32_mul(centered, invstd)
    affine = _f32_add(_f32_mul(normalized, weight), bias)

    tl.store(mean_ptr + row_ids, mean_1d, mask=row_ids < TOTAL_ROWS)
    tl.store(rsqrt_ptr + row_ids, invstd_1d, mask=row_ids < TOTAL_ROWS)
    tl.store(final_bf16_ptr + offsets, affine.to(tl.bfloat16), mask=mask)


# (T([25216,768], bf16), T([768], f32), T([128,197,768], f32), T([768], f32), T([768], f32), S([128,197,768]), S([25216,768]))
@oracle_impl(hardware="B200", point="f4c82f7a", BLOCK_H=1024)
def oracle_forward(inputs, *, BLOCK_H: int):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, _shape_param_0, _shape_param_1 = inputs

    add = torch.empty_strided(
        (BATCH, TOKENS, HIDDEN),
        (TOKENS * HIDDEN, HIDDEN, 1),
        device=arg0_1.device,
        dtype=torch.float32,
    )
    mean = torch.empty_strided(
        (BATCH, TOKENS, 1),
        (TOKENS, 1, 1),
        device=arg0_1.device,
        dtype=torch.float32,
    )
    rsqrt = torch.empty_strided(
        (BATCH, TOKENS, 1),
        (TOKENS, 1, 1),
        device=arg0_1.device,
        dtype=torch.float32,
    )
    final_base = torch.empty_strided(
        (BATCH, TOKENS, HIDDEN),
        (TOKENS * HIDDEN, HIDDEN, 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )

    _beit_layernorm_multiout_kernel[(lambda meta: (triton.cdiv(ROWS, meta["ROW_BLOCK"]),))](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        add,
        mean,
        rsqrt,
        final_base,
        TOTAL_ROWS=ROWS,
        HIDDEN_SIZE=HIDDEN,
        EPSILON=EPS,
        BLOCK_H=BLOCK_H,
    )
    return add, mean, rsqrt, final_base.view(tuple(int(dim) for dim in _shape_param_1))
