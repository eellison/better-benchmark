"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete ConvNeXt mixed-dtype channels-last pointwise scope in one storage-linear Triton kernel, preserving the `bf16 + f32 -> f32` add result and the sibling bf16 cast output with the exact dense channels-last strides; Inductor already lowers this isolated two-op region to the same mandatory two-read/two-write pointwise envelope, with no producer/consumer fusion, scatter-reduce, split-K, or algebraic elimination opportunity left inside the repro; the fix is BANDWIDTH_BOUND: record this as an at-floor pointwise memory-bandwidth case unless broader pointwise launch, indexing, or memory-codegen improvements move both implementations together."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


N = 128
C = 80
H = 56
W = 56
NUMEL = N * C * H * W


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
def _add_cast_kernel(
    x_ptr,
    y_ptr,
    out_f32_ptr,
    out_bf16_ptr,
    TOTAL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < TOTAL
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    value = _f32_add(x, y)
    tl.store(out_f32_ptr + offsets, value, mask=mask)
    tl.store(
        out_bf16_ptr + offsets,
        value.to(tl.bfloat16, fp_downcast_rounding="rtne"),
        mask=mask,
    )


@oracle_impl(hardware="B200", point="2a76d54a", BLOCK=1024, num_warps=4)
def oracle_forward(inputs, *, BLOCK: int, num_warps: int):
    x, y = inputs
    out_f32 = torch.empty_strided(
        (N, C, H, W),
        (C * H * W, 1, W * C, C),
        device=x.device,
        dtype=torch.float32,
    )
    out_bf16 = torch.empty_strided(
        (N, C, H, W),
        (C * H * W, 1, W * C, C),
        device=x.device,
        dtype=torch.bfloat16,
    )
    _add_cast_kernel[(triton.cdiv(NUMEL, BLOCK),)](
        x,
        y,
        out_f32,
        out_bf16,
        TOTAL=NUMEL,
        BLOCK=BLOCK,
        num_warps=num_warps,
        num_stages=4,
    )
    return out_f32, out_bf16
