"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete AlexNet bool-mask scale and metadata-view scope in one storage-linear Triton pointwise kernel, preserving the eager `bool -> bf16`, `* 2.0`, and bf16 activation multiply before returning the fresh contiguous buffer as `[128,256,6,6]`; Inductor already lowers this isolated convert/mul/mul/view graph to the same one-kernel memory-traffic envelope, with no surrounding producer or consumer to fuse and no reduction, scatter, split-K, or algebraic simplification opportunity inside the repro; the fix is BANDWIDTH_BOUND: record this as an at-floor pointwise case unless broader pointwise bandwidth, launch-overhead, or generic indexing improvements move both implementations together."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


N = 128
C = 256
H = 6
W = 6
NUMEL = N * C * H * W


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
def _mask_scale_kernel(
    mask_ptr,
    value_ptr,
    out_ptr,
    TOTAL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    in_bounds = offsets < TOTAL

    mask_values = tl.load(mask_ptr + offsets, mask=in_bounds, other=0).to(tl.float32)
    values = tl.load(value_ptr + offsets, mask=in_bounds, other=0.0).to(tl.float32)
    scale = _f32_mul(mask_values, 2.0)
    result = _f32_mul(values, scale)
    tl.store(
        out_ptr + offsets,
        result.to(tl.bfloat16, fp_downcast_rounding="rtne"),
        mask=in_bounds,
    )


@oracle_impl(hardware="B200", point="44864090", BLOCK=1024, num_warps=4)
def oracle_forward(inputs, *, BLOCK: int, num_warps: int):
    mask, values, _shape_param_0 = inputs
    out = torch.empty_strided(
        (N, C, H, W),
        (C * H * W, H * W, W, 1),
        device=values.device,
        dtype=torch.bfloat16,
    )
    _mask_scale_kernel[(triton.cdiv(NUMEL, BLOCK),)](
        mask,
        values,
        out,
        TOTAL=NUMEL,
        BLOCK=BLOCK,
        num_warps=num_warps,
        num_stages=4,
    )
    return out
