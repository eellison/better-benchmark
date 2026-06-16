"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete transformer residual accumulation scope as one storage-linear Triton kernel over the contiguous inputs, erasing the metadata-only 2D-to-3D views, widening each bf16 residual operand to fp32, preserving the captured left-associated add order, and storing the fresh contiguous fp32 output, whereas Inductor lowers the same view/convert/add/add/add graph through its generic pointwise scheduler; Inductor cannot do this today because pointwise codegen has no guarded B200 template for metadata-only view erasure plus mixed bf16/fp32 residual accumulation that emits the direct storage-order traversal with exact fp32 add boundaries; the fix is NEW_PATTERN: add a view-linearized mixed-dtype residual-add pointwise lowering for contiguous transformer activations."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


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
def _residual_add_kernel(
    residual0_ptr,
    base_ptr,
    residual1_ptr,
    residual2_ptr,
    out_ptr,
    n_elements,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements

    base = tl.load(base_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    residual0 = tl.load(residual0_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    residual1 = tl.load(residual1_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    residual2 = tl.load(residual2_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    out = _f32_add(base, residual0)
    out = _f32_add(out, residual1)
    out = _f32_add(out, residual2)
    tl.store(out_ptr + offsets, out, mask=mask)


# 18d1aea1: (T([8192,1024], bf16), T([8,1024,1024], f32), T([8192,1024], bf16), T([8192,1024], bf16), ...)
@oracle_impl(hardware="B200", point="18d1aea1", BLOCK=1024, num_warps=4, num_stages=4)
# dd4880ac: (T([16384,768], bf16), T([16,1024,768], f32), T([16384,768], bf16), T([16384,768], bf16), ...)
@oracle_impl(hardware="B200", point="dd4880ac", BLOCK=1024, num_warps=4, num_stages=4)
# 61418aa9: (T([16384,1024], bf16), T([64,256,1024], f32), T([16384,1024], bf16), T([16384,1024], bf16), ...)
@oracle_impl(hardware="B200", point="61418aa9", BLOCK=1024, num_warps=4, num_stages=4)
def oracle_forward(inputs, *, BLOCK, num_warps, num_stages):
    residual0, base, residual1, residual2, _shape_param_0, _shape_param_1, _shape_param_2 = inputs
    del _shape_param_1, _shape_param_2

    out_shape = tuple(int(dim) for dim in _shape_param_0)
    out = torch.empty_strided(
        out_shape,
        (out_shape[1] * out_shape[2], out_shape[2], 1),
        device=base.device,
        dtype=torch.float32,
    )
    n_elements = out.numel()
    grid = (triton.cdiv(n_elements, BLOCK),)
    _residual_add_kernel[grid](
        residual0,
        base,
        residual1,
        residual2,
        out,
        n_elements,
        BLOCK=BLOCK,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out
