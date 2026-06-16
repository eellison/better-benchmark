"""Gap diagnosis (classification: SCATTER_REDUCE): this oracle computes the complete XLNet masked embedding-gradient scatter scope by zero-initializing the `[32000,1024]` fp32 accumulation table, folding the three bf16 residual inputs, fp32 residual input, bool dropout scale, and token validity predicate into one duplicate-preserving row scatter-add, then adding the widened bf16 vocabulary base in the final dense pass, whereas Inductor lowers the view/squeeze casts, residual/dropout producer, validity mask, zero full, `_unsafe_masked_index_put_accumulate`, and dense add as generic pointwise plus indexed-update regions; Inductor cannot do this today because scheduler/codegen does not represent this graph as a structured masked row scatter-reduce with an exact producer and dense base-add epilogue; the fix is SCATTER_REDUCE: add a guarded embedding-gradient scatter lowering that folds validity checks into atomic row updates and fuses the residual/dropout producer while preserving the final base-plus-scatter output contract."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _add_rn(a, b):
    return tl.inline_asm_elementwise(
        "add.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _mul_rn(a, b):
    return tl.inline_asm_elementwise(
        "mul.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _zero_kernel(
    out_ptr,
    n_elements: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(out_ptr + offsets, tl.zeros((BLOCK,), tl.float32), mask=offsets < n_elements)


@triton.jit
def _xlnet_scatter_kernel(
    arg1_ptr,
    arg2_ptr,
    arg3_ptr,
    arg4_ptr,
    drop_ptr,
    index_ptr,
    out_ptr,
    hidden: tl.constexpr,
    vocab: tl.constexpr,
    total_sources: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    source = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    col_mask = cols < hidden

    token = tl.load(index_ptr + source, mask=source < total_sources, other=-1).to(tl.int64)
    active = (source < total_sources) & (token >= 0) & (token < vocab) & (token != -1)
    offsets = source * hidden + cols

    v0 = tl.load(arg2_ptr + offsets, mask=active & col_mask, other=0.0).to(tl.float32)
    v1 = tl.load(arg1_ptr + offsets, mask=active & col_mask, other=0.0).to(tl.float32)
    v2 = tl.load(arg3_ptr + offsets, mask=active & col_mask, other=0.0).to(tl.float32)
    v3 = tl.load(arg4_ptr + offsets, mask=active & col_mask, other=0.0).to(tl.float32)
    keep = tl.load(drop_ptr + offsets, mask=active & col_mask, other=0).to(tl.float32)

    add0 = _add_rn(v0, v1)
    add1 = _add_rn(add0, v2)
    add2 = _add_rn(add1, v3)
    scale = _mul_rn(keep, 1.1111111111111112)
    value = _mul_rn(add2, scale)

    tl.atomic_add(
        out_ptr + token * hidden + cols,
        value,
        sem="relaxed",
        mask=active & col_mask,
    )


@triton.jit
def _add_base_kernel(
    base_ptr,
    out_ptr,
    n_elements: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    base = tl.load(base_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    accum = tl.load(out_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + offsets, _add_rn(base, accum), mask=mask)


@oracle_impl(
    hardware="B200",
    point="7cee22f2",
    DENSE_BLOCK=1024,
    BLOCK_H=1024,
)
def oracle_forward(
    inputs,
    *,
    DENSE_BLOCK: int,
    BLOCK_H: int,
):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, *_shape_params = inputs
    out = torch.empty_strided((32000, 1024), (1024, 1), device=arg0_1.device, dtype=torch.float32)
    n_elements = 32000 * 1024

    _zero_kernel[(triton.cdiv(n_elements, DENSE_BLOCK),)](
        out,
        n_elements=n_elements,
        BLOCK=DENSE_BLOCK,
        num_warps=4,
    )
    _xlnet_scatter_kernel[(8192, triton.cdiv(1024, BLOCK_H))](
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        arg5_1,
        arg6_1,
        out,
        hidden=1024,
        vocab=32000,
        total_sources=8192,
        BLOCK_H=BLOCK_H,
        num_warps=8,
    )
    _add_base_kernel[(triton.cdiv(n_elements, DENSE_BLOCK),)](
        arg0_1,
        out,
        n_elements=n_elements,
        BLOCK=DENSE_BLOCK,
        num_warps=4,
    )
    return out
