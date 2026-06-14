"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete GhostNet bf16 ReLU, returned `[512, 1280]` metadata view, and sibling `relu <= 0` bool mask in one storage-linear Triton load/max/two-store pass, whereas Inductor already lowers this isolated pointwise fanout and view-only reshape into the same mandatory input-read, bf16-output-write, and bool-mask-write envelope with generic pointwise scheduling overhead. Inductor cannot materially improve this local repro through reduction templates, scatter handling, split-K, recomputation, or algebraic elimination because there is no producer or consumer context in scope and both returned tensors are observable outputs. The fix is BANDWIDTH_BOUND: record this as a pointwise memory/launch floor unless broader pointwise codegen, launch, or allocation overhead changes move both implementations."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _relu_mask_kernel(
    input_ptr,
    relu_out_ptr,
    mask_out_ptr,
    n_elements: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = offsets < n_elements
    x = tl.load(input_ptr + offsets, mask=valid, other=0.0)
    relu = tl.where(x != x, x, tl.maximum(x, 0.0))
    tl.store(relu_out_ptr + offsets, relu, mask=valid)
    tl.store(mask_out_ptr + offsets, x <= 0.0, mask=valid)


def _contiguous_stride(shape):
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= dim
    return tuple(reversed(stride))


# 21720c2b: (T([512,1280,1,1], bf16), S([512,1280]))
@oracle_impl(hardware="B200", point="21720c2b", BLOCK=1024, num_warps=4)
def oracle_forward(inputs, *, BLOCK: int, num_warps: int):
    arg0_1, _shape_param_0 = inputs
    view_shape = tuple(int(dim) for dim in _shape_param_0)
    mask_shape = tuple(arg0_1.shape)

    view = torch.empty_strided(
        view_shape,
        _contiguous_stride(view_shape),
        device=arg0_1.device,
        dtype=arg0_1.dtype,
    )
    le = torch.empty_strided(
        mask_shape,
        _contiguous_stride(mask_shape),
        device=arg0_1.device,
        dtype=torch.bool,
    )

    n_elements = arg0_1.numel()
    _relu_mask_kernel[(triton.cdiv(n_elements, BLOCK),)](
        arg0_1,
        view,
        le,
        n_elements,
        BLOCK=BLOCK,
        num_warps=num_warps,
        num_stages=3,
    )
    return view, le
