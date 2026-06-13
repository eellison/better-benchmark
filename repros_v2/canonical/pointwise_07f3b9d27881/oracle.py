"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete T5 bf16 bool-mask scale/where pointwise scope by writing the fresh contiguous `[8,1024,2048]` backing tensor in one storage-linear Triton pass, creating and returning the required bf16 scalar zero, the `[8192,2048]` view, and the `[2048,8192]` transpose alias, whereas Inductor already lowers this isolated dense pointwise and metadata-view graph to equivalent fused materialization with only generic indexing/codegen overhead; Inductor cannot materially improve this local repro through scheduler fusion, scatter-reduce, split-K, algebraic elimination, or recomputation because the output contract requires dense bf16 input/mask reads, bf16 rounded scale multiplication, and a large bf16 backing allocation; the fix is BANDWIDTH_BOUND: record this as a pointwise memory floor unless broader pointwise memory-codegen or launch/allocation work moves both implementations."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


SCALE_BF16 = 1.109375


@triton.jit
def _masked_scale_where_kernel(
    x_ptr,
    scale_mask_ptr,
    zero_mask_ptr,
    zero_ptr,
    out_ptr,
    total: tl.constexpr,
    scale: tl.constexpr,
    block_size: tl.constexpr,
):
    offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    mask = offsets < total

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    scale_mask = tl.load(scale_mask_ptr + offsets, mask=mask, other=0).to(tl.float32)
    zero_mask = tl.load(zero_mask_ptr + offsets, mask=mask, other=0) != 0
    scaled = x * (scale_mask * scale)
    out = tl.where(zero_mask, 0.0, scaled)
    tl.store(out_ptr + offsets, out, mask=mask)

    if tl.program_id(0) == 0:
        tl.store(zero_ptr, tl.full((), 0.0, tl.float32))


# 363c89c3: (T([8192,2048], bf16), T([8,1024,2048], b8), T([8,1024,2048], b8), S([8,1024,2048]), S([8192,2048]))
@oracle_impl(hardware="B200", point="363c89c3", BLOCK_SIZE=1024, num_warps=4)
def oracle_forward(inputs, *, BLOCK_SIZE: int, num_warps: int):
    arg0_1, arg1_1, arg2_1, _shape_param_0, _shape_param_1 = inputs
    base_shape = tuple(int(dim) for dim in _shape_param_0)
    flat_shape = tuple(int(dim) for dim in _shape_param_1)
    base = torch.empty_strided(
        base_shape,
        (base_shape[1] * base_shape[2], base_shape[2], 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    full = torch.empty((), device=arg0_1.device, dtype=torch.bfloat16)

    _masked_scale_where_kernel[(triton.cdiv(arg0_1.numel(), BLOCK_SIZE),)](
        arg0_1,
        arg1_1,
        arg2_1,
        full,
        base,
        total=arg0_1.numel(),
        scale=SCALE_BF16,
        block_size=BLOCK_SIZE,
        num_warps=num_warps,
        num_stages=1,
    )
    flat = base.view(flat_shape)
    return full, flat, flat.permute(1, 0)
