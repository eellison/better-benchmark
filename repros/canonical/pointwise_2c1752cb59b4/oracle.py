"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete bf16 XLNet dual bias-add layout scope in one storage-linear Triton kernel, materializing the two fresh add buffers and returning their captured non-contiguous `[256,512,64]` views, whereas Inductor lowers the view/permute/add/view chain through its generic pointwise output planner; Inductor cannot do this today because the scheduler does not recognize this paired XLNet head-layout bias-add as a dedicated shape-specialized multi-output materialization with alias-only view epilogues; the fix is SCHEDULER_FUSION: add a guarded XLNet bias-add layout schedule that emits both add outputs directly in storage order and returns the metadata views without generic shape-param planning."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK": 1024}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK": 2048}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK": 2048}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK": 4096}, num_warps=8, num_stages=4),
    ],
    key=["N_ELEMENTS"],
)
@triton.jit
def _xlnet_dual_bias_add_kernel(
    x_ptr,
    bias0_ptr,
    bias1_ptr,
    out0_ptr,
    out1_ptr,
    N_ELEMENTS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    bias_offsets = offsets % 1024

    x = tl.load(x_ptr + offsets).to(tl.float32)
    bias0 = tl.load(bias0_ptr + bias_offsets, eviction_policy="evict_last").to(tl.float32)
    bias1 = tl.load(bias1_ptr + bias_offsets, eviction_policy="evict_last").to(tl.float32)

    tl.store(out0_ptr + offsets, x + bias0)
    tl.store(out1_ptr + offsets, x + bias1)


@oracle_impl(hardware="B200", point="34c69289")
def oracle_forward(inputs):
    x, bias0, bias1, _shape0, _shape1, out_shape0, out_shape1 = inputs
    out0_base = torch.empty_strided(
        (512, 16, 16, 64),
        (16384, 1024, 64, 1),
        device=x.device,
        dtype=torch.bfloat16,
    )
    out1_base = torch.empty_strided(
        (512, 16, 16, 64),
        (16384, 1024, 64, 1),
        device=x.device,
        dtype=torch.bfloat16,
    )
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK"]),)
    _xlnet_dual_bias_add_kernel[grid](
        x,
        bias0,
        bias1,
        out0_base,
        out1_base,
        N_ELEMENTS=n_elements,
    )
    out_shape0 = tuple(int(dim) for dim in out_shape0)
    out_shape1 = tuple(int(dim) for dim in out_shape1)
    return (
        out0_base.as_strided(out_shape0, (64, 16384, 1)),
        out1_base.as_strided(out_shape1, (64, 16384, 1)),
    )
