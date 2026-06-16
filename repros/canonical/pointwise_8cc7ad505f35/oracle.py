"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete token-mask fanout scope in one storage-linear Triton pointwise kernel, including the `arg0 != 1` int32 materialization and the returned valid-token bool unsqueeze view backed by a fresh `[B,S]` mask for both Longformer and Roberta points, whereas Inductor already lowers this isolated integer-mask graph to the same mandatory int64 read plus int32 and bool output traffic with generic pointwise scheduling; Inductor cannot materially improve this local repro through scheduler fusion, scatter-reduce, split-K, algebraic elimination, or recompute fusion because there is no surrounding producer or consumer and both returned tensors must be materialized; the fix is BANDWIDTH_BOUND: record this as an at-floor pointwise fanout case unless broader pointwise launch or memory-codegen work moves both implementations."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _token_mask_fanout_kernel(
    x_ptr,
    ne_i32_ptr,
    valid_ptr,
    n_elements: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=1)
    non_pad = x != 1
    valid = (x >= 0) & (x < 50265) & non_pad

    tl.store(ne_i32_ptr + offsets, non_pad.to(tl.int32), mask=mask)
    tl.store(valid_ptr + offsets, valid, mask=mask)


def _launch(inputs, *, BLOCK: int, num_warps: int):
    (x,) = inputs
    out_i32 = torch.empty_strided(
        tuple(x.shape),
        tuple(x.stride()),
        device=x.device,
        dtype=torch.int32,
    )
    valid_base = torch.empty_strided(
        tuple(x.shape),
        tuple(x.stride()),
        device=x.device,
        dtype=torch.bool,
    )

    _token_mask_fanout_kernel[(triton.cdiv(x.numel(), BLOCK),)](
        x,
        out_i32,
        valid_base,
        n_elements=x.numel(),
        BLOCK=BLOCK,
        num_warps=num_warps,
    )
    return out_i32, valid_base.unsqueeze(-1)


# 0105f520: i64[8,1024] -> i32[8,1024], bool[8,1024,1]
@oracle_impl(hardware="B200", point="0105f520", BLOCK=1024, num_warps=4)
# 26cc4258: i64[32,512] -> i32[32,512], bool[32,512,1]
@oracle_impl(hardware="B200", point="26cc4258", BLOCK=1024, num_warps=4)
def oracle_forward(inputs, *, BLOCK: int, num_warps: int):
    return _launch(inputs, BLOCK=BLOCK, num_warps=num_warps)
