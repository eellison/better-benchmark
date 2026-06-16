"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete Gemma causal attention-mask fanout by materializing all five distinct bf16 `[1,1,1000,1000]` `0/-inf` bases in one Triton kernel and returning the exact zero-head-stride `[1,8,1000,1000]` expanded views, whereas Inductor lowers each repeated iota/compare/where/expand branch as generic pointwise output work; Inductor cannot do this today because the pointwise scheduler does not coalesce identical shape-derived causal-mask siblings into one multi-output store while preserving distinct output storages and expanded view metadata; the fix is SCHEDULER_FUSION: add graph-output sibling fusion for repeated generated causal masks with metadata-only head expansion."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


NUM_OUTPUTS = 5
BATCH = 1
HEADS = 8
Q_LEN = 1000
K_LEN = 1000
BASE_SHAPE = (BATCH, 1, Q_LEN, K_LEN)
BASE_STRIDE = (Q_LEN * K_LEN, Q_LEN * K_LEN, K_LEN, 1)
EXPANDED_SHAPE = (BATCH, HEADS, Q_LEN, K_LEN)
BASE_NUMEL = BATCH * Q_LEN * K_LEN


@triton.autotune(
    configs=[
        triton.Config({"BLOCK": 256}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK": 512}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK": 1024}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK": 2048}, num_warps=8, num_stages=4),
    ],
    key=["n_elements"],
)
@triton.jit
def _causal_mask_5_kernel(
    out0,
    out1,
    out2,
    out3,
    out4,
    n_elements: tl.constexpr,
    q_len: tl.constexpr,
    k_len: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    key = offsets % k_len
    query = (offsets // k_len) % q_len
    values = tl.where(key <= query, 0.0, -float("inf"))

    tl.store(out0 + offsets, values, mask=mask)
    tl.store(out1 + offsets, values, mask=mask)
    tl.store(out2 + offsets, values, mask=mask)
    tl.store(out3 + offsets, values, mask=mask)
    tl.store(out4 + offsets, values, mask=mask)


def _make_base(device):
    return torch.empty_strided(
        BASE_SHAPE,
        BASE_STRIDE,
        device=device,
        dtype=torch.bfloat16,
    )


# d7517139: Gemma shape-only five-output causal-mask fanout.
@oracle_impl(hardware="B200", point="d7517139")
def oracle_forward(inputs):
    del inputs
    device = torch.device("cuda", torch.cuda.current_device())
    bases = tuple(_make_base(device) for _ in range(NUM_OUTPUTS))

    grid = lambda meta: (triton.cdiv(BASE_NUMEL, meta["BLOCK"]),)
    _causal_mask_5_kernel[grid](
        bases[0],
        bases[1],
        bases[2],
        bases[3],
        bases[4],
        n_elements=BASE_NUMEL,
        q_len=Q_LEN,
        k_len=K_LEN,
    )
    return tuple(base.expand(EXPANDED_SHAPE) for base in bases)
