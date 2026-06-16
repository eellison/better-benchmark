"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete T5 bf16 causal relative-position attention scope, including the metadata view of the score tensor, logarithmic relative-position bucket lookup, embedding bias, causal bf16 minimum mask, returned `[8, 8, 1024, 1024]` bias/mask tensor, fp32 softmax score construction from bf16 operands, fp32 last-dimension amax/libdevice.exp/sum/div, bf16 probability cast, expand, and returned `[64, 1024, 1024]` view in one Triton row-softmax kernel, whereas Inductor lowers the decomposed iota/minimum/log/bucket/embedding/permute/where/add/amax/sub/exp/sum/div/cast/expand/view graph as generic producers feeding a separate reduction; Inductor cannot do this today because its pattern library does not recognize T5 relative-position bucket computation and causal masking as structured row-softmax inputs with a sibling returned bias/mask output; the fix is NEW_PATTERN: add a guarded T5 relative-position attention-softmax lowering that fuses bucket lookup, causal mask construction, returned bias/mask stores, bf16 cast boundaries, normalization, and final view emission into the row-kernel schedule."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


@triton.jit
def _t5_bias_mask_softmax_kernel(
    scores_ptr,
    rel_bias_ptr,
    add5_ptr,
    probs_ptr,
    N_HEADS: tl.constexpr,
    Q_LEN: tl.constexpr,
    K_LEN: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0)
    head_query = row // Q_LEN
    head = head_query - (head_query // N_HEADS) * N_HEADS
    query = row - head_query * Q_LEN

    cols = tl.arange(0, BLOCK_K)
    col_mask = cols < K_LEN
    causal = cols <= query
    distance = tl.where(causal, query - cols, 0)

    bucket = distance
    bucket = tl.where(distance >= 16, 16, bucket)
    bucket = tl.where(distance >= 19, 17, bucket)
    bucket = tl.where(distance >= 21, 18, bucket)
    bucket = tl.where(distance >= 24, 19, bucket)
    bucket = tl.where(distance >= 27, 20, bucket)
    bucket = tl.where(distance >= 31, 21, bucket)
    bucket = tl.where(distance >= 35, 22, bucket)
    bucket = tl.where(distance >= 40, 23, bucket)
    bucket = tl.where(distance >= 46, 24, bucket)
    bucket = tl.where(distance >= 52, 25, bucket)
    bucket = tl.where(distance >= 59, 26, bucket)
    bucket = tl.where(distance >= 67, 27, bucket)
    bucket = tl.where(distance >= 77, 28, bucket)
    bucket = tl.where(distance >= 87, 29, bucket)
    bucket = tl.where(distance >= 99, 30, bucket)
    bucket = tl.where(distance >= 113, 31, bucket)

    offsets = row * K_LEN + cols
    rel_bias = tl.load(
        rel_bias_ptr + bucket * N_HEADS + head,
        mask=col_mask,
        other=0.0,
    ).to(tl.float32)

    mask_value = -3.3895313892515355e38
    add5 = tl.where(causal, rel_bias, mask_value)
    tl.store(add5_ptr + offsets, add5, mask=col_mask)

    scores = tl.load(scores_ptr + offsets, mask=col_mask, other=0.0).to(tl.float32)
    logits = scores + rel_bias
    logits = tl.where(causal & col_mask, logits, -float("inf"))

    row_max = tl.max(logits, axis=0)
    numer = libdevice.exp(logits - row_max)
    numer = tl.where(causal & col_mask, numer, 0.0)
    denom = tl.sum(numer, axis=0)
    probs = numer / denom

    tl.store(probs_ptr + offsets, probs.to(tl.bfloat16), mask=col_mask)


# ea4c0a34: (T([64,1024,1024], bf16), T([32,8], bf16), S([8,8,1024,1024]), ...)
@oracle_impl(hardware="B200", point="ea4c0a34", BLOCK_K=1024, num_warps=8, num_stages=3)
def oracle_forward(inputs, *, BLOCK_K: int, num_warps: int, num_stages: int):
    arg0_1, arg1_1, _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5 = inputs
    del _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3

    q_len = int(arg0_1.shape[1])
    k_len = int(arg0_1.shape[2])
    n_heads = int(arg1_1.shape[1])
    n_rows = int(arg0_1.shape[0] * q_len)

    add5 = torch.empty_strided(
        tuple(int(dim) for dim in _shape_param_4),
        (n_heads * q_len * k_len, q_len * k_len, k_len, 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    probs = torch.empty_strided(
        tuple(int(dim) for dim in _shape_param_5),
        tuple(int(stride) for stride in arg0_1.stride()),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )

    _t5_bias_mask_softmax_kernel[(n_rows,)](
        arg0_1,
        arg1_1,
        add5,
        probs,
        N_HEADS=n_heads,
        Q_LEN=q_len,
        K_LEN=k_len,
        BLOCK_K=BLOCK_K,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return add5, probs
