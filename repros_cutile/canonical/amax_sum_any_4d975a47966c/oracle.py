"""cuTile port of amax_sum_any_4d975a47966c (NEW_PATTERN): bf16 softmax with
all--inf row fallback to zero. Uses BLOCK_M=4 to match Triton's tiling.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


K_LEN = 512


@ct.kernel
def _bf16_softmax_zero_scope_kernel(
    x_ptr,     # (rows, k_len) bf16
    out_ptr,   # (rows, k_len) bf16
    BLOCK_M: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
):
    row_block = ct.bid(0)
    scores_bf = ct.load(x_ptr, index=(row_block, 0), shape=(BLOCK_M, BLOCK_N))
    scores = ct.astype(scores_bf, ct.float32)
    neg_inf = ct.full(shape=(BLOCK_M, BLOCK_N), fill_value=-float("inf"), dtype=ct.float32)
    live = scores != neg_inf
    zero_i = ct.zeros((BLOCK_M, BLOCK_N), dtype=ct.int32)
    one_i = ct.full(shape=(BLOCK_M, BLOCK_N), fill_value=1, dtype=ct.int32)
    live_i = ct.where(live, one_i, zero_i)
    # has_any along axis=1 (columns dim): (BLOCK_M,)
    has_any = ct.max(live_i, axis=1) != 0
    has_any_2d = ct.reshape(has_any, (BLOCK_M, 1))

    row_max = ct.max(scores, axis=1, keepdims=True)
    safe_max = ct.where(has_any_2d,
                        row_max,
                        ct.zeros((BLOCK_M, 1), dtype=ct.float32))
    numer = ct.exp(scores - safe_max)
    numer = ct.where(live, numer,
                     ct.zeros((BLOCK_M, BLOCK_N), dtype=ct.float32))
    denom = ct.sum(numer, axis=1, keepdims=True)
    denom = ct.where(has_any_2d, denom,
                     ct.full(shape=(BLOCK_M, 1), fill_value=1.0, dtype=ct.float32))
    probs = ct.astype(numer / denom, ct.bfloat16)
    zeros_bf = ct.zeros((BLOCK_M, BLOCK_N), dtype=ct.bfloat16)
    out = ct.where(has_any_2d, probs, zeros_bf)
    ct.store(out_ptr, index=(row_block, 0), tile=out)


@ct.kernel
def _zero_bf16_kernel(
    out_ptr,
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    zeros = ct.zeros(shape=(BLOCK,), dtype=ct.bfloat16)
    ct.store(out_ptr, index=(pid,), tile=zeros)


def _contiguous_stride(shape):
    stride = []
    running = 1
    for size in reversed(shape):
        stride.append(running)
        running *= int(size)
    return tuple(reversed(stride))


@oracle_impl(
    hardware="B200",
    point="9135f859",
    BLOCK_M=4,
    BLOCK_N=512,
    ZERO_BLOCK=1024,
)
def oracle_forward(inputs, *, BLOCK_M: int, BLOCK_N: int, ZERO_BLOCK: int):
    arg0_1, _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3 = inputs
    del _shape_param_0, _shape_param_2

    full_shape = tuple(int(dim) for dim in _shape_param_1)  # (8, 64, 512, 512)
    view_shape = tuple(int(dim) for dim in _shape_param_3)  # (512, 512, 512)
    full_stride = _contiguous_stride(full_shape)

    full = torch.empty_strided(
        full_shape,
        full_stride,
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    where_out = torch.empty_strided(
        full_shape,
        full_stride,
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    rows = arg0_1.numel() // K_LEN
    x_2d = arg0_1.view(rows, K_LEN)
    where_2d = where_out.view(rows, K_LEN)

    stream = torch.cuda.current_stream()
    # Zero-fill full
    n_full = full.numel()
    ct.launch(
        stream,
        (ct.cdiv(n_full, ZERO_BLOCK), 1, 1),
        _zero_bf16_kernel,
        (full.view(-1), ZERO_BLOCK),
    )
    ct.launch(
        stream,
        (ct.cdiv(rows, BLOCK_M), 1, 1),
        _bf16_softmax_zero_scope_kernel,
        (x_2d, where_2d, BLOCK_M, BLOCK_N),
    )
    view_1 = where_out.view(view_shape)
    return (full, where_out, view_1)
