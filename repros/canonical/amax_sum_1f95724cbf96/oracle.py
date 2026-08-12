"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete BERT sliced-vocabulary bf16 log-softmax-and-cast scope in one Triton row kernel, including the column slice that drops the final three padded vocabulary entries, the `[2048,20005] -> [16,128,20005]` view semantics, bf16-to-fp32 promotion, stable last-dimension amax/libdevice.exp/sum/libdevice.log, final subtract, and visible bf16 output rounding, whereas Inductor lowers the captured slice/view/cast/amax/sub/exp/sum/log/sub/cast graph through generic reduction and pointwise kernels over a very large vocabulary row; Inductor cannot do this today because its pattern library does not recognize this sliced padded-vocabulary log-softmax with a final bf16 boundary as one guarded full-scope row-reduction plan with direct contiguous bf16 output stores; the fix is NEW_PATTERN: add a sliced-vocabulary log-softmax lowering that preserves the padded input stride, natural exp/log numerics, fp32 arithmetic boundaries, final bf16 cast, and direct output layout."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


PADDED_K = 20008


@triton.jit
def _log_softmax_bf16_kernel(
    x_ptr,
    out_ptr,
    N_ROWS: tl.constexpr,
    K_LEN: tl.constexpr,
    PADDED_K_CONST: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < K_LEN
    x = tl.load(
        x_ptr + row * PADDED_K_CONST + cols,
        mask=mask,
        other=-float("inf"),
    ).to(tl.float32)

    row_max = tl.max(x, axis=0)
    shifted = x - row_max
    numer = libdevice.exp(shifted)
    numer = tl.where(mask, numer, 0.0)
    denom = tl.sum(numer, axis=0)
    out = (shifted - libdevice.log(denom)).to(tl.bfloat16)
    tl.store(out_ptr + row * K_LEN + cols, out, mask=(row < N_ROWS) & mask)


def _resolve_shape(shape, numel):
    dims = [int(dim) for dim in shape]
    unknown = -1
    known = 1
    for idx, dim in enumerate(dims):
        if dim == -1:
            unknown = idx
        else:
            known *= dim
    if unknown >= 0:
        dims[unknown] = int(numel) // known
    return tuple(dims)


def _contiguous_stride(shape):
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= int(dim)
    return tuple(reversed(stride))


# 00ba3519: BERT_pytorch sliced bf16 vocab [2048,20008] -> bf16 log-softmax [16,128,20005].
@oracle_impl(
    hardware="B200",
    point="00ba3519",
    BLOCK_N=32768,
    num_warps=8,
    num_stages=3,
)
def oracle_forward(
    inputs,
    *,
    BLOCK_N: int,
    num_warps: int,
    num_stages: int,
):
    x, shape0 = inputs
    out_shape = _resolve_shape(shape0, x.numel())
    n_rows = int(out_shape[0] * out_shape[1])
    k_len = int(out_shape[2])

    out = torch.empty_strided(
        out_shape,
        _contiguous_stride(out_shape),
        device=x.device,
        dtype=torch.bfloat16,
    )
    _log_softmax_bf16_kernel[(n_rows,)](
        x,
        out,
        N_ROWS=n_rows,
        K_LEN=k_len,
        PADDED_K_CONST=PADDED_K,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out
