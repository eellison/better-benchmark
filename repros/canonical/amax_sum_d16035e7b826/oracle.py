"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete BERT sliced-vocabulary bf16-to-fp32 log-softmax scope in Triton, including the column slice that drops the final three padded vocabulary entries, the `[2048,20005] -> [16,128,20005]` view semantics, fp32 promotion, stable last-dimension amax/libdevice.exp/sum/libdevice.log, and the returned contiguous f32 `[16,128,20005]` tensor, whereas Inductor lowers the captured slice/view/cast/amax/sub/exp/sum/log/sub graph through generic reduction and pointwise kernels over a very large vocabulary row; Inductor cannot do this today because its pattern library does not recognize this sliced padded-vocabulary log-softmax as a guarded full-scope row-reduction plan with a direct contiguous output store; the fix is NEW_PATTERN: add a sliced-vocabulary log-softmax lowering that preserves the padded input stride, natural exp/log numerics, fp32 arithmetic boundaries, and direct output layout."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


PADDED_K = 20008


@triton.jit
def _log_softmax_onepass_kernel(
    x_ptr,
    out_ptr,
    n_rows: tl.constexpr,
    k_len: tl.constexpr,
    padded_k: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < k_len
    x = tl.load(x_ptr + row * padded_k + cols, mask=mask, other=-float("inf")).to(tl.float32)
    row_max = tl.max(x, axis=0)
    shifted = x - row_max
    numer = libdevice.exp(shifted)
    numer = tl.where(mask, numer, 0.0)
    denom = tl.sum(numer, axis=0)
    out = shifted - libdevice.log(denom)
    tl.store(out_ptr + row * k_len + cols, out, mask=(row < n_rows) & mask)


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


# 00ba3519: BERT_pytorch sliced bf16 vocab [2048,20008] -> f32 log-softmax [16,128,20005].
@oracle_impl(
    hardware="B200",
    point="00ba3519",
    BLOCK_N=32768,
    num_warps=8,
)
def oracle_forward(
    inputs,
    *,
    BLOCK_N: int,
    num_warps: int,
):
    x, shape0 = inputs
    out_shape = _resolve_shape(shape0, x.numel())
    n_rows = int(out_shape[0] * out_shape[1])
    k_len = int(out_shape[2])

    out = torch.empty_strided(
        out_shape,
        _contiguous_stride(out_shape),
        device=x.device,
        dtype=torch.float32,
    )
    _log_softmax_onepass_kernel[(n_rows,)](
        x,
        out,
        n_rows=n_rows,
        k_len=k_len,
        padded_k=PADDED_K,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
    )
    return out
