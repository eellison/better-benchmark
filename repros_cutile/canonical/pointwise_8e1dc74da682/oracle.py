"""cuTile port of pointwise_8e1dc74da682: bf16 elementwise add (two variants).

BANDWIDTH_BOUND: dense add for channels-last NCHW and Demucs BCT/TCB mixed
layout add.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _add_dense_kernel(x_ptr, y_ptr, out_ptr, BLOCK: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(x_ptr, index=(pid,), shape=(BLOCK,))
    y = ct.load(y_ptr, index=(pid,), shape=(BLOCK,))
    ct.store(out_ptr, index=(pid,), tile=x + y)


@oracle_impl(hardware="B200", point="092aaddd", BLOCK=256)
@oracle_impl(hardware="B200", point="fd885e3f", BLOCK=2048)
@oracle_impl(hardware="B200", point="fe14aec8", BLOCK=2048)
@oracle_impl(hardware="B200", point="f30f68d5", BLOCK=2048)
@oracle_impl(hardware="B200", point="b4ab9b24", BLOCK=2048)
@oracle_impl(hardware="B200", point="8bac800c", BLOCK=2048)
@oracle_impl(hardware="B200", point="8b187706", BLOCK=2048)
@oracle_impl(hardware="B200", point="7c24d0c8", BLOCK=2048)
@oracle_impl(hardware="B200", point="d0800b9f", BLOCK=2048)
def oracle_forward(inputs, *, BLOCK: int):
    x, y = inputs
    out = torch.empty_strided(
        tuple(int(dim) for dim in x.shape),
        tuple(int(stride) for stride in x.stride()),
        device=x.device,
        dtype=x.dtype,
    )
    n = x.numel()
    # Both operands share the same channels-last layout for this variant.
    # We do a storage-linear traversal via as_strided in flat storage order.
    x_flat = torch.as_strided(x, (n,), (1,))
    y_flat = torch.as_strided(y, (n,), (1,))
    out_flat = torch.as_strided(out, (n,), (1,))
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(n, BLOCK), 1, 1),
        _add_dense_kernel,
        (x_flat, y_flat, out_flat, BLOCK),
    )
    return out


@ct.kernel
def _add_demucs_kernel(
    x_ptr,  # [B, C, T] with stride (C, 1, B*C)  -> transposed storage
    y_ptr,  # [B, C, T] contiguous
    out_ptr,  # same as x
    B: ct.Constant[int],
    C: ct.Constant[int],
    T: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
    BLOCK_T: ct.Constant[int],
):
    b = ct.bid(0)
    c_tile = ct.bid(1)
    t_tile = ct.bid(2)
    # cuTile respects strides via Array view — load tile (1, BLOCK_C, BLOCK_T)
    x = ct.load(x_ptr, index=(b, c_tile, t_tile), shape=(1, BLOCK_C, BLOCK_T),
                padding_mode=ct.PaddingMode.ZERO)
    y = ct.load(y_ptr, index=(b, c_tile, t_tile), shape=(1, BLOCK_C, BLOCK_T),
                padding_mode=ct.PaddingMode.ZERO)
    ct.store(out_ptr, index=(b, c_tile, t_tile), tile=x + y)


@oracle_impl(hardware="B200", point="79c39c98", B=8, C=2048, T=92, BLOCK_C=128, BLOCK_T=8)
@oracle_impl(hardware="B200", point="1ffb91db", B=4, C=2048, T=92, BLOCK_C=128, BLOCK_T=8)
def oracle_forward_demucs(
    inputs,
    *,
    B: int,
    C: int,
    T: int,
    BLOCK_C: int,
    BLOCK_T: int,
):
    x, y = inputs
    out = torch.empty_strided(
        tuple(int(dim) for dim in x.shape),
        tuple(int(stride) for stride in x.stride()),
        device=x.device,
        dtype=x.dtype,
    )
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (B, ct.cdiv(C, BLOCK_C), ct.cdiv(T, BLOCK_T)),
        _add_demucs_kernel,
        (x, y, out, B, C, T, BLOCK_C, BLOCK_T),
    )
    return out
