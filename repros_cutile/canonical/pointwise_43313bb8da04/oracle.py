"""cuTile port of pointwise_43313bb8da04: Demucs relu(x) + arg1[:, :, 20:-21] bf16 add.

Uses a per-row kernel with BLOCK=512, iterating over row blocks of 512 lanes so
that all writes land in-bounds. The 1452-wide row is handled by three
(1024+256+partial) tiles ... actually cleanest: use a per-row tile of size 2048
that covers the whole row and rely on padded loads plus scatter-friendly store.

Simplest: flatten to 1D of length 8*512*1452 = 5,947,392 = 5808 * 1024, so
BLOCK=1024 divides evenly.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 8
CHANNELS = 512
TIME = 1452
ARG_TIME = 1493
SLICE_START = 20
NUMEL = BATCH * CHANNELS * TIME  # = 5947392


@ct.kernel
def _slice_relu_add_kernel(
    conv_ptr,        # bf16 flat [NUMEL]
    arg_slice_ptr,   # bf16 flat [NUMEL] (already sliced to [:, :, 20:20+TIME] contiguous)
    out_ptr,         # bf16 flat [NUMEL]
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    conv = ct.load(conv_ptr, index=(pid,), shape=(BLOCK,))
    conv_f = ct.astype(conv, ct.float32)
    is_nan = conv_f != conv_f
    relu = ct.where(is_nan, conv_f, ct.maximum(conv_f, 0.0))

    arg = ct.load(arg_slice_ptr, index=(pid,), shape=(BLOCK,))
    arg_f = ct.astype(arg, ct.float32)

    out_f = relu + arg_f
    out_bf16 = ct.astype(out_f, ct.bfloat16)
    ct.store(out_ptr, index=(pid,), tile=out_bf16)


@oracle_impl(hardware="B200", point="6a0b50df", BLOCK=1024)
def oracle_forward(inputs, *, BLOCK: int):
    conv, arg = inputs
    conv_flat = conv.contiguous().view(-1)
    # Slice arg[:, :, 20:20+TIME] must be materialized contiguous.
    arg_slice = arg[:, :, SLICE_START:SLICE_START + TIME].contiguous().view(-1)

    out = torch.empty_strided(
        (BATCH, CHANNELS, TIME),
        (CHANNELS * TIME, TIME, 1),
        device=conv.device,
        dtype=torch.bfloat16,
    )
    out_flat = out.view(-1)

    n = NUMEL
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(n, BLOCK), 1, 1),
        _slice_relu_add_kernel,
        (conv_flat, arg_slice, out_flat, BLOCK),
    )
    return out
