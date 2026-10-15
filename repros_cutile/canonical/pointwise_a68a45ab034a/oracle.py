"""cuTile port of pointwise_a68a45ab034a: Demucs bf16 slice+ReLU+add.

Match Triton's flat 1D layout: for each output element at offset i, compute
row = i // TIME, col = i - row*TIME, and load arg at row*ARG_TIME + SLICE_START
+ col.  This avoids a torch-side .contiguous() copy of the sliced arg tensor.
BLOCK=1024 divides NUMEL exactly.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 8
CHANNELS = 1024
TIME = 364
ARG_TIME = 372
SLICE_START = 4
NUMEL = BATCH * CHANNELS * TIME
OUT_SHAPE = (BATCH, CHANNELS, TIME)
OUT_STRIDE = (CHANNELS * TIME, TIME, 1)


@ct.kernel
def _slice_relu_add_kernel(
    conv_ptr,      # bf16 [NUMEL]
    arg_ptr,       # bf16 [BATCH*CHANNELS*ARG_TIME]
    out_ptr,       # bf16 [NUMEL]
    T: ct.Constant[int],
    ARG_T: ct.Constant[int],
    SLICE: ct.Constant[int],
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    conv = ct.load(conv_ptr, index=(pid,), shape=(BLOCK,))

    # offsets = pid * BLOCK + arange(BLOCK)
    offsets = pid * BLOCK + ct.arange(BLOCK, dtype=ct.int32)
    row = offsets // T
    col = offsets - row * T
    arg_off = row * ARG_T + SLICE + col
    sliced = ct.gather(arg_ptr, (arg_off,))

    zero = ct.full((BLOCK,), 0.0, dtype=ct.bfloat16)
    is_nan = conv != conv
    relu = ct.where(is_nan, conv, ct.where(conv > zero, conv, zero))
    ct.store(out_ptr, index=(pid,), tile=relu + sliced)


@oracle_impl(hardware="B200", point="c29579ff", BLOCK=1024)
def oracle_forward(inputs, *, BLOCK: int):
    conv, arg = inputs
    out = torch.empty_strided(
        OUT_SHAPE,
        OUT_STRIDE,
        device=conv.device,
        dtype=torch.bfloat16,
    )
    conv_1d = conv.reshape(-1)
    arg_1d = arg.reshape(-1)
    out_1d = out.view(-1)
    stream = torch.cuda.current_stream()
    grid = (ct.cdiv(NUMEL, BLOCK), 1, 1)
    ct.launch(
        stream,
        grid,
        _slice_relu_add_kernel,
        (conv_1d, arg_1d, out_1d, TIME, ARG_TIME, SLICE_START, BLOCK),
    )
    return out
