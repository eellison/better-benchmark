"""cuTile port of pointwise_cb5ec5359d48: Demucs bf16 slice/relu/add."""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 8
CHANNELS = 256
TIME = 5804
ARG_TIME = 5979
SLICE_START = 87
NUMEL = BATCH * CHANNELS * TIME
OUT_SHAPE = (BATCH, CHANNELS, TIME)
OUT_STRIDE = (CHANNELS * TIME, TIME, 1)


@ct.kernel
def _slice_relu_add_kernel(
    conv_ptr,
    arg_sliced_ptr,
    out_ptr,
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    conv = ct.load(conv_ptr, index=(pid,), shape=(BLOCK,))
    arg_val = ct.load(arg_sliced_ptr, index=(pid,), shape=(BLOCK,))
    conv_f = ct.astype(conv, ct.float32)
    zero = ct.zeros(shape=(BLOCK,), dtype=ct.float32)
    relu_f = ct.where(conv_f != conv_f, conv_f, ct.maximum(conv_f, zero))
    ct.store(out_ptr, index=(pid,), tile=ct.astype(relu_f, ct.bfloat16) + arg_val)


@oracle_impl(hardware="B200", point="839c4ad7", BLOCK=1024)
def oracle_forward(inputs, *, BLOCK: int):
    conv, arg = inputs
    out = torch.empty_strided(
        OUT_SHAPE,
        OUT_STRIDE,
        device=conv.device,
        dtype=torch.bfloat16,
    )
    n_rows = BATCH * CHANNELS
    arg2d = arg.view(n_rows, ARG_TIME)
    arg_sliced = arg2d[:, SLICE_START:SLICE_START + TIME].contiguous().view(-1)
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (NUMEL // BLOCK, 1, 1),
        _slice_relu_add_kernel,
        (conv.view(-1), arg_sliced, out.view(-1), BLOCK),
    )
    return out
