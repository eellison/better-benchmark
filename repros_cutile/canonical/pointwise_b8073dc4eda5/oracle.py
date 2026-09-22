"""cuTile port of pointwise_b8073dc4eda5: Demucs bf16 relu/slice/add + mask."""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 4
CHANNELS = 512
TIME = 1452
ARG_TIME = 1493
SLICE_START = 20
NUMEL = BATCH * CHANNELS * TIME
OUT_SHAPE = (BATCH, CHANNELS, TIME)
OUT_STRIDE = (CHANNELS * TIME, TIME, 1)


@ct.kernel
def _relu_slice_add_mask_kernel(
    conv_ptr,        # 1D, contiguous over rows*TIME
    arg_sliced_ptr,  # 1D, contiguous rows*TIME (already sliced)
    add_out_ptr,     # 1D, contiguous rows*TIME
    mask_out_ptr,    # 1D, contiguous rows*TIME
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    conv = ct.load(conv_ptr, index=(pid,), shape=(BLOCK,))
    arg_val = ct.load(arg_sliced_ptr, index=(pid,), shape=(BLOCK,))

    conv_f = ct.astype(conv, ct.float32)
    zero = ct.zeros(shape=(BLOCK,), dtype=ct.float32)
    # NaN-preserving relu: where NaN keep NaN, else max(x, 0)
    relu_f = ct.where(conv_f != conv_f, conv_f, ct.maximum(conv_f, zero))
    relu = ct.astype(relu_f, ct.bfloat16)
    ct.store(add_out_ptr, index=(pid,), tile=relu + arg_val)
    ct.store(mask_out_ptr, index=(pid,), tile=conv_f <= 0.0)


@oracle_impl(hardware="B200", point="46ecc6c3", BLOCK=1024)
def oracle_forward(inputs, *, BLOCK: int):
    conv, arg = inputs
    add_out = torch.empty_strided(
        OUT_SHAPE,
        OUT_STRIDE,
        device=conv.device,
        dtype=torch.bfloat16,
    )
    mask_out = torch.empty_strided(
        OUT_SHAPE,
        OUT_STRIDE,
        device=conv.device,
        dtype=torch.bool,
    )
    n_rows = BATCH * CHANNELS
    # arg is [B, C, ARG_TIME] — slice out [SLICE_START:SLICE_START+TIME] on the last
    # dim, then materialize contiguous so we get 1D storage matching conv.
    arg2d = arg.view(n_rows, ARG_TIME)
    arg_sliced = arg2d[:, SLICE_START:SLICE_START + TIME].contiguous().view(-1)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (NUMEL // BLOCK, 1, 1),
        _relu_slice_add_mask_kernel,
        (conv.view(-1), arg_sliced, add_out.view(-1), mask_out.view(-1), BLOCK),
    )
    return add_out, mask_out
