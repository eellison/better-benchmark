"""cuTile port of pointwise_08d21e743dbc (Demucs ReLU + slice + add + mask).

Slices the arg tensor in Python (into a contiguous view) so cuTile only sees
a flat 1D bf16 array of the exact output shape, avoiding OOB tiles.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 4
CHANNELS = 128
TIME = 23212
SLICE_START = 355
NUMEL = BATCH * CHANNELS * TIME
OUT_SHAPE = (BATCH, CHANNELS, TIME)
OUT_STRIDE = (CHANNELS * TIME, TIME, 1)


@ct.kernel
def _relu_slice_add_mask_kernel(
    conv_ptr,
    arg_ptr,
    add_out_ptr,
    mask_out_ptr,
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    conv = ct.load(conv_ptr, index=(pid,), shape=(BLOCK,))
    sliced = ct.load(arg_ptr, index=(pid,), shape=(BLOCK,))

    conv_f = ct.astype(conv, ct.float32)
    zero = ct.full(shape=(BLOCK,), fill_value=0.0, dtype=ct.float32)
    # NaN-preserving ReLU: if conv is NaN keep NaN, else max(conv, 0)
    is_nan = conv_f != conv_f
    relu_f = ct.where(is_nan, conv_f, ct.where(conv_f > zero, conv_f, zero))
    relu_bf16 = ct.astype(relu_f, ct.bfloat16)

    sliced_f = ct.astype(sliced, ct.float32)
    add = ct.astype(ct.astype(relu_bf16, ct.float32) + sliced_f, ct.bfloat16)
    ct.store(add_out_ptr, index=(pid,), tile=add)

    le = conv_f <= zero
    ct.store(mask_out_ptr, index=(pid,), tile=le)


@oracle_impl(hardware="B200", point="10dba222", BLOCK=2048)
def oracle_forward(inputs, *, BLOCK: int):
    conv, arg = inputs
    # Slice arg into a contiguous 1D-viewable buffer so cuTile can flat-index.
    sliced = arg[:, :, SLICE_START:SLICE_START + TIME].contiguous()

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
    conv_flat = conv.view(NUMEL)
    sliced_flat = sliced.view(NUMEL)
    add_flat = add_out.view(NUMEL)
    mask_flat = mask_out.view(NUMEL)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(NUMEL, BLOCK), 1, 1),
        _relu_slice_add_mask_kernel,
        (conv_flat, sliced_flat, add_flat, mask_flat, BLOCK),
    )
    return add_out, mask_out
