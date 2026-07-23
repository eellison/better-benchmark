"""cuTile port of pointwise_f68e5f0d951d: bf16 Demucs slice/ReLU/add pointwise."""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 8
CHANNELS = 64
TIME = 92844
ARG_TIME = 95696
SLICE_START = 1426


@ct.kernel
def _slice_relu_add_kernel(
    conv,     # (BATCH, CHANNELS, TIME) bf16
    arg,      # (BATCH, CHANNELS, ARG_TIME) bf16
    out,      # (BATCH, CHANNELS, TIME) bf16
    T: ct.Constant[int],
    SLICE_START_: ct.Constant[int],
    BLOCK: ct.Constant[int],
):
    b = ct.bid(0)
    c = ct.bid(1)
    t_block = ct.bid(2)
    # Load a block of T elements
    conv_tile = ct.load(conv, index=(b, c, t_block), shape=(1, 1, BLOCK))
    arg_tile = ct.load(arg, index=(b, c, 0), shape=(1, 1, BLOCK),
                       padding_mode=ct.PaddingMode.ZERO)  # dummy; we'll use offset
    # The above doesn't work with fixed offset. Let's use a flat view instead.
    # Fall back to view-based approach: for the arg tensor, we can index into
    # a different tile-space. Simpler: rely on cuTile to handle the offset
    # explicitly. Actually we can pre-slice `arg[:, :, SLICE_START:SLICE_START+TIME]`
    # outside the kernel (creates a strided view — cuTile respects strides).
    conv_f = ct.astype(conv_tile, ct.float32)
    # NaN-preserving relu
    zero = ct.zeros(shape=(1, 1, BLOCK), dtype=ct.float32)
    is_nan = conv_f != conv_f
    relu_val = ct.where(is_nan, conv_f, ct.max(conv_f, zero))
    # Note: ct.max is a reduction. Use elementwise max? Use where.
    relu_val = ct.where(is_nan, conv_f, ct.where(conv_f > zero, conv_f, zero))
    arg_f = ct.astype(arg_tile, ct.float32)
    result = ct.astype(relu_val + arg_f, ct.bfloat16)
    ct.store(out, index=(b, c, t_block), tile=result)


# The above is complex. Simpler: use a 1D flat kernel via contiguous views for
# both output and the pre-sliced arg tensor.
@ct.kernel
def _slice_relu_add_flat_kernel(
    conv_flat,  # (N,) bf16
    arg_slice,  # (N,) bf16 -- pre-sliced strided view
    out_flat,   # (N,) bf16
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    conv_tile = ct.load(conv_flat, index=(pid,), shape=(BLOCK,))
    arg_tile = ct.load(arg_slice, index=(pid,), shape=(BLOCK,))
    conv_f = ct.astype(conv_tile, ct.float32)
    arg_f = ct.astype(arg_tile, ct.float32)
    zero_t = ct.zeros(shape=(BLOCK,), dtype=ct.float32)
    is_nan = conv_f != conv_f
    relu_val = ct.where(is_nan, conv_f, ct.where(conv_f > zero_t, conv_f, zero_t))
    result = ct.astype(relu_val + arg_f, ct.bfloat16)
    ct.store(out_flat, index=(pid,), tile=result)


@oracle_impl(hardware="B200", point="042ea2d0", BLOCK=1024)
def oracle_forward(inputs, *, BLOCK):
    conv, arg = inputs
    out = torch.empty_strided(
        (BATCH, CHANNELS, TIME),
        (CHANNELS * TIME, TIME, 1),
        device=conv.device, dtype=torch.bfloat16,
    )
    # Pre-slice the arg tensor (strided view)
    arg_sliced = arg[:, :, SLICE_START:SLICE_START + TIME].contiguous()
    stream = torch.cuda.current_stream()
    n = out.numel()
    ct.launch(stream, (ct.cdiv(n, BLOCK), 1, 1),
              _slice_relu_add_flat_kernel,
              (conv.view(-1), arg_sliced.view(-1), out.view(-1), BLOCK))
    return out
