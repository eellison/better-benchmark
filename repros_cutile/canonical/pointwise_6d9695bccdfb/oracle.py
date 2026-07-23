"""cuTile port of pointwise_6d9695bccdfb: Demucs bf16 ReLU / slice add / bool mask.

Ports the Triton `_relu_slice_add_mask_kernel` to cuTile. Reads conv and
arg[:, :, 4:-4] (which we materialize to contiguous storage before the
launch), computes add = relu(conv) + arg_sliced and mask = conv <= 0.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 4
CHANNELS = 1024
TIME = 364
NUMEL = BATCH * CHANNELS * TIME  # 1490944 = 2^11 * 728


@ct.kernel
def _relu_slice_add_mask_kernel(
    conv,      # flat bf16 [NUMEL]
    arg,       # flat bf16 [NUMEL] (already sliced+contiguous)
    add_out,   # flat bf16 [NUMEL]
    mask_out,  # flat bool [NUMEL]
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    conv_val = ct.load(conv, index=(pid,), shape=(BLOCK,))
    arg_val = ct.load(arg, index=(pid,), shape=(BLOCK,))
    conv_f = ct.astype(conv_val, ct.float32)
    arg_f = ct.astype(arg_val, ct.float32)
    zero = ct.zeros((BLOCK,), dtype=ct.float32)
    relu = ct.where(conv_f != conv_f, conv_f, ct.where(conv_f > 0.0, conv_f, zero))
    add_val = relu + arg_f
    ct.store(add_out, index=(pid,), tile=ct.astype(add_val, ct.bfloat16))
    ct.store(mask_out, index=(pid,), tile=conv_f <= 0.0)


@oracle_impl(hardware="B200", point="46f37179", BLOCK=2048)
def oracle_forward(inputs, *, BLOCK: int):
    conv, arg = inputs
    add_out = torch.empty_strided(
        (BATCH, CHANNELS, TIME),
        (CHANNELS * TIME, TIME, 1),
        device=conv.device,
        dtype=torch.bfloat16,
    )
    mask_out = torch.empty_strided(
        (BATCH, CHANNELS, TIME),
        (CHANNELS * TIME, TIME, 1),
        device=conv.device,
        dtype=torch.bool,
    )
    arg_sliced = arg[:, :, 4:4 + TIME].contiguous()
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (NUMEL // BLOCK, 1, 1),
        _relu_slice_add_mask_kernel,
        (conv.view(NUMEL), arg_sliced.view(NUMEL), add_out.view(NUMEL), mask_out.view(NUMEL), BLOCK),
    )
    return add_out, mask_out
