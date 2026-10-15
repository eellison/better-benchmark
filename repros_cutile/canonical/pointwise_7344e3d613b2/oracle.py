"""cuTile port of pointwise_7344e3d613b2 (BANDWIDTH_BOUND): Demucs ReLU/slice/add/mask.

For each element in [4, 256, 5804]:
  relu = relu(conv)          (bf16, preserves NaN)
  slice = arg[..., 87:5891]  (bf16)
  add = relu + slice         (bf16)
  le = relu <= 0             (bool)

We flatten to 1D since NUMEL = 4 * 256 * 5804 is divisible by BLOCK=1024, avoiding OOB stores.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 4
CHANNELS = 256
TIME = 5804
ARG_TIME = 5979
SLICE_START = 87
NUMEL = BATCH * CHANNELS * TIME  # 5,943,296


@ct.kernel
def _relu_slice_add_mask_kernel(
    conv_ptr,        # bf16 [NUMEL] flat
    arg_slice_ptr,   # bf16 [NUMEL] flat (sliced view)
    add_out_ptr,     # bf16 [NUMEL]
    mask_out_ptr,    # bool [NUMEL]
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    conv = ct.load(conv_ptr, index=(pid,), shape=(BLOCK,))
    sliced = ct.load(arg_slice_ptr, index=(pid,), shape=(BLOCK,))

    conv_f = ct.astype(conv, ct.float32)
    zero_f = ct.full((BLOCK,), 0.0, dtype=ct.float32)
    is_nan = conv_f != conv_f
    relu = ct.where(is_nan, conv_f, ct.maximum(conv_f, zero_f))
    relu_bf16 = ct.astype(relu, ct.bfloat16)
    sliced_f = ct.astype(sliced, ct.float32)
    add = ct.astype(relu + sliced_f, ct.bfloat16)
    ct.store(add_out_ptr, index=(pid,), tile=add)
    # le = (relu <= 0) — relu(NaN)=NaN, NaN <= 0 = False; for negative conv, relu = 0 and 0<=0 True.
    le = relu <= zero_f
    ct.store(mask_out_ptr, index=(pid,), tile=le)


@oracle_impl(hardware="B200", point="89c51046", BLOCK=1024)
def oracle_forward(inputs, *, BLOCK):
    conv, arg = inputs
    # Flatten conv: bf16[4,256,5804] -> [NUMEL]. Contiguous.
    conv_flat = conv.contiguous().view(NUMEL)
    # arg[..., 87:87+TIME] as contiguous by copying.
    arg_slice = arg[..., SLICE_START:SLICE_START + TIME].contiguous().view(NUMEL)

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
    add_out_flat = add_out.view(NUMEL)
    mask_out_flat = mask_out.view(NUMEL)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (NUMEL // BLOCK, 1, 1),
        _relu_slice_add_mask_kernel,
        (conv_flat, arg_slice, add_out_flat, mask_out_flat, BLOCK),
    )
    return add_out, mask_out
