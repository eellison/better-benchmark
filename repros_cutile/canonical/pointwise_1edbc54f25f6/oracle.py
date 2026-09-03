"""cuTile port of pointwise_1edbc54f25f6: AlexNet bool-mask * bf16 * 2.0."""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


N = 128
C = 256
H = 6
W = 6
NUMEL = N * C * H * W


@ct.kernel
def _mask_scale_kernel(
    mask_ptr,
    value_ptr,
    out_ptr,
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    mask = ct.load(mask_ptr, index=(pid,), shape=(BLOCK,))
    value = ct.load(value_ptr, index=(pid,), shape=(BLOCK,))
    mask_f = ct.astype(mask, ct.float32)
    value_f = ct.astype(value, ct.float32)
    result = value_f * (mask_f * 2.0)
    ct.store(out_ptr, index=(pid,), tile=ct.astype(result, ct.bfloat16))


@oracle_impl(hardware="B200", point="44864090", BLOCK=1024)
def oracle_forward(inputs, *, BLOCK: int):
    mask, values, _shape_param_0 = inputs
    out = torch.empty_strided(
        (N, C, H, W),
        (C * H * W, H * W, W, 1),
        device=values.device,
        dtype=torch.bfloat16,
    )
    # cuTile 1D flat view
    mask_flat = mask.view(NUMEL)
    values_flat = values.view(NUMEL)
    out_flat = out.view(NUMEL)
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(NUMEL, BLOCK), 1, 1),
        _mask_scale_kernel,
        (mask_flat, values_flat, out_flat, BLOCK),
    )
    return out
