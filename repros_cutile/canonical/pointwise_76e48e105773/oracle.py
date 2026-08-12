"""cuTile port of pointwise_76e48e105773: MobileViT slice_scatter.

Copies first 60 dims of each head from a source [8192,240] to a padded
[512, 4, 16, 64] output. Tail 4 dims come from a base [512,4,16,64].
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 512
HEADS = 4
SEQ = 16
ACTIVE_DIM = 60
PADDED_DIM = 64


@ct.kernel
def _slice_scatter_kernel(
    src_reshaped,   # (512, 16, 4, 60) bf16 (view of source)
    base,           # (512, 4, 16, 64) bf16
    dst,            # (512, 4, 16, 64) bf16
    ACTIVE: ct.Constant[int],
    PADDED: ct.Constant[int],
):
    b = ct.bid(0)
    h = ct.bid(1)
    s = ct.bid(2)
    # Load src padded to PADDED (OOB filled with zero because src is only 60 wide)
    src_tile = ct.load(src_reshaped, index=(b, s, h, 0), shape=(1, 1, 1, PADDED),
                       padding_mode=ct.PaddingMode.ZERO)
    # Load full base tile
    base_tile = ct.load(base, index=(b, h, s, 0), shape=(1, 1, 1, PADDED))
    # Build mask indicating cols < ACTIVE
    cols = ct.arange(PADDED, dtype=ct.int32)
    active_mask_1d = cols < ct.full(shape=(PADDED,), fill_value=ACTIVE, dtype=ct.int32)
    active_mask = ct.reshape(active_mask_1d, (1, 1, 1, PADDED))
    result = ct.where(active_mask, src_tile, base_tile)
    ct.store(dst, index=(b, h, s, 0), tile=result)


@oracle_impl(hardware="B200", point="5ce0effd")
def oracle_forward(inputs):
    arg0_1, arg1_1 = inputs[:2]
    # arg0_1: [8192, 240] view as [512, 16, 240], further [512, 16, 4, 60].
    src_reshaped = arg0_1.view(BATCH, SEQ, HEADS, ACTIVE_DIM)
    out = torch.empty_strided(
        (BATCH, HEADS, SEQ, PADDED_DIM),
        (HEADS * SEQ * PADDED_DIM, SEQ * PADDED_DIM, PADDED_DIM, 1),
        device=arg0_1.device, dtype=arg0_1.dtype,
    )
    stream = torch.cuda.current_stream()
    ct.launch(stream, (BATCH, HEADS, SEQ), _slice_scatter_kernel,
              (src_reshaped, arg1_1, out, ACTIVE_DIM, PADDED_DIM))
    return out
