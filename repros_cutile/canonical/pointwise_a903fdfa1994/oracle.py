"""cuTile port of pointwise_a903fdfa1994: MobileViT slice_scatter of a permuted src.

Input arg0_1 [8192, 240] bf16 -> view [512, 16, 240] -> view [512, 16, 4, 60]
-> permute (0, 2, 1, 3) -> [512, 4, 16, 60]. Output tensors are:
  full: zeros [512, 4, 16, 64] bf16
  slice_scatter: full with permuted written into [..., :60].
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
def _slice_scatter_write_kernel(
    src_ptr,     # bf16 [B, H, S, ACTIVE_DIM]
    dst_ptr,     # bf16 [B, H, S, PADDED_DIM]
    BLOCK_D: ct.Constant[int],
):
    b = ct.bid(0)
    h = ct.bid(1)
    s = ct.bid(2)
    v = ct.load(src_ptr, index=(b, h, s, 0), shape=(1, 1, 1, BLOCK_D))
    ct.store(dst_ptr, index=(b, h, s, 0), tile=v)


@oracle_impl(hardware="B200", point="51fe7e77")
def oracle_forward(inputs):
    arg0_1 = inputs[0]
    device = arg0_1.device
    full = torch.zeros(
        (BATCH, HEADS, SEQ, PADDED_DIM),
        device=device,
        dtype=torch.bfloat16,
    )
    slice_scatter = torch.zeros(
        (BATCH, HEADS, SEQ, PADDED_DIM),
        device=device,
        dtype=torch.bfloat16,
    )
    # Build source: [B, H, S, ACTIVE_DIM]
    src = arg0_1.view(BATCH, SEQ, HEADS, ACTIVE_DIM).permute(0, 2, 1, 3).contiguous()

    # Write src into the first ACTIVE_DIM columns of slice_scatter.
    # BLOCK_D must be a power of two >= ACTIVE_DIM. Since ACTIVE_DIM=60,
    # BLOCK_D=64. But storing 64 elements would overwrite the pad region
    # (which is already 0, so that's fine). Alternatively, use ACTIVE_DIM
    # itself if power-of-two rules allow. 60 isn't pow2, so use 64.
    # Since destination has 64 wide, writing 64 elements works IF we make
    # the source have 64 (pad the source with zeros).
    src_padded = torch.zeros(
        (BATCH, HEADS, SEQ, PADDED_DIM),
        device=device,
        dtype=torch.bfloat16,
    )
    src_padded[:, :, :, :ACTIVE_DIM].copy_(src)

    BLOCK_D = PADDED_DIM
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (BATCH, HEADS, SEQ),
        _slice_scatter_write_kernel,
        (src_padded, slice_scatter, BLOCK_D),
    )
    return full, slice_scatter
