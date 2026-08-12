"""cuTile port of pointwise_5d4ed0a3898e: Whisper scaled attention-head clone.

Input bf16[1500, 384] is viewed as bf16[1, 1500, 6, 64], scaled by 0.125,
then permuted to bf16[1, 6, 1500, 64] contiguous.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _scaled_head_clone_kernel(
    in_ptr,   # bf16 [S, H, D]
    out_ptr,  # bf16 [H, S, D]
    BLOCK_D: ct.Constant[int],
):
    h = ct.bid(0)
    s = ct.bid(1)
    # Load tile of shape (1, 1, BLOCK_D) from [S, H, D]
    x = ct.load(in_ptr, index=(s, h, 0), shape=(1, 1, BLOCK_D))
    x_f = ct.astype(x, ct.float32)
    scaled = ct.astype(x_f * 0.125, ct.bfloat16)
    # Reshape from (1, 1, BLOCK_D) to a matching-rank (1, 1, BLOCK_D) for [H, S, D]
    ct.store(out_ptr, index=(h, s, 0), tile=scaled)


@oracle_impl(hardware="B200", point="da5fdead", BLOCK_D=64)
def oracle_forward(inputs, *, BLOCK_D):
    arg0_1, shape0, shape1 = inputs
    B = int(shape0[0])
    S = int(shape0[1])
    D = int(shape1[3])
    H = arg0_1.shape[1] // D

    out = torch.empty((B, H, S, D), device=arg0_1.device, dtype=arg0_1.dtype)
    # Reshape input as [S, H, D] (B=1) — squeeze the batch dim
    in_view = arg0_1.view(S, H, D)
    out_view = out.view(H, S, D)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (H, S, 1),
        _scaled_head_clone_kernel,
        (in_view, out_view, BLOCK_D),
    )
    return out
