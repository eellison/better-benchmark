"""cuTile port of pointwise_ee109ecb96bb: bf16 GPTNeo head layout transpose clone.

Materializes a bf16 [B*H, D, S] tensor by reading from the packed [B, S, H, D]
source and transposing (S, D) -> (D, S) per (B, H). Returns the [B*H, D, S] view
and its [B*H, S, D] alias permute.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _head_transpose_clone_kernel(
    src,  # (B, S, H, D)
    dst,  # (B, H, D, S)
    BLOCK_S: ct.Constant[int],
    BLOCK_D: ct.Constant[int],
):
    b = ct.bid(0)
    h = ct.bid(1)
    # For each (b, h) tile, load (1, BLOCK_S, 1, BLOCK_D) from src at
    # (b, 0, h, 0) and store transposed to dst at (b, h, 0, 0).
    values = ct.load(src, index=(b, 0, h, 0), shape=(1, BLOCK_S, 1, BLOCK_D))
    # Reshape to (BLOCK_S, BLOCK_D), transpose to (BLOCK_D, BLOCK_S).
    values = ct.reshape(values, (BLOCK_S, BLOCK_D))
    values_t = ct.transpose(values)
    values_t = ct.reshape(values_t, (1, 1, BLOCK_D, BLOCK_S))
    ct.store(dst, index=(b, h, 0, 0), tile=values_t)


@oracle_impl(hardware="B200", point="af0c9f46")
def oracle_forward(inputs):
    arg0_1, _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3 = inputs

    B = int(_shape_param_2[0])
    H = int(_shape_param_2[1])
    D = int(_shape_param_2[2])
    S = int(_shape_param_2[3])

    view_2 = torch.empty_strided(
        tuple(_shape_param_3),
        (D * S, S, 1),
        device=arg0_1.device,
        dtype=arg0_1.dtype,
    )
    # Source packed as [B, S, H, D]
    src4 = arg0_1.view(B, S, H, D)
    dst4 = view_2.view(B, H, D, S)

    stream = torch.cuda.current_stream()
    # Use BLOCK_S = S, BLOCK_D = D so each (b, h) is one tile.
    ct.launch(
        stream,
        (B, H, 1),
        _head_transpose_clone_kernel,
        (src4, dst4, S, D),
    )
    return view_2, view_2.permute(0, 2, 1)
