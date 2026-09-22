"""cuTile port of pointwise_b89a1eb0b38d: bf16 attention permute/clone.

Materializes [B, H, S, D] -> [B, S, H, D] into a contiguous [B*S, H*D] output.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _permute_clone_kernel(
    src,   # (B, H, S, D) bf16 (may be strided)
    dst,   # (B, S, H, D) bf16 contiguous
    BLOCK_D: ct.Constant[int],
):
    b = ct.bid(0)
    h = ct.bid(1)
    s = ct.bid(2)
    # Load one (D,) row and store into permuted position
    tile = ct.load(src, index=(b, h, s, 0), shape=(1, 1, 1, BLOCK_D))
    tile = ct.reshape(tile, (1, 1, 1, BLOCK_D))
    ct.store(dst, index=(b, s, h, 0), tile=tile)


@oracle_impl(hardware="B200", point="9fbac4b8", BLOCK_D=64)
@oracle_impl(hardware="B200", point="903ae292", BLOCK_D=64)
@oracle_impl(hardware="B200", point="976e9b58", BLOCK_D=64)
def oracle_forward(inputs, *, BLOCK_D):
    src, _shape0, out_shape_param = inputs
    B, H, S, D = int(src.shape[0]), int(src.shape[1]), int(src.shape[2]), int(src.shape[3])
    out_shape = tuple(int(x) for x in out_shape_param)
    # Output as [B, S, H*D] viewed as [B, S, H, D]
    out = torch.empty_strided(
        out_shape, (out_shape[1], 1),
        device=src.device, dtype=src.dtype,
    )
    out_4d = out.view(B, S, H, D)
    stream = torch.cuda.current_stream()
    ct.launch(stream, (B, H, S), _permute_clone_kernel, (src, out_4d, BLOCK_D))
    return out
