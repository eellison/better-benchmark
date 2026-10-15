"""cuTile port of pointwise_bbf412ad3164: XLNet layout clone/permute.

Input arg0_1 is bf16[256, 512, 64] viewed as [A=16, B=16, S=512, 1, D=64].
Output is bf16[8192, 1024] where flat[s*A + a, d*B + b] = arg0[a*B + b, s, d].

Implementation: view input as (A, B, S, D) and output as (S, A, D, B), then
run one kernel that copies (a, b) -> (a, b) between axes.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


A = 16
B = 16
S = 512
D = 64


@ct.kernel
def _xlnet_layout_kernel(
    src,  # (A, B, S, D)
    dst,  # (S, A, D, B)
    BLOCK_D: ct.Constant[int],
):
    a = ct.bid(0)
    b = ct.bid(1)
    s = ct.bid(2)
    # Load a (1, 1, 1, BLOCK_D) tile from src at index (a, b, s, 0).
    tile = ct.load(src, index=(a, b, s, 0), shape=(1, 1, 1, BLOCK_D))
    # Reshape to (1, 1, BLOCK_D, 1) and store at dst index (s, a, 0, b).
    tile2 = ct.reshape(tile, (1, 1, BLOCK_D, 1))
    ct.store(dst, index=(s, a, 0, b), tile=tile2)


@oracle_impl(hardware="B200", point="2cdbce9d")
def oracle_forward(inputs):
    arg0_1, *_ = inputs
    src4 = arg0_1.view(A, B, S, D)
    out = torch.empty_strided(
        (S * A, D * B),
        (D * B, 1),
        device=arg0_1.device,
        dtype=arg0_1.dtype,
    )
    # Reshape (S*A, D*B) -> (S, A, D, B), contiguous with strides (A*D*B, D*B, B, 1).
    dst4 = out.view(S, A, D, B)
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (A, B, S),
        _xlnet_layout_kernel,
        (src4, dst4, D),
    )
    return out
