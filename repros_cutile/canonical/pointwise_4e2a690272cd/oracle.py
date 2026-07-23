"""cuTile port of pointwise_4e2a690272cd: MT5/T5 attention-key layout clone.

Input arg0_1: bf16[N*H, S, D] (contiguous).
Output view_2: bf16[N*D, H*S] contiguous; permute alias transpose.

For each (n, h, s, d), output[n*D+d, h*S+s] = input[n*H+h, s, d].

We reshape input to [N, H, S, D] and output to [N, D, H, S], then compute via
tile of last two dims per (n, d).
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _key_layout_clone_kernel(
    src,  # bf16 [N, H, S, D] view of input
    dst,  # bf16 [N, D, H, S] view of output
    H_C: ct.Constant[int],
    S_C: ct.Constant[int],
):
    n = ct.bid(0)
    d = ct.bid(1)
    h = ct.bid(2)
    # Load one row of length S from src at (n, h, :, d)
    # Load shape (1, 1, S, 1) then reshape to (1, 1, 1, S)
    tile = ct.load(src, index=(n, h, 0, d), shape=(1, 1, S_C, 1))
    tile = ct.reshape(tile, (1, 1, 1, S_C))
    ct.store(dst, index=(n, d, h, 0), tile=tile)


def _launch(inputs):
    arg0_1, _shape_param_0, _shape_param_1, _shape_param_2 = inputs
    N = int(_shape_param_0[0])
    H = int(_shape_param_0[1])
    S = int(_shape_param_0[2])
    D = int(_shape_param_0[3])
    src4 = arg0_1.view(N, H, S, D)

    out = torch.empty_strided(
        tuple(_shape_param_2),  # (N*D, H*S)
        (int(_shape_param_2[1]), 1),
        device=arg0_1.device,
        dtype=arg0_1.dtype,
    )
    dst4 = out.view(N, D, H, S)

    stream = torch.cuda.current_stream()
    ct.launch(stream, (N, D, H), _key_layout_clone_kernel, (src4, dst4, H, S))
    return out, out.permute(1, 0)


@oracle_impl(hardware="B200", point="1e7ad64a")
@oracle_impl(hardware="B200", point="beb18eeb")
def oracle_forward(inputs):
    return _launch(inputs)
