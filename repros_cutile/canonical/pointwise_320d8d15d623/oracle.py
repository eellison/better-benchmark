"""cuTile port of pointwise_320d8d15d623: attention-head layout materialization.

Rewrites [B, S, H, D] contiguous input as [B, H, S, D] contiguous by copying
tiles with the head/seq axes swapped.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _head_layout_2d_kernel(
    src,           # bf16 [B, S, H, D]
    dst,           # bf16 [B, H, S, D]
    BLOCK_S: ct.Constant[int],
    BLOCK_D: ct.Constant[int],
):
    b = ct.bid(0)
    h = ct.bid(1)
    s_tile = ct.bid(2)

    # Load tile [1, BLOCK_S, 1, BLOCK_D] from src at (b, s_tile, h, 0)
    x = ct.load(src, index=(b, s_tile, h, 0), shape=(1, BLOCK_S, 1, BLOCK_D))
    x = ct.reshape(x, (1, 1, BLOCK_S, BLOCK_D))
    ct.store(dst, index=(b, h, s_tile, 0), tile=x)


def _resolve_shape(shape, numel):
    dims = [int(dim) for dim in shape]
    if -1 in dims:
        known = 1
        for dim in dims:
            if dim != -1:
                known *= dim
        dims[dims.index(-1)] = numel // known
    return tuple(dims)


@oracle_impl(hardware="B200", point="d20f46e2", BLOCK_S=16, BLOCK_D=64)
@oracle_impl(hardware="B200", point="ad7b2a2c", BLOCK_S=16, BLOCK_D=64)
@oracle_impl(hardware="B200", point="da5fdead", BLOCK_S=4, BLOCK_D=64)
def oracle_forward(inputs, *, BLOCK_S, BLOCK_D):
    arg0_1, _shape_param_0, _shape_param_1 = inputs
    batch, seq, _hidden = (int(dim) for dim in _shape_param_0)
    _, _, heads, head_dim = _resolve_shape(_shape_param_1, arg0_1.numel())

    # If BLOCK_S doesn't divide seq, fall back to BLOCK_S=1 to avoid OOB.
    if seq % BLOCK_S != 0:
        BLOCK_S = 1
    output = torch.empty_strided(
        (batch, heads, seq, head_dim),
        (heads * seq * head_dim, seq * head_dim, head_dim, 1),
        device=arg0_1.device,
        dtype=arg0_1.dtype,
    )
    # View input as [B, S, H, D] — contiguous.
    src4 = arg0_1.view(batch, seq, heads, head_dim)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (batch, heads, seq // BLOCK_S),
        _head_layout_2d_kernel,
        (src4, output, BLOCK_S, BLOCK_D),
    )
    return output
