"""cuTile port of pointwise_4f45960cc89d: XLNet head layout materialize.

Given input `[N=1024, H=16, D=64]`, produce `out[d*H + h, n] = input[n, h, d]`.
Equivalent to `input.permute(2, 1, 0).reshape(D*H, N)`. Ports the Triton
`_xlnet_layout_materialize_kernel` to cuTile as a straight (D, H, N)-tiled copy.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


N = 1024
H = 16
D = 64


@ct.kernel
def _xlnet_layout_materialize_kernel(
    src,   # bf16 [N, H, D]
    dst,   # bf16 [D, H, N]  (view of contiguous [D*H, N] output)
    BLOCK_N: ct.Constant[int],
    BLOCK_D: ct.Constant[int],
):
    d_blk = ct.bid(0)
    h_idx = ct.bid(1)
    n_blk = ct.bid(2)

    # Load src[n_blk*BLOCK_N:..., h_idx, d_blk*BLOCK_D:...] shape (BLOCK_N, 1, BLOCK_D)
    values = ct.load(src, index=(n_blk, h_idx, d_blk),
                     shape=(BLOCK_N, 1, BLOCK_D))
    # Reshape to (BLOCK_N, BLOCK_D), then transpose -> (BLOCK_D, BLOCK_N).
    values = ct.reshape(values, (BLOCK_N, BLOCK_D))
    values_t = ct.transpose(values)
    # Store as (BLOCK_D, 1, BLOCK_N) at dst[d_blk, h_idx, n_blk]
    values_t = ct.reshape(values_t, (BLOCK_D, 1, BLOCK_N))
    ct.store(dst, index=(d_blk, h_idx, n_blk), tile=values_t)


@oracle_impl(hardware="B200", point="a1c20a52", BLOCK_LOCAL=32, BLOCK_COLS=64)
def oracle_forward(inputs, *, BLOCK_LOCAL: int, BLOCK_COLS: int):
    x, shape = inputs
    # x: [N, H, D] contiguous (bf16), shape (1024, 16, 64)
    out_shape = (int(shape[1]), int(shape[2]))  # (D*H=1024, N=1024)
    out = torch.empty_strided(
        out_shape,
        (out_shape[1], 1),
        device=x.device,
        dtype=x.dtype,
    )
    # View out as [D, H, N] so index (d, h, n) corresponds to (d*H + h, n).
    out_view = out.view(D, H, N)

    BLOCK_N = 64
    BLOCK_D = 32
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(D, BLOCK_D), H, ct.cdiv(N, BLOCK_N)),
        _xlnet_layout_materialize_kernel,
        (x, out_view, BLOCK_N, BLOCK_D),
    )
    return out
