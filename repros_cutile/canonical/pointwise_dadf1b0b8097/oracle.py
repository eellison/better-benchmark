"""cuTile port of pointwise_dadf1b0b8097: scaled attention-head transpose clone.

Ports the Triton `_scaled_head_layout_transpose_clone_kernel`. Input is
`[B*S, H*D]` bf16 viewed as `[B, S, H, D]`; output is contiguous `[B, H, D, S]`
scaled by 0.334370152488211.

Since cuTile tile shapes must be powers of 2, we pick BLOCK_D as the largest
power of 2 that divides D (so no OOB stores).
"""

import json
from pathlib import Path

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


SCALE = 0.334370152488211


@ct.kernel
def _scaled_head_layout_transpose_clone_kernel(
    src,  # (B, S, H, chunk_count, BLOCK_D) or (B, S, H, D)
    dst,  # (B, H, chunk_count, BLOCK_D, S) or (B, H, D, S)
    CHUNK_COUNT: ct.Constant[int],
    BLOCK_D: ct.Constant[int],
    BLOCK_S: ct.Constant[int],
):
    b = ct.bid(0)
    h_block = ct.bid(1)  # H * chunk_count blocks
    s_block = ct.bid(2)
    h = h_block // CHUNK_COUNT
    chunk = h_block - h * CHUNK_COUNT
    if CHUNK_COUNT == 1:
        # Load (1, BLOCK_S, 1, BLOCK_D) from src at (b, s_start, h, 0).
        values = ct.load(
            src, index=(b, s_block, h, 0), shape=(1, BLOCK_S, 1, BLOCK_D)
        )
        # Reshape to (BLOCK_S, BLOCK_D) then transpose to (BLOCK_D, BLOCK_S).
        v2d = ct.reshape(values, (BLOCK_S, BLOCK_D))
        v2d_f = ct.astype(v2d, ct.float32)
        v2d_scaled = ct.astype(v2d_f * SCALE, ct.bfloat16)
        v2d_t = ct.transpose(v2d_scaled)  # (BLOCK_D, BLOCK_S)
        # Store to dst at (b, h, 0, s_start) with shape (1, 1, BLOCK_D, BLOCK_S)
        ct.store(
            dst,
            index=(b, h, 0, s_block),
            tile=ct.reshape(v2d_t, (1, 1, BLOCK_D, BLOCK_S)),
        )
    else:
        values = ct.load(
            src,
            index=(b, s_block, h, chunk, 0),
            shape=(1, BLOCK_S, 1, 1, BLOCK_D),
        )
        v2d = ct.reshape(values, (BLOCK_S, BLOCK_D))
        v2d_f = ct.astype(v2d, ct.float32)
        v2d_scaled = ct.astype(v2d_f * SCALE, ct.bfloat16)
        v2d_t = ct.transpose(v2d_scaled)
        ct.store(
            dst,
            index=(b, h, chunk, 0, s_block),
            tile=ct.reshape(v2d_t, (1, 1, 1, BLOCK_D, BLOCK_S)),
        )


def _largest_pow2_divisor(n):
    n = int(n)
    d = 1
    while d * 2 <= n and n % (d * 2) == 0:
        d *= 2
    return d


def oracle_forward(inputs, *, BLOCK_D, BLOCK_S):
    arg0_1, _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3 = inputs
    del _shape_param_0, _shape_param_1

    b, h, d, s = (int(dim) for dim in _shape_param_2)
    out_shape = tuple(_shape_param_3)

    view_2 = torch.empty_strided(
        out_shape,
        (d * s, s, 1),
        device=arg0_1.device,
        dtype=arg0_1.dtype,
    )

    # arg0_1 is [B*S, H*D]. View as [B, S, H, D].
    src4 = arg0_1.view(b, s, h, d)
    # view_2 is [B*H, D, S]. View as [B, H, D, S].
    dst4 = view_2.view(b, h, d, s)

    chunk_count = d // BLOCK_D
    stream = torch.cuda.current_stream()
    if chunk_count == 1:
        ct.launch(
            stream,
            (b, h, s // BLOCK_S),
            _scaled_head_layout_transpose_clone_kernel,
            (src4, dst4, 1, BLOCK_D, BLOCK_S),
        )
    else:
        src5 = arg0_1.view(b, s, h, chunk_count, BLOCK_D)
        dst5 = view_2.view(b, h, chunk_count, BLOCK_D, s)
        ct.launch(
            stream,
            (b, h * chunk_count, s // BLOCK_S),
            _scaled_head_layout_transpose_clone_kernel,
            (src5, dst5, chunk_count, BLOCK_D, BLOCK_S),
        )

    return view_2, view_2.permute(0, 2, 1)


_SHAPES = json.loads((Path(__file__).with_name("shapes.json")).read_text())
for _POINT in _SHAPES["points"]:
    _shape = _POINT["inputs"][3][1]
    _d = int(_shape[2])
    _s = int(_shape[3])
    _block_d = _largest_pow2_divisor(_d)
    _block_s = _largest_pow2_divisor(_s)
    oracle_forward = oracle_impl(
        hardware="B200",
        point=_POINT["shape_hash"],
        BLOCK_D=_block_d,
        BLOCK_S=_block_s,
    )(oracle_forward)
