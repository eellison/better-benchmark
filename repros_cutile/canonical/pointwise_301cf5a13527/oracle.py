"""cuTile port of pointwise_301cf5a13527: attention-head reshape/permute clone.

Ports the Triton `_head_layout_kernel`. The input is a packed `[B*S, H*D]` bf16
tensor viewed as `[B, S, H, D]`; the output is contiguous `[B, H, S, D]`
(reshaped to `[B*H, S, D]` for the return). The permute swaps axes S and H.

Since cuTile tile shapes must be powers of 2, we pick BLOCK_D as the largest
power of 2 that divides D. When BLOCK_D < D, we split D into (chunk_count,
BLOCK_D) and iterate with a 5D shape.
"""

import json
from pathlib import Path

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _head_layout_kernel_4d(
    src,  # (B, S, H, BLOCK_D)  contiguous [B, S, H, D]
    dst,  # (B, H, S, BLOCK_D)  contiguous [B, H, S, D]
    BLOCK_S: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
    BLOCK_D: ct.Constant[int],
):
    b = ct.bid(0)
    s_tile = ct.bid(1)
    h_tile = ct.bid(2)
    # Load a (BLOCK_S, BLOCK_H, BLOCK_D) slab from src (which is [B,S,H,D])
    # and store it permuted (S,H) -> (H,S) into dst.
    values = ct.load(
        src, index=(b, s_tile, h_tile, 0),
        shape=(1, BLOCK_S, BLOCK_H, BLOCK_D),
    )
    permuted = ct.permute(values, (0, 2, 1, 3))  # (1, BLOCK_H, BLOCK_S, BLOCK_D)
    ct.store(dst, index=(b, h_tile, s_tile, 0), tile=permuted)


@ct.kernel
def _head_layout_kernel_5d(
    src,  # (B, S, H, CHUNK_COUNT, BLOCK_D)
    dst,  # (B, H, S, CHUNK_COUNT, BLOCK_D)
    BLOCK_S: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
    BLOCK_D: ct.Constant[int],
):
    b = ct.bid(0)
    s_tile = ct.bid(1)
    h_tile_and_chunk = ct.bid(2)
    # Encode (h_tile, chunk) into bid(2): h_tile * CHUNK_COUNT + chunk.
    # But CHUNK_COUNT isn't a constant here — we pass it via the shape of dim 3
    # (which we load one chunk at a time via index=chunk).
    # Simpler: use a separate constant, but that requires another parameter.
    # Instead, we let CHUNK_COUNT be inferred via src.shape[3]. cuTile allows
    # dynamic dim indexing with a static tile shape of 1 in that dim.
    chunk_count = src.shape[3]
    h_tile = h_tile_and_chunk // chunk_count
    chunk = h_tile_and_chunk - h_tile * chunk_count
    values = ct.load(
        src, index=(b, s_tile, h_tile, chunk, 0),
        shape=(1, BLOCK_S, BLOCK_H, 1, BLOCK_D),
    )
    permuted = ct.permute(values, (0, 2, 1, 3, 4))
    ct.store(dst, index=(b, h_tile, s_tile, chunk, 0), tile=permuted)


def _largest_pow2_divisor(n):
    n = int(n)
    d = 1
    while d * 2 <= n and n % (d * 2) == 0:
        d *= 2
    return d


def _pick_block(n, cap):
    # Largest power-of-2 divisor of n, capped at `cap`.
    d = _largest_pow2_divisor(n)
    while d > cap:
        d //= 2
    return max(d, 1)


def oracle_forward(inputs, *, BLOCK_D, BLOCK_S, BLOCK_H):
    arg0_1, _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3 = inputs
    b, h, s, d = (int(dim) for dim in _shape_param_2)
    output_shape = tuple(int(dim) for dim in _shape_param_3)
    output = torch.empty_strided(
        output_shape,
        (s * d, d, 1),
        device=arg0_1.device,
        dtype=arg0_1.dtype,
    )
    chunk_count = d // BLOCK_D
    stream = torch.cuda.current_stream()
    if chunk_count == 1:
        src4 = arg0_1.view(b, s, h, d)
        dst4 = output.view(b, h, s, d)
        grid = (b, s // BLOCK_S, h // BLOCK_H)
        ct.launch(
            stream, grid, _head_layout_kernel_4d,
            (src4, dst4, BLOCK_S, BLOCK_H, BLOCK_D),
        )
    else:
        src5 = arg0_1.view(b, s, h, chunk_count, BLOCK_D)
        dst5 = output.view(b, h, s, chunk_count, BLOCK_D)
        grid = (b, s // BLOCK_S, (h // BLOCK_H) * chunk_count)
        ct.launch(
            stream, grid, _head_layout_kernel_5d,
            (src5, dst5, BLOCK_S, BLOCK_H, BLOCK_D),
        )
    return output


_SHAPES = json.loads((Path(__file__).with_name("shapes.json")).read_text())
for _POINT in _SHAPES["points"]:
    _h = int(_POINT["inputs"][3][1][1])
    _s = int(_POINT["inputs"][3][1][2])
    _d = int(_POINT["inputs"][3][1][3])
    _block_d = _largest_pow2_divisor(_d)
    # Cap BLOCK_H at 4 and BLOCK_S at 8 — keeps tile small while cutting the
    # launch count by ~BLOCK_S*BLOCK_H (typically 32x).
    _block_h = _pick_block(_h, 4)
    _block_s = _pick_block(_s, 8)
    oracle_forward = oracle_impl(
        hardware="B200",
        point=_POINT["shape_hash"],
        BLOCK_D=_block_d,
        BLOCK_S=_block_s,
        BLOCK_H=_block_h,
    )(oracle_forward)
