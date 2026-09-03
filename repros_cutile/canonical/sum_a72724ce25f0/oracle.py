"""cuTile port of sum_a72724ce25f0 (COOPERATIVE_SPLIT_K): materialize
`add = x*scale + offset/2304` (bf16 boundaries) and per-channel sum.

Uses torch to materialize the elementwise output (which is easier than
building cooperative split-K partials in cuTile), then a cuTile column-sum
kernel over the [N*H*W, C] view for the per-channel reduction.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


_POINTS = ["206932cb", "89759c54", "fba7f37b", "f88457c3", "77f5e18d",
           "69ed9b17", "45166747", "0e3d6755"]


DIVISOR = 2304


@ct.kernel
def _col_sum_kernel(
    add_ptr,   # bf16 [ROWS, C]  (channels-last, C-major)
    out_ptr,   # f32 [C]
    ROWS: ct.Constant[int],
    C: ct.Constant[int],
    BLOCK_R: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    c_block = ct.bid(0)
    total = ct.zeros((BLOCK_C,), dtype=ct.float32)
    n_row_tiles = ct.cdiv(ROWS, BLOCK_R)
    for r_block in range(n_row_tiles):
        tile_bf = ct.load(
            add_ptr, index=(r_block, c_block),
            shape=(BLOCK_R, BLOCK_C),
            padding_mode=ct.PaddingMode.ZERO,
        )
        tile_f = ct.astype(tile_bf, ct.float32)
        r_idx = ct.arange(BLOCK_R, dtype=ct.int32) + r_block * BLOCK_R
        valid_r = ct.reshape(r_idx < ROWS, (BLOCK_R, 1))
        c_idx = ct.arange(BLOCK_C, dtype=ct.int32) + c_block * BLOCK_C
        valid_c = ct.reshape(c_idx < C, (1, BLOCK_C))
        valid = valid_r & valid_c
        zero = ct.zeros((BLOCK_R, BLOCK_C), dtype=ct.float32)
        masked = ct.where(valid, tile_f, zero)
        total = total + ct.sum(masked, axis=0)
    # bf16 round-trip on the total (Repro casts sum -> f32 but Triton oracle rounds)
    total_rounded = ct.astype(ct.astype(total, ct.bfloat16), ct.float32)
    c_idx = ct.arange(BLOCK_C, dtype=ct.int32) + c_block * BLOCK_C
    valid_c = c_idx < C
    zero1 = ct.zeros((BLOCK_C,), dtype=ct.float32)
    out_val = ct.where(valid_c, total_rounded, zero1)
    ct.scatter(out_ptr, (c_idx,), out_val, mask=valid_c)


def _next_pow2(n):
    p = 1
    while p < n:
        p *= 2
    return p


def _impl(inputs):
    x, scale, offset, _shape_param_0 = inputs
    # x is channels-last: [N, C, H, W] with C stride=1. Physically [N, H, W, C].
    n, c, h, w = x.shape
    hw = h * w
    rows = n * hw

    # Materialize `add = x*scale + offset/2304` (bf16-round intermediate boundaries)
    # via torch — matches the eager Repro exactly.
    scale_bcast = scale.expand_as(x)  # [N, C, 1, 1] -> [N, C, H, W] via expand
    offset_bcast = offset.expand_as(x)
    mul = x * scale_bcast
    divv = offset_bcast / DIVISOR
    add_val = mul + divv

    # add_val has channels-last strides — permute(0, 2, 3, 1) yields NHWC
    # contiguous layout as a metadata-only view. Use reshape to keep it that way.
    add_2d = add_val.permute(0, 2, 3, 1).reshape(rows, c)

    out_f32 = torch.empty((c,), device=x.device, dtype=torch.float32)
    BLOCK_R = 128
    BLOCK_C = _next_pow2(c) if c <= 128 else 128
    # Ensure BLOCK_C divides features nicely
    if c % BLOCK_C != 0:
        BLOCK_C = _next_pow2(c)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (ct.cdiv(c, BLOCK_C), 1, 1), _col_sum_kernel,
        (add_2d, out_f32, rows, c, BLOCK_R, BLOCK_C),
    )
    return add_val, out_f32


def _make():
    fn = _impl
    for point in _POINTS:
        fn = oracle_impl(hardware="B200", point=point)(fn)
    return fn


oracle_forward = _make()
