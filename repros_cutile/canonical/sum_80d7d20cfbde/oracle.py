"""cuTile port of sum_80d7d20cfbde (SCHEDULER_FUSION): H-last ConvBert clone
plus column sum. Materializes [16384, 384] bf16 clone from a permuted (B,H,S,D)
input, plus its transposed alias and the f32 column sum (with bf16 round-trip).
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


B = 32
S_LEN = 512
HEADS = 6
HEAD_DIM = 64
ROWS = B * S_LEN  # 16384
COLS = HEADS * HEAD_DIM  # 384


@ct.kernel
def _copy_kernel_4d(
    src_ptr,   # bf16 [B, S, H, D]
    dst_ptr,   # bf16 [B, S, H, D]
    D_C: ct.Constant[int],
):
    b = ct.bid(0)
    s = ct.bid(1)
    h = ct.bid(2)
    tile = ct.load(src_ptr, index=(b, s, h, 0), shape=(1, 1, 1, D_C))
    ct.store(dst_ptr, index=(b, s, h, 0), tile=tile)


@ct.kernel
def _partial_column_sum_kernel(
    clone_ptr,     # bf16 [ROWS, COLS]
    partial_ptr,   # f32 [NUM_R_TILES, COLS]
    BLOCK_R: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    r_block = ct.bid(0)
    c_block = ct.bid(1)
    vals = ct.load(clone_ptr, index=(r_block, c_block), shape=(BLOCK_R, BLOCK_C))
    vals_f = ct.astype(vals, ct.float32)
    partial = ct.sum(vals_f, axis=0)
    partial_2d = ct.reshape(partial, (1, BLOCK_C))
    ct.store(partial_ptr, index=(r_block, c_block), tile=partial_2d)


@ct.kernel
def _final_reduce_kernel(
    partial_ptr,
    sum_ptr,
    BLOCK_C: ct.Constant[int],
    BLOCK_R_TILES: ct.Constant[int],
):
    c_block = ct.bid(0)
    partials = ct.load(
        partial_ptr,
        index=(0, c_block),
        shape=(BLOCK_R_TILES, BLOCK_C),
        padding_mode=ct.PaddingMode.ZERO,
    )
    total = ct.sum(partials, axis=0)
    total_bf = ct.astype(total, ct.bfloat16)
    total_f = ct.astype(total_bf, ct.float32)
    ct.store(sum_ptr, index=(c_block,), tile=total_f)


def _next_pow2(n):
    p = 1
    while p < n:
        p *= 2
    return p


@oracle_impl(
    hardware="B200",
    point="f85fe078",
    ROW_BLOCK=512,
    FINAL_BLOCK_COLS=32,
    COPY_Y_BLOCK=16,
    COPY_X_BLOCK=64,
    REDUCE_X_BLOCK=32,
)
def oracle_forward(
    inputs,
    *,
    ROW_BLOCK: int,
    FINAL_BLOCK_COLS: int,
    COPY_Y_BLOCK: int,
    COPY_X_BLOCK: int,
    REDUCE_X_BLOCK: int,
):
    x, _shape_param_0, shape_2d, sum_shape = inputs
    del _shape_param_0

    rows = int(shape_2d[0])
    cols = int(shape_2d[1])

    clone = torch.empty_strided(
        (rows, cols),
        (cols, 1),
        device=x.device,
        dtype=torch.bfloat16,
    )
    sum_out = torch.empty_strided(
        tuple(int(dim) for dim in sum_shape),
        (1,),
        device=x.device,
        dtype=torch.float32,
    )

    # x has physical layout (B, S, D, H) with contiguous strides (S*D*H, D*H, H, 1)
    # = (196608, 384, 6, 1). Logically it's (B, H, S, D) with strides (196608, 1, 384, 6).
    # We want clone = permute(x, [0,2,1,3]) contiguous = (B, S, H, D) contig.
    # x.permute(0, 2, 1, 3) gives (B, S, H, D) with strides (196608, 384, 1, 6).
    x_perm = x.permute(0, 2, 1, 3)  # (B, S, H, D)
    clone_4d = clone.view(B, S_LEN, HEADS, HEAD_DIM)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (B, S_LEN, HEADS),
        _copy_kernel_4d,
        (x_perm, clone_4d, HEAD_DIM),
    )

    # Column sum
    num_row_tiles = rows // ROW_BLOCK  # 32
    partial = torch.empty_strided(
        (num_row_tiles, cols),
        (cols, 1),
        device=x.device,
        dtype=torch.float32,
    )
    ct.launch(
        stream,
        (num_row_tiles, cols // FINAL_BLOCK_COLS, 1),
        _partial_column_sum_kernel,
        (clone, partial, ROW_BLOCK, FINAL_BLOCK_COLS),
    )
    block_partials = _next_pow2(num_row_tiles)
    ct.launch(
        stream,
        (cols // FINAL_BLOCK_COLS, 1, 1),
        _final_reduce_kernel,
        (partial, sum_out, FINAL_BLOCK_COLS, block_partials),
    )
    return clone, torch.as_strided(clone, (cols, rows), (1, cols)), sum_out
