"""cuTile port of pointwise_f84d2f33f15b: Longformer iota+as_strided+clone -> int64.

Reshape the flat int64[NUMEL] output to [OUTER, WINDOW, HEAD_DIM] and compute
each element from its (outer, window, dim) coordinates.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


OUTER = 96 * 3          # 288
WINDOW = 512
HEAD_DIM = 64
NUMEL = OUTER * WINDOW * HEAD_DIM
STRIDE_OUTER = WINDOW * HEAD_DIM  # 32768
STRIDE_SOURCE_BATCH = 64
STRIDE_SOURCE_OVERLAP = 1572864
STRIDE_SOURCE_WINDOW = 6144


@ct.kernel
def _longformer_iota_kernel(
    out_ptr,  # int64 [OUTER, WINDOW, HEAD_DIM]
    BLOCK_W: ct.Constant[int],
    HEAD_DIM_: ct.Constant[int],
    STRIDE_SOURCE_BATCH_: ct.Constant[int],
    STRIDE_SOURCE_OVERLAP_: ct.Constant[int],
    STRIDE_SOURCE_WINDOW_: ct.Constant[int],
):
    outer = ct.bid(0)
    window_block = ct.bid(1)

    # batch = outer // 3, overlap = outer % 3 (compile-time known per-block)
    # But we can express it by computing values from the actual coordinate arrays.
    window = window_block * BLOCK_W + ct.arange(BLOCK_W, dtype=ct.int64)
    dim = ct.arange(HEAD_DIM_, dtype=ct.int64)
    outer_i64 = ct.astype(ct.full(shape=(1,), fill_value=outer, dtype=ct.int32), ct.int64)
    batch = outer_i64 * (1 // 1)  # placeholder: use scalar arithmetic
    # Use scalar Python arithmetic on `outer` — it's a runtime int32 scalar.
    # ct doesn't allow that yet; do it via broadcasted arrays.
    # batch = outer // 3, overlap = outer - batch * 3, all scalars.

    window_2d = ct.reshape(window, (BLOCK_W, 1))
    dim_2d = ct.reshape(dim, (1, HEAD_DIM_))
    # Compute per-element values.
    # values = batch * SB + overlap * SO + window * SW + dim
    # batch, overlap are scalars derived from `outer`.
    # We can multiply outer by S_batch/S_overlap via scalar computations at
    # kernel launch time. But cuTile kernel-side scalar arithmetic on `outer`
    # is fine via broadcast:
    outer_arr = outer_i64  # shape (1,)
    # batch_arr = outer_arr // 3
    batch_arr = outer_arr // 3
    overlap_arr = outer_arr - batch_arr * 3
    base_offset = batch_arr * STRIDE_SOURCE_BATCH_ + overlap_arr * STRIDE_SOURCE_OVERLAP_
    base_2d = ct.reshape(base_offset, (1, 1))
    values = base_2d + window_2d * STRIDE_SOURCE_WINDOW_ + dim_2d

    ct.store(out_ptr, index=(outer, window_block, 0), tile=ct.reshape(values, (1, BLOCK_W, HEAD_DIM_)))


@oracle_impl(hardware="B200", point="d7517139")
def oracle_forward(inputs):
    del inputs
    out = torch.empty_strided(
        (NUMEL,),
        (1,),
        device=torch.device("cuda", 0),
        dtype=torch.int64,
    )
    out_3d = out.view(OUTER, WINDOW, HEAD_DIM)
    BLOCK_W = 16
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (OUTER, ct.cdiv(WINDOW, BLOCK_W), 1),
        _longformer_iota_kernel,
        (out_3d, BLOCK_W, HEAD_DIM,
         STRIDE_SOURCE_BATCH, STRIDE_SOURCE_OVERLAP, STRIDE_SOURCE_WINDOW),
    )
    return out
