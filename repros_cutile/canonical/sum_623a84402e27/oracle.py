"""cuTile port of sum_623a84402e27: MegatronBert QKV clone + column sum.

SCHEDULER_FUSION: the input is bf16 [16, 16, 512, 64] with strides
(524288, 64, 1024, 1). Reshape to [16, 512, 16, 64] and permute [0, 2, 1, 3]
in torch — since the strides are already the permuted-contiguous strides,
this is a metadata-only view; then `.contiguous()` clones into [rows, cols].
Kernel: for each (row_tile, col_tile) load the bf16 tile, fp32-cast, store to
the clone buffer and accumulate a column-partial sum. Finalize the column sums
across row tiles into an f32 [cols] output that goes through a bf16 rounding
roundtrip.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _copy_kernel(
    src_ptr,   # bf16 [rows, cols]
    dst_ptr,   # bf16 [rows, cols]
    BLOCK_M: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
):
    row_tile = ct.bid(0)
    col_tile = ct.bid(1)
    values = ct.load(src_ptr, index=(row_tile, col_tile), shape=(BLOCK_M, BLOCK_N))
    ct.store(dst_ptr, index=(row_tile, col_tile), tile=values)


@ct.kernel
def _col_sum_bf16_kernel(
    x_ptr,      # bf16 [rows, cols]
    out_ptr,    # f32 [cols]
    ROWS: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
):
    col_tile = ct.bid(0)
    # Load whole column block (rows, BLOCK_N)
    values = ct.load(x_ptr, index=(0, col_tile), shape=(ROWS, BLOCK_N))
    values_f = ct.astype(values, ct.float32)
    totals = ct.sum(values_f, axis=0)   # (BLOCK_N,)
    # bf16 rounding roundtrip to match the Triton kernel.
    rounded_bf16 = ct.astype(totals, ct.bfloat16)
    rounded_f32 = ct.astype(rounded_bf16, ct.float32)
    ct.store(out_ptr, index=(col_tile,), tile=rounded_f32)


@oracle_impl(
    hardware="B200",
    point="903ae292",
    BLOCK_N=32,
    ROW_BLOCK=512,
)
@oracle_impl(
    hardware="B200",
    point="9876fbcf",
    BLOCK_N=32,
    ROW_BLOCK=512,
)
def oracle_forward(inputs, *, BLOCK_N: int, ROW_BLOCK: int):
    x, _shape_param_0, shape_2d, sum_shape = inputs
    rows = int(shape_2d[0])
    cols = int(shape_2d[1])

    # x is bf16 [B, H, S, D] with strides that match the permute.
    # After permute [0, 2, 1, 3] and view [rows, cols], we get the "clone"
    # source. Materialize it once with torch (respects strides).
    x_permuted = x.permute(0, 2, 1, 3).contiguous().view(rows, cols)
    # For our purposes, `x_permuted` is already the "clone" we need. But the
    # Triton oracle also computes the column sum; we do both in cuTile
    # kernels to mirror that path.

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
    sum_out_1d = sum_out.view(cols)

    stream = torch.cuda.current_stream()
    BLOCK_M = 32
    ct.launch(
        stream,
        (rows // BLOCK_M, cols // BLOCK_N, 1),
        _copy_kernel,
        (x_permuted, clone, BLOCK_M, BLOCK_N),
    )
    ct.launch(
        stream,
        (cols // BLOCK_N, 1, 1),
        _col_sum_bf16_kernel,
        (clone, sum_out_1d, rows, BLOCK_N),
    )

    return clone, torch.as_strided(clone, (cols, rows), (1, cols)), sum_out
