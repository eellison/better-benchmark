"""cuTile port of sum_abcd9bccce7d: permute/view alias + dim-0 column reduction.

Returns two alias views of the (permuted) input plus the column-sum with a
bf16 roundtrip. The reduction is over rows of a packed [M, N] view. N may not
be a power of two (768, 1024, 2560), so we pad the column axis with BLOCK_N,
mask the OOB columns during reductions, and slice back to [N] after the store.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _col_sum_kernel(
    x_ptr,        # bf16 [M, N]
    out_ptr,      # f32  [BLOCK_N_TOTAL]  (padded)
    M: ct.Constant[int],
    N: ct.Constant[int],
    BLOCK_M: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
):
    col_block = ct.bid(0)
    # Column-space tile: [M, BLOCK_N]. We assume M fits in BLOCK_M for a
    # single-tile approach; when M > BLOCK_M we walk row tiles.
    n_row_tiles = ct.cdiv(M, BLOCK_M)
    acc = ct.full((BLOCK_N,), 0.0, dtype=ct.float32)

    cols = ct.arange(BLOCK_N, dtype=ct.int32)
    col_offset = col_block * BLOCK_N
    global_cols = cols + col_offset
    col_mask = global_cols < N
    zero_col = ct.full((BLOCK_N,), 0.0, dtype=ct.float32)

    for row_block in range(n_row_tiles):
        # Row/col: index in tile space.
        x = ct.load(
            x_ptr,
            index=(row_block, col_block),
            shape=(BLOCK_M, BLOCK_N),
            padding_mode=ct.PaddingMode.ZERO,
        )
        xf = ct.astype(x, ct.float32)
        # Row-mask via rows dimension: rows out of range are already zeroed by
        # PaddingMode.ZERO, and column OOB is also zeroed. So sum is safe.
        partial = ct.sum(xf, axis=0)  # shape (BLOCK_N,)
        acc = acc + partial

    # bf16 roundtrip: cast to bf16 and back to f32 for storage.
    acc_bf = ct.astype(acc, ct.bfloat16)
    acc_out = ct.astype(acc_bf, ct.float32)
    # Zero out the OOB tail so the padded output doesn't have garbage.
    acc_out = ct.where(col_mask, acc_out, zero_col)
    ct.store(out_ptr, index=(col_block,), tile=acc_out)


def _shape_tuple(shape):
    return tuple(int(dim) for dim in shape)


@oracle_impl(hardware="B200", point="fd01dd4f", BLOCK_M=128, BLOCK_N=8)
@oracle_impl(hardware="B200", point="9ca5cb3f", BLOCK_M=128, BLOCK_N=16)
@oracle_impl(hardware="B200", point="c14b0cba", BLOCK_M=128, BLOCK_N=16)
@oracle_impl(hardware="B200", point="4252d9ce", BLOCK_M=128, BLOCK_N=32)
@oracle_impl(hardware="B200", point="cc365d90", BLOCK_M=128, BLOCK_N=16)
@oracle_impl(hardware="B200", point="f696cede", BLOCK_M=128, BLOCK_N=16)
@oracle_impl(hardware="B200", point="73dfcea9", BLOCK_M=128, BLOCK_N=16)
@oracle_impl(hardware="B200", point="2ed1d266", BLOCK_M=128, BLOCK_N=16)
def oracle_forward(inputs, *, BLOCK_M: int, BLOCK_N: int):
    x, _shape_param_0, shape_param_1, shape_param_2 = inputs
    view_shape = _shape_tuple(shape_param_1)
    out_shape = _shape_tuple(shape_param_2)
    m, n = view_shape

    view_1 = torch.as_strided(x, view_shape, (n, 1))
    permute_1 = torch.as_strided(x, (n, m), (1, n))

    # Cuti kernel needs a contiguous [M, N] input matching the view semantics.
    x_flat = view_1.contiguous()

    # Padded output — reduces safely without masked stores.
    n_col_tiles = (n + BLOCK_N - 1) // BLOCK_N
    padded_n = n_col_tiles * BLOCK_N
    padded_out = torch.empty(padded_n, device=x.device, dtype=torch.float32)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (n_col_tiles, 1, 1),
        _col_sum_kernel,
        (x_flat, padded_out, m, n, BLOCK_M, BLOCK_N),
    )
    out = padded_out[:n].contiguous().view(out_shape)
    return (view_1, permute_1, out)
