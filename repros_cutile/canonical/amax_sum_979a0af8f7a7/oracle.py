"""cuTile port of amax_sum_979a0af8f7a7: T5/MT5 additive-bias bf16 softmax.

Adds a bias tensor (with a non-trivial layout — head is the innermost dim
via strides) to bf16 attention scores, then computes stable softmax and
casts to bf16.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _bf16_add_softmax_kernel(
    x_ptr,       # bf16 [BH, Q, K]
    bias_ptr,    # bf16 [B, Q, K, H] as_strided (contiguous storage)
    out_ptr,     # bf16 [BH, Q, K]
    HEADS: ct.Constant[int],
    Q: ct.Constant[int],
    BLOCK_M: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
):
    # Batch BLOCK_M rows per program to cut launch count by BLOCK_M.
    row_block = ct.bid(0)
    # All BLOCK_M rows in one program share the same (batch, head): we require
    # BLOCK_M to divide Q. flat_bh identifies which (batch, head) slice we're in;
    # query_tile is the tile index along the query axis (in units of BLOCK_M).
    q_tiles = Q // BLOCK_M
    flat_bh = row_block // q_tiles
    query_tile = row_block % q_tiles
    batch = flat_bh // HEADS
    head = flat_bh % HEADS

    # Load BLOCK_M contiguous rows: shape (BLOCK_M, BLOCK_N).
    x_rows = ct.load(x_ptr, index=(row_block, 0), shape=(BLOCK_M, BLOCK_N))
    # bias[batch, query_start:query_start+BLOCK_M, :, head] — tile-space index
    # for shape (1, BLOCK_M, BLOCK_N, 1) is (batch, query_tile, 0, head).
    bias_tile = ct.load(
        bias_ptr,
        index=(batch, query_tile, 0, head),
        shape=(1, BLOCK_M, BLOCK_N, 1),
    )
    bias_2d = ct.reshape(bias_tile, (BLOCK_M, BLOCK_N))

    x_f = ct.astype(x_rows, ct.float32)
    bias_f = ct.astype(bias_2d, ct.float32)
    scores = x_f + bias_f

    row_max = ct.max(scores, axis=1, keepdims=True)
    numer = ct.exp(scores - row_max)
    denom = ct.sum(numer, axis=1, keepdims=True)
    probs = numer / denom
    ct.store(out_ptr, index=(row_block, 0), tile=ct.astype(probs, ct.bfloat16))


@oracle_impl(hardware="B200", point="d3b915a9", BLOCK_M=8, BLOCK_N=128)
@oracle_impl(hardware="B200", point="679762f8", BLOCK_M=1, BLOCK_N=1024)
def oracle_forward(inputs, *, BLOCK_M: int, BLOCK_N: int):
    arg0_1, arg1_1, _shape_param_0, _shape_param_1, _shape_param_2 = inputs
    del _shape_param_0, _shape_param_1

    out_shape = tuple(int(dim) for dim in _shape_param_2)
    out = torch.empty_strided(
        out_shape,
        (out_shape[1] * out_shape[2], out_shape[2], 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )

    q_len = int(arg0_1.shape[1])
    k_len = int(arg0_1.shape[2])
    n_rows = int(arg0_1.shape[0] * q_len)
    batch = int(arg1_1.shape[0])
    heads = int(arg1_1.shape[1])

    # arg1_1 has shape [B, H, Q, K] with strides [Q*K*H, 1, K*H, H] —
    # storage is contiguous [B, Q, K, H]. Reinterpret as [B, Q, K, H] via
    # as_strided (no copy). The earlier permute+contiguous+view was dead
    # code — as_strided always overwrites it.
    bias_bqkh = torch.as_strided(
        arg1_1, (batch, q_len, k_len, heads),
        (q_len * k_len * heads, k_len * heads, heads, 1),
    )

    out_2d = out.view(n_rows, k_len)

    stream = torch.cuda.current_stream()
    # Batch BLOCK_M rows per program: grid (n_rows / BLOCK_M, 1, 1).
    grid_m = n_rows // BLOCK_M
    ct.launch(
        stream,
        (grid_m, 1, 1),
        _bf16_add_softmax_kernel,
        (arg0_1.view(n_rows, k_len), bias_bqkh, out_2d,
         heads, q_len, BLOCK_M, k_len),
    )
    return out
