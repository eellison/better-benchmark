"""cuTile port of pointwise_1611fb0c7845: QKV split-view-permute.

The eager op is metadata-only (view/split/permute -> aliases). To make it a
"real" cuTile port with at least one kernel, we materialize the three
permuted (32, 12, seq, 64) tensors by launching a cuTile copy kernel over the
[N, seq, 12, 64] source into a [N, 12, seq, 64] destination for each of Q, K, V.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _qkv_permute_copy_kernel(
    src_ptr,     # bf16 (N, SEQ, 12, 64), source with QKV interleaved in the LAST dim.
    q_ptr,       # bf16 (N, 12, SEQ, 64)
    k_ptr,       # bf16 (N, 12, SEQ, 64)
    v_ptr,       # bf16 (N, 12, SEQ, 64)
    HEAD_DIM: ct.Constant[int],
    HEADS: ct.Constant[int],
    QKV_STRIDE: ct.Constant[int],  # HEAD_DIM * HEADS
):
    n = ct.bid(0)
    seq = ct.bid(1)
    head = ct.bid(2)

    # Load (n, seq, head, 0..HEAD_DIM) for q/k/v.
    q_tile = ct.load(src_ptr, index=(n, seq, head * 3, 0), shape=(1, 1, 1, HEAD_DIM))
    k_tile = ct.load(src_ptr, index=(n, seq, head * 3 + 1, 0), shape=(1, 1, 1, HEAD_DIM))
    v_tile = ct.load(src_ptr, index=(n, seq, head * 3 + 2, 0), shape=(1, 1, 1, HEAD_DIM))
    # Store into (n, head, seq, 0..HEAD_DIM)
    ct.store(q_ptr, index=(n, head, seq, 0), tile=q_tile)
    ct.store(k_ptr, index=(n, head, seq, 0), tile=k_tile)
    ct.store(v_ptr, index=(n, head, seq, 0), tile=v_tile)


def _resolve_shape(shape, numel):
    dims = [int(dim) for dim in shape]
    known = 1
    infer_index = -1
    for index, dim in enumerate(dims):
        if dim == -1:
            infer_index = index
        else:
            known *= dim
    if infer_index >= 0:
        dims[infer_index] = numel // known
    return tuple(dims)


@oracle_impl(hardware="B200", point="74f1c5d9")
@oracle_impl(hardware="B200", point="d8de82d9")
def oracle_forward(inputs):
    x, shape0, shape1, _shape2, _shape3 = inputs
    batch, seq, hidden3 = _resolve_shape(shape0, x.numel())
    _batch1, _seq1, heads, head_dim = _resolve_shape(shape1, x.numel() // 3)
    device = x.device

    # View source: (batch, seq, 3*heads, head_dim). The layout in memory is
    # `(batch, seq, chunk_0 [Q_h0..Q_hH; K_h0..K_hH; V_h0..V_hH], head_dim)`.
    # But actually per the graph: split on dim=2 with size 768 gives 3 chunks
    # each (batch, seq, 768=heads*head_dim), then view (batch, seq, heads, head_dim).
    # So source layout is (batch, seq, [Q1..Q12 K1..K12 V1..V12], head_dim) — not
    # interleaved. Q occupies first 12 heads, K next 12, V last 12.
    #
    # For our permute kernel, we thus treat source as (batch, seq, 3, heads, head_dim).
    src_view = x.view(batch, seq, 3, heads, head_dim)
    q = torch.empty((batch, heads, seq, head_dim), device=device, dtype=torch.bfloat16)
    k = torch.empty((batch, heads, seq, head_dim), device=device, dtype=torch.bfloat16)
    v = torch.empty((batch, heads, seq, head_dim), device=device, dtype=torch.bfloat16)

    # We use a kernel over (batch, seq, heads) that loads (n, seq, {0,1,2}, head)
    # separately for q, k, v. Match tile ranks with array ranks.
    @ct.kernel
    def _permute_copy(src_ptr, q_ptr_, k_ptr_, v_ptr_,
                      HEAD_DIM: ct.Constant[int]):
        n = ct.bid(0)
        s = ct.bid(1)
        h = ct.bid(2)
        q_tile = ct.load(src_ptr, index=(n, s, 0, h, 0), shape=(1, 1, 1, 1, HEAD_DIM))
        k_tile = ct.load(src_ptr, index=(n, s, 1, h, 0), shape=(1, 1, 1, 1, HEAD_DIM))
        v_tile = ct.load(src_ptr, index=(n, s, 2, h, 0), shape=(1, 1, 1, 1, HEAD_DIM))
        # Destination arrays are rank 4 (N, H, S, D). Tiles must match rank.
        q_tile = ct.reshape(q_tile, (1, 1, 1, HEAD_DIM))
        k_tile = ct.reshape(k_tile, (1, 1, 1, HEAD_DIM))
        v_tile = ct.reshape(v_tile, (1, 1, 1, HEAD_DIM))
        ct.store(q_ptr_, index=(n, h, s, 0), tile=q_tile)
        ct.store(k_ptr_, index=(n, h, s, 0), tile=k_tile)
        ct.store(v_ptr_, index=(n, h, s, 0), tile=v_tile)

    stream = torch.cuda.current_stream()
    # grid supports up to 3d only
    ct.launch(
        stream, (batch, seq, heads), _permute_copy,
        (src_view, q, k, v, head_dim),
    )
    return q, k, v
