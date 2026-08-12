"""cuTile port of pointwise_a42e20fae37b: QKV view/permute/unbind.

Copies (batch, seq, 3, heads, head_dim) source into three (batch, heads, seq,
head_dim) outputs using a cuTile permute-copy kernel.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _qkv_permute_copy(src_ptr, q_ptr_, k_ptr_, v_ptr_,
                      HEAD_DIM: ct.Constant[int]):
    n = ct.bid(0)
    s = ct.bid(1)
    h = ct.bid(2)
    q_tile = ct.load(src_ptr, index=(n, s, 0, h, 0), shape=(1, 1, 1, 1, HEAD_DIM))
    k_tile = ct.load(src_ptr, index=(n, s, 1, h, 0), shape=(1, 1, 1, 1, HEAD_DIM))
    v_tile = ct.load(src_ptr, index=(n, s, 2, h, 0), shape=(1, 1, 1, 1, HEAD_DIM))
    q_tile = ct.reshape(q_tile, (1, 1, 1, HEAD_DIM))
    k_tile = ct.reshape(k_tile, (1, 1, 1, HEAD_DIM))
    v_tile = ct.reshape(v_tile, (1, 1, 1, HEAD_DIM))
    ct.store(q_ptr_, index=(n, h, s, 0), tile=q_tile)
    ct.store(k_ptr_, index=(n, h, s, 0), tile=k_tile)
    ct.store(v_ptr_, index=(n, h, s, 0), tile=v_tile)


@ct.kernel
def _qkv_permute_pad_copy(
    src_ptr,       # bf16 (batch, seq, hidden3)  where hidden3 = 3*heads*head_dim
    q_ptr,         # bf16 (batch, heads, seq, head_dim)
    k_ptr,         # bf16 (batch, heads, seq, head_dim)
    v_ptr,         # bf16 (batch, heads, seq, head_dim)
    HEADS: ct.Constant[int],
    HEAD_DIM: ct.Constant[int],
    PAD_DIM: ct.Constant[int],  # next pow2 >= head_dim
):
    """Handles non-pow2 head_dim by loading padded [PAD_DIM] slices and using
    ct.scatter to write only valid head_dim columns."""
    n = ct.bid(0)
    s = ct.bid(1)
    h = ct.bid(2)

    # Compute flat offsets within src's last dim for q, k, v at head h.
    q_offset = (0 * HEADS + h) * HEAD_DIM  # column offset within hidden3
    k_offset = (1 * HEADS + h) * HEAD_DIM
    v_offset = (2 * HEADS + h) * HEAD_DIM

    # We load PAD_DIM elements starting from these column offsets. cuTile
    # requires shape=(pow2), which PAD_DIM satisfies. Out-of-lane elements
    # (> head_dim within this "head") happen when PAD_DIM > head_dim; those
    # can spill into the next head. Instead, use gather with per-element
    # indices scoped to valid columns.
    col = ct.arange(PAD_DIM, dtype=ct.int64)
    valid = col < HEAD_DIM
    q_cols = col + q_offset
    k_cols = col + k_offset
    v_cols = col + v_offset

    # src is (batch, seq, hidden3); load using gather with 2D index (row=n*seq+s, col=q_cols).
    # Actually simpler: broadcast (batch_id, seq_id) as scalar tiles.
    n_tile = ct.full((PAD_DIM,), 0, dtype=ct.int64) + ct.astype(n, ct.int64)
    s_tile = ct.full((PAD_DIM,), 0, dtype=ct.int64) + ct.astype(s, ct.int64)
    q_vals = ct.gather(src_ptr, (n_tile, s_tile, q_cols))
    k_vals = ct.gather(src_ptr, (n_tile, s_tile, k_cols))
    v_vals = ct.gather(src_ptr, (n_tile, s_tile, v_cols))

    # Scatter to (n, h, s, 0..head_dim). Compute flat scatter index using
    # per-element indices.
    out_col = ct.arange(PAD_DIM, dtype=ct.int64)
    n_out = ct.full((PAD_DIM,), 0, dtype=ct.int64) + ct.astype(n, ct.int64)
    h_out = ct.full((PAD_DIM,), 0, dtype=ct.int64) + ct.astype(h, ct.int64)
    s_out = ct.full((PAD_DIM,), 0, dtype=ct.int64) + ct.astype(s, ct.int64)
    ct.scatter(q_ptr, (n_out, h_out, s_out, out_col), q_vals, mask=valid)
    ct.scatter(k_ptr, (n_out, h_out, s_out, out_col), k_vals, mask=valid)
    ct.scatter(v_ptr, (n_out, h_out, s_out, out_col), v_vals, mask=valid)


def _resolve_shape(shape, numel):
    out = [int(dim) for dim in shape]
    known = 1
    infer_index = -1
    for index, dim in enumerate(out):
        if dim == -1:
            infer_index = index
        else:
            known *= dim
    if infer_index >= 0:
        out[infer_index] = numel // known
    return tuple(out)


def _next_pow2(x):
    p = 1
    while p < x:
        p *= 2
    return p


@oracle_impl(hardware="B200", point="3d46c2a2")
@oracle_impl(hardware="B200", point="36a70cbd")
@oracle_impl(hardware="B200", point="79a90d04")
@oracle_impl(hardware="B200", point="2e43c9fa")
@oracle_impl(hardware="B200", point="3b895fcd")
@oracle_impl(hardware="B200", point="5bfb1085")
def oracle_forward(inputs):
    x, _shape0, shape1_arg = inputs
    shape1 = _resolve_shape(shape1_arg, x.numel())
    batch, seq, three, heads, head_dim = shape1
    device = x.device

    q = torch.empty((batch, heads, seq, head_dim), device=device, dtype=torch.bfloat16)
    k = torch.empty((batch, heads, seq, head_dim), device=device, dtype=torch.bfloat16)
    v = torch.empty((batch, heads, seq, head_dim), device=device, dtype=torch.bfloat16)

    stream = torch.cuda.current_stream()
    is_pow2 = (head_dim & (head_dim - 1)) == 0
    if is_pow2:
        src_view = x.view(batch, seq, three, heads, head_dim)
        ct.launch(
            stream, (batch, seq, heads), _qkv_permute_copy,
            (src_view, q, k, v, head_dim),
        )
    else:
        # Reshape to (batch, seq, hidden3)
        hidden3 = three * heads * head_dim
        src_flat = x.view(batch, seq, hidden3)
        pad_dim = _next_pow2(head_dim)
        ct.launch(
            stream, (batch, seq, heads), _qkv_permute_pad_copy,
            (src_flat, q, k, v, heads, head_dim, pad_dim),
        )
    return q, k, v
