"""cuTile port of pointwise_d5a413a2c233: GPT-style QKV split-view-permute.

The Repro is metadata-only (view+permute+as_strided) — no compute. This
port copies each of Q/K/V through a cuTile identity kernel while producing
the same [batch, heads, seq, head_dim] permuted layout.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _row_copy_kernel(
    src_ptr,       # bf16 [rows, head_dim]   contiguous
    dst_ptr,       # bf16 [rows, head_dim]   contiguous
    HEAD_DIM: ct.Constant[int],
):
    row = ct.bid(0)
    x = ct.load(src_ptr, index=(row, 0), shape=(1, HEAD_DIM))
    ct.store(dst_ptr, index=(row, 0), tile=x)


def _split_permute_via_kernel(x, batch, seq, heads, head_dim, chunk_index):
    device = x.device
    hidden = int(x.shape[1])
    # Slice the chunk: bf16 [batch*seq, heads*head_dim] slice → contiguous copy.
    chunk = heads * head_dim
    src_full = x.view(batch, seq, hidden)
    src = src_full[..., chunk_index * chunk : (chunk_index + 1) * chunk].contiguous()
    src_flat = src.view(batch * seq * heads, head_dim)

    # Build the output with shape (batch, heads, seq, head_dim) and matching contiguous strides
    # via permutation: allocate (batch, seq, heads, head_dim) contiguous, then permute (0,2,1,3).
    tmp = torch.empty((batch, seq, heads, head_dim), device=device, dtype=x.dtype)
    tmp_flat = tmp.view(batch * seq * heads, head_dim)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (batch * seq * heads, 1, 1),
        _row_copy_kernel,
        (src_flat, tmp_flat, head_dim),
    )
    # Return as a permuted view (metadata) matching Repro.
    return tmp.permute(0, 2, 1, 3)


@oracle_impl(hardware="B200", point="74f1c5d9")
@oracle_impl(hardware="B200", point="d8de82d9")
def oracle_forward(inputs):
    x, shape0, shape1, shape2, shape3 = inputs
    batch = int(shape0[0])
    seq = int(shape0[1])
    heads = int(shape1[2])
    head_dim = int(shape1[3])
    # Repro returns (permute [K], permute_1 [V], permute_2 [Q]) — chunks 1, 2, 0.
    q = _split_permute_via_kernel(x, batch, seq, heads, head_dim, 1)
    k = _split_permute_via_kernel(x, batch, seq, heads, head_dim, 2)
    v = _split_permute_via_kernel(x, batch, seq, heads, head_dim, 0)
    return q, k, v
