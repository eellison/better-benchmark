"""cuTile port of pointwise_c0f1f15a4e49: Longformer triple layout-copy.

Materializes three fresh contiguous [8192, 768] bf16 outputs from the same
permuted (1,0,2) view of the source [8, 1024, 768].
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 8
SEQ = 1024
DIM = 768


@ct.kernel
def _triple_layout_copy_kernel(
    src,  # (SEQ, BATCH, DIM) bf16 — a permuted view of the input
    out0,  # (SEQ, BATCH, DIM) bf16
    out1,
    out2,
    BLOCK: ct.Constant[int],
):
    s = ct.bid(0)
    b = ct.bid(1)
    d_tile = ct.bid(2)
    vals = ct.load(src, index=(s, b, d_tile), shape=(1, 1, BLOCK))
    ct.store(out0, index=(s, b, d_tile), tile=vals)
    ct.store(out1, index=(s, b, d_tile), tile=vals)
    ct.store(out2, index=(s, b, d_tile), tile=vals)


@oracle_impl(hardware="B200", point="2973ba0c", BLOCK=128)
def oracle_forward(inputs, *, BLOCK):
    x, shape0, _shape1, _shape2 = inputs
    out_shape = tuple(int(dim) for dim in shape0)  # e.g. (8192, 768)

    # x has logical shape [BATCH, SEQ, DIM]; we want to view it as
    # [SEQ, BATCH, DIM] via permute (which is a stride change, not a copy).
    src_view = x.permute(1, 0, 2)  # (SEQ, BATCH, DIM), non-contiguous

    out0 = torch.empty_strided(out_shape, (out_shape[1], 1), device=x.device, dtype=x.dtype)
    out1 = torch.empty_strided(out_shape, (out_shape[1], 1), device=x.device, dtype=x.dtype)
    out2 = torch.empty_strided(out_shape, (out_shape[1], 1), device=x.device, dtype=x.dtype)

    # Reshape each output to (SEQ, BATCH, DIM) — [8192, 768] as [1024, 8, 768].
    out0_v = out0.view(SEQ, BATCH, DIM)
    out1_v = out1.view(SEQ, BATCH, DIM)
    out2_v = out2.view(SEQ, BATCH, DIM)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (SEQ, BATCH, DIM // BLOCK),
        _triple_layout_copy_kernel,
        (src_view, out0_v, out1_v, out2_v, BLOCK),
    )
    return out0, out1, out2
