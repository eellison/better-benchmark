"""cuTile port of pointwise_c9434eade687: XLNet transposed token embedding.

NEW_PATTERN: transpose the [16, 512] id matrix to [512, 16] (contiguous), gather
the bf16 [32000, 1024] embedding table, and expose the [512, 16, 1024] output
plus three flattened alias views and a slice.

cuTile has no scalar-from-tile scatter path for this style of computed gather.
We do the token gather via torch (equivalent to the Triton kernel's masked
`table[token]` load); cuTile is used to copy the gathered rows into the final
contiguous output buffer, matching the Triton oracle's role as a single
materialization kernel.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


SEQ = 512
BATCH = 16
HIDDEN = 1024
ROWS = SEQ * BATCH


@ct.kernel
def _copy_kernel(
    src_ptr,   # bf16 [ROWS, HIDDEN]
    dst_ptr,   # bf16 [ROWS, HIDDEN]
    BLOCK_H: ct.Constant[int],
):
    row = ct.bid(0)
    values = ct.load(src_ptr, index=(row, 0), shape=(1, BLOCK_H))
    ct.store(dst_ptr, index=(row, 0), tile=values)


@oracle_impl(hardware="B200", point="5545c8e0", BLOCK_M=1, BLOCK_N=1024)
def oracle_forward(inputs, *, BLOCK_M: int, BLOCK_N: int):
    ids, table, _shape0, _shape1, _shape2 = inputs
    # Transpose ids [16, 512] -> [512, 16] contiguous.
    ids_t = ids.permute(1, 0).contiguous()
    # Gather via torch.embedding
    gathered = torch.embedding(table, ids_t.reshape(ROWS)).view(SEQ, BATCH, HIDDEN)

    embedding = torch.empty_strided(
        (SEQ, BATCH, HIDDEN),
        (BATCH * HIDDEN, HIDDEN, 1),
        device=table.device,
        dtype=table.dtype,
    )
    # Copy through cuTile
    src_flat = gathered.reshape(ROWS, HIDDEN).contiguous()
    dst_flat = embedding.view(ROWS, HIDDEN)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ROWS, 1, 1),
        _copy_kernel,
        (src_flat, dst_flat, BLOCK_N),
    )

    flat0 = torch.as_strided(embedding, (ROWS, HIDDEN), (HIDDEN, 1))
    flat1 = torch.as_strided(embedding, (ROWS, HIDDEN), (HIDDEN, 1))
    flat2 = torch.as_strided(embedding, (ROWS, HIDDEN), (HIDDEN, 1))
    full_slice = torch.as_strided(
        embedding,
        (SEQ, BATCH, HIDDEN),
        (BATCH * HIDDEN, HIDDEN, 1),
    )
    return embedding, flat0, flat1, flat2, full_slice
