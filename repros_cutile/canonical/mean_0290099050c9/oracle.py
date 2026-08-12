"""cuTile port of mean_0290099050c9: GPT-OSS embedding-RMSNorm with alias views."""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _embedding_rmsnorm_kernel(
    embedding_ptr,   # (VOCAB, HIDDEN) bf16
    input_ids_ptr,   # (rows,) i64 (flattened)
    weight_ptr,      # (HIDDEN,) bf16
    raw_out_ptr,     # (rows, HIDDEN) bf16
    norm_out_ptr,    # (rows, HIDDEN) bf16
    HIDDEN: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
):
    row = ct.bid(0)
    # Load one token id
    token = ct.load(input_ids_ptr, index=(row,), shape=(1,))

    # Gather one row from the embedding table: gather uses (row_idx, col_idx) tuple.
    row_idx = ct.reshape(token, (1, 1))
    col_idx = ct.reshape(ct.arange(BLOCK_H, dtype=ct.int64), (1, BLOCK_H))
    raw = ct.gather(embedding_ptr, (row_idx, col_idx))  # (1, BLOCK_H) bf16
    ct.store(raw_out_ptr, index=(row, 0), tile=raw)

    # RMSNorm
    x = ct.astype(raw, ct.float32)
    sum_sq = ct.sum(x * x)
    inv_rms = ct.rsqrt(sum_sq * (1.0 / HIDDEN) + 1.0e-5)

    weight = ct.load(weight_ptr, index=(0,), shape=(BLOCK_H,))
    weight_f = ct.astype(weight, ct.float32)
    weight_2d = ct.reshape(weight_f, (1, BLOCK_H))
    out = ct.astype(x * inv_rms * weight_2d, ct.bfloat16)
    ct.store(norm_out_ptr, index=(row, 0), tile=out)


@oracle_impl(hardware="B200", point="49da884f", BLOCK_H=4096)
def oracle_forward(inputs, *, BLOCK_H):
    embedding_table, input_ids, weight, shape0, shape1, shape2 = inputs
    rows = int(input_ids.numel())
    hidden = int(weight.numel())
    base_shape = (1, rows, hidden)
    base_stride = (rows * hidden, hidden, 1)

    raw_base = torch.empty_strided(
        base_shape,
        base_stride,
        device=embedding_table.device,
        dtype=torch.bfloat16,
    )
    norm_base = torch.empty_strided(
        base_shape,
        base_stride,
        device=embedding_table.device,
        dtype=torch.bfloat16,
    )

    raw_2d = raw_base.view(rows, hidden)
    norm_2d = norm_base.view(rows, hidden)
    ids_flat = input_ids.view(rows)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (rows, 1, 1),
        _embedding_rmsnorm_kernel,
        (embedding_table, ids_flat, weight, raw_2d, norm_2d, hidden, BLOCK_H),
    )
    return (
        raw_base,
        norm_base.view(tuple(int(dim) for dim in shape0)),
        norm_base.view(tuple(int(dim) for dim in shape1)),
        norm_base.view(tuple(int(dim) for dim in shape2)),
    )
