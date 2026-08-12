"""cuTile port of mean_2ac293b38233: Qwen3/Mistral bf16 embedding + RMSNorm.

NEW_PATTERN: gather bf16 embedding row for each token in input_ids (torch), then
compute fp32 mean-square rsqrt (eps=1e-6) row norm in cuTile, cast to bf16,
apply per-feature weight, and return raw + three view aliases of the
normalized backing tensor.

cuTile does not support scalar indexing into an on-device tile for the
computed embedding-row offset, so the embedding gather is done via
torch.embedding on the host launcher and the fp32 RMSNorm/weight-mul is done
in one cuTile row kernel.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _rmsnorm_bf16_kernel(
    raw_ptr,         # bf16 [rows, HIDDEN]
    weight_ptr,      # bf16 [HIDDEN]
    norm_out_ptr,    # bf16 [rows, HIDDEN]
    HIDDEN: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
):
    row = ct.bid(0)
    raw = ct.load(raw_ptr, index=(row, 0), shape=(1, BLOCK_H))
    x = ct.astype(raw, ct.float32)
    sum_sq = ct.sum(x * x)
    inv_rms = ct.rsqrt(sum_sq * (1.0 / HIDDEN) + 1.0e-6)
    norm_bf16 = ct.astype(x * inv_rms, ct.bfloat16)

    weight = ct.load(weight_ptr, index=(0,), shape=(BLOCK_H,))
    weight_f = ct.astype(weight, ct.float32)
    weight_2d = ct.reshape(weight_f, (1, BLOCK_H))
    out = ct.astype(weight_2d * ct.astype(norm_bf16, ct.float32), ct.bfloat16)
    ct.store(norm_out_ptr, index=(row, 0), tile=out)


@oracle_impl(hardware="B200", point="79ba99e2", BLOCK_H=1024)
@oracle_impl(hardware="B200", point="c790717e", BLOCK_H=4096)
def oracle_forward(inputs, *, BLOCK_H: int):
    embedding_table, input_ids, weight, shape0, shape1, shape2 = inputs
    rows = int(input_ids.numel())
    hidden = int(weight.numel())
    base_shape = (1, rows, hidden)
    base_stride = (rows * hidden, hidden, 1)

    # Embedding gather via torch — cuTile doesn't handle a computed embedding
    # row offset from a tile easily. Store the raw output into raw_base and
    # feed the same tensor into the RMSNorm kernel.
    raw_base = torch.empty_strided(
        base_shape,
        base_stride,
        device=embedding_table.device,
        dtype=torch.bfloat16,
    )
    raw_flat = raw_base.view(rows, hidden)
    torch.index_select(embedding_table, 0, input_ids.view(rows).long(), out=raw_flat)

    norm_base = torch.empty_strided(
        base_shape,
        base_stride,
        device=embedding_table.device,
        dtype=torch.bfloat16,
    )
    norm_flat = norm_base.view(rows, hidden)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (rows, 1, 1),
        _rmsnorm_bf16_kernel,
        (raw_flat, weight, norm_flat, hidden, BLOCK_H),
    )
    return (
        raw_base,
        norm_base.view(tuple(int(dim) for dim in shape0)),
        norm_base.view(tuple(int(dim) for dim in shape1)),
        norm_base.view(tuple(int(dim) for dim in shape2)),
    )
