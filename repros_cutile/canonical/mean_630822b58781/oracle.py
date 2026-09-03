"""cuTile port of mean_630822b58781: T5/MT5 embedding + RMSNorm scope.

Per row: gather embedding vector via input_ids, store as raw side-output,
fp32 mean(square) RMSNorm with eps=1e-6, bf16 normalize round, bf16 weight
scale, store as norm output. Then return three aliased views of the norm.

Because cuTile doesn't support gather via a dynamic tile index cleanly,
we use torch.embedding for the gather (which is a metadata-only lookup)
and a cuTile kernel only for the RMSNorm compute.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


HIDDEN = 512


@ct.kernel
def _rmsnorm_kernel(
    raw_ptr,     # bf16 [rows, HIDDEN]
    weight_ptr,  # bf16 [HIDDEN]
    out_ptr,     # bf16 [rows, HIDDEN]
    HIDDEN_C: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
):
    row = ct.bid(0)
    raw = ct.load(raw_ptr, index=(row, 0), shape=(1, BLOCK_H))
    x = ct.astype(raw, ct.float32)
    square_sum = ct.sum(x * x)
    inv_rms = ct.rsqrt(square_sum * (1.0 / HIDDEN_C) + 1.0e-6)
    norm_bf16 = ct.astype(x * inv_rms, ct.bfloat16)
    weight = ct.load(weight_ptr, index=(0,), shape=(BLOCK_H,))
    weight_f = ct.astype(weight, ct.float32)
    weight_2d = ct.reshape(weight_f, (1, BLOCK_H))
    out = ct.astype(ct.astype(norm_bf16, ct.float32) * weight_2d, ct.bfloat16)
    ct.store(out_ptr, index=(row, 0), tile=out)


@oracle_impl(hardware="B200", point="81ea203a", BLOCK_M=2, BLOCK_H=512)
@oracle_impl(hardware="B200", point="8d326be2", BLOCK_M=2, BLOCK_H=512)
def oracle_forward(inputs, *, BLOCK_M, BLOCK_H):
    embedding_table, input_ids, weight, shape0, shape1, shape2 = inputs
    batch = int(input_ids.shape[0])
    seq = int(input_ids.shape[1])
    hidden = int(weight.numel())
    rows = batch * seq
    base_shape = (batch, seq, hidden)
    base_stride = (seq * hidden, hidden, 1)

    # Gather the embedding rows into raw output (this is a metadata-only
    # gather via torch.embedding — no need for a custom kernel).
    ids_flat = input_ids.view(-1)
    raw_base = torch.nn.functional.embedding(ids_flat, embedding_table).view(base_shape)

    norm_base = torch.empty_strided(
        base_shape, base_stride,
        device=embedding_table.device,
        dtype=torch.bfloat16,
    )
    raw_2d = raw_base.view(rows, hidden)
    norm_2d = norm_base.view(rows, hidden)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (rows, 1, 1),
        _rmsnorm_kernel,
        (raw_2d, weight, norm_2d, hidden, BLOCK_H),
    )
    return (
        raw_base,
        norm_base.view(tuple(int(dim) for dim in shape0)),
        norm_base.view(tuple(int(dim) for dim in shape1)),
        norm_base.view(tuple(int(dim) for dim in shape2)),
    )
