"""cuTile port of mean_104f67c766f1: Gemma embedding-scale RMSNorm row kernel.

Ports the Triton `_embedding_scaled_rmsnorm_kernel` — for each row: gather
embedding by token id, multiply by scalar scale (bf16), row-mean-square,
rsqrt with eps, affine with (weight.float() + 1.0), final bf16 cast.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _embedding_scaled_rmsnorm_kernel(
    embedding_ptr,   # bf16 [VOCAB, HIDDEN]
    input_ids_ptr,   # i64 [rows]
    scale_ptr,       # bf16 scalar (shape [1])
    weight_ptr,      # bf16 [HIDDEN]
    scaled_out_ptr,  # bf16 [rows, HIDDEN]
    norm_out_ptr,    # bf16 [rows, HIDDEN]
    HIDDEN: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
):
    row = ct.bid(0)

    # Load the token id for this row (a scalar).
    token_tile = ct.load(input_ids_ptr, index=(row,), shape=(1,))
    # Use as row index into the embedding table.
    cols = ct.arange(BLOCK_H, dtype=ct.int32)
    token_broadcast = ct.broadcast_to(token_tile, (BLOCK_H,))
    embed = ct.gather(embedding_ptr, (token_broadcast, cols))

    scale_tile = ct.load(scale_ptr, index=(0,), shape=(1,))
    scale_f32 = ct.astype(scale_tile, ct.float32)
    embed_f32 = ct.astype(embed, ct.float32)
    scaled_bf16 = ct.astype(embed_f32 * scale_f32, ct.bfloat16)

    # scaled_out has been reshaped to 2D [rows, HIDDEN] at launch.
    scaled_2d = ct.reshape(scaled_bf16, (1, BLOCK_H))
    ct.store(scaled_out_ptr, index=(row, 0), tile=scaled_2d)

    scaled_f32 = ct.astype(scaled_bf16, ct.float32)
    sum_sq = ct.sum(scaled_f32 * scaled_f32)
    inv_rms = ct.rsqrt(sum_sq * (1.0 / HIDDEN) + 1.0e-6)

    weight = ct.load(weight_ptr, index=(0,), shape=(BLOCK_H,))
    weight_f = ct.astype(weight, ct.float32)
    out = scaled_f32 * inv_rms * (weight_f + 1.0)
    out_bf16 = ct.astype(out, ct.bfloat16)
    out_2d = ct.reshape(out_bf16, (1, BLOCK_H))
    ct.store(norm_out_ptr, index=(row, 0), tile=out_2d)


@oracle_impl(hardware="B200", point="65410254", BLOCK_H=4096)
@oracle_impl(hardware="B200", point="402c18a0", BLOCK_H=4096)
def oracle_forward(inputs, *, BLOCK_H: int):
    embedding, input_ids, scale, weight, shape0, shape1, shape2 = inputs
    rows = int(input_ids.numel())
    hidden = int(embedding.shape[1])
    base_shape = (1, rows, hidden)

    if BLOCK_H < hidden:
        raise NotImplementedError(
            f"cuTile port unsupported: BLOCK_H={BLOCK_H} < hidden={hidden}"
        )
    if hidden != BLOCK_H:
        # cuTile requires tile shapes to be powers of 2 and store the whole
        # tile. For hidden that doesn't fit BLOCK_H exactly, we need matching.
        pass

    scaled_out = torch.empty_strided(
        base_shape,
        (rows * hidden, hidden, 1),
        device=embedding.device,
        dtype=torch.bfloat16,
    )
    norm_base = torch.empty_strided(
        base_shape,
        (rows * hidden, hidden, 1),
        device=embedding.device,
        dtype=torch.bfloat16,
    )

    # Flatten input_ids to 1D of length `rows`.
    ids_flat = input_ids.view(-1)
    # scale is a 0-d bf16 tensor; view as [1] for cuTile load
    scale_1d = scale.view(1)

    # Reshape 3D outputs to 2D for cuTile 2D tile stores.
    scaled_2d = scaled_out.view(rows, hidden)
    norm_2d = norm_base.view(rows, hidden)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (rows, 1, 1),
        _embedding_scaled_rmsnorm_kernel,
        (embedding, ids_flat, scale_1d, weight, scaled_2d, norm_2d, hidden, BLOCK_H),
    )
    return (
        scaled_out,
        norm_base.view(tuple(int(dim) for dim in shape0)),
        norm_base.view(tuple(int(dim) for dim in shape1)),
        norm_base.view(tuple(int(dim) for dim in shape2)),
    )
