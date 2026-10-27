"""cuTile port of var_mean_fda6b7a40722: DistillGPT2 embedding LayerNorm.

- Uses torch's embedding op to gather token/position embeddings before the kernel.
- The LN row kernel loads add[row], reduces over hidden=768 (padded to BLOCK_H=1024 with zero).
- Also emits an all-False bool mask output (adjacent position differences equal 1).
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


EPS = 1.0e-5


@ct.kernel
def _layernorm_row_kernel(
    add_ptr,        # bf16 [rows, HIDDEN_PAD], with HIDDEN valid cols
    weight_ptr,     # bf16 [HIDDEN_PAD]
    bias_ptr,       # bf16 [HIDDEN_PAD]
    norm_out_ptr,   # bf16 [rows, HIDDEN_PAD]
    HIDDEN: ct.Constant[int],
    EPSILON: ct.Constant[float],
    BLOCK_H: ct.Constant[int],
):
    row = ct.bid(0)

    add = ct.load(
        add_ptr,
        index=(row, 0),
        shape=(1, BLOCK_H),
    )
    add_f = ct.astype(add, ct.float32)

    # Valid columns mask (HIDDEN out of BLOCK_H may have zero-padding at high cols)
    # We used torch's zero-padding when we allocated the buffers; sum works fine.
    mean = ct.sum(add_f) * (1.0 / HIDDEN)
    centered = add_f - mean
    # Zero out the padded high cols before variance sum
    # Use ct.arange to build a column mask
    cols = ct.arange(BLOCK_H, dtype=ct.int32)
    col_valid = cols < HIDDEN
    col_valid_2d = ct.reshape(col_valid, (1, BLOCK_H))
    centered_masked = ct.where(col_valid_2d, centered, 0.0)
    variance = ct.sum(centered_masked * centered_masked) * (1.0 / HIDDEN)
    invstd = ct.rsqrt(variance + EPSILON)
    normalized = centered * invstd

    weight = ct.astype(
        ct.load(weight_ptr, index=(0,), shape=(BLOCK_H,)),
        ct.float32,
    )
    bias = ct.astype(
        ct.load(bias_ptr, index=(0,), shape=(BLOCK_H,)),
        ct.float32,
    )
    weight_2d = ct.reshape(weight, (1, BLOCK_H))
    bias_2d = ct.reshape(bias, (1, BLOCK_H))

    affine = normalized * weight_2d + bias_2d
    ct.store(norm_out_ptr, index=(row, 0), tile=ct.astype(affine, ct.bfloat16))


@oracle_impl(hardware="B200", point="7a2cdb11", BLOCK_M=2, BLOCK_H=1024)
def oracle_forward(inputs, *, BLOCK_M: int, BLOCK_H: int):
    token_table, token_ids, position_table, weight, bias, _shape0, _shape1 = inputs
    batch = int(token_ids.shape[0])
    seq_len = int(token_ids.shape[1])
    hidden = int(token_table.shape[1])
    rows = batch * seq_len

    # Compute embeddings + add in torch (fine — it's out of kernel scope)
    # token embed: [batch, seq_len, hidden]
    token_embed = torch.embedding(token_table, token_ids)
    # position embed for positions 0..seq_len-1
    pos_iota = torch.arange(seq_len, device=token_table.device, dtype=torch.int64)
    position_embed = torch.embedding(position_table, pos_iota).unsqueeze(0)
    add_bf16 = (token_embed.to(torch.float32) + position_embed.to(torch.float32)).to(torch.bfloat16)

    # add_bf16 shape [batch, seq_len, hidden]. For the kernel, we need [rows, HIDDEN_PAD]
    # where HIDDEN_PAD = BLOCK_H. If BLOCK_H > hidden, we need padded storage.
    # Simplest: allocate rows x BLOCK_H buffer, zero, copy hidden slice in.
    add_padded = torch.zeros((rows, BLOCK_H), device=token_table.device, dtype=torch.bfloat16)
    add_padded[:, :hidden] = add_bf16.view(rows, hidden)

    weight_padded = torch.zeros((BLOCK_H,), device=token_table.device, dtype=torch.bfloat16)
    weight_padded[:hidden] = weight
    bias_padded = torch.zeros((BLOCK_H,), device=token_table.device, dtype=torch.bfloat16)
    bias_padded[:hidden] = bias

    norm_padded = torch.empty((rows, BLOCK_H), device=token_table.device, dtype=torch.bfloat16)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (rows, 1, 1),
        _layernorm_row_kernel,
        (add_padded, weight_padded, bias_padded, norm_padded, hidden, EPS, BLOCK_H),
    )

    # Slice the valid part of norm and reshape to expected view shape [rows, hidden]
    norm_out = norm_padded[:, :hidden].contiguous()
    # ne mask is always False since positions are consecutive (differences = 1)
    mask_out = torch.zeros((batch, seq_len), device=token_table.device, dtype=torch.bool)

    return add_bf16, norm_out, mask_out
