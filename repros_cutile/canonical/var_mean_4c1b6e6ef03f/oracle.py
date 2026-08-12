"""cuTile port of var_mean_4c1b6e6ef03f: Blenderbot/Pegasus embedding + LayerNorm.

For each row: token embedding gather + position embedding gather, bf16 add,
fp32 var_mean, rsqrt(eps=1e-5), affine bf16 cast. Returns the add tensor and
three alias views of the normalized output.

Mirrors Triton's structure: one kernel per row, all ops fused. For h=2560,
partition the row into 5 chunks of 512 (BLOCK_H=512, 2560=5*512), avoiding
`torch.zeros` padded-tensor allocations. For h=1024, use BLOCK_H=1024.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


EPS = 1.0e-5


@ct.kernel
def _embedding_ln_h2560_kernel(
    token_table_ptr,    # bf16 [VOCAB, 2560]
    token_ids_ptr,      # i64 [rows]
    position_table_ptr, # bf16 [SEQ_LEN, 2560]
    weight_ptr,         # bf16 [2560]
    bias_ptr,           # bf16 [2560]
    add_out_ptr,        # bf16 [rows, 5, 512] (view)
    norm_out_ptr,       # bf16 [rows, 5, 512] (view)
    SEQ_LEN: ct.Constant[int],
    HIDDEN: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
):
    row = ct.bid(0)

    token_tile = ct.load(token_ids_ptr, index=(row,), shape=(1,))
    row_arr = ct.full((1,), row, dtype=ct.int32)
    pos_id = row_arr % SEQ_LEN

    # Chunk 0: cols [0, 512)
    cols0 = ct.arange(BLOCK_H, dtype=ct.int32)
    token_bc = ct.broadcast_to(token_tile, (BLOCK_H,))
    pos_bc = ct.broadcast_to(pos_id, (BLOCK_H,))
    t0 = ct.gather(token_table_ptr, (token_bc, cols0))
    p0 = ct.gather(position_table_ptr, (pos_bc, cols0))
    add0 = ct.astype(ct.astype(t0, ct.float32) + ct.astype(p0, ct.float32), ct.bfloat16)

    # Chunk 1: cols [512, 1024)
    cols1 = cols0 + BLOCK_H
    t1 = ct.gather(token_table_ptr, (token_bc, cols1))
    p1 = ct.gather(position_table_ptr, (pos_bc, cols1))
    add1 = ct.astype(ct.astype(t1, ct.float32) + ct.astype(p1, ct.float32), ct.bfloat16)

    # Chunk 2: cols [1024, 1536)
    cols2 = cols0 + 2 * BLOCK_H
    t2 = ct.gather(token_table_ptr, (token_bc, cols2))
    p2 = ct.gather(position_table_ptr, (pos_bc, cols2))
    add2 = ct.astype(ct.astype(t2, ct.float32) + ct.astype(p2, ct.float32), ct.bfloat16)

    # Chunk 3: cols [1536, 2048)
    cols3 = cols0 + 3 * BLOCK_H
    t3 = ct.gather(token_table_ptr, (token_bc, cols3))
    p3 = ct.gather(position_table_ptr, (pos_bc, cols3))
    add3 = ct.astype(ct.astype(t3, ct.float32) + ct.astype(p3, ct.float32), ct.bfloat16)

    # Chunk 4: cols [2048, 2560)
    cols4 = cols0 + 4 * BLOCK_H
    t4 = ct.gather(token_table_ptr, (token_bc, cols4))
    p4 = ct.gather(position_table_ptr, (pos_bc, cols4))
    add4 = ct.astype(ct.astype(t4, ct.float32) + ct.astype(p4, ct.float32), ct.bfloat16)

    # Store add outputs (5 tile-space columns).
    ct.store(add_out_ptr, index=(row, 0, 0), tile=ct.reshape(add0, (1, 1, BLOCK_H)))
    ct.store(add_out_ptr, index=(row, 1, 0), tile=ct.reshape(add1, (1, 1, BLOCK_H)))
    ct.store(add_out_ptr, index=(row, 2, 0), tile=ct.reshape(add2, (1, 1, BLOCK_H)))
    ct.store(add_out_ptr, index=(row, 3, 0), tile=ct.reshape(add3, (1, 1, BLOCK_H)))
    ct.store(add_out_ptr, index=(row, 4, 0), tile=ct.reshape(add4, (1, 1, BLOCK_H)))

    # Compute mean across all 5 chunks.
    x0 = ct.astype(add0, ct.float32)
    x1 = ct.astype(add1, ct.float32)
    x2 = ct.astype(add2, ct.float32)
    x3 = ct.astype(add3, ct.float32)
    x4 = ct.astype(add4, ct.float32)
    sum_all = ct.sum(x0) + ct.sum(x1) + ct.sum(x2) + ct.sum(x3) + ct.sum(x4)
    mean = sum_all * (1.0 / HIDDEN)

    c0 = x0 - mean
    c1 = x1 - mean
    c2 = x2 - mean
    c3 = x3 - mean
    c4 = x4 - mean
    var_all = (
        ct.sum(c0 * c0) + ct.sum(c1 * c1) + ct.sum(c2 * c2)
        + ct.sum(c3 * c3) + ct.sum(c4 * c4)
    )
    variance = var_all * (1.0 / HIDDEN)
    invstd = ct.rsqrt(variance + EPS)

    # Load weight/bias (5 tiles).
    w0 = ct.astype(ct.load(weight_ptr, index=(0,), shape=(BLOCK_H,)), ct.float32)
    w1 = ct.astype(ct.load(weight_ptr, index=(1,), shape=(BLOCK_H,)), ct.float32)
    w2 = ct.astype(ct.load(weight_ptr, index=(2,), shape=(BLOCK_H,)), ct.float32)
    w3 = ct.astype(ct.load(weight_ptr, index=(3,), shape=(BLOCK_H,)), ct.float32)
    w4 = ct.astype(ct.load(weight_ptr, index=(4,), shape=(BLOCK_H,)), ct.float32)
    b0 = ct.astype(ct.load(bias_ptr, index=(0,), shape=(BLOCK_H,)), ct.float32)
    b1 = ct.astype(ct.load(bias_ptr, index=(1,), shape=(BLOCK_H,)), ct.float32)
    b2 = ct.astype(ct.load(bias_ptr, index=(2,), shape=(BLOCK_H,)), ct.float32)
    b3 = ct.astype(ct.load(bias_ptr, index=(3,), shape=(BLOCK_H,)), ct.float32)
    b4 = ct.astype(ct.load(bias_ptr, index=(4,), shape=(BLOCK_H,)), ct.float32)

    o0 = c0 * invstd * w0 + b0
    o1 = c1 * invstd * w1 + b1
    o2 = c2 * invstd * w2 + b2
    o3 = c3 * invstd * w3 + b3
    o4 = c4 * invstd * w4 + b4

    ct.store(norm_out_ptr, index=(row, 0, 0),
             tile=ct.reshape(ct.astype(o0, ct.bfloat16), (1, 1, BLOCK_H)))
    ct.store(norm_out_ptr, index=(row, 1, 0),
             tile=ct.reshape(ct.astype(o1, ct.bfloat16), (1, 1, BLOCK_H)))
    ct.store(norm_out_ptr, index=(row, 2, 0),
             tile=ct.reshape(ct.astype(o2, ct.bfloat16), (1, 1, BLOCK_H)))
    ct.store(norm_out_ptr, index=(row, 3, 0),
             tile=ct.reshape(ct.astype(o3, ct.bfloat16), (1, 1, BLOCK_H)))
    ct.store(norm_out_ptr, index=(row, 4, 0),
             tile=ct.reshape(ct.astype(o4, ct.bfloat16), (1, 1, BLOCK_H)))


@ct.kernel
def _embedding_ln_block_kernel(
    token_table_ptr,    # bf16 [VOCAB, HIDDEN]
    token_ids_ptr,      # i64 [rows]
    position_table_ptr, # bf16 [SEQ_LEN, HIDDEN]
    weight_ptr,         # bf16 [HIDDEN]
    bias_ptr,           # bf16 [HIDDEN]
    add_out_ptr,        # bf16 [rows, HIDDEN]
    norm_out_ptr,       # bf16 [rows, HIDDEN]
    SEQ_LEN: ct.Constant[int],
    HIDDEN: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
):
    row = ct.bid(0)

    token_tile = ct.load(token_ids_ptr, index=(row,), shape=(1,))
    row_arr = ct.full((1,), row, dtype=ct.int32)
    pos_id = row_arr % SEQ_LEN

    cols = ct.arange(BLOCK_H, dtype=ct.int32)
    token_bc = ct.broadcast_to(token_tile, (BLOCK_H,))
    token = ct.gather(token_table_ptr, (token_bc, cols))
    pos_bc = ct.broadcast_to(pos_id, (BLOCK_H,))
    position = ct.gather(position_table_ptr, (pos_bc, cols))

    add_bf16 = ct.astype(
        ct.astype(token, ct.float32) + ct.astype(position, ct.float32),
        ct.bfloat16,
    )
    ct.store(add_out_ptr, index=(row, 0), tile=ct.reshape(add_bf16, (1, BLOCK_H)))

    x_f = ct.astype(add_bf16, ct.float32)
    mean = ct.sum(x_f) * (1.0 / HIDDEN)
    centered = x_f - mean
    variance = ct.sum(centered * centered) * (1.0 / HIDDEN)
    invstd = ct.rsqrt(variance + EPS)

    weight = ct.astype(ct.load(weight_ptr, index=(0,), shape=(BLOCK_H,)), ct.float32)
    bias = ct.astype(ct.load(bias_ptr, index=(0,), shape=(BLOCK_H,)), ct.float32)

    affine = centered * invstd * weight + bias
    ct.store(norm_out_ptr, index=(row, 0),
             tile=ct.reshape(ct.astype(affine, ct.bfloat16), (1, BLOCK_H)))


HIDDEN_2560 = 2560
BLOCK_H_2560 = 512  # 2560 / 5


def _run_h2560(inputs):
    token_table, token_ids, position_table, weight, bias, shape0, shape1, shape2 = inputs
    batch = int(token_ids.shape[0])
    seq_len = int(token_ids.shape[1])
    hidden = int(token_table.shape[1])
    assert hidden == HIDDEN_2560, f"expected hidden=2560, got {hidden}"
    rows = batch * seq_len

    add_out = torch.empty_strided(
        (batch, seq_len, hidden),
        (seq_len * hidden, hidden, 1),
        device=token_table.device,
        dtype=torch.bfloat16,
    )
    norm_out = torch.empty_strided(
        (rows, hidden),
        (hidden, 1),
        device=token_table.device,
        dtype=torch.bfloat16,
    )

    ids_flat = token_ids.view(-1)
    # Metadata-only reshape: (rows, 5, 512) view of (rows, 2560).
    add_3d = add_out.view(rows, 5, BLOCK_H_2560)
    norm_3d = norm_out.view(rows, 5, BLOCK_H_2560)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (rows, 1, 1),
        _embedding_ln_h2560_kernel,
        (token_table, ids_flat, position_table, weight, bias,
         add_3d, norm_3d, seq_len, hidden, BLOCK_H_2560),
    )

    return (
        add_out,
        norm_out.view(tuple(int(d) for d in shape0)),
        norm_out.view(tuple(int(d) for d in shape1)),
        norm_out.view(tuple(int(d) for d in shape2)),
    )


def _run_block(inputs, *, BLOCK_H: int):
    token_table, token_ids, position_table, weight, bias, shape0, shape1, shape2 = inputs
    batch = int(token_ids.shape[0])
    seq_len = int(token_ids.shape[1])
    hidden = int(token_table.shape[1])
    if BLOCK_H != hidden:
        raise NotImplementedError(
            f"cuTile port only supports BLOCK_H == hidden for block kernel; "
            f"BLOCK_H={BLOCK_H} hidden={hidden}"
        )
    rows = batch * seq_len

    add_out = torch.empty_strided(
        (batch, seq_len, hidden),
        (seq_len * hidden, hidden, 1),
        device=token_table.device,
        dtype=torch.bfloat16,
    )
    norm_out = torch.empty_strided(
        (rows, hidden),
        (hidden, 1),
        device=token_table.device,
        dtype=torch.bfloat16,
    )

    ids_flat = token_ids.view(-1)
    add_2d = add_out.view(rows, hidden)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (rows, 1, 1),
        _embedding_ln_block_kernel,
        (token_table, ids_flat, position_table, weight, bias,
         add_2d, norm_out, seq_len, hidden, BLOCK_H),
    )

    return (
        add_out,
        norm_out.view(tuple(int(d) for d in shape0)),
        norm_out.view(tuple(int(d) for d in shape1)),
        norm_out.view(tuple(int(d) for d in shape2)),
    )


# fa552165: hidden=2560 (BlenderbotForCausalLM)
@oracle_impl(hardware="B200", point="fa552165")
# cd33d4f9: hidden=2560 (BlenderbotForConditionalGeneration)
@oracle_impl(hardware="B200", point="cd33d4f9")
def oracle_forward_h2560(inputs):
    return _run_h2560(inputs)


# f842bd4d: hidden=1024 (PegasusForCausalLM)
@oracle_impl(hardware="B200", point="f842bd4d", BLOCK_H=1024)
def oracle_forward_h1024(inputs, *, BLOCK_H: int):
    return _run_block(inputs, BLOCK_H=BLOCK_H)


def oracle_forward(inputs):
    raise RuntimeError(
        "registered oracle_impl dispatch should select a point-specific implementation"
    )
