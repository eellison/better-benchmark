"""cuTile port of mean_mean_ee4f8b590982: Gemma Q/K RMSNorm + precomputed-RoPE.

For each row (batch=1, so row = pos*heads + head):
  x [HEAD_DIM=256] bf16
  RMSNorm with weight_shifted=(weight+1.0), applying bf16 rounding.
  Rotate: rot = concat(-x[HALF:], x[:HALF]) with HALF=128
  RoPE: out = round(value*cos) + round(rot*sin)
Query pass -> query_rope; Key pass -> key_rope + repeated_key.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


HEAD_DIM = 256
HALF_DIM = 128
EPS = 1.0e-6


@ct.kernel
def _rmsnorm_rope_kernel(
    x_ptr,         # bf16 [rows, HEAD_DIM]  (rows = seq * heads)
    weight_ptr,    # bf16 [HEAD_DIM]
    cos_ptr,       # bf16 [seq, HEAD_DIM]
    sin_ptr,       # bf16 [seq, HEAD_DIM]
    out_ptr,       # bf16 [rows, HEAD_DIM] contiguous view of the strided output
    NUM_HEADS: ct.Constant[int],
):
    row = ct.bid(0)
    pos = row // NUM_HEADS

    x_low = ct.load(x_ptr, index=(row, 0), shape=(1, HALF_DIM))
    x_high = ct.load(x_ptr, index=(row, 1), shape=(1, HALF_DIM))
    x_low_f = ct.astype(x_low, ct.float32)
    x_high_f = ct.astype(x_high, ct.float32)

    sum_sq_low = ct.sum(x_low_f * x_low_f, axis=1, keepdims=True)
    sum_sq_high = ct.sum(x_high_f * x_high_f, axis=1, keepdims=True)
    inv_rms = ct.rsqrt((sum_sq_low + sum_sq_high) * (1.0 / HEAD_DIM) + EPS)

    w_low = ct.load(weight_ptr, index=(0,), shape=(HALF_DIM,))
    w_high = ct.load(weight_ptr, index=(1,), shape=(HALF_DIM,))
    # Gemma shifts: weight = weight + 1.0
    w_low_2d = ct.reshape(ct.astype(w_low, ct.float32), (1, HALF_DIM)) + 1.0
    w_high_2d = ct.reshape(ct.astype(w_high, ct.float32), (1, HALF_DIM)) + 1.0

    norm_low = x_low_f * inv_rms
    norm_high = x_high_f * inv_rms
    value_low_bf = ct.astype(norm_low * w_low_2d, ct.bfloat16)
    value_high_bf = ct.astype(norm_high * w_high_2d, ct.bfloat16)

    # rot: low_half = -x_high (with weight = w_high), high_half = x_low (weight = w_low)
    rot_low_f = 0.0 - (norm_high * w_high_2d)   # matches -(value_high) after bf16 round
    rot_high_f = norm_low * w_low_2d
    rot_low_bf = ct.astype(rot_low_f, ct.bfloat16)   # -x_high rounded
    rot_high_bf = ct.astype(rot_high_f, ct.bfloat16)

    # Load cos/sin halves at pos
    cos_low = ct.load(cos_ptr, index=(pos, 0), shape=(1, HALF_DIM))
    cos_high = ct.load(cos_ptr, index=(pos, 1), shape=(1, HALF_DIM))
    sin_low = ct.load(sin_ptr, index=(pos, 0), shape=(1, HALF_DIM))
    sin_high = ct.load(sin_ptr, index=(pos, 1), shape=(1, HALF_DIM))

    val_low_f = ct.astype(value_low_bf, ct.float32)
    val_high_f = ct.astype(value_high_bf, ct.float32)
    cos_low_f = ct.astype(cos_low, ct.float32)
    cos_high_f = ct.astype(cos_high, ct.float32)
    sin_low_f = ct.astype(sin_low, ct.float32)
    sin_high_f = ct.astype(sin_high, ct.float32)
    rot_low_bf_f = ct.astype(rot_low_bf, ct.float32)
    rot_high_bf_f = ct.astype(rot_high_bf, ct.float32)

    out_low = ct.astype(val_low_f * cos_low_f, ct.bfloat16)
    out_high = ct.astype(val_high_f * cos_high_f, ct.bfloat16)
    rot_low_a = ct.astype(rot_low_bf_f * sin_low_f, ct.bfloat16)
    rot_high_a = ct.astype(rot_high_bf_f * sin_high_f, ct.bfloat16)

    final_low = ct.astype(out_low, ct.float32) + ct.astype(rot_low_a, ct.float32)
    final_high = ct.astype(out_high, ct.float32) + ct.astype(rot_high_a, ct.float32)
    ct.store(out_ptr, index=(row, 0), tile=ct.astype(final_low, ct.bfloat16))
    ct.store(out_ptr, index=(row, 1), tile=ct.astype(final_high, ct.bfloat16))


@ct.kernel
def _rmsnorm_rope_key_repeat_kernel(
    x_ptr,           # bf16 [rows, HEAD_DIM]   rows = seq * kv_heads
    weight_ptr,      # bf16 [HEAD_DIM]
    cos_ptr,         # bf16 [seq, HEAD_DIM]
    sin_ptr,         # bf16 [seq, HEAD_DIM]
    key_out_ptr,     # bf16 [rows, HEAD_DIM]   contiguous view of key_rope
    repeated_ptr,    # bf16 [q_heads, seq, HEAD_DIM]   contiguous
    NUM_KV_HEADS: ct.Constant[int],
    REPEAT: ct.Constant[int],
):
    row = ct.bid(0)
    pos = row // NUM_KV_HEADS
    kv_head = row - pos * NUM_KV_HEADS

    x_low = ct.load(x_ptr, index=(row, 0), shape=(1, HALF_DIM))
    x_high = ct.load(x_ptr, index=(row, 1), shape=(1, HALF_DIM))
    x_low_f = ct.astype(x_low, ct.float32)
    x_high_f = ct.astype(x_high, ct.float32)
    sum_sq_low = ct.sum(x_low_f * x_low_f, axis=1, keepdims=True)
    sum_sq_high = ct.sum(x_high_f * x_high_f, axis=1, keepdims=True)
    inv_rms = ct.rsqrt((sum_sq_low + sum_sq_high) * (1.0 / HEAD_DIM) + EPS)

    w_low = ct.load(weight_ptr, index=(0,), shape=(HALF_DIM,))
    w_high = ct.load(weight_ptr, index=(1,), shape=(HALF_DIM,))
    w_low_2d = ct.reshape(ct.astype(w_low, ct.float32), (1, HALF_DIM)) + 1.0
    w_high_2d = ct.reshape(ct.astype(w_high, ct.float32), (1, HALF_DIM)) + 1.0

    norm_low = x_low_f * inv_rms
    norm_high = x_high_f * inv_rms
    value_low_bf = ct.astype(norm_low * w_low_2d, ct.bfloat16)
    value_high_bf = ct.astype(norm_high * w_high_2d, ct.bfloat16)

    rot_low_f = 0.0 - (norm_high * w_high_2d)
    rot_high_f = norm_low * w_low_2d
    rot_low_bf = ct.astype(rot_low_f, ct.bfloat16)
    rot_high_bf = ct.astype(rot_high_f, ct.bfloat16)

    cos_low = ct.load(cos_ptr, index=(pos, 0), shape=(1, HALF_DIM))
    cos_high = ct.load(cos_ptr, index=(pos, 1), shape=(1, HALF_DIM))
    sin_low = ct.load(sin_ptr, index=(pos, 0), shape=(1, HALF_DIM))
    sin_high = ct.load(sin_ptr, index=(pos, 1), shape=(1, HALF_DIM))

    val_low_f = ct.astype(value_low_bf, ct.float32)
    val_high_f = ct.astype(value_high_bf, ct.float32)
    cos_low_f = ct.astype(cos_low, ct.float32)
    cos_high_f = ct.astype(cos_high, ct.float32)
    sin_low_f = ct.astype(sin_low, ct.float32)
    sin_high_f = ct.astype(sin_high, ct.float32)
    rot_low_bf_f = ct.astype(rot_low_bf, ct.float32)
    rot_high_bf_f = ct.astype(rot_high_bf, ct.float32)

    out_low = ct.astype(val_low_f * cos_low_f, ct.bfloat16)
    out_high = ct.astype(val_high_f * cos_high_f, ct.bfloat16)
    rot_low_a = ct.astype(rot_low_bf_f * sin_low_f, ct.bfloat16)
    rot_high_a = ct.astype(rot_high_bf_f * sin_high_f, ct.bfloat16)

    final_low_bf = ct.astype(
        ct.astype(out_low, ct.float32) + ct.astype(rot_low_a, ct.float32),
        ct.bfloat16,
    )
    final_high_bf = ct.astype(
        ct.astype(out_high, ct.float32) + ct.astype(rot_high_a, ct.float32),
        ct.bfloat16,
    )
    ct.store(key_out_ptr, index=(row, 0), tile=final_low_bf)
    ct.store(key_out_ptr, index=(row, 1), tile=final_high_bf)

    # Repeated key: for r in [0, REPEAT): out_head = kv_head*REPEAT + r
    final_low_3d = ct.reshape(final_low_bf, (1, 1, HALF_DIM))
    final_high_3d = ct.reshape(final_high_bf, (1, 1, HALF_DIM))
    for r in ct.static_iter(range(REPEAT)):
        out_head = kv_head * REPEAT + r
        ct.store(repeated_ptr, index=(out_head, pos, 0), tile=final_low_3d)
        ct.store(repeated_ptr, index=(out_head, pos, 1), tile=final_high_3d)


@oracle_impl(hardware="B200", point="2c02d9cc")
def oracle_forward(inputs):
    (
        arg0_1,   # bf16 [1000, 2048]
        arg1_1,   # bf16 [256]
        arg2_1,   # bf16 [1, 1000, 256]
        arg3_1,   # bf16 [1, 1000, 256]
        arg4_1,   # bf16 [1000, 1024]
        arg5_1,   # bf16 [256]
        *_shape_params,
    ) = inputs
    _shape_param_5 = _shape_params[-1]  # (1,8,1000,256) — repeated key shape

    seq_len = int(arg2_1.shape[1])
    batch = 1
    q_heads = int(arg0_1.shape[1]) // HEAD_DIM
    kv_heads = int(arg4_1.shape[1]) // HEAD_DIM
    repeat = q_heads // kv_heads

    arg0_2d = arg0_1.view(seq_len * q_heads, HEAD_DIM)
    arg4_2d = arg4_1.view(seq_len * kv_heads, HEAD_DIM)
    cos_2d = arg2_1.view(seq_len, HEAD_DIM)
    sin_2d = arg3_1.view(seq_len, HEAD_DIM)

    query_rope = torch.empty_strided(
        (batch, q_heads, seq_len, HEAD_DIM),
        (q_heads * seq_len * HEAD_DIM, HEAD_DIM, q_heads * HEAD_DIM, 1),
        device=arg0_1.device, dtype=torch.bfloat16,
    )
    key_rope = torch.empty_strided(
        (batch, kv_heads, seq_len, HEAD_DIM),
        (kv_heads * seq_len * HEAD_DIM, HEAD_DIM, kv_heads * HEAD_DIM, 1),
        device=arg0_1.device, dtype=torch.bfloat16,
    )
    repeated_key = torch.empty(
        tuple(int(d) for d in _shape_param_5),
        device=arg0_1.device, dtype=torch.bfloat16,
    )

    # For the strided outputs: memory layout is (seq*heads*HEAD_DIM) contiguous;
    # we can flatten to [rows, HEAD_DIM] via as_strided.
    query_rope_flat = query_rope.as_strided(
        (batch * seq_len * q_heads, HEAD_DIM), (HEAD_DIM, 1),
        storage_offset=query_rope.storage_offset(),
    )
    key_rope_flat = key_rope.as_strided(
        (batch * seq_len * kv_heads, HEAD_DIM), (HEAD_DIM, 1),
        storage_offset=key_rope.storage_offset(),
    )
    repeated_key_3d = repeated_key.view(q_heads, seq_len, HEAD_DIM)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (seq_len * q_heads, 1, 1),
        _rmsnorm_rope_kernel,
        (arg0_2d, arg1_1, cos_2d, sin_2d, query_rope_flat, q_heads),
    )
    ct.launch(
        stream, (seq_len * kv_heads, 1, 1),
        _rmsnorm_rope_key_repeat_kernel,
        (arg4_2d, arg5_1, cos_2d, sin_2d, key_rope_flat, repeated_key_3d,
         kv_heads, repeat),
    )
    return query_rope, key_rope, repeated_key
