"""cuTile port of mean_mean_37961b6339c1: Qwen3 query/key RMSNorm + RoPE.

Mirrors Triton's two-kernel structure. Each kernel:
  - Loads one HEAD_DIM tile per row of (batch, seq, head).
  - Computes RMSNorm with bf16 rounding boundaries between mul-by-invrms and
    mul-by-weight (matches Triton's PTX rn.f32 semantics via bf16 round-trip).
  - Emits RoPE by loading the rotated-index tile via ct.gather.
Query kernel writes one output; key kernel writes key + REPEAT copies of same.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


HEAD_DIM = 128
HALF_DIM = 64
EPS = 1.0e-6


@ct.kernel
def _query_rmsnorm_rope_kernel(
    x_ptr,        # bf16 (rows, HEAD_DIM)
    weight_ptr,   # bf16 (HEAD_DIM,)
    cos_ptr,      # bf16 (seq, HEAD_DIM)
    sin_ptr,      # bf16 (seq, HEAD_DIM)
    out_ptr,      # bf16 (rows, HEAD_DIM) contiguous view of query_rope layout
    NUM_HEADS: ct.Constant[int],
):
    row = ct.bid(0)
    pos = row // NUM_HEADS

    x = ct.load(x_ptr, index=(row, 0), shape=(1, HEAD_DIM))
    x_f = ct.astype(x, ct.float32)
    sum_sq = ct.sum(x_f * x_f, axis=1, keepdims=True)
    inv_rms = ct.rsqrt(sum_sq * (1.0 / HEAD_DIM) + EPS)

    # Rotation indices: rot_cols[c] = c+HALF if c<HALF else c-HALF
    cols = ct.arange(HEAD_DIM, dtype=ct.int32)
    rot_cols = ct.where(cols < HALF_DIM, cols + HALF_DIM, cols - HALF_DIM)
    sign = ct.where(cols < HALF_DIM, -1.0, 1.0)
    sign_2d = ct.reshape(sign, (1, HEAD_DIM))

    # Load rotated tile and rotated weight via gather from the (row*HEAD_DIM + rot_cols) space.
    # Row-local gather: base offset row * HEAD_DIM, columns rot_cols.
    # x is (rows, HEAD_DIM) so we index (row, rot_cols).
    row_idx = ct.full(shape=(HEAD_DIM,), fill_value=row, dtype=ct.int32)
    x_rot = ct.gather(x_ptr, (row_idx, rot_cols))
    x_rot_f = ct.astype(x_rot, ct.float32)
    x_rot_f = ct.reshape(x_rot_f, (1, HEAD_DIM))

    weight = ct.load(weight_ptr, index=(0,), shape=(HEAD_DIM,))
    rot_weight = ct.gather(weight_ptr, rot_cols)
    weight_f = ct.reshape(ct.astype(weight, ct.float32), (1, HEAD_DIM))
    rot_weight_f = ct.reshape(ct.astype(rot_weight, ct.float32), (1, HEAD_DIM))

    value_bf = ct.astype(x_f * inv_rms, ct.bfloat16)
    value = ct.astype(ct.astype(value_bf, ct.float32) * weight_f, ct.bfloat16)
    rot_bf = ct.astype(x_rot_f * inv_rms, ct.bfloat16)
    rotated = ct.astype(ct.astype(rot_bf, ct.float32) * rot_weight_f, ct.bfloat16)
    rotated_signed = ct.astype(rotated, ct.float32) * sign_2d

    cos = ct.load(cos_ptr, index=(pos, 0), shape=(1, HEAD_DIM))
    sin = ct.load(sin_ptr, index=(pos, 0), shape=(1, HEAD_DIM))
    cos_f = ct.astype(cos, ct.float32)
    sin_f = ct.astype(sin, ct.float32)

    out_a = ct.astype(ct.astype(value, ct.float32) * cos_f, ct.bfloat16)
    out_b = ct.astype(rotated_signed * sin_f, ct.bfloat16)
    out = ct.astype(out_a, ct.float32) + ct.astype(out_b, ct.float32)
    ct.store(out_ptr, index=(row, 0), tile=ct.astype(out, ct.bfloat16))


@ct.kernel
def _key_rmsnorm_rope_repeat_kernel(
    x_ptr,           # bf16 (rows, HEAD_DIM)
    weight_ptr,      # bf16 (HEAD_DIM,)
    cos_ptr,         # bf16 (seq, HEAD_DIM)
    sin_ptr,         # bf16 (seq, HEAD_DIM)
    key_out_ptr,     # bf16 (rows, HEAD_DIM)
    repeat_out_ptr,  # bf16 (q_heads, seq, HEAD_DIM)  contig
    NUM_KV_HEADS: ct.Constant[int],
    REPEAT: ct.Constant[int],
):
    row = ct.bid(0)
    pos = row // NUM_KV_HEADS
    kv_head = row - pos * NUM_KV_HEADS

    x = ct.load(x_ptr, index=(row, 0), shape=(1, HEAD_DIM))
    x_f = ct.astype(x, ct.float32)
    sum_sq = ct.sum(x_f * x_f, axis=1, keepdims=True)
    inv_rms = ct.rsqrt(sum_sq * (1.0 / HEAD_DIM) + EPS)

    cols = ct.arange(HEAD_DIM, dtype=ct.int32)
    rot_cols = ct.where(cols < HALF_DIM, cols + HALF_DIM, cols - HALF_DIM)
    sign = ct.where(cols < HALF_DIM, -1.0, 1.0)
    sign_2d = ct.reshape(sign, (1, HEAD_DIM))

    row_idx = ct.full(shape=(HEAD_DIM,), fill_value=row, dtype=ct.int32)
    x_rot = ct.gather(x_ptr, (row_idx, rot_cols))
    x_rot_f = ct.reshape(ct.astype(x_rot, ct.float32), (1, HEAD_DIM))

    weight = ct.load(weight_ptr, index=(0,), shape=(HEAD_DIM,))
    rot_weight = ct.gather(weight_ptr, rot_cols)
    weight_f = ct.reshape(ct.astype(weight, ct.float32), (1, HEAD_DIM))
    rot_weight_f = ct.reshape(ct.astype(rot_weight, ct.float32), (1, HEAD_DIM))

    value_bf = ct.astype(x_f * inv_rms, ct.bfloat16)
    value = ct.astype(ct.astype(value_bf, ct.float32) * weight_f, ct.bfloat16)
    rot_bf = ct.astype(x_rot_f * inv_rms, ct.bfloat16)
    rotated = ct.astype(ct.astype(rot_bf, ct.float32) * rot_weight_f, ct.bfloat16)
    rotated_signed = ct.astype(rotated, ct.float32) * sign_2d

    cos = ct.load(cos_ptr, index=(pos, 0), shape=(1, HEAD_DIM))
    sin = ct.load(sin_ptr, index=(pos, 0), shape=(1, HEAD_DIM))
    cos_f = ct.astype(cos, ct.float32)
    sin_f = ct.astype(sin, ct.float32)

    out_a = ct.astype(ct.astype(value, ct.float32) * cos_f, ct.bfloat16)
    out_b = ct.astype(rotated_signed * sin_f, ct.bfloat16)
    final = ct.astype(ct.astype(out_a, ct.float32) + ct.astype(out_b, ct.float32),
                      ct.bfloat16)

    ct.store(key_out_ptr, index=(row, 0), tile=final)

    final_3d = ct.reshape(final, (1, 1, HEAD_DIM))
    for r in ct.static_iter(range(REPEAT)):
        out_head = kv_head * REPEAT + r
        ct.store(repeat_out_ptr, index=(out_head, pos, 0), tile=final_3d)


@oracle_impl(hardware="B200", point="f8472361")
def oracle_forward(inputs):
    (
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1,
        *_shape_params,
    ) = inputs
    _shape_param_5 = _shape_params[-1]

    seq_len = int(arg2_1.shape[1])
    batch = int(arg0_1.shape[0]) // seq_len
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
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    key_rope = torch.empty_strided(
        (batch, kv_heads, seq_len, HEAD_DIM),
        (kv_heads * seq_len * HEAD_DIM, HEAD_DIM, kv_heads * HEAD_DIM, 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    repeated_key = torch.empty(
        tuple(int(dim) for dim in _shape_param_5),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )

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
        _query_rmsnorm_rope_kernel,
        (arg0_2d, arg1_1, cos_2d, sin_2d, query_rope_flat, q_heads),
    )
    ct.launch(
        stream, (seq_len * kv_heads, 1, 1),
        _key_rmsnorm_rope_repeat_kernel,
        (arg4_2d, arg5_1, cos_2d, sin_2d, key_rope_flat, repeated_key_3d,
         kv_heads, repeat),
    )
    return query_rope, key_rope, repeated_key
