"""cuTile port of pointwise_5c9e535343f4: GPT-J RoPE (rotate-half interleaved).

For each rotary-dim position: rotate-half + direct products with two coefficient
sets. Non-rotary positions pass through. Emits zero64/zero256 side outputs,
and two 4D output tensors plus their permutes.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


SEQ = 128
HEADS = 16
HEAD_DIM = 256
ROTARY_DIM = 64
HIDDEN = HEADS * HEAD_DIM  # 4096
NUMEL = SEQ * HIDDEN  # 524288


@ct.kernel
def _gptj_rope_kernel(
    arg0_ptr,           # bf16 (16, 256, 128) — [heads, head_dim, seq] flat
    arg1_ptr,           # bf16 (16, 128, 256) — [heads, seq, head_dim] flat
    rotate_coeff_ptr,   # bf16 (SEQ, ROTARY_DIM/2) — cos/sin coeffs (paired)
    direct_coeff_ptr,   # bf16 (SEQ, ROTARY_DIM/2)
    zero64_ptr,         # bf16 (SEQ*HEADS*ROTARY_DIM,) - all zeros
    zero256_ptr,        # bf16 (SEQ*HEADS*HEAD_DIM,) - all zeros
    out0_ptr,           # bf16 (SEQ, HIDDEN)
    out1_ptr,           # bf16 (SEQ, HIDDEN)
    NUMEL_C: ct.Constant[int],
    ZERO64_NUMEL_C: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    pid = ct.bid(0)
    offs = pid * BLOCK_C + ct.arange(BLOCK_C, dtype=ct.int32)
    valid = offs < NUMEL_C

    dim = offs - (offs // HEAD_DIM) * HEAD_DIM
    row = offs // HEAD_DIM
    seq = row // HEADS
    head = row - seq * HEADS

    is_rotary = dim < ROTARY_DIM
    rotary_valid = valid & is_rotary
    pair = dim // 2
    odd = (dim - (pair * 2)) == 1
    paired_dim = ct.where(odd, dim - 1, dim + 1)

    coeff_offs = seq * (ROTARY_DIM // 2) + pair
    rotate_coeff = ct.gather(rotate_coeff_ptr, coeff_offs, mask=rotary_valid,
                             padding_value=0)
    direct_coeff = ct.gather(direct_coeff_ptr, coeff_offs, mask=rotary_valid,
                             padding_value=0)
    rotate_f = ct.astype(rotate_coeff, ct.float32)
    direct_f = ct.astype(direct_coeff, ct.float32)

    # arg0 flat layout: (16 heads, 256 head_dim, 128 seq) — so index is:
    # head * (256*128) + dim * 128 + seq
    arg0_offs = head * 32768 + dim * SEQ + seq
    arg0_pair_offs = head * 32768 + paired_dim * SEQ + seq
    # arg1 flat layout: (16 heads, 128 seq, 256 head_dim) — index:
    # head * (128*256) + seq * 256 + dim
    arg1_offs = head * 32768 + seq * HEAD_DIM + dim
    arg1_pair_offs = head * 32768 + seq * HEAD_DIM + paired_dim

    x0 = ct.gather(arg0_ptr, arg0_offs, mask=valid, padding_value=0)
    x0_pair = ct.gather(arg0_ptr, arg0_pair_offs, mask=rotary_valid, padding_value=0)
    x1 = ct.gather(arg1_ptr, arg1_offs, mask=valid, padding_value=0)
    x1_pair = ct.gather(arg1_ptr, arg1_pair_offs, mask=rotary_valid, padding_value=0)

    x0_f = ct.astype(x0, ct.float32)
    x1_f = ct.astype(x1, ct.float32)
    x0_pair_f = ct.astype(x0_pair, ct.float32)
    x1_pair_f = ct.astype(x1_pair, ct.float32)

    # Rotate products via round-to-nearest bf16 boundary.
    x0_rot_prod = ct.astype(ct.astype(x0_pair_f * rotate_f, ct.bfloat16), ct.float32)
    x1_rot_prod = ct.astype(ct.astype(x1_pair_f * rotate_f, ct.bfloat16), ct.float32)
    x0_rot = ct.where(odd, -x0_rot_prod, x0_rot_prod)
    x1_rot = ct.where(odd, -x1_rot_prod, x1_rot_prod)
    x0_dir = ct.astype(ct.astype(x0_f * direct_f, ct.bfloat16), ct.float32)
    x1_dir = ct.astype(ct.astype(x1_f * direct_f, ct.bfloat16), ct.float32)
    y0_rotary = ct.astype(ct.astype(x0_rot + x0_dir, ct.bfloat16), ct.float32)
    y1_rotary = ct.astype(ct.astype(x1_rot + x1_dir, ct.bfloat16), ct.float32)

    y0 = ct.astype(ct.where(is_rotary, y0_rotary, x0_f), ct.bfloat16)
    y1 = ct.astype(ct.where(is_rotary, y1_rotary, x1_f), ct.bfloat16)

    ct.scatter(out0_ptr, offs, y0, mask=valid)
    ct.scatter(out1_ptr, offs, y1, mask=valid)

    zero_bf = ct.zeros((BLOCK_C,), dtype=ct.bfloat16)
    ct.scatter(zero256_ptr, offs, zero_bf, mask=valid)
    zero64_valid = valid & (offs < ZERO64_NUMEL_C)
    ct.scatter(zero64_ptr, offs, zero_bf, mask=zero64_valid)


@oracle_impl(hardware="B200", point="40872e86")
def oracle_forward(inputs):
    arg0_1, arg1_1, arg2_1, arg3_1 = inputs[:4]
    device = arg0_1.device

    zero64 = torch.empty_strided(
        (1, SEQ, HEADS, ROTARY_DIM),
        (SEQ * HEADS * ROTARY_DIM, HEADS * ROTARY_DIM, ROTARY_DIM, 1),
        device=device, dtype=torch.bfloat16,
    )
    zero256 = torch.empty_strided(
        (1, SEQ, HEADS, HEAD_DIM),
        (NUMEL, HIDDEN, HEAD_DIM, 1),
        device=device, dtype=torch.bfloat16,
    )
    out0 = torch.empty_strided((SEQ, HIDDEN), (HIDDEN, 1),
                               device=device, dtype=torch.bfloat16)
    out1 = torch.empty_strided((SEQ, HIDDEN), (HIDDEN, 1),
                               device=device, dtype=torch.bfloat16)

    # arg2/arg3 are bf16 [1, 128, 1, 32, 1] — but each has 32 lanes.
    # They're expanded to [1, 128, 1, 32, 2] via broadcast expansion.
    # After expand+clone+view to [1, 128, 1, 64], we get 64 coefficients per seq.
    # Since the last dim is 1 and expanded to 2, each pair (2i, 2i+1) is identical.
    # So the effective per-pair (32) coefficient value is arg2_1[0, seq, 0, pair, 0].
    # Reshape arg2_1 to [SEQ, 32] and arg3_1 similarly.
    rotate_coeff = arg2_1.reshape(SEQ, ROTARY_DIM // 2).contiguous()
    direct_coeff = arg3_1.reshape(SEQ, ROTARY_DIM // 2).contiguous()

    arg0_flat = arg0_1.reshape(-1)
    arg1_flat = arg1_1.reshape(-1)
    rotate_flat = rotate_coeff.reshape(-1)
    direct_flat = direct_coeff.reshape(-1)
    zero64_flat = zero64.reshape(-1)
    zero256_flat = zero256.reshape(-1)
    out0_flat = out0.reshape(-1)
    out1_flat = out1.reshape(-1)

    BLOCK = 512
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(NUMEL, BLOCK), 1, 1),
        _gptj_rope_kernel,
        (
            arg0_flat, arg1_flat, rotate_flat, direct_flat,
            zero64_flat, zero256_flat, out0_flat, out1_flat,
            NUMEL, SEQ * HEADS * ROTARY_DIM, BLOCK,
        ),
    )
    return zero64, zero256, out0, out0.permute(1, 0), out1, out1.permute(1, 0)
