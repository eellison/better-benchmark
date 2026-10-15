"""cuTile port of pointwise_0e23e54703ef: GPT-J RoPE pair kernel.

Structure:
- Torch: build positions [1, 128] via arange (host-side gather setup).
- Torch: gather the RoPE table entries to [1, 128, 64] (row-per-seq).
- cuTile: do the pair-swap + bf16 rounding + rope combine per (seq, head).
- Torch: build the "ne" and "positions" side outputs.

cuTile's bf16 round-trip is RTNE by default, so the Triton oracle's
_round_bf16_to_fp32(x) becomes astype(astype(x, bf16), f32).
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


SEQ = 128
HEADS = 16
HEAD_DIM = 256
ROTARY_DIM = 64
ROTARY_PAIRS = ROTARY_DIM // 2


@ct.kernel
def _rope_pair_kernel(
    x0_ptr,        # bf16 [SEQ, HEADS, HEAD_DIM]  (contiguous view: [SEQ*HEADS, HEAD_DIM])
    x1_ptr,        # bf16 [SEQ*HEADS, HEAD_DIM]
    rot_coef_ptr,  # bf16 [SEQ, ROTARY_DIM]  (rot coefficient repeated per pair)
    dir_coef_ptr,  # bf16 [SEQ, ROTARY_DIM]  (direct coefficient repeated per pair)
    out0_ptr,      # f32 [SEQ*HEADS, HEAD_DIM]  contiguous flat view
    out1_ptr,      # f32 [SEQ*HEADS, HEAD_DIM]  contiguous flat view
    NUM_HEADS: ct.Constant[int],
):
    row = ct.bid(0)
    seq_idx = row // NUM_HEADS

    # Rotary segment (first ROTARY_DIM columns)
    x0_rot = ct.load(x0_ptr, index=(row, 0), shape=(1, ROTARY_DIM))
    x1_rot = ct.load(x1_ptr, index=(row, 0), shape=(1, ROTARY_DIM))
    coef_rot = ct.load(rot_coef_ptr, index=(seq_idx, 0), shape=(1, ROTARY_DIM))
    coef_dir = ct.load(dir_coef_ptr, index=(seq_idx, 0), shape=(1, ROTARY_DIM))

    x0_rot_f = ct.astype(x0_rot, ct.float32)
    x1_rot_f = ct.astype(x1_rot, ct.float32)
    coef_rot_f = ct.astype(coef_rot, ct.float32)
    coef_dir_f = ct.astype(coef_dir, ct.float32)

    # Pair-swap along the last axis: reshape (1, PAIRS, 2), swap the 2-axis, flatten.
    # Since ct.flip isn't obviously present, build the swap via masks:
    #   pair (a, b) -> swapped (b, a) = a*[0,1] + b*[1,0]
    # We can extract even (a) and odd (b) via ct.sum with a one-hot mask.
    two_idx = ct.arange(2, dtype=ct.int32)
    mask_even = ct.reshape(ct.astype(two_idx == 0, ct.float32), (1, 1, 2))
    mask_odd = ct.reshape(ct.astype(two_idx == 1, ct.float32), (1, 1, 2))

    x0_pairs = ct.reshape(x0_rot_f, (1, ROTARY_PAIRS, 2))
    a0 = ct.sum(x0_pairs * mask_even, axis=2, keepdims=True)  # (1, PAIRS, 1)
    b0 = ct.sum(x0_pairs * mask_odd, axis=2, keepdims=True)
    # swapped: pair (b, a)
    swap0_pairs = b0 * mask_even + a0 * mask_odd
    x0_pair = ct.reshape(swap0_pairs, (1, ROTARY_DIM))

    x1_pairs = ct.reshape(x1_rot_f, (1, ROTARY_PAIRS, 2))
    a1 = ct.sum(x1_pairs * mask_even, axis=2, keepdims=True)
    b1 = ct.sum(x1_pairs * mask_odd, axis=2, keepdims=True)
    swap1_pairs = b1 * mask_even + a1 * mask_odd
    x1_pair = ct.reshape(swap1_pairs, (1, ROTARY_DIM))

    # rot_sign: -1 for even cols, +1 for odd cols
    idx = ct.arange(ROTARY_DIM, dtype=ct.int32)
    idx_2d = ct.reshape(idx, (1, ROTARY_DIM))
    is_odd = (idx_2d - (idx_2d // 2) * 2) == 1
    rot_sign = ct.where(is_odd, 1.0, 0.0 - 1.0)

    y0_direct = ct.astype(ct.astype(x0_rot_f * coef_dir_f, ct.bfloat16), ct.float32)
    y0_rotate = ct.astype(ct.astype((x0_pair * rot_sign) * coef_rot_f, ct.bfloat16), ct.float32)
    y1_direct = ct.astype(ct.astype(x1_rot_f * coef_dir_f, ct.bfloat16), ct.float32)
    y1_rotate = ct.astype(ct.astype((x1_pair * rot_sign) * coef_rot_f, ct.bfloat16), ct.float32)
    y0_rotary = ct.astype(ct.astype(y0_direct + y0_rotate, ct.bfloat16), ct.float32)
    y1_rotary = ct.astype(ct.astype(y1_direct + y1_rotate, ct.bfloat16), ct.float32)

    ct.store(out0_ptr, index=(row, 0), tile=y0_rotary)
    ct.store(out1_ptr, index=(row, 0), tile=y1_rotary)

    # Tail (non-rotary): promote bf16 -> f32 as-is
    for k in ct.static_iter(range(1, HEAD_DIM // ROTARY_DIM)):
        tail0 = ct.load(x0_ptr, index=(row, k), shape=(1, ROTARY_DIM))
        tail1 = ct.load(x1_ptr, index=(row, k), shape=(1, ROTARY_DIM))
        ct.store(out0_ptr, index=(row, k), tile=ct.astype(tail0, ct.float32))
        ct.store(out1_ptr, index=(row, k), tile=ct.astype(tail1, ct.float32))


@oracle_impl(hardware="B200", point="42922299")
def oracle_forward(inputs):
    arg0_1, arg1_1, arg2_1 = inputs[:3]
    device = arg1_1.device
    NUMEL = SEQ * HEADS * HEAD_DIM

    # Build positions [1, 128]
    pos_base = torch.arange(SEQ, dtype=torch.int64, device=device)
    unsqueeze = pos_base.unsqueeze(0)

    # Torch pre-gather: emulate Repro's index_gather producing [1, 128, 64]
    # where table[batch=0, i, :] = arg0_1[i, :] (identity gather when repeat idx=arange).
    # arg0_1 is bf16[2048, 64] — actually a repeated storage; the effective gather
    # for our position range is arg0_1[:SEQ, :]. From the repro:
    #   repeat_1: "i64[1, 128, 64]" is unsqueeze(pos_base)*expand — same idx for each of 64 cols.
    #   gather from repeat[0] using repeat_1 -> table[unsqueeze[0, i], j] for j = 0..63.
    # unsqueeze holds arange(SEQ), so gather picks arg0_1[i, j].
    table = arg0_1[:SEQ, :]  # [SEQ, 64] bf16
    # Split into rotate coeffs and direct coeffs (first ROTARY_PAIRS = last two halves).
    # From Triton: coeff_rotate at seq*TABLE_DIM + pair (cols 0..31); coeff_direct at seq*TABLE_DIM + ROTARY_PAIRS + pair (cols 32..63).
    coef_rot_pairs = table[:, :ROTARY_PAIRS]     # bf16 [SEQ, 32]
    coef_dir_pairs = table[:, ROTARY_PAIRS:]     # bf16 [SEQ, 32]
    # Repeat each pair value to fill both slots (rotary_dim = 64):
    coef_rot = coef_rot_pairs.repeat_interleave(2, dim=1).contiguous()   # [SEQ, 64] bf16
    coef_dir = coef_dir_pairs.repeat_interleave(2, dim=1).contiguous()

    x0_2d = arg1_1.view(SEQ * HEADS, HEAD_DIM)
    x1_2d = arg2_1.view(SEQ * HEADS, HEAD_DIM)

    out0_base = torch.empty_strided(
        (1, HEADS, SEQ, HEAD_DIM),
        (NUMEL, HEAD_DIM, HEADS * HEAD_DIM, 1),
        device=device, dtype=torch.float32,
    )
    out1_base = torch.empty_strided(
        (1, HEADS, HEAD_DIM, SEQ),
        (NUMEL, HEAD_DIM, 1, HEADS * HEAD_DIM),
        device=device, dtype=torch.float32,
    )
    view_5 = out0_base.as_strided((HEADS, SEQ, HEAD_DIM), (HEAD_DIM, HEADS * HEAD_DIM, 1))
    view_11 = out1_base.as_strided((HEADS, HEAD_DIM, SEQ), (HEAD_DIM, 1, HEADS * HEAD_DIM))
    ne = torch.zeros((1, SEQ), device=device, dtype=torch.bool)

    # Flat views (see comments in previous version — both are contiguous under storage flat).
    out0_flat = out0_base.as_strided(
        (SEQ * HEADS, HEAD_DIM), (HEAD_DIM, 1),
        storage_offset=out0_base.storage_offset(),
    )
    out1_flat = out1_base.as_strided(
        (SEQ * HEADS, HEAD_DIM), (HEAD_DIM, 1),
        storage_offset=out1_base.storage_offset(),
    )

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (SEQ * HEADS, 1, 1), _rope_pair_kernel,
        (x0_2d, x1_2d, coef_rot, coef_dir, out0_flat, out1_flat, HEADS),
    )
    return unsqueeze, view_5, view_11, ne
