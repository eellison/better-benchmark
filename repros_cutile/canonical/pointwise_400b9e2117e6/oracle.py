"""cuTile port of pointwise_400b9e2117e6: Gemma-2-2B rotary embedding + GQA expand.

Applies RoPE to Q (8 heads, dim 256) and K (4 heads), then expands K by
group-query-attention factor of 2 to reach 8 heads. Uses cuTile for the
per-row-pair RoPE + concat kernel.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _rope_kernel(
    x_ptr,           # bf16 [rows, D]
    cos_ptr,         # bf16 [rows, D]  (broadcast: same for all heads)
    sin_ptr,         # bf16 [rows, D]
    out_ptr,         # bf16 [rows, D]
    HALF_D: ct.Constant[int],
    D: ct.Constant[int],
):
    row = ct.bid(0)
    x = ct.load(x_ptr, index=(row, 0), shape=(1, D))
    cos = ct.load(cos_ptr, index=(row, 0), shape=(1, D))
    sin = ct.load(sin_ptr, index=(row, 0), shape=(1, D))

    # Compute rotate_half(x): [-x[D/2:], x[:D/2]]
    # Load x, split into two halves.
    x_low = ct.load(x_ptr, index=(row, 0), shape=(1, HALF_D))
    x_high = ct.load(x_ptr, index=(row, 1), shape=(1, HALF_D))
    # rotate = concat([-x_high, x_low], dim=-1)
    # Store rotate in temporary — but we need to build it as one tile.
    # We can construct via ct.where masking:
    # For position < HALF_D: use -x_high[i]
    # For position >= HALF_D: use x_low[i - HALF_D]
    # Alternative: do two separate stores.
    # Simplest: compute the two halves separately.

    # Left half of output tile (positions 0..HALF_D):
    # out_low = x_low * cos_low + (-x_high) * sin_low
    cos_low = ct.load(cos_ptr, index=(row, 0), shape=(1, HALF_D))
    sin_low = ct.load(sin_ptr, index=(row, 0), shape=(1, HALF_D))
    cos_high = ct.load(cos_ptr, index=(row, 1), shape=(1, HALF_D))
    sin_high = ct.load(sin_ptr, index=(row, 1), shape=(1, HALF_D))

    x_low_f = ct.astype(x_low, ct.float32)
    x_high_f = ct.astype(x_high, ct.float32)
    cos_low_f = ct.astype(cos_low, ct.float32)
    sin_low_f = ct.astype(sin_low, ct.float32)
    cos_high_f = ct.astype(cos_high, ct.float32)
    sin_high_f = ct.astype(sin_high, ct.float32)

    # x_bf16 * cos_bf16 rounded to bf16, plus rotate_bf16 * sin_bf16.
    # In original: mul = x * cos (bf16), then cat_val = rotate * sin (bf16), then add.
    mul_low_bf = ct.astype(x_low_f * cos_low_f, ct.bfloat16)
    mul_high_bf = ct.astype(x_high_f * cos_high_f, ct.bfloat16)

    # rotate_half's low half is -x_high, high half is x_low.
    rotate_low_bf = ct.astype(-x_high_f, ct.bfloat16)
    rotate_high_bf = ct.astype(x_low_f, ct.bfloat16)

    mul1_low_bf = ct.astype(ct.astype(rotate_low_bf, ct.float32) * sin_low_f, ct.bfloat16)
    mul1_high_bf = ct.astype(ct.astype(rotate_high_bf, ct.float32) * sin_high_f, ct.bfloat16)

    add_low_bf = ct.astype(ct.astype(mul_low_bf, ct.float32) + ct.astype(mul1_low_bf, ct.float32), ct.bfloat16)
    add_high_bf = ct.astype(ct.astype(mul_high_bf, ct.float32) + ct.astype(mul1_high_bf, ct.float32), ct.bfloat16)

    ct.store(out_ptr, index=(row, 0), tile=add_low_bf)
    ct.store(out_ptr, index=(row, 1), tile=add_high_bf)


def _shape_tuple(shape):
    return tuple(int(dim) for dim in shape)


@oracle_impl(hardware="B200", point="9e133e29")
def oracle_forward(inputs):
    arg0_1, arg1_1, arg2_1, arg3_1, *_shape_params = inputs
    device = arg0_1.device
    # arg0_1: bf16 [1000, 2048] -> [1, 1000, 8, 256] -> [1, 8, 1000, 256]
    # arg3_1: bf16 [1000, 1024] -> [1, 1000, 4, 256] -> [1, 4, 1000, 256]
    # arg1_1: bf16 [1, 1000, 256] cos
    # arg2_1: bf16 [1, 1000, 256] sin
    seq = 1000
    d = 256
    half_d = 128

    # Q shape: [1, 8, 1000, 256] (permuted from [1, 1000, 8, 256]).
    q_reshaped = arg0_1.view(1, seq, 8, d).permute(0, 2, 1, 3).contiguous()
    # K shape: [1, 4, 1000, 256].
    k_reshaped = arg3_1.view(1, seq, 4, d).permute(0, 2, 1, 3).contiguous()

    # Flatten to (heads * seq, d).
    q_flat = q_reshaped.view(8 * seq, d)
    k_flat = k_reshaped.view(4 * seq, d)

    # cos/sin: [1, 1, 1000, 256] broadcast across heads. Repeat per head.
    cos_1d = arg1_1.view(seq, d)
    sin_1d = arg2_1.view(seq, d)
    # For Q: repeat per 8 heads -> [8*seq, 256] where each 1000-block uses same cos/sin.
    # Cheaper: broadcast at load time by using ct.load with stride 0? Since cuTile
    # respects strides, we can create a stride-0 view.
    cos_q = cos_1d.unsqueeze(0).expand(8, seq, d).contiguous().view(8 * seq, d)
    sin_q = sin_1d.unsqueeze(0).expand(8, seq, d).contiguous().view(8 * seq, d)
    cos_k = cos_1d.unsqueeze(0).expand(4, seq, d).contiguous().view(4 * seq, d)
    sin_k = sin_1d.unsqueeze(0).expand(4, seq, d).contiguous().view(4 * seq, d)

    q_out = torch.empty_like(q_flat)
    k_out = torch.empty_like(k_flat)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (8 * seq, 1, 1), _rope_kernel,
        (q_flat, cos_q, sin_q, q_out, half_d, half_d),
    )
    ct.launch(
        stream, (4 * seq, 1, 1), _rope_kernel,
        (k_flat, cos_k, sin_k, k_out, half_d, half_d),
    )

    # Reshape outputs.
    q_reshaped_out = q_out.view(1, 8, seq, d)
    k_reshaped_out = k_out.view(1, 4, seq, d)

    # Expand K: unsqueeze(2) -> [1, 4, 1, 1000, 256] expand -> [1, 4, 2, 1000, 256]
    # then clone contiguous, then view [1, 8, 1000, 256].
    k_unsqueezed = k_reshaped_out.unsqueeze(2)
    k_expanded = k_unsqueezed.expand(1, 4, 2, seq, d).contiguous()
    k_view = k_expanded.view(1, 8, seq, d)

    # slice_5 = add_1[:, -4095:, :, :] on dim 2 -> since add_1.shape[2]=1000, this is all rows
    # But slice dim is 2 with negative start -4095 -> effectively index 0. So it's the full tensor.
    # torch: slice.Tensor(add_1, 2, -4095, 9223372036854775807) — dim=2 has size 4? Wait no, `add_1` has shape [1, 4, 1000, 256]. On dim 2 which has 1000 elements, start=-4095 -> 0 (clamped since negative and abs>size).
    slice_5 = k_reshaped_out  # equivalent to no slice

    return q_reshaped_out, k_view, slice_5
