"""cuTile port of amax_amax_any_f8e425e72787: Visformer safe scaled softmax + constant_pad.

Input: bf16[768, 200, 200], sliced to bf16[768, 196, 196] then viewed as
[128, 6, 196, 196]. Emits raw_amax (f32), scaled_amax (f32), all_finite (bool),
denom (f32), padded bf16[768, 200, 200], and permute of compact bf16[768, 196, 196].
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


N_HEADS = 768        # = 128 * 6
K_LEN = 196          # non-power-of-2
BLOCK_N = 256        # next pow2 of K_LEN
SCALE = 0.125


@ct.kernel
def _visformer_safe_scaled_softmax_pad_kernel(
    x_ptr,             # bf16 [N_HEADS, 200, 200] (arg0)
    amax_ptr,          # f32 flat [N_HEADS*K_LEN]
    scaled_amax_ptr,   # f32 flat [N_HEADS*K_LEN]
    all_finite_ptr,    # bool flat [N_HEADS*K_LEN]
    denom_ptr,         # f32 flat [N_HEADS*K_LEN]
    compact_flat_ptr,  # bf16 flat [N_HEADS*K_LEN*K_LEN]
    padded_flat_ptr,   # bf16 flat [N_HEADS*200*200]
    BLOCK_N_: ct.Constant[int],
    K_LEN_: ct.Constant[int],
    SCALE_: ct.Constant[float],
    PADDED_W: ct.Constant[int],
):
    flat_head = ct.bid(0)
    query = ct.bid(1)

    x = ct.load(
        x_ptr, index=(flat_head, query, 0), shape=(1, 1, BLOCK_N_),
        padding_mode=ct.PaddingMode.ZERO,
    )
    x_1d = ct.reshape(x, (BLOCK_N_,))
    raw = ct.astype(x_1d, ct.float32)

    cols = ct.arange(BLOCK_N_, dtype=ct.int32)
    col_mask = cols < K_LEN_
    neg_inf = ct.full((BLOCK_N_,), float("-inf"), dtype=ct.float32)
    raw_masked = ct.where(col_mask, raw, neg_inf)

    scaled_masked = raw_masked * SCALE_

    raw_max = ct.max(raw_masked)
    scaled_max = ct.max(scaled_masked)

    abs_val = abs(scaled_masked)
    inf_tile = ct.full((BLOCK_N_,), float("inf"), dtype=ct.float32)
    is_finite = (scaled_masked == scaled_masked) & (abs_val != inf_tile)
    zero_i32 = ct.zeros((BLOCK_N_,), dtype=ct.int32)
    one_i32 = ct.full((BLOCK_N_,), 1, dtype=ct.int32)
    invalid_flag = ct.where(col_mask & (~is_finite), one_i32, zero_i32)
    any_invalid = ct.max(invalid_flag) != 0
    all_finite = ~any_invalid

    shifted_unscaled = (raw_masked - raw_max) * SCALE_
    shifted_scaled = scaled_masked - scaled_max
    shifted = ct.where(all_finite, shifted_unscaled, shifted_scaled)

    numer = ct.exp(shifted)
    zero_f = ct.zeros((BLOCK_N_,), dtype=ct.float32)
    numer_masked = ct.where(col_mask, numer, zero_f)
    denom = ct.sum(numer_masked)
    probs = numer_masked / denom

    row_flat = flat_head * K_LEN_ + query
    ct.store(amax_ptr, index=(row_flat,), tile=ct.reshape(raw_max, (1,)))
    ct.store(scaled_amax_ptr, index=(row_flat,), tile=ct.reshape(scaled_max, (1,)))
    ct.store(all_finite_ptr, index=(row_flat,), tile=ct.reshape(all_finite, (1,)))
    ct.store(denom_ptr, index=(row_flat,), tile=ct.reshape(denom, (1,)))

    probs_bf = ct.astype(probs, ct.bfloat16)

    # Store to compact bf16[N_HEADS*K_LEN*K_LEN] flat.
    compact_off = row_flat * K_LEN_ + cols
    ct.scatter(compact_flat_ptr, compact_off, probs_bf, mask=col_mask)

    # Store to padded bf16[N_HEADS*PADDED_W*PADDED_W] flat.
    padded_off = flat_head * (PADDED_W * PADDED_W) + query * PADDED_W + cols
    ct.scatter(padded_flat_ptr, padded_off, probs_bf, mask=col_mask)


@oracle_impl(hardware="B200", point="df5bbfd4", BLOCK_M=8, BLOCK_N=256)
def oracle_forward(inputs, *, BLOCK_M, BLOCK_N):
    del BLOCK_M
    arg0_1, _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3 = inputs
    device = arg0_1.device

    padded_shape = tuple(int(dim) for dim in arg0_1.shape)  # (768, 200, 200)
    padded_h = padded_shape[1]
    padded_w = padded_shape[2]
    n_heads = padded_shape[0]

    view_shape = tuple(int(dim) for dim in _shape_param_0)  # (128, 6, 196, 196)
    compact_shape = tuple(int(dim) for dim in _shape_param_2)  # (768, 196, 196)
    q_len = view_shape[-2]
    k_len = view_shape[-1]
    assert q_len == K_LEN and k_len == K_LEN

    reduction_shape = (view_shape[0], view_shape[1], view_shape[2], 1)
    reduction_stride = (view_shape[1] * view_shape[2], view_shape[2], 1, 1)
    raw_amax = torch.empty_strided(reduction_shape, reduction_stride, device=device, dtype=torch.float32)
    scaled_amax = torch.empty_strided(reduction_shape, reduction_stride, device=device, dtype=torch.float32)
    all_finite = torch.empty_strided(reduction_shape, reduction_stride, device=device, dtype=torch.bool)
    denom = torch.empty_strided(reduction_shape, reduction_stride, device=device, dtype=torch.float32)
    compact = torch.empty_strided(
        compact_shape,
        (compact_shape[1] * compact_shape[2], compact_shape[2], 1),
        device=device, dtype=torch.bfloat16,
    )
    padded = torch.zeros(padded_shape, device=device, dtype=torch.bfloat16)

    amax_flat = raw_amax.view(-1)
    scaled_amax_flat = scaled_amax.view(-1)
    all_finite_flat = all_finite.view(-1)
    denom_flat = denom.view(-1)
    compact_flat = compact.view(-1)
    padded_flat = padded.view(-1)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (n_heads, K_LEN, 1),
        _visformer_safe_scaled_softmax_pad_kernel,
        (
            arg0_1, amax_flat, scaled_amax_flat, all_finite_flat, denom_flat,
            compact_flat, padded_flat, BLOCK_N, K_LEN, SCALE, padded_w,
        ),
    )
    return raw_amax, scaled_amax, all_finite, denom, padded, compact.permute(0, 2, 1)
