"""cuTile port of any_amax_amax_af50781fc699: XLNet relative-shift softmax.

For each row r in [0, 256*512): compute (content[r, :] + rel[shifted]),
apply softmax with finite-guard branching between two normalizations.

Row -> group = row // 512, query = row % 512.
rel_offset(row, col) = group * 524288 + 512 + query * 1023 + col.
Output: bf16[256, 512, 512].
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


N_HEAD_GROUPS = 256   # 16 * 16
K_LEN = 512
REL_WIDTH = 1023
REL_STRIDE = 512 * 1024  # 524288
SCALE = 0.125
BLOCK_N = K_LEN  # 512


@ct.kernel
def _xlnet_relshift_softmax_kernel(
    content_flat_ptr,   # bf16 (N_HEAD_GROUPS * K_LEN * K_LEN,)
    rel_flat_ptr,       # bf16 (256 * 512 * 1024,)
    out_flat_ptr,       # bf16 (N_HEAD_GROUPS * K_LEN * K_LEN,)
    BLOCK_N_: ct.Constant[int],
    K_LEN_: ct.Constant[int],
):
    row = ct.bid(0)
    group = row // K_LEN_
    query = row - group * K_LEN_

    cols = ct.arange(BLOCK_N_, dtype=ct.int32)
    content_offsets = row * K_LEN_ + cols
    rel_offsets = group * REL_STRIDE + K_LEN_ + query * REL_WIDTH + cols

    content = ct.gather(content_flat_ptr, content_offsets)
    rel = ct.gather(rel_flat_ptr, rel_offsets)
    content_f = ct.astype(content, ct.float32)
    rel_f = ct.astype(rel, ct.float32)

    added_bf = ct.astype(content_f + rel_f, ct.bfloat16)
    unscaled = ct.astype(added_bf, ct.float32)
    scaled_bf = ct.astype(unscaled * SCALE, ct.bfloat16)
    scaled = ct.astype(scaled_bf, ct.float32)

    # Finite check
    abs_scaled = ct.astype(scaled, ct.float32)
    abs_val = abs(abs_scaled)
    inf_tile = ct.full((BLOCK_N_,), float("inf"), dtype=ct.float32)
    is_finite = (scaled == scaled) & (abs_val != inf_tile)
    invalid_flag = ct.where(~is_finite, ct.full((BLOCK_N_,), 1, dtype=ct.int32),
                            ct.zeros((BLOCK_N_,), dtype=ct.int32))
    any_invalid = ct.max(invalid_flag) != 0
    has_invalid = any_invalid

    unscaled_max = ct.max(unscaled)
    scaled_max = ct.max(scaled)
    shifted_unscaled = (unscaled - unscaled_max) * SCALE
    shifted_scaled = scaled - scaled_max
    shifted = ct.where(has_invalid, shifted_scaled, shifted_unscaled)

    numer = ct.exp(shifted)
    denom = ct.sum(numer)
    probs = numer / denom

    ct.scatter(out_flat_ptr, content_offsets, ct.astype(probs, ct.bfloat16))


@oracle_impl(hardware="B200", point="99deda8b")
def oracle_forward(inputs):
    content, rel, _s0, _s1, _s2, _s3, _s4, _s5, out_shape = inputs
    out_shape = tuple(int(dim) for dim in out_shape)
    out = torch.empty_strided(
        out_shape,
        (out_shape[1] * out_shape[2], out_shape[2], 1),
        device=content.device, dtype=torch.bfloat16,
    )

    content_flat = content.reshape(-1)
    rel_flat = rel.reshape(-1)
    out_flat = out.reshape(-1)
    n_rows = out_shape[0] * out_shape[1]

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (n_rows, 1, 1),
        _xlnet_relshift_softmax_kernel,
        (content_flat, rel_flat, out_flat, BLOCK_N, K_LEN),
    )
    return out
