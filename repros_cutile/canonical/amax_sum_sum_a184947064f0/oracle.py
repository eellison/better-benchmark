"""cuTile port of amax_sum_sum_a184947064f0: MoE combine + RMSNorm.

Two cuTile kernels mirror the Triton oracle:
1. Invert permutation.
2. Routed combine (in-kernel gate softmax over TOPK + routed sum) + residual
   add + RMSNorm (in-kernel square-sum + rsqrt + affine).
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


TOKENS = 1000
TOPK = 4
ROUTED_ROWS = TOKENS * TOPK
HIDDEN = 2880
EPS = 1.0e-5


@ct.kernel
def _invert_permutation_kernel(
    perm_ptr,      # i64 [N]
    inverse_ptr,   # i64 [N]
    N: ct.Constant[int],
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    offsets = pid * BLOCK + ct.arange(BLOCK, dtype=ct.int32)
    mask = offsets < N
    dest = ct.load(perm_ptr, index=(pid,), shape=(BLOCK,),
                   padding_mode=ct.PaddingMode.ZERO)
    ct.scatter(inverse_ptr, (ct.astype(dest, ct.int32),),
               ct.astype(offsets, ct.int64), mask=mask)


@ct.kernel
def _routed_combine_rmsnorm_kernel(
    expert_ptr,          # bf16 [NUM_EXPERTS, HIDDEN]
    expert_index_ptr,    # i64  [ROUTED_ROWS]  arg1_1 flattened
    payload_ptr,         # bf16 [ROUTED_ROWS, HIDDEN]  arg2_1
    gate_logits_ptr,     # bf16 [TOKENS, TOPK]  arg3_1
    inverse_ptr,         # i64  [ROUTED_ROWS] from kernel 1
    skip_mask_ptr,       # b8   [ROUTED_ROWS, 1] — treat as [ROUTED_ROWS] flat
    residual_ptr,        # bf16 [TOKENS, HIDDEN]  arg6_1 viewed
    weight_ptr,          # bf16 [HIDDEN]  arg7_1
    add_out_ptr,         # bf16 [TOKENS, HIDDEN]  routed+residual bf16
    norm_out_ptr,        # bf16 [TOKENS, HIDDEN]  RMSNormed bf16
    HIDDEN_SIZE: ct.Constant[int],
    TOPK_SIZE: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
):
    token = ct.bid(0)

    # ---- Gate softmax over TOPK columns ----
    logits = ct.astype(
        ct.load(gate_logits_ptr, index=(token, 0), shape=(1, TOPK_SIZE)),
        ct.float32,
    )
    logits_1d = ct.reshape(logits, (TOPK_SIZE,))
    row_max = ct.max(logits_1d)
    numer = ct.exp(logits_1d - row_max)
    denom = ct.sum(numer)
    probs = numer / denom
    probs_bf = ct.astype(ct.astype(probs, ct.bfloat16), ct.float32)  # [TOPK]

    # ---- Routed sum ----
    col_idx = ct.arange(BLOCK_H, dtype=ct.int32)
    col_valid = col_idx < HIDDEN_SIZE
    col_valid_1d = col_valid  # 1D
    routed_sum = ct.full((BLOCK_H,), 0.0, dtype=ct.float32)
    for slot in ct.static_iter(range(TOPK_SIZE)):
        # inverse[token * TOPK + slot] -> src index
        src_off = ct.full((1,), token * TOPK_SIZE + slot, dtype=ct.int32)
        src_tile = ct.gather(inverse_ptr, (src_off,))
        src = ct.reshape(ct.astype(src_tile, ct.int32), ())

        # expert_index[src]
        expert_id_tile = ct.gather(expert_index_ptr,
                                   (ct.reshape(src, (1,)),))
        expert_id = ct.reshape(ct.astype(expert_id_tile, ct.int32), ())
        # skip_mask[src]
        skipped_tile = ct.gather(skip_mask_ptr, (ct.reshape(src, (1,)),))
        skipped = ct.reshape(skipped_tile, ())

        # gate = probs[slot]: derive via equality mask + sum (as Triton kernel does)
        gate_cols = ct.arange(TOPK_SIZE, dtype=ct.int32)
        slot_bcast = ct.full((TOPK_SIZE,), slot, dtype=ct.int32)
        gate_val = ct.sum(
            ct.where(gate_cols == slot_bcast, probs_bf,
                     ct.full((TOPK_SIZE,), 0.0, dtype=ct.float32))
        )

        # Load payload[src, :HIDDEN]
        payload = ct.astype(
            ct.gather(payload_ptr,
                      (ct.full((BLOCK_H,), src, dtype=ct.int32) * HIDDEN_SIZE + col_idx,),
                      mask=col_valid_1d),
            ct.float32,
        )
        # Load expert[expert_id, :HIDDEN]
        expert = ct.astype(
            ct.gather(expert_ptr,
                      (ct.full((BLOCK_H,), expert_id, dtype=ct.int32) * HIDDEN_SIZE + col_idx,),
                      mask=col_valid_1d),
            ct.float32,
        )
        add_bf = ct.astype(payload + expert, ct.bfloat16)
        contrib_f = ct.astype(add_bf, ct.float32) * gate_val
        contrib_bf = ct.astype(contrib_f, ct.bfloat16)
        contrib_f = ct.astype(contrib_bf, ct.float32)
        # Zero out if skipped
        contrib_final = ct.where(skipped, ct.full((BLOCK_H,), 0.0, dtype=ct.float32),
                                 contrib_f)
        routed_sum = routed_sum + contrib_final

    routed_sum_bf = ct.astype(ct.astype(routed_sum, ct.bfloat16), ct.float32)

    # Residual add
    residual = ct.astype(
        ct.load(residual_ptr, index=(token, 0), shape=(1, BLOCK_H),
                padding_mode=ct.PaddingMode.ZERO),
        ct.float32,
    )
    residual_1d = ct.reshape(residual, (BLOCK_H,))
    add_val_f = residual_1d + routed_sum_bf
    add_val_bf = ct.astype(add_val_f, ct.bfloat16)
    # Store add_out
    ct.store(add_out_ptr, index=(token, 0),
             tile=ct.reshape(add_val_bf, (1, BLOCK_H)))

    add_val_f2 = ct.astype(add_val_bf, ct.float32)
    zeroed = ct.where(col_valid, add_val_f2 * add_val_f2,
                      ct.full((BLOCK_H,), 0.0, dtype=ct.float32))
    square_sum = ct.sum(zeroed)
    inv_rms = ct.rsqrt(square_sum * (1.0 / HIDDEN_SIZE) + EPS)
    weight = ct.astype(
        ct.load(weight_ptr, index=(0,), shape=(BLOCK_H,),
                padding_mode=ct.PaddingMode.ZERO), ct.float32,
    )
    out = add_val_f2 * inv_rms * weight
    ct.store(norm_out_ptr, index=(token, 0),
             tile=ct.reshape(ct.astype(out, ct.bfloat16), (1, BLOCK_H)))


def _contiguous_stride(shape):
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= int(dim)
    return tuple(reversed(stride))


@oracle_impl(hardware="B200", point="038e722e", BLOCK_H=4096)
def oracle_forward(inputs, *, BLOCK_H: int):
    (
        arg0_1,  # expert weights bf16 [NUM_EXPERTS, HIDDEN]
        arg1_1,  # expert index i64 [ROUTED_ROWS]
        arg2_1,  # payload bf16 [ROUTED_ROWS, HIDDEN]
        arg3_1,  # gate logits bf16 [TOKENS, TOPK]
        arg4_1,  # permutation i64 [ROUTED_ROWS]
        arg5_1,  # skip mask b8 [ROUTED_ROWS, 1]
        arg6_1,  # residual bf16 [1, TOKENS, HIDDEN]
        arg7_1,  # weight bf16 [HIDDEN]
        _shape_param_0, _shape_param_1, shape2, shape3, shape4, shape5,
    ) = inputs
    device = arg6_1.device
    add_shape = tuple(int(d) for d in shape2)
    routed_rows = TOKENS * TOPK

    add_out = torch.empty_strided(
        add_shape, _contiguous_stride(add_shape),
        device=device, dtype=torch.bfloat16,
    )
    norm = torch.empty_strided(
        add_shape, _contiguous_stride(add_shape),
        device=device, dtype=torch.bfloat16,
    )
    inverse = torch.empty((routed_rows,), device=device, dtype=torch.int64)

    BLOCK_INV = 1024
    stream = torch.cuda.current_stream()
    ct.launch(
        stream, ((routed_rows + BLOCK_INV - 1) // BLOCK_INV, 1, 1),
        _invert_permutation_kernel,
        (arg4_1, inverse, routed_rows, BLOCK_INV),
    )

    add_out_2d = add_out.view(TOKENS, HIDDEN)
    norm_2d = norm.view(TOKENS, HIDDEN)
    residual_2d = arg6_1.view(TOKENS, HIDDEN)
    payload_flat = arg2_1.view(routed_rows * HIDDEN)
    expert_flat = arg0_1.view(-1)
    skip_flat = arg5_1.view(routed_rows)

    ct.launch(
        stream, (TOKENS, 1, 1),
        _routed_combine_rmsnorm_kernel,
        (expert_flat, arg1_1, payload_flat, arg3_1, inverse,
         skip_flat, residual_2d, arg7_1,
         add_out_2d, norm_2d,
         HIDDEN, TOPK, BLOCK_H),
    )

    return (
        add_out,
        norm.view(tuple(int(d) for d in shape3)),
        norm.view(tuple(int(d) for d in shape4)),
        norm.view(tuple(int(d) for d in shape5)),
    )
