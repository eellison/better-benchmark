"""cuTile port of amax_sum_sum_4053b14a3062: GPT-OSS MoE combine + RMSNorm.

Permutation inverse and per-slot expert gather are done in torch outside the
kernel; a cuTile kernel does the routed sum reduction over TOPK slots, bf16
rounding at each step, residual add, RMS normalization with eps=1e-5, and
final bf16 affine output.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


TOKENS = 1000
TOPK = 4
ROUTED_ROWS = TOKENS * TOPK  # 4000
HIDDEN = 2880  # non-power-of-2; we tile with 4096 and mask


@ct.kernel
def _routed_combine_rmsnorm_kernel(
    payload_ptr,       # bf16 [ROUTED_ROWS, HIDDEN] - inverse-permuted, already added w/ expert and gate
    skip_mask_ptr,     # b8   [ROUTED_ROWS]  (permuted)
    residual_ptr,      # bf16 [TOKENS, HIDDEN]
    weight_ptr,        # bf16 [HIDDEN]
    norm_out_ptr,      # bf16 [TOKENS, HIDDEN]
    HIDDEN_SIZE: ct.Constant[int],
    TOPK_SIZE: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
):
    token = ct.bid(0)
    cols = ct.arange(BLOCK_H, dtype=ct.int32)
    col_mask_1d = cols < HIDDEN_SIZE
    col_mask = ct.reshape(col_mask_1d, (1, BLOCK_H))
    zero_f = ct.full((1, BLOCK_H), 0.0, dtype=ct.float32)

    # Accumulate routed sum in fp32
    routed_sum = ct.zeros((1, BLOCK_H), dtype=ct.float32)
    for slot in ct.static_iter(range(TOPK_SIZE)):
        src_row = token * TOPK_SIZE + slot
        # skipped is the permuted skip_mask
        skipped_b = ct.load(skip_mask_ptr, index=(src_row,), shape=(1,))
        skipped_2d = ct.reshape(skipped_b, (1, 1))

        payload_bf = ct.load(payload_ptr, index=(src_row, 0), shape=(1, BLOCK_H),
                             padding_mode=ct.PaddingMode.ZERO)
        payload_f = ct.astype(payload_bf, ct.float32)
        payload_f = ct.where(col_mask, payload_f, zero_f)
        # Bf16 rounding
        payload_bf_rd = ct.astype(payload_f, ct.bfloat16)
        contrib = ct.astype(payload_bf_rd, ct.float32)
        # Apply skip
        zero_bc = ct.full((1, BLOCK_H), 0.0, dtype=ct.float32)
        skip_bc = ct.where(skipped_2d, zero_bc, contrib)
        routed_sum = routed_sum + skip_bc

    # bf16 round the routed sum
    routed_sum_bf = ct.astype(routed_sum, ct.bfloat16)
    residual_bf = ct.load(residual_ptr, index=(token, 0), shape=(1, BLOCK_H),
                          padding_mode=ct.PaddingMode.ZERO)
    add_f = ct.astype(routed_sum_bf, ct.float32) + ct.astype(residual_bf, ct.float32)
    add_bf = ct.astype(add_f, ct.bfloat16)
    add_out = ct.astype(add_bf, ct.float32)

    # RMS
    sq = ct.where(col_mask, add_out * add_out, zero_f)
    sq_sum = ct.sum(sq, axis=1, keepdims=True)
    inv_rms = ct.rsqrt(sq_sum * (1.0 / HIDDEN_SIZE) + 1.0e-5)
    weight_bf = ct.load(weight_ptr, index=(0,), shape=(BLOCK_H,),
                        padding_mode=ct.PaddingMode.ZERO)
    weight_f = ct.reshape(ct.astype(weight_bf, ct.float32), (1, BLOCK_H))
    out_f = add_out * inv_rms * weight_f
    out_bf = ct.astype(out_f, ct.bfloat16)

    # Masked store: only write [token, cols<HIDDEN_SIZE]
    # Since HIDDEN=2880 is not a power of 2, we need to write to a padded output
    # and slice, OR the caller can pass HIDDEN-sized output. Here we do the
    # simplest: pad output to BLOCK_H and slice in caller.
    ct.store(norm_out_ptr, index=(token, 0), tile=out_bf)


# 038e722e: GPT-OSS MoE combine + RMSNorm, hidden=2880, tokens=1000, topk=4.
@oracle_impl(hardware="B200", point="038e722e", BLOCK_H=4096)
def oracle_forward(inputs, *, BLOCK_H: int):
    (
        arg0_1,   # bf16 [32, 2880] expert weights
        arg1_1,   # i64  [4000] expert ids (permuted)
        arg2_1,   # bf16 [4000, 2880] payload (permuted)
        arg3_1,   # bf16 [1000, 4] gate logits
        arg4_1,   # i64  [4000] permutation
        arg5_1,   # b8   [4000, 1] skip mask (permuted)
        arg6_1,   # bf16 [1, 1000, 2880] residual
        arg7_1,   # bf16 [2880] rms weight
        _sp0, _sp1, _sp2, _sp3,
    ) = inputs
    device = arg2_1.device

    # Step 1: index gather - expert[arg1_1]
    index_expert = arg0_1[arg1_1]  # bf16 [4000, 2880]
    add = arg2_1 + index_expert   # bf16 [4000, 2880]

    # Step 2: softmax gate over [1000, 4]
    gate_f = arg3_1.float()
    amax = gate_f.amax(dim=1, keepdim=True)
    exp = torch.exp(gate_f - amax)
    denom = exp.sum(dim=1, keepdim=True)
    div = exp / denom
    probs_bf = div.bfloat16()   # [1000, 4]

    # Step 3: view [4000], gather via arg4_1 (permutation): gate[permutation[i]]
    gate_flat = probs_bf.view(-1)     # [4000]
    gate_perm = gate_flat[arg4_1]     # [4000]
    unsqueeze = gate_perm.unsqueeze(-1)  # [4000, 1]

    # Step 4: mul
    mul = add * unsqueeze             # bf16 [4000, 2880]

    # Step 5: where(skip_mask, 0, mul)
    where = torch.where(arg5_1, torch.tensor(0.0, dtype=torch.bfloat16, device=device), mul)

    # Step 6: build inverse permutation via index_put
    empty = torch.empty(ROUTED_ROWS, dtype=torch.int64, device=device)
    iota = torch.arange(ROUTED_ROWS, dtype=torch.int64, device=device)
    inverse = empty.index_put_((arg4_1,), iota)  # [4000] such that inverse[arg4_1[i]] = i

    # Step 7: gather where by inverse
    routed = where[inverse]  # bf16 [4000, 2880]

    # Now the cuTile kernel does:
    # for each token in 0..999:
    #   accum = sum(bf16_round(routed[token*4+slot, :]))  (with skip_mask handling — but skip is
    #     already applied via `where`, so we just sum)
    # actually the skip_mask is already applied in `where`, so we don't need it in kernel.
    # Then residual add, RMSNorm.
    # Pad payload to BLOCK_H width
    padded_payload = torch.zeros((ROUTED_ROWS, BLOCK_H), device=device, dtype=torch.bfloat16)
    padded_payload[:, :HIDDEN].copy_(routed)
    padded_residual = torch.zeros((TOKENS, BLOCK_H), device=device, dtype=torch.bfloat16)
    padded_residual[:, :HIDDEN].copy_(arg6_1.contiguous().view(TOKENS, HIDDEN))
    padded_weight = torch.zeros(BLOCK_H, device=device, dtype=torch.bfloat16)
    padded_weight[:HIDDEN].copy_(arg7_1)
    # skip_mask is not used since we already applied it in torch
    # Reuse arg5_1 as a "false" mask (all zeros) — or make our own
    zero_skip = torch.zeros(ROUTED_ROWS, device=device, dtype=torch.bool)

    padded_out = torch.zeros((TOKENS, BLOCK_H), device=device, dtype=torch.bfloat16)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (TOKENS, 1, 1),
        _routed_combine_rmsnorm_kernel,
        (padded_payload, zero_skip, padded_residual, padded_weight, padded_out,
         HIDDEN, TOPK, BLOCK_H),
    )

    # Slice off the padded tail
    out = padded_out[:, :HIDDEN].contiguous()
    # Return with expected strides
    view_3_shape = tuple(int(d) for d in _sp3)
    view_3 = torch.empty_strided(
        view_3_shape, (HIDDEN, 1), device=device, dtype=torch.bfloat16)
    view_3.copy_(out.view(view_3_shape))
    return view_3
