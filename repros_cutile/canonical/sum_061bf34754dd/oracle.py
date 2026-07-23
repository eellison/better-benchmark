"""cuTile port of sum_061bf34754dd: Demucs bf16 gated cat + row sum.

Single kernel matching Triton's `_direct_channel_kernel`: one CTA per channel,
streams K_TOTAL = BATCH*TIME rows in BLOCK_T chunks, materializes the two cat
halves via scatter into the (B, C_TOTAL, T) output, and accumulates per-channel
partial sums via bf16-rounded values with an f32 accumulator, writing the
bf16-rounded final sums at the end.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 4
C_HALF = 512
C_TOTAL = 1024
TIME = 1493
K_TOTAL = BATCH * TIME
BLOCK_T = 1024


def _cdiv(a, b):
    return (a + b - 1) // b


NUM_ITERS = _cdiv(K_TOTAL, BLOCK_T)


@ct.kernel
def _direct_channel_kernel(
    lhs_flat_ptr,     # bf16 flat [BATCH*C_HALF*TIME]
    rhs_flat_ptr,     # bf16 flat [BATCH*C_HALF*TIME]
    gate_flat_ptr,    # bf16 flat [BATCH*C_TOTAL*TIME]
    out_flat_ptr,     # bf16 flat [BATCH*C_TOTAL*TIME]
    sum_first_ptr,    # f32 [C_HALF]
    sum_second_ptr,   # f32 [C_HALF]
    K_TOTAL_: ct.Constant[int],
    C_HALF_: ct.Constant[int],
    C_TOTAL_: ct.Constant[int],
    TIME_: ct.Constant[int],
    BLOCK_T_: ct.Constant[int],
    NUM_ITERS_: ct.Constant[int],
):
    c = ct.bid(0)

    acc_first = ct.zeros((1,), dtype=ct.float32)
    acc_second = ct.zeros((1,), dtype=ct.float32)

    for it in ct.static_iter(range(NUM_ITERS_)):
        lanes = ct.arange(BLOCK_T_, dtype=ct.int32)
        k = it * BLOCK_T_ + lanes
        active = k < K_TOTAL_
        batch = k // TIME_
        time = k - batch * TIME_

        half_offsets = batch * (C_HALF_ * TIME_) + c * TIME_ + time
        full_offsets = batch * (C_TOTAL_ * TIME_) + c * TIME_ + time

        lhs = ct.astype(ct.gather(lhs_flat_ptr, half_offsets, mask=active, padding_value=0), ct.float32)
        rhs = ct.astype(ct.gather(rhs_flat_ptr, half_offsets, mask=active, padding_value=0), ct.float32)
        add_unrounded = lhs + rhs
        add_rounded = ct.astype(ct.astype(add_unrounded, ct.bfloat16), ct.float32)

        gate_a = ct.astype(ct.gather(gate_flat_ptr, full_offsets, mask=active, padding_value=0), ct.float32)
        gate_b_offsets = full_offsets + C_HALF_ * TIME_
        gate_b = ct.astype(ct.gather(gate_flat_ptr, gate_b_offsets, mask=active, padding_value=0), ct.float32)

        sigmoid = 1.0 / (1.0 + ct.exp(-gate_b))
        gate_deriv = (1.0 - sigmoid) * sigmoid

        first = ct.astype(sigmoid * add_unrounded, ct.bfloat16)
        second = ct.astype(gate_deriv * gate_a * add_unrounded, ct.bfloat16)
        first_for_sum = ct.astype(sigmoid * add_rounded, ct.bfloat16)
        second_for_sum = ct.astype(gate_deriv * gate_a * add_rounded, ct.bfloat16)

        ct.scatter(out_flat_ptr, full_offsets, first, mask=active)
        ct.scatter(out_flat_ptr, gate_b_offsets, second, mask=active)

        first_for_sum_f = ct.astype(first_for_sum, ct.float32)
        second_for_sum_f = ct.astype(second_for_sum, ct.float32)
        first_masked = ct.where(active, first_for_sum_f, 0.0)
        second_masked = ct.where(active, second_for_sum_f, 0.0)
        acc_first = acc_first + ct.sum(first_masked, keepdims=True)
        acc_second = acc_second + ct.sum(second_masked, keepdims=True)

    first_rounded = ct.astype(ct.astype(acc_first, ct.bfloat16), ct.float32)
    second_rounded = ct.astype(ct.astype(acc_second, ct.bfloat16), ct.float32)
    ct.store(sum_first_ptr, index=(c,), tile=first_rounded)
    ct.store(sum_second_ptr, index=(c,), tile=second_rounded)


@oracle_impl(hardware="B200", point="bdbb44ef", BLOCK_T=BLOCK_T)
def oracle_forward(inputs, *, BLOCK_T: int):
    lhs, rhs, gate = inputs
    device = gate.device

    out = torch.empty_strided(
        (BATCH, C_TOTAL, TIME),
        (C_TOTAL * TIME, TIME, 1),
        device=device,
        dtype=torch.bfloat16,
    )
    sum_out = torch.empty_strided((C_TOTAL,), (1,), device=device, dtype=torch.float32)

    lhs_flat = lhs.reshape(-1)
    rhs_flat = rhs.reshape(-1)
    gate_flat = gate.reshape(-1)
    out_flat = out.view(-1)
    sum_first = sum_out[:C_HALF]
    sum_second = sum_out[C_HALF:]

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (C_HALF, 1, 1),
        _direct_channel_kernel,
        (lhs_flat, rhs_flat, gate_flat, out_flat, sum_first, sum_second,
         K_TOTAL, C_HALF, C_TOTAL, TIME, BLOCK_T, NUM_ITERS),
    )
    return out, sum_out
