"""cuTile port of sum_407f0323586d (SCHEDULER_FUSION): NFNet bf16 residual-SiLU-derivative
that materializes a bf16 output and reduces per-channel sum. Channels-last layout.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


N = 128
C = 128
H = 56
W = 56
K_TOTAL = N * H * W  # 401408


@ct.kernel
def _materialize_partial_sum_kernel(
    x0_ptr,      # bf16 [K_TOTAL, C]
    x1_ptr,      # bf16 [K_TOTAL, C]
    gate_ptr,    # bf16 [K_TOTAL, C]
    out_ptr,     # bf16 [K_TOTAL, C]
    partial_ptr, # f32 [num_partials, C]
    C_C: ct.Constant[int],
    BLOCK_K: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    k_block = ct.bid(0)
    c_block = ct.bid(1)
    x0 = ct.load(x0_ptr, index=(k_block, c_block), shape=(BLOCK_K, BLOCK_C))
    x1 = ct.load(x1_ptr, index=(k_block, c_block), shape=(BLOCK_K, BLOCK_C))
    gate = ct.load(gate_ptr, index=(k_block, c_block), shape=(BLOCK_K, BLOCK_C))
    x0_f = ct.astype(x0, ct.float32)
    x1_f = ct.astype(x1, ct.float32)
    gate_f = ct.astype(gate, ct.float32)
    # residual = round_bf16(x0 + x1) then convert back to f32
    residual = ct.astype(ct.astype(x0_f + x1_f, ct.bfloat16), ct.float32)
    # sigmoid(gate) = 1 / (1 + exp(-gate))
    sigmoid_ = 1.0 / (1.0 + ct.exp(-gate_f))
    value = residual * sigmoid_ * (gate_f * (1.0 - sigmoid_) + 1.0)
    value_bf16 = ct.astype(value, ct.bfloat16)
    ct.store(out_ptr, index=(k_block, c_block), tile=value_bf16)
    # Reduce over K axis (axis=0) -> [BLOCK_C]
    partial = ct.sum(ct.astype(value_bf16, ct.float32), axis=0)
    # partial: [BLOCK_C]. Store into partial_ptr at (k_block, c_block)
    # partial_ptr shape [num_partials, C], so index (k_block, c_block) shape (1, BLOCK_C)
    partial_2d = ct.reshape(partial, (1, BLOCK_C))
    ct.store(partial_ptr, index=(k_block, c_block), tile=partial_2d)


@ct.kernel
def _final_channel_sum_kernel(
    partial_ptr,  # f32 [num_partials, C]
    sum_ptr,      # f32 [C]
    NUM_PARTIALS: ct.Constant[int],
    C_C: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
    BLOCK_PARTIALS: ct.Constant[int],
):
    c_block = ct.bid(0)
    partials = ct.load(
        partial_ptr, index=(0, c_block), shape=(BLOCK_PARTIALS, BLOCK_C),
        padding_mode=ct.PaddingMode.ZERO,
    )
    total = ct.sum(partials, axis=0)
    # Round-trip bf16 -> f32
    total_bf = ct.astype(total, ct.bfloat16)
    total_f = ct.astype(total_bf, ct.float32)
    ct.store(sum_ptr, index=(c_block,), tile=total_f)


def _next_pow2(n):
    p = 1
    while p < n:
        p *= 2
    return p


@oracle_impl(hardware="B200", point="97f7c01b", BLOCK_K=256, BLOCK_C=32, FINAL_BLOCK_C=32)
def oracle_forward(inputs, *, BLOCK_K: int, BLOCK_C: int, FINAL_BLOCK_C: int):
    x0, x1, gate = inputs
    num_partials = (K_TOTAL + BLOCK_K - 1) // BLOCK_K  # 401408/256 = 1568

    out = torch.empty_strided(
        (N, C, H, W),
        tuple(int(s) for s in x0.stride()),
        device=x0.device,
        dtype=torch.bfloat16,
    )
    partials = torch.empty_strided(
        (num_partials, C),
        (C, 1),
        device=x0.device,
        dtype=torch.float32,
    )
    sum_out = torch.empty_strided((C,), (1,), device=x0.device, dtype=torch.float32)

    # Reshape channels-last data to [K_TOTAL, C] via as_strided:
    # Physical stride of C axis is 1, stride of K axis is C. Storage order is
    # k*C + c for the [K_TOTAL, C] view.
    x0_2d = torch.as_strided(x0, (K_TOTAL, C), (C, 1))
    x1_2d = torch.as_strided(x1, (K_TOTAL, C), (C, 1))
    gate_2d = torch.as_strided(gate, (K_TOTAL, C), (C, 1))
    out_2d = torch.as_strided(out, (K_TOTAL, C), (C, 1))

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (num_partials, C // BLOCK_C, 1),
        _materialize_partial_sum_kernel,
        (x0_2d, x1_2d, gate_2d, out_2d, partials, C, BLOCK_K, BLOCK_C),
    )
    block_partials = _next_pow2(num_partials)
    ct.launch(
        stream,
        (C // FINAL_BLOCK_C, 1, 1),
        _final_channel_sum_kernel,
        (partials, sum_out, num_partials, C, FINAL_BLOCK_C, block_partials),
    )
    return out, sum_out
