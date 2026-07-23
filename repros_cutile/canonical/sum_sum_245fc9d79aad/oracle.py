"""cuTile port of sum_sum_245fc9d79aad: MobileNetV3 hard-sigmoid gated
avg-pool backward + sibling channel reductions + BN backward dense epilogue.

Matches Triton's three-kernel plan:
1. `_partial_reduce_kernel`: 2D grid (cdiv(C, BLOCK_C), num_k_tiles).
   Loads gated_x, act, gate, pool, mean and computes the producer inline.
   Emits per-(chunk, channel_block) partial_sum and partial_dot.
2. `_finalize_kernel`: grid (C,). Reduces partials across chunks and computes
   invstd-scaled stats (mean_term, prod_coeff, output_scale) plus sum_out and
   scale_grad output vectors.
3. `_epilogue_kernel`: grid (cdiv(total, EPILOGUE_BLOCK),). Recomputes the
   producer and applies BN backward using the finalized stats.

Producer is `where(act <= -3, fill, where(act < 3, base * (act/3+0.5), base))`
where `base = round_bf16(gated_x * hard_sigmoid(gate) + pool * (1/196))`.
All arithmetic uses cuTile default RTNE rounding (matches Triton PTX).
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


SCALE = 9.964923469387754e-06
POOL_INV = 0.00510204081632653  # 1/196


def _next_pow2(v: int) -> int:
    return 1 << (int(v) - 1).bit_length()


@ct.kernel
def _partial_reduce_kernel(
    gate_ptr,           # f32 [N*C]
    gated_x_ptr,        # bf16 [total]
    pool_ptr,           # bf16 [N*C]
    act_ptr,            # f32 [total]
    fill_ptr,           # f32 [1]
    mean_ptr,           # f32 [C]
    centered_x_ptr,     # bf16 [total]
    partial_sum_ptr,    # f32 [NUM_K_TILES, C]
    partial_prod_ptr,   # f32 [NUM_K_TILES, C]
    C: ct.Constant[int],
    HW: ct.Constant[int],
    K_TOTAL: ct.Constant[int],
    BLOCK_K: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    c_block = ct.bid(0)
    r_block = ct.bid(1)
    r_offsets = r_block * BLOCK_K + ct.arange(BLOCK_K, dtype=ct.int32)
    c_offsets = c_block * BLOCK_C + ct.arange(BLOCK_C, dtype=ct.int32)
    r_2d = ct.reshape(r_offsets, (BLOCK_K, 1))
    c_2d = ct.reshape(c_offsets, (1, BLOCK_C))
    K_full = ct.full((BLOCK_K, 1), K_TOTAL, dtype=ct.int32)
    C_full = ct.full((1, BLOCK_C), C, dtype=ct.int32)
    mask = (r_2d < K_full) & (c_2d < C_full)
    zero_i2d = ct.zeros((BLOCK_K, BLOCK_C), dtype=ct.int32)
    r_full = r_2d + zero_i2d
    c_full = c_2d + zero_i2d
    r_safe = ct.where(mask, r_full, zero_i2d)
    c_safe = ct.where(mask, c_full, zero_i2d)

    flat_offsets = r_safe * C + c_safe
    n = r_safe // HW
    nc_offsets = n * C + c_safe

    act = ct.gather(act_ptr, flat_offsets, mask=mask)
    fill_val = ct.load(fill_ptr, index=(0,), shape=(1,))
    fill_broadcast = ct.reshape(fill_val, (1, 1)) + ct.zeros(
        (BLOCK_K, BLOCK_C), dtype=ct.float32,
    )

    gate_src = ct.gather(gate_ptr, nc_offsets, mask=mask)
    gate = gate_src + 3.0
    zero_f2d = ct.zeros((BLOCK_K, BLOCK_C), dtype=ct.float32)
    six_f = ct.full((BLOCK_K, BLOCK_C), 6.0, dtype=ct.float32)
    gate = ct.where(gate < zero_f2d, zero_f2d, gate)
    gate = ct.where(gate > six_f, six_f, gate)
    gate_bf = ct.astype(gate * (1.0 / 6.0), ct.bfloat16)
    gate_f = ct.astype(gate_bf, ct.float32)

    gated_x = ct.astype(
        ct.gather(gated_x_ptr, flat_offsets, mask=mask), ct.float32,
    )
    pooled = ct.astype(ct.gather(pool_ptr, nc_offsets, mask=mask), ct.float32)
    gated_bf = ct.astype(gated_x * gate_f, ct.bfloat16)
    gated_f = ct.astype(gated_bf, ct.float32)
    pool_grad_bf = ct.astype(pooled * POOL_INV, ct.bfloat16)
    pool_grad_f = ct.astype(pool_grad_bf, ct.float32)
    base_bf = ct.astype(gated_f + pool_grad_f, ct.bfloat16)
    base = ct.astype(base_bf, ct.float32)

    middle_gate = act * (1.0 / 3.0) + 0.5
    middle = base * middle_gate
    three_f = ct.full((BLOCK_K, BLOCK_C), 3.0, dtype=ct.float32)
    neg_three_f = ct.full((BLOCK_K, BLOCK_C), -3.0, dtype=ct.float32)
    out = ct.where(act < three_f, middle, base)
    out = ct.where(act <= neg_three_f, fill_broadcast, out)
    producer_bf = ct.astype(out, ct.bfloat16)
    producer = ct.astype(producer_bf, ct.float32)

    mean_1d = ct.load(mean_ptr, index=(c_block,), shape=(BLOCK_C,))
    mean_2d = ct.reshape(mean_1d, (1, BLOCK_C)) + zero_f2d
    centered_x = ct.astype(
        ct.gather(centered_x_ptr, flat_offsets, mask=mask), ct.float32,
    )
    centered = centered_x - mean_2d

    sum_values = ct.where(mask, producer, zero_f2d)
    dot_values = ct.where(mask, producer * centered, zero_f2d)

    partial_sum = ct.sum(sum_values, axis=0)
    partial_dot = ct.sum(dot_values, axis=0)
    ct.store(
        partial_sum_ptr, index=(r_block, c_block),
        tile=ct.reshape(partial_sum, (1, BLOCK_C)),
    )
    ct.store(
        partial_prod_ptr, index=(r_block, c_block),
        tile=ct.reshape(partial_dot, (1, BLOCK_C)),
    )


@ct.kernel
def _finalize_kernel(
    partial_sum_ptr,     # f32 [NUM_K_TILES, C]
    partial_prod_ptr,    # f32 [NUM_K_TILES, C]
    invstd_ptr,          # f32 [C]
    weight_ptr,          # f32 [C]
    sum_out_ptr,         # f32 [C]
    mean_term_ptr,       # f32 [C]
    prod_coeff_ptr,      # f32 [C]
    output_scale_ptr,    # f32 [C]
    scale_grad_ptr,      # f32 [C]
    NUM_K_TILES: ct.Constant[int],
    BLOCK_TILES: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    c_block = ct.bid(0)
    chunks = ct.arange(BLOCK_TILES, dtype=ct.int32)
    c = c_block * BLOCK_C + ct.arange(BLOCK_C, dtype=ct.int32)
    chunks_2d = ct.reshape(chunks, (BLOCK_TILES, 1))
    c_2d = ct.reshape(c, (1, BLOCK_C))
    NT_full = ct.full((BLOCK_TILES, 1), NUM_K_TILES, dtype=ct.int32)
    mask = chunks_2d < NT_full
    zero_i2d = ct.zeros((BLOCK_TILES, BLOCK_C), dtype=ct.int32)
    zero_f2d = ct.zeros((BLOCK_TILES, BLOCK_C), dtype=ct.float32)

    chunks_full = chunks_2d + zero_i2d
    c_full = c_2d + zero_i2d
    chunks_safe = ct.where(mask, chunks_full, zero_i2d)

    partial_sum = ct.gather(
        partial_sum_ptr, (chunks_safe, c_full), mask=mask,
    )
    partial_prod = ct.gather(
        partial_prod_ptr, (chunks_safe, c_full), mask=mask,
    )
    partial_sum = ct.where(mask, partial_sum, zero_f2d)
    partial_prod = ct.where(mask, partial_prod, zero_f2d)
    sum_value = ct.sum(partial_sum, axis=0)
    prod_value = ct.sum(partial_prod, axis=0)

    invstd = ct.load(invstd_ptr, index=(c_block,), shape=(BLOCK_C,))
    weight = ct.load(weight_ptr, index=(c_block,), shape=(BLOCK_C,))

    mean_term = sum_value * SCALE
    prod_scaled = prod_value * SCALE
    invstd_sq = invstd * invstd
    prod_coeff = prod_scaled * invstd_sq
    output_scale = invstd * weight
    scale_grad = prod_value * invstd

    ct.store(sum_out_ptr, index=(c_block,), tile=sum_value)
    ct.store(mean_term_ptr, index=(c_block,), tile=mean_term)
    ct.store(prod_coeff_ptr, index=(c_block,), tile=prod_coeff)
    ct.store(output_scale_ptr, index=(c_block,), tile=output_scale)
    ct.store(scale_grad_ptr, index=(c_block,), tile=scale_grad)


@ct.kernel
def _epilogue_kernel(
    gate_ptr,           # f32 [N*C]
    gated_x_ptr,        # bf16 [total]
    pool_ptr,           # bf16 [N*C]
    act_ptr,            # f32 [total]
    fill_ptr,           # f32 [1]
    mean_ptr,           # f32 [C]
    centered_x_ptr,     # bf16 [total]
    mean_term_ptr,      # f32 [C]
    prod_coeff_ptr,     # f32 [C]
    output_scale_ptr,   # f32 [C]
    out_ptr,            # bf16 [total]
    C: ct.Constant[int],
    HW: ct.Constant[int],
    TOTAL: ct.Constant[int],
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    offsets = pid * BLOCK + ct.arange(BLOCK, dtype=ct.int32)
    TOTAL_full = ct.full((BLOCK,), TOTAL, dtype=ct.int32)
    mask = offsets < TOTAL_full
    zero_i = ct.zeros((BLOCK,), dtype=ct.int32)
    zero_f = ct.zeros((BLOCK,), dtype=ct.float32)
    safe_off = ct.where(mask, offsets, zero_i)

    c = safe_off - (safe_off // C) * C
    k = safe_off // C
    n = k // HW
    nc_off = n * C + c

    act = ct.gather(act_ptr, safe_off, mask=mask)
    fill_val = ct.load(fill_ptr, index=(0,), shape=(1,))
    fill_broadcast = ct.reshape(fill_val, (1,)) + zero_f

    gate_src = ct.gather(gate_ptr, nc_off, mask=mask)
    gate = gate_src + 3.0
    six_f = ct.full((BLOCK,), 6.0, dtype=ct.float32)
    gate = ct.where(gate < zero_f, zero_f, gate)
    gate = ct.where(gate > six_f, six_f, gate)
    gate_bf = ct.astype(gate * (1.0 / 6.0), ct.bfloat16)
    gate_f = ct.astype(gate_bf, ct.float32)

    gated_x = ct.astype(
        ct.gather(gated_x_ptr, safe_off, mask=mask), ct.float32,
    )
    pooled = ct.astype(
        ct.gather(pool_ptr, nc_off, mask=mask), ct.float32,
    )
    gated_bf = ct.astype(gated_x * gate_f, ct.bfloat16)
    gated_f = ct.astype(gated_bf, ct.float32)
    pool_grad_bf = ct.astype(pooled * POOL_INV, ct.bfloat16)
    pool_grad_f = ct.astype(pool_grad_bf, ct.float32)
    base_bf = ct.astype(gated_f + pool_grad_f, ct.bfloat16)
    base = ct.astype(base_bf, ct.float32)

    middle_gate = act * (1.0 / 3.0) + 0.5
    middle = base * middle_gate
    three_f = ct.full((BLOCK,), 3.0, dtype=ct.float32)
    neg_three_f = ct.full((BLOCK,), -3.0, dtype=ct.float32)
    out = ct.where(act < three_f, middle, base)
    out = ct.where(act <= neg_three_f, fill_broadcast, out)
    producer_bf = ct.astype(out, ct.bfloat16)
    producer = ct.astype(producer_bf, ct.float32)

    mean = ct.gather(mean_ptr, c, mask=mask)
    centered_x = ct.astype(
        ct.gather(centered_x_ptr, safe_off, mask=mask), ct.float32,
    )
    centered = centered_x - mean

    prod_coeff = ct.gather(prod_coeff_ptr, c, mask=mask)
    mean_term = ct.gather(mean_term_ptr, c, mask=mask)
    output_scale = ct.gather(output_scale_ptr, c, mask=mask)

    correction = centered * prod_coeff
    residual = producer - correction
    residual = residual - mean_term
    out_val = residual * output_scale
    out_bf = ct.astype(out_val, ct.bfloat16)
    ct.scatter(out_ptr, offsets, out_bf, mask=mask)


@oracle_impl(hardware="B200", point="5816bf68")
@oracle_impl(hardware="B200", point="06aedd03")
@oracle_impl(hardware="B200", point="230be935")
@oracle_impl(hardware="B200", point="212c1566")
@oracle_impl(hardware="B200", point="d6f080d2")
@oracle_impl(hardware="B200", point="3468f7bd")
@oracle_impl(hardware="B200", point="26a83dcc")
@oracle_impl(hardware="B200", point="50dc27bc")
def oracle_forward(inputs, **_kwargs):
    (
        gate,        # arg0 f32 [N,C,1,1]
        gated_x,     # arg1 bf16 [N,C,H,W] channels-last
        pooled,      # arg2 bf16 [N,C,1,1]
        act,         # arg3 f32 [N,C,H,W] channels-last
        fill,        # arg4 f32 scalar
        mean,        # arg5 f32 [1,C,1,1]
        centered_x,  # arg6 bf16 [N,C,H,W] channels-last
        invstd,      # arg7 f32 [1,C,1,1]
        weight,      # arg8 f32 [C]
        _shape,
    ) = inputs
    n, c, h, w = gated_x.shape
    hw = h * w
    k_total = n * hw
    total = k_total * c
    device = gated_x.device

    # Pick pow2 BLOCK sizes.
    # BLOCK_C divides C (C is 480/672/960 — all multiples of 16).
    block_c = 16
    # BLOCK_K is largest pow2 dividing k_total, capped at 512.
    block_k = 1
    for cand in (512, 256, 128, 64, 32, 16, 8, 4, 2, 1):
        if k_total % cand == 0:
            block_k = cand
            break
    num_k_tiles = k_total // block_k
    epilogue_block = 1024
    while total % epilogue_block != 0 and epilogue_block > 1:
        epilogue_block //= 2

    # Flat views (physical channels-last storage: n,h,w,c order).
    def _flat(t, size):
        if t.is_contiguous():
            return t.view(size)
        return t.as_strided((size,), (1,))

    gate_flat = gate.reshape(n * c).contiguous()  # f32 (N*C,) in n,c order
    pooled_flat = pooled.reshape(n * c).contiguous()  # bf16 (N*C,) in n,c order
    fill_1d = fill.reshape(1).contiguous().to(torch.float32)
    mean_flat = mean.reshape(c).contiguous()
    invstd_flat = invstd.reshape(c).contiguous()
    weight_flat = weight.reshape(c).contiguous()

    gated_x_flat = _flat(gated_x, total)
    act_flat = _flat(act, total)
    centered_x_flat = _flat(centered_x, total)

    out = torch.empty_strided(
        tuple(gated_x.shape), tuple(gated_x.stride()),
        device=device, dtype=torch.bfloat16,
    )
    out_flat = _flat(out, total)

    partial_sum = torch.empty_strided(
        (num_k_tiles, c), (c, 1), device=device, dtype=torch.float32,
    )
    partial_prod = torch.empty_like(partial_sum)
    sum_out = torch.empty_strided((c,), (1,), device=device, dtype=torch.float32)
    mean_term = torch.empty_strided((c,), (1,), device=device, dtype=torch.float32)
    prod_coeff = torch.empty_strided((c,), (1,), device=device, dtype=torch.float32)
    output_scale = torch.empty_strided((c,), (1,), device=device, dtype=torch.float32)
    scale_grad = torch.empty_strided((c,), (1,), device=device, dtype=torch.float32)

    block_tiles = _next_pow2(num_k_tiles)
    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (c // block_c, num_k_tiles, 1), _partial_reduce_kernel,
        (gate_flat, gated_x_flat, pooled_flat, act_flat, fill_1d,
         mean_flat, centered_x_flat,
         partial_sum, partial_prod, c, hw, k_total, block_k, block_c),
    )
    ct.launch(
        stream, (c // block_c, 1, 1), _finalize_kernel,
        (partial_sum, partial_prod, invstd_flat, weight_flat,
         sum_out, mean_term, prod_coeff, output_scale, scale_grad,
         num_k_tiles, block_tiles, block_c),
    )
    ct.launch(
        stream, (ct.cdiv(total, epilogue_block), 1, 1), _epilogue_kernel,
        (gate_flat, gated_x_flat, pooled_flat, act_flat, fill_1d,
         mean_flat, centered_x_flat,
         mean_term, prod_coeff, output_scale, out_flat,
         c, hw, total, epilogue_block),
    )

    return sum_out, scale_grad, out
