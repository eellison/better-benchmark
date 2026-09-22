"""cuTile port of sum_sum_3280cef0c732: GhostNet BN backward split-K.

Matches Triton's 3-kernel structure. All reductions and derived per-channel
scalar math run inside cuTile kernels; torch only handles allocation and
layout permutation.

  1. `_producer_add1_kernel` — computes the bf16 `add_out` producer (SE-gate
     hardsigmoid path over N*FULL_C*HW).
  2. `_partial_reduce_kernel` — for tail channels [C:FULL_C], applies the BN
     inference + ReLU + where path, then accumulates partial sums.
  3. `_finalize_kernel` — combines partials and computes per-channel scalars.
  4. `_epilogue_kernel` — per-element BN-backward output writeback.

Note: the Triton reference fuses the producer store + partial-reduce into a
single kernel via a channel mask (tail-only reduction). cuTile's tile-based
layout doesn't express that mixed producer/reduce pattern cleanly, so we
split the fused producer step into two kernels but keep every reduction and
elementwise op inside a `@ct.kernel`.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


N = 512
FULL_C = 672
C = 336
H = 14
W = 14
HW = H * W
R = N * HW
INV_R = 9.964923469387754e-06


def _next_power_of_2(v: int) -> int:
    return 1 << (int(v) - 1).bit_length()


@ct.kernel
def _producer_add1_kernel(
    arg0_ptr,        # bf16 [N*FULL_C]
    arg1_ptr,        # bf16 [N*H*W*FULL_C] channels-last flat
    arg2_ptr,        # bf16 [N*FULL_C]
    add_ptr,         # bf16 [N*H*W*FULL_C] channels-last flat
    N_HW_FC: ct.Constant[int],
    FC: ct.Constant[int],
    HW_: ct.Constant[int],
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    offsets = pid * BLOCK + ct.arange(BLOCK, dtype=ct.int64)
    mask = offsets < N_HW_FC

    c_idx = offsets % FC
    row_flat = offsets // FC
    n = row_flat // HW_
    param_idx = n * FC + c_idx

    gate_source = ct.astype(ct.gather(arg0_ptr, param_idx, mask=mask), ct.float32)
    shifted = gate_source + 3.0
    zero_t = ct.full((BLOCK,), 0.0, dtype=ct.float32)
    six_t = ct.full((BLOCK,), 6.0, dtype=ct.float32)
    clamped = ct.where(shifted < zero_t, zero_t, shifted)
    clamped = ct.where(clamped > six_t, six_t, clamped)
    gate = ct.astype(ct.astype(clamped / 6.0, ct.bfloat16), ct.float32)

    spatial = ct.astype(ct.gather(arg1_ptr, offsets, mask=mask), ct.float32)
    scaled = ct.astype(ct.astype(spatial * gate, ct.bfloat16), ct.float32)
    bias = ct.astype(ct.gather(arg2_ptr, param_idx, mask=mask), ct.float32)
    averaged = ct.astype(ct.astype(bias * (1.0 / 196.0), ct.bfloat16), ct.float32)
    add_value = ct.astype(scaled + averaged, ct.bfloat16)

    ct.scatter(add_ptr, offsets, add_value, mask=mask)


@ct.kernel
def _partial_reduce_kernel(
    add_ptr,           # bf16 (R, C)  — tail channels flattened
    x_ptr,             # bf16 (R, C)  — arg3 flattened
    fill_ptr,          # bf16 (1,)
    mean_ptr,          # f32  (C,)
    invstd_ptr,        # f32  (C,)
    weight_ptr,        # f32  (C,)  (affine weight = arg6)
    bias_ptr,          # f32  (C,)  (affine bias   = arg7)
    partial_sum_ptr,   # f32  (NUM_TILES, C)
    partial_dot_ptr,   # f32  (NUM_TILES, C)
    R_: ct.Constant[int],
    C_C: ct.Constant[int],
    BLOCK_R: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    r_block = ct.bid(0)
    c_block = ct.bid(1)

    add_v = ct.load(add_ptr, index=(r_block, c_block), shape=(BLOCK_R, BLOCK_C),
                    padding_mode=ct.PaddingMode.ZERO)
    x = ct.load(x_ptr, index=(r_block, c_block), shape=(BLOCK_R, BLOCK_C),
                padding_mode=ct.PaddingMode.ZERO)
    mean = ct.load(mean_ptr, index=(c_block,), shape=(BLOCK_C,),
                   padding_mode=ct.PaddingMode.ZERO)
    invstd = ct.load(invstd_ptr, index=(c_block,), shape=(BLOCK_C,),
                     padding_mode=ct.PaddingMode.ZERO)
    weight = ct.load(weight_ptr, index=(c_block,), shape=(BLOCK_C,),
                     padding_mode=ct.PaddingMode.ZERO)
    bias = ct.load(bias_ptr, index=(c_block,), shape=(BLOCK_C,),
                   padding_mode=ct.PaddingMode.ZERO)

    x_f = ct.astype(x, ct.float32)
    mean_2d = ct.reshape(mean, (1, BLOCK_C))
    invstd_2d = ct.reshape(invstd, (1, BLOCK_C))
    weight_2d = ct.reshape(weight, (1, BLOCK_C))
    bias_2d = ct.reshape(bias, (1, BLOCK_C))

    centered = x_f - mean_2d
    affine = centered * invstd_2d * weight_2d + bias_2d
    # Round to bf16 then back to f32 for comparison (matches Triton)
    affine_bf = ct.astype(affine, ct.bfloat16)

    fill = ct.load(fill_ptr, index=(0,), shape=(1,))
    fill_bc = ct.reshape(fill, (1, 1))
    fill_2d = fill_bc + ct.zeros((BLOCK_R, BLOCK_C), dtype=ct.bfloat16)
    zero_bf = ct.astype(0.0, ct.bfloat16)
    where_bf = ct.where(affine_bf <= zero_bf, fill_2d, add_v)
    where_f = ct.astype(where_bf, ct.float32)

    prod = where_f * centered

    # Row/col mask for tail.
    r_idx = ct.arange(BLOCK_R, dtype=ct.int32) + r_block * BLOCK_R
    c_idx = ct.arange(BLOCK_C, dtype=ct.int32) + c_block * BLOCK_C
    r_valid = ct.reshape(r_idx < R_, (BLOCK_R, 1))
    c_valid = ct.reshape(c_idx < C_C, (1, BLOCK_C))
    valid = r_valid & c_valid
    zero_f = ct.zeros((BLOCK_R, BLOCK_C), dtype=ct.float32)
    where_m = ct.where(valid, where_f, zero_f)
    prod_m = ct.where(valid, prod, zero_f)

    p_sum = ct.sum(where_m, axis=0, keepdims=True)
    p_dot = ct.sum(prod_m, axis=0, keepdims=True)
    ct.store(partial_sum_ptr, index=(r_block, c_block), tile=p_sum)
    ct.store(partial_dot_ptr, index=(r_block, c_block), tile=p_dot)


@ct.kernel
def _finalize_kernel(
    partial_sum_ptr,   # f32 (NUM_TILES, C)
    partial_dot_ptr,   # f32 (NUM_TILES, C)
    invstd_ptr,        # f32 (C,)
    weight_ptr,        # f32 (C,)
    sum_out_ptr,       # f32 (C,)
    vec_out_ptr,       # f32 (C,) = dot * invstd
    mean_term_ptr,     # f32 (C,) = sum * SCALE
    dot_coeff_ptr,     # f32 (C,) = dot * SCALE * invstd^2
    out_scale_ptr,     # f32 (C,) = invstd * weight
    NUM_TILES: ct.Constant[int],
    BLOCK_TILES: ct.Constant[int],
    C_C: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
    INV_R_: ct.Constant[float],
):
    c_block = ct.bid(0)
    tile = ct.load(partial_sum_ptr, index=(0, c_block),
                   shape=(BLOCK_TILES, BLOCK_C),
                   padding_mode=ct.PaddingMode.ZERO)
    tile2 = ct.load(partial_dot_ptr, index=(0, c_block),
                    shape=(BLOCK_TILES, BLOCK_C),
                    padding_mode=ct.PaddingMode.ZERO)
    t_idx = ct.arange(BLOCK_TILES, dtype=ct.int32)
    c_idx = ct.arange(BLOCK_C, dtype=ct.int32) + c_block * BLOCK_C
    t_valid = ct.reshape(t_idx < NUM_TILES, (BLOCK_TILES, 1))
    c_valid = ct.reshape(c_idx < C_C, (1, BLOCK_C))
    valid = t_valid & c_valid
    zero_f = ct.zeros((BLOCK_TILES, BLOCK_C), dtype=ct.float32)
    tile = ct.where(valid, tile, zero_f)
    tile2 = ct.where(valid, tile2, zero_f)

    sum_value = ct.sum(tile, axis=0)
    dot_value = ct.sum(tile2, axis=0)

    invstd = ct.load(invstd_ptr, index=(c_block,), shape=(BLOCK_C,),
                     padding_mode=ct.PaddingMode.ZERO)
    weight = ct.load(weight_ptr, index=(c_block,), shape=(BLOCK_C,),
                     padding_mode=ct.PaddingMode.ZERO)
    invstd_sq = invstd * invstd
    dot_scaled = dot_value * INV_R_
    mean_term = sum_value * INV_R_
    dot_coeff = dot_scaled * invstd_sq
    out_scale = invstd * weight
    vec = dot_value * invstd

    ct.store(sum_out_ptr, index=(c_block,), tile=sum_value)
    ct.store(vec_out_ptr, index=(c_block,), tile=vec)
    ct.store(mean_term_ptr, index=(c_block,), tile=mean_term)
    ct.store(dot_coeff_ptr, index=(c_block,), tile=dot_coeff)
    ct.store(out_scale_ptr, index=(c_block,), tile=out_scale)


@ct.kernel
def _epilogue_kernel(
    add_ptr,           # bf16 (R, C) — tail slice of add_out
    x_ptr,             # bf16 (R, C)
    fill_ptr,          # bf16 (1,)
    mean_ptr,          # f32 (C,)
    invstd_ptr,        # f32 (C,)
    weight_ptr,        # f32 (C,)   (arg6)
    bias_ptr,          # f32 (C,)   (arg7)
    mean_term_ptr,     # f32 (C,)
    dot_coeff_ptr,     # f32 (C,)
    out_scale_ptr,     # f32 (C,)
    out_ptr,           # bf16 (R, C)
    BLOCK_C: ct.Constant[int],
):
    pid = ct.bid(0)
    c_block = ct.bid(1)
    add_v = ct.load(add_ptr, index=(pid, c_block), shape=(1, BLOCK_C),
                    padding_mode=ct.PaddingMode.ZERO)
    x = ct.load(x_ptr, index=(pid, c_block), shape=(1, BLOCK_C),
                padding_mode=ct.PaddingMode.ZERO)
    mean = ct.load(mean_ptr, index=(c_block,), shape=(BLOCK_C,),
                   padding_mode=ct.PaddingMode.ZERO)
    invstd = ct.load(invstd_ptr, index=(c_block,), shape=(BLOCK_C,),
                     padding_mode=ct.PaddingMode.ZERO)
    weight = ct.load(weight_ptr, index=(c_block,), shape=(BLOCK_C,),
                     padding_mode=ct.PaddingMode.ZERO)
    bias = ct.load(bias_ptr, index=(c_block,), shape=(BLOCK_C,),
                   padding_mode=ct.PaddingMode.ZERO)
    mean_term = ct.load(mean_term_ptr, index=(c_block,), shape=(BLOCK_C,),
                        padding_mode=ct.PaddingMode.ZERO)
    dot_coeff = ct.load(dot_coeff_ptr, index=(c_block,), shape=(BLOCK_C,),
                        padding_mode=ct.PaddingMode.ZERO)
    out_scale = ct.load(out_scale_ptr, index=(c_block,), shape=(BLOCK_C,),
                        padding_mode=ct.PaddingMode.ZERO)

    x_f = ct.astype(x, ct.float32)
    mean_2d = ct.reshape(mean, (1, BLOCK_C))
    invstd_2d = ct.reshape(invstd, (1, BLOCK_C))
    weight_2d = ct.reshape(weight, (1, BLOCK_C))
    bias_2d = ct.reshape(bias, (1, BLOCK_C))
    centered = x_f - mean_2d
    affine = centered * invstd_2d * weight_2d + bias_2d
    affine_bf = ct.astype(affine, ct.bfloat16)

    fill = ct.load(fill_ptr, index=(0,), shape=(1,))
    fill_bc = ct.reshape(fill, (1, 1))
    fill_2d = fill_bc + ct.zeros((1, BLOCK_C), dtype=ct.bfloat16)
    zero_bf = ct.astype(0.0, ct.bfloat16)
    where_bf = ct.where(affine_bf <= zero_bf, fill_2d, add_v)
    where_f = ct.astype(where_bf, ct.float32)

    mean_term_2d = ct.reshape(mean_term, (1, BLOCK_C))
    dot_coeff_2d = ct.reshape(dot_coeff, (1, BLOCK_C))
    out_scale_2d = ct.reshape(out_scale, (1, BLOCK_C))

    adjusted = where_f - centered * dot_coeff_2d - mean_term_2d
    out = adjusted * out_scale_2d
    ct.store(out_ptr, index=(pid, c_block), tile=ct.astype(out, ct.bfloat16))


@oracle_impl(hardware="B200", point="d01c3ddd", BLOCK=1024)
def oracle_forward(inputs, *, BLOCK: int):
    (arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, _shape_param_0) = inputs
    device = arg1_1.device

    # Producer output in channels-last layout.
    add_out = torch.empty_strided(
        (N, FULL_C, H, W),
        (FULL_C * HW, 1, FULL_C * W, FULL_C),
        device=device, dtype=torch.bfloat16,
    )
    add_flat_buf = torch.empty(N * FULL_C * HW, device=device, dtype=torch.bfloat16)
    arg1_perm_flat = arg1_1.permute(0, 2, 3, 1).contiguous().view(-1)
    arg0_flat = arg0_1.view(N, FULL_C).contiguous().view(-1)
    arg2_flat = arg2_1.view(N, FULL_C).contiguous().view(-1)

    total = N * FULL_C * HW
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(total, BLOCK), 1, 1),
        _producer_add1_kernel,
        (arg0_flat, arg1_perm_flat, arg2_flat, add_flat_buf, total, FULL_C, HW, BLOCK),
    )
    # Copy channels-last-flat back to strided layout (N, FULL_C, H, W).
    add_out.copy_(add_flat_buf.view(N, H, W, FULL_C).permute(0, 3, 1, 2))

    # Prepare tail slice as (R, C) for partial-reduce.
    # add_flat_buf is (N, H, W, FULL_C) contiguous. Slice tail = [..., C:FULL_C].
    add_nhwc = add_flat_buf.view(N, H, W, FULL_C)
    add_tail = add_nhwc[..., C:FULL_C].contiguous().view(R, C)

    # arg3 is bf16 [N, C, H, W] channels-last (stride C*HW, 1, W*C, C).
    x_nhwc = arg3_1.permute(0, 2, 3, 1).contiguous()
    x_2d = x_nhwc.view(R, C)

    mean_1d = arg4_1.view(C).contiguous()
    invstd_1d = arg5_1.view(C).contiguous()
    weight_1d = arg6_1.view(C).contiguous().to(torch.float32)
    bias_1d = arg7_1.view(C).contiguous().to(torch.float32)
    fill_1d = arg8_1.view(1).contiguous()

    BLOCK_R = 512
    BLOCK_C = 16
    num_tiles = (R + BLOCK_R - 1) // BLOCK_R
    block_tiles = _next_power_of_2(num_tiles)
    num_c_blocks = (C + BLOCK_C - 1) // BLOCK_C

    partial_sum = torch.empty((num_tiles, C), device=device, dtype=torch.float32)
    partial_dot = torch.empty((num_tiles, C), device=device, dtype=torch.float32)
    sum_out = torch.empty((C,), device=device, dtype=torch.float32)
    vec_out = torch.empty((C,), device=device, dtype=torch.float32)
    mean_term = torch.empty((C,), device=device, dtype=torch.float32)
    dot_coeff = torch.empty((C,), device=device, dtype=torch.float32)
    out_scale = torch.empty((C,), device=device, dtype=torch.float32)

    ct.launch(
        stream,
        (num_tiles, num_c_blocks, 1),
        _partial_reduce_kernel,
        (add_tail, x_2d, fill_1d, mean_1d, invstd_1d, weight_1d, bias_1d,
         partial_sum, partial_dot, R, C, BLOCK_R, BLOCK_C),
    )
    ct.launch(
        stream,
        (num_c_blocks, 1, 1),
        _finalize_kernel,
        (partial_sum, partial_dot, invstd_1d, weight_1d,
         sum_out, vec_out, mean_term, dot_coeff, out_scale,
         num_tiles, block_tiles, C, BLOCK_C, INV_R),
    )

    out_flat = torch.empty((R, C), device=device, dtype=torch.bfloat16)
    ct.launch(
        stream,
        (R, num_c_blocks, 1),
        _epilogue_kernel,
        (add_tail, x_2d, fill_1d, mean_1d, invstd_1d, weight_1d, bias_1d,
         mean_term, dot_coeff, out_scale, out_flat, BLOCK_C),
    )

    out_cl = torch.empty_strided(
        (N, C, H, W),
        (C * HW, 1, C * W, C),
        device=device, dtype=torch.bfloat16,
    )
    out_cl.copy_(out_flat.view(N, H, W, C).permute(0, 3, 1, 2))

    return add_out, sum_out, vec_out, out_cl
