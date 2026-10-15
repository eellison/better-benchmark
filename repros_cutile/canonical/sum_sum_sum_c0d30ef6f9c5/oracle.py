"""cuTile port of sum_sum_sum_c0d30ef6f9c5: RepVGG dual BN backward + ReLU-mask.

Three-kernel structure mirroring the Triton oracle:
  - _partial_kernel: per (c_block, k_block) tile: compute the masked ReLU
    producer and accumulate partial (sum, dot_b, dot_a) reductions along HW.
  - _finalize_kernel: combine per-channel partials, emit sum_1/sum_3/mul_12/
    mul_21 and stats needed by the epilogue.
  - _epilogue_kernel: dense per-tile grad_a/grad_b epilogue over NHWC.

Uses metadata-only channels-last permute(0,2,3,1).reshape(NHW, C) — the
input tensors have stride (N*H*W, 1, W*C, C) so permute+reshape are free.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


SCALE = 0.00015943877551020407

BLOCK_K = 512
BLOCK_C = 32
BLOCK_TILES = 16  # power-of-2 upper bound for cdiv(6272, 512) == 13


def _next_pow2(v):
    r = 1
    while r < v:
        r <<= 1
    return r


@ct.kernel
def _partial_kernel(
    arg0_nc_ptr,          # bf16 [N, C]
    arg1_ptr,             # bf16 [NHW, C]
    arg6_ptr,             # bf16 [NHW, C]
    mean_a_ptr,           # f32 [C]
    invstd_a_ptr,         # f32 [C]
    weight_a_ptr,         # f32 [C]
    bias_a_ptr,           # f32 [C]
    mean_b_ptr,           # f32 [C]
    invstd_b_ptr,         # f32 [C]
    weight_b_ptr,         # f32 [C]
    bias_b_ptr,           # f32 [C]
    partials_sum_ptr,     # f32 [C, BLOCK_TILES]
    partials_dot_b_ptr,   # f32 [C, BLOCK_TILES]
    partials_dot_a_ptr,   # f32 [C, BLOCK_TILES]
    N: ct.Constant[int],
    C: ct.Constant[int],
    HW: ct.Constant[int],
    TOTAL: ct.Constant[int],
    BLOCK_K_: ct.Constant[int],
    BLOCK_C_: ct.Constant[int],
):
    c_block = ct.bid(0)
    k_block = ct.bid(1)
    c_off = ct.arange(BLOCK_C_, dtype=ct.int32) + c_block * BLOCK_C_
    k_off = ct.arange(BLOCK_K_, dtype=ct.int32) + k_block * BLOCK_K_
    c_mask = c_off < C
    k_mask = k_off < TOTAL
    mask = ct.reshape(k_mask, (BLOCK_K_, 1)) & ct.reshape(c_mask, (1, BLOCK_C_))

    def _load_c(ptr):
        v = ct.load(
            ptr, index=(c_block,), shape=(BLOCK_C_,),
            padding_mode=ct.PaddingMode.ZERO,
        )
        return ct.reshape(ct.astype(v, ct.float32), (1, BLOCK_C_))

    mean_a = _load_c(mean_a_ptr)
    invstd_a = _load_c(invstd_a_ptr)
    weight_a = _load_c(weight_a_ptr)
    bias_a = _load_c(bias_a_ptr)
    mean_b = _load_c(mean_b_ptr)
    invstd_b = _load_c(invstd_b_ptr)
    weight_b = _load_c(weight_b_ptr)
    bias_b = _load_c(bias_b_ptr)

    x_a = ct.astype(
        ct.load(arg1_ptr, index=(k_block, c_block),
                shape=(BLOCK_K_, BLOCK_C_),
                padding_mode=ct.PaddingMode.ZERO),
        ct.float32,
    )
    x_b = ct.astype(
        ct.load(arg6_ptr, index=(k_block, c_block),
                shape=(BLOCK_K_, BLOCK_C_),
                padding_mode=ct.PaddingMode.ZERO),
        ct.float32,
    )
    centered_a = x_a - mean_a
    centered_b = x_b - mean_b
    branch_a = centered_a * invstd_a * weight_a + bias_a
    branch_b = centered_b * invstd_b * weight_b + bias_b
    branch_a_bf = ct.astype(branch_a, ct.bfloat16)
    branch_b_bf = ct.astype(branch_b, ct.bfloat16)
    add_bf = ct.astype(branch_a_bf + branch_b_bf, ct.bfloat16)
    zero_bf = ct.full((BLOCK_K_, BLOCK_C_), 0.0, dtype=ct.bfloat16)
    le = add_bf <= zero_bf

    n_of_k = ct.reshape(k_off // HW, (BLOCK_K_, 1))
    br_out = ct.zeros((BLOCK_K_, BLOCK_C_), dtype=ct.float32)
    for n_i in ct.static_iter(range(N)):
        sel = ct.reshape(n_of_k == n_i, (BLOCK_K_, 1))
        row_i = ct.load(
            arg0_nc_ptr, index=(n_i, c_block),
            shape=(1, BLOCK_C_),
            padding_mode=ct.PaddingMode.ZERO,
        )
        row_i_f = ct.astype(row_i, ct.float32)
        row_scaled_bf = ct.astype(row_i_f * (1.0 / 49.0), ct.bfloat16)
        row_scaled_f = ct.astype(row_scaled_bf, ct.float32)
        br_out = ct.where(sel, row_scaled_f, br_out)

    zero_f = ct.full((BLOCK_K_, BLOCK_C_), 0.0, dtype=ct.float32)
    where_value = ct.where(le, zero_f, br_out)
    where_value = ct.where(mask, where_value, zero_f)

    partial_sum = ct.sum(where_value, axis=0)
    centered_b_m = ct.where(mask, centered_b, zero_f)
    centered_a_m = ct.where(mask, centered_a, zero_f)
    partial_dot_b = ct.sum(where_value * centered_b_m, axis=0)
    partial_dot_a = ct.sum(where_value * centered_a_m, axis=0)

    ct.store(
        partials_sum_ptr, index=(c_block, k_block),
        tile=ct.reshape(partial_sum, (BLOCK_C_, 1)),
    )
    ct.store(
        partials_dot_b_ptr, index=(c_block, k_block),
        tile=ct.reshape(partial_dot_b, (BLOCK_C_, 1)),
    )
    ct.store(
        partials_dot_a_ptr, index=(c_block, k_block),
        tile=ct.reshape(partial_dot_a, (BLOCK_C_, 1)),
    )


@ct.kernel
def _finalize_kernel(
    partials_sum_ptr,      # f32 [C, BLOCK_TILES]
    partials_dot_b_ptr,
    partials_dot_a_ptr,
    invstd_a_ptr,          # f32 [C]
    weight_a_ptr,          # f32 [C]
    invstd_b_ptr,          # f32 [C]
    weight_b_ptr,          # f32 [C]
    sum_1_ptr,             # f32 [C]
    sum_3_ptr,             # f32 [C]
    mul_12_ptr,            # f32 [C]
    mul_21_ptr,            # f32 [C]
    stats_ptr,             # f32 [5, C]
    C: ct.Constant[int],
    NUM_TILES: ct.Constant[int],
    BLOCK_TILES_: ct.Constant[int],
    BLOCK_C_: ct.Constant[int],
    SCALE_VALUE: ct.Constant[float],
):
    c_block = ct.bid(0)
    c_off = ct.arange(BLOCK_C_, dtype=ct.int32) + c_block * BLOCK_C_
    c_mask = c_off < C
    t_off = ct.arange(BLOCK_TILES_, dtype=ct.int32)
    t_mask = t_off < NUM_TILES

    def _reduce(ptr):
        v = ct.load(
            ptr, index=(c_block, 0),
            shape=(BLOCK_C_, BLOCK_TILES_),
            padding_mode=ct.PaddingMode.ZERO,
        )
        m2d = ct.reshape(c_mask, (BLOCK_C_, 1)) & ct.reshape(t_mask, (1, BLOCK_TILES_))
        zero_f = ct.full((BLOCK_C_, BLOCK_TILES_), 0.0, dtype=ct.float32)
        v = ct.where(m2d, v, zero_f)
        return ct.sum(v, axis=1)

    sum_value = _reduce(partials_sum_ptr)
    dot_b = _reduce(partials_dot_b_ptr)
    dot_a = _reduce(partials_dot_a_ptr)

    invstd_a = ct.astype(
        ct.load(invstd_a_ptr, index=(c_block,), shape=(BLOCK_C_,),
                padding_mode=ct.PaddingMode.ZERO),
        ct.float32,
    )
    weight_a = ct.astype(
        ct.load(weight_a_ptr, index=(c_block,), shape=(BLOCK_C_,),
                padding_mode=ct.PaddingMode.ZERO),
        ct.float32,
    )
    invstd_b = ct.astype(
        ct.load(invstd_b_ptr, index=(c_block,), shape=(BLOCK_C_,),
                padding_mode=ct.PaddingMode.ZERO),
        ct.float32,
    )
    weight_b = ct.astype(
        ct.load(weight_b_ptr, index=(c_block,), shape=(BLOCK_C_,),
                padding_mode=ct.PaddingMode.ZERO),
        ct.float32,
    )

    ct.store(sum_1_ptr, index=(c_block,), tile=sum_value)
    ct.store(sum_3_ptr, index=(c_block,), tile=sum_value)
    ct.store(mul_12_ptr, index=(c_block,), tile=dot_b * invstd_b)
    ct.store(mul_21_ptr, index=(c_block,), tile=dot_a * invstd_a)
    ct.store(stats_ptr, index=(0, c_block),
             tile=ct.reshape(sum_value * SCALE_VALUE, (1, BLOCK_C_)))
    ct.store(stats_ptr, index=(1, c_block),
             tile=ct.reshape(dot_b * SCALE_VALUE * invstd_b * invstd_b, (1, BLOCK_C_)))
    ct.store(stats_ptr, index=(2, c_block),
             tile=ct.reshape(invstd_b * weight_b, (1, BLOCK_C_)))
    ct.store(stats_ptr, index=(3, c_block),
             tile=ct.reshape(dot_a * SCALE_VALUE * invstd_a * invstd_a, (1, BLOCK_C_)))
    ct.store(stats_ptr, index=(4, c_block),
             tile=ct.reshape(invstd_a * weight_a, (1, BLOCK_C_)))


@ct.kernel
def _epilogue_kernel(
    arg0_nc_ptr,           # bf16 [N, C]
    arg1_ptr,              # bf16 [NHW, C]
    arg6_ptr,              # bf16 [NHW, C]
    mean_a_ptr,            # f32 [C]
    invstd_a_ptr,          # f32 [C]
    weight_a_ptr,          # f32 [C]
    bias_a_ptr,            # f32 [C]
    mean_b_ptr,            # f32 [C]
    invstd_b_ptr,          # f32 [C]
    weight_b_ptr,          # f32 [C]
    bias_b_ptr,            # f32 [C]
    stats_ptr,             # f32 [5, C]
    out_b_ptr,             # bf16 [NHW, C]
    out_a_ptr,             # bf16 [NHW, C]
    N: ct.Constant[int],
    C: ct.Constant[int],
    HW: ct.Constant[int],
    TOTAL: ct.Constant[int],
    BLOCK_K_: ct.Constant[int],
    BLOCK_C_: ct.Constant[int],
):
    c_block = ct.bid(0)
    k_block = ct.bid(1)
    c_off = ct.arange(BLOCK_C_, dtype=ct.int32) + c_block * BLOCK_C_
    k_off = ct.arange(BLOCK_K_, dtype=ct.int32) + k_block * BLOCK_K_
    c_mask = c_off < C
    k_mask = k_off < TOTAL

    def _load_c(ptr):
        v = ct.load(
            ptr, index=(c_block,), shape=(BLOCK_C_,),
            padding_mode=ct.PaddingMode.ZERO,
        )
        return ct.reshape(ct.astype(v, ct.float32), (1, BLOCK_C_))

    mean_a = _load_c(mean_a_ptr)
    invstd_a = _load_c(invstd_a_ptr)
    weight_a = _load_c(weight_a_ptr)
    bias_a = _load_c(bias_a_ptr)
    mean_b = _load_c(mean_b_ptr)
    invstd_b = _load_c(invstd_b_ptr)
    weight_b = _load_c(weight_b_ptr)
    bias_b = _load_c(bias_b_ptr)

    x_a = ct.astype(
        ct.load(arg1_ptr, index=(k_block, c_block),
                shape=(BLOCK_K_, BLOCK_C_),
                padding_mode=ct.PaddingMode.ZERO),
        ct.float32,
    )
    x_b = ct.astype(
        ct.load(arg6_ptr, index=(k_block, c_block),
                shape=(BLOCK_K_, BLOCK_C_),
                padding_mode=ct.PaddingMode.ZERO),
        ct.float32,
    )
    centered_a = x_a - mean_a
    centered_b = x_b - mean_b
    branch_a = centered_a * invstd_a * weight_a + bias_a
    branch_b = centered_b * invstd_b * weight_b + bias_b
    branch_a_bf = ct.astype(branch_a, ct.bfloat16)
    branch_b_bf = ct.astype(branch_b, ct.bfloat16)
    add_bf = ct.astype(branch_a_bf + branch_b_bf, ct.bfloat16)
    zero_bf = ct.full((BLOCK_K_, BLOCK_C_), 0.0, dtype=ct.bfloat16)
    le = add_bf <= zero_bf

    n_of_k = ct.reshape(k_off // HW, (BLOCK_K_, 1))
    br_out = ct.zeros((BLOCK_K_, BLOCK_C_), dtype=ct.float32)
    for n_i in ct.static_iter(range(N)):
        sel = ct.reshape(n_of_k == n_i, (BLOCK_K_, 1))
        row_i = ct.load(
            arg0_nc_ptr, index=(n_i, c_block),
            shape=(1, BLOCK_C_),
            padding_mode=ct.PaddingMode.ZERO,
        )
        row_i_f = ct.astype(row_i, ct.float32)
        row_scaled_bf = ct.astype(row_i_f * (1.0 / 49.0), ct.bfloat16)
        row_scaled_f = ct.astype(row_scaled_bf, ct.float32)
        br_out = ct.where(sel, row_scaled_f, br_out)

    zero_f = ct.full((BLOCK_K_, BLOCK_C_), 0.0, dtype=ct.float32)
    where_value = ct.where(le, zero_f, br_out)

    mt = ct.reshape(
        ct.astype(
            ct.load(stats_ptr, index=(0, c_block),
                    shape=(1, BLOCK_C_),
                    padding_mode=ct.PaddingMode.ZERO),
            ct.float32,
        ),
        (1, BLOCK_C_),
    )
    cb = ct.reshape(
        ct.astype(
            ct.load(stats_ptr, index=(1, c_block),
                    shape=(1, BLOCK_C_),
                    padding_mode=ct.PaddingMode.ZERO),
            ct.float32,
        ),
        (1, BLOCK_C_),
    )
    sb = ct.reshape(
        ct.astype(
            ct.load(stats_ptr, index=(2, c_block),
                    shape=(1, BLOCK_C_),
                    padding_mode=ct.PaddingMode.ZERO),
            ct.float32,
        ),
        (1, BLOCK_C_),
    )
    ca = ct.reshape(
        ct.astype(
            ct.load(stats_ptr, index=(3, c_block),
                    shape=(1, BLOCK_C_),
                    padding_mode=ct.PaddingMode.ZERO),
            ct.float32,
        ),
        (1, BLOCK_C_),
    )
    sa = ct.reshape(
        ct.astype(
            ct.load(stats_ptr, index=(4, c_block),
                    shape=(1, BLOCK_C_),
                    padding_mode=ct.PaddingMode.ZERO),
            ct.float32,
        ),
        (1, BLOCK_C_),
    )

    out_b = ((where_value - centered_b * cb) - mt) * sb
    out_a = ((where_value - centered_a * ca) - mt) * sa
    ct.store(out_b_ptr, index=(k_block, c_block),
             tile=ct.astype(out_b, ct.bfloat16))
    ct.store(out_a_ptr, index=(k_block, c_block),
             tile=ct.astype(out_a, ct.bfloat16))


def _shape_tuple(shape):
    return tuple(int(dim) for dim in shape)


@oracle_impl(hardware="B200", point="ac965caa")
def oracle_forward(inputs):
    (a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10,
     sp0, sp1, sp2, sp3, sp4, sp5, sp6) = inputs
    device = a0.device
    N, C, H, W = int(a1.shape[0]), int(a1.shape[1]), int(a1.shape[2]), int(a1.shape[3])
    HW = H * W
    TOTAL = N * HW
    NUM_TILES = (TOTAL + BLOCK_K - 1) // BLOCK_K
    assert NUM_TILES <= BLOCK_TILES, (NUM_TILES, BLOCK_TILES)

    view = a0.view(*_shape_tuple(sp0))
    a0_nc = view.squeeze(3).squeeze(2).contiguous()

    a1_nhw = a1.permute(0, 2, 3, 1).reshape(TOTAL, C)
    a6_nhw = a6.permute(0, 2, 3, 1).reshape(TOTAL, C)

    mean_a = a2.squeeze(0).squeeze(-1).squeeze(-1)
    invstd_a = a3.squeeze(0).squeeze(-1).squeeze(-1)
    weight_a = a4
    bias_a = a5
    mean_b = a7.squeeze(0).squeeze(-1).squeeze(-1)
    invstd_b = a8.squeeze(0).squeeze(-1).squeeze(-1)
    weight_b = a9
    bias_b = a10

    partials_sum = torch.empty((C, BLOCK_TILES), device=device, dtype=torch.float32)
    partials_dot_b = torch.empty((C, BLOCK_TILES), device=device, dtype=torch.float32)
    partials_dot_a = torch.empty((C, BLOCK_TILES), device=device, dtype=torch.float32)

    sum_1 = torch.empty((C,), device=device, dtype=torch.float32)
    sum_3 = torch.empty((C,), device=device, dtype=torch.float32)
    mul_12 = torch.empty((C,), device=device, dtype=torch.float32)
    mul_21 = torch.empty((C,), device=device, dtype=torch.float32)
    stats = torch.empty((5, C), device=device, dtype=torch.float32)
    scalar_zero = torch.zeros((), device=device, dtype=torch.bfloat16)

    grad_b_flat = torch.empty((TOTAL, C), device=device, dtype=torch.bfloat16)
    grad_a_flat = torch.empty((TOTAL, C), device=device, dtype=torch.bfloat16)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        ((C + BLOCK_C - 1) // BLOCK_C, NUM_TILES, 1),
        _partial_kernel,
        (a0_nc, a1_nhw, a6_nhw,
         mean_a, invstd_a, weight_a, bias_a,
         mean_b, invstd_b, weight_b, bias_b,
         partials_sum, partials_dot_b, partials_dot_a,
         N, C, HW, TOTAL, BLOCK_K, BLOCK_C),
    )
    ct.launch(
        stream,
        ((C + BLOCK_C - 1) // BLOCK_C, 1, 1),
        _finalize_kernel,
        (partials_sum, partials_dot_b, partials_dot_a,
         invstd_a, weight_a, invstd_b, weight_b,
         sum_1, sum_3, mul_12, mul_21, stats,
         C, NUM_TILES, BLOCK_TILES, BLOCK_C, SCALE),
    )
    ct.launch(
        stream,
        ((C + BLOCK_C - 1) // BLOCK_C, NUM_TILES, 1),
        _epilogue_kernel,
        (a0_nc, a1_nhw, a6_nhw,
         mean_a, invstd_a, weight_a, bias_a,
         mean_b, invstd_b, weight_b, bias_b,
         stats, grad_b_flat, grad_a_flat,
         N, C, HW, TOTAL, BLOCK_K, BLOCK_C),
    )

    grad_b = grad_b_flat.view(N, H, W, C).permute(0, 3, 1, 2)
    grad_a = grad_a_flat.view(N, H, W, C).permute(0, 3, 1, 2)
    return scalar_zero, sum_1, mul_12, grad_b, sum_3, mul_21, grad_a
