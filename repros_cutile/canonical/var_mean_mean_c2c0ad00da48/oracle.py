"""cuTile port of var_mean_mean_c2c0ad00da48: MnasNet training-BN + ReLU + spatial mean + seeded dropout.

Multi-kernel plan (all cuTile):
1. `_bn_partial_stats_kernel`: reduces the NCHW input into per-channel partial
   (sum, sum_sq) over BLOCK_E chunks along the flattened N*H*W axis.
2. `_bn_finalize_stats_kernel`: combines partials into (mean, invstd), stores
   saved_mean/invstd, updates running_mean/running_var via 0.9997/0.00029999... .
3. `_bn_relu_spatial_mean_dropout_kernel`: affine + bf16 round + relu + spatial
   mean + seeded-dropout keep-prob 0.8 (pre-generated random tensor).

The dropout uses a randomly-drawn seed via `inductor_seeds`, so we mirror that
by calling the same op path in oracle_forward and passing the resulting random
tensor to the kernel.

Layout: arg0_1 is NCHW-contiguous bf16[32, 1280, 7, 7]. We flatten as
[N*H*W, C] via permute+reshape (channels-last logical layout), do BN over C.
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


EPS = 1.0e-5
MOMENTUM = 0.9997
ONE_MINUS_MOMENTUM = 0.00029999999999996696
UNBIAS_CORRECTION = 1.0006381620931717
DROPOUT_KEEP = 0.8
DROPOUT_SCALE = 1.0 / 0.8


def _next_power_of_2(value):
    return 1 << (int(value) - 1).bit_length()


@ct.kernel
def _bn_partial_stats_kernel(
    x_ptr,       # bf16 [E, C]  (channels-last logical layout)
    psum_ptr,    # f32  [num_chunks, C]
    psq_ptr,     # f32  [num_chunks, C]
    E: ct.Constant[int],
    BLOCK_E: ct.Constant[int],
    C_BLOCK: ct.Constant[int],
):
    chunk = ct.bid(0)
    cblk = ct.bid(1)

    x = ct.load(
        x_ptr, index=(chunk, cblk), shape=(BLOCK_E, C_BLOCK),
        padding_mode=ct.PaddingMode.ZERO,
    )
    x_f = ct.astype(x, ct.float32)

    e_arange = ct.arange(BLOCK_E, dtype=ct.int32)
    e_global = chunk * BLOCK_E + e_arange
    e_valid = e_global < E
    e_valid_2d = ct.reshape(e_valid, (BLOCK_E, 1))
    zero = ct.full((BLOCK_E, C_BLOCK), 0.0, dtype=ct.float32)
    x_masked = ct.where(e_valid_2d, x_f, zero)

    sums = ct.sum(x_masked, axis=0, keepdims=True)
    sums_sq = ct.sum(x_masked * x_masked, axis=0, keepdims=True)

    ct.store(psum_ptr, index=(chunk, cblk), tile=sums)
    ct.store(psq_ptr, index=(chunk, cblk), tile=sums_sq)


@ct.kernel
def _bn_finalize_stats_kernel(
    psum_ptr,           # f32 [num_chunks_pad, C]
    psq_ptr,            # f32 [num_chunks_pad, C]
    running_mean_ptr,   # f32 [C]  (read+updated)
    running_var_ptr,    # f32 [C]  (read+updated)
    saved_mean_ptr,     # f32 [C]
    invstd_ptr,         # f32 [C]
    NUM_CHUNKS: ct.Constant[int],
    BLOCK_CHUNKS: ct.Constant[int],
    C_BLOCK: ct.Constant[int],
    INV_E: ct.Constant[float],
    EPS_: ct.Constant[float],
    UNBIAS: ct.Constant[float],
    MOM: ct.Constant[float],
    ONE_MINUS_MOM: ct.Constant[float],
):
    cblk = ct.bid(0)

    psum = ct.load(
        psum_ptr, index=(0, cblk), shape=(BLOCK_CHUNKS, C_BLOCK),
        padding_mode=ct.PaddingMode.ZERO,
    )
    psq = ct.load(
        psq_ptr, index=(0, cblk), shape=(BLOCK_CHUNKS, C_BLOCK),
        padding_mode=ct.PaddingMode.ZERO,
    )

    chunks = ct.arange(BLOCK_CHUNKS, dtype=ct.int32)
    chunk_mask = chunks < NUM_CHUNKS
    chunk_mask_2d = ct.reshape(chunk_mask, (BLOCK_CHUNKS, 1))
    zero = ct.full((BLOCK_CHUNKS, C_BLOCK), 0.0, dtype=ct.float32)

    total = ct.sum(ct.where(chunk_mask_2d, psum, zero), axis=0)
    total_sq = ct.sum(ct.where(chunk_mask_2d, psq, zero), axis=0)

    mean = total * INV_E
    ex2 = total_sq * INV_E
    diff = ex2 - mean * mean
    zero_c = ct.full((C_BLOCK,), 0.0, dtype=ct.float32)
    var = ct.where(diff > zero_c, diff, zero_c)
    invstd = ct.rsqrt(var + EPS_)

    old_mean = ct.load(running_mean_ptr, index=(cblk,), shape=(C_BLOCK,))
    old_var = ct.load(running_var_ptr, index=(cblk,), shape=(C_BLOCK,))
    new_mean = mean * ONE_MINUS_MOM + old_mean * MOM
    new_var = (var * UNBIAS) * ONE_MINUS_MOM + old_var * MOM

    ct.store(saved_mean_ptr, index=(cblk,), tile=mean)
    ct.store(invstd_ptr, index=(cblk,), tile=invstd)
    ct.store(running_mean_ptr, index=(cblk,), tile=new_mean)
    ct.store(running_var_ptr, index=(cblk,), tile=new_var)


@ct.kernel
def _bn_relu_spatial_mean_dropout_kernel(
    x_ptr,           # bf16 [N*HW_PAD, C]  (padded)
    weight_ptr,      # f32 [C]
    bias_ptr,        # f32 [C]
    saved_mean_ptr,  # f32 [C]
    invstd_ptr,      # f32 [C]
    random_ptr,      # f32 [N, C]
    lt_ptr,          # b8  [N, C]
    out_ptr,         # bf16 [N, C]
    HW: ct.Constant[int],
    INV_HW: ct.Constant[float],
    HW_PAD: ct.Constant[int],
    C_BLOCK: ct.Constant[int],
    SCALE: ct.Constant[float],
):
    n = ct.bid(0)
    cblk = ct.bid(1)

    x = ct.load(x_ptr, index=(n, cblk), shape=(HW_PAD, C_BLOCK))
    x_f = ct.astype(x, ct.float32)

    mean = ct.load(saved_mean_ptr, index=(cblk,), shape=(C_BLOCK,))
    invstd = ct.load(invstd_ptr, index=(cblk,), shape=(C_BLOCK,))
    weight = ct.load(weight_ptr, index=(cblk,), shape=(C_BLOCK,))
    bias = ct.load(bias_ptr, index=(cblk,), shape=(C_BLOCK,))
    mean_2d = ct.reshape(mean, (1, C_BLOCK))
    invstd_2d = ct.reshape(invstd, (1, C_BLOCK))
    weight_2d = ct.reshape(weight, (1, C_BLOCK))
    bias_2d = ct.reshape(bias, (1, C_BLOCK))

    centered = x_f - mean_2d
    normalized = centered * invstd_2d
    scaled = normalized * weight_2d
    affine = scaled + bias_2d
    affine_bf16 = ct.astype(affine, ct.bfloat16)
    zero_bf16 = ct.full((HW_PAD, C_BLOCK), 0.0, dtype=ct.bfloat16)
    relu = ct.where(affine_bf16 < zero_bf16, zero_bf16, affine_bf16)

    hw_arange = ct.arange(HW_PAD, dtype=ct.int32)
    hw_valid = hw_arange < HW
    hw_valid_2d = ct.reshape(hw_valid, (HW_PAD, 1))
    zero_f = ct.full((HW_PAD, C_BLOCK), 0.0, dtype=ct.float32)
    relu_f = ct.astype(relu, ct.float32)
    relu_masked = ct.where(hw_valid_2d, relu_f, zero_f)
    pooled = ct.sum(relu_masked, axis=0, keepdims=True) * INV_HW
    pooled_bf16 = ct.astype(pooled, ct.bfloat16)   # shape (1, C_BLOCK)

    # Dropout on the per-(n, c) pooled tile
    random_row = ct.load(random_ptr, index=(n, cblk), shape=(1, C_BLOCK))
    threshold = ct.full((1, C_BLOCK), 0.8, dtype=ct.float32)
    lt_mask = random_row < threshold
    ct.store(lt_ptr, index=(n, cblk), tile=lt_mask)

    zero_bf_1 = ct.full((1, C_BLOCK), 0.0, dtype=ct.bfloat16)
    one_bf_1 = ct.full((1, C_BLOCK), 1.0, dtype=ct.bfloat16)
    keep_bf = ct.where(lt_mask, one_bf_1, zero_bf_1)
    div_bf = ct.astype(
        ct.astype(keep_bf, ct.float32) * SCALE,
        ct.bfloat16,
    )
    dropped = ct.astype(
        ct.astype(pooled_bf16, ct.float32) * ct.astype(div_bf, ct.float32),
        ct.bfloat16,
    )
    ct.store(out_ptr, index=(n, cblk), tile=dropped)


@oracle_impl(hardware="B200", point="441beb73")
def oracle_forward(inputs):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, _shape_param_0 = inputs
    n = int(arg0_1.shape[0])
    c = int(arg0_1.shape[1])
    h = int(arg0_1.shape[2])
    w = int(arg0_1.shape[3])
    hw = h * w
    e = n * hw
    device = arg0_1.device

    BLOCK_E = 256
    C_BLOCK = 32
    num_chunks = (e + BLOCK_E - 1) // BLOCK_E
    block_chunks = _next_power_of_2(num_chunks)
    assert c % C_BLOCK == 0

    # NCHW-contiguous. permute(0,2,3,1) -> NHWC logical then reshape to [E, C]
    x_flat = arg0_1.permute(0, 2, 3, 1).reshape(e, c).contiguous()

    partial_sum = torch.empty((num_chunks, c), device=device, dtype=torch.float32)
    partial_sq = torch.empty((num_chunks, c), device=device, dtype=torch.float32)
    saved_mean = torch.empty_strided(
        (1, c, 1, 1), (c, 1, 1, 1), device=device, dtype=torch.float32,
    )
    invstd = torch.empty_strided(
        (1, c, 1, 1), (c, 1, 1, 1), device=device, dtype=torch.float32,
    )

    stream = torch.cuda.current_stream()

    # 1) Partial reduction
    ct.launch(
        stream,
        (num_chunks, c // C_BLOCK, 1),
        _bn_partial_stats_kernel,
        (x_flat, partial_sum, partial_sq, e, BLOCK_E, C_BLOCK),
    )

    # 2) Finalize
    saved_mean_1d = saved_mean.view(c)
    invstd_1d = invstd.view(c)
    ct.launch(
        stream,
        (c // C_BLOCK, 1, 1),
        _bn_finalize_stats_kernel,
        (partial_sum, partial_sq, arg1_1, arg2_1, saved_mean_1d, invstd_1d,
         num_chunks, block_chunks, C_BLOCK,
         1.0 / float(e), EPS, UNBIAS_CORRECTION, MOMENTUM, ONE_MINUS_MOMENTUM),
    )

    # 3) Padded input for tile-based spatial mean, plus dropout using
    # inductor_seeds like the Repro does.
    HW_PAD = _next_power_of_2(hw)
    padded_x = torch.zeros((n, HW_PAD, c), device=device, dtype=torch.bfloat16)
    padded_x[:, :hw, :].copy_(x_flat.view(n, hw, c))
    padded_x_2d = padded_x.view(n * HW_PAD, c)

    # rsqrt output = invstd side output as-is
    rsqrt_out = invstd  # shape (1, c, 1, 1) f32

    # inductor_seeds is drawn as needed
    inductor_seeds = torch.ops.prims.inductor_seeds.default(1, device)
    seed = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 0)
    random = torch.ops.prims.inductor_random.default((n, c), seed, "rand")

    lt = torch.empty((n, c), device=device, dtype=torch.bool)
    dropped_out = torch.empty((n, c), device=device, dtype=torch.bfloat16)

    ct.launch(
        stream,
        (n, c // C_BLOCK, 1),
        _bn_relu_spatial_mean_dropout_kernel,
        (padded_x_2d, arg3_1, arg4_1, saved_mean_1d, invstd_1d,
         random, lt, dropped_out,
         hw, 1.0 / float(hw), HW_PAD, C_BLOCK, DROPOUT_SCALE),
    )

    return saved_mean, rsqrt_out, lt, dropped_out, arg1_1, arg2_1
