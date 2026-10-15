"""cuTile port of mean_var_a955dfb90471: BERT embedding + dropout + LayerNorm.

The seed tensor is generated fresh (inductor_seeds), so it's stochastic
across runs; the check_oracle harness detects and skips these outputs.

Outputs: (inductor_seeds, gt, mul_1=dropout_out, sqrt, sub, view=bf16 flat).
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_COUNT = 61
SEED_INDEX = 0
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.1111111111111112
DENOM_EPS = 1.0e-6


@ct.kernel
def _embedding_layernorm_kernel(
    add_1_ptr,      # f32 [ROWS, HIDDEN]  the pre-dropout add: embed + slice + embed_1
    random_ptr,     # f32 [ROWS, HIDDEN]
    weight_ptr,     # f32 [HIDDEN]
    bias_ptr,       # f32 [HIDDEN]
    gt_ptr,         # b8 [ROWS, HIDDEN]
    mul_1_ptr,      # f32 [ROWS, HIDDEN]  dropout_out
    sqrt_ptr,       # f32 [ROWS, 1]
    sub_ptr,        # f32 [ROWS, HIDDEN]
    view_ptr,       # bf16 [ROWS, HIDDEN]
    HIDDEN_: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
    ROW_BLOCK: ct.Constant[int],
    DROPOUT_P_: ct.Constant[float],
    DROPOUT_SCALE_: ct.Constant[float],
    DENOM_EPS_: ct.Constant[float],
):
    row_block = ct.bid(0)

    add_1 = ct.load(add_1_ptr, index=(row_block, 0), shape=(ROW_BLOCK, BLOCK_H))
    rand = ct.load(random_ptr, index=(row_block, 0), shape=(ROW_BLOCK, BLOCK_H))

    # dropout: gt = random > 0.1 (f32 direct); mul_1 = keep * add_1 * 1.111...
    thresh_f = ct.full((ROW_BLOCK, BLOCK_H), DROPOUT_P_, dtype=ct.float32)
    keep = rand > thresh_f
    ct.store(gt_ptr, index=(row_block, 0), tile=keep)

    zero_f = ct.full((ROW_BLOCK, BLOCK_H), 0.0, dtype=ct.float32)
    dropped = ct.where(keep, add_1, zero_f)
    mul_1 = dropped * DROPOUT_SCALE_
    ct.store(mul_1_ptr, index=(row_block, 0), tile=mul_1)

    # LayerNorm-style: mean, var (unbiased with correction=1.0), sqrt
    inv_h = 1.0 / HIDDEN_
    mean_1d = ct.sum(mul_1, axis=1, keepdims=True) * inv_h
    sub = mul_1 - mean_1d
    ct.store(sub_ptr, index=(row_block, 0), tile=sub)

    # Variance with Bessel correction (correction=1): sum((x-mean)^2)/(HIDDEN-1)
    var_1d = ct.sum(sub * sub, axis=1, keepdims=True) * (1.0 / (HIDDEN_ - 1))
    sqrt_1d = ct.sqrt(var_1d)
    ct.store(sqrt_ptr, index=(row_block, 0), tile=sqrt_1d)

    weight = ct.load(weight_ptr, index=(0,), shape=(BLOCK_H,))
    bias = ct.load(bias_ptr, index=(0,), shape=(BLOCK_H,))
    weight_2d = ct.reshape(weight, (1, BLOCK_H))
    bias_2d = ct.reshape(bias, (1, BLOCK_H))

    mul_2 = weight_2d * sub
    denom = sqrt_1d + DENOM_EPS_
    div = mul_2 / denom
    add_3 = div + bias_2d
    ct.store(view_ptr, index=(row_block, 0), tile=ct.astype(add_3, ct.bfloat16))


def _shape_tuple(shape):
    return tuple(int(dim) for dim in shape)


@oracle_impl(hardware="B200", point="4ea50a71", BLOCK_H=1024, ROW_BLOCK=1)
def oracle_forward(inputs, *, BLOCK_H: int, ROW_BLOCK: int):
    (arg0_1,  # f32 [20005, 768] word embedding table
     arg1_1,  # i64 [16, 128] word indices
     arg2_1,  # f32 [1, 512, 768] positional embedding
     arg3_1,  # f32 [3, 768] type embedding
     arg4_1,  # i64 [16, 128] type indices
     arg5_1,  # f32 [768] weight
     arg6_1,  # f32 [768] bias
     shape0,  # (16, 128, 768)
     shape1,  # (2048, 768)
     ) = inputs
    base_shape = _shape_tuple(shape0)
    flat_shape = _shape_tuple(shape1)
    batch = int(base_shape[0])
    seq = int(base_shape[1])
    hidden = int(base_shape[2])
    rows = batch * seq
    device = arg0_1.device

    # Precompute add_1 via torch ops: embedding + slice + embedding_1
    embed = torch.ops.aten.embedding.default(arg0_1, arg1_1, 0)
    slice_1 = arg2_1[:, :seq, :]
    add_e = embed + slice_1
    embed_1 = torch.ops.aten.embedding.default(arg3_1, arg4_1, 0)
    add_1 = add_e + embed_1
    add_1_2d = add_1.contiguous().view(rows, hidden)

    # inductor_seeds — this is stochastic. Generate fresh each call.
    inductor_seeds = torch.ops.prims.inductor_seeds.default(SEED_COUNT, device)
    seed = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, SEED_INDEX)
    random = torch.ops.prims.inductor_random.default(
        base_shape, seed, "rand")
    random_2d = random.contiguous().view(rows, hidden)

    gt = torch.empty_strided(base_shape, (seq * hidden, hidden, 1),
                             device=device, dtype=torch.bool)
    mul_1 = torch.empty_strided(base_shape, (seq * hidden, hidden, 1),
                                device=device, dtype=torch.float32)
    sqrt = torch.empty_strided((batch, seq, 1), (seq, 1, 1),
                               device=device, dtype=torch.float32)
    sub = torch.empty_strided(base_shape, (seq * hidden, hidden, 1),
                              device=device, dtype=torch.float32)
    view_bf = torch.empty_strided((rows, hidden), (hidden, 1),
                                  device=device, dtype=torch.bfloat16)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(rows, ROW_BLOCK), 1, 1),
        _embedding_layernorm_kernel,
        (add_1_2d, random_2d, arg5_1, arg6_1,
         gt.view(rows, hidden),
         mul_1.view(rows, hidden),
         sqrt.view(rows, 1),
         sub.view(rows, hidden),
         view_bf,
         hidden, BLOCK_H, ROW_BLOCK, DROPOUT_P, DROPOUT_SCALE, DENOM_EPS),
    )

    view = view_bf.view(flat_shape)
    return inductor_seeds, gt, mul_1, sqrt, sub, view
