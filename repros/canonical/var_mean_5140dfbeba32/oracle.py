"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete bf16 dropout-residual LayerNorm scope in one Triton row kernel, including seed-index-29 Inductor RNG, bf16 random-mask rounding, bf16 dropout scaling, the fp32 residual add, population var_mean over the hidden dimension with eps=1e-7, the normalized and affine fp32 side outputs, the bf16 flattened final view, and the rsqrt/hidden side output, whereas Inductor currently schedules the stochastic dropout producer, row-normalization template, and multiple live output epilogues through generic fusion boundaries; Inductor cannot do this today because the normalization scheduler does not sink an Inductor RNG producer with required bf16 cast boundaries and all live consumers into one fixed-hidden row-normalization plan; the fix is SCHEDULER_FUSION: teach the norm scheduler to fuse seeded dropout/residual producers with normalized, affine, bf16-view, mask, and inverse-std side-output stores while preserving the captured dtype boundaries."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


SEQ_LEN = 512
SEED_INDEX = 29
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.1111111111111112
EPS = 1.0e-7


@triton.jit
def _dropout_residual_layernorm_full_kernel(
    src_ptr,
    seeds_ptr,
    random_ptr,
    residual_ptr,
    weight_ptr,
    bias_ptr,
    mask_ptr,
    normalized_ptr,
    affine_ptr,
    out_bf16_ptr,
    side_ptr,
    hidden: tl.constexpr,
    total_rows: tl.constexpr,
    block_h: tl.constexpr,
    row_block: tl.constexpr,
    seed_index: tl.constexpr,
    dropout_p: tl.constexpr,
    dropout_scale: tl.constexpr,
    eps: tl.constexpr,
    use_random_ptr: tl.constexpr,
):
    row_ids = tl.program_id(0) * row_block + tl.arange(0, row_block)
    cols = tl.arange(0, block_h)
    offsets = row_ids[:, None] * hidden + cols[None, :]
    row_mask = row_ids[:, None] < total_rows
    col_mask = cols[None, :] < hidden
    valid = row_mask & col_mask

    src = tl.load(src_ptr + offsets, mask=valid, other=0.0).to(tl.float32)
    residual = tl.load(residual_ptr + offsets, mask=valid, other=0.0).to(tl.float32)

    if use_random_ptr:
        random = tl.load(random_ptr + offsets, mask=valid, other=0.0).to(tl.bfloat16)
    else:
        seed = tl.load(seeds_ptr + seed_index)
        random = tl.rand(seed, offsets.to(tl.uint32)).to(tl.bfloat16)
    threshold = tl.full((row_block, block_h), dropout_p, tl.float32).to(tl.bfloat16)
    keep = random > threshold
    dropped = tl.where(keep, src * dropout_scale, 0.0).to(tl.bfloat16).to(tl.float32)
    x = residual + dropped

    x_for_sum = tl.where(valid, x, 0.0)
    mean_vec = tl.sum(x_for_sum, axis=1) / hidden
    centered = x - mean_vec[:, None]
    centered_for_var = tl.where(valid, centered, 0.0)
    var_vec = tl.sum(centered_for_var * centered_for_var, axis=1) / hidden
    invstd_vec = tl.rsqrt(var_vec + eps)
    normalized = centered * invstd_vec[:, None]

    weight = tl.load(weight_ptr + cols, mask=cols < hidden, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + cols, mask=cols < hidden, other=0.0).to(tl.float32)
    affine = normalized * weight[None, :] + bias[None, :]

    tl.store(mask_ptr + offsets, keep, mask=valid)
    tl.store(normalized_ptr + offsets, normalized, mask=valid)
    tl.store(affine_ptr + offsets, affine, mask=valid)
    tl.store(out_bf16_ptr + offsets, affine, mask=valid)
    tl.store(side_ptr + row_ids, invstd_vec / hidden, mask=row_ids < total_rows)


@oracle_impl(
    hardware="B200",
    point="55aa5fd0",
    HIDDEN=1536,
    ROWS=4096,
    BATCH=8,
    BLOCK_H=2048,
    ROW_BLOCK=1,
    RNG_OFFSET_DELTA=48,
    num_warps=8,
    num_stages=3,
)
@oracle_impl(
    hardware="B200",
    point="243d7832",
    HIDDEN=768,
    ROWS=16384,
    BATCH=32,
    BLOCK_H=1024,
    ROW_BLOCK=1,
    RNG_OFFSET_DELTA=96,
    num_warps=4,
    num_stages=3,
)
@oracle_impl(
    hardware="B200",
    point="d9ecc504",
    HIDDEN=256,
    ROWS=32768,
    BATCH=64,
    BLOCK_H=256,
    ROW_BLOCK=1,
    RNG_OFFSET_DELTA=64,
    num_warps=4,
    num_stages=3,
)
def oracle_forward(
    inputs,
    *,
    HIDDEN,
    ROWS,
    BATCH,
    BLOCK_H,
    ROW_BLOCK,
    RNG_OFFSET_DELTA,
    num_warps,
    num_stages,
):
    src, seeds, residual, weight, bias, shape0, _shape1, shape2 = inputs
    mask = torch.empty_strided(
        tuple(shape0),
        (SEQ_LEN * HIDDEN, HIDDEN, 1),
        device=src.device,
        dtype=torch.bool,
    )
    normalized = torch.empty_strided(
        tuple(shape0),
        (SEQ_LEN * HIDDEN, HIDDEN, 1),
        device=src.device,
        dtype=torch.float32,
    )
    affine = torch.empty_strided(
        tuple(shape0),
        (SEQ_LEN * HIDDEN, HIDDEN, 1),
        device=src.device,
        dtype=torch.float32,
    )
    out_bf16 = torch.empty_strided(
        tuple(shape2),
        (HIDDEN, 1),
        device=src.device,
        dtype=torch.bfloat16,
    )
    side = torch.empty_strided(
        (BATCH, SEQ_LEN, 1),
        (SEQ_LEN, 1, 1),
        device=src.device,
        dtype=torch.float32,
    )

    grid = (triton.cdiv(ROWS, ROW_BLOCK),)
    _dropout_residual_layernorm_full_kernel[grid](
        src,
        seeds,
        src,
        residual,
        weight,
        bias,
        mask,
        normalized,
        affine,
        out_bf16,
        side,
        hidden=HIDDEN,
        total_rows=ROWS,
        block_h=BLOCK_H,
        row_block=ROW_BLOCK,
        seed_index=SEED_INDEX,
        dropout_p=DROPOUT_P,
        dropout_scale=DROPOUT_SCALE,
        eps=EPS,
        use_random_ptr=False,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    if not torch.cuda.is_current_stream_capturing():
        state = torch.cuda.get_rng_state(src.device)
        previous_state = state.clone()
        offset = int.from_bytes(bytes(previous_state[8:16].cpu().tolist()), "little")
        previous_offset = (offset - RNG_OFFSET_DELTA) % (1 << 64)
        previous_state[8:16] = torch.tensor(
            list(previous_offset.to_bytes(8, "little")),
            dtype=torch.uint8,
        )
        torch.cuda.set_rng_state(previous_state, src.device)
        seed = torch.ops.prims.inductor_lookup_seed.default(seeds, SEED_INDEX)
        random = torch.ops.prims.inductor_random.default(tuple(shape0), seed, "rand")
        torch.cuda.set_rng_state(state, src.device)
        _dropout_residual_layernorm_full_kernel[grid](
            src,
            seeds,
            random,
            residual,
            weight,
            bias,
            mask,
            normalized,
            affine,
            out_bf16,
            side,
            hidden=HIDDEN,
            total_rows=ROWS,
            block_h=BLOCK_H,
            row_block=ROW_BLOCK,
            seed_index=SEED_INDEX,
            dropout_p=DROPOUT_P,
            dropout_scale=DROPOUT_SCALE,
            eps=EPS,
            use_random_ptr=True,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    return mask, normalized, affine, out_bf16, side
