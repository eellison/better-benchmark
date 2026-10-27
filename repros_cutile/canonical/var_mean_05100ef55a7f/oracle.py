"""cuTile port of var_mean_05100ef55a7f: MegatronBERT seeded dropout + LayerNorm.

Returns (gt, add, mul_2, view_1, div) where:
  gt: dropout keep mask
  add: residual + scaled_dropout (f32)
  mul_2: normalized (f32) = (add - mean) * rsqrt
  view_1: bf16 affine output = (normalized * scale + bias)
  div: rsqrt / HIDDEN
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX = 43
BATCH = 16
SEQ = 512
ROWS = BATCH * SEQ
HIDDEN = 1024
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.1111111111111112
EPS = 1.0e-12


@ct.kernel
def _dropout_layernorm_kernel(
    flat_ptr,      # bf16 [rows, HIDDEN]
    random_ptr,    # f32 [rows, HIDDEN]
    residual_ptr,  # f32 [rows, HIDDEN]
    weight_ptr,    # f32 [HIDDEN]
    bias_ptr,      # f32 [HIDDEN]
    gt_ptr,        # b8 [rows, HIDDEN]
    add_ptr,       # f32 [rows, HIDDEN]
    norm_ptr,      # f32 [rows, HIDDEN] mul_2
    out_ptr,       # bf16 [rows, HIDDEN]
    div_ptr,       # f32 [rows]
    HIDDEN_C: ct.Constant[int],
):
    row = ct.bid(0)

    flat_bf = ct.load(flat_ptr, index=(row, 0), shape=(1, HIDDEN_C))
    rand_f = ct.load(random_ptr, index=(row, 0), shape=(1, HIDDEN_C))
    residual = ct.load(residual_ptr, index=(row, 0), shape=(1, HIDDEN_C))

    rand_bf = ct.astype(rand_f, ct.bfloat16)
    thresh_bf = ct.full((1, HIDDEN_C), DROPOUT_P, dtype=ct.bfloat16)
    keep = rand_bf > thresh_bf
    ct.store(gt_ptr, index=(row, 0), tile=keep)

    zero_bf = ct.full((1, HIDDEN_C), 0.0, dtype=ct.bfloat16)
    dropped_bf = ct.where(keep, flat_bf, zero_bf)
    scaled_bf = ct.astype(
        ct.astype(dropped_bf, ct.float32) * DROPOUT_SCALE, ct.bfloat16
    )

    add_val = residual + ct.astype(scaled_bf, ct.float32)
    ct.store(add_ptr, index=(row, 0), tile=add_val)

    mean_val = ct.sum(add_val) * (1.0 / HIDDEN_C)
    centered = add_val - mean_val
    variance = ct.sum(centered * centered) * (1.0 / HIDDEN_C)
    invstd = ct.rsqrt(variance + EPS)

    normalized = centered * invstd
    ct.store(norm_ptr, index=(row, 0), tile=normalized)

    weight = ct.load(weight_ptr, index=(0,), shape=(HIDDEN_C,))
    bias = ct.load(bias_ptr, index=(0,), shape=(HIDDEN_C,))
    weight_2d = ct.reshape(weight, (1, HIDDEN_C))
    bias_2d = ct.reshape(bias, (1, HIDDEN_C))
    affine = normalized * weight_2d + bias_2d
    ct.store(out_ptr, index=(row, 0), tile=ct.astype(affine, ct.bfloat16))

    div_val = invstd * (1.0 / HIDDEN_C)
    ct.store(div_ptr, index=(row,),
             tile=ct.reshape(ct.full((1,), div_val, dtype=ct.float32), (1,)))


def _state_u64(state, start):
    return int.from_bytes(bytes(state[start : start + 8].tolist()), "little")


def _put_state_u64(state, start, value):
    state[start : start + 8] = torch.tensor(
        list(int(value).to_bytes(8, "little", signed=False)),
        dtype=state.dtype,
        device=state.device,
    )


def _inductor_random_for_eager_check(shape, seed, *, device):
    numel = 1
    for dim in shape:
        numel *= int(dim)
    props = torch.cuda.get_device_properties(device)
    block_size = 256
    unroll = 4
    curand4_engine_calls = 4
    blocks_per_sm = props.max_threads_per_multi_processor // block_size
    grid = min(
        (numel + block_size - 1) // block_size,
        props.multi_processor_count * blocks_per_sm,
    )
    advance = (
        ((numel - 1) // (block_size * grid * unroll) + 1)
        * curand4_engine_calls
        * 2
    )
    state = torch.cuda.get_rng_state(device)
    offset = _state_u64(state, 8)
    if offset >= advance:
        rewound = state.clone()
        _put_state_u64(rewound, 8, offset - advance)
        torch.cuda.set_rng_state(rewound, device)
        random = torch.ops.prims.inductor_random.default(shape, seed, "rand")
        torch.cuda.set_rng_state(state, device)
        return random
    return torch.ops.prims.inductor_random.default(shape, seed, "rand")


def _shape_tuple(shape):
    return tuple(int(dim) for dim in shape)


@oracle_impl(hardware="B200", point="cfc55f11", BLOCK_H=1024)
def oracle_forward(inputs, *, BLOCK_H: int):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, _shape0, shape1, _shape2 = inputs

    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        raise NotImplementedError(
            "cuTile port unsupported inside CUDA graph capture (seeded RNG)."
        )

    random_shape = _shape_tuple(shape1)
    device = arg0_1.device

    gt = torch.empty((BATCH, SEQ, HIDDEN), device=device, dtype=torch.bool)
    add = torch.empty((BATCH, SEQ, HIDDEN), device=device, dtype=torch.float32)
    normalized = torch.empty((BATCH, SEQ, HIDDEN), device=device, dtype=torch.float32)
    out = torch.empty((ROWS, HIDDEN), device=device, dtype=torch.bfloat16)
    div = torch.empty((BATCH, SEQ, 1), device=device, dtype=torch.float32)

    seed = torch.ops.prims.inductor_lookup_seed.default(arg1_1, SEED_INDEX)
    random = _inductor_random_for_eager_check(random_shape, seed, device=device)

    flat_2d = arg0_1.view(ROWS, HIDDEN)
    residual_2d = arg2_1.contiguous().view(ROWS, HIDDEN)
    random_2d = random.contiguous().view(ROWS, HIDDEN)
    gt_2d = gt.view(ROWS, HIDDEN)
    add_2d = add.view(ROWS, HIDDEN)
    normalized_2d = normalized.view(ROWS, HIDDEN)
    out_2d = out.view(ROWS, HIDDEN)
    div_1d = div.view(ROWS)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ROWS, 1, 1),
        _dropout_layernorm_kernel,
        (flat_2d, random_2d, residual_2d, arg3_1, arg4_1,
         gt_2d, add_2d, normalized_2d, out_2d, div_1d, HIDDEN),
    )
    return gt, add, normalized, out, div
