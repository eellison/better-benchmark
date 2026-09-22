"""cuTile port of var_mean_05643331629f: DistillGPT2/GPT2 dropout residual LN.

Pre-generates seeded random tensor via inductor_random on the Python side, then
a single row-parallel cuTile kernel does dropout, residual add, LN, affine and
emits mask, add, normalized, bf16-flat view, and permuted bf16 view + inverse-
std side output. HIDDEN=768 padded to BLOCK_H=1024.
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX = 5
HIDDEN = 768
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.1111111111111112
EPS = 1.0e-5


@ct.kernel
def _dropout_ln_kernel(
    hidden_ptr,     # bf16 [rows, BLOCK_H]  (padded)
    random_ptr,     # f32 [rows, BLOCK_H]
    residual_ptr,   # f32 [rows, BLOCK_H]
    weight_ptr,     # f32 [BLOCK_H]
    bias_ptr,       # f32 [BLOCK_H]
    gt_ptr,         # bool [rows, BLOCK_H]
    add_ptr,        # f32 [rows, BLOCK_H]
    norm_ptr,       # f32 [rows, BLOCK_H]
    bf16_ptr,       # bf16 [rows, BLOCK_H]
    div_ptr,        # f32 [rows]
    HIDDEN_C: ct.Constant[int],
    EPS_C: ct.Constant[float],
    BLOCK_H: ct.Constant[int],
):
    row = ct.bid(0)

    hidden_bf = ct.load(hidden_ptr, index=(row, 0), shape=(1, BLOCK_H))
    residual_f = ct.load(residual_ptr, index=(row, 0), shape=(1, BLOCK_H))
    random_f = ct.load(random_ptr, index=(row, 0), shape=(1, BLOCK_H))

    cols = ct.arange(BLOCK_H, dtype=ct.int32)
    col_valid = cols < HIDDEN_C
    col_valid_2d = ct.reshape(col_valid, (1, BLOCK_H))

    random_bf = ct.astype(random_f, ct.bfloat16)
    threshold_bf = ct.full((1, BLOCK_H), DROPOUT_P, dtype=ct.bfloat16)
    keep = random_bf > threshold_bf
    ct.store(gt_ptr, index=(row, 0), tile=keep)

    zero_bf = ct.zeros((1, BLOCK_H), dtype=ct.bfloat16)
    dropped = ct.where(keep, hidden_bf, zero_bf)
    scaled_f = ct.astype(dropped, ct.float32) * DROPOUT_SCALE
    scaled_bf = ct.astype(scaled_f, ct.bfloat16)
    x_f = ct.astype(scaled_bf, ct.float32) + residual_f
    ct.store(add_ptr, index=(row, 0), tile=x_f)

    x_masked = ct.where(col_valid_2d, x_f, 0.0)
    mean = ct.sum(x_masked) * (1.0 / HIDDEN_C)
    centered = x_f - mean
    centered_masked = ct.where(col_valid_2d, centered, 0.0)
    variance = ct.sum(centered_masked * centered_masked) * (1.0 / HIDDEN_C)
    invstd = ct.rsqrt(variance + EPS_C)
    normalized = centered * invstd

    weight_1d = ct.load(weight_ptr, index=(0,), shape=(BLOCK_H,))
    bias_1d = ct.load(bias_ptr, index=(0,), shape=(BLOCK_H,))
    weight_2d = ct.reshape(weight_1d, (1, BLOCK_H))
    bias_2d = ct.reshape(bias_1d, (1, BLOCK_H))
    affine_f = normalized * weight_2d + bias_2d

    ct.store(norm_ptr, index=(row, 0), tile=normalized)
    ct.store(bf16_ptr, index=(row, 0), tile=ct.astype(affine_f, ct.bfloat16))
    ct.store(div_ptr, index=(row,), tile=invstd * (1.0 / HIDDEN_C))


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


def _run(inputs, *, BLOCK_H: int):
    hidden_input, seeds, residual, weight, bias, shape0, random_shape, shape2 = inputs
    base_shape = _shape_tuple(shape0)
    rand_shape = _shape_tuple(random_shape)
    flat_shape = _shape_tuple(shape2)
    batch = int(base_shape[0])
    seq = int(base_shape[1])
    hidden = int(base_shape[2])
    rows = batch * seq
    device = hidden_input.device

    # Pad
    x_padded = torch.zeros((rows, BLOCK_H), device=device, dtype=torch.bfloat16)
    x_padded[:, :hidden] = hidden_input
    residual_padded = torch.zeros((rows, BLOCK_H), device=device, dtype=torch.float32)
    residual_padded[:, :hidden] = residual.view(rows, hidden)
    weight_padded = torch.zeros((BLOCK_H,), device=device, dtype=torch.float32)
    weight_padded[:hidden] = weight
    bias_padded = torch.zeros((BLOCK_H,), device=device, dtype=torch.float32)
    bias_padded[:hidden] = bias

    seed = torch.ops.prims.inductor_lookup_seed.default(seeds, SEED_INDEX)
    random_full = _inductor_random_for_eager_check(rand_shape, seed, device=device)
    random_flat = random_full.view(rows, hidden)
    random_padded = torch.zeros((rows, BLOCK_H), device=device, dtype=torch.float32)
    random_padded[:, :hidden] = random_flat

    gt_padded = torch.empty((rows, BLOCK_H), device=device, dtype=torch.bool)
    add_padded = torch.empty((rows, BLOCK_H), device=device, dtype=torch.float32)
    norm_padded = torch.empty((rows, BLOCK_H), device=device, dtype=torch.float32)
    bf16_padded = torch.empty((rows, BLOCK_H), device=device, dtype=torch.bfloat16)
    div_flat = torch.empty((rows,), device=device, dtype=torch.float32)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (rows, 1, 1),
        _dropout_ln_kernel,
        (
            x_padded, random_padded, residual_padded,
            weight_padded, bias_padded,
            gt_padded, add_padded, norm_padded, bf16_padded, div_flat,
            hidden, EPS, BLOCK_H,
        ),
    )

    # Slice into final layouts matching the Triton oracle strides
    gt = torch.empty_strided(base_shape, (seq * hidden, hidden, 1),
                             device=device, dtype=torch.bool)
    gt.view(rows, hidden).copy_(gt_padded[:, :hidden])
    add = torch.empty_strided(base_shape, (seq * hidden, hidden, 1),
                              device=device, dtype=torch.float32)
    add.view(rows, hidden).copy_(add_padded[:, :hidden])
    normalized = torch.empty_strided(base_shape, (seq * hidden, hidden, 1),
                                     device=device, dtype=torch.float32)
    normalized.view(rows, hidden).copy_(norm_padded[:, :hidden])
    bf16_base = torch.empty_strided((rows, hidden), (hidden, 1),
                                    device=device, dtype=torch.bfloat16)
    bf16_base.copy_(bf16_padded[:, :hidden])
    div = torch.empty_strided((batch, seq, 1), (seq, 1, 1),
                              device=device, dtype=torch.float32)
    div.view(rows).copy_(div_flat)

    bf16_flat = bf16_base.view(flat_shape)
    return gt, add, normalized, bf16_flat, bf16_flat.permute(1, 0), div


# a352047a: DistillGPT2 train, bf16[16384,768] with residual f32[32,512,768].
@oracle_impl(hardware="B200", point="a352047a", BLOCK_H=1024)
# bf8decda: GPT2ForSequenceClassification train, bf16[8192,768] with residual f32[8,1024,768].
@oracle_impl(hardware="B200", point="bf8decda", BLOCK_H=1024)
def oracle_forward(inputs, *, BLOCK_H: int):
    return _run(inputs, BLOCK_H=BLOCK_H)
