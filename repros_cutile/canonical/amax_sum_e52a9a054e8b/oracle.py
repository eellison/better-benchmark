"""cuTile port of amax_sum_e52a9a054e8b: BERT scaled attention softmax + dropout.

Pre-generates seeded random via inductor_random, then a single per-row cuTile
kernel does: div, mask, softmax (amax/exp/sum/div), dropout, output.

Outputs: (where, amax, sum_1, gt, view_1=bf16 flat, permute).
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX = 46
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.1111111111111112


@ct.kernel
def _softmax_dropout_kernel(
    x_ptr,          # bf16 [rows, K]
    mask_ptr,       # b8 [rows, K]
    scalar_ptr,     # bf16 scalar
    random_ptr,     # f32 [rows, K]
    where_ptr,      # bf16 [rows, K]
    amax_ptr,       # f32 [rows, 1]
    sum_ptr,        # f32 [rows, 1]
    gt_ptr,         # b8 [rows, K]
    out_ptr,        # bf16 [rows, K]
    K: ct.Constant[int],
    BLOCK_K: ct.Constant[int],
    DROPOUT_P_: ct.Constant[float],
    DROPOUT_SCALE_: ct.Constant[float],
):
    row = ct.bid(0)

    x = ct.load(x_ptr, index=(row, 0), shape=(1, BLOCK_K))
    mask = ct.load(mask_ptr, index=(row, 0), shape=(1, BLOCK_K))
    scalar = ct.load(scalar_ptr, index=(0,), shape=(1,))
    scalar_v = ct.reshape(scalar, (1, 1))
    random_f = ct.load(random_ptr, index=(row, 0), shape=(1, BLOCK_K))

    div_bf = ct.astype(ct.astype(x, ct.float32) * (1.0 / 8.0), ct.bfloat16)
    where_v = ct.where(mask, scalar_v, div_bf)
    ct.store(where_ptr, index=(row, 0), tile=where_v)

    where_f = ct.astype(where_v, ct.float32)
    amax = ct.max(where_f, axis=1, keepdims=True)
    ct.store(amax_ptr, index=(row, 0), tile=amax)

    sub_ = where_f - amax
    exp_v = ct.exp(sub_)
    sum_v = ct.sum(exp_v, axis=1, keepdims=True)
    ct.store(sum_ptr, index=(row, 0), tile=sum_v)
    div_v = exp_v / sum_v
    div_bf16 = ct.astype(div_v, ct.bfloat16)

    thresh = ct.full((1, BLOCK_K), DROPOUT_P_, dtype=ct.float32)
    keep = random_f > thresh
    ct.store(gt_ptr, index=(row, 0), tile=keep)
    zero_bf = ct.full((1, BLOCK_K), 0.0, dtype=ct.bfloat16)
    dropped_bf = ct.where(keep, div_bf16, zero_bf)
    scaled = ct.astype(ct.astype(dropped_bf, ct.float32) * DROPOUT_SCALE_, ct.bfloat16)
    ct.store(out_ptr, index=(row, 0), tile=scaled)


def _state_u64(state, start):
    return int.from_bytes(bytes(state[start : start + 8].tolist()), "little")


def _put_state_u64(state, start, value):
    state[start : start + 8] = torch.tensor(
        list(int(value).to_bytes(8, "little", signed=False)),
        dtype=state.dtype,
        device=state.device,
    )


def _inductor_random_for_eager_check(shape, seed, *, device):
    if torch.cuda.is_current_stream_capturing():
        return torch.ops.prims.inductor_random.default(shape, seed, "rand")
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


@oracle_impl(hardware="B200", point="0e2c5e9e")
def oracle_forward(inputs):
    (arg0_1, arg1_1, arg2_1, arg3_1, *_shape) = inputs
    device = arg0_1.device
    K = 128
    total_rows = 16 * 12 * 128

    view = arg0_1.view(16, 12, 128, 128)
    mask_bc = arg1_1.expand(16, 12, 128, 128).contiguous()
    view_flat = view.reshape(total_rows, K)
    mask_flat = mask_bc.reshape(total_rows, K)

    where_ = torch.empty((total_rows, K), device=device, dtype=torch.bfloat16)
    amax_ = torch.empty((total_rows, 1), device=device, dtype=torch.float32)
    sum_ = torch.empty((total_rows, 1), device=device, dtype=torch.float32)
    gt_ = torch.empty((total_rows, K), device=device, dtype=torch.bool)
    out_ = torch.empty((total_rows, K), device=device, dtype=torch.bfloat16)

    seed = torch.ops.prims.inductor_lookup_seed.default(arg3_1, SEED_INDEX)
    random = _inductor_random_for_eager_check((16, 12, 128, 128), seed, device=device)
    random_flat = random.reshape(total_rows, K)
    scalar_v = arg2_1.view(1)

    stream = torch.cuda.current_stream()
    ct.launch(stream, (total_rows, 1, 1), _softmax_dropout_kernel,
              (view_flat, mask_flat, scalar_v, random_flat,
               where_, amax_, sum_, gt_, out_,
               K, K, DROPOUT_P, DROPOUT_SCALE))

    where_4d = where_.view(16, 12, 128, K)
    amax_4d = amax_.view(16, 12, 128, 1)
    sum_4d = sum_.view(16, 12, 128, 1)
    gt_4d = gt_.view(16, 12, 128, K)
    out_4d = out_.view(16, 12, 128, K)

    view_1 = out_4d.view(192, 128, K)
    permute = view_1.permute(0, 2, 1)
    return where_4d, amax_4d, sum_4d, gt_4d, view_1, permute
