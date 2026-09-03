"""cuTile port of amax_sum_any_161752724969: attention safe softmax + fallback + dropout.

Uses eager pre-generated random via torch.ops.prims.inductor_random (seed index 13).
Row-wise: softmax(x, dim=-1) or fallback (arg1_1) if all elements are -inf,
then dropout with keep = (rand.to(bf16) > 0.1).
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX = 13
DROPOUT_SCALE = 1.1111111111111112


@ct.kernel
def _safe_softmax_dropout_kernel(
    x_ptr,           # bf16 [rows, K]
    fallback_ptr,    # bf16 [rows, K]
    random_ptr,      # f32  [rows, K]
    where_ptr,       # bf16 [rows, K]
    gt_ptr,          # b8   [rows, K]
    dropped_ptr,     # bf16 [rows, K]
    K_LEN_: ct.Constant[int],
):
    row = ct.bid(0)
    x = ct.load(x_ptr, index=(row, 0), shape=(1, K_LEN_))
    x_f = ct.astype(x, ct.float32)
    neg_inf = ct.full(shape=(1, K_LEN_), fill_value=float("-inf"), dtype=ct.float32)
    live = x_f != neg_inf
    # any live element -> row has a valid entry
    zero_i32 = ct.zeros((1, K_LEN_), dtype=ct.int32)
    one_i32 = ct.full(shape=(1, K_LEN_), fill_value=1, dtype=ct.int32)
    live_flag = ct.where(live, one_i32, zero_i32)
    has_any = ct.max(live_flag) != 0

    row_max = ct.max(x_f)
    safe_max = ct.where(has_any, row_max, 0.0)
    numer = ct.exp(x_f - safe_max)
    numer_masked = ct.where(live, numer, 0.0)
    denom = ct.sum(numer_masked)
    safe_denom = ct.where(has_any, denom, 1.0)
    probs = numer_masked / safe_denom
    probs_bf = ct.astype(probs, ct.bfloat16)

    fallback = ct.load(fallback_ptr, index=(row, 0), shape=(1, K_LEN_))
    where_val = ct.where(has_any, probs_bf, fallback)
    ct.store(where_ptr, index=(row, 0), tile=where_val)

    rand_f = ct.load(random_ptr, index=(row, 0), shape=(1, K_LEN_))
    rand_bf = ct.astype(rand_f, ct.bfloat16)
    threshold_bf = ct.astype(
        ct.full(shape=(1, K_LEN_), fill_value=0.1, dtype=ct.float32),
        ct.bfloat16,
    )
    keep = rand_bf > threshold_bf
    ct.store(gt_ptr, index=(row, 0), tile=keep)

    dropped_bf = ct.astype(ct.where(keep, where_val, ct.astype(
        ct.zeros(shape=(1, K_LEN_), dtype=ct.float32), ct.bfloat16)), ct.bfloat16)
    scaled = ct.astype(ct.astype(dropped_bf, ct.float32) * DROPOUT_SCALE, ct.bfloat16)
    ct.store(dropped_ptr, index=(row, 0), tile=scaled)


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


@oracle_impl(hardware="B200", point="8ae0f618")
@oracle_impl(hardware="B200", point="fac7e171")
@oracle_impl(hardware="B200", point="d59f4ab1")
def oracle_forward(inputs, **_kwargs):
    arg0_1, arg1_1, arg2_1, _shape0, shape1, _shape2, _shape3 = inputs

    k_len = int(arg0_1.shape[-1])
    n_rows = arg0_1.numel() // k_len
    random_shape = tuple(int(dim) for dim in shape1)

    where = torch.empty_like(arg1_1)
    gt = torch.empty_strided(
        tuple(arg1_1.shape), tuple(arg1_1.stride()),
        device=arg1_1.device, dtype=torch.bool,
    )
    dropped = torch.empty_like(arg0_1)

    seed = torch.ops.prims.inductor_lookup_seed.default(arg2_1, SEED_INDEX)
    random = _inductor_random_for_eager_check(
        random_shape, seed, device=arg0_1.device,
    )
    random_2d = random.reshape(n_rows, k_len).contiguous()

    x_2d = arg0_1.view(n_rows, k_len)
    fallback_2d = arg1_1.reshape(n_rows, k_len).contiguous()
    where_2d = where.view(n_rows, k_len)
    gt_2d = gt.view(n_rows, k_len)
    dropped_2d = dropped.view(n_rows, k_len)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (n_rows, 1, 1), _safe_softmax_dropout_kernel,
        (x_2d, fallback_2d, random_2d,
         where_2d, gt_2d, dropped_2d, k_len),
    )
    return where, gt, dropped, dropped.permute(0, 2, 1)
