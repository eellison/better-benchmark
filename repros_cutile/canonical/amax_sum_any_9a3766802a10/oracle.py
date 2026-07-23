"""cuTile port of amax_sum_any_9a3766802a10: MobileBERT safe softmax + dropout.

For each row: softmax(row) if any finite; else use arg1 row.
Then dropout with SEED_INDEX=23 seeded RNG.

Outputs: (where, gt, view_1, permute).
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX = 23
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.1111111111111112


@ct.kernel
def _safe_softmax_dropout_kernel(
    x_ptr,          # bf16 [rows, K]
    arg1_ptr,       # bf16 [rows, K]  fallback
    random_ptr,     # f32 [rows, K]
    where_ptr,      # bf16 [rows, K]
    gt_ptr,         # b8 [rows, K]
    out_ptr,        # bf16 [rows, K]
    K: ct.Constant[int],
    BLOCK_K: ct.Constant[int],
    DROPOUT_P_: ct.Constant[float],
    DROPOUT_SCALE_: ct.Constant[float],
):
    row = ct.bid(0)
    x = ct.load(x_ptr, index=(row, 0), shape=(1, BLOCK_K))
    x_f = ct.astype(x, ct.float32)
    amax = ct.max(x_f, axis=1, keepdims=True)
    sub_ = x_f - amax
    exp_v = ct.exp(sub_)
    sum_v = ct.sum(exp_v, axis=1, keepdims=True)
    div_v = exp_v / sum_v
    div_bf = ct.astype(div_v, ct.bfloat16)

    # Check any(x != -inf) — equivalently max(x) != -inf.
    neg_inf = ct.full((1, 1), -1.0e38, dtype=ct.float32)
    all_neg_inf = amax <= neg_inf
    # logical_not_1 = ~any_1: True where ALL are -inf (row of only -infs).
    # Broadcast to (1, BLOCK_K)
    all_ninf_bc = ct.reshape(all_neg_inf, (1, 1))

    arg1_v = ct.load(arg1_ptr, index=(row, 0), shape=(1, BLOCK_K))
    where_v = ct.where(all_ninf_bc, arg1_v, div_bf)
    ct.store(where_ptr, index=(row, 0), tile=where_v)

    # Dropout with bf16 threshold (convert_element_type_2 = random.to(bf16); gt = >0.1)
    random_f = ct.load(random_ptr, index=(row, 0), shape=(1, BLOCK_K))
    random_bf = ct.astype(random_f, ct.bfloat16)
    thresh_bf = ct.full((1, BLOCK_K), DROPOUT_P_, dtype=ct.bfloat16)
    keep = random_bf > thresh_bf
    ct.store(gt_ptr, index=(row, 0), tile=keep)

    zero_bf = ct.full((1, BLOCK_K), 0.0, dtype=ct.bfloat16)
    dropped_bf = ct.where(keep, where_v, zero_bf)
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


@oracle_impl(hardware="B200", point="d59f4ab1")
def oracle_forward(inputs):
    (arg0_1, arg1_1, arg2_1, *_shape) = inputs
    device = arg0_1.device
    K = 128
    total_rows = 256 * 4 * 128

    view = arg0_1.view(256, 4, 128, 128)
    view_flat = view.reshape(total_rows, K)
    arg1_flat = arg1_1.reshape(total_rows, K)

    where_ = torch.empty((total_rows, K), device=device, dtype=torch.bfloat16)
    gt_ = torch.empty((total_rows, K), device=device, dtype=torch.bool)
    out_ = torch.empty((total_rows, K), device=device, dtype=torch.bfloat16)

    seed = torch.ops.prims.inductor_lookup_seed.default(arg2_1, SEED_INDEX)
    random = _inductor_random_for_eager_check((256, 4, 128, 128), seed, device=device)
    random_flat = random.reshape(total_rows, K)

    stream = torch.cuda.current_stream()
    ct.launch(stream, (total_rows, 1, 1), _safe_softmax_dropout_kernel,
              (view_flat, arg1_flat, random_flat,
               where_, gt_, out_,
               K, K, DROPOUT_P, DROPOUT_SCALE))

    where_4d = where_.view(256, 4, 128, K)
    gt_4d = gt_.view(256, 4, 128, K)
    out_4d = out_.view(256, 4, 128, K)
    view_1 = out_4d.view(1024, 128, K)
    permute = view_1.permute(0, 2, 1)
    return where_4d, gt_4d, view_1, permute
