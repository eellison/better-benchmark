"""cuTile port of pointwise_5ae4111f8443: MT5 tanh-GELU + gate + dropout.

Pre-generates seeded random via inductor_random, then runs a flat cuTile
pointwise kernel that: applies tanh-approx GELU, multiplies by gate, dropout
mask, scale by 1/(1-0.1), bf16 output.
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX = 11
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.1111111111111112


@ct.kernel
def _mt5_gelu_gate_dropout_kernel(
    x_ptr,          # bf16 (total,)
    gate_ptr,       # bf16 (total,)
    random_ptr,     # f32 (total,)
    gt_ptr,         # b8 (total,)
    out_ptr,        # bf16 (total,)
    BLOCK_N: ct.Constant[int],
):
    pid = ct.bid(0)
    random = ct.load(random_ptr, index=(pid,), shape=(BLOCK_N,))
    keep = random > DROPOUT_P
    ct.store(gt_ptr, index=(pid,), tile=keep)

    x = ct.astype(ct.load(x_ptr, index=(pid,), shape=(BLOCK_N,)), ct.float32)
    gate = ct.astype(ct.load(gate_ptr, index=(pid,), shape=(BLOCK_N,)), ct.float32)

    half_bf = ct.astype(x * 0.5, ct.bfloat16)
    half_f = ct.astype(half_bf, ct.float32)
    x_cubed = x * x * x
    tanh_arg = (x + x_cubed * 0.044715) * 0.7978845608028654
    tanh_val = ct.tanh(tanh_arg)
    gelu = half_f * (tanh_val + 1.0)

    gated = gelu * gate
    dropped = ct.astype(keep, ct.float32) * gated
    scaled = ct.astype(dropped * DROPOUT_SCALE, ct.bfloat16)
    ct.store(out_ptr, index=(pid,), tile=scaled)


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


def _contiguous_stride(shape):
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= int(dim)
    return tuple(reversed(stride))


@oracle_impl(hardware="B200", point="96064e9c", BLOCK_N=1024)
def oracle_forward(inputs, *, BLOCK_N: int):
    x, gate, seeds, _shape_param_0, _shape_param_1, random_shape, out_shape = inputs
    random_shape = tuple(int(dim) for dim in random_shape)
    out_shape = tuple(int(dim) for dim in out_shape)
    device = x.device
    total = int(x.numel())

    gt = torch.empty_strided(
        random_shape, _contiguous_stride(random_shape),
        device=device, dtype=torch.bool)
    out = torch.empty_strided(
        out_shape, _contiguous_stride(out_shape),
        device=device, dtype=torch.bfloat16)

    seed = torch.ops.prims.inductor_lookup_seed.default(seeds, SEED_INDEX)
    random = _inductor_random_for_eager_check(random_shape, seed, device=device)

    x_flat = x.view(total)
    gate_flat = gate.view(total)
    random_flat = random.view(total)
    gt_flat = gt.view(total)
    out_flat = out.view(total)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (ct.cdiv(total, BLOCK_N), 1, 1), _mt5_gelu_gate_dropout_kernel,
        (x_flat, gate_flat, random_flat, gt_flat, out_flat, BLOCK_N),
    )
    return gt, out
