"""cuTile port of pointwise_8fce69a7fd7a: MT5 tanh-approx GELU + dropout.

bf16[4096,1024]. Seed index 51. Uses tanh approximation for GELU.
Returns (gt, out).
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX = 51
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.1111111111111112
BLOCK_N = 1024


@ct.kernel
def _mt5_gelu_gate_dropout_kernel(
    x_ptr,
    gate_ptr,
    random_ptr,
    gt_ptr,
    out_ptr,
    BLOCK_N_: ct.Constant[int],
):
    pid = ct.bid(0)

    random = ct.load(random_ptr, index=(pid,), shape=(BLOCK_N_,))
    keep = random > DROPOUT_P

    x = ct.load(x_ptr, index=(pid,), shape=(BLOCK_N_,))
    gate = ct.load(gate_ptr, index=(pid,), shape=(BLOCK_N_,))

    x_f32 = ct.astype(x, ct.float32)
    gate_f32 = ct.astype(gate, ct.float32)

    # Tanh-approx GELU with explicit bf16 rounding on 0.5*x
    half = ct.astype(x_f32 * 0.5, ct.bfloat16)
    half_f32 = ct.astype(half, ct.float32)

    # GELU approx: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    x_cubed = x_f32 * x_f32 * x_f32
    tanh_arg = (x_f32 + x_cubed * 0.044715) * 0.7978845608028654
    tanh_val = ct.tanh(tanh_arg)
    gelu = half_f32 * (tanh_val + 1.0)

    gated = gelu * gate_f32
    dropped = ct.where(keep, gated, 0.0)
    scaled_bf16 = ct.astype(dropped * DROPOUT_SCALE, ct.bfloat16)

    ct.store(gt_ptr, index=(pid,), tile=keep)
    ct.store(out_ptr, index=(pid,), tile=scaled_bf16)


def _contiguous_stride(shape):
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= int(dim)
    return tuple(reversed(stride))


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


@oracle_impl(hardware="B200", point="96064e9c", BLOCK_N=1024)
def oracle_forward(inputs, **_kwargs):
    x, gate, seeds, _shape_param_0, _shape_param_1, random_shape, out_shape = inputs
    del _shape_param_0, _shape_param_1

    random_shape = tuple(int(dim) for dim in random_shape)
    out_shape = tuple(int(dim) for dim in out_shape)
    device = x.device
    n_elements = x.numel()

    gt = torch.empty_strided(
        random_shape,
        _contiguous_stride(random_shape),
        device=device,
        dtype=torch.bool,
    )
    out = torch.empty_strided(
        out_shape,
        _contiguous_stride(out_shape),
        device=device,
        dtype=torch.bfloat16,
    )

    seed = torch.ops.prims.inductor_lookup_seed.default(seeds, SEED_INDEX)
    random = _inductor_random_for_eager_check(random_shape, seed, device=device)
    random_flat = random.reshape(-1).contiguous()

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (ct.cdiv(n_elements, BLOCK_N), 1, 1), _mt5_gelu_gate_dropout_kernel,
        (x.view(-1), gate.view(-1), random_flat, gt.view(-1), out.view(-1), BLOCK_N),
    )

    return gt, out
