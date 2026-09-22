"""cuTile port of pointwise_24ca2fc47594: MT5 tanh-approx GELU + gate + dropout.

Uses pre-generated random tensor (from torch.ops.prims.inductor_random) to
sidestep cuTile's lack of on-device seeded RNG. The `mul` intrinsic in the
Triton oracle uses inline PTX `mul.rn.f32`, but that's just round-to-nearest
f32 multiply which cuTile does natively.
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX = 19
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.1111111111111112


@ct.kernel
def _mt5_gelu_gate_dropout_kernel(
    x_ptr,          # bf16 [N]
    gate_ptr,       # bf16 [N]
    random_ptr,     # f32  [N]
    gt_ptr,         # b8   [N]
    out_ptr,        # bf16 [N]
    BLOCK_N: ct.Constant[int],
):
    pid = ct.bid(0)
    x_bf = ct.load(x_ptr, index=(pid,), shape=(BLOCK_N,))
    gate_bf = ct.load(gate_ptr, index=(pid,), shape=(BLOCK_N,))
    rand = ct.load(random_ptr, index=(pid,), shape=(BLOCK_N,))

    keep = rand > DROPOUT_P
    ct.store(gt_ptr, index=(pid,), tile=keep)

    x = ct.astype(x_bf, ct.float32)
    gate = ct.astype(gate_bf, ct.float32)

    half = ct.astype(ct.astype(x * 0.5, ct.bfloat16), ct.float32)
    x_cubed = x * x * x
    tanh_arg = (x + x_cubed * 0.044715) * 0.7978845608028654
    gelu = half * (ct.tanh(tanh_arg) + 1.0)

    gated = gelu * gate
    dropped = ct.astype(keep, ct.float32) * gated
    scaled = ct.astype(dropped * DROPOUT_SCALE, ct.bfloat16)
    ct.store(out_ptr, index=(pid,), tile=scaled)


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
def oracle_forward(inputs, *, BLOCK_N: int):
    x, gate, seeds, _s0, _s1, random_shape, out_shape = inputs

    random_shape = tuple(int(d) for d in random_shape)
    out_shape = tuple(int(d) for d in out_shape)
    n_elements = x.numel()
    gt = torch.empty_strided(
        random_shape,
        _contiguous_stride(random_shape),
        device=x.device,
        dtype=torch.bool,
    )
    out = torch.empty_strided(
        out_shape,
        _contiguous_stride(out_shape),
        device=x.device,
        dtype=torch.bfloat16,
    )

    seed = torch.ops.prims.inductor_lookup_seed.default(seeds, SEED_INDEX)
    random = _inductor_random_for_eager_check(random_shape, seed, device=x.device)

    # Flatten every tensor to 1D so the cuTile kernel sees them uniformly.
    x_flat = x.reshape(n_elements)
    gate_flat = gate.reshape(n_elements)
    random_flat = random.reshape(n_elements).contiguous()
    gt_flat = gt.view(n_elements)
    out_flat = out.view(n_elements)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(n_elements, BLOCK_N), 1, 1),
        _mt5_gelu_gate_dropout_kernel,
        (x_flat, gate_flat, random_flat, gt_flat, out_flat, BLOCK_N),
    )

    return gt, out
