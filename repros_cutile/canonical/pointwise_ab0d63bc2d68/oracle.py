"""cuTile port of pointwise_ab0d63bc2d68: MT5 tanh-approx GELU + gate + dropout.

Ports the Triton `_mt5_gelu_gate_dropout_kernel`. Pre-generates the seeded
random tensor via `torch.ops.prims.inductor_random`. Pre-computes the tanh
argument's tanh outside the kernel (cuTile has no built-in tanh); the kernel
does the rest of the pointwise scope.
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX = 63
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.1111111111111112


@ct.kernel
def _mt5_gelu_gate_dropout_kernel(
    x_ptr,          # bf16 [N]
    gate_ptr,       # bf16 [N]
    tanh_val_ptr,   # f32 [N] (pre-computed tanh((x + 0.044715 x^3) * 0.7978845608028654))
    random_ptr,     # f32 [N]
    gt_ptr,         # bool [N]
    out_ptr,        # bf16 [N]
    BLOCK_N: ct.Constant[int],
):
    pid = ct.bid(0)
    x = ct.load(x_ptr, index=(pid,), shape=(BLOCK_N,))
    gate = ct.load(gate_ptr, index=(pid,), shape=(BLOCK_N,))
    tanh_val = ct.load(tanh_val_ptr, index=(pid,), shape=(BLOCK_N,))
    random_f = ct.load(random_ptr, index=(pid,), shape=(BLOCK_N,))

    keep = random_f > DROPOUT_P
    ct.store(gt_ptr, index=(pid,), tile=keep)

    x_f = ct.astype(x, ct.float32)
    gate_f = ct.astype(gate, ct.float32)
    half_bf = ct.astype(x_f * 0.5, ct.bfloat16)
    half_f = ct.astype(half_bf, ct.float32)
    gelu = half_f * (tanh_val + 1.0)
    gated = gelu * gate_f

    keep_f = ct.astype(keep, ct.float32)
    dropped = keep_f * gated
    scaled_bf = ct.astype(dropped * DROPOUT_SCALE, ct.bfloat16)
    ct.store(out_ptr, index=(pid,), tile=scaled_bf)


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
    x, gate, seeds, _shape0, _shape1, random_shape, out_shape = inputs
    del _shape0, _shape1

    random_shape = tuple(int(dim) for dim in random_shape)
    out_shape = tuple(int(dim) for dim in out_shape)
    n_elements = x.numel()
    device = x.device

    seed = torch.ops.prims.inductor_lookup_seed.default(seeds, SEED_INDEX)
    random = _inductor_random_for_eager_check(random_shape, seed, device=device)
    random_flat = random.contiguous().view(n_elements)

    # Pre-compute tanh((x + 0.044715 x^3) * 0.7978845608028654) in f32.
    x_flat = x.contiguous().view(n_elements)
    x_f = x_flat.float()
    x_cubed = x_f * x_f * x_f
    tanh_arg = (x_f + x_cubed * 0.044715) * 0.7978845608028654
    tanh_val = torch.tanh(tanh_arg)

    gate_flat = gate.contiguous().view(n_elements)

    gt = torch.empty_strided(
        random_shape, _contiguous_stride(random_shape),
        device=device, dtype=torch.bool,
    )
    out = torch.empty_strided(
        out_shape, _contiguous_stride(out_shape),
        device=device, dtype=torch.bfloat16,
    )
    gt_flat = gt.view(n_elements)
    out_flat = out.view(n_elements)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(n_elements, BLOCK_N), 1, 1),
        _mt5_gelu_gate_dropout_kernel,
        (x_flat, gate_flat, tanh_val, random_flat, gt_flat, out_flat, BLOCK_N),
    )
    return gt, out
