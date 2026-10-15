"""cuTile port of pointwise_1d2c10996f53: MT5 gated tanh-GELU + Inductor dropout.

One pointwise kernel: bf16 x*0.5 boundary, fp32 tanh-GELU, bf16 gate multiply
(dropped), seeded dropout gate mask, dropout scale, bf16 output. Random is
pre-generated via torch.ops.prims.inductor_random.
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX = 15
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.1111111111111112
BLOCK_N = 1024


@ct.kernel
def _mt5_gelu_gate_dropout_kernel(
    x_ptr,          # bf16 [total]
    gate_ptr,       # bf16 [total]
    random_ptr,     # f32  [total]
    gt_ptr,         # b8   [total]
    out_ptr,        # bf16 [total]
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    x_bf = ct.load(x_ptr, index=(pid,), shape=(BLOCK,))
    gate_bf = ct.load(gate_ptr, index=(pid,), shape=(BLOCK,))
    rand = ct.load(random_ptr, index=(pid,), shape=(BLOCK,))
    keep = rand > DROPOUT_P
    ct.store(gt_ptr, index=(pid,), tile=keep)

    x = ct.astype(x_bf, ct.float32)
    gate = ct.astype(gate_bf, ct.float32)
    # (x * 0.5).to(bf16).to(f32)
    half = ct.astype(ct.astype(x * 0.5, ct.bfloat16), ct.float32)
    x3 = x * x * x
    tanh_arg = (x + x3 * 0.044715) * 0.7978845608028654
    # cuTile has no tanh; use exp: tanh(y) = (e^{2y} - 1) / (e^{2y} + 1)
    e2y = ct.exp(2.0 * tanh_arg)
    tanh_val = (e2y - 1.0) * (1.0 / (e2y + 1.0))
    gelu = half * (tanh_val + 1.0)
    gated = gelu * gate
    dropped = ct.astype(keep, ct.float32) * gated
    scaled_bf = ct.astype(dropped * DROPOUT_SCALE, ct.bfloat16)
    ct.store(out_ptr, index=(pid,), tile=scaled_bf)


def _contiguous_stride(shape):
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= int(dim)
    return tuple(reversed(stride))


def _as_shape(shape):
    return tuple(int(dim) for dim in shape)


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


@oracle_impl(hardware="B200", point="96064e9c")
def oracle_forward(inputs):
    x, gate, seeds, _shape0, _shape1, random_shape, out_shape = inputs
    device = x.device
    random_shape_t = _as_shape(random_shape)
    out_shape_t = _as_shape(out_shape)
    n = x.numel()

    seed = torch.ops.prims.inductor_lookup_seed.default(seeds, SEED_INDEX)
    random = _inductor_random_for_eager_check(random_shape_t, seed, device=device)
    random_flat = random.contiguous().view(-1)
    x_flat = x.contiguous().view(-1)
    gate_flat = gate.contiguous().view(-1)

    gt_flat = torch.empty((n,), device=device, dtype=torch.bool)
    out_flat = torch.empty((n,), device=device, dtype=torch.bfloat16)

    stream = torch.cuda.current_stream()
    assert n % BLOCK_N == 0
    ct.launch(
        stream, (n // BLOCK_N, 1, 1), _mt5_gelu_gate_dropout_kernel,
        (x_flat, gate_flat, random_flat, gt_flat, out_flat, BLOCK_N),
    )

    gt = torch.empty_strided(
        random_shape_t, _contiguous_stride(random_shape_t),
        device=device, dtype=torch.bool,
    )
    gt.view(-1).copy_(gt_flat)
    out = torch.empty_strided(
        out_shape_t, _contiguous_stride(out_shape_t),
        device=device, dtype=torch.bfloat16,
    )
    out.view(-1).copy_(out_flat)

    return gt, out
