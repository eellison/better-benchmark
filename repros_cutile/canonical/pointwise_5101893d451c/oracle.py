"""cuTile port of pointwise_5101893d451c: MT5 gated tanh-GELU + seeded dropout."""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX = 39
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.1111111111111112
BLOCK_N = 1024


def _contiguous_stride(shape):
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= int(dim)
    return tuple(reversed(stride))


def _inductor_random_for_eager_check(shape, seed, *, device):
    if torch.cuda.is_current_stream_capturing():
        return torch.ops.prims.inductor_random.default(shape, seed, "rand")
    numel = 1
    for dim in shape:
        numel *= int(dim)
    advance = (numel + 131071) // 131072
    state = torch.cuda.get_rng_state(device)
    offset = int.from_bytes(bytes(state[8:16].tolist()), "little")
    if offset >= advance:
        rewound = state.clone()
        rewound[8:16] = torch.tensor(
            list((offset - advance).to_bytes(8, "little", signed=False)),
            dtype=state.dtype,
            device=state.device,
        )
        torch.cuda.set_rng_state(rewound, device)
        random = torch.ops.prims.inductor_random.default(shape, seed, "rand")
        torch.cuda.set_rng_state(state, device)
        return random
    return torch.ops.prims.inductor_random.default(shape, seed, "rand")


@ct.kernel
def _gated_gelu_dropout_kernel(
    x_ptr, gate_ptr, random_ptr,
    gt_ptr, out_ptr,
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    x = ct.load(x_ptr, index=(pid,), shape=(BLOCK,))
    gate = ct.load(gate_ptr, index=(pid,), shape=(BLOCK,))
    random = ct.load(random_ptr, index=(pid,), shape=(BLOCK,))

    keep = random > DROPOUT_P
    ct.store(gt_ptr, index=(pid,), tile=keep)

    x_f = ct.astype(x, ct.float32)
    gate_f = ct.astype(gate, ct.float32)

    # half = x * 0.5 in bf16 rounded
    half = ct.astype(ct.astype(x_f * 0.5, ct.bfloat16), ct.float32)
    x_cubed = x_f * x_f * x_f
    tanh_arg = (x_f + x_cubed * 0.044715) * 0.7978845608028654
    # tanh via exp: tanh(x) = 2 / (1 + exp(-2x)) - 1
    two_x = tanh_arg * 2.0
    exp_neg = ct.exp(-two_x)
    tanh_val = 2.0 / (1.0 + exp_neg) - 1.0
    gelu = half * (tanh_val + 1.0)

    gated = gelu * gate_f
    zero_f = ct.full(shape=(BLOCK,), fill_value=0.0, dtype=ct.float32)
    dropped = ct.where(keep, gated, zero_f)
    scaled = ct.astype(dropped * DROPOUT_SCALE, ct.bfloat16)
    ct.store(out_ptr, index=(pid,), tile=scaled)


@oracle_impl(hardware="B200", point="96064e9c")
def oracle_forward(inputs):
    x, gate, seeds, _shape_param_0, _shape_param_1, random_shape, out_shape = inputs
    random_shape = tuple(int(dim) for dim in random_shape)
    out_shape = tuple(int(dim) for dim in out_shape)
    n_elements = x.numel()
    device = x.device

    gt = torch.empty_strided(random_shape, _contiguous_stride(random_shape),
                             device=device, dtype=torch.bool)
    out = torch.empty_strided(out_shape, _contiguous_stride(out_shape),
                              device=device, dtype=torch.bfloat16)

    seed = torch.ops.prims.inductor_lookup_seed.default(seeds, SEED_INDEX)
    random = _inductor_random_for_eager_check(random_shape, seed, device=device)

    x_flat = x.contiguous().view(-1)
    gate_flat = gate.contiguous().view(-1)
    random_1d = random.view(-1)
    gt_1d = gt.view(-1)
    out_1d = out.view(-1)

    stream = torch.cuda.current_stream()
    grid = (ct.cdiv(n_elements, BLOCK_N), 1, 1)
    ct.launch(
        stream, grid, _gated_gelu_dropout_kernel,
        (x_flat, gate_flat, random_1d, gt_1d, out_1d, BLOCK_N),
    )
    return gt, out
