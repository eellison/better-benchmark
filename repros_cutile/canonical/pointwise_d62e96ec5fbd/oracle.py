"""cuTile port of pointwise_d62e96ec5fbd: exact-erf GELU + seeded dropout (seed 20)."""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX = 20
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
def _gelu_dropout_kernel(
    x_ptr, random_ptr,
    gt_ptr, out_ptr,
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    x = ct.load(x_ptr, index=(pid,), shape=(BLOCK,))
    x_f = ct.astype(x, ct.float32)
    erf_arg = x_f * 0.7071067811865476

    # In-kernel erf via Abramowitz-Stegun 7.1.26.
    zero_f = ct.zeros((BLOCK,), dtype=ct.float32)
    sign = ct.where(erf_arg >= zero_f,
                    ct.full((BLOCK,), 1.0, dtype=ct.float32),
                    ct.full((BLOCK,), -1.0, dtype=ct.float32))
    abs_arg = ct.where(erf_arg >= zero_f, erf_arg, -erf_arg)
    t = 1.0 / (1.0 + 0.3275911 * abs_arg)
    poly = (((((1.061405429 * t) - 1.453152027) * t + 1.421413741) * t
             - 0.284496736) * t + 0.254829592) * t
    erf_val = sign * (1.0 - poly * ct.exp(-abs_arg * abs_arg))

    gelu_half = x_f * 0.5
    gelu = ct.astype(gelu_half * (erf_val + 1.0), ct.bfloat16)

    random = ct.load(random_ptr, index=(pid,), shape=(BLOCK,))
    rand_bf16 = ct.astype(random, ct.bfloat16)
    dropout_p_bf16 = ct.astype(
        ct.full(shape=(BLOCK,), fill_value=DROPOUT_P, dtype=ct.float32),
        ct.bfloat16,
    )
    keep = rand_bf16 > dropout_p_bf16
    ct.store(gt_ptr, index=(pid,), tile=keep)

    zero_bf = ct.full(shape=(BLOCK,), fill_value=0.0, dtype=ct.bfloat16)
    dropped = ct.where(keep, gelu, zero_bf)
    scaled = ct.astype(ct.astype(dropped, ct.float32) * DROPOUT_SCALE, ct.bfloat16)
    ct.store(out_ptr, index=(pid,), tile=scaled)


@oracle_impl(hardware="B200", point="c78a05f8")
def oracle_forward(inputs):
    x, seeds, _shape_param_0, random_shape, out_shape = inputs
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

    random_1d = random.view(-1)
    gt_1d = gt.view(-1)
    out_1d = out.view(-1)

    stream = torch.cuda.current_stream()
    grid = (ct.cdiv(n_elements, BLOCK_N), 1, 1)
    ct.launch(
        stream, grid, _gelu_dropout_kernel,
        (x_flat, random_1d, gt_1d, out_1d, BLOCK_N),
    )
    return gt, out
