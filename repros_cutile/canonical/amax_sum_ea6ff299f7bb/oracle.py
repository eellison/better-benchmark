"""cuTile port of amax_sum_ea6ff299f7bb: BERT scaled masked attention softmax + dropout.

Row kernel per (batch, head, q) that: divides scores by 8.0 (bf16 rounded),
applies boolean mask (broadcast on heads) with bf16 scalar fill, computes
bf16 masked scores side output, fp32 amax and softmax sum side outputs,
then seeded dropout via pre-computed random tensor. Returns
(where, amax, sum, gt, bf16_out, permute).
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX = 31
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.1111111111111112


@ct.kernel
def _scaled_masked_softmax_dropout_kernel(
    x_ptr,       # bf16 [B, H, Q, K]
    mask_ptr,    # b8   [B, 1, Q, K]
    random_ptr,  # f32  [B, H, Q, K]
    where_ptr,   # bf16 [B, H, Q, K]
    amax_ptr,    # f32  [B, H, Q, 1]
    sum_ptr,     # f32  [B, H, Q, 1]
    gt_ptr,      # b8   [B, H, Q, K]
    out_ptr,     # bf16 [B, H, Q, K]
    FILL: ct.Constant[float],
    BLOCK_N: ct.Constant[int],
):
    b = ct.bid(0)
    h = ct.bid(1)
    q = ct.bid(2)

    raw_bf = ct.load(x_ptr, index=(b, h, q, 0), shape=(1, 1, 1, BLOCK_N))
    # bf16-rounded division by 8.0: cast to fp32, scale, back to bf16.
    scaled_bf = ct.astype(ct.astype(raw_bf, ct.float32) * 0.125, ct.bfloat16)
    m = ct.load(mask_ptr, index=(b, 0, q, 0), shape=(1, 1, 1, BLOCK_N))
    fill_tile = ct.full((1, 1, 1, BLOCK_N), FILL, dtype=ct.bfloat16)
    masked = ct.where(m, fill_tile, scaled_bf)
    ct.store(where_ptr, index=(b, h, q, 0), tile=masked)

    scores = ct.astype(masked, ct.float32)
    row_max = ct.max(scores, axis=3, keepdims=True)
    shifted = scores - row_max
    numer = ct.exp(shifted)
    denom = ct.sum(numer, axis=3, keepdims=True)
    probs = numer / denom

    ct.store(amax_ptr, index=(b, h, q, 0), tile=row_max)
    ct.store(sum_ptr, index=(b, h, q, 0), tile=denom)

    rand_f = ct.load(random_ptr, index=(b, h, q, 0), shape=(1, 1, 1, BLOCK_N))
    threshold = ct.full((1, 1, 1, BLOCK_N), DROPOUT_P, dtype=ct.float32)
    keep = rand_f > threshold
    ct.store(gt_ptr, index=(b, h, q, 0), tile=keep)

    zero_f = ct.full((1, 1, 1, BLOCK_N), 0.0, dtype=ct.float32)
    dropped = ct.where(keep, probs, zero_f)
    scaled_out = ct.astype(dropped * DROPOUT_SCALE, ct.bfloat16)
    ct.store(out_ptr, index=(b, h, q, 0), tile=scaled_out)


def _contiguous_stride(shape):
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= int(dim)
    return tuple(reversed(stride))


def _resolve_shape(shape, numel):
    dims = [int(d) for d in shape]
    unknown = -1
    known = 1
    for i, d in enumerate(dims):
        if d == -1:
            unknown = i
        else:
            known *= d
    if unknown >= 0:
        dims[unknown] = int(numel) // known
    return tuple(dims)


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


@oracle_impl(hardware="B200", point="0e2c5e9e", BLOCK_N=128)
def oracle_forward(inputs, *, BLOCK_N: int):
    arg0_1, arg1_1, arg2_1, arg3_1, shape0, shape1, _shape2, shape3 = inputs

    numel = int(arg0_1.numel())
    view_shape = _resolve_shape(shape0, numel)
    random_shape = _resolve_shape(shape1, numel)
    out_shape = _resolve_shape(shape3, numel)

    B, H, Q, K = view_shape
    row_shape = (B, H, Q, 1)
    device = arg0_1.device

    where = torch.empty_strided(
        view_shape, _contiguous_stride(view_shape),
        device=device, dtype=torch.bfloat16,
    )
    amax = torch.empty_strided(
        row_shape, _contiguous_stride(row_shape),
        device=device, dtype=torch.float32,
    )
    sum_1 = torch.empty_strided(
        row_shape, _contiguous_stride(row_shape),
        device=device, dtype=torch.float32,
    )
    gt = torch.empty_strided(
        view_shape, _contiguous_stride(view_shape),
        device=device, dtype=torch.bool,
    )
    out_4d = torch.empty_strided(
        view_shape, _contiguous_stride(view_shape),
        device=device, dtype=torch.bfloat16,
    )

    seed = torch.ops.prims.inductor_lookup_seed.default(arg3_1, SEED_INDEX)
    random = _inductor_random_for_eager_check(random_shape, seed, device=device)

    x_4d = arg0_1.view(B, H, Q, K)
    random_4d = random.view(B, H, Q, K)
    fill_val = float(arg2_1.item())

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (B, H, Q),
        _scaled_masked_softmax_dropout_kernel,
        (x_4d, arg1_1, random_4d, where, amax, sum_1, gt, out_4d,
         fill_val, BLOCK_N),
    )

    out_flat = out_4d.view(out_shape)
    return where, amax, sum_1, gt, out_flat, out_flat.permute(0, 2, 1)
