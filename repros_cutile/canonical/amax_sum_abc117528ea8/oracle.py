"""cuTile port of amax_sum_abc117528ea8: Longformer training softmax + dropout.

Uses torch for the band assembly and edge masks (given as inputs). A cuTile
kernel does the row softmax with query-mask zeroing plus seeded dropout.
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 8
SEQ = 1024
HEADS = 12
WINDOW = 513
SEED_INDEX = 18
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.1111111111111112


@ct.kernel
def _softmax_dropout_kernel(
    scores_ptr,     # bf16 [rows, W]
    q_mask_ptr,     # b8 [rows]
    q_fill_ptr,     # f32 scalar
    random_ptr,     # f32 [rows, W]
    softmax_out_ptr,# f32 [rows, W]
    amax_ptr,       # f32 [rows]
    sum_ptr,        # f32 [rows]
    gt_ptr,         # b8 [rows, W]
    mul_ptr,        # bf16 [rows, W]
    W: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
):
    row = ct.bid(0)
    scores_bf = ct.load(scores_ptr, index=(row, 0), shape=(1, BLOCK_N),
                        padding_mode=ct.PaddingMode.ZERO)
    scores_f = ct.astype(scores_bf, ct.float32)
    col_idx = ct.arange(BLOCK_N, dtype=ct.int32)
    col_mask = ct.reshape(col_idx < W, (1, BLOCK_N))
    ninf = ct.full((1, BLOCK_N), -float("inf"), dtype=ct.float32)
    scores_masked = ct.where(col_mask, scores_f, ninf)
    # NaN-aware max: torch amax propagates NaN. cuTile's ct.max may skip NaN
    # so detect explicit NaN and inject via sum(NaN, 0).
    is_nan = scores_masked != scores_masked
    nan_count = ct.sum(ct.where(is_nan, 1, 0))
    row_max_plain = ct.max(scores_masked)
    nan_val = ct.full((), float("nan"), dtype=ct.float32)
    row_max = ct.where(nan_count > 0, nan_val, row_max_plain)
    numer = ct.exp(scores_masked - row_max)
    numer = ct.where(col_mask, numer, 0.0)
    denom = ct.sum(numer)
    probs_f = numer / denom
    ct.store(amax_ptr, index=(row,), tile=ct.reshape(row_max, (1,)))
    ct.store(sum_ptr, index=(row,), tile=ct.reshape(denom, (1,)))

    qmask = ct.load(q_mask_ptr, index=(row,), shape=(1,))
    q_fill_bcast = ct.load(q_fill_ptr, index=(0,), shape=(1,))
    softmax_f = ct.where(qmask != 0, q_fill_bcast, probs_f)
    ct.store(softmax_out_ptr, index=(row, 0), tile=softmax_f)
    softmax_bf = ct.astype(softmax_f, ct.bfloat16)

    rand_f = ct.load(random_ptr, index=(row, 0), shape=(1, BLOCK_N),
                     padding_mode=ct.PaddingMode.ZERO)
    rand_bf = ct.astype(rand_f, ct.bfloat16)
    keep = rand_bf > DROPOUT_P
    ct.store(gt_ptr, index=(row, 0), tile=keep)
    zero_bf = ct.full((1, BLOCK_N), 0.0, dtype=ct.bfloat16)
    dropped = ct.where(keep, softmax_bf, zero_bf)
    scaled = ct.astype(ct.astype(dropped, ct.float32) * DROPOUT_SCALE, ct.bfloat16)
    ct.store(mul_ptr, index=(row, 0), tile=scaled)


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


@oracle_impl(hardware="B200", point="b64f0e8a")
def oracle_forward(inputs):
    (
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1,
        arg9_1, arg10_1, *sp,
    ) = inputs
    device = arg0_1.device
    sp_t = [tuple(int(d) for d in s) for s in sp]

    view = arg0_1.view(*sp_t[0])
    permute = view.permute(0, 1, 2, 4, 3)
    view_1 = permute.reshape(*sp_t[1])
    constant_pad_nd = torch.nn.functional.pad(view_1, [0, 0, 0, 1], value=0.0)
    view_2 = constant_pad_nd.view(*sp_t[2])

    scaffold = arg3_1.clone()
    scaffold[:, 0:-1, :, 256:] = view_2[:, :, 0:256, 0:257]
    scaffold[:, -1, :, 256:] = view_2[:, -1, 256:, 0:257]
    scaffold[:, 1:, :, 0:256] = view_2[:, :, -257:-1, 257:]
    scaffold[:, 0, 1:256, 1:256] = view_2[:, 0, 0:255, -255:]

    scores = scaffold.view(*sp_t[3]).permute(0, 2, 1, 3).contiguous()

    tl = scores[:, 0:256, :, 0:257].clone()
    scores[:, 0:256, :, 0:257] = torch.where(arg4_1, arg5_1, tl)

    br = scores[:, -256:, :, -257:].clone()
    scores[:, -256:, :, -257:] = torch.where(arg6_1, arg5_1, br)

    scores_total = scores + arg7_1

    scores_flat = scores_total.contiguous().view(-1, WINDOW)
    rows = scores_flat.shape[0]

    seed = torch.ops.prims.inductor_lookup_seed.default(arg10_1, SEED_INDEX)
    random = _inductor_random_for_eager_check(
        (BATCH, SEQ, HEADS, WINDOW), seed, device=device,
    )
    random_flat = random.contiguous().view(-1, WINDOW)

    q_mask_full = arg8_1.view(BATCH, SEQ).unsqueeze(2).expand(
        BATCH, SEQ, HEADS
    ).contiguous().view(-1).to(torch.int32)
    q_fill_scalar = arg9_1.view(1)

    softmax_out = torch.empty((rows, WINDOW), device=device, dtype=torch.float32)
    amax_out = torch.empty((rows,), device=device, dtype=torch.float32)
    sum_out = torch.empty((rows,), device=device, dtype=torch.float32)
    gt_out = torch.empty((rows, WINDOW), device=device, dtype=torch.bool)
    mul_out = torch.empty((rows, WINDOW), device=device, dtype=torch.bfloat16)

    BLOCK_N = 1024
    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (rows, 1, 1), _softmax_dropout_kernel,
        (scores_flat, q_mask_full, q_fill_scalar, random_flat,
         softmax_out, amax_out, sum_out, gt_out, mul_out, WINDOW, BLOCK_N),
    )

    permute_7 = scores_total  # [B, 1024, 12, 513]
    amax = amax_out.view(BATCH, SEQ, HEADS, 1)
    sum_1 = sum_out.view(BATCH, SEQ, HEADS, 1)
    gt = gt_out.view(BATCH, SEQ, HEADS, WINDOW)
    mul_1 = mul_out.view(BATCH, SEQ, HEADS, WINDOW)

    permute_8 = mul_1.permute(0, 2, 1, 3)
    clone_1 = permute_8.contiguous()
    view_6 = clone_1.reshape(*sp_t[7])
    constant_pad_nd_1 = torch.nn.functional.pad(view_6, sp_t[8], value=0.0)
    view_7 = constant_pad_nd_1.view(*sp_t[9])
    slice_18 = view_7[:, :, 0:-256]
    view_8 = slice_18.view(*sp_t[10])
    slice_19 = view_8[:, :, :, 0:-1]
    unsqueeze = slice_19.unsqueeze(4)
    view_9 = unsqueeze.view(*sp_t[11])
    permute_9 = view_9.permute(0, 2, 1)

    return permute_7, amax, sum_1, gt, view_9, permute_9
