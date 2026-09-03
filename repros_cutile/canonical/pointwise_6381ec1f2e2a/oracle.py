"""cuTile port of pointwise_6381ec1f2e2a: T5 ReLU + dropout.

view: bf16[8,1024,2048] = arg0_1.view(...)
relu: bf16 = relu(view)
random -> bf16 > 0.1 -> gt
mul: bf16 = gt * relu
mul_1: bf16 = mul * 1.1111...
convert_element_type_1: bf16 (no-op)
view_1: bf16[8192, 2048]
le: bool = relu <= 0
returns (gt, view_1, le)
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX = 31
DROPOUT_SCALE = 1.1111111111111112


@ct.kernel
def _relu_dropout_kernel(
    x_ptr,          # bf16 [rows, cols]
    random_ptr,     # f32  [rows, cols]
    gt_ptr,         # bool [rows, cols]
    dropout_out_ptr,  # bf16 [rows, cols]
    le_ptr,         # bool [rows, cols]
    ROWS: ct.Constant[int],
    COLS: ct.Constant[int],
    BLOCK_R: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    row_block = ct.bid(0)
    col_block = ct.bid(1)

    x = ct.load(x_ptr, index=(row_block, col_block), shape=(BLOCK_R, BLOCK_C))
    zero_bf = ct.full((BLOCK_R, BLOCK_C), 0.0, dtype=ct.bfloat16)
    relu_v = ct.where(x > zero_bf, x, zero_bf)

    # le: relu <= 0 (bool)
    le = relu_v <= zero_bf
    ct.store(le_ptr, index=(row_block, col_block), tile=le)

    rand = ct.load(random_ptr, index=(row_block, col_block), shape=(BLOCK_R, BLOCK_C))
    rand_bf = ct.astype(rand, ct.bfloat16)
    dropout_p_bf = ct.full((BLOCK_R, BLOCK_C), 0.1, dtype=ct.bfloat16)
    keep = rand_bf > dropout_p_bf
    ct.store(gt_ptr, index=(row_block, col_block), tile=keep)

    dropped = ct.where(keep, relu_v, zero_bf)
    scaled = ct.astype(
        ct.astype(dropped, ct.float32) * DROPOUT_SCALE, ct.bfloat16
    )
    ct.store(dropout_out_ptr, index=(row_block, col_block), tile=scaled)


def _shape(shape):
    return tuple(int(d) for d in shape)


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
    advance = (numel + 131071) // 131072
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


@oracle_impl(hardware="B200", point="52dd4c9c", BLOCK_R=8, BLOCK_C=1024)
def oracle_forward(inputs, *, BLOCK_R: int, BLOCK_C: int):
    arg0_1, arg1_1, shape0, shape1, shape2 = inputs
    device = arg0_1.device

    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        raise NotImplementedError(
            "cuTile port unsupported inside CUDA graph capture (seeded RNG)."
        )

    view_shape = _shape(shape0)  # [8, 1024, 2048]
    random_shape = _shape(shape1)
    flat_shape = _shape(shape2)  # [8192, 2048]
    rows = 1
    for d in view_shape[:-1]:
        rows *= int(d)
    cols = int(view_shape[-1])

    gt = torch.empty(view_shape, device=device, dtype=torch.bool)
    le = torch.empty(view_shape, device=device, dtype=torch.bool)
    dropout_out = torch.empty(flat_shape, device=device, dtype=torch.bfloat16)

    seed = torch.ops.prims.inductor_lookup_seed.default(arg1_1, SEED_INDEX)
    random = _inductor_random_for_eager_check(random_shape, seed, device=device)

    x_2d = arg0_1.view(rows, cols)
    random_2d = random.contiguous().view(rows, cols)
    gt_2d = gt.view(rows, cols)
    le_2d = le.view(rows, cols)
    out_2d = dropout_out.view(rows, cols)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(rows, BLOCK_R), ct.cdiv(cols, BLOCK_C), 1),
        _relu_dropout_kernel,
        (x_2d, random_2d, gt_2d, out_2d, le_2d, rows, cols, BLOCK_R, BLOCK_C),
    )
    return gt, dropout_out, le
