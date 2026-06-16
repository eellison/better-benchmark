"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the
complete Demucs stochastic irregular gather scope by replaying the captured
Inductor random streams, composing the sign flip, stereo swap, time-window
gather, group permutation, visible `[4,4,2,382788]` output materialization, and
sibling `[4,2,382788]` f32 source sum in one Triton pass. Inductor lowers the
random sort, three randint-indexed gathers, cat/view chain, returned materialized
gather tensor, and dependent dim-1 reduction as generic scheduled regions with
large intermediate traffic. It cannot do this today because scheduler/codegen
does not have a stochastic irregular gather-reduce template that keeps the
returned gather tensor live while accumulating the sibling reduction from the
same loaded values and preserving the exact Inductor RNG/index semantics; the
fix is NEW_PATTERN: add a Demucs-style fused random-index composition template
with visible side-output stores and a reduction epilogue."""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import triton
import triton.language as tl

from oracle_harness import oracle_impl


OUT_T = 382788
IN_T = 426888
INPUT_N_STRIDE = 5 * 2 * IN_T
INPUT_SOURCE_STRIDE = 2 * IN_T
REPRO_RNG_ADVANCE = 40


def _state_u64(state, offset):
    return int.from_bytes(bytes(state[offset : offset + 8].tolist()), "little")


def _put_state_u64(state, offset, value):
    state[offset : offset + 8] = torch.tensor(
        list(int(value).to_bytes(8, "little")),
        dtype=state.dtype,
        device=state.device,
    )


def _repro_random_inputs(device):
    if torch.cuda.is_current_stream_capturing():
        restore_state = None
    else:
        state = torch.cuda.get_rng_state(device)
        offset = _state_u64(state, 8)
        restore_state = state
        if offset >= REPRO_RNG_ADVANCE:
            rewound = state.clone()
            _put_state_u64(rewound, 8, offset - REPRO_RNG_ADVANCE)
            torch.cuda.set_rng_state(rewound, device)
        else:
            restore_state = None

    seeds = torch.ops.prims.inductor_seeds.default(
        4, device=torch.device(type="cuda", index=device.index or 0)
    )
    group_seed = torch.ops.prims.inductor_lookup_seed.default(seeds, 3)
    group_random = torch.ops.prims.inductor_random.default(
        [1, 4, 4, 1, 1], group_seed, "rand"
    )
    group_indices = torch.ops.aten.sort.default(group_random, 1)[1]

    sign_seed = torch.ops.prims.inductor_lookup_seed.default(seeds, 0)
    sign_bits = torch.ops.prims.inductor_randint.default(
        0, 2, [4, 4, 1, 1], sign_seed
    )
    stereo_seed = torch.ops.prims.inductor_lookup_seed.default(seeds, 1)
    stereo = torch.ops.prims.inductor_randint.default(0, 2, [4, 4, 1, 1], stereo_seed)
    offset_seed = torch.ops.prims.inductor_lookup_seed.default(seeds, 2)
    time_offsets = torch.ops.prims.inductor_randint.default(
        0, 44100, [4, 4, 1, 1], offset_seed
    )

    if restore_state is not None:
        torch.cuda.set_rng_state(restore_state, device)

    return group_indices, sign_bits, stereo, time_offsets


@triton.jit
def _materialize_and_sum_kernel(
    x_ptr,
    perm_ptr,
    sign_ptr,
    stereo_ptr,
    offset_ptr,
    out_ptr,
    sum_ptr,
    BLOCK_T: tl.constexpr,
):
    tile_t = tl.program_id(0)
    out_group = tl.program_id(1)
    out_stereo = tl.program_id(2)
    t = tile_t * BLOCK_T + tl.arange(0, BLOCK_T)
    mask = t < 382788

    acc = tl.zeros((BLOCK_T,), dtype=tl.float32)
    for source in tl.static_range(0, 4):
        input_group = tl.load(perm_ptr + out_group * 4 + source)
        sign_bit = tl.load(sign_ptr + input_group * 4 + source)
        sign = sign_bit.to(tl.float32) * 2.0 - 1.0
        stereo_index = tl.load(stereo_ptr + input_group * 4 + source)
        effective_stereo = tl.where(out_stereo == 0, stereo_index, 1 - stereo_index)
        time_offset = tl.load(offset_ptr + input_group * 4 + source)

        input_offsets = (
            input_group * 4268880
            + (source + 1) * 853776
            + effective_stereo * 426888
            + time_offset
            + t
        )
        value = tl.load(x_ptr + input_offsets, mask=mask, other=0.0) * sign
        out_offsets = ((out_group * 4 + source) * 2 + out_stereo) * 382788 + t
        tl.store(out_ptr + out_offsets, value, mask=mask)
        acc += value

    sum_offsets = (out_group * 2 + out_stereo) * 382788 + t
    tl.store(sum_ptr + sum_offsets, acc, mask=mask)


@oracle_impl(
    hardware="B200",
    point="764a0dde",
    BLOCK_T=4096,
    num_warps=8,
)
def oracle_forward(inputs, *, BLOCK_T: int, num_warps: int):
    x, *_shape_params = inputs
    group_indices, sign_bits, stereo, time_offsets = _repro_random_inputs(x.device)

    out = torch.empty_strided(
        (4, 4, 2, OUT_T),
        (4 * 2 * OUT_T, 2 * OUT_T, OUT_T, 1),
        device=x.device,
        dtype=torch.float32,
    )
    summed = torch.empty_strided(
        (4, 2, OUT_T),
        (2 * OUT_T, OUT_T, 1),
        device=x.device,
        dtype=torch.float32,
    )

    _materialize_and_sum_kernel[(triton.cdiv(OUT_T, BLOCK_T), 4, 2)](
        x,
        group_indices,
        sign_bits,
        stereo,
        time_offsets,
        out,
        summed,
        BLOCK_T=BLOCK_T,
        num_warps=num_warps,
        num_stages=4,
    )
    return out, summed
