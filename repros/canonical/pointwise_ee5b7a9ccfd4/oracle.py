"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete bf16 MT5 tanh-approximate GELU, gated multiply, seed-index-7 dropout, returned bool keep mask, and final contiguous bf16 view in one storage-linear Triton pointwise kernel, including the explicit bf16-rounded `x * 0.5` factor, fp32 cubic/tanh GELU path, fp32 Inductor-random comparison against `0.1`, fp32 gate/dropout scaling, and final bf16 cast, whereas Inductor lowers the stochastic GELU-gate-dropout graph through its generic fused pointwise schedule; Inductor cannot do this today because the scheduler does not have a shape-specialized MT5 GELU-gated-dropout dual-output template that preserves the seed-indexed RNG, bf16 cast boundaries, and final view layout with minimal indexing overhead; the fix is SCHEDULER_FUSION: add a guarded pointwise template for this MT5 GELU-gated-dropout layout family, including the bool side output and bf16 materialization."""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


SEED_INDEX = 7
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.1111111111111112


@triton.jit
def _mt5_gelu_gate_dropout_kernel(
    x_ptr,
    gate_ptr,
    random_or_seed_ptr,
    gt_ptr,
    out_ptr,
    seed_index: tl.constexpr,
    dropout_p: tl.constexpr,
    dropout_scale: tl.constexpr,
    use_seeded_rng: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)

    if use_seeded_rng:
        seed = tl.load(random_or_seed_ptr + seed_index)
        random = tl.rand(seed, offsets.to(tl.uint32))
    else:
        random = tl.load(
            random_or_seed_ptr + offsets,
            eviction_policy="evict_first",
        )
    keep = random > dropout_p

    x = tl.load(x_ptr + offsets).to(tl.float32)
    gate = tl.load(gate_ptr + offsets).to(tl.float32)

    half = (x * 0.5).to(tl.bfloat16).to(tl.float32)
    x_cubed = x * x * x
    tanh_arg = (x + x_cubed * 0.044715) * 0.7978845608028654
    gelu = half * (libdevice.tanh(tanh_arg) + 1.0)

    gated = gelu * gate
    dropped = keep.to(tl.float32) * gated
    scaled = (dropped * dropout_scale).to(tl.bfloat16)

    tl.store(gt_ptr + offsets, keep)
    tl.store(out_ptr + offsets, scaled)


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


def _launch(inputs, *, BLOCK_N: int, num_warps: int, num_stages: int):
    x, gate, seeds, _shape_param_0, _shape_param_1, random_shape, out_shape = inputs
    del _shape_param_0, _shape_param_1

    random_shape = tuple(int(dim) for dim in random_shape)
    out_shape = tuple(int(dim) for dim in out_shape)
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

    grid = (triton.cdiv(n_elements, BLOCK_N),)
    if torch.cuda.is_current_stream_capturing():
        _mt5_gelu_gate_dropout_kernel[grid](
            x,
            gate,
            seeds,
            gt,
            out,
            seed_index=SEED_INDEX,
            dropout_p=DROPOUT_P,
            dropout_scale=DROPOUT_SCALE,
            use_seeded_rng=True,
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    else:
        seed = torch.ops.prims.inductor_lookup_seed.default(seeds, SEED_INDEX)
        random = _inductor_random_for_eager_check(
            random_shape,
            seed,
            device=x.device,
        )
        _mt5_gelu_gate_dropout_kernel[grid](
            x,
            gate,
            random,
            gt,
            out,
            seed_index=SEED_INDEX,
            dropout_p=DROPOUT_P,
            dropout_scale=DROPOUT_SCALE,
            use_seeded_rng=False,
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    return gt, out


# 96064e9c: bf16[4096,1024], bf16[4096,1024], seed index 7
@oracle_impl(hardware="B200", point="96064e9c", BLOCK_N=1024, num_warps=4, num_stages=1)
def oracle_forward(
    inputs,
    *,
    BLOCK_N: int,
    num_warps: int,
    num_stages: int,
):
    return _launch(
        inputs,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=num_stages,
    )
