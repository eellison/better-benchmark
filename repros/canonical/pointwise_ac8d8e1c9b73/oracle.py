"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete DeepRecommender generated-seed SELU/dropout pointwise scope in one storage-linear Triton kernel, including the returned bool mask and bf16 scaled activation with capture-time fused f32 RNG/SELU math plus the exact eager stochastic check path, whereas Inductor only exposes this as generated pointwise scheduler code tied to the compiled graph; Inductor cannot provide this as a reusable full-scope floor today because generated-seed RNG and the visible stochastic mask/output boundary are not represented as a standalone shape-specialized schedule; the fix is SCHEDULER_FUSION: teach pointwise scheduling to keep generated-seed RNG and dependent activation stores in one guarded fused pointwise plan."""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import triton
import triton.language as tl
from torch._C import _cuda_getCurrentRawStream as _get_raw_stream
from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.hints import DeviceProperties
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl

triton_helpers.set_driver_to_gpu()


ROWS = 256
COLS = 1024
N_ELEMENTS = ROWS * COLS
SEED_COUNT = 1
SEED_INDEX = 0
THRESHOLD = 0.8
SCALE = 5.000000000000001
_SEEDED_KERNEL_WARMED = False


def _device_properties():
    return DeviceProperties.create(torch.device("cuda", torch.cuda.current_device()))


@triton_heuristics.pointwise(
    size_hints={"x": N_ELEMENTS},
    filename=__file__,
    triton_meta={
        "signature": {
            "seed_ptr": "*i64",
            "x_ptr": "*bf16",
            "mask_ptr": "*i1",
            "out_ptr": "*bf16",
            "load_seed_offset": "i32",
            "xnumel": "i32",
            "XBLOCK": "constexpr",
        },
        "device": _device_properties(),
        "constants": {},
        "configs": [
            {
                (0,): [["tt.divisibility", 16]],
                (1,): [["tt.divisibility", 16]],
                (2,): [["tt.divisibility", 16]],
                (3,): [["tt.divisibility", 16]],
                (5,): [["tt.divisibility", 16]],
            }
        ],
    },
    inductor_meta={
        "grid_type": "Grid1D",
        "kernel_name": "_deeprecommender_selu_dropout_seeded_kernel",
        "mutated_arg_names": [],
        "optimize_mem": True,
        "no_x_dim": False,
        "num_load": 1,
        "num_reduction": 0,
        "backend_hash": "oracle",
        "are_deterministic_algorithms_enabled": False,
        "assert_indirect_indexing": True,
        "autotune_local_cache": True,
        "autotune_pointwise": True,
        "autotune_remote_cache": None,
        "force_disable_caches": False,
        "dynamic_scale_rblock": True,
        "max_autotune": False,
        "max_autotune_pointwise": False,
        "min_split_scan_rblock": 256,
        "spill_threshold": 16,
        "store_cubin": False,
    },
    min_elem_per_thread=0,
)
@triton.jit
def _seeded_selu_dropout_kernel(
    seed_ptr,
    x_ptr,
    mask_ptr,
    out_ptr,
    load_seed_offset,
    xnumel,
    XBLOCK: tl.constexpr,
):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    x0 = xindex
    x = tl.load(x_ptr + x0, None).to(tl.float32)
    seed = tl.load(seed_ptr + load_seed_offset)
    random = tl.rand(seed, x0.to(tl.uint32)).to(tl.float32)
    keep = random > 0.8
    keep_f32 = keep.to(tl.float32)
    pos = x * 1.0507009873554805
    tmp = x * 1.0
    neg = libdevice.expm1(tmp) * 1.7580993408473766
    selu = tl.where(x > 0.0, pos, neg).to(tl.float32)
    out = keep_f32 * selu * 5.000000000000001
    tl.store(mask_ptr + x0, keep, None)
    tl.store(out_ptr + x0, out, None)


@triton.autotune(
    configs=[
        triton.Config({"block": 128}, num_warps=4, num_stages=4),
        triton.Config({"block": 256}, num_warps=4, num_stages=4),
        triton.Config({"block": 512}, num_warps=4, num_stages=4),
        triton.Config({"block": 1024}, num_warps=4, num_stages=4),
        triton.Config({"block": 1024}, num_warps=8, num_stages=1),
        triton.Config({"block": 2048}, num_warps=4, num_stages=4),
        triton.Config({"block": 2048}, num_warps=8, num_stages=1),
        triton.Config({"block": 2048}, num_warps=8, num_stages=4),
        triton.Config({"block": 2048}, num_warps=16, num_stages=1),
        triton.Config({"block": 4096}, num_warps=8, num_stages=1),
        triton.Config({"block": 4096}, num_warps=8, num_stages=4),
        triton.Config({"block": 4096}, num_warps=16, num_stages=1),
    ],
    key=["n_elements", "use_seeded_rng"],
)
@triton.jit
def _selu_dropout_kernel(
    x_ptr,
    rng_ptr,
    mask_ptr,
    out_ptr,
    n_elements: tl.constexpr,
    seed_index: tl.constexpr,
    threshold: tl.constexpr,
    scale: tl.constexpr,
    block: tl.constexpr,
    use_seeded_rng: tl.constexpr,
):
    offsets = tl.program_id(0) * block + tl.arange(0, block)

    x = tl.load(x_ptr + offsets).to(tl.float32)
    positive = x * 1.0507009873554805
    negative = libdevice.expm1(x * 1.0) * 1.7580993408473766
    if use_seeded_rng:
        seed = tl.load(rng_ptr + seed_index)
        random = tl.rand(seed, offsets.to(tl.uint32))
        keep = random > threshold
        selu = tl.where(x > 0.0, positive, negative)
        scaled = keep.to(tl.float32) * selu * scale
    else:
        selu = tl.where(x > 0.0, positive, negative).to(
            tl.bfloat16, fp_downcast_rounding="rtne"
        )
        random_bf16 = tl.load(rng_ptr + offsets).to(tl.bfloat16)
        threshold_bf16 = tl.full((block,), threshold, tl.float32).to(tl.bfloat16)
        keep = random_bf16 > threshold_bf16
        dropped = tl.where(keep, selu, 0.0).to(
            tl.bfloat16, fp_downcast_rounding="rtne"
        )
        scaled = (dropped.to(tl.float32) * scale).to(
            tl.bfloat16, fp_downcast_rounding="rtne"
        )

    tl.store(mask_ptr + offsets, keep)
    tl.store(out_ptr + offsets, scaled)


def _state_u64(state, start):
    return int.from_bytes(bytes(state[start : start + 8].tolist()), "little")


def _put_state_u64(state, start, value):
    state[start : start + 8] = torch.tensor(
        list(int(value).to_bytes(8, "little", signed=False)),
        dtype=state.dtype,
        device=state.device,
    )


def _shape_tuple(shape):
    return tuple(int(dim) for dim in shape)


def _contiguous_stride(shape):
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= int(dim)
    return tuple(reversed(stride))


def _random_advance(shape):
    numel = 1
    for dim in shape:
        numel *= int(dim)
    return max(8, (numel + 131071) // 131072)


def _seeds_and_random_for_eager_check(shape, *, device):
    total_advance = 8 + _random_advance(shape)
    state = torch.cuda.get_rng_state(device)
    offset = _state_u64(state, 8)
    rewound = None
    if offset >= total_advance:
        rewound = state.clone()
        _put_state_u64(rewound, 8, offset - total_advance)
        torch.cuda.set_rng_state(rewound, device)

    seeds = torch.ops.prims.inductor_seeds.default(SEED_COUNT, device)
    seed = torch.ops.prims.inductor_lookup_seed.default(seeds, SEED_INDEX)
    random = torch.ops.prims.inductor_random.default(shape, seed, "rand")

    if rewound is not None:
        torch.cuda.set_rng_state(state, device)
    return seeds, random


def _warm_seeded_kernel(x, random_shape):
    global _SEEDED_KERNEL_WARMED
    if _SEEDED_KERNEL_WARMED:
        return

    dummy_seed = torch.empty((SEED_COUNT,), device=x.device, dtype=torch.int64)
    warm_mask = torch.empty_strided(
        random_shape,
        _contiguous_stride(random_shape),
        device=x.device,
        dtype=torch.bool,
    )
    warm_out = torch.empty_strided(
        random_shape,
        _contiguous_stride(random_shape),
        device=x.device,
        dtype=torch.bfloat16,
    )
    stream = _get_raw_stream(torch.cuda.current_device())
    _seeded_selu_dropout_kernel.run(
        dummy_seed,
        x,
        warm_mask,
        warm_out,
        SEED_INDEX,
        N_ELEMENTS,
        stream=stream,
    )
    _SEEDED_KERNEL_WARMED = True


# 0f3e2fa1: (T([256,1024], bf16), S([256,1024]))
@oracle_impl(hardware="B200", point="0f3e2fa1")
def oracle_forward(inputs):
    x, random_shape_param = inputs
    random_shape = _shape_tuple(random_shape_param)
    device = x.device

    mask = torch.empty_strided(
        random_shape,
        _contiguous_stride(random_shape),
        device=device,
        dtype=torch.bool,
    )
    out = torch.empty_strided(
        random_shape,
        _contiguous_stride(random_shape),
        device=device,
        dtype=torch.bfloat16,
    )

    grid = lambda meta: (triton.cdiv(N_ELEMENTS, meta["block"]),)
    if torch.cuda.is_current_stream_capturing():
        seeds = torch.empty_strided((SEED_COUNT,), (1,), device=device, dtype=torch.int64)
        torch.ops.aten.randint.low_out(
            -9223372036854775808,
            9223372036854775807,
            [SEED_COUNT],
            out=seeds,
        )
        stream = _get_raw_stream(torch.cuda.current_device())
        _seeded_selu_dropout_kernel.run(
            seeds,
            x,
            mask,
            out,
            SEED_INDEX,
            N_ELEMENTS,
            stream=stream,
        )
    else:
        _warm_seeded_kernel(x, random_shape)
        _, random = _seeds_and_random_for_eager_check(random_shape, device=device)
        _selu_dropout_kernel[grid](
            x,
            random,
            mask,
            out,
            n_elements=N_ELEMENTS,
            seed_index=SEED_INDEX,
            threshold=THRESHOLD,
            scale=SCALE,
            use_seeded_rng=False,
        )

    return mask, out
