"""
Full-scope Triton oracle for sum_c7a56ee42903.

The compiled repro computes:

    out[c] = sum_{n,h,w} arg146_1[n, c, h, w]

for one contiguous f32[8, 2, 640, 959] input and returns one contiguous f32[2]
tensor. This oracle covers that complete computation with the same input,
output shape, dtype, and stride; it is not a reduction-subset microbenchmark.

Gap diagnosis (classification: BANDWIDTH_BOUND): the oracle differs from
Inductor only by using a hand-written split reduction over the contiguous
channel slabs, storing one partial per tile before a tiny finalizer. Inductor
already lowers this simple sum as a fused reduction over the same full input,
so there is no missing scheduler fusion, scatter-reduce, algebraic rewrite, or
dependent epilogue fusion to unlock. The relevant Inductor change is tuning the
existing reduction template for this very small output and large input, but if
the full-scope Triton timing does not beat both required compile configs and
the historical best, this artifact is diagnosis-only rather than a true floor.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl

from oracle_harness import (
    oracle_impl,
    bench_oracle,
    bench_oracle_all_shapes,
    check_oracle,
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    has_stochastic_ops,
)


REPRO_ID = "sum_c7a56ee42903"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"

N = 8
C = 2
H = 640
W = 959
HW = H * W
TOTAL_SEGMENTS = N * C
HISTORICAL_BEST_COMPILE_US = 19.519999623298645
DEFAULT_BLOCK_ELEMENTS = 4096
DEFAULT_NUM_WARPS = 1

COMPILE_CONFIGS = [
    ("coordinate_descent_tuning=True", {"coordinate_descent_tuning": True}),
    (
        (
            "combo_kernels=True,combo_kernel_per_subkernel_blocks=True,"
            "coordinate_descent_tuning=True,benchmark_combo_kernel=True,"
            "triton.multi_kernel=3"
        ),
        {
            "combo_kernels": True,
            "combo_kernel_per_subkernel_blocks": True,
            "coordinate_descent_tuning": True,
            "benchmark_combo_kernel": True,
            "triton.multi_kernel": 3,
        },
    ),
]

sys.path.insert(0, str(REPO_ROOT))


@triton.jit
def _sum_partials_kernel(
    x_ptr,
    partials_ptr,
    N_: tl.constexpr,
    C_: tl.constexpr,
    HW_: tl.constexpr,
    NUM_TILES_PER_SEGMENT: tl.constexpr,
    BLOCK_ELEMENTS: tl.constexpr,
):
    segment = tl.program_id(0)
    tile = tl.program_id(1)
    offsets = tile * BLOCK_ELEMENTS + tl.arange(0, BLOCK_ELEMENTS)
    mask = offsets < HW_
    values = tl.load(x_ptr + segment * HW_ + offsets, mask=mask, other=0.0).to(tl.float32)
    partial = tl.sum(values, axis=0)

    c = segment - (segment // C_) * C_
    partial_index = c * (N_ * NUM_TILES_PER_SEGMENT) + (segment // C_) * NUM_TILES_PER_SEGMENT + tile
    tl.store(partials_ptr + partial_index, partial)


@triton.jit
def _sum_pair_partials_kernel(
    x_ptr,
    partials_ptr,
    N_: tl.constexpr,
    C_: tl.constexpr,
    HW_: tl.constexpr,
    NUM_TILES_PER_N: tl.constexpr,
    BLOCK_ELEMENTS: tl.constexpr,
):
    n = tl.program_id(0)
    tile = tl.program_id(1)
    offsets = tile * BLOCK_ELEMENTS + tl.arange(0, BLOCK_ELEMENTS)
    mask = offsets < HW_
    base = n * C_ * HW_ + offsets

    values0 = tl.load(x_ptr + base, mask=mask, other=0.0).to(tl.float32)
    values1 = tl.load(x_ptr + base + HW_, mask=mask, other=0.0).to(tl.float32)
    partial0 = tl.sum(values0, axis=0)
    partial1 = tl.sum(values1, axis=0)

    partial_offset = n * NUM_TILES_PER_N + tile
    partials_per_channel = N_ * NUM_TILES_PER_N
    tl.store(partials_ptr + partial_offset, partial0)
    tl.store(partials_ptr + partials_per_channel + partial_offset, partial1)


@triton.jit
def _finalize_partials_kernel(
    partials_ptr,
    out_ptr,
    NUM_PARTIALS_PER_CHANNEL: tl.constexpr,
    BLOCK_PARTIALS: tl.constexpr,
):
    c = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_PARTIALS)
    mask = offsets < NUM_PARTIALS_PER_CHANNEL
    values = tl.load(partials_ptr + c * NUM_PARTIALS_PER_CHANNEL + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + c, tl.sum(values, axis=0))


def _load_repro_module():
    spec = importlib.util.spec_from_file_location(f"{REPRO_ID}_repro", REPRO_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load repro module from {REPRO_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_inputs() -> tuple[object, ...]:
    module = _load_repro_module()
    return tuple(x.cuda() if isinstance(x, torch.Tensor) else x for x in module.make_inputs())


def _num_tiles_per_segment(block_elements: int) -> int:
    return triton.cdiv(HW, block_elements)


def _num_partials_per_channel(block_elements: int) -> int:
    return N * _num_tiles_per_segment(block_elements)


def _make_workspace(
    device: torch.device,
    block_elements: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    partials = torch.empty((C, _num_partials_per_channel(block_elements)), device=device, dtype=torch.float32)
    out = torch.empty_strided((C,), (1,), device=device, dtype=torch.float32)
    return partials, out


def _validate_input(x: torch.Tensor, out: torch.Tensor | None = None) -> None:
    if x.device.type != "cuda":
        raise RuntimeError("Triton oracle requires CUDA inputs")
    if tuple(x.shape) != (N, C, H, W):
        raise ValueError(f"unexpected input shape: {tuple(x.shape)}")
    if x.dtype != torch.float32:
        raise ValueError(f"unexpected input dtype: {x.dtype}")
    if not x.is_contiguous():
        raise ValueError(f"oracle expects the default contiguous input stride, got {x.stride()}")
    if out is not None:
        if tuple(out.shape) != (C,) or out.stride() != (1,) or out.dtype != torch.float32:
            raise ValueError("unexpected output buffer layout")


def _oracle_into(
    x: torch.Tensor,
    partials: torch.Tensor,
    out: torch.Tensor,
    block_elements: int,
    num_warps: int,
    paired: bool,
) -> torch.Tensor:
    _validate_input(x, out)
    if block_elements <= 0 or (block_elements & (block_elements - 1)) != 0:
        raise ValueError("block_elements must be a positive power of two")

    num_tiles = _num_tiles_per_segment(block_elements)
    num_partials = _num_partials_per_channel(block_elements)
    if tuple(partials.shape) != (C, num_partials) or partials.dtype != torch.float32:
        raise ValueError("unexpected partials workspace layout")

    if paired:
        _sum_pair_partials_kernel[(N, num_tiles)](
            x,
            partials,
            N_=N,
            C_=C,
            HW_=HW,
            NUM_TILES_PER_N=num_tiles,
            BLOCK_ELEMENTS=block_elements,
            num_warps=num_warps,
        )
    else:
        _sum_partials_kernel[(TOTAL_SEGMENTS, num_tiles)](
            x,
            partials,
            N_=N,
            C_=C,
            HW_=HW,
            NUM_TILES_PER_SEGMENT=num_tiles,
            BLOCK_ELEMENTS=block_elements,
            num_warps=num_warps,
        )
    _finalize_partials_kernel[(C,)](
        partials,
        out,
        NUM_PARTIALS_PER_CHANNEL=num_partials,
        BLOCK_PARTIALS=triton.next_power_of_2(num_partials),
        num_warps=1,
    )
    return out


def oracle_fused(
    x: torch.Tensor,
    *,
    block_elements: int = DEFAULT_BLOCK_ELEMENTS,
    num_warps: int = DEFAULT_NUM_WARPS,
    paired: bool = True,
) -> torch.Tensor:
    partials, out = _make_workspace(x.device, block_elements)
    return _oracle_into(x, partials, out, block_elements, num_warps, paired)


@oracle_impl(hardware="H100", shapes="(T([8, 2, 640, 959], f32))")
def oracle_forward(inputs: tuple[object, ...] | list[object]) -> torch.Tensor:
    (x,) = inputs
    return oracle_fused(x)


def reference_output(inputs: tuple[object, ...]) -> torch.Tensor:
    module = _load_repro_module()
    model = module.Repro().cuda()
    with torch.no_grad():
        return model(*inputs)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=f"Oracle for {REPRO_ID}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--check", action="store_true",
                        help="Verify correctness against eager Repro")
    parser.add_argument("--bench", action="store_true",
                        help="Benchmark oracle vs torch.compile")
    parser.add_argument("--rtol", type=float, default=1e-2,
                        help="Relative tolerance for correctness check")
    parser.add_argument("--atol", type=float, default=1e-2,
                        help="Absolute tolerance for correctness check")
    parser.add_argument("--warmup", type=int, default=25,
                        help="Warmup iterations for benchmark")
    parser.add_argument("--rep", type=int, default=200,
                        help="Repetitions for benchmark")
    parser.add_argument("--no-skip-stochastic", action="store_true",
                        help="Disable auto-detection and skipping of stochastic outputs")
    parser.add_argument("--all-shapes", action="store_true",
                        help="Benchmark across all shapes from shapes.txt")
    parser.add_argument("--show-hw", action="store_true",
                        help="Print GPU hardware info and exit")
    args = parser.parse_args()

    if args.show_hw:
        import json
        print(json.dumps(get_hardware_info(), indent=2))
        return

    if not args.check and not args.bench:
        args.check = args.bench = True

    inputs = _harness_get_inputs(REPRO_DIR)
    instance = _harness_get_repro_instance(REPRO_DIR)

    if has_stochastic_ops(REPRO_PATH):
        print(f"NOTE: {REPRO_ID} contains stochastic ops; affected outputs will be auto-skipped")

    if args.check:
        print(f"Checking {REPRO_ID}...")
        ok = check_oracle(
            oracle_forward,
            instance,
            inputs,
            atol=args.atol,
            rtol=args.rtol,
            skip_stochastic=not args.no_skip_stochastic,
        )
        status = "PASS" if ok else "FAIL"
        print(f"Correctness: {status}")
        if not ok:
            sys.exit(1)

    if args.bench:
        print(f"Benchmarking {REPRO_ID}...")
        if args.all_shapes:
            results = bench_oracle_all_shapes(
                oracle_forward,
                REPRO_DIR,
                REPRO_ID,
                warmup=args.warmup,
                rep=args.rep,
            )
            for result in results:
                if result["status"] == "BAD_ORACLE":
                    print(f"WARNING: oracle is slower than compile "
                          f"for {result['repro_id']} (ratio={result['ratio']:.3f}x)")
        else:
            result = bench_oracle(
                oracle_forward,
                instance,
                inputs,
                REPRO_ID,
                warmup=args.warmup,
                rep=args.rep,
            )
            if result["status"] == "BAD_ORACLE":
                print(f"WARNING: oracle is slower than compile "
                      f"(ratio={result['ratio']:.3f}x)")


if __name__ == "__main__":
    main()
