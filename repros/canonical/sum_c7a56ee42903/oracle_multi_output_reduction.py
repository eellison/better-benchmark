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


def oracle_forward(inputs: tuple[object, ...] | list[object]) -> torch.Tensor:
    (x,) = inputs
    return oracle_fused(x)


def reference_output(inputs: tuple[object, ...]) -> torch.Tensor:
    module = _load_repro_module()
    model = module.Repro().cuda()
    with torch.no_grad():
        return model(*inputs)


def _max_diff(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    diff = (actual.float() - expected.float()).abs()
    rel = diff / (expected.float().abs() + 1.0e-8)
    return float(diff.max().item()), float(rel.max().item())


def run_check(
    rtol: float,
    atol: float,
    block_elements: int,
    num_warps: int,
    paired: bool,
) -> bool:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Triton oracle check")

    torch.manual_seed(0)
    inputs = make_inputs()
    with torch.no_grad():
        ref = reference_output(inputs)
        actual = oracle_fused(
            *inputs,
            block_elements=block_elements,
            num_warps=num_warps,
            paired=paired,
        )
        torch.cuda.synchronize()

    shape_ok = actual.shape == ref.shape
    dtype_ok = actual.dtype == ref.dtype
    stride_ok = actual.stride() == ref.stride()
    max_abs, max_rel = _max_diff(actual, ref)
    close_ok = torch.allclose(actual.float(), ref.float(), rtol=rtol, atol=atol)
    ok = shape_ok and dtype_ok and stride_ok and close_ok
    print(
        f"output[0]: shape={list(actual.shape)} dtype={actual.dtype} stride={actual.stride()} "
        f"max_abs={max_abs:.6e} max_rel={max_rel:.6e} allclose={close_ok} "
        f"shape_match={shape_ok} stride_match={stride_ok} dtype_match={dtype_ok}"
    )
    print(f"Correctness: {'PASS' if ok else 'FAIL'}")
    return ok


def _compile_with_config(model: torch.nn.Module, inputs: tuple[object, ...], config: dict[str, object]):
    import torch._dynamo
    import torch._inductor.config as inductor_config

    torch._dynamo.reset()
    with inductor_config.patch(config):
        compiled = torch.compile(model)
        for _ in range(3):
            compiled(*inputs)
        torch.cuda.synchronize()
    return compiled


def run_bench(
    rep: int,
    warmup: int,
    no_compile: bool,
    block_elements: int,
    num_warps: int,
    paired: bool,
) -> dict[str, float]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Triton oracle benchmark")

    torch.manual_seed(0)
    inputs = make_inputs()
    (x,) = inputs
    partials, out = _make_workspace(x.device, block_elements)

    timings: dict[str, float] = {}
    with torch.no_grad():
        _oracle_into(x, partials, out, block_elements, num_warps, paired)
        torch.cuda.synchronize()
        oracle_us = triton.testing.do_bench(
            lambda: _oracle_into(x, partials, out, block_elements, num_warps, paired),
            warmup=warmup,
            rep=rep,
            return_mode="min",
        ) * 1000.0
    timings["oracle_fused"] = oracle_us
    print(
        f"oracle_fused full-scope sum: {oracle_us:.3f} us "
        f"(block_elements={block_elements}, num_warps={num_warps}, paired={paired})"
    )

    if not no_compile:
        module = _load_repro_module()
        for label, config in COMPILE_CONFIGS:
            model = module.Repro().cuda()
            with torch.no_grad():
                compiled = _compile_with_config(model, inputs, config)
                compiled_us = triton.testing.do_bench(
                    lambda: compiled(*inputs),
                    warmup=warmup,
                    rep=rep,
                    return_mode="min",
                ) * 1000.0
            timings[label] = compiled_us
            print(f"torch.compile {label}: {compiled_us:.3f} us")

    best_required_compile = min(
        (timings[label] for label, _ in COMPILE_CONFIGS if label in timings),
        default=float("inf"),
    )
    best_reference = min(best_required_compile, HISTORICAL_BEST_COMPILE_US)
    valid_floor = oracle_us < best_reference
    print(f"historical best_compile_us: {HISTORICAL_BEST_COMPILE_US:.3f} us")
    print(f"Valid floor: {'yes' if valid_floor else 'no (diagnosis-only)'}")
    print(
        json.dumps(
            {
                "repro_id": REPRO_ID,
                "oracle_us": round(oracle_us, 3),
                "best_required_compile_us": (
                    round(best_required_compile, 3)
                    if best_required_compile != float("inf")
                    else None
                ),
                "historical_best_compile_us": round(HISTORICAL_BEST_COMPILE_US, 3),
                "valid_floor": valid_floor,
                "classification": "BANDWIDTH_BOUND",
            }
        )
    )
    return timings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="run correctness check against repro.py")
    parser.add_argument("--bench", action="store_true", help="run timing benchmark")
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=2e-1)
    parser.add_argument("--rep", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--block-elements", type=int, default=DEFAULT_BLOCK_ELEMENTS)
    parser.add_argument("--num-warps", type=int, default=DEFAULT_NUM_WARPS)
    parser.add_argument("--segment-mode", action="store_true", help="use one program per (n, c, tile)")
    parser.add_argument("--no-compile", action="store_true", help="only benchmark the oracle")
    args = parser.parse_args()

    if not args.check and not args.bench:
        args.check = True
        args.bench = True

    if args.check and not run_check(
        rtol=args.rtol,
        atol=args.atol,
        block_elements=args.block_elements,
        num_warps=args.num_warps,
        paired=not args.segment_mode,
    ):
        sys.exit(1)
    if args.bench:
        run_bench(
            rep=args.rep,
            warmup=args.warmup,
            no_compile=args.no_compile,
            block_elements=args.block_elements,
            num_warps=args.num_warps,
            paired=not args.segment_mode,
        )


if __name__ == "__main__":
    with torch.no_grad():
        main()
