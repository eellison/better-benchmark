"""
Full-scope Triton oracle for sum_sum_990e3735ef09.

Gap diagnosis (classification: BANDWIDTH_BOUND): The oracle consumes the same
`mm_2`, scale vector, normalized activation tensor, row scale tensor, and shape
parameters as repro.py, then computes the two sibling row reductions
`sum(mm_2 * arg12_1)` and `sum(mm_2 * arg12_1 * arg114_1)` inside one
multi-accumulator Triton row kernel before writing the dependent full
`[4096, 4096]` permuted output with matching dtype and stride. This differs
from Inductor by forcing the reductions and their full epilogue into one
row-owned schedule with no intermediate reduction tensors; Inductor cannot use
this exact schedule as a guaranteed win today because its reduction scheduler
must balance occupancy, vector width, and dependent epilogue traffic across many
row sizes rather than specializing this single Albert layer-norm backward shape.
If the measured oracle does not beat both required compile configs and the
historical best compile timing, this artifact is diagnosis-only and the
practical fix classification remains BANDWIDTH_BOUND.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl


REPRO_ID = "sum_sum_990e3735ef09"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"

ROWS = 8 * 512
K = 4096
HISTORICAL_BEST_COMPILE_US = 39.29600119590759

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
def _full_scope_row_kernel(
    mm_ptr,
    scale_ptr,
    normed_ptr,
    row_scale_ptr,
    out_ptr,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_K)
    offsets = row * BLOCK_K + cols

    mm = tl.load(mm_ptr + offsets).to(tl.float32)
    scale = tl.load(scale_ptr + cols).to(tl.float32)
    normed = tl.load(normed_ptr + offsets).to(tl.float32)
    x = mm * scale

    sum_x = tl.sum(x, axis=0)
    sum_x_normed = tl.sum(x * normed, axis=0)

    row_scale = tl.load(row_scale_ptr + row).to(tl.float32)
    out = row_scale * (x * BLOCK_K - sum_x - normed * sum_x_normed)
    tl.store(out_ptr + offsets, out)


@triton.jit
def _full_scope_row_tiled_kernel(
    mm_ptr,
    scale_ptr,
    normed_ptr,
    row_scale_ptr,
    out_ptr,
    BLOCK_K: tl.constexpr,
    NUM_TILES: tl.constexpr,
    K_: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_K)
    sum_x = tl.full((), 0.0, tl.float32)
    sum_x_normed = tl.full((), 0.0, tl.float32)

    for tile in tl.static_range(0, NUM_TILES):
        k = tile * BLOCK_K + cols
        mask = k < K_
        offsets = row * K_ + k
        mm = tl.load(mm_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        scale = tl.load(scale_ptr + k, mask=mask, other=0.0).to(tl.float32)
        normed = tl.load(normed_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        x = mm * scale
        sum_x += tl.sum(x, axis=0)
        sum_x_normed += tl.sum(x * normed, axis=0)

    row_scale = tl.load(row_scale_ptr + row).to(tl.float32)
    for tile in tl.static_range(0, NUM_TILES):
        k = tile * BLOCK_K + cols
        mask = k < K_
        offsets = row * K_ + k
        mm = tl.load(mm_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        scale = tl.load(scale_ptr + k, mask=mask, other=0.0).to(tl.float32)
        normed = tl.load(normed_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        x = mm * scale
        out = row_scale * (x * K_ - sum_x - normed * sum_x_normed)
        tl.store(out_ptr + offsets, out, mask=mask)


@triton.jit
def _partial_reduce_kernel(
    mm_ptr,
    scale_ptr,
    normed_ptr,
    partial_sum_x_ptr,
    partial_sum_x_normed_ptr,
    BLOCK_K: tl.constexpr,
    NUM_TILES: tl.constexpr,
    K_: tl.constexpr,
):
    row = tl.program_id(0)
    tile = tl.program_id(1)
    cols = tile * BLOCK_K + tl.arange(0, BLOCK_K)
    mask = cols < K_
    offsets = row * K_ + cols

    mm = tl.load(mm_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    scale = tl.load(scale_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    normed = tl.load(normed_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    x = mm * scale

    partial_offset = row * NUM_TILES + tile
    tl.store(partial_sum_x_ptr + partial_offset, tl.sum(x, axis=0))
    tl.store(partial_sum_x_normed_ptr + partial_offset, tl.sum(x * normed, axis=0))


@triton.jit
def _finalize_reduce_kernel(
    partial_sum_x_ptr,
    partial_sum_x_normed_ptr,
    sum_x_ptr,
    sum_x_normed_ptr,
    NUM_TILES: tl.constexpr,
    BLOCK_TILES: tl.constexpr,
):
    row = tl.program_id(0)
    tiles = tl.arange(0, BLOCK_TILES)
    mask = tiles < NUM_TILES
    offsets = row * NUM_TILES + tiles

    partial_sum_x = tl.load(partial_sum_x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    partial_sum_x_normed = tl.load(
        partial_sum_x_normed_ptr + offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    tl.store(sum_x_ptr + row, tl.sum(partial_sum_x, axis=0))
    tl.store(sum_x_normed_ptr + row, tl.sum(partial_sum_x_normed, axis=0))


@triton.jit
def _split_epilogue_kernel(
    mm_ptr,
    scale_ptr,
    normed_ptr,
    row_scale_ptr,
    sum_x_ptr,
    sum_x_normed_ptr,
    out_ptr,
    BLOCK_K: tl.constexpr,
    K_: tl.constexpr,
):
    row = tl.program_id(0)
    tile = tl.program_id(1)
    cols = tile * BLOCK_K + tl.arange(0, BLOCK_K)
    mask = cols < K_
    offsets = row * K_ + cols

    mm = tl.load(mm_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    scale = tl.load(scale_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    normed = tl.load(normed_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    sum_x = tl.load(sum_x_ptr + row).to(tl.float32)
    sum_x_normed = tl.load(sum_x_normed_ptr + row).to(tl.float32)
    row_scale = tl.load(row_scale_ptr + row).to(tl.float32)

    x = mm * scale
    out = row_scale * (x * K_ - sum_x - normed * sum_x_normed)
    tl.store(out_ptr + offsets, out, mask=mask)


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


def get_inputs() -> tuple[object, ...]:
    """Load the exact inputs used by repro.py."""
    return make_inputs()


def get_repro_instance() -> torch.nn.Module:
    """Create the exact eager reference module from repro.py."""
    return _load_repro_module().Repro().cuda()


def oracle_fused(
    mm_2: torch.Tensor,
    arg12_1: torch.Tensor,
    arg114_1: torch.Tensor,
    arg124_1: torch.Tensor,
    _shape_param_0,
    _shape_param_1,
) -> torch.Tensor:
    assert mm_2.shape == (ROWS, K)
    assert arg12_1.shape == (K,)
    assert arg114_1.shape == (8, 512, K)
    assert arg124_1.shape == (8, 512, 1)
    assert mm_2.is_contiguous()
    assert arg12_1.is_contiguous()
    assert arg114_1.is_contiguous()
    assert arg124_1.is_contiguous()

    out = torch.empty_strided(
        (K, ROWS),
        (1, K),
        device=mm_2.device,
        dtype=torch.float32,
    )
    _full_scope_row_kernel[(ROWS,)](
        mm_2,
        arg12_1,
        arg114_1,
        arg124_1,
        out,
        BLOCK_K=K,
        num_warps=8,
    )
    return out


def oracle_forward(inputs: tuple[object, ...]) -> torch.Tensor:
    """Run the full repro.py computation with the same input/output contract."""
    return oracle_fused(*inputs)


def _oracle_fused_tiled(inputs: tuple[object, ...], block_k: int, num_warps: int) -> torch.Tensor:
    mm_2, arg12_1, arg114_1, arg124_1, _shape_param_0, _shape_param_1 = inputs
    out = torch.empty_strided(
        (K, ROWS),
        (1, K),
        device=mm_2.device,
        dtype=torch.float32,
    )
    _full_scope_row_tiled_kernel[(ROWS,)](
        mm_2,
        arg12_1,
        arg114_1,
        arg124_1,
        out,
        BLOCK_K=block_k,
        NUM_TILES=triton.cdiv(K, block_k),
        K_=K,
        num_warps=num_warps,
    )
    return out


def _oracle_fused_split(
    inputs: tuple[object, ...],
    reduce_block_k: int,
    epilogue_block_k: int,
    reduce_warps: int,
    epilogue_warps: int,
) -> torch.Tensor:
    mm_2, arg12_1, arg114_1, arg124_1, _shape_param_0, _shape_param_1 = inputs
    num_tiles = triton.cdiv(K, reduce_block_k)
    partial_sum_x = torch.empty((ROWS, num_tiles), device=mm_2.device, dtype=torch.float32)
    partial_sum_x_normed = torch.empty((ROWS, num_tiles), device=mm_2.device, dtype=torch.float32)
    sum_x = torch.empty((ROWS,), device=mm_2.device, dtype=torch.float32)
    sum_x_normed = torch.empty((ROWS,), device=mm_2.device, dtype=torch.float32)
    out = torch.empty_strided(
        (K, ROWS),
        (1, K),
        device=mm_2.device,
        dtype=torch.float32,
    )

    _partial_reduce_kernel[(ROWS, num_tiles)](
        mm_2,
        arg12_1,
        arg114_1,
        partial_sum_x,
        partial_sum_x_normed,
        BLOCK_K=reduce_block_k,
        NUM_TILES=num_tiles,
        K_=K,
        num_warps=reduce_warps,
    )
    _finalize_reduce_kernel[(ROWS,)](
        partial_sum_x,
        partial_sum_x_normed,
        sum_x,
        sum_x_normed,
        NUM_TILES=num_tiles,
        BLOCK_TILES=triton.next_power_of_2(num_tiles),
        num_warps=1,
    )
    _split_epilogue_kernel[(ROWS, triton.cdiv(K, epilogue_block_k))](
        mm_2,
        arg12_1,
        arg114_1,
        arg124_1,
        sum_x,
        sum_x_normed,
        out,
        BLOCK_K=epilogue_block_k,
        K_=K,
        num_warps=epilogue_warps,
    )
    return out


def reference_outputs(inputs: tuple[object, ...]) -> torch.Tensor:
    model = get_repro_instance()
    with torch.no_grad():
        return model(*inputs)


def _max_diff(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    diff = (actual.float() - expected.float()).abs()
    rel = diff / (expected.float().abs() + 1e-8)
    return diff.max().item(), rel.max().item()


def run_check(rtol: float, atol: float) -> bool:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Triton oracle check")

    torch.manual_seed(0)
    inputs = get_inputs()
    with torch.no_grad():
        ref = reference_outputs(inputs)
        actual = oracle_forward(inputs)
        torch.cuda.synchronize()

    max_abs, max_rel = _max_diff(actual, ref)
    output_ok = torch.allclose(actual.float(), ref.float(), rtol=rtol, atol=atol)
    stride_ok = actual.stride() == ref.stride()
    shape_ok = actual.shape == ref.shape
    dtype_ok = actual.dtype == ref.dtype
    ok = output_ok and stride_ok and shape_ok and dtype_ok
    print(
        f"output: shape={list(actual.shape)} stride={actual.stride()} "
        f"dtype={actual.dtype} max_abs={max_abs:.6e} max_rel={max_rel:.6e} "
        f"allclose={output_ok} shape_match={shape_ok} "
        f"stride_match={stride_ok} dtype_match={dtype_ok}"
    )
    print(f"Correctness: {'PASS' if ok else 'FAIL'}")
    return ok


def _compile_with_config(model: torch.nn.Module, inputs: tuple[object, ...], config: dict[str, object]):
    import torch._dynamo
    import torch._inductor.config as inductor_config

    torch._dynamo.reset()
    with inductor_config.patch(config), torch.no_grad():
        compiled = torch.compile(model)
        for _ in range(5):
            compiled(*inputs)
        torch.cuda.synchronize()
    return compiled


def _bench_callable(fn, warmup: int, rep: int) -> float:
    return triton.testing.do_bench(
        fn,
        warmup=warmup,
        rep=rep,
        return_mode="min",
    ) * 1000.0


def run_bench(rep: int, warmup: int, no_compile: bool, tune_variants: bool) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Triton oracle benchmark")

    torch.manual_seed(0)
    inputs = get_inputs()
    timings: dict[str, float] = {}

    with torch.no_grad():
        oracle_forward(inputs)
        torch.cuda.synchronize()
        oracle_us = _bench_callable(lambda: oracle_forward(inputs), warmup, rep)
    timings["oracle_fused"] = oracle_us
    print(f"oracle_fused full-scope row dual reduction + transposed epilogue: {oracle_us:.3f} us")

    if tune_variants:
        for block_k, num_warps in ((2048, 8), (1024, 4)):
            with torch.no_grad():
                _oracle_fused_tiled(inputs, block_k, num_warps)
                torch.cuda.synchronize()
                variant_us = _bench_callable(
                    lambda bk=block_k, nw=num_warps: _oracle_fused_tiled(inputs, bk, nw),
                    warmup,
                    rep,
                )
            label = f"diagnostic_tiled_block{block_k}_warps{num_warps}"
            timings[label] = variant_us
            print(f"{label}: {variant_us:.3f} us")
        for reduce_block_k, epilogue_block_k, reduce_warps, epilogue_warps in (
            (1024, 1024, 4, 4),
            (1024, 512, 4, 4),
            (512, 512, 4, 4),
            (2048, 1024, 8, 4),
            (1024, 2048, 4, 8),
        ):
            with torch.no_grad():
                _oracle_fused_split(
                    inputs,
                    reduce_block_k,
                    epilogue_block_k,
                    reduce_warps,
                    epilogue_warps,
                )
                torch.cuda.synchronize()
                variant_us = _bench_callable(
                    lambda rb=reduce_block_k, eb=epilogue_block_k, rw=reduce_warps, ew=epilogue_warps: (
                        _oracle_fused_split(inputs, rb, eb, rw, ew)
                    ),
                    warmup,
                    rep,
                )
            label = (
                f"diagnostic_split_reduce{reduce_block_k}_epilogue{epilogue_block_k}_"
                f"warps{reduce_warps}_{epilogue_warps}"
            )
            timings[label] = variant_us
            print(f"{label}: {variant_us:.3f} us")

    if not no_compile:
        module = _load_repro_module()
        for label, config in COMPILE_CONFIGS:
            model = module.Repro().cuda()
            with torch.no_grad():
                compiled = _compile_with_config(model, inputs, config)
                compiled_us = _bench_callable(lambda: compiled(*inputs), warmup, rep)
            timings[label] = compiled_us
            print(f"torch.compile {label}: {compiled_us:.3f} us")

    required_compile_values = [timings[label] for label, _ in COMPILE_CONFIGS if label in timings]
    best_required_compile = min(required_compile_values) if required_compile_values else math.inf
    comparison = min(best_required_compile, HISTORICAL_BEST_COMPILE_US)
    valid_floor = oracle_us < comparison
    print(f"historical best_compile_us: {HISTORICAL_BEST_COMPILE_US:.3f} us")
    print(f"Valid floor: {'yes' if valid_floor else 'no (diagnosis-only)'}")

    result = {
        "repro_id": REPRO_ID,
        "oracle_us": round(oracle_us, 3),
        "best_required_compile_us": (
            round(best_required_compile, 3) if best_required_compile != math.inf else None
        ),
        "historical_best_compile_us": round(HISTORICAL_BEST_COMPILE_US, 3),
        "valid_floor": valid_floor,
        "classification": "BANDWIDTH_BOUND",
        "compile_configs_us": {
            label: round(timings[label], 3) for label, _ in COMPILE_CONFIGS if label in timings
        },
    }
    print(json.dumps(result))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="run correctness check against repro.py")
    parser.add_argument("--bench", action="store_true", help="run timing benchmark")
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument("--atol", type=float, default=1e-2)
    parser.add_argument("--rep", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--no-compile", action="store_true", help="only benchmark the oracle")
    parser.add_argument(
        "--tune-variants",
        action="store_true",
        help="also time diagnostic lower-register tiled variants",
    )
    args = parser.parse_args()

    if not args.check and not args.bench:
        args.check = True
        args.bench = True

    if args.check and not run_check(rtol=args.rtol, atol=args.atol):
        sys.exit(1)
    if args.bench:
        run_bench(
            rep=args.rep,
            warmup=args.warmup,
            no_compile=args.no_compile,
            tune_variants=args.tune_variants,
        )


if __name__ == "__main__":
    with torch.no_grad():
        main()
