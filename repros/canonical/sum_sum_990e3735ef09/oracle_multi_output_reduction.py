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


@oracle_impl(hardware="H100", shapes="(T([4096, 4096], f32), T([4096], f32), T([8, 512, 4096], f32), T([8, 512, 1], f32), S([8, 512, 4096]), S([4096, 4096]))")
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
