"""
Oracle for pointwise_34dd8044bfe1.

Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the full
stochastic `inductor_seeds -> inductor_lookup_seed -> inductor_random(randn) ->
unsqueeze -> unsqueeze -> view` region by drawing the same two Inductor seeds,
materializing every `float32` randn element with Triton's `tl.randn(seed,
offset)` semantics, and returning the exact requested contiguous
`float32[12, 64, 64]` layout, whereas Inductor already lowers the observable
work to the same seed draw plus one fused random-generation kernel and treats
the view-only layout ops as metadata; Inductor cannot materially remove this
work because the fresh stochastic tensor and RNG-state advancement are required,
so the fix is BANDWIDTH_BOUND: do not add a fake constant/empty oracle; this
full-scope same-semantics RNG oracle records the measured stochastic
materialization floor for the required compile configurations.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch
import torch._inductor.inductor_prims  # noqa: F401 - registers prims.* RNG ops

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - keeps py_compile useful without Triton.
    triton = None
    tl = None

from oracle_harness import (
    oracle_impl,
    bench_oracle,
    bench_oracle_all_shapes,
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    has_stochastic_ops,
)


REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPRO_PATH = REPRO_DIR / "repro.py"
CLASSIFICATION = "BANDWIDTH_BOUND"

RANDOM_SOURCE_SHAPE = (12, 64, 1, 64)
RANDOM_NUMEL = math.prod(RANDOM_SOURCE_SHAPE)
DTYPE = torch.float32
BLOCK_N = 1024
SEED_LOW = -9223372036854775808
SEED_HIGH = 9223372036854775807


def get_inputs() -> list[Any]:
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance() -> torch.nn.Module:
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


def _contiguous_stride(shape: tuple[int, ...]) -> tuple[int, ...]:
    stride: list[int] = []
    running = 1
    for size in reversed(shape):
        stride.append(running)
        running *= size
    return tuple(reversed(stride))


def _validate_inputs(inputs: list[Any] | tuple[Any, ...]) -> tuple[int, ...]:
    if len(inputs) != 1:
        raise ValueError(f"{REPRO_ID} expects one shape input, got {len(inputs)}")
    shape_arg = inputs[0]
    if isinstance(shape_arg, torch.Size):
        out_shape = tuple(int(x) for x in shape_arg)
    elif isinstance(shape_arg, (list, tuple)):
        out_shape = tuple(int(x) for x in shape_arg)
    else:
        raise TypeError(f"expected shape list/tuple input, got {type(shape_arg).__name__}")
    if math.prod(out_shape) != RANDOM_NUMEL:
        raise ValueError(
            f"view shape {out_shape} has {math.prod(out_shape)} elements, "
            f"expected {RANDOM_NUMEL}"
        )
    return out_shape


def _source_has_inductor_random() -> bool:
    return "prims.inductor_random" in REPRO_PATH.read_text()


if triton is not None:

    @triton.jit
    def _randn_kernel(
        seeds_ptr,
        out_ptr,
        N: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < N
        seed = tl.load(seeds_ptr)
        values = tl.randn(seed, offsets.to(tl.uint32))
        tl.store(out_ptr + offsets, values, mask=mask)


@oracle_impl(hardware="H100", shapes="(S([12, 64, 64]))")
def oracle_forward(inputs: list[Any] | tuple[Any, ...]) -> torch.Tensor:
    """Run the full stochastic repro scope and return the exact output layout."""
    if triton is None:
        raise RuntimeError("Triton is required for oracle_layout.py")

    out_shape = _validate_inputs(inputs)
    device = torch.device("cuda", 0)

    # Repro.forward requests two seeds and uses the first for randn.  Inductor
    # lowers that prim to randint.low_out, so use the same seed draw here.
    seeds = torch.empty_strided((2,), (1,), device=device, dtype=torch.int64)
    torch.ops.aten.randint.low_out(SEED_LOW, SEED_HIGH, [2], out=seeds)
    out = torch.empty_strided(
        out_shape,
        _contiguous_stride(out_shape),
        device=device,
        dtype=DTYPE,
    )
    grid = (triton.cdiv(RANDOM_NUMEL, BLOCK_N),)
    _randn_kernel[grid](
        seeds,
        out,
        N=RANDOM_NUMEL,
        BLOCK=BLOCK_N,
        num_warps=4,
    )
    return out


def _as_tuple(value: Any) -> tuple[torch.Tensor, ...]:
    if isinstance(value, torch.Tensor):
        return (value,)
    if isinstance(value, (tuple, list)):
        return tuple(item for item in value if isinstance(item, torch.Tensor))
    return ()


def _randn_stats(tensor: torch.Tensor) -> tuple[float, float, bool]:
    values = tensor.float()
    finite = bool(torch.isfinite(values).all().item())
    mean = float(values.mean().item())
    std = float(values.std(unbiased=False).item())
    return mean, std, finite


@torch.no_grad()
def run_check(rtol: float, atol: float) -> bool:
    """Check full scope for stochastic randn without bitwise value comparison.

    The eager repro and oracle both draw fresh Inductor RNG seeds, so matching
    exact values would be the wrong invariant.  This check enforces output
    count, shape, dtype, stride, finite data, non-constant fresh draws, and
    randn distribution sanity for the full materialized output.
    """
    del rtol, atol  # Stochastic values are intentionally not allclose-checked.

    inputs = get_inputs()
    instance = get_repro_instance()

    expected = _as_tuple(instance(*inputs))
    actual = _as_tuple(oracle_forward(inputs))
    actual_again = _as_tuple(oracle_forward(inputs))
    torch.cuda.synchronize()

    if len(actual) != len(expected):
        print(
            f"  SCOPE_MISMATCH: oracle produces {len(actual)} outputs, "
            f"eager produces {len(expected)}"
        )
        print("Correctness: FAIL")
        return False

    all_ok = True
    for idx, (act, exp) in enumerate(zip(actual, expected)):
        shape_ok = act.shape == exp.shape
        dtype_ok = act.dtype == exp.dtype == DTYPE
        stride_ok = act.stride() == exp.stride()
        fresh_ok = idx < len(actual_again) and not torch.equal(act, actual_again[idx])

        exp_mean, exp_std, exp_finite = _randn_stats(exp)
        act_mean, act_std, act_finite = _randn_stats(act)
        distribution_ok = (
            exp_finite
            and act_finite
            and abs(exp_mean) < 0.06
            and abs(act_mean) < 0.06
            and 0.90 < exp_std < 1.10
            and 0.90 < act_std < 1.10
        )

        ok = shape_ok and dtype_ok and stride_ok and fresh_ok and distribution_ok
        all_ok = all_ok and ok
        status = "PASS" if ok else "FAIL"
        print(
            f"  output {idx}: {status} stochastic "
            f"(shape={list(act.shape)} dtype={act.dtype} stride={act.stride()} "
            f"eager_mean={exp_mean:.4f} eager_std={exp_std:.4f} "
            f"oracle_mean={act_mean:.4f} oracle_std={act_std:.4f} "
            f"fresh_draw={fresh_ok})"
        )
        if not shape_ok:
            print(f"    shape mismatch: oracle={list(act.shape)} eager={list(exp.shape)}")
        if not dtype_ok:
            print(f"    dtype mismatch: oracle={act.dtype} eager={exp.dtype}")
        if not stride_ok:
            print(f"    stride mismatch: oracle={act.stride()} eager={exp.stride()}")
        if not distribution_ok:
            print("    randn sanity check failed")

    print("  stochastic values are not compared bit-exactly; both paths draw fresh randn data")
    print(f"Correctness: {'PASS' if all_ok else 'FAIL'}")
    return all_ok


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
                        help="Accepted for template compatibility")
    parser.add_argument("--atol", type=float, default=1e-2,
                        help="Accepted for template compatibility")
    parser.add_argument("--warmup", type=int, default=25,
                        help="Warmup iterations for benchmark")
    parser.add_argument("--rep", type=int, default=200,
                        help="Repetitions for benchmark")
    parser.add_argument("--no-skip-stochastic", action="store_true",
                        help="Accepted; stochastic randn is always checked structurally")
    parser.add_argument("--all-shapes", action="store_true",
                        help="Benchmark across all shapes from shapes.txt")
    parser.add_argument("--show-hw", action="store_true",
                        help="Print GPU hardware info and exit")
    args = parser.parse_args()

    if args.show_hw:
        print(json.dumps(get_hardware_info(), indent=2))
        return

    if not args.check and not args.bench:
        args.check = args.bench = True

    inputs = get_inputs()
    instance = get_repro_instance()

    if has_stochastic_ops(REPRO_PATH) or _source_has_inductor_random():
        print(
            f"NOTE: {REPRO_ID} contains prims.inductor_random; "
            "checking layout and randn distribution, not exact random values"
        )

    if args.check:
        print(f"Checking {REPRO_ID}...")
        ok = run_check(rtol=args.rtol, atol=args.atol)
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
                    print(
                        f"WARNING: oracle is slower than compile for "
                        f"{result['repro_id']} (ratio={result['ratio']:.3f}x)"
                    )
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
                print(f"WARNING: oracle is slower than compile (ratio={result['ratio']:.3f}x)")


if __name__ == "__main__":
    main()
