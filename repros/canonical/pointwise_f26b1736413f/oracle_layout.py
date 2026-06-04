"""
Oracle for pointwise_f26b1736413f.

Gap diagnosis (classification: SCHEDULER_FUSION): this oracle covers the full
stochastic layout-only repro by using the compiled-equivalent seed tensor for
`prims.inductor_seeds` and one Triton kernel that generates three independent
`tl.randn(seed_i, flat_offset)` streams, casts each stream to fp16 on store, and
returns the three contiguous `[12, 64, 64]` tensors. Inductor currently lowers
the three sibling `inductor_random -> convert_element_type ->
singleton view/permute/view` chains as three separate pointwise RNG kernels
even though the layout ops collapse to contiguous views and the seed offsets
are independent. Inductor cannot do this today because its scheduler does not
fuse independent multi-output RNG pointwise producers that read different
offsets from the same seed tensor; the fix is SCHEDULER_FUSION: allow one
multi-output pointwise RNG kernel over a common shape with separate seed loads
and output stores.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None

from oracle_harness import (
    bench_oracle,
    bench_oracle_all_shapes,
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
)


REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPRO_PATH = REPRO_DIR / "repro.py"

OUTPUT_SHAPE = (12, 64, 64)
OUTPUT_STRIDE = (4096, 64, 1)
NUMEL = math.prod(OUTPUT_SHAPE)
BLOCK_N = 128


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _randn3_fp16_kernel(
        seeds_ptr,
        out0_ptr,
        out1_ptr,
        out2_ptr,
        N: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < N
        random_offsets = offsets.to(tl.uint32)

        seed0 = tl.load(seeds_ptr + 0)
        seed1 = tl.load(seeds_ptr + 1)
        seed2 = tl.load(seeds_ptr + 2)

        vals0 = tl.randn(seed0, random_offsets)
        vals1 = tl.randn(seed1, random_offsets)
        vals2 = tl.randn(seed2, random_offsets)

        tl.store(out0_ptr + offsets, vals0, mask=mask)
        tl.store(out1_ptr + offsets, vals1, mask=mask)
        tl.store(out2_ptr + offsets, vals2, mask=mask)


def _shape_tuple(value) -> tuple[int, ...]:
    if isinstance(value, torch.Size):
        return tuple(value)
    if isinstance(value, (list, tuple)):
        return tuple(int(dim) for dim in value)
    raise TypeError(f"expected shape-param list/tuple, got {type(value).__name__}")


def _validate_inputs(inputs) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    if len(inputs) != 3:
        raise ValueError(f"{REPRO_ID} expects three shape-param inputs, got {len(inputs)}")
    shapes = tuple(_shape_tuple(inp) for inp in inputs)
    for idx, shape in enumerate(shapes):
        if shape != OUTPUT_SHAPE:
            raise ValueError(
                f"input shape param {idx} is {shape}, expected fixed repro shape {OUTPUT_SHAPE}"
            )
    return shapes


def _make_output(shape: tuple[int, ...], device: torch.device) -> torch.Tensor:
    return torch.empty_strided(
        shape,
        OUTPUT_STRIDE,
        device=device,
        dtype=torch.float16,
    )


def _make_inductor_seed_tensor(device: torch.device) -> torch.Tensor:
    """Match compiled Inductor's lowering of `prims.inductor_seeds.default(3)`."""
    seeds = torch.empty_strided((3,), (1,), device=device, dtype=torch.int64)
    torch.ops.aten.randint.low_out(
        -9223372036854775808,
        9223372036854775807,
        [3],
        out=seeds,
    )
    return seeds


def oracle_forward(inputs):
    """Run the full-scope stochastic oracle.

    The repro inputs are only three shape parameters. The user-visible work is
    three independent f32 `randn` tensors, each cast to fp16 and returned through
    metadata-only singleton view/permute/view ops. This oracle preserves the
    same output arity, shape, dtype, and contiguous layout while fusing the three
    RNG streams into one Triton launch.
    """
    if triton is None:
        raise RuntimeError("triton is required for this oracle")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this oracle")

    shape0, shape1, shape2 = _validate_inputs(inputs)
    device = torch.device("cuda", torch.cuda.current_device())
    seeds = _make_inductor_seed_tensor(device)

    out0 = _make_output(shape0, device)
    out1 = _make_output(shape1, device)
    out2 = _make_output(shape2, device)

    grid = (triton.cdiv(NUMEL, BLOCK_N),)
    _randn3_fp16_kernel[grid](
        seeds,
        out0,
        out1,
        out2,
        N=NUMEL,
        BLOCK=BLOCK_N,
        num_warps=4,
    )
    return (out0, out1, out2)


def _as_tuple(value) -> tuple:
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    return (value,)


def _max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).abs().max().item())


def _stats_ok(tensor: torch.Tensor) -> tuple[bool, float, float]:
    values = tensor.float()
    mean = float(values.mean().item())
    std = float(values.std(unbiased=False).item())
    ok = torch.isfinite(values).all().item() and abs(mean) < 0.10 and 0.60 < std < 1.40
    return bool(ok), mean, std


@torch.no_grad()
def run_check(rtol: float, atol: float) -> bool:
    """Check metadata vs eager and RNG values vs compiled Inductor.

    Eager `prims.inductor_random` does not provide a stable value oracle for the
    seed tensor that compiled Inductor lowers to `tl.randn(seed, offset)`.
    Therefore this check uses eager for scope metadata and uses a compiled repro
    run under the same restored CUDA RNG state for bit-exact stochastic values.
    """
    print(f"Checking {REPRO_ID}...")
    inputs = get_inputs()
    instance = get_repro_instance()

    eager = _as_tuple(instance(*inputs))
    actual = _as_tuple(oracle_forward(inputs))
    torch.cuda.synchronize()

    if len(actual) != len(eager):
        print(f"  SCOPE_MISMATCH: eager outputs={len(eager)} oracle outputs={len(actual)}")
        print("Correctness: FAIL")
        return False

    ok = True
    for idx, (got, ref) in enumerate(zip(actual, eager)):
        metadata_ok = (
            got.shape == ref.shape
            and got.dtype == ref.dtype
            and got.stride() == ref.stride()
            and got.storage_offset() == ref.storage_offset()
        )
        stats_ok, mean, std = _stats_ok(got)
        item_ok = metadata_ok and stats_ok
        ok = ok and item_ok
        status = "PASS" if item_ok else "FAIL"
        print(
            f"  output {idx}: {status} metadata shape={list(got.shape)} "
            f"dtype={got.dtype} stride={tuple(got.stride())} "
            f"stochastic_stats mean={mean:.4f} std={std:.4f}"
        )

    pairwise_distinct = not torch.equal(actual[0], actual[1]) and not torch.equal(actual[0], actual[2])
    pairwise_distinct = pairwise_distinct and not torch.equal(actual[1], actual[2])
    print(f"  stochastic streams distinct: {'PASS' if pairwise_distinct else 'FAIL'}")
    ok = ok and pairwise_distinct

    compiled = torch.compile(instance)
    compiled(*inputs)
    oracle_forward(inputs)
    torch.cuda.synchronize()

    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state()

    torch.set_rng_state(cpu_state)
    torch.cuda.set_rng_state(cuda_state)
    compiled_ref = _as_tuple(compiled(*inputs))
    torch.cuda.synchronize()

    torch.set_rng_state(cpu_state)
    torch.cuda.set_rng_state(cuda_state)
    compiled_equiv = _as_tuple(oracle_forward(inputs))
    torch.cuda.synchronize()

    for idx, (got, ref) in enumerate(zip(compiled_equiv, compiled_ref)):
        same_layout = got.shape == ref.shape and got.dtype == ref.dtype and got.stride() == ref.stride()
        same_values = torch.allclose(got.float(), ref.float(), rtol=rtol, atol=atol)
        exact_values = torch.equal(got, ref)
        item_ok = same_layout and same_values
        ok = ok and item_ok
        status = "PASS" if item_ok else "FAIL"
        print(
            f"  compiled-seed output {idx}: {status} "
            f"exact={exact_values} max_abs={_max_abs(got, ref):.2e}"
        )

    print(f"Correctness: {'PASS' if ok else 'FAIL'}")
    return ok


def _has_inductor_random() -> bool:
    return "inductor_random" in REPRO_PATH.read_text()


def main():
    parser = argparse.ArgumentParser(
        description=f"Oracle for {REPRO_ID}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--check", action="store_true", help="Verify correctness against Repro")
    parser.add_argument("--bench", action="store_true", help="Benchmark oracle vs torch.compile")
    parser.add_argument("--rtol", type=float, default=1e-2, help="Relative tolerance")
    parser.add_argument("--atol", type=float, default=1e-2, help="Absolute tolerance")
    parser.add_argument("--warmup", type=int, default=25, help="Warmup iterations")
    parser.add_argument("--rep", type=int, default=200, help="Benchmark repetitions")
    parser.add_argument(
        "--no-skip-stochastic",
        action="store_true",
        help="Accepted for template compatibility; stochastic values are checked explicitly",
    )
    parser.add_argument("--all-shapes", action="store_true", help="Benchmark across shapes.txt")
    parser.add_argument("--show-hw", action="store_true", help="Print GPU hardware info and exit")
    args = parser.parse_args()

    if args.show_hw:
        import json

        print(json.dumps(get_hardware_info(), indent=2))
        return

    if not args.check and not args.bench:
        args.check = args.bench = True

    if _has_inductor_random():
        print(
            f"NOTE: {REPRO_ID} contains prims.inductor_random; "
            "metadata is checked against eager and seeded values against compiled Inductor"
        )

    inputs = get_inputs()
    instance = get_repro_instance()

    if args.check:
        if not run_check(args.rtol, args.atol):
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
