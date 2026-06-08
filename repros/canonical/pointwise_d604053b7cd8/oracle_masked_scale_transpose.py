"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete T5 bool-mask scale/where pointwise scope by writing the fresh contiguous `[8, 1024, 2048]` backing tensor in one storage-linear Triton pass and returning the required `[2048, 8192]` transpose view, whereas Inductor already lowers this isolated pointwise graph to equivalent fused materialization with only generic index/codegen overhead; Inductor cannot materially improve this local repro through fusion, scatter-reduce, split-K, algebraic elimination, or recomputation because the output contract requires the large f32 backing allocation plus dense mask/input reads and f32 stores; the fix is BANDWIDTH_BOUND: record this as a pointwise memory floor unless broader pointwise memory-codegen or launch/allocation work moves both implementations."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - keeps py_compile useful without Triton.
    triton = None
    tl = None

# --- Configuration (auto-derived from file location) ---
REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPRO_PATH = REPRO_DIR / "repro.py"

# Import shared oracle infrastructure. Run first:
#   python -m pip install --no-build-isolation -e .
# Use the installed oracle_harness package; run editable install before checks.
# Do not add custom benchmark functions. bench_oracle() owns timing so CUDAGraph,
# GPU locking, and interleaved oracle/compile measurement are preserved.
from oracle_harness import (
    bench_oracle,
    bench_oracle_all_shapes,
    check_oracle,
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    has_stochastic_ops,
)


SCALE = 1.1111111111111112
BLOCK_SIZE = 1024


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


def _shape_tuple(value: object, name: str) -> tuple[int, ...]:
    try:
        return tuple(int(dim) for dim in value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError(f"{name} must be an iterable shape, got {type(value)!r}") from exc


def _validate_inputs(
    inputs: tuple[object, ...] | list[object],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, int, int], tuple[int, int]]:
    if len(inputs) != 6:
        raise ValueError(f"{REPRO_ID} expects 6 inputs, got {len(inputs)}")

    mm_183, arg147_1, arg429_1, full_1, shape0, shape1 = inputs
    if not isinstance(mm_183, torch.Tensor):
        raise TypeError(f"mm_183 must be a tensor, got {type(mm_183)!r}")
    if not isinstance(arg147_1, torch.Tensor):
        raise TypeError(f"arg147_1 must be a tensor, got {type(arg147_1)!r}")
    if not isinstance(arg429_1, torch.Tensor):
        raise TypeError(f"arg429_1 must be a tensor, got {type(arg429_1)!r}")
    if not isinstance(full_1, torch.Tensor):
        raise TypeError(f"full_1 must be a tensor, got {type(full_1)!r}")

    view_shape = _shape_tuple(shape0, "_shape_param_0")
    flat_shape = _shape_tuple(shape1, "_shape_param_1")
    if len(view_shape) != 3 or len(flat_shape) != 2:
        raise ValueError(f"unexpected shape params: {view_shape}, {flat_shape}")
    if flat_shape != tuple(mm_183.shape):
        raise ValueError(f"mm_183 shape {tuple(mm_183.shape)} does not match {flat_shape}")
    if view_shape != tuple(arg147_1.shape) or view_shape != tuple(arg429_1.shape):
        raise ValueError(
            "mask shapes must match _shape_param_0, got "
            f"{tuple(arg147_1.shape)} and {tuple(arg429_1.shape)} vs {view_shape}"
        )
    if flat_shape[0] != view_shape[0] * view_shape[1] or flat_shape[1] != view_shape[2]:
        raise ValueError(f"shape params are not compatible: {view_shape}, {flat_shape}")

    if mm_183.dtype is not torch.float32:
        raise ValueError(f"mm_183 must be float32, got {mm_183.dtype}")
    if arg147_1.dtype is not torch.bool or arg429_1.dtype is not torch.bool:
        raise ValueError(f"masks must be bool, got {arg147_1.dtype} and {arg429_1.dtype}")
    if full_1.dtype is not torch.float32 or full_1.ndim != 0:
        raise ValueError(f"full_1 must be a float32 scalar tensor, got {full_1.dtype} {tuple(full_1.shape)}")

    for name, tensor in (("mm_183", mm_183), ("arg147_1", arg147_1), ("arg429_1", arg429_1)):
        if not tensor.is_cuda:
            raise ValueError(f"{name} must be a CUDA tensor")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous, got stride={tuple(tensor.stride())}")
    if not full_1.is_cuda:
        raise ValueError("full_1 must be a CUDA tensor")

    return mm_183, arg147_1, arg429_1, full_1, view_shape, flat_shape


if triton is not None:

    @triton.jit
    def _masked_scale_where_kernel(
        mm_ptr,
        scale_mask_ptr,
        full_mask_ptr,
        full_ptr,
        out_ptr,
        total: tl.constexpr,
        scale: tl.constexpr,
        block_size: tl.constexpr,
    ):
        offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
        mask = offsets < total

        values = tl.load(mm_ptr + offsets, mask=mask, other=0.0)
        scale_mask = tl.load(scale_mask_ptr + offsets, mask=mask, other=0).to(tl.float32)
        full_mask = tl.load(full_mask_ptr + offsets, mask=mask, other=0)
        full_value = tl.load(full_ptr)

        scaled = values * (scale_mask * scale)
        out = tl.where(full_mask, full_value, scaled)
        tl.store(out_ptr + offsets, out, mask=mask)


def oracle_forward(inputs):
    """Run the oracle computation.

    SCOPE INVARIANT: Must accept the same inputs as Repro.forward() and return
    the same outputs (same count, same shapes, same dtypes, same strides).

    Args:
        inputs: tuple of tensors/values from get_inputs(), identical to what
                Repro.forward() receives via Repro()(*inputs).

    Returns:
        Same output structure as Repro()(*inputs).
    """
    if triton is None:
        raise RuntimeError("Triton is required for this oracle")

    mm_183, arg147_1, arg429_1, full_1, view_shape, flat_shape = _validate_inputs(inputs)
    base = torch.empty_strided(
        view_shape,
        (view_shape[1] * view_shape[2], view_shape[2], 1),
        device=mm_183.device,
        dtype=mm_183.dtype,
    )

    total = mm_183.numel()
    grid = (triton.cdiv(total, BLOCK_SIZE),)
    _masked_scale_where_kernel[grid](
        mm_183,
        arg147_1,
        arg429_1,
        full_1,
        base,
        total=total,
        scale=SCALE,
        block_size=BLOCK_SIZE,
        num_warps=4,
        num_stages=1,
    )
    return base.view(flat_shape).permute(1, 0)


def _check_output_contract(
    actual: torch.Tensor,
    expected: torch.Tensor,
    inputs: tuple[object, ...] | list[object],
) -> bool:
    base_actual = actual._base
    base_expected = expected._base
    base_ok = (
        (base_actual is None and base_expected is None)
        or (
            base_actual is not None
            and base_expected is not None
            and tuple(base_actual.shape) == tuple(base_expected.shape)
            and tuple(base_actual.stride()) == tuple(base_expected.stride())
            and base_actual.storage_offset() == base_expected.storage_offset()
        )
    )
    input_alias_ok = all(
        actual.data_ptr() != value.data_ptr()
        for value in inputs
        if isinstance(value, torch.Tensor)
    )
    return (
        isinstance(actual, torch.Tensor)
        and tuple(actual.shape) == tuple(expected.shape)
        and actual.dtype == expected.dtype
        and tuple(actual.stride()) == tuple(expected.stride())
        and actual.storage_offset() == expected.storage_offset()
        and (actual._base is not None) == (expected._base is not None)
        and base_ok
        and input_alias_ok
    )


# --- CLI entry point ---
def main():
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

    # Handle --show-hw early
    if args.show_hw:
        import json
        print(json.dumps(get_hardware_info(), indent=2))
        return

    # Default: run both --check and --bench
    if not args.check and not args.bench:
        args.check = args.bench = True

    inputs = get_inputs()
    instance = get_repro_instance()

    # Report if stochastic ops detected in source
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
        with torch.no_grad():
            expected = instance(*inputs)
            actual = oracle_forward(inputs)
            if actual.is_cuda:
                torch.cuda.synchronize()
        contract_ok = _check_output_contract(actual, expected, inputs)
        print(
            f"  output 0 contract: {'PASS' if contract_ok else 'FAIL'} "
            f"(shape={list(actual.shape)} stride={list(actual.stride())} "
            f"storage_offset={actual.storage_offset()} view={actual._base is not None})"
        )
        ok = ok and contract_ok
        print(f"Correctness: {'PASS' if ok else 'FAIL'}")
        if not ok:
            sys.exit(1)

    if args.bench:
        print(f"Benchmarking {REPRO_ID}...")
        if args.all_shapes:
            results = bench_oracle_all_shapes(
                oracle_forward, REPRO_DIR, REPRO_ID,
                warmup=args.warmup, rep=args.rep,
            )
            for result in results:
                if result["status"] == "BAD_ORACLE":
                    print(f"WARNING: oracle is slower than compile for "
                          f"{result['repro_id']} (ratio={result['ratio']:.3f}x)")
        else:
            # The shared harness owns timing so graph capture, GPU locking, and
            # interleaved oracle/compile measurement stay intact.
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
