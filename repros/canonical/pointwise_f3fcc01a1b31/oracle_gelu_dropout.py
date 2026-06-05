"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete Bart training GELU/dropout pointwise scope in one shape-specialized flat Triton pass, including the `[2048,3072] -> [4,512,3072]` metadata view, exact-erf GELU, seed-index-2 Inductor RNG dropout, dropout scaling, and final contiguous `[2048,3072]` view, whereas Inductor already lowers the same full scope to an equivalent fused stochastic pointwise kernel; Inductor cannot materially improve this local repro today because the remaining cost is dominated by mandatory activation reads, erf/RNG/dropout math, and output stores rather than an avoidable scheduler split, scatter/reduce, split-K, or algebraic rewrite; the fix is BANDWIDTH_BOUND: record this as at floor for performance, with not_true_floor status for this oracle attempt because exact eager stochastic equality is skipped."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
import torch._inductor.inductor_prims  # noqa: F401 - registers prims RNG ops

try:
    import triton
    import triton.language as tl
except ImportError:
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
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    check_oracle,
    bench_oracle,
    bench_oracle_all_shapes,
    get_hardware_info,
    get_shape_key,
    has_stochastic_ops,
)


ROWS = 2048
COLS = 3072
BATCH = 4
SEQ_LEN = 512
INPUT_SHAPE = (ROWS, COLS)
VIEW_SHAPE = (BATCH, SEQ_LEN, COLS)
OUTPUT_SHAPE = INPUT_SHAPE
OUTPUT_STRIDE = (COLS, 1)
SEED_INDEX = 2
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.1111111111111112
GELU_SCALE = 0.7071067811865476
EXACT_STOCHASTIC_EQUALITY = False


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---
if triton is not None:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_SIZE": 256}, num_warps=4, num_stages=4),
            triton.Config({"BLOCK_SIZE": 512}, num_warps=4, num_stages=4),
            triton.Config({"BLOCK_SIZE": 1024}, num_warps=4, num_stages=4),
            triton.Config({"BLOCK_SIZE": 2048}, num_warps=8, num_stages=4),
            triton.Config({"BLOCK_SIZE": 4096}, num_warps=8, num_stages=4),
        ],
        key=["N"],
    )
    @triton.jit
    def _gelu_dropout_kernel(
        addmm_ptr,
        seeds_ptr,
        output_ptr,
        N: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N
        x = tl.load(addmm_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

        gelu = (x * 0.5) * (tl.erf(x * 0.7071067811865476) + 1.0)
        seed = tl.load(seeds_ptr + 2)
        random = tl.rand(seed, offsets.to(tl.uint32))
        keep = random > 0.1
        output = tl.where(keep, gelu * 1.1111111111111112, 0.0)
        tl.store(output_ptr + offsets, output, mask=mask)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    if isinstance(value, torch.Size):
        return tuple(int(dim) for dim in value)
    if isinstance(value, (list, tuple)):
        return tuple(int(dim) for dim in value)
    raise TypeError(f"expected shape parameter list/tuple, got {type(value).__name__}")


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, ...], tuple[int, ...]]:
    if len(inputs) != 4:
        raise ValueError(f"{REPRO_ID} expects 4 inputs, got {len(inputs)}")

    addmm_4, inductor_seeds, shape0, shape1 = inputs
    if not isinstance(addmm_4, torch.Tensor):
        raise TypeError("addmm_4 must be a tensor")
    if not isinstance(inductor_seeds, torch.Tensor):
        raise TypeError("inductor_seeds must be a tensor")
    if tuple(addmm_4.shape) != INPUT_SHAPE:
        raise ValueError(f"addmm_4 has shape {tuple(addmm_4.shape)}, expected {INPUT_SHAPE}")
    if addmm_4.dtype != torch.float32:
        raise TypeError(f"addmm_4 has dtype {addmm_4.dtype}, expected torch.float32")
    if inductor_seeds.dtype != torch.int64:
        raise TypeError(f"inductor_seeds has dtype {inductor_seeds.dtype}, expected torch.int64")
    if inductor_seeds.ndim != 1 or inductor_seeds.shape[0] <= SEED_INDEX:
        raise ValueError(f"inductor_seeds must contain seed index {SEED_INDEX}")
    if not addmm_4.is_contiguous():
        raise ValueError(f"addmm_4 must be contiguous, got stride={tuple(addmm_4.stride())}")
    if inductor_seeds.device != addmm_4.device:
        raise ValueError("addmm_4 and inductor_seeds must be on the same device")

    view_shape = _shape_tuple(shape0)
    output_shape = _shape_tuple(shape1)
    if view_shape != VIEW_SHAPE:
        raise ValueError(f"unexpected first view shape {view_shape}, expected {VIEW_SHAPE}")
    if output_shape != OUTPUT_SHAPE:
        raise ValueError(f"unexpected final view shape {output_shape}, expected {OUTPUT_SHAPE}")
    return addmm_4, inductor_seeds, view_shape, output_shape


def _torch_full_scope(inputs: list[Any] | tuple[Any, ...]) -> torch.Tensor:
    addmm_4, inductor_seeds, view_shape, output_shape = _validate_inputs(inputs)
    view_default = torch.ops.aten.view.default(addmm_4, view_shape)
    mul_tensor = torch.ops.aten.mul.Tensor(view_default, 0.5)
    mul_tensor_1 = torch.ops.aten.mul.Tensor(view_default, GELU_SCALE)
    erf_default = torch.ops.aten.erf.default(mul_tensor_1)
    add_tensor = torch.ops.aten.add.Tensor(erf_default, 1)
    mul_tensor_2 = torch.ops.aten.mul.Tensor(mul_tensor, add_tensor)
    seed = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, SEED_INDEX)
    random = torch.ops.prims.inductor_random.default(VIEW_SHAPE, seed, "rand")
    keep = torch.ops.aten.gt.Scalar(random, DROPOUT_P)
    dropped = torch.ops.aten.mul.Tensor(keep, mul_tensor_2)
    scaled = torch.ops.aten.mul.Tensor(dropped, DROPOUT_SCALE)
    return torch.ops.aten.view.default(scaled, output_shape)


def oracle_forward(inputs):
    """Run the full Repro.forward scope: GELU, seeded dropout, and final view.

    SCOPE INVARIANT: Must accept the same inputs as Repro.forward() and return
    the same outputs (same count, same shapes, same dtypes, same strides).

    Args:
        inputs: tuple of tensors/values from get_inputs(), identical to what
                Repro.forward() receives via Repro()(*inputs).

    Returns:
        Same output structure as Repro()(*inputs).
    """
    addmm_4, inductor_seeds, _view_shape, _output_shape = _validate_inputs(inputs)
    if triton is None or addmm_4.device.type != "cuda":
        return _torch_full_scope(inputs)

    output = torch.empty_strided(
        OUTPUT_SHAPE,
        OUTPUT_STRIDE,
        device=addmm_4.device,
        dtype=torch.float32,
    )
    n_elements = addmm_4.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    _gelu_dropout_kernel[grid](
        addmm_4,
        inductor_seeds,
        output,
        N=n_elements,
    )
    return output


def _has_inductor_random() -> bool:
    return "inductor_random" in REPRO_PATH.read_text()


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
    if has_stochastic_ops(REPRO_PATH) or _has_inductor_random():
        print(f"NOTE: {REPRO_ID} contains stochastic ops; affected outputs will be auto-skipped")
    if not EXACT_STOCHASTIC_EQUALITY:
        print("NOTE: exact stochastic equality is not established; floor status not_true_floor")

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
