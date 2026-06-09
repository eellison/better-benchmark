"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the full Whisper reshape-plus-add scope as one contiguous Triton vector add from the flattened `[12000, 384]` view and the `[8, 1500, 384]` residual into a fresh contiguous output, whereas Inductor already emits a single fused pointwise kernel for the same view/add work but carries generic indexing and launch overhead around mandatory read/write traffic; Inductor cannot remove the dominant work today because the repro contract requires reading both inputs and materializing the full `float32` output tensor with the same layout; the fix is BANDWIDTH_BOUND: no scheduler or algebraic rewrite can substantially beat the memory-traffic floor beyond minor pointwise index specialization."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

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
    oracle_impl,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    check_oracle,
    bench_oracle,
    bench_oracle_all_shapes,
    get_hardware_info,
    get_shape_key,
    has_stochastic_ops,
)


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
            triton.Config({"BLOCK_N": 256}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_N": 1024}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_N": 2048}, num_warps=8, num_stages=3),
            triton.Config({"BLOCK_N": 4096}, num_warps=8, num_stages=3),
        ],
        key=["N"],
    )
    @triton.jit
    def _flat_view_add_kernel(
        addmm_ptr,
        add_ptr,
        out_ptr,
        N: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = offsets < N

        lhs = tl.load(addmm_ptr + offsets, mask=mask, other=0.0)
        rhs = tl.load(add_ptr + offsets, mask=mask, other=0.0)
        tl.store(out_ptr + offsets, lhs + rhs, mask=mask)


def _contiguous_stride(shape: tuple[int, ...]) -> tuple[int, ...]:
    stride = []
    running = 1
    for size in reversed(shape):
        stride.append(running)
        running *= int(size)
    return tuple(reversed(stride))


def _as_shape_tuple(shape_param) -> tuple[int, ...]:
    shape = tuple(int(dim) for dim in shape_param)
    if any(dim < 0 for dim in shape):
        raise ValueError(f"expected concrete positive shape parameter, got {shape_param}")
    return shape


def _same_storage(lhs: torch.Tensor, rhs: torch.Tensor) -> bool:
    return lhs.untyped_storage().data_ptr() == rhs.untyped_storage().data_ptr()


def _check_output_contract(eager: torch.Tensor, oracle: torch.Tensor, inputs) -> bool:
    layout_ok = (
        tuple(oracle.shape) == tuple(eager.shape)
        and oracle.dtype == eager.dtype
        and tuple(oracle.stride()) == tuple(eager.stride())
        and oracle.storage_offset() == eager.storage_offset()
    )
    alias_ok = True
    for inp in inputs:
        if isinstance(inp, torch.Tensor):
            alias_ok = alias_ok and (_same_storage(oracle, inp) == _same_storage(eager, inp))
    return layout_ok and alias_ok


@oracle_impl(hardware="H100", shapes="(T([12000, 384], f32), T([8, 1500, 384], f32), S([8, 1500, 384]))")
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

    addmm_4, add_2, shape_param = inputs
    output_shape = _as_shape_tuple(shape_param)
    if tuple(add_2.shape) != output_shape:
        raise ValueError(f"shape parameter {output_shape} does not match add input {tuple(add_2.shape)}")
    if addmm_4.numel() != add_2.numel():
        raise ValueError(f"view input numel {addmm_4.numel()} does not match add input numel {add_2.numel()}")
    if addmm_4.dtype != add_2.dtype:
        raise ValueError(f"expected matching input dtypes, got {addmm_4.dtype} and {add_2.dtype}")
    if not addmm_4.is_contiguous() or not add_2.is_contiguous():
        raise ValueError(f"expected contiguous inputs, got strides {tuple(addmm_4.stride())} and {tuple(add_2.stride())}")

    output = torch.empty_strided(
        output_shape,
        _contiguous_stride(output_shape),
        device=add_2.device,
        dtype=add_2.dtype,
    )
    numel = add_2.numel()
    grid = lambda META: (triton.cdiv(numel, META["BLOCK_N"]),)
    _flat_view_add_kernel[grid](
        addmm_4,
        add_2,
        output,
        N=numel,
    )
    return output


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
            eager = instance(*inputs)
            oracle_out = oracle_forward(inputs)
            if oracle_out.is_cuda:
                torch.cuda.synchronize()
        contract_ok = _check_output_contract(eager, oracle_out, inputs)
        print(
            f"  output 0 layout/alias: {'PASS' if contract_ok else 'FAIL'} "
            f"(shape={list(oracle_out.shape)} stride={list(oracle_out.stride())} "
            f"storage_offset={oracle_out.storage_offset()})"
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
