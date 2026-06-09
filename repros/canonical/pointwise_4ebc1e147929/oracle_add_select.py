"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete reshape/add/select region in one contiguous Triton add kernel that writes the selected [128, 768] result directly, whereas Inductor already lowers this metadata reshape plus singleton-dimension select around the add to equivalent single-kernel pointwise work; Inductor cannot materially improve it today because the remaining cost is one launch plus reading two f32 tensors and writing one f32 tensor, not an unfused scheduler pattern; the fix is BANDWIDTH_BOUND: classify this repro as at the memory/launch floor rather than adding a new Inductor transformation."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - keeps py_compile usable without Triton.
    triton = None
    tl = None

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


REPRO_ID = "pointwise_4ebc1e147929"
REPRO_DIR = Path(__file__).resolve().parent
REPRO_PATH = REPRO_DIR / "repro.py"

M = 128
N = 768
NUMEL = M * N
BLOCK_SIZE = 1024


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _add_select_kernel(
        addmm_ptr,
        view_ptr,
        out_ptr,
        total: tl.constexpr,
        block_size: tl.constexpr,
    ):
        offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
        mask = offsets < total

        # Offsets are identical for addmm[i, j], view[i, 0, j], and out[i, j]
        # because the captured view has a singleton middle dimension.
        addmm = tl.load(addmm_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        view = tl.load(view_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        tl.store(out_ptr + offsets, addmm + view, mask=mask)


def _validate_inputs(inputs: list[object]) -> tuple[torch.Tensor, torch.Tensor]:
    if triton is None:
        raise RuntimeError("Triton is required for this oracle")
    if len(inputs) != 3:
        raise ValueError(f"expected 3 inputs, got {len(inputs)}")

    addmm_51, view_129, shape_param = inputs
    if not isinstance(addmm_51, torch.Tensor) or not isinstance(view_129, torch.Tensor):
        raise TypeError("expected tensor inputs addmm_51 and view_129")
    if addmm_51.device.type != "cuda" or view_129.device.type != "cuda":
        raise RuntimeError("Triton oracle requires CUDA tensor inputs")
    if addmm_51.dtype != torch.float32 or view_129.dtype != torch.float32:
        raise TypeError(f"expected f32 inputs, got {addmm_51.dtype} and {view_129.dtype}")
    if tuple(addmm_51.shape) != (M, N):
        raise ValueError(f"unexpected addmm_51 shape: {tuple(addmm_51.shape)}")
    if tuple(view_129.shape) != (M, 1, N):
        raise ValueError(f"unexpected view_129 shape: {tuple(view_129.shape)}")
    if tuple(int(dim) for dim in shape_param) != (M, 1, N):
        raise ValueError(f"unexpected reshape parameter: {shape_param}")
    if tuple(addmm_51.stride()) != (N, 1):
        raise ValueError(f"unexpected addmm_51 stride: {tuple(addmm_51.stride())}")
    if tuple(view_129.stride()) != (N, N, 1):
        raise ValueError(f"unexpected view_129 stride: {tuple(view_129.stride())}")

    return addmm_51, view_129


@oracle_impl(hardware="H100", shapes="(T([128, 768], f32), T([128, 1, 768], f32), S([128, 1, 768]))")
def oracle_forward(inputs):
    """Compute the exact Repro.forward output for reshape(addmm)+view select(dim=1,index=0)."""
    addmm_51, view_129 = _validate_inputs(inputs)
    out = torch.empty_strided((M, N), (N, 1), device=addmm_51.device, dtype=torch.float32)
    grid = (triton.cdiv(NUMEL, BLOCK_SIZE),)
    _add_select_kernel[grid](
        addmm_51,
        view_129,
        out,
        total=NUMEL,
        block_size=BLOCK_SIZE,
        num_warps=4,
    )
    return out


def main():
    parser = argparse.ArgumentParser(
        description=f"Oracle for {REPRO_ID}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--check", action="store_true", help="Verify correctness against eager Repro")
    parser.add_argument("--bench", action="store_true", help="Benchmark oracle vs torch.compile")
    parser.add_argument("--rtol", type=float, default=1e-2, help="Relative tolerance for correctness check")
    parser.add_argument("--atol", type=float, default=1e-2, help="Absolute tolerance for correctness check")
    parser.add_argument("--warmup", type=int, default=25, help="Warmup iterations for benchmark")
    parser.add_argument("--rep", type=int, default=200, help="Repetitions for benchmark")
    parser.add_argument(
        "--no-skip-stochastic",
        action="store_true",
        help="Disable auto-detection and skipping of stochastic outputs",
    )
    parser.add_argument("--all-shapes", action="store_true", help="Benchmark across all shapes from shapes.txt")
    parser.add_argument("--show-hw", action="store_true", help="Print GPU hardware info and exit")
    args = parser.parse_args()

    if args.show_hw:
        import json

        print(json.dumps(get_hardware_info(), indent=2))
        return

    if not args.check and not args.bench:
        args.check = args.bench = True

    inputs = get_inputs()
    instance = get_repro_instance()

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
        print(f"Correctness: {'PASS' if ok else 'FAIL'}")
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
