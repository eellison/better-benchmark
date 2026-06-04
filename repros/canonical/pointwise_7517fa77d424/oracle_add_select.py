"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete view(addmm_51)->add(view_104)->select(dim=1,index=0) chain as one Triton pointwise kernel that reads both f32 inputs once and writes the selected f32[128,768] output, whereas Inductor already lowers this repro to the same essential fused pointwise memory traffic; Inductor cannot materially beat this today because there is no remaining algebraic, fusion, or scheduling boundary to remove after the size-1 view/select collapse, and the required change is BANDWIDTH_BOUND: treat this pattern as at floor unless a broader launch-overhead or memory-system improvement applies."""
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
    bench_oracle,
    bench_oracle_all_shapes,
    check_oracle,
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    has_stochastic_ops,
)


REPRO_ID = "pointwise_7517fa77d424"
REPRO_DIR = Path(__file__).resolve().parent
REPRO_PATH = REPRO_DIR / "repro.py"


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
        view_104_ptr,
        out_ptr,
        total: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < total
        addmm = tl.load(addmm_ptr + offsets, mask=mask)
        view_104 = tl.load(view_104_ptr + offsets, mask=mask)
        tl.store(out_ptr + offsets, addmm + view_104, mask=mask)


def _validate_inputs(inputs: tuple[object, ...] | list[object]) -> tuple[torch.Tensor, torch.Tensor]:
    if triton is None:
        raise RuntimeError("Triton is required for this oracle")
    if len(inputs) != 3:
        raise ValueError(f"expected 3 inputs, got {len(inputs)}")

    addmm_51, view_104, shape_param = inputs
    if not isinstance(addmm_51, torch.Tensor) or not isinstance(view_104, torch.Tensor):
        raise TypeError("expected tensor inputs for addmm_51 and view_104")
    if addmm_51.device.type != "cuda" or view_104.device.type != "cuda":
        raise RuntimeError("Triton oracle requires CUDA tensor inputs")
    if addmm_51.dtype != torch.float32 or view_104.dtype != torch.float32:
        raise ValueError("expected f32 addmm_51 and view_104 inputs")
    if tuple(addmm_51.shape) != (128, 768):
        raise ValueError(f"unexpected addmm_51 shape: {tuple(addmm_51.shape)}")
    if tuple(view_104.shape) != (128, 1, 768):
        raise ValueError(f"unexpected view_104 shape: {tuple(view_104.shape)}")
    if tuple(shape_param) != (128, 1, 768):
        raise ValueError(f"unexpected view shape parameter: {shape_param}")
    if not addmm_51.is_contiguous() or not view_104.is_contiguous():
        raise ValueError("oracle expects the contiguous captured repro inputs")
    return addmm_51, view_104


def oracle_forward(inputs):
    """Compute the exact Repro.forward output for the captured view/add/select scope."""
    addmm_51, view_104 = _validate_inputs(inputs)
    out = torch.empty_strided((128, 768), (768, 1), device=addmm_51.device, dtype=torch.float32)
    total = out.numel()
    block = 1024
    grid = (triton.cdiv(total, block),)
    _add_select_kernel[grid](addmm_51, view_104, out, total, BLOCK=block, num_warps=4)
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
