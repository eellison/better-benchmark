"""Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle allocates the 24 returned contiguous bool tensors and fills them directly with constant True in one Triton kernel, whereas Inductor currently lowers the no-input graph through float full, view/broadcast metadata, a broadcast multiply, and repeated bool conversions; Inductor cannot do this today because its algebraic simplifier does not prove that full(1) multiplied by its broadcasted transpose is nonzero everywhere and fold all downstream bool casts into constant True output stores; the fix is ALGEBRAIC_ELIMINATION: canonicalize nonzero constant full/broadcast arithmetic feeding bool conversions into direct constant bool materialization with duplicated-output CSE handled before scheduling."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - keeps py_compile useful.
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


REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPRO_PATH = REPRO_DIR / "repro.py"

OUT_SHAPE = (8, 1, 512, 512)
OUT_STRIDE = (262144, 262144, 512, 1)
OUT_NUMEL = 8 * 1 * 512 * 512
OUT_COUNT = 24
BLOCK_SIZE = 1024
CUDA_DEVICE = torch.device("cuda", 0)


def get_inputs() -> list[object]:
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance() -> torch.nn.Module:
    """Create Repro() for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR).eval()


if triton is not None:

    @triton.jit
    def _fill_true_24_kernel(
        out0,
        out1,
        out2,
        out3,
        out4,
        out5,
        out6,
        out7,
        out8,
        out9,
        out10,
        out11,
        out12,
        out13,
        out14,
        out15,
        out16,
        out17,
        out18,
        out19,
        out20,
        out21,
        out22,
        out23,
        n_elements: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n_elements
        values = offsets == offsets
        tl.store(out0 + offsets, values, mask=mask)
        tl.store(out1 + offsets, values, mask=mask)
        tl.store(out2 + offsets, values, mask=mask)
        tl.store(out3 + offsets, values, mask=mask)
        tl.store(out4 + offsets, values, mask=mask)
        tl.store(out5 + offsets, values, mask=mask)
        tl.store(out6 + offsets, values, mask=mask)
        tl.store(out7 + offsets, values, mask=mask)
        tl.store(out8 + offsets, values, mask=mask)
        tl.store(out9 + offsets, values, mask=mask)
        tl.store(out10 + offsets, values, mask=mask)
        tl.store(out11 + offsets, values, mask=mask)
        tl.store(out12 + offsets, values, mask=mask)
        tl.store(out13 + offsets, values, mask=mask)
        tl.store(out14 + offsets, values, mask=mask)
        tl.store(out15 + offsets, values, mask=mask)
        tl.store(out16 + offsets, values, mask=mask)
        tl.store(out17 + offsets, values, mask=mask)
        tl.store(out18 + offsets, values, mask=mask)
        tl.store(out19 + offsets, values, mask=mask)
        tl.store(out20 + offsets, values, mask=mask)
        tl.store(out21 + offsets, values, mask=mask)
        tl.store(out22 + offsets, values, mask=mask)
        tl.store(out23 + offsets, values, mask=mask)


def oracle_forward(inputs: list[object] | tuple[object, ...]) -> tuple[torch.Tensor, ...]:
    """Run the full Repro.forward scope with direct constant bool output stores."""
    if inputs:
        raise ValueError(f"{REPRO_ID} expects no inputs, got {len(inputs)}")
    if triton is None:
        raise RuntimeError("Triton is required for the timed oracle")
    if not torch.cuda.is_available():
        raise RuntimeError(f"{REPRO_ID} produces CUDA outputs, but CUDA is unavailable")

    outputs = tuple(
        torch.empty_strided(
            OUT_SHAPE,
            OUT_STRIDE,
            device=CUDA_DEVICE,
            dtype=torch.bool,
        )
        for _ in range(OUT_COUNT)
    )
    grid = (triton.cdiv(OUT_NUMEL, BLOCK_SIZE),)
    _fill_true_24_kernel[grid](
        *outputs,
        n_elements=OUT_NUMEL,
        BLOCK=BLOCK_SIZE,
        num_warps=4,
        num_stages=3,
    )
    return outputs


def _check_layout_and_alias(outputs: tuple[torch.Tensor, ...]) -> bool:
    if len(outputs) != OUT_COUNT:
        return False
    data_ptrs = set()
    for output in outputs:
        if (
            tuple(output.shape) != OUT_SHAPE
            or tuple(output.stride()) != OUT_STRIDE
            or output.dtype is not torch.bool
            or not output.is_cuda
            or output.storage_offset() != 0
        ):
            return False
        data_ptrs.add(output.data_ptr())
    return len(data_ptrs) == OUT_COUNT


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
        with torch.no_grad():
            layout_outputs = oracle_forward(inputs)
            torch.cuda.synchronize()
        layout_ok = _check_layout_and_alias(layout_outputs)
        print(
            f"  outputs layout/alias: {'PASS' if layout_ok else 'FAIL'} "
            f"(count={len(layout_outputs)} shape={list(layout_outputs[0].shape)} "
            f"stride={layout_outputs[0].stride()} dtype={layout_outputs[0].dtype})"
        )
        ok = ok and layout_ok
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
