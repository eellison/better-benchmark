"""
Oracle for pointwise_e1a420e661f1.

Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the full
permute-clone-view scope as one direct Triton materialization from
`arg81_1[0, c, h, w]` to the final contiguous `float32[64, 768]` output,
whereas Inductor already lowers the isolated graph to a generic one-launch
layout materialization for the fresh clone output; Inductor cannot materially
improve this kernel today because the user-visible clone requires reading and
writing all 49,152 float32 elements and this capture has no surrounding
consumer to absorb the materialization; the fix is BANDWIDTH_BOUND: no local
scheduler/codegen change is indicated beyond broader graph fusion that removes
the clone materialization outside this repro boundary.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

try:
    import triton
    import triton.language as tl
except ModuleNotFoundError:  # pragma: no cover - keeps py_compile useful.
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

N = 1
C = 12
H = 64
W = 64
OUT_M = H
OUT_N = C * W
OUT_STRIDE = (OUT_N, 1)
BLOCK_C = 16
BLOCK_W = 64


def get_inputs() -> tuple[object, ...]:
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return tuple(_harness_get_inputs(REPRO_DIR))


def get_repro_instance() -> torch.nn.Module:
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR).eval()


if triton is not None:

    @triton.jit
    def _permute_clone_kernel(
        input_ptr,
        output_ptr,
        BLOCK_C_: tl.constexpr,
        BLOCK_W_: tl.constexpr,
    ):
        h = tl.program_id(0)
        channels = tl.arange(0, BLOCK_C_)
        widths = tl.arange(0, BLOCK_W_)
        mask = channels[:, None] < 12

        input_offsets = channels[:, None] * 4096 + h * 64 + widths[None, :]
        output_offsets = h * 768 + channels[:, None] * 64 + widths[None, :]
        values = tl.load(input_ptr + input_offsets, mask=mask, other=0.0)
        tl.store(output_ptr + output_offsets, values, mask=mask)


def _validate_inputs(inputs: tuple[object, ...]) -> tuple[torch.Tensor, tuple[int, int]]:
    if len(inputs) != 2:
        raise ValueError(f"{REPRO_ID} expects 2 inputs, got {len(inputs)}")

    arg81_1, shape_param = inputs
    if not isinstance(arg81_1, torch.Tensor):
        raise TypeError("expected arg81_1 to be a tensor")
    if arg81_1.shape != (N, C, H, W):
        raise ValueError(f"unexpected arg81_1 shape: {tuple(arg81_1.shape)}")
    if arg81_1.dtype is not torch.float32:
        raise TypeError(f"expected float32 arg81_1, got {arg81_1.dtype}")
    if not arg81_1.is_cuda:
        raise ValueError("oracle requires CUDA input")
    if not arg81_1.is_contiguous():
        raise ValueError("oracle expects the captured contiguous input layout")

    try:
        out_shape = tuple(int(dim) for dim in shape_param)
    except TypeError as exc:
        raise TypeError("expected _shape_param_0 to be an iterable shape") from exc
    if out_shape != (OUT_M, OUT_N):
        raise ValueError(f"unexpected _shape_param_0: {out_shape}")

    return arg81_1, out_shape


def oracle_forward(inputs: tuple[object, ...]) -> torch.Tensor:
    """Run the full permute -> contiguous clone -> unsafe_view -> view scope."""
    if triton is None:
        raise RuntimeError("triton is required for this oracle")

    arg81_1, out_shape = _validate_inputs(inputs)
    output = torch.empty_strided(
        out_shape,
        OUT_STRIDE,
        device=arg81_1.device,
        dtype=arg81_1.dtype,
    )
    _permute_clone_kernel[(OUT_M,)](
        arg81_1,
        output,
        BLOCK_C_=BLOCK_C,
        BLOCK_W_=BLOCK_W,
        num_warps=4,
        num_stages=1,
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description=f"Oracle for {REPRO_ID}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify correctness against eager Repro",
    )
    parser.add_argument(
        "--bench",
        action="store_true",
        help="Benchmark oracle vs torch.compile",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-2,
        help="Relative tolerance for correctness check",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-2,
        help="Absolute tolerance for correctness check",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=25,
        help="Warmup iterations for benchmark",
    )
    parser.add_argument(
        "--rep",
        type=int,
        default=200,
        help="Repetitions for benchmark",
    )
    parser.add_argument(
        "--no-skip-stochastic",
        action="store_true",
        help="Disable auto-detection and skipping of stochastic outputs",
    )
    parser.add_argument(
        "--all-shapes",
        action="store_true",
        help="Benchmark across all shapes from shapes.txt",
    )
    parser.add_argument(
        "--show-hw",
        action="store_true",
        help="Print GPU hardware info and exit",
    )
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
