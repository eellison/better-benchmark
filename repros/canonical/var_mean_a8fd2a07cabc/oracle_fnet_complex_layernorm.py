"""Gap diagnosis (classification: NEW_PATTERN): this oracle exactly replays the complete FNet residual LayerNorm-to-complex64 scope, including the view, residual add, fp32 population var_mean over hidden size 768, eps=1e-12 rsqrt, affine scale/bias, and complex64 conversion, whereas Inductor currently lowers the real-valued normalization and real-to-complex materialization as generic scheduled work; Inductor cannot do this today because its normalization template/codegen does not support a fixed-hidden LayerNorm epilogue that preserves the exact complex64 conversion contract; the fix is NEW_PATTERN: add a residual-LayerNorm-to-complex64 template that folds the view, add, var_mean, affine, and exact real/zero-imaginary complex stores into one lowering."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPRO_PATH = REPRO_DIR / "repro.py"

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


ROWS = 16384
BATCH = 32
SEQ_LEN = 512
HIDDEN = 768
EPS = 1.0e-12


def get_inputs() -> list[Any]:
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance() -> torch.nn.Module:
    return _harness_get_repro_instance(REPRO_DIR)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    if isinstance(value, torch.Size):
        return tuple(int(dim) for dim in value)
    if isinstance(value, (list, tuple)):
        return tuple(int(dim) for dim in value)
    raise TypeError(f"expected shape parameter list/tuple, got {type(value).__name__}")


def _require_tensor(
    name: str,
    value: Any,
    shape: tuple[int, ...],
    dtype: torch.dtype,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value).__name__}")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} has shape {tuple(value.shape)}, expected {shape}")
    if value.dtype != dtype:
        raise TypeError(f"{name} has dtype {value.dtype}, expected {dtype}")
    if not value.is_contiguous():
        raise ValueError(f"{name} must be contiguous, got stride={value.stride()}")
    return value


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(inputs) != 5:
        raise ValueError(f"{REPRO_ID} expects 5 inputs, got {len(inputs)}")

    addmm = _require_tensor("addmm_22", inputs[0], (ROWS, HIDDEN), torch.float32)
    residual = _require_tensor(
        "add_86", inputs[1], (BATCH, SEQ_LEN, HIDDEN), torch.float32
    )
    weight = _require_tensor("arg96_1", inputs[2], (HIDDEN,), torch.float32)
    bias = _require_tensor("arg97_1", inputs[3], (HIDDEN,), torch.float32)

    if _shape_tuple(inputs[4]) != (BATCH, SEQ_LEN, HIDDEN):
        raise ValueError(f"unexpected view shape parameter: {inputs[4]!r}")

    device = addmm.device
    for tensor in (residual, weight, bias):
        if tensor.device != device:
            raise ValueError("all tensor inputs must be on the same device")
    return addmm, residual, weight, bias


@oracle_impl(hardware="H100", shapes="(T([16384, 768], f32), T([32, 512, 768], f32), T([768], f32), T([768], f32), S([32, 512, 768]))")
def oracle_forward(inputs: list[Any] | tuple[Any, ...]) -> torch.Tensor:
    """Run the exact full Repro.forward computation."""
    addmm, residual, weight, bias = _validate_inputs(inputs)
    view_default = torch.ops.aten.view.default(addmm, inputs[4])
    add_tensor = torch.ops.aten.add.Tensor(view_default, residual)
    variance, mean = torch.ops.aten.var_mean.correction(
        add_tensor, [2], correction=0, keepdim=True
    )
    normalized = torch.ops.aten.mul.Tensor(
        torch.ops.aten.sub.Tensor(add_tensor, mean),
        torch.ops.aten.rsqrt.default(torch.ops.aten.add.Tensor(variance, EPS)),
    )
    affine = torch.ops.aten.add.Tensor(
        torch.ops.aten.mul.Tensor(normalized, weight),
        bias,
    )
    return torch.ops.prims.convert_element_type.default(affine, torch.complex64)


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
