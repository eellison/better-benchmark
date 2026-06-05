"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete captured bool-cast, scalar multiply, elementwise multiply, reshape, mask-to-zero, and `[0, 2, 3]` sum as one Triton column-block reduction that reads only the original `gt`, `mm`, and `le` inputs and writes the final f32 `[1280]` output, whereas Inductor lowers the decomposed reshape/where/reduction schedule through generic fusion boundaries around the masked reduction; Inductor cannot do this today because its scheduler does not canonicalize size-1 reshape dimensions and a boolean `where` producer into a single layout-specific masked column reduction; the fix is SCHEDULER_FUSION: inline the predicate and pointwise producers into the reduction schedule and drop the metadata-only reshape before codegen."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - keeps py_compile useful without Triton.
    triton = None
    tl = None

from oracle_harness import (
    check_oracle,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
)


REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name

ROWS = 512
COLS = 1280
SCALE = 1.25

COMPILE_CONFIGS = [
    ("coordinate_descent_tuning", {"coordinate_descent_tuning": True}),
    (
        "combo_looped_cd",
        {
            "combo_kernels": True,
            "combo_kernel_per_subkernel_blocks": True,
            "coordinate_descent_tuning": True,
            "benchmark_combo_kernel": True,
            "triton.multi_kernel": 3,
        },
    ),
]


def get_inputs() -> list[Any]:
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance() -> torch.nn.Module:
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _masked_sum_kernel(
        gt_ptr,
        mm_ptr,
        le_ptr,
        out_ptr,
        n_cols: tl.constexpr,
        scale: tl.constexpr,
        BLOCK_COLS: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
    ):
        col_offsets = tl.program_id(0) * BLOCK_COLS + tl.arange(0, BLOCK_COLS)
        row_offsets = tl.arange(0, BLOCK_ROWS)
        offsets = row_offsets[:, None] * n_cols + col_offsets[None, :]
        mask = col_offsets[None, :] < n_cols

        gt = tl.load(gt_ptr + offsets, mask=mask, other=0)
        le = tl.load(le_ptr + offsets, mask=mask, other=1)
        mm = tl.load(mm_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

        keep = gt & (le == 0)
        values = tl.where(keep, mm * scale, 0.0)
        reduced = tl.sum(values, axis=0)
        tl.store(out_ptr + col_offsets, reduced, mask=col_offsets < n_cols)


def _require_triton_cuda() -> None:
    if triton is None:
        raise RuntimeError("Triton is required for oracle_masked_sum.py")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for oracle_masked_sum.py")


def _require_tensor(
    name: str,
    value: Any,
    shape: tuple[int, ...],
    stride: tuple[int, ...],
    dtype: torch.dtype,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value)!r}")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} has shape {tuple(value.shape)}, expected {shape}")
    if tuple(value.stride()) != stride:
        raise ValueError(f"{name} has stride {tuple(value.stride())}, expected {stride}")
    if value.dtype is not dtype:
        raise TypeError(f"{name} has dtype {value.dtype}, expected {dtype}")
    if not value.is_cuda:
        raise RuntimeError(f"{name} must be a CUDA tensor")
    return value


def _validate_inputs(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(inputs) != 4:
        raise ValueError(f"{REPRO_ID} expects 4 inputs, got {len(inputs)}")

    gt, mm, le, shape_param = inputs
    gt_t = _require_tensor("gt", gt, (ROWS, COLS), (COLS, 1), torch.bool)
    mm_t = _require_tensor("mm", mm, (ROWS, COLS), (COLS, 1), torch.float32)
    le_t = _require_tensor("le", le, (ROWS, COLS, 1, 1), (COLS, 1, 1, 1), torch.bool)

    if list(shape_param) != [ROWS, COLS, 1, 1]:
        raise ValueError(f"unexpected reshape parameter: {shape_param!r}")
    if gt_t.device != mm_t.device or gt_t.device != le_t.device:
        raise ValueError("all tensor inputs must be on the same CUDA device")
    return gt_t, mm_t, le_t


def _launch_oracle(
    gt: torch.Tensor,
    mm: torch.Tensor,
    le: torch.Tensor,
    out: torch.Tensor,
    *,
    block_cols: int,
) -> torch.Tensor:
    _require_triton_cuda()
    if tuple(out.shape) != (COLS,) or out.dtype is not torch.float32:
        raise ValueError(f"out must be f32[{COLS}], got shape={tuple(out.shape)} dtype={out.dtype}")
    if out.device != mm.device or not out.is_cuda:
        raise RuntimeError("out must be a CUDA tensor on the input device")

    grid = (triton.cdiv(COLS, block_cols),)
    _masked_sum_kernel[grid](
        gt,
        mm,
        le,
        out,
        n_cols=COLS,
        scale=SCALE,
        BLOCK_COLS=block_cols,
        BLOCK_ROWS=ROWS,
        num_warps=8,
        num_stages=3,
    )
    return out


def oracle_forward(inputs: list[Any] | tuple[Any, ...], *, block_cols: int = 16) -> torch.Tensor:
    """Run the full Repro.forward computation.

    SCOPE INVARIANT: accepts the same inputs as Repro.forward() and returns the
    same single contiguous f32 `[1280]` tensor.
    """
    gt, mm, le = _validate_inputs(inputs)
    out = torch.empty_strided((COLS,), (1,), device=mm.device, dtype=torch.float32)
    return _launch_oracle(gt, mm, le, out, block_cols=block_cols)


def _bench_cuda(fn, *, warmup: int, rep: int) -> float:
    from triton.testing import do_bench

    return do_bench(fn, warmup=warmup, rep=rep, return_mode="min") * 1000.0


def _compile_with_config(instance: torch.nn.Module, inputs: list[Any], config: dict[str, object], warmup: int):
    import torch._dynamo
    import torch._inductor.config as inductor_config

    torch._dynamo.reset()
    model = instance.__class__().cuda()
    with inductor_config.patch(config):
        compiled = torch.compile(model)
        with torch.no_grad():
            for _ in range(max(1, warmup)):
                compiled(*inputs)
            torch.cuda.synchronize()
    return compiled


def run_check(args: argparse.Namespace) -> bool:
    inputs = get_inputs()
    instance = get_repro_instance()
    ok = check_oracle(
        lambda values: oracle_forward(values, block_cols=args.block_cols),
        instance,
        inputs,
        atol=args.atol,
        rtol=args.rtol,
        skip_stochastic=False,
    )
    print(f"Correctness: {'PASS' if ok else 'FAIL'}")
    return bool(ok)


def run_bench(args: argparse.Namespace) -> dict[str, object]:
    inputs = get_inputs()
    instance = get_repro_instance()
    gt, mm, le = _validate_inputs(inputs)
    out = torch.empty_strided((COLS,), (1,), device=mm.device, dtype=torch.float32)

    with torch.no_grad():
        _launch_oracle(gt, mm, le, out, block_cols=args.block_cols)
        torch.cuda.synchronize()
        oracle_us = _bench_cuda(
            lambda: _launch_oracle(gt, mm, le, out, block_cols=args.block_cols),
            warmup=args.warmup,
            rep=args.rep,
        )

    compile_results: dict[str, float] = {}
    with torch.no_grad():
        for label, config in COMPILE_CONFIGS:
            compiled = _compile_with_config(instance, inputs, config, warmup=args.warmup)
            compile_results[label] = _bench_cuda(lambda: compiled(*inputs), warmup=args.warmup, rep=args.rep)
            del compiled
            torch.cuda.empty_cache()

    best_compile_us = min(compile_results.values())
    result = {
        "repro_id": REPRO_ID,
        "oracle_us": round(oracle_us, 3),
        "compile_us": round(compile_results["coordinate_descent_tuning"], 3),
        "combo_compile_us": round(compile_results["combo_looped_cd"], 3),
        "best_compile_us": round(best_compile_us, 3),
        "ratio": round(best_compile_us / oracle_us, 3),
        "true_floor": oracle_us < best_compile_us,
        "status": "GOOD" if oracle_us < best_compile_us else "DIAGNOSIS_ONLY",
        "block_cols": args.block_cols,
    }
    print(json.dumps(result))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=f"Oracle for {REPRO_ID}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--check", action="store_true", help="Verify correctness against eager Repro")
    parser.add_argument("--bench", action="store_true", help="Benchmark oracle and required compile configs")
    parser.add_argument("--rtol", type=float, default=1e-2, help="Relative tolerance for correctness check")
    parser.add_argument("--atol", type=float, default=1e-2, help="Absolute tolerance for correctness check")
    parser.add_argument("--warmup", type=int, default=25, help="Warmup iterations for benchmark")
    parser.add_argument("--rep", type=int, default=200, help="Repetitions for benchmark")
    parser.add_argument("--block-cols", type=int, default=16, help="Triton output columns per program")
    args = parser.parse_args()

    if not args.check and not args.bench:
        args.check = args.bench = True

    _require_triton_cuda()
    with torch.no_grad():
        if args.check and not run_check(args):
            sys.exit(1)
        if args.bench:
            run_bench(args)


if __name__ == "__main__":
    main()
