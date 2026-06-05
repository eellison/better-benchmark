"""Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle computes the complete Longformer training bias-add, degenerate seed-index-34 dropout, residual LayerNorm, affine epilogue, final `[8192,768]` view, and `rsqrt/768` side output by folding the captured `rand > 1e-30`/scale-1.0 dropout producer to the identity before one fixed-hidden Triton row-normalization kernel, whereas Inductor currently keeps the seeded RNG/mask/mul producer live and schedules it around the norm template; Inductor cannot do this today because its algebraic simplifier does not canonicalize near-zero-threshold Inductor dropout masks before normalization scheduling; the fix is ALGEBRAIC_ELIMINATION: teach Inductor to fold degenerate dropout/mask/mul producers and feed the simplified bias/residual expression directly into the LayerNorm template with the inverse-std side-output store."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
import torch._inductor.inductor_prims  # noqa: F401 - registers prims RNG ops for eager fallback.

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


REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPRO_PATH = REPRO_DIR / "repro.py"

ROWS = 8192
BATCH = 8
SEQ_LEN = 1024
HIDDEN = 768
INPUT_SHAPE = (ROWS, HIDDEN)
RESIDUAL_SHAPE = (BATCH, SEQ_LEN, HIDDEN)
OUTPUT_SHAPE = (ROWS, HIDDEN)
OUTPUT_STRIDE = (HIDDEN, 1)
SIDE_SHAPE = (BATCH, SEQ_LEN, 1)
SIDE_STRIDE = (SEQ_LEN, 1, 1)
SEED_COUNT = 36
SEED_INDEX = 34
EPS = 1.0e-5
BLOCK_H = 1024
CLASSIFICATION = "ALGEBRAIC_ELIMINATION"


def get_inputs() -> list[Any]:
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance() -> torch.nn.Module:
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.autotune(
        configs=[
            triton.Config({"ROW_BLOCK": 1}, num_warps=1, num_stages=3),
            triton.Config({"ROW_BLOCK": 1}, num_warps=2, num_stages=3),
            triton.Config({"ROW_BLOCK": 1}, num_warps=4, num_stages=3),
            triton.Config({"ROW_BLOCK": 2}, num_warps=1, num_stages=3),
            triton.Config({"ROW_BLOCK": 2}, num_warps=2, num_stages=3),
            triton.Config({"ROW_BLOCK": 2}, num_warps=4, num_stages=3),
            triton.Config({"ROW_BLOCK": 4}, num_warps=1, num_stages=3),
            triton.Config({"ROW_BLOCK": 4}, num_warps=2, num_stages=3),
            triton.Config({"ROW_BLOCK": 4}, num_warps=4, num_stages=3),
            triton.Config({"ROW_BLOCK": 4}, num_warps=8, num_stages=3),
        ],
        key=["hidden"],
    )
    @triton.jit
    def _bias_residual_layernorm_side_kernel(
        mm_ptr,
        pre_dropout_bias_ptr,
        residual_ptr,
        weight_ptr,
        bias_ptr,
        out_ptr,
        side_ptr,
        hidden: tl.constexpr,
        eps: tl.constexpr,
        block_h: tl.constexpr,
        total_rows: tl.constexpr,
        ROW_BLOCK: tl.constexpr,
    ):
        row_ids = tl.program_id(0) * ROW_BLOCK + tl.arange(0, ROW_BLOCK)
        rows = row_ids[:, None]
        cols = tl.arange(0, block_h)[None, :]
        row_mask_1d = row_ids < total_rows
        row_mask = row_mask_1d[:, None]
        col_mask = cols < hidden
        mask = row_mask & col_mask
        offsets = rows * hidden + cols

        mm = tl.load(
            mm_ptr + offsets,
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        pre_dropout_bias = tl.load(
            pre_dropout_bias_ptr + cols,
            mask=col_mask,
            other=0.0,
            eviction_policy="evict_last",
        ).to(tl.float32)
        residual = tl.load(
            residual_ptr + offsets,
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        x = mm + pre_dropout_bias + residual

        x_for_reduce = tl.where(mask, x, 0.0)
        sum_x = tl.sum(x_for_reduce, axis=1)
        sum_x2 = tl.sum(x_for_reduce * x_for_reduce, axis=1)
        mean_1d = sum_x / hidden
        variance = sum_x2 / hidden - mean_1d * mean_1d
        invstd_1d = tl.rsqrt(tl.maximum(variance, 0.0) + eps)

        weight = tl.load(
            weight_ptr + cols,
            mask=col_mask,
            other=0.0,
            eviction_policy="evict_last",
        ).to(tl.float32)
        bias = tl.load(
            bias_ptr + cols,
            mask=col_mask,
            other=0.0,
            eviction_policy="evict_last",
        ).to(tl.float32)
        y = (x - mean_1d[:, None]) * invstd_1d[:, None] * weight + bias

        tl.store(out_ptr + offsets, y, mask=mask)
        tl.store(side_ptr + row_ids, invstd_1d / hidden, mask=row_mask_1d)


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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(inputs) != 8:
        raise ValueError(f"{REPRO_ID} expects 8 inputs, got {len(inputs)}")

    mm_47 = _require_tensor("mm_47", inputs[0], INPUT_SHAPE, torch.float32)
    pre_dropout_bias = _require_tensor("arg186_1", inputs[1], (HIDDEN,), torch.float32)
    seeds = _require_tensor("inductor_seeds", inputs[2], (SEED_COUNT,), torch.int64)
    residual = _require_tensor("add_131", inputs[3], RESIDUAL_SHAPE, torch.float32)
    weight = _require_tensor("arg187_1", inputs[4], (HIDDEN,), torch.float32)
    bias = _require_tensor("arg188_1", inputs[5], (HIDDEN,), torch.float32)

    if _shape_tuple(inputs[6]) != RESIDUAL_SHAPE:
        raise ValueError(f"unexpected input view shape parameter: {inputs[6]!r}")
    if _shape_tuple(inputs[7]) != OUTPUT_SHAPE:
        raise ValueError(f"unexpected output view shape parameter: {inputs[7]!r}")

    devices = {value.device for value in (mm_47, pre_dropout_bias, seeds, residual, weight, bias)}
    if len(devices) != 1:
        raise ValueError(f"all tensor inputs must be on one device, got {devices}")
    return mm_47, pre_dropout_bias, seeds, residual, weight, bias


def _torch_full_scope(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    mm_47, pre_dropout_bias, _seeds, residual, weight, bias = _validate_inputs(inputs)
    x = torch.ops.aten.view.default(mm_47, RESIDUAL_SHAPE)
    x = torch.ops.aten.add.Tensor(x, pre_dropout_bias)
    x = torch.ops.aten.add.Tensor(x, residual)
    variance, mean = torch.ops.aten.var_mean.correction(
        x, [2], correction=0, keepdim=True
    )
    invstd = torch.ops.aten.rsqrt.default(variance + EPS)
    y = (x - mean) * invstd * weight + bias
    return torch.ops.aten.view.default(y, OUTPUT_SHAPE), invstd / HIDDEN


def oracle_forward(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the repro scope after folding the degenerate dropout producer."""
    mm_47, pre_dropout_bias, _seeds, residual, weight, bias = _validate_inputs(inputs)
    if triton is None or not mm_47.is_cuda:
        return _torch_full_scope(inputs)

    output = torch.empty_strided(
        OUTPUT_SHAPE,
        OUTPUT_STRIDE,
        device=mm_47.device,
        dtype=torch.float32,
    )
    side = torch.empty_strided(
        SIDE_SHAPE,
        SIDE_STRIDE,
        device=mm_47.device,
        dtype=torch.float32,
    )

    grid = lambda meta: (triton.cdiv(ROWS, meta["ROW_BLOCK"]),)
    _bias_residual_layernorm_side_kernel[grid](
        mm_47,
        pre_dropout_bias,
        residual,
        weight,
        bias,
        output,
        side,
        hidden=HIDDEN,
        eps=EPS,
        block_h=BLOCK_H,
        total_rows=ROWS,
    )
    return output, side


def _repro_has_stochastic_ops() -> bool:
    return has_stochastic_ops(REPRO_PATH) or "inductor_random" in REPRO_PATH.read_text()


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

    if _repro_has_stochastic_ops():
        print(
            f"NOTE: {REPRO_ID} contains prims.inductor_random; this oracle "
            "folds the degenerate rand > 1e-30 dropout mask, and the shared "
            "checker may skip stochastic output values"
        )

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
