"""Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle computes the complete Longformer training residual LayerNorm scope, including the `[8192,768] -> [8,1024,768]` view, seed-index-32 degenerate `rand > 1e-30` dropout elimination, residual add, population `var_mean` over hidden size 768, eps=1e-5 affine epilogue, final `permute(1,0,2).clone().view([8192,768])` layout, and `rsqrt/768` side output, whereas Inductor keeps the input-seeded RNG, comparison, and no-op multiply live through normalization scheduling and then handles the sequence-major output layout generically; Inductor cannot do this today because its algebraic simplifier does not prove near-zero-threshold unit-scale Inductor dropout masks are identity before norm-template and layout codegen; the fix is ALGEBRAIC_ELIMINATION: fold degenerate dropout producers before the LayerNorm template and emit the sequence-major final store plus inverse-std side output directly."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
import torch._inductor.inductor_prims  # noqa: F401 - registers prims RNG ops for fallback paths.

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - keeps py_compile usable without Triton.
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
    has_stochastic_ops,
)


def get_inputs() -> list[Any]:
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance() -> torch.nn.Module:
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---
BATCH = 8
SEQ_LEN = 1024
HIDDEN = 768
ROWS = BATCH * SEQ_LEN
SEED_COUNT = 36
SEED_INDEX = 32
EPS = 1.0e-5
BLOCK_H = 1024

INPUT_2D_SHAPE = (ROWS, HIDDEN)
INPUT_VIEW_SHAPE = (BATCH, SEQ_LEN, HIDDEN)
OUTPUT_SHAPE = (ROWS, HIDDEN)
OUTPUT_STRIDE = (HIDDEN, 1)
SIDE_SHAPE = (BATCH, SEQ_LEN, 1)
SIDE_STRIDE = (SEQ_LEN, 1, 1)


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
    def _identity_dropout_layernorm_permute_kernel(
        addmm_ptr,
        residual_ptr,
        weight_ptr,
        bias_ptr,
        output_ptr,
        side_ptr,
        hidden: tl.constexpr,
        batch: tl.constexpr,
        seq_len: tl.constexpr,
        eps: tl.constexpr,
        block_h: tl.constexpr,
        total_rows: tl.constexpr,
        ROW_BLOCK: tl.constexpr,
    ):
        out_row_ids = tl.program_id(0) * ROW_BLOCK + tl.arange(0, ROW_BLOCK)
        seq_ids = out_row_ids // batch
        batch_ids = out_row_ids - seq_ids * batch
        in_row_ids = batch_ids * seq_len + seq_ids

        cols = tl.arange(0, block_h)[None, :]
        row_mask_1d = out_row_ids < total_rows
        row_mask = row_mask_1d[:, None]
        col_mask = cols < hidden
        mask = row_mask & col_mask

        load_offsets = in_row_ids[:, None] * hidden + cols
        store_offsets = out_row_ids[:, None] * hidden + cols

        addmm = tl.load(
            addmm_ptr + load_offsets,
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        residual = tl.load(
            residual_ptr + load_offsets,
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        x = addmm + residual

        x_for_reduce = tl.where(mask, x, 0.0)
        mean_1d = tl.sum(x_for_reduce, axis=1) / hidden
        centered = x - mean_1d[:, None]
        centered_for_var = tl.where(mask, centered, 0.0)
        variance = tl.sum(centered_for_var * centered_for_var, axis=1) / hidden
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
        y = centered * invstd_1d[:, None] * weight + bias

        tl.store(output_ptr + store_offsets, y, mask=mask)
        tl.store(side_ptr + in_row_ids, invstd_1d / hidden, mask=row_mask_1d)


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
        raise ValueError(f"{name} must be contiguous, got stride={tuple(value.stride())}")
    return value


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(inputs) != 7:
        raise ValueError(f"{REPRO_ID} expects 7 inputs, got {len(inputs)}")

    addmm = _require_tensor("addmm_21", inputs[0], INPUT_2D_SHAPE, torch.float32)
    seeds = _require_tensor("inductor_seeds", inputs[1], (SEED_COUNT,), torch.int64)
    residual = _require_tensor("add_127", inputs[2], INPUT_VIEW_SHAPE, torch.float32)
    weight = _require_tensor("arg177_1", inputs[3], (HIDDEN,), torch.float32)
    bias = _require_tensor("arg178_1", inputs[4], (HIDDEN,), torch.float32)

    if _shape_tuple(inputs[5]) != INPUT_VIEW_SHAPE:
        raise ValueError(f"unexpected input view shape parameter: {inputs[5]!r}")
    if _shape_tuple(inputs[6]) != OUTPUT_SHAPE:
        raise ValueError(f"unexpected output view shape parameter: {inputs[6]!r}")

    device = addmm.device
    for tensor in (seeds, residual, weight, bias):
        if tensor.device != device:
            raise ValueError("all tensor inputs must be on the same device")

    return addmm, seeds, residual, weight, bias


def _torch_full_scope(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    addmm, _seeds, residual, weight, bias = _validate_inputs(inputs)
    x = torch.ops.aten.view.default(addmm, INPUT_VIEW_SHAPE)
    x = torch.ops.aten.add.Tensor(x, residual)
    variance, mean = torch.ops.aten.var_mean.correction(
        x, [2], correction=0, keepdim=True
    )
    invstd = torch.ops.aten.rsqrt.default(variance + EPS)
    y = torch.ops.aten.sub.Tensor(x, mean)
    y = torch.ops.aten.mul.Tensor(y, invstd)
    y = torch.ops.aten.mul.Tensor(y, weight)
    y = torch.ops.aten.add.Tensor(y, bias)
    y = torch.ops.aten.permute.default(y, [1, 0, 2])
    y = torch.ops.aten.clone.default(y, memory_format=torch.contiguous_format)
    y = torch.ops.aten.view.default(y, OUTPUT_SHAPE)
    return y, invstd / HIDDEN


def oracle_forward(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the full repro scope after folding the degenerate dropout producer."""
    addmm, _seeds, residual, weight, bias = _validate_inputs(inputs)
    if triton is None or not addmm.is_cuda:
        return _torch_full_scope(inputs)

    output = torch.empty_strided(
        OUTPUT_SHAPE,
        OUTPUT_STRIDE,
        device=addmm.device,
        dtype=torch.float32,
    )
    side = torch.empty_strided(
        SIDE_SHAPE,
        SIDE_STRIDE,
        device=addmm.device,
        dtype=torch.float32,
    )

    grid = lambda meta: (triton.cdiv(ROWS, meta["ROW_BLOCK"]),)
    _identity_dropout_layernorm_permute_kernel[grid](
        addmm,
        residual,
        weight,
        bias,
        output,
        side,
        hidden=HIDDEN,
        batch=BATCH,
        seq_len=SEQ_LEN,
        eps=EPS,
        block_h=BLOCK_H,
        total_rows=ROWS,
    )
    return output, side


def _repro_has_stochastic_ops() -> bool:
    return has_stochastic_ops(REPRO_PATH) or "inductor_random" in REPRO_PATH.read_text()


# --- CLI entry point ---
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
    if _repro_has_stochastic_ops():
        print(
            f"NOTE: {REPRO_ID} contains input-seeded Inductor RNG, but this "
            "oracle eliminates the captured identity dropout mask"
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
