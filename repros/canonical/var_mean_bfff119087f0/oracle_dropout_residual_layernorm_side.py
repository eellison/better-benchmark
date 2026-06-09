"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete MegatronBERT seeded-dropout residual LayerNorm scope in one hidden-size-1024 Triton row kernel, including the `[8192, 1024] -> [16, 512, 1024]` view, seed-index-48 p=0.1 dropout on the projected activation, residual add, fp32 population `var_mean(..., dim=2, correction=0, keepdim=True)`, eps=1e-12 affine scale/bias, final `[8192, 1024]` view, and sibling `rsqrt / 1024` side output, whereas Inductor already lowers this norm-template-canonicalization graph into the same fused RNG/dropout/residual/LayerNorm/side-output envelope and CUDAGraph timing shows no usable local gap; Inductor cannot materially improve this local repro through scheduler fusion, scatter-reduce, split-K, algebraic elimination, or recompute fusion because the mandatory activation/residual/affine reads, stochastic mask generation, row reduction, output store, and side-output store dominate; the fix is BANDWIDTH_BOUND: record this repro as at_floor/not_true_floor unless broader normalization-template, RNG, or bandwidth work moves both implementations. Exact stochastic equality for the dropout-dependent output is not proven: the standard check skips both floating outputs, and an explicit no-skip check failed output 0 while the side output matched tolerance."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
import torch._inductor.inductor_prims  # noqa: F401 - registers prims RNG ops.

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - keeps syntax checks usable without Triton.
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
    has_stochastic_ops,
)


BATCH = 16
SEQ_LEN = 512
HIDDEN = 1024
N_ROWS = BATCH * SEQ_LEN

INPUT_SHAPE = (N_ROWS, HIDDEN)
ACTIVATION_SHAPE = (BATCH, SEQ_LEN, HIDDEN)
OUTPUT_SHAPE = (N_ROWS, HIDDEN)
OUTPUT_STRIDE = (HIDDEN, 1)
SIDE_SHAPE = (BATCH, SEQ_LEN, 1)
SIDE_STRIDE = (SEQ_LEN, 1, 1)

SEED_COUNT = 49
SEED_INDEX = 48
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.1111111111111112
EPS = 1.0e-12
BLOCK_H = 1024
EXACT_STOCHASTIC_EQUALITY = False


def get_inputs() -> list[Any]:
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance() -> torch.nn.Module:
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---
if triton is not None:

    @triton.autotune(
        configs=[
            triton.Config({"ROW_BLOCK": 1}, num_warps=4, num_stages=3),
            triton.Config({"ROW_BLOCK": 1}, num_warps=8, num_stages=3),
            triton.Config({"ROW_BLOCK": 2}, num_warps=4, num_stages=3),
            triton.Config({"ROW_BLOCK": 2}, num_warps=8, num_stages=3),
            triton.Config({"ROW_BLOCK": 4}, num_warps=8, num_stages=3),
        ],
        key=["hidden"],
    )
    @triton.jit
    def _dropout_residual_layernorm_side_kernel(
        addmm_ptr,
        seeds_ptr,
        residual_ptr,
        weight_ptr,
        bias_ptr,
        out_ptr,
        invstd_div_ptr,
        hidden: tl.constexpr,
        seed_index: tl.constexpr,
        dropout_p: tl.constexpr,
        dropout_scale: tl.constexpr,
        eps: tl.constexpr,
        block_h: tl.constexpr,
        total_rows: tl.constexpr,
        ROW_BLOCK: tl.constexpr,
    ):
        row_ids = tl.program_id(0) * ROW_BLOCK + tl.arange(0, ROW_BLOCK)
        cols = tl.arange(0, block_h)
        row_mask = row_ids < total_rows
        col_mask = cols < hidden
        mask = row_mask[:, None] & col_mask[None, :]
        offsets = row_ids[:, None] * hidden + cols[None, :]

        projected = tl.load(
            addmm_ptr + offsets,
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        residual = tl.load(
            residual_ptr + offsets,
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)

        seed = tl.load(seeds_ptr + seed_index)
        random = tl.rand(seed, offsets.to(tl.uint32))
        dropped = tl.where(random > dropout_p, projected * dropout_scale, 0.0)
        x = residual + dropped

        reduce_x = tl.where(mask, x, 0.0)
        mean = tl.sum(reduce_x, axis=1) / hidden
        centered = x - mean[:, None]
        variance = tl.sum(tl.where(mask, centered * centered, 0.0), axis=1) / hidden
        invstd = tl.rsqrt(variance + eps)

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
        normalized = centered * invstd[:, None]
        out = normalized * weight[None, :] + bias[None, :]

        tl.store(out_ptr + offsets, out, mask=mask)
        tl.store(invstd_div_ptr + row_ids, invstd / hidden, mask=row_mask)


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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, ...], tuple[int, ...]]:
    if len(inputs) != 7:
        raise ValueError(f"{REPRO_ID} expects 7 inputs, got {len(inputs)}")

    addmm, seeds, residual, weight, bias, shape0, shape1 = inputs
    addmm = _require_tensor("addmm_143", addmm, INPUT_SHAPE, torch.float32)
    seeds = _require_tensor("inductor_seeds", seeds, (SEED_COUNT,), torch.int64)
    residual = _require_tensor("add_165", residual, ACTIVATION_SHAPE, torch.float32)
    weight = _require_tensor("arg390_1", weight, (HIDDEN,), torch.float32)
    bias = _require_tensor("arg391_1", bias, (HIDDEN,), torch.float32)

    shape0_tuple = _shape_tuple(shape0)
    shape1_tuple = _shape_tuple(shape1)
    if shape0_tuple != ACTIVATION_SHAPE:
        raise ValueError(f"unexpected input view shape parameter: {shape0!r}")
    if shape1_tuple != OUTPUT_SHAPE:
        raise ValueError(f"unexpected output view shape parameter: {shape1!r}")

    device = addmm.device
    for tensor in (seeds, residual, weight, bias):
        if tensor.device != device:
            raise ValueError("all tensor inputs must be on the same device")

    return addmm, seeds, residual, weight, bias, shape0_tuple, shape1_tuple


def _torch_full_scope(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    addmm, seeds, residual, weight, bias, shape0, shape1 = _validate_inputs(inputs)
    viewed = torch.ops.aten.view.default(addmm, shape0)
    seed = torch.ops.prims.inductor_lookup_seed.default(seeds, SEED_INDEX)
    random = torch.ops.prims.inductor_random.default(shape0, seed, "rand")
    keep = torch.ops.aten.gt.Scalar(random, DROPOUT_P)
    dropped = torch.ops.aten.mul.Tensor(keep, viewed)
    dropped = torch.ops.aten.mul.Tensor(dropped, DROPOUT_SCALE)
    x = torch.ops.aten.add.Tensor(residual, dropped)
    variance, mean = torch.ops.aten.var_mean.correction(
        x, [2], correction=0, keepdim=True
    )
    invstd = torch.ops.aten.rsqrt.default(variance + EPS)
    normalized = torch.ops.aten.mul.Tensor(x - mean, invstd)
    affine = torch.ops.aten.add.Tensor(normalized * weight, bias)
    return torch.ops.aten.view.default(affine, shape1), invstd / HIDDEN


@oracle_impl(hardware="H100", shapes="(T([8192, 1024], f32), T([49], i64), T([16, 512, 1024], f32), T([1024], f32), T([1024], f32), S([16, 512, 1024]), S([8192, 1024]))")
def oracle_forward(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the full Repro.forward seeded-dropout residual LayerNorm scope."""
    addmm, seeds, residual, weight, bias, _shape0, _shape1 = _validate_inputs(inputs)
    if triton is None or not addmm.is_cuda:
        return _torch_full_scope(inputs)

    output = torch.empty_strided(
        OUTPUT_SHAPE,
        OUTPUT_STRIDE,
        device=addmm.device,
        dtype=torch.float32,
    )
    invstd_div = torch.empty_strided(
        SIDE_SHAPE,
        SIDE_STRIDE,
        device=addmm.device,
        dtype=torch.float32,
    )

    grid = lambda meta: (triton.cdiv(N_ROWS, meta["ROW_BLOCK"]),)
    _dropout_residual_layernorm_side_kernel[grid](
        addmm,
        seeds,
        residual,
        weight,
        bias,
        output,
        invstd_div,
        hidden=HIDDEN,
        seed_index=SEED_INDEX,
        dropout_p=DROPOUT_P,
        dropout_scale=DROPOUT_SCALE,
        eps=EPS,
        block_h=BLOCK_H,
        total_rows=N_ROWS,
    )
    return output, invstd_div


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
