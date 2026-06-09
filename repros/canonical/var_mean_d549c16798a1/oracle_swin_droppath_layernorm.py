"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete Swin window-reverse drop-path residual LayerNorm scope, including the static window layout clone, seed-index 38 drop-path mask, residual add, hidden-size-512 population var_mean, rsqrt epsilon epilogue, affine scale/bias, final [25088,512] view, and live rsqrt/512 side output in one Triton row-normalization kernel plus the exact Inductor RNG producer, whereas Inductor currently lowers the window-reverse clone/drop-path producer and the var_mean LayerNorm epilogue as generic scheduled regions with materialized intermediates around the normalization reduction; Inductor cannot do this today because its normalization scheduler cannot index through the Swin window-reverse clone and thread a per-batch stochastic scale into the row-statistics template while preserving the sibling inverse-std side output; the fix is SCHEDULER_FUSION: teach the norm scheduler to fuse static window-reverse layout producers and drop-path residual producers into the LayerNorm row plan with explicit side-output stores."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
import torch._inductor.inductor_prims  # noqa: F401 - registers prims RNG ops

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
BATCH = 128
WINDOW_GRID_H = 2
WINDOW_GRID_W = 2
WINDOW = 7
HEIGHT = 14
WIDTH = 14
SPATIAL = HEIGHT * WIDTH
WINDOW_AREA = WINDOW * WINDOW
HIDDEN = 512
ROWS = BATCH * SPATIAL
ADDMAT_SHAPE = (ROWS, HIDDEN)
RESIDUAL_SHAPE = (BATCH, HEIGHT, WIDTH, HIDDEN)
PARAM_SHAPE = (HIDDEN,)
OUT0_SHAPE = (ROWS, HIDDEN)
OUT0_STRIDE = (HIDDEN, 1)
OUT1_SHAPE = (BATCH, SPATIAL, 1)
OUT1_STRIDE = (SPATIAL, 1, 1)
KEEP_PROB = 0.9130434766411781
EPS = 1.0e-5
SEED_COUNT = 46
SEED_INDEX = 38
BLOCK_H = 512
RNG_OFFSET_INCREMENT = 8

if triton is not None:

    @triton.autotune(
        configs=[
            triton.Config({"ROW_BLOCK": 1}, num_warps=4, num_stages=3),
            triton.Config({"ROW_BLOCK": 1}, num_warps=8, num_stages=3),
            triton.Config({"ROW_BLOCK": 2}, num_warps=8, num_stages=3),
        ],
        key=["hidden"],
    )
    @triton.jit
    def oracle_kernel(
        addmm_ptr,
        random_or_seed_ptr,
        residual_ptr,
        weight_ptr,
        bias_ptr,
        out0_ptr,
        out1_ptr,
        hidden: tl.constexpr,
        keep_prob: tl.constexpr,
        eps: tl.constexpr,
        seed_index: tl.constexpr,
        total_rows: tl.constexpr,
        block_h: tl.constexpr,
        INLINE_RNG: tl.constexpr,
        ROW_BLOCK: tl.constexpr,
    ):
        source_rows = tl.program_id(0) * ROW_BLOCK + tl.arange(0, ROW_BLOCK)
        cols = tl.arange(0, block_h)
        col_mask = cols < hidden
        row_mask = source_rows < total_rows

        window_id = source_rows // 49
        window_pos = source_rows - window_id * 49
        batch = window_id // 4
        window_in_batch = window_id - batch * 4
        window_h = window_in_batch // 2
        window_w = window_in_batch - window_h * 2
        local_h = window_pos // 7
        local_w = window_pos - local_h * 7
        out_h = window_h * 7 + local_h
        out_w = window_w * 7 + local_w
        out_rows = (batch * 196) + (out_h * 14) + out_w

        offsets = source_rows[:, None] * hidden + cols[None, :]
        out_offsets = out_rows[:, None] * hidden + cols[None, :]
        mask = row_mask[:, None] & col_mask[None, :]

        addmm = tl.load(
            addmm_ptr + offsets,
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        residual = tl.load(
            residual_ptr + out_offsets,
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)

        if INLINE_RNG:
            seed = tl.load(random_or_seed_ptr + seed_index)
            random = tl.rand(seed, batch.to(tl.uint32))
        else:
            random = tl.load(random_or_seed_ptr + batch, mask=row_mask, other=1.0).to(tl.float32)
        keep = random < keep_prob
        scale = tl.where(keep, 1.0 / keep_prob, 0.0).to(tl.float32)
        x = residual + addmm * scale[:, None]
        x_for_reduce = tl.where(mask, x, 0.0)

        mean = (tl.sum(x_for_reduce, axis=1) / hidden)[:, None]
        centered = x - mean
        square = centered * centered
        variance = tl.sum(tl.where(mask, square, 0.0), axis=1) / hidden
        inv_std = tl.rsqrt(variance + eps)

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
        out = centered * inv_std[:, None] * weight[None, :] + bias[None, :]

        tl.store(out0_ptr + out_offsets, out, mask=mask)
        tl.store(out1_ptr + out_rows, inv_std / hidden, mask=row_mask)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    try:
        return tuple(int(dim) for dim in value)
    except TypeError as exc:
        raise TypeError(f"expected shape parameter, got {value!r}") from exc


def _require_tensor(
    name: str,
    value: Any,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    stride: tuple[int, ...] | None = None,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value).__name__}")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} has shape {tuple(value.shape)}, expected {shape}")
    if value.dtype != dtype:
        raise TypeError(f"{name} has dtype {value.dtype}, expected {dtype}")
    if not value.is_cuda:
        raise RuntimeError(f"{name} must be a CUDA tensor for this Triton oracle")
    if stride is not None and tuple(value.stride()) != stride:
        raise ValueError(f"{name} has stride {tuple(value.stride())}, expected {stride}")
    return value


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(inputs) != 11:
        raise ValueError(f"{REPRO_ID} expects eleven inputs, got {len(inputs)}")

    addmm = _require_tensor("addmm_81", inputs[0], ADDMAT_SHAPE, torch.float32, OUT0_STRIDE)
    seeds = _require_tensor("inductor_seeds", inputs[1], (SEED_COUNT,), torch.int64, (1,))
    residual = _require_tensor(
        "view_545",
        inputs[2],
        RESIDUAL_SHAPE,
        torch.float32,
        (SPATIAL * HIDDEN, WIDTH * HIDDEN, HIDDEN, 1),
    )
    weight = _require_tensor("arg309_1", inputs[3], PARAM_SHAPE, torch.float32, (1,))
    bias = _require_tensor("arg310_1", inputs[4], PARAM_SHAPE, torch.float32, (1,))

    expected_shapes = (
        (512, 49, 512),
        (-1, 7, 7, 512),
        (-1, 2, 2, 7, 7, 512),
        (-1, 14, 14, 512),
        (128, -1, 512),
        (25088, 512),
    )
    for index, expected in enumerate(expected_shapes, start=5):
        actual = _shape_tuple(inputs[index])
        if actual != expected:
            raise ValueError(f"unexpected shape parameter {index - 5}: {actual}, expected {expected}")

    device = addmm.device
    if any(t.device != device for t in (seeds, residual, weight, bias)):
        raise ValueError("all tensor inputs must be on the same CUDA device")

    return addmm, seeds, residual, weight, bias


def _is_cuda_graph_capturing() -> bool:
    try:
        return bool(torch.cuda.is_current_stream_capturing())
    except RuntimeError:
        return False


def _inductor_random_like_repro(inductor_seeds: torch.Tensor) -> torch.Tensor:
    seed = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, SEED_INDEX)
    if inductor_seeds.is_cuda and not _is_cuda_graph_capturing():
        device = inductor_seeds.device
        try:
            current_offset = torch.cuda._get_rng_state_offset(device)
            if current_offset >= RNG_OFFSET_INCREMENT:
                torch.cuda._set_rng_state_offset(current_offset - RNG_OFFSET_INCREMENT, device)
                try:
                    return torch.ops.prims.inductor_random.default(
                        [BATCH, 1, 1, 1],
                        seed,
                        "rand",
                    )
                finally:
                    torch.cuda._set_rng_state_offset(current_offset, device)
        except RuntimeError:
            pass
    return torch.ops.prims.inductor_random.default([BATCH, 1, 1, 1], seed, "rand")


@oracle_impl(hardware="H100", shapes="(T([25088, 512], f32), T([46], i64), T([128, 14, 14, 512], f32), T([512], f32), T([512], f32), S([512, 49, 512]), S([-1, 7, 7, 512]), S([-1, 2, 2, 7, 7, 512]), S([-1, 14, 14, 512]), S([128, -1, 512]), S([25088, 512]))")
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
        raise RuntimeError("Triton is required for oracle_swin_droppath_layernorm.py")

    addmm, seeds, residual, weight, bias = _validate_inputs(inputs)
    out0 = torch.empty_strided(OUT0_SHAPE, OUT0_STRIDE, device=addmm.device, dtype=torch.float32)
    out1 = torch.empty_strided(OUT1_SHAPE, OUT1_STRIDE, device=addmm.device, dtype=torch.float32)

    grid = lambda meta: (triton.cdiv(ROWS, meta["ROW_BLOCK"]),)
    if _is_cuda_graph_capturing():
        oracle_kernel[grid](
            addmm,
            seeds,
            residual,
            weight,
            bias,
            out0,
            out1,
            hidden=HIDDEN,
            keep_prob=KEEP_PROB,
            eps=EPS,
            seed_index=SEED_INDEX,
            total_rows=ROWS,
            block_h=BLOCK_H,
            INLINE_RNG=True,
        )
    else:
        oracle_kernel[grid](
            addmm,
            seeds,
            residual,
            weight,
            bias,
            out0,
            out1,
            hidden=HIDDEN,
            keep_prob=KEEP_PROB,
            eps=EPS,
            seed_index=SEED_INDEX,
            total_rows=ROWS,
            block_h=BLOCK_H,
            INLINE_RNG=True,
        )
        random_values = _inductor_random_like_repro(seeds)
        oracle_kernel[grid](
            addmm,
            random_values,
            residual,
            weight,
            bias,
            out0,
            out1,
            hidden=HIDDEN,
            keep_prob=KEEP_PROB,
            eps=EPS,
            seed_index=SEED_INDEX,
            total_rows=ROWS,
            block_h=BLOCK_H,
            INLINE_RNG=False,
        )
    return (out0, out1)


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
