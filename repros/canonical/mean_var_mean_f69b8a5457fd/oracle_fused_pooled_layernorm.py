"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete BEiT pooled-token LayerNorm scope, including the metadata-only `[25216, 768] -> [128, 197, 768]` view, `gamma * addmm + residual`, class-token drop, mean over the remaining 196 tokens, hidden-size-768 population var_mean, affine epilogue, and the live `rsqrt / 768` side output, whereas tuned Inductor already lowers this fixed norm-template-canonicalization region to the same practical two-kernel memory-traffic envelope; Inductor cannot materially improve this today through local scheduler fusion, scatter-reduce, split-K, algebraic-elimination, or recompute-fusion because the remaining work is dominated by mandatory activation reads, the compact pooled buffer, one row reduction, affine traffic, and launch overhead; the fix is BANDWIDTH_BOUND: record this repro as at_floor unless broader normalization-template, launch, or memory-traffic work moves both implementations."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

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
TOKENS = 197
POOLED_TOKENS = 196
HIDDEN = 768
ADDMM_SHAPE = (BATCH * TOKENS, HIDDEN)
RESIDUAL_SHAPE = (BATCH, TOKENS, HIDDEN)
OUTPUT_SHAPE = (BATCH, HIDDEN)
SIDE_SHAPE = (BATCH, 1)
EPS = 1.0e-6

if triton is not None:

    @triton.jit
    def _pooled_sum_kernel(
        residual_ptr,
        gamma_ptr,
        addmm_ptr,
        pooled_ptr,
        xnumel: tl.constexpr,
        xblock: tl.constexpr,
        rblock: tl.constexpr,
    ):
        xidx = tl.program_id(0) * xblock + tl.arange(0, xblock)[:, None]
        ridx = tl.arange(0, rblock)[None, :]
        xmask = xidx < xnumel
        channel = xidx % 768
        batch = xidx // 768

        gamma = tl.load(gamma_ptr + channel, mask=xmask, other=0.0).to(tl.float32)
        accum = tl.zeros((xblock, rblock), dtype=tl.float32)
        for roffset in tl.range(0, 196, rblock):
            token = roffset + ridx
            rmask = token < 196
            offsets = 768 + channel + 768 * token + 151296 * batch
            addmm = tl.load(
                addmm_ptr + offsets,
                mask=xmask & rmask,
                other=0.0,
                eviction_policy="evict_first",
            ).to(tl.float32)
            residual = tl.load(
                residual_ptr + offsets,
                mask=xmask & rmask,
                other=0.0,
                eviction_policy="evict_first",
            ).to(tl.float32)
            accum += tl.where(rmask, residual + gamma * addmm, 0.0)

        pooled_sum = tl.sum(accum, axis=1)[:, None]
        tl.store(pooled_ptr + xidx, pooled_sum, mask=xmask)

    @triton.jit
    def _layernorm_side_kernel(
        in_out_ptr,
        weight_ptr,
        bias_ptr,
        side_ptr,
        block_h: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, block_h)
        mask = cols < 768
        offsets = row * 768 + cols

        pooled_sum = tl.load(
            in_out_ptr + offsets,
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        pooled = pooled_sum * (1.0 / 196.0)
        mean = tl.sum(tl.where(mask, pooled, 0.0), axis=0) * (1.0 / 768.0)
        centered = tl.where(mask, pooled - mean, 0.0)
        variance = tl.sum(centered * centered, axis=0) * (1.0 / 768.0)
        invstd = tl.rsqrt(variance + 1.0e-6)

        weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        out = centered * invstd * weight + bias
        tl.store(in_out_ptr + offsets, out, mask=mask)
        tl.store(side_ptr + row, invstd * (1.0 / 768.0))


def _shape_tuple(value: Any) -> tuple[int, ...]:
    try:
        return tuple(int(dim) for dim in value)
    except TypeError as exc:
        raise TypeError(f"expected shape parameter, got {type(value).__name__}") from exc


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
    if len(inputs) != 6:
        raise ValueError(f"{REPRO_ID} expects six inputs, got {len(inputs)}")

    addmm = _require_tensor("addmm_47", inputs[0], ADDMM_SHAPE, torch.float32)
    gamma = _require_tensor("arg213_1", inputs[1], (HIDDEN,), torch.float32)
    residual = _require_tensor("add_79", inputs[2], RESIDUAL_SHAPE, torch.float32)
    weight = _require_tensor("arg220_1", inputs[3], (HIDDEN,), torch.float32)
    bias = _require_tensor("arg221_1", inputs[4], (HIDDEN,), torch.float32)
    if _shape_tuple(inputs[5]) != RESIDUAL_SHAPE:
        raise ValueError(f"unexpected reshape shape parameter: {inputs[5]!r}")

    device = addmm.device
    if not (gamma.device == residual.device == weight.device == bias.device == device):
        raise ValueError("all tensor inputs must be on the same device")
    return addmm, gamma, residual, weight, bias


def _torch_full_scope(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    addmm, gamma, residual, weight, bias = _validate_inputs(inputs)
    view = torch.ops.aten.view.default(addmm, _shape_tuple(inputs[5]))
    x = torch.ops.aten.add.Tensor(residual, torch.ops.aten.mul.Tensor(gamma, view))
    pooled = torch.ops.aten.mean.dim(
        torch.ops.aten.slice.Tensor(x, 1, 1, 9223372036854775807), [1]
    )
    variance, mean = torch.ops.aten.var_mean.correction(
        pooled, [1], correction=0, keepdim=True
    )
    invstd = torch.ops.aten.rsqrt.default(variance + EPS)
    out = (pooled - mean) * invstd * weight + bias
    return out, invstd / HIDDEN


def oracle_forward(inputs):
    """Run the complete Repro.forward computation.

    SCOPE INVARIANT: accepts the same six inputs as Repro.forward() and returns
    the same float32 [128,768] affine LayerNorm output plus [128,1] side output.
    """
    addmm, gamma, residual, weight, bias = _validate_inputs(inputs)
    if triton is None or not addmm.is_cuda:
        return _torch_full_scope(inputs)

    output = torch.empty_strided(
        OUTPUT_SHAPE,
        (HIDDEN, 1),
        device=addmm.device,
        dtype=torch.float32,
    )
    side = torch.empty_strided(
        SIDE_SHAPE,
        (1, 1),
        device=addmm.device,
        dtype=torch.float32,
    )
    _pooled_sum_kernel[(triton.cdiv(BATCH * HIDDEN, 64),)](
        residual,
        gamma,
        addmm,
        output,
        BATCH * HIDDEN,
        xblock=64,
        rblock=8,
        num_warps=4,
    )
    _layernorm_side_kernel[(BATCH,)](
        output,
        weight,
        bias,
        side,
        block_h=1024,
        num_warps=8,
    )
    return output, side


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
