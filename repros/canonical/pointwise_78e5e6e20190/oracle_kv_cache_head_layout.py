"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete Llama inference KV-cache update and key head-layout materialization in one Triton launch, writing the `[0:32, 1:33, :, :]` cache slice from the packed real view while directly producing the fresh contiguous `[B*H, D, S]` attention-key output, whereas Inductor lowers the captured `slice/copy/slice_scatter/permute/clone/view/copy_` chain as generic layout and mutation operations; Inductor cannot do this today because it does not recognize the cache-slice update plus immediately consumed head-layout clone as one guarded indexed store pattern with a mutable output alias; the fix is NEW_PATTERN: add a KV-cache update/materialize template that fuses the slice overwrite with direct head-major output generation while preserving the returned input alias."""
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

if triton is not None:

    @triton.autotune(
        configs=[
            triton.Config({"SBLOCK": 64, "DBLOCK": 16}, num_warps=4, num_stages=3),
            triton.Config({"SBLOCK": 64, "DBLOCK": 32}, num_warps=4, num_stages=3),
            triton.Config({"SBLOCK": 64, "DBLOCK": 64}, num_warps=8, num_stages=3),
        ],
        key=["OUT_SEQ", "HEAD_DIM"],
    )
    @triton.jit
    def _kv_cache_head_layout_kernel(
        cache_ptr,
        packed_real_ptr,
        output_ptr,
        CACHE_SEQ: tl.constexpr,
        HEADS: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        OUT_SEQ: tl.constexpr,
        SBLOCK: tl.constexpr,
        DBLOCK: tl.constexpr,
    ):
        batch_head = tl.program_id(0)
        dim_block = tl.program_id(1)
        batch = batch_head // HEADS
        head = batch_head - batch * HEADS

        seq = tl.arange(0, SBLOCK)[:, None]
        dim = dim_block * DBLOCK + tl.arange(0, DBLOCK)[None, :]
        mask = (seq < OUT_SEQ) & (dim < HEAD_DIM)

        cache_offsets = ((batch * CACHE_SEQ + seq) * HEADS + head) * HEAD_DIM + dim
        packed_seq = tl.maximum(seq - 1, 0)
        packed_offsets = ((batch * (OUT_SEQ - 1) + packed_seq) * HEADS + head) * HEAD_DIM + dim
        values = tl.load(
            tl.where(seq == 0, cache_ptr + cache_offsets, packed_real_ptr + packed_offsets),
            mask=mask,
            other=0.0,
        )

        out_seq = tl.arange(0, SBLOCK)[None, :]
        out_dim = dim_block * DBLOCK + tl.arange(0, DBLOCK)[:, None]
        output_offsets = batch_head * HEAD_DIM * OUT_SEQ + out_dim * OUT_SEQ + out_seq
        output_mask = (out_dim < HEAD_DIM) & (out_seq < OUT_SEQ)
        tl.store(output_ptr + output_offsets, tl.trans(values), mask=output_mask)
        tl.store(cache_ptr + cache_offsets, values, mask=mask & (seq != 0))


def _shape_tuple(value: Any) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple, torch.Size)):
        raise TypeError(f"expected shape list/tuple, got {type(value)!r}")
    return tuple(int(dim) for dim in value)


def _numel(shape: tuple[int, ...]) -> int:
    result = 1
    for dim in shape:
        result *= dim
    return result


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int, int], tuple[int, int, int], int, int, int]:
    if triton is None:
        raise RuntimeError("Triton is required for oracle_kv_cache_head_layout.py")
    if len(inputs) != 5:
        raise ValueError(f"{REPRO_ID} expects five inputs, got {len(inputs)}")

    cache, packed_real, shape0, shape1, shape2 = inputs
    if not isinstance(cache, torch.Tensor) or not isinstance(packed_real, torch.Tensor):
        raise TypeError(f"{REPRO_ID} expects its first two inputs to be tensors")
    if not cache.is_cuda or not packed_real.is_cuda:
        raise ValueError(f"{REPRO_ID} expects CUDA tensor inputs")
    if cache.dtype != torch.float32 or packed_real.dtype != torch.float32:
        raise TypeError(
            f"{REPRO_ID} expects fp32 tensors, got {cache.dtype} and {packed_real.dtype}"
        )
    if tuple(cache.shape) != (64, 1024, 8, 64):
        raise ValueError(f"unexpected cache shape {tuple(cache.shape)}")
    if tuple(packed_real.shape) != (32, 32, 8, 32, 2):
        raise ValueError(f"unexpected packed real shape {tuple(packed_real.shape)}")
    if not cache.is_contiguous():
        raise ValueError(f"{REPRO_ID} expects contiguous cache, got stride={tuple(cache.stride())}")
    if not packed_real.is_contiguous():
        raise ValueError(
            f"{REPRO_ID} expects contiguous packed real input, got stride={tuple(packed_real.stride())}"
        )

    packed_view_shape = _shape_tuple(shape0)
    if packed_view_shape != (32, 32, 8, 64):
        raise ValueError(f"unexpected packed view shape {packed_view_shape}")
    if _numel(packed_view_shape) != packed_real.numel():
        raise ValueError(f"packed view shape {packed_view_shape} does not match input numel")

    expanded_shape = _shape_tuple(shape1)
    if expanded_shape != (32, 8, 64, 33):
        raise ValueError(f"unexpected expanded shape {expanded_shape}")

    output_shape = _shape_tuple(shape2)
    if output_shape != (256, 64, 33):
        raise ValueError(f"unexpected output shape {output_shape}")

    output_stride = (64 * 33, 33, 1)
    return cache, packed_real, output_shape, output_stride, 1024, 8, 64


@oracle_impl(hardware="H100", shapes="(T([64, 1024, 8, 64], f32), T([32, 32, 8, 32, 2], f32), S([32, 32, 8, 64]), S([32, 8, 64, 33]), S([256, 64, 33]))")
def oracle_forward(inputs):
    """Run the full Repro.forward scope, including the input cache copy_ alias."""
    cache, packed_real, output_shape, output_stride, cache_seq, heads, head_dim = _validate_inputs(inputs)
    output = torch.empty_strided(
        output_shape,
        output_stride,
        device=cache.device,
        dtype=cache.dtype,
    )

    grid = lambda meta: (output_shape[0], triton.cdiv(head_dim, meta["DBLOCK"]))
    _kv_cache_head_layout_kernel[grid](
        cache,
        packed_real,
        output,
        CACHE_SEQ=cache_seq,
        HEADS=heads,
        HEAD_DIM=head_dim,
        OUT_SEQ=output_shape[2],
    )
    return (output, cache)


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
