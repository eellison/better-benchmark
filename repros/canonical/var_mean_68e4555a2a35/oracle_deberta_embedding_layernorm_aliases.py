"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete DeBERTa word/position embedding add, hidden-size-1536 Welford var_mean, exact `libdevice.rsqrt` affine LayerNorm epilogue, and three aliasing `[4096, 1536]` views in one fixed-hidden Triton row kernel, but parent `bench_oracle()` runs measure it at the same floor as Inductor's fused reduction for this full scope; Inductor is not missing a profitable local scheduler, scatter-reduce, split-K, algebraic, or recompute transformation here because the mandatory gathered embedding reads, Welford row reduction, affine parameter reads, and final store dominate either implementation; the fix is BANDWIDTH_BOUND: record this as an at-floor normalization/embedding alias case unless broader normalization-template or memory-traffic work improves both paths."""
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
BATCH = 8
SEQ_LEN = 512
ROWS = BATCH * SEQ_LEN
HIDDEN = 1536
WORD_VOCAB = 128100
POSITION_VOCAB = 512
BASE_OUTPUT_SHAPE = (BATCH, SEQ_LEN, HIDDEN)
BASE_OUTPUT_STRIDE = (SEQ_LEN * HIDDEN, HIDDEN, 1)
VIEW_OUTPUT_SHAPE = (ROWS, HIDDEN)
VIEW_OUTPUT_STRIDE = (HIDDEN, 1)
BLOCK_H = 2048
EPS = 1.0e-7
CLASSIFICATION = "RECOMPUTE_FUSION"

if triton is not None:
    from torch._inductor.runtime import triton_helpers
    from torch._inductor.runtime.triton_helpers import libdevice

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_M": 1}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_M": 1}, num_warps=8, num_stages=3),
            triton.Config({"BLOCK_M": 2}, num_warps=8, num_stages=3),
        ],
        key=[],
    )
    @triton.jit
    def _deberta_embedding_layernorm_aliases_kernel(
        word_embedding_ptr,
        word_ids_ptr,
        position_embedding_ptr,
        position_ids_ptr,
        weight_ptr,
        bias_ptr,
        out_ptr,
        hidden: tl.constexpr,
        seq_len: tl.constexpr,
        word_vocab: tl.constexpr,
        position_vocab: tl.constexpr,
        eps: tl.constexpr,
        block_h: tl.constexpr,
        rows_total: tl.constexpr,
        BLOCK_M: tl.constexpr,
    ):
        rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
        cols = tl.arange(0, block_h)[None, :]
        row_mask = rows < rows_total
        col_mask = cols < hidden
        mask = row_mask & col_mask

        seq = rows % seq_len
        word_id = tl.load(word_ids_ptr + rows, mask=row_mask, other=0)
        position_id = tl.load(position_ids_ptr + seq, mask=row_mask, other=0)

        tl.device_assert(
            (0 <= word_id) & (word_id < word_vocab),
            "index out of bounds: 0 <= word_id < 128100",
        )
        tl.device_assert(
            (0 <= position_id) & (position_id < position_vocab),
            "index out of bounds: 0 <= position_id < 512",
        )

        word = tl.load(
            word_embedding_ptr + word_id * hidden + cols,
            mask=mask,
            eviction_policy="evict_last",
            other=0.0,
        )
        position = tl.load(
            position_embedding_ptr + position_id * hidden + cols,
            mask=mask,
            eviction_policy="evict_last",
            other=0.0,
        )
        x = word + position

        mean_state = tl.zeros((BLOCK_M, block_h), tl.float32)
        m2_state = tl.zeros((BLOCK_M, block_h), tl.float32)
        count_state = tl.zeros((BLOCK_M, block_h), tl.float32)
        mean_next, m2_next, count_next = triton_helpers.welford_reduce(
            x, mean_state, m2_state, count_state, True
        )
        mean_state = tl.where(mask, mean_next, mean_state)
        m2_state = tl.where(mask, m2_next, m2_state)
        count_state = tl.where(mask, count_next, count_state)
        row_mean, row_m2, _row_count = triton_helpers.welford(
            mean_state, m2_state, count_state, 1
        )

        mean = row_mean[:, None]
        m2 = row_m2[:, None]
        variance = m2 / tl.full((1, 1), 1536.0, tl.float32)
        invstd = libdevice.rsqrt(variance + tl.full((1, 1), 1e-07, tl.float32))

        affine_weight = tl.load(
            weight_ptr + cols,
            mask=col_mask,
            eviction_policy="evict_last",
            other=0.0,
        )
        affine_bias = tl.load(
            bias_ptr + cols,
            mask=col_mask,
            eviction_policy="evict_last",
            other=0.0,
        )

        centered = x - mean
        normalized = centered * invstd
        scaled = normalized * affine_weight
        out = scaled + affine_bias
        offsets = rows * hidden + cols
        tl.store(out_ptr + offsets, out, mask=mask)


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
    if not value.is_cuda:
        raise RuntimeError(f"{name} must be a CUDA tensor for this Triton oracle")
    if not value.is_contiguous():
        raise ValueError(f"{name} must be contiguous, got stride={value.stride()}")
    return value


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    tuple[int, int],
    tuple[int, int],
    tuple[int, int],
]:
    if len(inputs) != 9:
        raise ValueError(f"{REPRO_ID} expects 9 inputs, got {len(inputs)}")

    (
        word_embedding,
        word_ids,
        position_embedding,
        position_ids,
        weight,
        bias,
        out_shape_0,
        out_shape_1,
        out_shape_2,
    ) = inputs

    word_embedding_t = _require_tensor(
        "arg2_1", word_embedding, (WORD_VOCAB, HIDDEN), torch.float32
    )
    word_ids_t = _require_tensor("arg0_1", word_ids, (BATCH, SEQ_LEN), torch.int64)
    position_embedding_t = _require_tensor(
        "arg3_1", position_embedding, (POSITION_VOCAB, HIDDEN), torch.float32
    )
    position_ids_t = _require_tensor(
        "arg1_1", position_ids, (1, SEQ_LEN), torch.int64
    )
    weight_t = _require_tensor("arg4_1", weight, (HIDDEN,), torch.float32)
    bias_t = _require_tensor("arg5_1", bias, (HIDDEN,), torch.float32)

    device = word_embedding_t.device
    tensor_inputs = (
        word_ids_t,
        position_embedding_t,
        position_ids_t,
        weight_t,
        bias_t,
    )
    if any(value.device != device for value in tensor_inputs):
        raise ValueError("all tensor inputs must be on the same CUDA device")

    output_shapes = (
        _shape_tuple(out_shape_0),
        _shape_tuple(out_shape_1),
        _shape_tuple(out_shape_2),
    )
    if any(shape != VIEW_OUTPUT_SHAPE for shape in output_shapes):
        raise ValueError(
            f"unexpected output view shapes: {out_shape_0!r}, {out_shape_1!r}, {out_shape_2!r}"
        )

    return (
        word_embedding_t,
        word_ids_t,
        position_embedding_t,
        position_ids_t,
        weight_t,
        bias_t,
        output_shapes[0],
        output_shapes[1],
        output_shapes[2],
    )


@oracle_impl(hardware="H100", shapes="(T([128100, 1536], f32), T([8, 512], i64, gen=Index(128100)), T([512, 1536], f32), T([1, 512], i64, gen=Index(512)), T([1536], f32), T([1536], f32), S([4096, 1536]), S([4096, 1536]), S([4096, 1536]))")
def oracle_forward(inputs):
    """Run the complete Repro.forward embedding gather + LayerNorm computation.

    SCOPE INVARIANT: Must accept the same inputs as Repro.forward() and return
    the same outputs (same count, same shapes, same dtypes, same strides).

    Args:
        inputs: tuple of tensors/values from get_inputs(), identical to what
                Repro.forward() receives via Repro()(*inputs).

    Returns:
        Same output structure as Repro()(*inputs).
    """
    if triton is None:
        raise RuntimeError("Triton is required for oracle_deberta_embedding_layernorm_aliases.py")

    (
        word_embedding,
        word_ids,
        position_embedding,
        position_ids,
        weight,
        bias,
        out_shape_0,
        out_shape_1,
        out_shape_2,
    ) = _validate_inputs(inputs)

    out_storage = torch.empty_strided(
        BASE_OUTPUT_SHAPE,
        BASE_OUTPUT_STRIDE,
        device=word_embedding.device,
        dtype=torch.float32,
    )
    grid = lambda meta: (triton.cdiv(ROWS, meta["BLOCK_M"]),)
    _deberta_embedding_layernorm_aliases_kernel[grid](
        word_embedding,
        word_ids,
        position_embedding,
        position_ids,
        weight,
        bias,
        out_storage,
        hidden=HIDDEN,
        seq_len=SEQ_LEN,
        word_vocab=WORD_VOCAB,
        position_vocab=POSITION_VOCAB,
        eps=EPS,
        block_h=BLOCK_H,
        rows_total=ROWS,
    )

    view_0 = torch.ops.aten.view.default(out_storage, out_shape_0)
    view_1 = torch.ops.aten.view.default(out_storage, out_shape_1)
    view_2 = torch.ops.aten.view.default(out_storage, out_shape_2)
    return (view_0, view_1, view_2)


def _storage_alias_key(value: torch.Tensor) -> tuple[int, int]:
    return (value.untyped_storage().data_ptr(), int(value.storage_offset()))


def _check_layout_and_aliases(instance, inputs) -> bool:
    with torch.no_grad():
        expected = instance(*inputs)
        actual = oracle_forward(inputs)
        if any(isinstance(value, torch.Tensor) and value.is_cuda for value in actual):
            torch.cuda.synchronize()

    expected_list = list(expected)
    actual_list = list(actual)
    layout_ok = all(
        tuple(actual_value.shape) == tuple(expected_value.shape)
        and actual_value.stride() == expected_value.stride()
        and actual_value.dtype == expected_value.dtype
        for expected_value, actual_value in zip(expected_list, actual_list)
    )
    expected_alias_ok = len({_storage_alias_key(value) for value in expected_list}) == 1
    actual_alias_ok = len({_storage_alias_key(value) for value in actual_list}) == 1
    ok = layout_ok and expected_alias_ok and actual_alias_ok
    print(
        f"  layout/alias: {'PASS' if ok else 'FAIL'} "
        f"(expected_stride={expected_list[0].stride()}, oracle_stride={actual_list[0].stride()}, "
        f"expected_alias={expected_alias_ok}, oracle_alias={actual_alias_ok})"
    )
    return ok


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
        ok = _check_layout_and_aliases(instance, inputs) and ok
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
