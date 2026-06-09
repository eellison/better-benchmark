"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete LayoutLM training embedding-LayerNorm-dropout scope as a canonical full-scope graph, including word and position gathers, canonicalized all-zero bbox coordinate embeddings, token-type embedding, fp32 hidden-size-768 population var_mean, affine epilogue, Inductor-style dropout, final [16384, 768] view, and the rsqrt/768 side output, whereas tuned Inductor already reaches the same timing envelope for the captured decomposed full/select/sub/embedding, normalization, stochastic dropout, view, and side-output graph; Inductor cannot materially improve this local repro through norm-template canonicalization alone because the remaining cost is the required embedding/affine reads, fixed-width row reduction, RNG mask, output store, and side-output store rather than avoidable intermediate traffic; the fix is BANDWIDTH_BOUND: record this as at-floor timing unless a broader embedding, RNG, or LayerNorm codegen improvement moves both paths. Exact stochastic equality is not established when correctness skips the dropout output, so such runs are not_true_floor."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
import torch._inductor.inductor_prims  # noqa: F401 - registers prims.* RNG ops

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


def get_inputs() -> list[Any]:
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance() -> torch.nn.Module:
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle implementation ---
BATCH = 32
SEQ_LEN = 512
ROWS = BATCH * SEQ_LEN
HIDDEN = 768
WORD_VOCAB = 30522
POSITION_VOCAB = 512
BBOX_VOCAB = 1024
TOKEN_TYPES = 2
OUTPUT_SHAPE = (ROWS, HIDDEN)
EPS = 1.0e-12
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.1111111111111112
SEED_COUNT = 25
SEED_INDEX = 0
EXACT_STOCHASTIC_EQUALITY = False
CLASSIFICATION = "BANDWIDTH_BOUND"
_COMPILED_CANONICAL = None


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
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    tuple[int, int],
]:
    if len(inputs) != 12:
        raise ValueError(f"{REPRO_ID} expects 12 inputs, got {len(inputs)}")

    word_embedding = _require_tensor("arg1_1", inputs[0], (WORD_VOCAB, HIDDEN), torch.float32)
    word_ids = _require_tensor("arg0_1", inputs[1], (BATCH, SEQ_LEN), torch.int64)
    position_embedding = _require_tensor("arg3_1", inputs[2], (POSITION_VOCAB, HIDDEN), torch.float32)
    position_ids = _require_tensor("arg2_1", inputs[3], (1, SEQ_LEN), torch.int64)
    x_position_embedding = _require_tensor("arg4_1", inputs[4], (BBOX_VOCAB, HIDDEN), torch.float32)
    y_position_embedding = _require_tensor("arg5_1", inputs[5], (BBOX_VOCAB, HIDDEN), torch.float32)
    h_position_embedding = _require_tensor("arg6_1", inputs[6], (BBOX_VOCAB, HIDDEN), torch.float32)
    w_position_embedding = _require_tensor("arg7_1", inputs[7], (BBOX_VOCAB, HIDDEN), torch.float32)
    token_type_embedding = _require_tensor("arg8_1", inputs[8], (TOKEN_TYPES, HIDDEN), torch.float32)
    weight = _require_tensor("arg9_1", inputs[9], (HIDDEN,), torch.float32)
    bias = _require_tensor("arg10_1", inputs[10], (HIDDEN,), torch.float32)

    output_shape = _shape_tuple(inputs[11])
    if output_shape != OUTPUT_SHAPE:
        raise ValueError(f"unexpected output view shape: {inputs[11]!r}")

    device = word_embedding.device
    tensor_inputs = (
        word_ids,
        position_embedding,
        position_ids,
        x_position_embedding,
        y_position_embedding,
        h_position_embedding,
        w_position_embedding,
        token_type_embedding,
        weight,
        bias,
    )
    if not word_embedding.is_cuda:
        raise RuntimeError("CUDA tensors are required for this oracle")
    if any(tensor.device != device for tensor in tensor_inputs):
        raise ValueError("all tensor inputs must be on the same device")

    return (
        word_embedding,
        word_ids,
        position_embedding,
        position_ids,
        x_position_embedding,
        y_position_embedding,
        h_position_embedding,
        w_position_embedding,
        token_type_embedding,
        weight,
        bias,
        output_shape,
    )


def _make_inductor_seed(device: torch.device) -> torch.Tensor:
    seeds = torch.ops.prims.inductor_seeds.default(SEED_COUNT, device=device)
    return torch.ops.prims.inductor_lookup_seed.default(seeds, SEED_INDEX)


def _canonical_torch_forward(
    word_embedding: torch.Tensor,
    word_ids: torch.Tensor,
    position_embedding: torch.Tensor,
    position_ids: torch.Tensor,
    x_position_embedding: torch.Tensor,
    y_position_embedding: torch.Tensor,
    h_position_embedding: torch.Tensor,
    w_position_embedding: torch.Tensor,
    token_type_embedding: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    output_shape: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    word = torch.ops.aten.embedding.default(word_embedding, word_ids, 0)
    position = torch.ops.aten.embedding.default(position_embedding, position_ids)
    x = torch.ops.aten.add.Tensor(word, position)
    x = torch.ops.aten.add.Tensor(x, x_position_embedding[0])
    x = torch.ops.aten.add.Tensor(x, y_position_embedding[0])
    x = torch.ops.aten.add.Tensor(x, x_position_embedding[0])
    x = torch.ops.aten.add.Tensor(x, y_position_embedding[0])
    x = torch.ops.aten.add.Tensor(x, h_position_embedding[0])
    x = torch.ops.aten.add.Tensor(x, w_position_embedding[0])
    x = torch.ops.aten.add.Tensor(x, token_type_embedding[0])

    var, mean = torch.ops.aten.var_mean.correction(x, [2], correction=0, keepdim=True)
    invstd = torch.ops.aten.rsqrt.default(torch.ops.aten.add.Tensor(var, EPS))
    y = torch.ops.aten.mul.Tensor(torch.ops.aten.sub.Tensor(x, mean), invstd)
    y = torch.ops.aten.mul.Tensor(y, weight)
    y = torch.ops.aten.add.Tensor(y, bias)

    seed = _make_inductor_seed(word_embedding.device)
    random = torch.ops.prims.inductor_random.default((BATCH, SEQ_LEN, HIDDEN), seed, "rand")
    keep = torch.ops.aten.gt.Scalar(random, DROPOUT_P)
    y = torch.ops.aten.mul.Tensor(keep, y)
    y = torch.ops.aten.mul.Tensor(y, DROPOUT_SCALE)
    return torch.ops.aten.view.default(y, output_shape), torch.ops.aten.div.Tensor(invstd, HIDDEN)


def _get_compiled_canonical():
    global _COMPILED_CANONICAL
    if _COMPILED_CANONICAL is None:
        _COMPILED_CANONICAL = torch.compile(_canonical_torch_forward, fullgraph=True)
    return _COMPILED_CANONICAL


@oracle_impl(hardware="H100", shapes="(T([30522, 768], f32), T([32, 512], i64, gen=Index(30522)), T([512, 768], f32), T([1, 512], i64, gen=Index(512)), T([1024, 768], f32), T([1024, 768], f32), T([1024, 768], f32), T([1024, 768], f32), T([2, 768], f32), T([768], f32), T([768], f32), S([16384, 768]))")
def oracle_forward(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the complete Repro.forward embedding, LayerNorm, dropout, view, and side-output scope.

    SCOPE INVARIANT: Must accept the same inputs as Repro.forward() and return
    the same outputs (same count, same shapes, same dtypes, same strides).

    Args:
        inputs: tuple of tensors/values from get_inputs(), identical to what
                Repro.forward() receives via Repro()(*inputs).

    Returns:
        Same output structure as Repro()(*inputs).
    """
    (
        word_embedding,
        word_ids,
        position_embedding,
        position_ids,
        x_position_embedding,
        y_position_embedding,
        h_position_embedding,
        w_position_embedding,
        token_type_embedding,
        weight,
        bias,
        output_shape,
    ) = _validate_inputs(inputs)

    return _get_compiled_canonical()(
        word_embedding,
        word_ids,
        position_embedding,
        position_ids,
        x_position_embedding,
        y_position_embedding,
        h_position_embedding,
        w_position_embedding,
        token_type_embedding,
        weight,
        bias,
        output_shape,
    )


def _has_inductor_random() -> bool:
    return "inductor_random" in REPRO_PATH.read_text()


def _check_layout(instance: torch.nn.Module, inputs: list[Any]) -> bool:
    with torch.no_grad():
        expected = instance(*inputs)
        actual = oracle_forward(inputs)
        if any(isinstance(out, torch.Tensor) and out.is_cuda for out in actual):
            torch.cuda.synchronize()

    expected_list = expected if isinstance(expected, tuple) else (expected,)
    actual_list = actual if isinstance(actual, tuple) else (actual,)
    if len(expected_list) != len(actual_list):
        print(f"  layout: FAIL (expected_outputs={len(expected_list)}, oracle_outputs={len(actual_list)})")
        return False

    ok = True
    for index, (expected_out, actual_out) in enumerate(zip(expected_list, actual_list)):
        item_ok = (
            isinstance(expected_out, torch.Tensor)
            and isinstance(actual_out, torch.Tensor)
            and tuple(actual_out.shape) == tuple(expected_out.shape)
            and actual_out.stride() == expected_out.stride()
            and actual_out.dtype == expected_out.dtype
        )
        print(
            f"  output {index} layout: {'PASS' if item_ok else 'FAIL'} "
            f"(expected_shape={tuple(expected_out.shape)}, oracle_shape={tuple(actual_out.shape)}, "
            f"expected_stride={expected_out.stride()}, oracle_stride={actual_out.stride()}, "
            f"expected_dtype={expected_out.dtype}, oracle_dtype={actual_out.dtype})"
        )
        ok = ok and item_ok
    return ok


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
    if has_stochastic_ops(REPRO_PATH) or _has_inductor_random():
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
        ok = _check_layout(instance, inputs) and ok
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
