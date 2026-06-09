"""
Oracle for var_mean_0990c69ae9bb

Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the
complete BERT embedding-plus-layernorm forward in one Triton row kernel,
including the word embedding gather, token-type gather from `arg2[arg1]`,
position embedding gather, fp32 row `var_mean`, affine epilogue, f16 cast, and
three returned `[512,768]` views, whereas the required coordinate-descent
Inductor config already emits a one-kernel full-scope implementation within the
oracle's launch/noise envelope; Inductor cannot materially improve this today
through scheduler fusion, scatter-reduce, split-K, algebraic elimination, or
recompute fusion because the remaining cost is the mandatory indexed embedding
reads, affine reads, output write, and one small row-normalization launch rather
than a demonstrated avoidable materialization; the fix is BANDWIDTH_BOUND:
record this as an at-floor diagnosis unless a broader launch-overhead or
normalization-template improvement beats the measured Triton floor.
"""
from __future__ import annotations

import argparse
import json
import math
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
    oracle_impl,
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

ROWS = 512
HIDDEN = 768
VOCAB = 30522
TOKEN_TYPES = 2
EPS = 1.0e-12
CLASSIFICATION = "BANDWIDTH_BOUND"

COMPILE_CONFIGS = [
    ("coordinate_descent_tuning=True", {"coordinate_descent_tuning": True}),
    (
        "combo_kernels=True,combo_kernel_per_subkernel_blocks=True,"
        "coordinate_descent_tuning=True,benchmark_combo_kernel=True,"
        "triton.multi_kernel=3",
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
    return _harness_get_repro_instance(REPRO_DIR).eval()


if triton is not None:

    @triton.jit
    def _embedding_layernorm_kernel(
        word_embedding_ptr,
        word_ids_ptr,
        token_type_source_ptr,
        position_ids_ptr,
        token_type_embedding_ptr,
        position_embedding_ptr,
        weight_ptr,
        bias_ptr,
        out_ptr,
        hidden: tl.constexpr,
        eps: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK_N)
        mask = cols < hidden

        word_id = tl.load(word_ids_ptr + row)
        position_id = tl.load(position_ids_ptr + row)
        token_type_id = tl.load(token_type_source_ptr + position_id)

        word = tl.load(
            word_embedding_ptr + word_id * hidden + cols,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        token_type = tl.load(
            token_type_embedding_ptr + token_type_id * hidden + cols,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        position = tl.load(
            position_embedding_ptr + position_id * hidden + cols,
            mask=mask,
            other=0.0,
        ).to(tl.float32)

        x = tl.where(mask, word + token_type + position, 0.0)
        mean = tl.sum(x, axis=0) / hidden
        sq_mean = tl.sum(x * x, axis=0) / hidden
        variance = tl.maximum(sq_mean - mean * mean, 0.0)
        inv_std = tl.rsqrt(variance + eps)

        weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        y = (x - mean) * inv_std * weight + bias

        tl.store(out_ptr + row * hidden + cols, y, mask=mask)


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
    list[int],
    list[int],
    list[int],
    list[int],
]:
    if len(inputs) != 12:
        raise ValueError(f"{REPRO_ID} expects 12 inputs, got {len(inputs)}")

    (
        word_embedding,
        word_ids,
        token_type_source,
        position_ids,
        token_type_embedding,
        position_embedding,
        weight,
        bias,
        token_type_shape,
        out_shape_0,
        out_shape_1,
        out_shape_2,
    ) = inputs

    tensor_inputs = (
        word_embedding,
        word_ids,
        token_type_source,
        position_ids,
        token_type_embedding,
        position_embedding,
        weight,
        bias,
    )
    if not all(isinstance(item, torch.Tensor) for item in tensor_inputs):
        raise TypeError("the first eight oracle inputs must be tensors")
    if not all(item.is_cuda for item in tensor_inputs):
        raise RuntimeError("CUDA tensors are required for the Triton oracle")

    expected_shapes = (
        (VOCAB, HIDDEN),
        (1, ROWS),
        (1, ROWS),
        (1, ROWS),
        (TOKEN_TYPES, HIDDEN),
        (ROWS, HIDDEN),
        (HIDDEN,),
        (HIDDEN,),
    )
    for idx, (tensor, shape) in enumerate(zip(tensor_inputs, expected_shapes)):
        if tuple(tensor.shape) != shape:
            raise ValueError(f"input {idx} shape {tuple(tensor.shape)} != {shape}")
        if not tensor.is_contiguous():
            raise ValueError(f"input {idx} must be contiguous, got stride={tensor.stride()}")

    if word_embedding.dtype != torch.float16:
        raise TypeError(f"word embedding must be f16, got {word_embedding.dtype}")
    if token_type_embedding.dtype != torch.float16 or position_embedding.dtype != torch.float16:
        raise TypeError("token-type and position embeddings must be f16")
    if weight.dtype != torch.float16 or bias.dtype != torch.float16:
        raise TypeError("layernorm weight and bias must be f16")
    if word_ids.dtype != torch.int64 or token_type_source.dtype != torch.int64 or position_ids.dtype != torch.int64:
        raise TypeError("index tensors must be i64")

    if list(token_type_shape) != [1, ROWS]:
        raise ValueError(f"unexpected token-type expand shape: {token_type_shape!r}")
    for idx, shape in enumerate((out_shape_0, out_shape_1, out_shape_2), start=9):
        if list(shape) != [ROWS, HIDDEN]:
            raise ValueError(f"unexpected output shape input {idx}: {shape!r}")

    return (
        word_embedding,
        word_ids,
        token_type_source,
        position_ids,
        token_type_embedding,
        position_embedding,
        weight,
        bias,
        token_type_shape,
        out_shape_0,
        out_shape_1,
        out_shape_2,
    )


@oracle_impl(hardware="H100", shapes="(T([30522, 768], f16), T([1, 512], i64, gen=Index(30522)), T([1, 512], i64, gen=Index(2)), T([1, 512], i64, gen=Index(512)), T([2, 768], f16), T([512, 768], f16), T([768], f16), T([768], f16), S([1, 512]), S([512, 768]), S([512, 768]), S([512, 768]))")
def oracle_forward(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the full Repro.forward embedding + layernorm computation."""
    if triton is None:
        raise RuntimeError("Triton is required for oracle_embedding_layernorm.py")

    (
        word_embedding,
        word_ids,
        token_type_source,
        position_ids,
        token_type_embedding,
        position_embedding,
        weight,
        bias,
        _token_type_shape,
        out_shape_0,
        out_shape_1,
        out_shape_2,
    ) = _validate_inputs(inputs)

    out_storage = torch.empty_strided(
        (1, ROWS, HIDDEN),
        (ROWS * HIDDEN, HIDDEN, 1),
        device=word_embedding.device,
        dtype=word_embedding.dtype,
    )
    _embedding_layernorm_kernel[(ROWS,)](
        word_embedding,
        word_ids,
        token_type_source,
        position_ids,
        token_type_embedding,
        position_embedding,
        weight,
        bias,
        out_storage,
        hidden=HIDDEN,
        eps=EPS,
        BLOCK_N=triton.next_power_of_2(HIDDEN),
        num_warps=8,
        num_stages=3,
    )

    return (
        out_storage.view(out_shape_0),
        out_storage.view(out_shape_1),
        out_storage.view(out_shape_2),
    )


def _normalize_outputs(out: Any) -> list[torch.Tensor]:
    if isinstance(out, torch.Tensor):
        return [out]
    if isinstance(out, (tuple, list)):
        tensors: list[torch.Tensor] = []
        for item in out:
            tensors.extend(_normalize_outputs(item))
        return tensors
    return []


def _check_layout(instance: torch.nn.Module, inputs: list[Any]) -> bool:
    with torch.no_grad():
        expected = _normalize_outputs(instance(*inputs))
        actual = _normalize_outputs(oracle_forward(inputs))
        torch.cuda.synchronize()

    ok = len(expected) == len(actual)
    for idx, (expected_tensor, actual_tensor) in enumerate(zip(expected, actual)):
        this_ok = expected_tensor.stride() == actual_tensor.stride()
        print(
            f"  output {idx} layout: {'PASS' if this_ok else 'FAIL'} "
            f"(expected_stride={expected_tensor.stride()}, oracle_stride={actual_tensor.stride()})"
        )
        ok = ok and this_ok

    if len(actual) >= 3:
        alias_ok = actual[0].data_ptr() == actual[1].data_ptr() == actual[2].data_ptr()
        print(f"  output aliasing: {'PASS' if alias_ok else 'FAIL'}")
        ok = ok and alias_ok
    return ok


def _do_bench(fn: Any, *, warmup: int, rep: int) -> float:
    if triton is None:
        raise RuntimeError("Triton is required for benchmarking")
    from triton.testing import do_bench

    return do_bench(fn, warmup=warmup, rep=rep, return_mode="min") * 1000.0


def _bench_cuda_graph(fn: Any, *, warmup: int, rep: int) -> float:
    with torch.no_grad():
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        try:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                fn()
            torch.cuda.synchronize()
            return _do_bench(lambda: graph.replay(), warmup=warmup, rep=rep)
        except Exception as exc:
            print(f"WARNING: CUDA graph capture failed; falling back to direct timing ({exc})")
            torch.cuda.synchronize()
            return _do_bench(fn, warmup=warmup, rep=rep)


def _bench_compile_config(
    label: str,
    config: dict[str, object],
    inputs: list[Any],
    *,
    warmup: int,
    rep: int,
) -> float:
    import torch._dynamo
    import torch._inductor.config as inductor_config

    torch._dynamo.reset()
    instance = get_repro_instance()
    with inductor_config.patch(config):
        compiled = torch.compile(instance)
        holder: list[Any] = [None]
        with torch.no_grad():
            for _ in range(5):
                holder[0] = compiled(*inputs)
            torch.cuda.synchronize()
            compile_us = _bench_cuda_graph(
                lambda: holder.__setitem__(0, compiled(*inputs)),
                warmup=warmup,
                rep=rep,
            )
    torch.cuda.synchronize()
    print(f"torch.compile {label}: {compile_us:.3f} us")
    return compile_us


def run_bench(*, warmup: int, rep: int) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Triton oracle benchmark")

    inputs = get_inputs()
    holder: list[Any] = [None]
    with torch.no_grad():
        holder[0] = oracle_forward(inputs)
        torch.cuda.synchronize()
        oracle_us = _bench_cuda_graph(
            lambda: holder.__setitem__(0, oracle_forward(inputs)),
            warmup=warmup,
            rep=rep,
        )

    compile_results: dict[str, float] = {}
    for label, config in COMPILE_CONFIGS:
        compile_results[label] = _bench_compile_config(
            label,
            config,
            inputs,
            warmup=warmup,
            rep=rep,
        )

    cd_compile_us = compile_results["coordinate_descent_tuning=True"]
    best_compile_us = min(compile_results.values()) if compile_results else math.nan
    ratio = best_compile_us / oracle_us if oracle_us > 0 else math.nan
    if ratio > 1.05:
        status = "GOOD"
    elif ratio < 0.95:
        status = "BAD_ORACLE"
    else:
        status = "AT_FLOOR"

    result = {
        "repro_id": REPRO_ID,
        "oracle_us": round(oracle_us, 3),
        "compile_us": round(cd_compile_us, 3),
        "best_compile_us": round(best_compile_us, 3),
        "compile_configs_us": {key: round(value, 3) for key, value in compile_results.items()},
        "ratio": round(ratio, 3),
        "status": status,
        "classification": CLASSIFICATION,
    }
    print(json.dumps(result))
    print(f"oracle full-scope embedding_layernorm: {oracle_us:.3f} us")
    print(f"best required compile config: {best_compile_us:.3f} us")
    return result


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

    inputs = _harness_get_inputs(REPRO_DIR)
    instance = _harness_get_repro_instance(REPRO_DIR)

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
        status = "PASS" if ok else "FAIL"
        print(f"Correctness: {status}")
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
                    print(f"WARNING: oracle is slower than compile "
                          f"for {result['repro_id']} (ratio={result['ratio']:.3f}x)")
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
