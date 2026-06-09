"""
Full-scope oracle for amax_sum_sum_098a3fee06f9.

Gap diagnosis (classification: SCATTER_REDUCE): this oracle fuses the
two-column softmax-backward-like update, the zero materialization, the
duplicate-preserving index_put accumulate, and the final `[8192, 2] -> [2,
8192]` transposed layout into one Triton program that writes the returned
strided tensor directly. Inductor currently lowers the decomposed graph as a
small reduction/update followed by a separate zero fill, scatter/add, view, and
permute pipeline; the missed opportunity is recognizing that this tiny
scatter-reduce epilogue can materialize the final transposed output directly
while preserving atomic add semantics for duplicate coordinates.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
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

N_ROWS = 8
N_CHANNELS = 2
SCATTER_DIM = 1024
OUT_POSITIONS = N_ROWS * SCATTER_DIM
OUT_STORAGE_SIZE = OUT_POSITIONS * N_CHANNELS

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


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _softmax_scatter_transpose_kernel(
        arg293_ptr,
        arg230_ptr,
        arg232_ptr,
        arg231_ptr,
        arg229_ptr,
        arg294_ptr,
        arg78_ptr,
        arg228_ptr,
        out_ptr,
        block_out: tl.constexpr,
        n_rows: tl.constexpr,
        scatter_dim: tl.constexpr,
    ):
        offsets = tl.arange(0, block_out)
        tl.store(out_ptr + offsets, tl.zeros((block_out,), tl.float32))

        rows = tl.arange(0, n_rows)
        div = tl.load(arg293_ptr).to(tl.float32) / tl.load(arg230_ptr).to(tl.float32)

        pred_row = tl.load(arg231_ptr + rows).to(tl.int1)
        mask0 = tl.load(arg232_ptr + rows * 2).to(tl.int1)
        mask1 = tl.load(arg232_ptr + rows * 2 + 1).to(tl.int1)
        scale = tl.where(pred_row, div, 0.0)
        grad0 = tl.where(mask0, -scale, 0.0)
        grad1 = tl.where(mask1, -scale, 0.0)
        grad_sum = grad0 + grad1

        x0 = tl.load(arg229_ptr + rows * 2).to(tl.float32)
        x1 = tl.load(arg229_ptr + rows * 2 + 1).to(tl.float32)
        row_max = tl.maximum(x0, x1)
        e0 = tl.exp(x0 - row_max)
        e1 = tl.exp(x1 - row_max)
        denom = e0 + e1
        prob0 = e0 / denom
        prob1 = e1 / denom

        add0 = tl.load(arg294_ptr + rows * 2).to(tl.float32) + grad0 - prob0 * grad_sum
        add1 = tl.load(arg294_ptr + rows * 2 + 1).to(tl.float32) + grad1 - prob1 * grad_sum

        idx0 = tl.load(arg78_ptr + rows).to(tl.int64)
        idx1 = tl.load(arg228_ptr + rows).to(tl.int64)
        out_position = idx0 * scatter_dim + idx1

        tl.atomic_add(out_ptr + out_position * 2, add0, sem="relaxed")
        tl.atomic_add(out_ptr + out_position * 2 + 1, add1, sem="relaxed")


def _require_triton_cuda() -> None:
    if triton is None:
        raise RuntimeError("Triton is required for this oracle")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this Triton oracle")


def _launch_oracle(
    arg293: torch.Tensor,
    arg230: torch.Tensor,
    arg232: torch.Tensor,
    arg231: torch.Tensor,
    arg229: torch.Tensor,
    arg294: torch.Tensor,
    arg78: torch.Tensor,
    arg228: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    _require_triton_cuda()
    if arg293.shape != () or arg230.shape != ():
        raise ValueError("expected two scalar f32 inputs")
    if arg232.shape != (N_ROWS, N_CHANNELS) or arg232.dtype != torch.bool:
        raise ValueError(f"expected bool mask shape {(N_ROWS, N_CHANNELS)}")
    if arg231.shape != (N_ROWS, 1) or arg231.dtype != torch.bool:
        raise ValueError(f"expected bool row mask shape {(N_ROWS, 1)}")
    if arg229.shape != (N_ROWS, N_CHANNELS) or arg294.shape != (N_ROWS, N_CHANNELS):
        raise ValueError(f"expected f32 update tensors shape {(N_ROWS, N_CHANNELS)}")
    if arg78.shape != (N_ROWS,) or arg228.shape != (N_ROWS,):
        raise ValueError(f"expected int64 index tensors shape {(N_ROWS,)}")
    if out.shape != (N_CHANNELS, OUT_POSITIONS) or out.stride() != (1, N_CHANNELS):
        raise ValueError("expected output shape [2,8192] with transposed stride (1,2)")

    _softmax_scatter_transpose_kernel[(1,)](
        arg293,
        arg230,
        arg232,
        arg231,
        arg229,
        arg294,
        arg78,
        arg228,
        out,
        block_out=OUT_STORAGE_SIZE,
        n_rows=N_ROWS,
        scatter_dim=SCATTER_DIM,
        num_warps=8,
    )
    return out


@oracle_impl(hardware="H100", shapes="(T([], f32), T([], f32), T([8, 2], b8), T([8, 1], b8), T([8, 2], f32), T([8, 2], f32), T([8], i64, gen=Index(8)), T([8], i64, gen=Index(8)), S([8192, 2]))")
def oracle_forward(inputs) -> torch.Tensor:
    """Run the full returned-output oracle computation."""
    (
        arg293,
        arg230,
        arg232,
        arg231,
        arg229,
        arg294,
        arg78,
        arg228,
        shape_param,
    ) = inputs
    if tuple(shape_param) != (OUT_POSITIONS, N_CHANNELS):
        raise ValueError(f"unexpected view shape parameter: {shape_param}")

    out = torch.empty_strided(
        (N_CHANNELS, OUT_POSITIONS),
        (1, N_CHANNELS),
        device=arg229.device,
        dtype=torch.float32,
    )
    return _launch_oracle(arg293, arg230, arg232, arg231, arg229, arg294, arg78, arg228, out)


def _bench_cuda(fn, warmup: int, rep: int) -> float:
    from triton.testing import do_bench

    return do_bench(fn, warmup=warmup, rep=rep, return_mode="min") * 1000.0


def _compile_with_config(instance, inputs, config: dict[str, object], warmup: int):
    import torch._dynamo
    import torch._inductor.config as inductor_config

    torch._dynamo.reset()
    model = instance.__class__().cuda()
    with inductor_config.patch(config):
        compiled = torch.compile(model)
        for _ in range(max(1, warmup)):
            compiled(*inputs)
        torch.cuda.synchronize()
    return compiled


def run_check(args: argparse.Namespace) -> bool:
    inputs = get_inputs()
    instance = get_repro_instance()
    ok = check_oracle(
        lambda values: oracle_forward(values),
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

    with torch.no_grad():
        oracle_forward(inputs)
        torch.cuda.synchronize()
        oracle_us = _bench_cuda(lambda: oracle_forward(inputs), warmup=args.warmup, rep=args.rep)

    compile_results: dict[str, float] = {}
    with torch.no_grad():
        for label, config in COMPILE_CONFIGS:
            compiled = _compile_with_config(instance, inputs, config, warmup=args.warmup)
            compile_results[label] = _bench_cuda(lambda: compiled(*inputs), warmup=args.warmup, rep=args.rep)
            del compiled
            torch.cuda.empty_cache()

    best_required_compile_us = min(compile_results.values())
    true_floor = oracle_us < best_required_compile_us
    result = {
        "repro_id": REPRO_ID,
        "oracle_us": round(oracle_us, 3),
        "compile_us": round(compile_results["coordinate_descent_tuning"], 3),
        "combo_compile_us": round(compile_results["combo_looped_cd"], 3),
        "best_required_compile_us": round(best_required_compile_us, 3),
        "ratio": round(best_required_compile_us / oracle_us, 3),
        "true_floor": true_floor,
        "status": "GOOD" if true_floor else "DIAGNOSIS_ONLY",
        "classification": "SCATTER_REDUCE",
    }
    print(json.dumps(result))
    if not true_floor:
        print("WARNING: oracle is slower than a required compile baseline")
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
