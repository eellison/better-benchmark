"""
Full-scope Triton diagnostic artifact for sum_sum_cc11724ac17a.

Gap diagnosis (classification: BANDWIDTH_BOUND): this artifact consumes the
same five tensor inputs plus two shape parameters as repro.py, computes the
`(mul_296 + view(mm_126)) * arg9_1` producer, accumulates both hidden-dimension
row sums, applies the dependent layer-norm-backward epilogue, and writes the
returned `float32[4096, 4096]` transpose-view output with stride `(1, 4096)` in
one Triton row kernel. It differs from Inductor only by using a hand-written
one-row, 4096-column Triton schedule; Inductor already emits a single fused
`add/mul/sub/sum/view` reduction kernel for this full scope, and the historical
best compiled timing is faster than this artifact. There is no scheduler-fusion
or algebraic Inductor change to claim here; any remaining improvement would be a
low-level tiling/autotuning constant-factor win for an already bandwidth-heavy
fused kernel.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl

from oracle_harness import (
    bench_oracle,
    bench_oracle_all_shapes,
    check_oracle,
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    has_stochastic_ops,
)


REPRO_ID = "sum_sum_cc11724ac17a"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"

ROWS = 8 * 512
HIDDEN = 4096
HISTORICAL_BEST_COMPILE_US = 53.05600166320801

COMPILE_CONFIGS = [
    ("coordinate_descent_tuning=True", {"coordinate_descent_tuning": True}),
    (
        (
            "combo_kernels=True,combo_kernel_per_subkernel_blocks=True,"
            "coordinate_descent_tuning=True,benchmark_combo_kernel=True,"
            "triton.multi_kernel=3"
        ),
        {
            "combo_kernels": True,
            "combo_kernel_per_subkernel_blocks": True,
            "coordinate_descent_tuning": True,
            "benchmark_combo_kernel": True,
            "triton.multi_kernel": 3,
        },
    ),
]

sys.path.insert(0, str(REPO_ROOT))


@triton.jit
def _layernorm_backward_transpose_kernel(
    mm_ptr,
    mul_ptr,
    weight_ptr,
    rhs_ptr,
    gate_ptr,
    out_ptr,
    HIDDEN_: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    offsets = row * HIDDEN_ + cols

    mm = tl.load(mm_ptr + offsets).to(tl.float32)
    mul = tl.load(mul_ptr + offsets).to(tl.float32)
    weight = tl.load(weight_ptr + cols).to(tl.float32)
    rhs = tl.load(rhs_ptr + offsets).to(tl.float32)

    producer = (mul + mm) * weight
    sum_producer = tl.sum(producer, axis=0)
    sum_producer_rhs = tl.sum(producer * rhs, axis=0)
    gate = tl.load(gate_ptr + row).to(tl.float32)
    out = gate * (producer * HIDDEN_ - sum_producer - rhs * sum_producer_rhs)

    # The returned eager tensor is a permute view with stride (1, 4096).  Writing
    # row-major backing storage here makes logical output[col, row] equal out.
    tl.store(out_ptr + offsets, out)


def _load_repro_module():
    spec = importlib.util.spec_from_file_location(f"{REPRO_ID}_repro", REPRO_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load repro module from {REPRO_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def get_inputs() -> tuple[object, ...]:
    module = _load_repro_module()
    return tuple(x.cuda() if isinstance(x, torch.Tensor) else x for x in module.make_inputs())


def get_repro_instance() -> torch.nn.Module:
    module = _load_repro_module()
    return module.Repro().cuda()


def _validate_inputs(inputs: tuple[object, ...]) -> None:
    (
        mm_126,
        mul_296,
        arg9_1,
        arg30_1,
        arg175_1,
        shape0,
        shape1,
    ) = inputs
    if tuple(shape0) != (8, 512, HIDDEN):
        raise ValueError(f"unexpected first view shape parameter: {shape0}")
    if tuple(shape1) != (ROWS, HIDDEN):
        raise ValueError(f"unexpected second view shape parameter: {shape1}")
    if mm_126.shape != (ROWS, HIDDEN):
        raise ValueError(f"unexpected mm_126 shape: {tuple(mm_126.shape)}")
    if mul_296.shape != (8, 512, HIDDEN):
        raise ValueError(f"unexpected mul_296 shape: {tuple(mul_296.shape)}")
    if arg9_1.shape != (HIDDEN,):
        raise ValueError(f"unexpected arg9_1 shape: {tuple(arg9_1.shape)}")
    if arg30_1.shape != (8, 512, HIDDEN):
        raise ValueError(f"unexpected arg30_1 shape: {tuple(arg30_1.shape)}")
    if arg175_1.shape != (8, 512, 1):
        raise ValueError(f"unexpected arg175_1 shape: {tuple(arg175_1.shape)}")

    tensors = (mm_126, mul_296, arg9_1, arg30_1, arg175_1)
    if any(t.device.type != "cuda" for t in tensors):
        raise RuntimeError("Triton oracle requires CUDA inputs")
    if any(t.dtype != torch.float32 for t in tensors):
        raise ValueError("oracle expects all tensor inputs to be float32")
    if not mm_126.is_contiguous():
        raise ValueError("oracle expects contiguous mm_126")
    if not mul_296.is_contiguous():
        raise ValueError("oracle expects contiguous mul_296")
    if not arg9_1.is_contiguous():
        raise ValueError("oracle expects contiguous arg9_1")
    if not arg30_1.is_contiguous():
        raise ValueError("oracle expects contiguous arg30_1")
    if not arg175_1.is_contiguous():
        raise ValueError("oracle expects contiguous arg175_1")


def oracle_fused(
    mm_126: torch.Tensor,
    mul_296: torch.Tensor,
    arg9_1: torch.Tensor,
    arg30_1: torch.Tensor,
    arg175_1: torch.Tensor,
    shape0,
    shape1,
) -> torch.Tensor:
    inputs = (mm_126, mul_296, arg9_1, arg30_1, arg175_1, shape0, shape1)
    _validate_inputs(inputs)

    out = torch.empty_strided(
        (HIDDEN, ROWS),
        (1, HIDDEN),
        device=mm_126.device,
        dtype=torch.float32,
    )
    _layernorm_backward_transpose_kernel[(ROWS,)](
        mm_126,
        mul_296,
        arg9_1,
        arg30_1,
        arg175_1,
        out,
        HIDDEN_=HIDDEN,
        BLOCK_N=HIDDEN,
        num_warps=8,
    )
    return out


def oracle_forward(inputs: tuple[object, ...]) -> torch.Tensor:
    """Run the full-scope oracle with the same input tuple as Repro.forward."""
    return oracle_fused(*inputs)


def reference_output(inputs: tuple[object, ...]) -> torch.Tensor:
    model = get_repro_instance()
    with torch.no_grad():
        return model(*inputs)


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
