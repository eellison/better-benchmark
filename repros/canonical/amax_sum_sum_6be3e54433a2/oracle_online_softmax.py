"""Gap diagnosis (classification: SCATTER_REDUCE): this oracle computes the complete GPT-Neo two-class softmax-backward-like update, atomically accumulates the `[32,2]` values into the indexed zero `[32,128,2]` scatter-add destination, and writes the viewed/permuted/padded `[4,4096]` return directly, whereas Inductor currently lowers the row reductions, `index_put(accumulate=True)`, view/permute, and pad as separate generic kernels over materialized intermediates; Inductor cannot do this today because scheduler/codegen does not recognize this tiny indexed scatter-add as the producer of a final layout-changing padded side output; the fix is to add a lowering that fuses the row-local softmax-backward arithmetic with duplicate-safe indexed scatter accumulation into the final returned layout."""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - keeps py_compile useful without Triton.
    triton = None
    tl = None


REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"

sys.path.insert(0, str(REPO_ROOT))

from oracle_harness import (
    bench_oracle,
    bench_oracle_all_shapes,
    check_oracle,
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    has_stochastic_ops,
)

ROWS = 32
CLASSES = 2
SCATTER_M = 32
SCATTER_N = 128
OUT_ROWS = 4
OUT_COLS = SCATTER_M * SCATTER_N
OUT_NUMEL = OUT_ROWS * OUT_COLS

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


if triton is not None:

    @triton.jit
    def _zero_kernel(out_ptr, n_elements: tl.constexpr, BLOCK: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        tl.store(out_ptr + offsets, tl.zeros((BLOCK,), tl.float32), mask=offsets < n_elements)

    @triton.jit
    def _scatter_softmax_backward_kernel(
        scale_num_ptr,
        scale_den_ptr,
        class_mask_ptr,
        row_mask_ptr,
        logits_ptr,
        residual_ptr,
        idx_m_ptr,
        idx_n_ptr,
        out_ptr,
        BLOCK_ROWS: tl.constexpr,
        ROWS_: tl.constexpr,
        CLASSES_: tl.constexpr,
        SCATTER_M_: tl.constexpr,
        SCATTER_N_: tl.constexpr,
        OUT_COLS_: tl.constexpr,
    ):
        rows = tl.arange(0, BLOCK_ROWS)
        active = rows < ROWS_

        scale = tl.load(scale_num_ptr).to(tl.float32) / tl.load(scale_den_ptr).to(tl.float32)
        row_enabled = tl.load(row_mask_ptr + rows, mask=active, other=0).to(tl.int1)

        mask0 = tl.load(class_mask_ptr + rows * CLASSES_ + 0, mask=active, other=0).to(tl.int1)
        mask1 = tl.load(class_mask_ptr + rows * CLASSES_ + 1, mask=active, other=0).to(tl.int1)
        grad0 = tl.where(mask0 & row_enabled, -scale, 0.0)
        grad1 = tl.where(mask1 & row_enabled, -scale, 0.0)
        grad_sum = grad0 + grad1

        x0 = tl.load(logits_ptr + rows * CLASSES_ + 0, mask=active, other=-float("inf")).to(tl.float32)
        x1 = tl.load(logits_ptr + rows * CLASSES_ + 1, mask=active, other=-float("inf")).to(tl.float32)
        row_max = tl.maximum(x0, x1)
        exp0 = tl.exp(x0 - row_max)
        exp1 = tl.exp(x1 - row_max)
        denom = exp0 + exp1
        softmax0 = exp0 / denom
        softmax1 = exp1 / denom

        update0 = tl.load(residual_ptr + rows * CLASSES_ + 0, mask=active, other=0.0).to(tl.float32)
        update1 = tl.load(residual_ptr + rows * CLASSES_ + 1, mask=active, other=0.0).to(tl.float32)
        update0 += grad0 - softmax0 * grad_sum
        update1 += grad1 - softmax1 * grad_sum

        dst_m = tl.load(idx_m_ptr + rows, mask=active, other=0).to(tl.int64)
        dst_n = tl.load(idx_n_ptr + rows, mask=active, other=0).to(tl.int64)
        valid_dst = active & (dst_m >= 0) & (dst_m < SCATTER_M_) & (dst_n >= 0) & (dst_n < SCATTER_N_)
        dst = dst_m * SCATTER_N_ + dst_n

        tl.atomic_add(out_ptr + dst, update0, sem="relaxed", mask=valid_dst)
        tl.atomic_add(out_ptr + OUT_COLS_ + dst, update1, sem="relaxed", mask=valid_dst)


def _load_repro_module():
    spec = importlib.util.spec_from_file_location(f"{REPRO_ID}_repro", REPRO_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load repro module from {REPRO_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def get_inputs():
    return _harness_get_inputs(REPRO_DIR)


def _require_triton_cuda() -> None:
    if triton is None:
        raise RuntimeError("Triton is required for this oracle")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this Triton oracle")


def _validate_inputs(inputs: tuple[object, ...] | list[object]) -> tuple[torch.Tensor, ...]:
    if len(inputs) != 9:
        raise ValueError(f"expected 9 repro inputs, got {len(inputs)}")
    scale_num, scale_den, class_mask, row_mask, logits, residual, idx_m, idx_n, shape_param = inputs
    del shape_param
    tensors = (scale_num, scale_den, class_mask, row_mask, logits, residual, idx_m, idx_n)
    if not all(isinstance(value, torch.Tensor) for value in tensors):
        raise TypeError("oracle expects the tensor inputs produced by make_inputs()")
    if logits.device.type != "cuda":
        raise RuntimeError("Triton oracle requires CUDA inputs")
    if scale_num.shape != () or scale_den.shape != ():
        raise ValueError("scale inputs must be scalar tensors")
    if class_mask.shape != (ROWS, CLASSES) or class_mask.dtype is not torch.bool:
        raise ValueError(f"unexpected class mask: shape={tuple(class_mask.shape)} dtype={class_mask.dtype}")
    if row_mask.shape != (ROWS, 1) or row_mask.dtype is not torch.bool:
        raise ValueError(f"unexpected row mask: shape={tuple(row_mask.shape)} dtype={row_mask.dtype}")
    if logits.shape != (ROWS, CLASSES) or logits.dtype is not torch.float32:
        raise ValueError(f"unexpected logits: shape={tuple(logits.shape)} dtype={logits.dtype}")
    if residual.shape != (ROWS, CLASSES) or residual.dtype is not torch.float32:
        raise ValueError(f"unexpected residual: shape={tuple(residual.shape)} dtype={residual.dtype}")
    if idx_m.shape != (ROWS,) or idx_m.dtype is not torch.int64:
        raise ValueError(f"unexpected first index: shape={tuple(idx_m.shape)} dtype={idx_m.dtype}")
    if idx_n.shape != (ROWS,) or idx_n.dtype is not torch.int64:
        raise ValueError(f"unexpected second index: shape={tuple(idx_n.shape)} dtype={idx_n.dtype}")
    return tensors


def _launch_oracle(
    scale_num: torch.Tensor,
    scale_den: torch.Tensor,
    class_mask: torch.Tensor,
    row_mask: torch.Tensor,
    logits: torch.Tensor,
    residual: torch.Tensor,
    idx_m: torch.Tensor,
    idx_n: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    _require_triton_cuda()
    if out.shape != (OUT_ROWS, OUT_COLS) or out.dtype is not torch.float32:
        raise ValueError(f"out must be f32[{OUT_ROWS}, {OUT_COLS}], got shape={tuple(out.shape)} dtype={out.dtype}")

    zero_block = 256
    _zero_kernel[(triton.cdiv(OUT_NUMEL, zero_block),)](
        out,
        n_elements=OUT_NUMEL,
        BLOCK=zero_block,
        num_warps=4,
    )
    _scatter_softmax_backward_kernel[(1,)](
        scale_num,
        scale_den,
        class_mask,
        row_mask,
        logits,
        residual,
        idx_m,
        idx_n,
        out,
        BLOCK_ROWS=triton.next_power_of_2(ROWS),
        ROWS_=ROWS,
        CLASSES_=CLASSES,
        SCATTER_M_=SCATTER_M,
        SCATTER_N_=SCATTER_N,
        OUT_COLS_=OUT_COLS,
        num_warps=1,
    )
    return out


def oracle_forward(inputs: tuple[object, ...] | list[object]) -> torch.Tensor:
    tensors = _validate_inputs(inputs)
    out = torch.empty((OUT_ROWS, OUT_COLS), device=tensors[0].device, dtype=torch.float32)
    return _launch_oracle(*tensors, out)


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
    instance = _harness_get_repro_instance(REPRO_DIR)
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
    instance = _harness_get_repro_instance(REPRO_DIR)
    tensors = _validate_inputs(inputs)
    out = torch.empty((OUT_ROWS, OUT_COLS), device=tensors[0].device, dtype=torch.float32)

    logical_bytes = (
        2 * 4
        + ROWS * CLASSES
        + ROWS
        + ROWS * CLASSES * 4
        + ROWS * CLASSES * 4
        + ROWS * 8 * 2
        + OUT_NUMEL * 4
        + ROWS * CLASSES * 4
    )

    with torch.no_grad():
        _launch_oracle(*tensors, out)
        torch.cuda.synchronize()
        oracle_us = _bench_cuda(lambda: _launch_oracle(*tensors, out), warmup=args.warmup, rep=args.rep)

    compile_results: dict[str, float] = {}
    with torch.no_grad():
        for label, config in COMPILE_CONFIGS:
            compiled = _compile_with_config(instance, inputs, config, warmup=args.warmup)
            us = _bench_cuda(lambda: compiled(*inputs), warmup=args.warmup, rep=args.rep)
            compile_results[label] = us
            del compiled
            torch.cuda.empty_cache()

    best_required_compile_us = min(compile_results.values())
    true_floor = oracle_us < best_required_compile_us
    result = {
        "repro_id": REPRO_ID,
        "oracle_us": round(oracle_us, 3),
        "oracle_bw_tb_s": round(logical_bytes / (oracle_us * 1e-6) / 1e12, 3),
        "compile_us": round(compile_results["coordinate_descent_tuning"], 3),
        "combo_compile_us": round(compile_results["combo_looped_cd"], 3),
        "best_required_compile_us": round(best_required_compile_us, 3),
        "ratio": round(best_required_compile_us / oracle_us, 3),
        "true_floor": true_floor,
        "status": "GOOD" if true_floor else "DIAGNOSIS_ONLY",
        "classification": "SCATTER_REDUCE",
    }
    print(json.dumps(result))
    print(f"oracle shape: update=f32[{ROWS}, {CLASSES}], indexed output=f32[{OUT_ROWS}, {OUT_COLS}]")
    print(f"direct logical traffic: {logical_bytes / 1e6:.3f} MB")
    if not true_floor:
        print("diagnosis_only: not_true_floor because a required compile config is faster than the oracle")
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
