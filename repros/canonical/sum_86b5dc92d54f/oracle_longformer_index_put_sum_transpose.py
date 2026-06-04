"""Gap diagnosis (classification: SCATTER_REDUCE): this oracle computes the full Longformer duplicate-index `index_put(accumulate=True)` return tuple by reusing the accumulated one-dimensional scatter buffer for both the `[768]` hidden-dimension sum and the required `[768, 2048]` transposed side-output view, whereas Inductor currently lowers the duplicate-index scatter as a generic atomic `index_put` and then schedules the `as_strided`/`view`/`permute`/`sum` consumers as separate layout and reduction work; Inductor cannot do this today because its scheduler/codegen does not recognize a one-dimensional scatter-add whose non-contiguous view aliases feed both a reduction epilogue and a layout-changing materialized side output; the fix is SCATTER_REDUCE: add an indexed scatter-reduce lowering that maps the captured `as_strided` alias, accumulates the hidden-dimension sum, and emits the transposed side store from the same scatter/load tile."""
from __future__ import annotations

import argparse
import importlib.util
import math
import sys
import time
from pathlib import Path
from typing import Callable

import torch

try:
    import triton
    import triton.language as tl
except ModuleNotFoundError:  # pragma: no cover - keeps py_compile usable without Triton.
    triton = None
    tl = None


REPRO_ID = "sum_86b5dc92d54f"
SHAPE_LABEL = "torchbench_hf_longformer_train_005_d621b103"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

BLOCK = 1024
REDUCE_BLOCK = 2048
HIDDEN = 768
SCATTER_SIZE = 24 * 3 * 512 * 64



def _load_repro_module():
    spec = importlib.util.spec_from_file_location(f"{REPRO_ID}_repro", REPRO_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load repro module from {REPRO_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _config_on_device(config: dict, device: torch.device) -> dict:
    return {
        "inputs": [
            {**spec, "device": str(device)}
            if isinstance(spec, dict) and spec.get("kind") == "tensor"
            else spec
            for spec in config["inputs"]
        ]
    }


def make_inputs(device: torch.device) -> tuple[object, ...]:
    from repro_harness import load_shape_configs, make_inputs_from_config

    configs = load_shape_configs(str(REPRO_PATH))
    if configs:
        inputs = make_inputs_from_config(
            _config_on_device(next(iter(configs.values())), device)
        )
    else:
        module = _load_repro_module()
        inputs = module.make_inputs()

    moved: list[object] = []
    for value in inputs:
        if isinstance(value, torch.Tensor):
            moved.append(value.to(device=device))
        else:
            moved.append(value)
    return tuple(moved)


def get_inputs() -> tuple[object, ...]:
    return make_inputs(torch.device("cuda" if torch.cuda.is_available() else "cpu"))


def get_repro_instance() -> torch.nn.Module:
    return _load_repro_module().Repro().to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))


def oracle_forward(inputs: tuple[object, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    return oracle_longformer_index_put_sum_transpose(*inputs)


def _longformer_update_values(
    bmm_46: torch.Tensor,
    shape_param_0: list[int],
    shape_param_1: list[int],
) -> torch.Tensor:
    chunked = torch.ops.aten.view.default(bmm_46, shape_param_0)
    transposed = torch.ops.aten.permute.default(chunked, [0, 1, 4, 3, 2])
    transposed = torch.ops.aten.permute.default(transposed, [0, 1, 3, 4, 2])
    squeezed = torch.ops.aten.squeeze.dim(transposed, 4)
    contiguous = torch.ops.aten.clone.default(
        squeezed,
        memory_format=torch.contiguous_format,
    )
    return torch.ops.aten.view.default(contiguous, shape_param_1)


if triton is not None:

    @triton.jit
    def _scatter_index_add_kernel(
        bmm_ptr,
        index_ptr,
        scatter_ptr,
        TOTAL: tl.constexpr,
        BLOCK_: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_ + tl.arange(0, BLOCK_)
        mask = offsets < TOTAL

        dim = offsets % 64
        tmp = offsets // 64
        seq = tmp % 512
        tmp = tmp // 512
        chunk = tmp % 3
        block24 = tmp // 3
        bmm_offsets = ((block24 * 3 + chunk) * 64 + dim) * 512 + seq

        values = tl.load(bmm_ptr + bmm_offsets, mask=mask, other=0.0).to(tl.float32)
        indices = tl.load(index_ptr + offsets, mask=mask, other=0).to(tl.int64)
        tl.atomic_add(scatter_ptr + indices, values, sem="relaxed", mask=mask)

    @triton.jit
    def _hidden_sum_kernel(
        scatter_ptr,
        out_ptr,
        BLOCK_: tl.constexpr,
    ):
        hidden = tl.program_id(0)
        rows = tl.arange(0, BLOCK_)
        values = tl.load(scatter_ptr + rows * 768 + hidden).to(tl.float32)
        tl.store(out_ptr + hidden, tl.sum(values, axis=0))


def _oracle_triton(
    bmm_46: torch.Tensor,
    full_15: torch.Tensor,
    view_38: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if triton is None:
        raise RuntimeError("triton is required for the Triton oracle")
    scattered = full_15.clone()
    _scatter_index_add_kernel[(triton.cdiv(SCATTER_SIZE, BLOCK),)](
        bmm_46,
        view_38,
        scattered,
        TOTAL=SCATTER_SIZE,
        BLOCK_=BLOCK,
        num_warps=4,
    )
    reduced = torch.empty((HIDDEN,), device=bmm_46.device, dtype=torch.float32)
    _hidden_sum_kernel[(HIDDEN,)](
        scattered,
        reduced,
        BLOCK_=REDUCE_BLOCK,
        num_warps=8,
    )
    return reduced, scattered.view(2048, HIDDEN).permute(1, 0)


def _oracle_torch(
    bmm_46: torch.Tensor,
    full_15: torch.Tensor,
    view_38: torch.Tensor,
    _shape_param_0,
    _shape_param_1,
    _shape_param_4,
    _shape_param_5,
    _shape_param_6,
) -> tuple[torch.Tensor, torch.Tensor]:
    updates = _longformer_update_values(bmm_46, _shape_param_0, _shape_param_1)
    scattered = torch.ops.aten.clone.default(full_15)
    scattered.index_add_(0, view_38, updates)

    logical = torch.ops.aten.view.default(scattered, _shape_param_4)
    reduced = torch.ops.aten.view.default(
        torch.ops.aten.sum.dim_IntList(logical, [0, 1], True),
        _shape_param_5,
    )
    transposed = torch.ops.aten.permute.default(
        torch.ops.aten.view.default(logical, _shape_param_6),
        [1, 0],
    )
    return reduced, transposed


def oracle_longformer_index_put_sum_transpose(
    bmm_46: torch.Tensor,
    full_15: torch.Tensor,
    view_38: torch.Tensor,
    _shape_param_0,
    _shape_param_1,
    _shape_param_2,
    _shape_param_3,
    _shape_param_4,
    _shape_param_5,
    _shape_param_6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Full-pattern oracle for index_put(accumulate=True) -> sum + transpose."""
    del _shape_param_2, _shape_param_3

    if bmm_46.device.type == "cuda" and triton is not None:
        return _oracle_triton(bmm_46, full_15, view_38)
    return _oracle_torch(
        bmm_46,
        full_15,
        view_38,
        _shape_param_0,
        _shape_param_1,
        _shape_param_4,
        _shape_param_5,
        _shape_param_6,
    )


def reference_outputs(
    inputs: tuple[object, ...],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    module = _load_repro_module()
    if device.type != "cuda":
        module.device = lambda *unused_args, **unused_kwargs: device
    model = module.Repro().to(device)
    return model(*inputs)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _diff_stats(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float, float]:
    diff = (actual.float() - expected.float()).abs()
    rel = diff / (expected.float().abs() + 1e-8)
    return diff.max().item(), diff.mean().item(), rel.max().item()


def run_check(device: torch.device, rtol: float, atol: float) -> bool:
    torch.manual_seed(0)
    inputs = make_inputs(device)
    with torch.no_grad():
        expected = reference_outputs(inputs, device)
        actual = oracle_longformer_index_put_sum_transpose(*inputs)
        synchronize(device)

    if len(actual) != len(expected):
        print(f"tuple_length_match=False actual={len(actual)} expected={len(expected)}")
        return False

    ok = True
    for idx, (got, ref) in enumerate(zip(actual, expected)):
        max_abs, mean_abs, max_rel = _diff_stats(got, ref)
        value_ok = torch.allclose(got.float(), ref.float(), rtol=rtol, atol=atol)
        dtype_ok = got.dtype == ref.dtype
        stride_ok = got.stride() == ref.stride()
        shape_ok = got.shape == ref.shape
        ok = ok and value_ok and dtype_ok and stride_ok and shape_ok
        print(
            f"output[{idx}]: shape={list(got.shape)} dtype={got.dtype} "
            f"stride={got.stride()} max_abs={max_abs:.6e} "
            f"mean_abs={mean_abs:.6e} max_rel={max_rel:.6e} "
            f"allclose={value_ok} dtype_match={dtype_ok} "
            f"stride_match={stride_ok} shape_match={shape_ok}"
        )

    print(f"Correctness: {'PASS' if ok else 'FAIL'}")
    return ok


def benchmark(fn: Callable[[], object], device: torch.device, warmup: int, rep: int) -> float:
    for _ in range(warmup):
        fn()
    synchronize(device)

    if device.type == "cuda":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        best_ms = math.inf
        for _ in range(rep):
            start.record()
            fn()
            end.record()
            torch.cuda.synchronize(device)
            best_ms = min(best_ms, start.elapsed_time(end))
        return best_ms * 1000.0

    best_s = math.inf
    for _ in range(rep):
        start = time.perf_counter()
        fn()
        synchronize(device)
        best_s = min(best_s, time.perf_counter() - start)
    return best_s * 1_000_000.0


def run_bench(device: torch.device, warmup: int, rep: int, compare_eager: bool) -> None:
    torch.manual_seed(0)
    inputs = make_inputs(device)

    with torch.no_grad():
        oracle_longformer_index_put_sum_transpose(*inputs)
        synchronize(device)
        oracle_us = benchmark(
            lambda: oracle_longformer_index_put_sum_transpose(*inputs),
            device,
            warmup,
            rep,
        )
        print(
            f"oracle_longformer_index_put_sum_transpose: {oracle_us:.3f} us "
            f"shape={SHAPE_LABEL} device={device} warmup={warmup} rep={rep}"
        )

        if compare_eager:
            module = _load_repro_module()
            if device.type != "cuda":
                module.device = lambda *unused_args, **unused_kwargs: device
            repro = module.Repro().to(device)
            repro(*inputs)
            synchronize(device)
            repro_us = benchmark(lambda: repro(*inputs), device, warmup, rep)
            print(
                f"repro_eager: {repro_us:.3f} us "
                f"shape={SHAPE_LABEL} device={device} warmup={warmup} rep={rep}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="run correctness check")
    parser.add_argument("--bench", action="store_true", help="run timing benchmark")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--rep", type=int, default=50)
    parser.add_argument(
        "--compare-eager",
        action="store_true",
        help="include eager repro timing in --bench output",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.check and not args.bench:
        args.check = True

    device = torch.device(args.device)
    if args.check and not run_check(device=device, rtol=args.rtol, atol=args.atol):
        sys.exit(1)
    if args.bench:
        run_bench(
            device=device,
            warmup=args.warmup,
            rep=args.rep,
            compare_eager=args.compare_eager,
        )


if __name__ == "__main__":
    main()
