"""Gap diagnosis (classification: SCATTER_REDUCE): this oracle treats the Qwen MoE grouped-mm row update as an indexed row scatter-add into the `[2048, 2048]` hidden-state matrix and derives the RMSNorm weight-gradient reduction plus transposed input-gradient side output from that single accumulated matrix, whereas Inductor currently materializes the `where` source, lowers `index_put(accumulate=True)` into a generic scatter buffer, then schedules the RMSNorm reduction and transpose-producing pointwise consumer as separate work; Inductor cannot do this today because its scheduler/codegen does not model duplicate-index row scatter updates as a structured scatter-reduce producer with sibling reduction and materialized-layout epilogues; the fix is SCATTER_REDUCE: add an indexed row-scatter/RMSNorm-backward lowering that accumulates duplicate expert rows once and fuses the downstream per-hidden reduction with the transposed gradient store."""
from __future__ import annotations

import argparse
import importlib.util
import math
import sys
import time
from pathlib import Path
from typing import Callable

import torch


REPRO_ID = "sum_sum_36f20fb8bfa8"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"
SHAPE_LABEL = "vllm_qwen_qwen3-30b-a3b_001_3b4ca287"

BATCH = 4
SEQ_LEN = 512
HIDDEN = 2048

sys.path.insert(0, str(REPO_ROOT))


def _load_repro_module():
    spec = importlib.util.spec_from_file_location(f"{REPRO_ID}_repro", REPRO_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load repro module from {REPRO_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_inputs(device: torch.device) -> tuple[object, ...]:
    from repro_harness import load_shape_configs, make_inputs_from_config

    configs = load_shape_configs(str(REPRO_PATH))
    if configs:
        config = next(iter(configs.values()))
        config = {
            "inputs": [
                {**spec, "device": str(device)}
                if isinstance(spec, dict) and spec.get("kind") == "tensor"
                else spec
                for spec in config["inputs"]
            ]
        }
        inputs = make_inputs_from_config(config)
    else:
        module = _load_repro_module()
        inputs = module.make_inputs()

    return tuple(
        value.to(device=device) if isinstance(value, torch.Tensor) else value
        for value in inputs
    )


def structured_scatter_rmsnorm_reduce(
    arg171_1: torch.Tensor,
    full_3: torch.Tensor,
    _grouped_mm_3: torch.Tensor,
    arg169_1: torch.Tensor,
    mm_3: torch.Tensor,
    arg42_1: torch.Tensor,
    arg159_1: torch.Tensor,
    arg160_1: torch.Tensor,
    convert_element_type_5: torch.Tensor,
    _shape_param_0,
    _shape_param_1,
    _shape_param_2,
    _shape_param_3,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Source-structured version of the captured indexed scatter plus RMSNorm bwd."""
    scatter_source = torch.ops.aten.where.self(arg171_1, full_3, _grouped_mm_3)
    scatter_base = torch.ops.aten.full.default(
        [HIDDEN, HIDDEN],
        0,
        dtype=torch.bfloat16,
        layout=torch.strided,
        device=mm_3.device,
        pin_memory=False,
    )
    scatter_accum = torch.ops.aten.index_put.default(
        scatter_base,
        [arg169_1],
        scatter_source,
        True,
    )
    hidden_matrix = torch.ops.aten.add.Tensor(scatter_accum, mm_3)
    hidden_rows = torch.ops.aten.view.default(hidden_matrix, _shape_param_0)

    x_f32 = torch.ops.prims.convert_element_type.default(arg159_1, torch.float32)
    xhat_bf16 = torch.ops.prims.convert_element_type.default(
        torch.ops.aten.mul.Tensor(x_f32, arg160_1),
        torch.bfloat16,
    )

    weight_grad = torch.ops.aten.view.default(
        torch.ops.aten.sum.dim_IntList(
            torch.ops.aten.mul.Tensor(hidden_rows, xhat_bf16),
            [0, 1],
            True,
        ),
        _shape_param_1,
    )

    dy_weighted_bf16 = torch.ops.aten.mul.Tensor(hidden_rows, arg42_1)
    dy_weighted = torch.ops.prims.convert_element_type.default(
        dy_weighted_bf16,
        torch.float32,
    )
    row_dot = torch.ops.aten.sum.dim_IntList(
        torch.ops.aten.mul.Tensor(dy_weighted, x_f32),
        [2],
        True,
    )
    correction_scale = torch.ops.aten.mul.Tensor(
        torch.ops.aten.mul.Scalar(row_dot, -0.5),
        torch.ops.aten.pow.Tensor_Scalar(arg160_1, 3),
    )
    correction = torch.ops.aten.mul.Tensor(
        torch.ops.aten.div.Scalar(
            torch.ops.aten.expand.default(correction_scale, _shape_param_2),
            HIDDEN,
        ),
        torch.ops.aten.mul.Scalar(
            torch.ops.aten.pow.Tensor_Scalar(x_f32, 1.0),
            2.0,
        ),
    )
    input_grad = torch.ops.aten.add.Tensor(
        torch.ops.aten.mul.Tensor(dy_weighted, arg160_1),
        correction,
    )
    input_grad_bf16 = torch.ops.prims.convert_element_type.default(
        input_grad,
        torch.bfloat16,
    )
    transposed_side = torch.ops.aten.permute.default(
        torch.ops.aten.view.default(
            torch.ops.aten.add.Tensor(convert_element_type_5, input_grad_bf16),
            _shape_param_3,
        ),
        [1, 0],
    )
    return weight_grad, transposed_side


def reference_outputs(
    inputs: tuple[object, ...],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    module = _load_repro_module()
    module.device = lambda *unused_args, **unused_kwargs: device
    model = module.Repro().to(device=device)
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
        actual = structured_scatter_rmsnorm_reduce(*inputs)
        synchronize(device)

    ok = True
    for idx, (got, ref) in enumerate(zip(actual, expected)):
        max_abs, mean_abs, max_rel = _diff_stats(got, ref)
        value_ok = torch.allclose(got.float(), ref.float(), rtol=rtol, atol=atol)
        dtype_ok = got.dtype == ref.dtype
        stride_ok = got.stride() == ref.stride()
        ok = ok and value_ok and dtype_ok and stride_ok
        print(
            f"output[{idx}]: shape={list(got.shape)} dtype={got.dtype} "
            f"stride={got.stride()} max_abs={max_abs:.6g} "
            f"mean_abs={mean_abs:.6g} max_rel={max_rel:.6g} "
            f"allclose={value_ok} dtype_match={dtype_ok} stride_match={stride_ok}"
        )

    print(f"Correctness: {'PASS' if ok else 'FAIL'}")
    return ok


def benchmark(fn: Callable[[], object], device: torch.device, warmup: int, rep: int) -> float:
    for _ in range(warmup):
        fn()
    synchronize(device)

    best_s = math.inf
    for _ in range(rep):
        start = time.perf_counter()
        fn()
        synchronize(device)
        best_s = min(best_s, time.perf_counter() - start)
    return best_s * 1_000_000.0


def run_bench(
    device: torch.device,
    warmup: int,
    rep: int,
    include_compile: bool,
) -> None:
    torch.manual_seed(0)
    inputs = make_inputs(device)
    module = _load_repro_module()
    module.device = lambda *unused_args, **unused_kwargs: device
    repro = module.Repro().to(device=device)

    with torch.no_grad():
        oracle_us = benchmark(
            lambda: structured_scatter_rmsnorm_reduce(*inputs),
            device,
            warmup,
            rep,
        )
        repro_eager_us = benchmark(lambda: repro(*inputs), device, warmup, rep)

    print(
        f"oracle_eager_us={oracle_us:.3f} shape={SHAPE_LABEL} "
        f"device={device} warmup={warmup} rep={rep}"
    )
    print(
        f"repro_eager_us={repro_eager_us:.3f} shape={SHAPE_LABEL} "
        f"device={device} warmup={warmup} rep={rep}"
    )

    if not include_compile:
        return

    compiled_repro = torch.compile(repro)
    compiled_oracle = torch.compile(structured_scatter_rmsnorm_reduce)
    with torch.no_grad():
        compiled_repro(*inputs)
        compiled_oracle(*inputs)
        synchronize(device)
        compiled_repro_us = benchmark(lambda: compiled_repro(*inputs), device, warmup, rep)
        compiled_oracle_us = benchmark(lambda: compiled_oracle(*inputs), device, warmup, rep)

    print(
        f"repro_compiled_us={compiled_repro_us:.3f} shape={SHAPE_LABEL} "
        f"device={device} warmup={warmup} rep={rep}"
    )
    print(
        f"oracle_compiled_us={compiled_oracle_us:.3f} shape={SHAPE_LABEL} "
        f"device={device} warmup={warmup} rep={rep}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="run correctness check")
    parser.add_argument("--bench", action="store_true", help="run timing benchmark")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--rtol", type=float, default=0.0)
    parser.add_argument("--atol", type=float, default=0.0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--rep", type=int, default=20)
    parser.add_argument(
        "--include-compile",
        action="store_true",
        help="also benchmark torch.compile variants",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.check and not args.bench:
        args.check = True
        args.bench = True

    device = torch.device(args.device)
    if args.check and not run_check(device=device, rtol=args.rtol, atol=args.atol):
        sys.exit(1)
    if args.bench:
        run_bench(
            device=device,
            warmup=args.warmup,
            rep=args.rep,
            include_compile=args.include_compile,
        )


if __name__ == "__main__":
    main()
