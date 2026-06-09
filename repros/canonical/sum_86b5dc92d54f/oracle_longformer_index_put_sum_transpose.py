"""Gap diagnosis (classification: SCATTER_REDUCE): this oracle computes the full Longformer duplicate-index `index_put(accumulate=True)` return tuple by reusing the accumulated one-dimensional scatter buffer for both the `[768]` hidden-dimension sum and the required `[768, 2048]` transposed side-output view, whereas Inductor currently lowers the duplicate-index scatter as a generic atomic `index_put` and then schedules the `as_strided`/`view`/`permute`/`sum` consumers as separate layout and reduction work; Inductor cannot do this today because its scheduler/codegen does not recognize a one-dimensional scatter-add whose non-contiguous view aliases feed both a reduction epilogue and a layout-changing materialized side output; the fix is SCATTER_REDUCE: add an indexed scatter-reduce lowering that maps the captured `as_strided` alias, accumulates the hidden-dimension sum, and emits the transposed side store from the same scatter/load tile."""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch

try:
    import triton
    import triton.language as tl
except ModuleNotFoundError:  # pragma: no cover - keeps py_compile usable without Triton.
    triton = None
    tl = None


REPRO_ID = "sum_86b5dc92d54f"

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


@oracle_impl(hardware="H100", shapes="(T([72, 64, 512], f32), T([1572864], f32), T([2359296], i64, gen=Index(1572864)), S([24, 3, 64, 512, 1]), S([2359296]), S([24, 1024, 64]), S([2, 12, 1024, 64]), S([1024, 2, 768]), S([768]), S([2048, 768]))")
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
