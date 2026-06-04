"""
Gap diagnosis: this oracle computes the row-wise attention-gradient ``fma`` once, writes the required full ``[8192, 49, 49]`` output, and in the same Triton pass accumulates per-block partial sums into the relative-position bias table with indexed atomics; Inductor currently emits one kernel to compute the row sums plus materialized ``fma`` output, a copy kernel for the bias-table base, and a second reduction kernel that rereads the inputs and recomputes ``fma`` before the indexed ``index_put(accumulate=True)``. Inductor cannot express this today because the profitable fusion crosses two different reductions and a materialized sibling output: the first reduction is per attention row over keys, while the second is a batch reduction followed by duplicate-index accumulation into ``[169, 16]``. Classification: NEW_PATTERN; the fix would be a dedicated fused materialized-producer plus indexed partial-reduction template for relative-position-bias style scatter-reduces.
"""
from __future__ import annotations

import argparse
import importlib.util
import math
import sys
from pathlib import Path
from typing import Callable

import torch
import torch._inductor.config as inductor_config
import triton
import triton.language as tl


REPRO_ID = "sum_sum_2dc2fbd588d9"
SHAPE_LABEL = "timm_swin_base_patch4_window7_224_train_f07fc870"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"

BATCH = 512
HEADS = 16
QUERY = 49
KEY = 49
BIAS_ROWS = 169



@triton.jit
def _fused_fma_relative_bias_kernel(
    bmm_ptr,
    div_ptr,
    index_ptr,
    bias_out_ptr,
    fma_out_ptr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
) -> None:
    n_block = tl.program_id(0)
    head = tl.program_id(1)
    query = tl.program_id(2)

    n_offsets = n_block * BLOCK_N + tl.arange(0, BLOCK_N)[:, None]
    key_offsets = tl.arange(0, BLOCK_K)[None, :]
    n_mask = n_offsets < 512
    key_mask = key_offsets < 49
    mask = n_mask & key_mask

    bmm_offsets = ((n_offsets * 16 + head) * 49 + query) * 49 + key_offsets
    div_offsets = n_offsets * 38912 + head * 2432 + query * 49 + key_offsets

    bmm = tl.load(bmm_ptr + bmm_offsets, mask=mask, other=0.0)
    div = tl.load(div_ptr + div_offsets, mask=mask, other=0.0)
    product = bmm * div
    row_sum = tl.sum(tl.where(key_mask, product, 0.0), axis=1)[:, None]
    fma = tl.fma(-div, row_sum, product)

    tl.store(fma_out_ptr + bmm_offsets, fma, mask=mask)

    partial = tl.sum(tl.where(n_mask, fma, 0.0), axis=0)
    flat_key_offsets = tl.arange(0, BLOCK_K)
    flat_key_mask = flat_key_offsets < 49
    bias_indices = tl.load(index_ptr + query * 49 + flat_key_offsets, mask=flat_key_mask, other=0)
    bias_indices = tl.where(bias_indices < 0, bias_indices + 169, bias_indices)
    tl.atomic_add(
        bias_out_ptr + bias_indices * 16 + head,
        partial,
        sem="relaxed",
        mask=flat_key_mask,
    )


def _load_repro_module():
    spec = importlib.util.spec_from_file_location(f"{REPRO_ID}_repro", REPRO_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load repro module from {REPRO_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_inputs(device: torch.device) -> tuple:
    from repro_harness import load_shape_configs, make_inputs_from_config

    configs = load_shape_configs(str(REPRO_PATH))
    if not configs:
        module = _load_repro_module()
        inputs = module.make_inputs()
    else:
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

    moved = []
    for value in inputs:
        if isinstance(value, torch.Tensor):
            moved.append(value.to(device=device))
        else:
            moved.append(value)
    return tuple(moved)


def torch_direct_oracle(
    bmm_125: torch.Tensor,
    div_10: torch.Tensor,
    primals_75: torch.Tensor,
    full_default_2: torch.Tensor,
    shape_param_0,
    shape_param_1,
    shape_param_2,
) -> tuple[torch.Tensor, torch.Tensor]:
    reshaped = torch.ops.aten.reshape.default(bmm_125, shape_param_0)
    product = reshaped * div_10
    row_sum = product.sum(dim=-1, keepdim=True)
    fma = torch.ops.prims.fma.default(-div_10, row_sum, product)

    bias_values = fma.sum(dim=0).permute(1, 2, 0).reshape(shape_param_1)
    bias_indices = primals_75.reshape(-1)
    bias_out = torch.ops.aten.index_put.default(
        full_default_2,
        [bias_indices],
        bias_values,
        True,
    )
    return bias_out, fma.reshape(shape_param_2)


def triton_fused_oracle(
    bmm_125: torch.Tensor,
    div_10: torch.Tensor,
    primals_75: torch.Tensor,
    full_default_2: torch.Tensor,
    _shape_param_0,
    _shape_param_1,
    _shape_param_2,
    *,
    block_n: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not bmm_125.is_cuda:
        raise RuntimeError("triton-fused oracle requires CUDA inputs")

    bias_out = full_default_2.clone()
    fma_out = torch.empty_like(bmm_125)
    grid = (triton.cdiv(BATCH, block_n), HEADS, QUERY)
    _fused_fma_relative_bias_kernel[grid](
        bmm_125,
        div_10,
        primals_75,
        bias_out,
        fma_out,
        BLOCK_N=block_n,
        BLOCK_K=64,
        num_warps=8,
    )
    return bias_out, fma_out


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def max_abs_diff(actual: tuple[torch.Tensor, ...], expected: tuple[torch.Tensor, ...]) -> float:
    return max((a.float() - e.float()).abs().max().item() for a, e in zip(actual, expected))


def allclose(actual: tuple[torch.Tensor, ...], expected: tuple[torch.Tensor, ...], rtol: float, atol: float) -> bool:
    return all(torch.allclose(a.float(), e.float(), rtol=rtol, atol=atol) for a, e in zip(actual, expected))


def benchmark(fn: Callable[[], object], device: torch.device, warmup: int, rep: int) -> float:
    if device.type == "cuda":
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep, return_mode="min")
        return ms * 1000.0

    import time

    for _ in range(warmup):
        fn()
    best_s = math.inf
    for _ in range(rep):
        start = time.perf_counter()
        fn()
        best_s = min(best_s, time.perf_counter() - start)
    return best_s * 1_000_000.0


def compiled_repro_fn(inputs: tuple, config: dict[str, object]) -> Callable[[], object]:
    torch._dynamo.reset()
    module = _load_repro_module().Repro()
    with inductor_config.patch(config):
        compiled = torch.compile(module)
        with torch.no_grad():
            for _ in range(3):
                compiled(*inputs)
            torch.cuda.synchronize()
    return lambda: compiled(*inputs)


def parse_args() -> argparse.Namespace:
    default_impl = "triton-fused" if torch.cuda.is_available() else "torch-direct"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--impl", choices=("triton-fused", "torch-direct"), default=default_impl)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--block-n", type=int, default=64)
    parser.add_argument("--check", action="store_true", help="Verify oracle against repro.py.")
    parser.add_argument("--bench", action="store_true", help="Benchmark the oracle and compiled baselines.")
    parser.add_argument("--skip-baselines", action="store_true", help="Only benchmark the oracle.")
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--rep", type=int, default=100)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--atol", type=float, default=1e-2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    inputs = make_inputs(device)

    if args.impl == "triton-fused":
        oracle_fn = lambda: triton_fused_oracle(*inputs, block_n=args.block_n)
    else:
        oracle_fn = lambda: torch_direct_oracle(*inputs)

    with torch.no_grad():
        out = oracle_fn()
        synchronize(device)

        if args.check:
            ref = _load_repro_module().Repro().to(device)(*inputs)
            synchronize(device)
            diff = max_abs_diff(out, ref)
            correct = allclose(out, ref, rtol=args.rtol, atol=args.atol)
            print(f"correct={correct} max_abs_diff={diff:.6g} rtol={args.rtol} atol={args.atol}")
            if not correct:
                raise SystemExit(1)

        if not args.bench:
            return

        oracle_us = benchmark(oracle_fn, device, args.warmup, args.rep)
        print(
            f"oracle_us={oracle_us:.3f} impl={args.impl} device={device} "
            f"warmup={args.warmup} rep={args.rep}"
        )

        if args.skip_baselines:
            return
        if device.type != "cuda":
            print("skipping compiled baselines on non-CUDA device")
            return

        baseline_configs = [
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
        for label, config in baseline_configs:
            fn = compiled_repro_fn(inputs, config)
            compiled_us = benchmark(fn, device, args.warmup, args.rep)
            print(f"{label}_us={compiled_us:.3f}")


if __name__ == "__main__":
    main()
