"""Gap diagnosis (classification: SCATTER_REDUCE): this oracle computes the full Llama RoPE return tuple by applying the grouped-head sum, half-rotation slice-scatter, and both required transposed stores directly from the source tensors, whereas Inductor currently materializes the grouped sum, rotary `slice_scatter` reconstructions, clones, views, and permutes as separate generic pointwise/layout work; Inductor cannot do this today because its scheduler/codegen does not model rotary slice-scatter reconstruction with a grouped-head reduction and materialized layout-changing side outputs as one structured scatter-reduce template; the fix is SCATTER_REDUCE: add a RoPE slice-scatter lowering that accumulates grouped heads once, applies the half-rotation in source space, and emits the transposed side-output stores directly."""
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
except ModuleNotFoundError:  # pragma: no cover - allows CPU-only syntax checks.
    triton = None
    tl = None


REPRO_ID = "sum_a3c9eb6b27f1"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"
SHAPE_LABEL = "vllm_meta-llama_Llama-3.2-1B_001_c9503b4c"

BATCH = 4
HEADS = 32
GROUPS = 8
HEADS_PER_GROUP = HEADS // GROUPS
SEQ = 512
HEAD_DIM = 64
HALF_DIM = HEAD_DIM // 2
TOKENS = BATCH * SEQ
HIDDEN = HEADS * HEAD_DIM
GROUP_HIDDEN = GROUPS * HEAD_DIM
FULL_TOTAL = TOKENS * HIDDEN
GROUP_TOTAL = TOKENS * GROUP_HIDDEN
BLOCK_SIZE = 1024
TABLE_BLOCK_SIZE = 256

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
        inputs = _load_repro_module().make_inputs()

    return tuple(
        value.to(device=device) if isinstance(value, torch.Tensor) else value
        for value in inputs
    )


def _rotary_tables(arg2_1: torch.Tensor, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    positions = torch.arange(SEQ, device=arg2_1.device, dtype=torch.float32)
    freqs = positions[:, None] * arg2_1[None, :]
    angles = torch.cat((freqs, freqs), dim=1)
    return torch.sin(angles).to(dtype), torch.cos(angles).to(dtype)


def _apply_rope_torch(x: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor) -> torch.Tensor:
    sin = sin[None, None, :, :]
    cos = cos[None, None, :, :]
    x_sin = x * sin
    rotated = torch.cat((x_sin[..., HALF_DIM:], -x_sin[..., :HALF_DIM]), dim=-1)
    return rotated + x * cos


def oracle_torch(
    getitem_1: torch.Tensor,
    arg2_1: torch.Tensor,
    getitem: torch.Tensor,
    _shape_param_0,
    _shape_param_1,
    _shape_param_2,
    _shape_param_3,
    _shape_param_4,
    _shape_param_5,
    _shape_param_6,
    _shape_param_7,
    _shape_param_8,
) -> tuple[torch.Tensor, torch.Tensor]:
    del (
        _shape_param_0,
        _shape_param_1,
        _shape_param_2,
        _shape_param_3,
        _shape_param_4,
        _shape_param_5,
        _shape_param_6,
        _shape_param_7,
        _shape_param_8,
    )

    sin, cos = _rotary_tables(arg2_1, getitem.dtype)
    grouped = getitem_1.view(BATCH, GROUPS, HEADS_PER_GROUP, SEQ, HEAD_DIM).sum(dim=2)

    grouped_rope = _apply_rope_torch(grouped, sin, cos)
    full_rope = _apply_rope_torch(getitem, sin, cos)

    out0 = (
        grouped_rope.permute(0, 2, 1, 3)
        .contiguous()
        .view(TOKENS, GROUP_HIDDEN)
        .permute(1, 0)
    )
    out1 = (
        full_rope.permute(0, 2, 1, 3)
        .contiguous()
        .view(TOKENS, HIDDEN)
        .permute(1, 0)
    )
    return out0, out1


if triton is not None:

    @triton.jit
    def _rotary_table_kernel(
        arg2_ptr,
        sin_ptr,
        cos_ptr,
        TABLE_BLOCK_SIZE_: tl.constexpr,
    ):
        offsets = tl.program_id(0) * TABLE_BLOCK_SIZE_ + tl.arange(0, TABLE_BLOCK_SIZE_)
        mask = offsets < 32768
        seq = offsets // 64
        dim = offsets - seq * 64
        freq_dim = dim % 32
        freq = seq.to(tl.float32) * tl.load(arg2_ptr + freq_dim, mask=mask, other=0.0)
        tl.store(sin_ptr + offsets, tl.sin(freq).to(tl.bfloat16), mask=mask)
        tl.store(cos_ptr + offsets, tl.cos(freq).to(tl.bfloat16), mask=mask)


    @triton.jit
    def _full_rope_kernel(
        x_ptr,
        sin_ptr,
        cos_ptr,
        out_ptr,
        BLOCK_SIZE_: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_SIZE_ + tl.arange(0, BLOCK_SIZE_)
        mask = offsets < 4194304

        hidden = offsets % 2048
        token = offsets // 2048
        batch = token // 512
        seq = token - batch * 512
        head = hidden // 64
        dim = hidden - head * 64

        src_dim = tl.where(dim < 32, dim + 32, dim - 32)
        sign = tl.where(dim < 32, 1.0, -1.0)
        x_offset = ((batch * 32 + head) * 512 + seq) * 64 + dim
        src_offset = ((batch * 32 + head) * 512 + seq) * 64 + src_dim

        x_val = tl.load(x_ptr + x_offset, mask=mask, other=0.0).to(tl.float32)
        src_val = tl.load(x_ptr + src_offset, mask=mask, other=0.0).to(tl.float32)
        sin_val = tl.load(sin_ptr + seq * 64 + src_dim, mask=mask, other=0.0).to(tl.float32)
        cos_val = tl.load(cos_ptr + seq * 64 + dim, mask=mask, other=0.0).to(tl.float32)

        rotated = (src_val * sin_val).to(tl.bfloat16).to(tl.float32) * sign
        scaled = (x_val * cos_val).to(tl.bfloat16).to(tl.float32)
        tl.store(out_ptr + offsets, (rotated + scaled).to(tl.bfloat16), mask=mask)


    @triton.jit
    def _grouped_rope_kernel(
        x_ptr,
        sin_ptr,
        cos_ptr,
        out_ptr,
        BLOCK_SIZE_: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_SIZE_ + tl.arange(0, BLOCK_SIZE_)
        mask = offsets < 1048576

        hidden = offsets % 512
        token = offsets // 512
        batch = token // 512
        seq = token - batch * 512
        group = hidden // 64
        dim = hidden - group * 64

        src_dim = tl.where(dim < 32, dim + 32, dim - 32)
        sign = tl.where(dim < 32, 1.0, -1.0)
        head_base = group * 4

        offset0 = ((batch * 32 + head_base + 0) * 512 + seq) * 64 + dim
        offset1 = ((batch * 32 + head_base + 1) * 512 + seq) * 64 + dim
        offset2 = ((batch * 32 + head_base + 2) * 512 + seq) * 64 + dim
        offset3 = ((batch * 32 + head_base + 3) * 512 + seq) * 64 + dim
        src_offset0 = ((batch * 32 + head_base + 0) * 512 + seq) * 64 + src_dim
        src_offset1 = ((batch * 32 + head_base + 1) * 512 + seq) * 64 + src_dim
        src_offset2 = ((batch * 32 + head_base + 2) * 512 + seq) * 64 + src_dim
        src_offset3 = ((batch * 32 + head_base + 3) * 512 + seq) * 64 + src_dim

        x_sum = (
            tl.load(x_ptr + offset0, mask=mask, other=0.0).to(tl.float32)
            + tl.load(x_ptr + offset1, mask=mask, other=0.0).to(tl.float32)
            + tl.load(x_ptr + offset2, mask=mask, other=0.0).to(tl.float32)
            + tl.load(x_ptr + offset3, mask=mask, other=0.0).to(tl.float32)
        ).to(tl.bfloat16).to(tl.float32)
        src_sum = (
            tl.load(x_ptr + src_offset0, mask=mask, other=0.0).to(tl.float32)
            + tl.load(x_ptr + src_offset1, mask=mask, other=0.0).to(tl.float32)
            + tl.load(x_ptr + src_offset2, mask=mask, other=0.0).to(tl.float32)
            + tl.load(x_ptr + src_offset3, mask=mask, other=0.0).to(tl.float32)
        ).to(tl.bfloat16).to(tl.float32)

        sin_val = tl.load(sin_ptr + seq * 64 + src_dim, mask=mask, other=0.0).to(tl.float32)
        cos_val = tl.load(cos_ptr + seq * 64 + dim, mask=mask, other=0.0).to(tl.float32)
        rotated = (src_sum * sin_val).to(tl.bfloat16).to(tl.float32) * sign
        scaled = (x_sum * cos_val).to(tl.bfloat16).to(tl.float32)
        tl.store(out_ptr + offsets, (rotated + scaled).to(tl.bfloat16), mask=mask)


def oracle_triton(
    getitem_1: torch.Tensor,
    arg2_1: torch.Tensor,
    getitem: torch.Tensor,
    *shape_params: object,
) -> tuple[torch.Tensor, torch.Tensor]:
    del shape_params
    if triton is None:
        raise RuntimeError("triton is not available")
    if getitem.device.type != "cuda":
        raise RuntimeError("triton oracle requires CUDA inputs")

    sin = torch.empty((SEQ, HEAD_DIM), device=getitem.device, dtype=getitem.dtype)
    cos = torch.empty_like(sin)
    _rotary_table_kernel[(triton.cdiv(SEQ * HEAD_DIM, TABLE_BLOCK_SIZE),)](
        arg2_1,
        sin,
        cos,
        TABLE_BLOCK_SIZE_=TABLE_BLOCK_SIZE,
        num_warps=8,
    )
    base0 = torch.empty((TOKENS, GROUP_HIDDEN), device=getitem.device, dtype=getitem.dtype)
    base1 = torch.empty((TOKENS, HIDDEN), device=getitem.device, dtype=getitem.dtype)

    _grouped_rope_kernel[(triton.cdiv(GROUP_TOTAL, BLOCK_SIZE),)](
        getitem_1,
        sin,
        cos,
        base0,
        BLOCK_SIZE_=BLOCK_SIZE,
        num_warps=4,
    )
    _full_rope_kernel[(triton.cdiv(FULL_TOTAL, BLOCK_SIZE),)](
        getitem,
        sin,
        cos,
        base1,
        BLOCK_SIZE_=BLOCK_SIZE,
        num_warps=4,
    )
    return base0.permute(1, 0), base1.permute(1, 0)


def oracle_rope_scatter_reduce(
    *inputs: object,
    impl: str = "auto",
) -> tuple[torch.Tensor, torch.Tensor]:
    first_tensor = inputs[0]
    if not isinstance(first_tensor, torch.Tensor):
        raise TypeError("first input must be a tensor")
    if impl == "auto":
        impl = "triton" if first_tensor.device.type == "cuda" and triton is not None else "torch"
    if impl == "triton":
        return oracle_triton(*inputs)
    if impl == "torch":
        return oracle_torch(*inputs)
    raise ValueError(f"unknown impl: {impl}")


def reference_outputs(
    inputs: tuple[object, ...],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    module = _load_repro_module()
    module.device = lambda *unused_args, **unused_kwargs: device
    model = module.Repro().to(device)
    return model(*inputs)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _max_diff(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    diff = (actual.float() - expected.float()).abs()
    rel = diff / (expected.float().abs() + 1e-8)
    return diff.max().item(), rel.max().item()


def run_check(device: torch.device, impl: str, rtol: float, atol: float) -> bool:
    torch.manual_seed(0)
    inputs = make_inputs(device)
    with torch.no_grad():
        expected = reference_outputs(inputs, device)
        actual = oracle_rope_scatter_reduce(*inputs, impl=impl)
        synchronize(device)

    if len(actual) != len(expected):
        print(f"tuple length mismatch: actual={len(actual)} expected={len(expected)}")
        return False

    ok = True
    for idx, (got, ref) in enumerate(zip(actual, expected)):
        max_abs, max_rel = _max_diff(got, ref)
        value_ok = torch.allclose(got.float(), ref.float(), rtol=rtol, atol=atol)
        stride_ok = got.stride() == ref.stride()
        dtype_ok = got.dtype == ref.dtype
        ok = ok and value_ok and stride_ok and dtype_ok
        print(
            f"output[{idx}]: shape={list(got.shape)} stride={got.stride()} "
            f"max_abs={max_abs:.6e} max_rel={max_rel:.6e} "
            f"allclose={value_ok} stride_match={stride_ok} dtype_match={dtype_ok}"
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


def run_bench(device: torch.device, impl: str, warmup: int, rep: int) -> None:
    torch.manual_seed(0)
    inputs = make_inputs(device)
    actual_impl = impl
    if actual_impl == "auto":
        actual_impl = "triton" if device.type == "cuda" and triton is not None else "torch"

    oracle_fn = lambda: oracle_rope_scatter_reduce(*inputs, impl=actual_impl)
    with torch.no_grad():
        oracle_fn()
        synchronize(device)
        oracle_us = benchmark(oracle_fn, device, warmup, rep)

    print(
        f"oracle_rope_scatter_reduce: {oracle_us:.3f} us "
        f"impl={actual_impl} shape={SHAPE_LABEL} device={device} "
        f"warmup={warmup} rep={rep}"
    )


def main() -> None:
    default_impl = "triton" if triton is not None and torch.cuda.is_available() else "torch"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="compare the full repro.py return tuple against the oracle")
    parser.add_argument("--bench", action="store_true", help="benchmark the selected oracle implementation")
    parser.add_argument("--impl", choices=("auto", "triton", "torch"), default=default_impl)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--rep", type=int, default=100)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-2)
    args = parser.parse_args()

    if not args.check and not args.bench:
        parser.error("select at least one mode: --check and/or --bench")

    device = torch.device(args.device)
    if args.check and not run_check(device=device, impl=args.impl, rtol=args.rtol, atol=args.atol):
        sys.exit(1)
    if args.bench:
        run_bench(device=device, impl=args.impl, warmup=args.warmup, rep=args.rep)


if __name__ == "__main__":
    main()
