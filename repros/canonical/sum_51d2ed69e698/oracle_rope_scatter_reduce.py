"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the full Mistral grouped-head sum, RoPE half-rotation algebra, and both returned transposed layouts directly in Triton, whereas Inductor currently lowers the view/sum/sin/cos/slice_scatter/add/clone/view/permute graph as separate generic reduction, pointwise, materialization, and layout kernels; Inductor cannot do this today because its scheduler/codegen does not recognize grouped-head RoPE plus final transpose-layout stores as one fused producer with strided output epilogues; the fix is SCHEDULER_FUSION: add an Inductor grouped-RoPE fusion that accumulates grouped heads once, applies the paired half-rotation epilogue, and writes the complete returned strides directly."""
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
except ModuleNotFoundError:  # pragma: no cover - keeps py_compile working without Triton.
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

REPRO_ID = "sum_51d2ed69e698"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"
SHAPE_LABEL = "vllm_mistralai_Mistral-7B-Instruct-v0.3_001_3e8b8880"

BATCH = 4
HEADS = 32
GROUPS = 8
HEADS_PER_GROUP = HEADS // GROUPS
SEQ = 512
HEAD_DIM = 128
HALF_DIM = HEAD_DIM // 2
TOKENS = BATCH * SEQ
HIDDEN = HEADS * HEAD_DIM
GROUP_HIDDEN = GROUPS * HEAD_DIM
FULL_TOTAL = TOKENS * HIDDEN
GROUP_TOTAL = TOKENS * GROUP_HIDDEN
TABLE_TOTAL = SEQ * HEAD_DIM
BLOCK_SIZE = 1024
TABLE_BLOCK_SIZE = 256



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


def _rotary_tables_torch(arg1_1: torch.Tensor, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    positions = torch.arange(SEQ, device=arg1_1.device, dtype=torch.float32)
    freqs = positions[:, None] * arg1_1[None, :]
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
    arg1_1: torch.Tensor,
    getitem: torch.Tensor,
    *shape_params: object,
) -> tuple[torch.Tensor, torch.Tensor]:
    del shape_params
    sin, cos = _rotary_tables_torch(arg1_1, getitem.dtype)
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
        arg1_ptr,
        sin_ptr,
        cos_ptr,
        TABLE_BLOCK_SIZE_: tl.constexpr,
    ):
        offsets = tl.program_id(0) * TABLE_BLOCK_SIZE_ + tl.arange(0, TABLE_BLOCK_SIZE_)
        mask = offsets < 65536
        seq = offsets // 128
        dim = offsets - seq * 128
        freq_dim = dim % 64
        freq = seq.to(tl.float32) * tl.load(arg1_ptr + freq_dim, mask=mask, other=0.0)
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
        mask = offsets < 8388608

        hidden = offsets % 4096
        token = offsets // 4096
        batch = token // 512
        seq = token - batch * 512
        head = hidden // 128
        dim = hidden - head * 128

        src_dim = tl.where(dim < 64, dim + 64, dim - 64)
        sign = tl.where(dim < 64, 1.0, -1.0)
        x_offset = ((batch * 32 + head) * 512 + seq) * 128 + dim
        src_offset = ((batch * 32 + head) * 512 + seq) * 128 + src_dim

        x_val = tl.load(x_ptr + x_offset, mask=mask, other=0.0).to(tl.float32)
        src_val = tl.load(x_ptr + src_offset, mask=mask, other=0.0).to(tl.float32)
        sin_val = tl.load(sin_ptr + seq * 128 + src_dim, mask=mask, other=0.0).to(tl.float32)
        cos_val = tl.load(cos_ptr + seq * 128 + dim, mask=mask, other=0.0).to(tl.float32)

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
        mask = offsets < 2097152

        hidden = offsets % 1024
        token = offsets // 1024
        batch = token // 512
        seq = token - batch * 512
        group = hidden // 128
        dim = hidden - group * 128

        src_dim = tl.where(dim < 64, dim + 64, dim - 64)
        sign = tl.where(dim < 64, 1.0, -1.0)
        head_base = group * 4

        offset0 = ((batch * 32 + head_base + 0) * 512 + seq) * 128 + dim
        offset1 = ((batch * 32 + head_base + 1) * 512 + seq) * 128 + dim
        offset2 = ((batch * 32 + head_base + 2) * 512 + seq) * 128 + dim
        offset3 = ((batch * 32 + head_base + 3) * 512 + seq) * 128 + dim
        src_offset0 = ((batch * 32 + head_base + 0) * 512 + seq) * 128 + src_dim
        src_offset1 = ((batch * 32 + head_base + 1) * 512 + seq) * 128 + src_dim
        src_offset2 = ((batch * 32 + head_base + 2) * 512 + seq) * 128 + src_dim
        src_offset3 = ((batch * 32 + head_base + 3) * 512 + seq) * 128 + src_dim

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

        sin_val = tl.load(sin_ptr + seq * 128 + src_dim, mask=mask, other=0.0).to(tl.float32)
        cos_val = tl.load(cos_ptr + seq * 128 + dim, mask=mask, other=0.0).to(tl.float32)
        rotated = (src_sum * sin_val).to(tl.bfloat16).to(tl.float32) * sign
        scaled = (x_sum * cos_val).to(tl.bfloat16).to(tl.float32)
        tl.store(out_ptr + offsets, (rotated + scaled).to(tl.bfloat16), mask=mask)


def oracle_triton(
    getitem_1: torch.Tensor,
    arg1_1: torch.Tensor,
    getitem: torch.Tensor,
    *shape_params: object,
) -> tuple[torch.Tensor, torch.Tensor]:
    del shape_params
    if triton is None:
        raise RuntimeError("triton is not available")
    if getitem.device.type != "cuda":
        raise RuntimeError("triton oracle requires CUDA inputs")
    if getitem_1.dtype is not torch.bfloat16 or getitem.dtype is not torch.bfloat16:
        raise TypeError("expected bf16 source tensors")

    sin = torch.empty((SEQ, HEAD_DIM), device=getitem.device, dtype=getitem.dtype)
    cos = torch.empty_like(sin)
    _rotary_table_kernel[(triton.cdiv(TABLE_TOTAL, TABLE_BLOCK_SIZE),)](
        arg1_1,
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
    out = model(*inputs)
    if not isinstance(out, tuple):
        raise TypeError(f"expected tuple output from repro, got {type(out)!r}")
    return out


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


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


@oracle_impl(hardware="H100", shapes="(T([4, 32, 512, 128], bf16), T([64], f32), T([4, 32, 512, 128], bf16), S([4, 8, 4, 512, 128]), S([1, 64, 1]), S([1, 1, 512]), S([1, 512, 2, 64]), S([1, 512, 128]), S([4, 512, 1024]), S([2048, 1024]), S([4, 512, 4096]), S([2048, 4096]))")
def oracle_forward(inputs):
    return oracle_triton(*inputs)


def main():
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
        print(f"Correctness: {'PASS' if ok else 'FAIL'}")
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
