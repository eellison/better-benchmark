"""
Full-scope oracle for sum_sum_f10bfbcf6729 (LearningToPaint BN-backward tail).

Gap diagnosis (classification: BANDWIDTH_BOUND): The timed oracle consumes the
same original inputs as repro.py and returns the same `[1024, 512, 4, 4]`
gradient tensor plus `[512]` vector. It does the avg-pool backward producer
algebraically as `where(arg102 <= 0, 0, mm / 16)`, shares that producer across
two channel reductions, and then uses the sibling sums in a full output
epilogue. Inductor's tuned coordinate-descent reduction path already runs this
full graph near the byte-accounted floor for this shape, so a handwritten
multi-accumulator oracle does not expose an actionable missing fusion beyond
normal reduction tiling. The fix classification is BANDWIDTH_BOUND: keep the
artifact as a full-scope diagnosis, but do not treat it as a floor improvement
unless it beats both measured compile configurations.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl


REPRO_ID = "sum_sum_f10bfbcf6729"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"

N = 1024
C = 512
H = 4
W = 4
HW = H * W
TOTAL_REDUCE = N * HW
NUMEL = N * C * HW
AVGPOOL_SCALE = 0.0625
BN_SCALE = 6.103515625e-05



@triton.jit
def _dual_reduce_channel_kernel(
    mm_ptr,
    mask_input_ptr,
    centered_source_ptr,
    mean_ptr,
    invstd_ptr,
    sum1_ptr,
    sum2_ptr,
    vector_out_ptr,
    C_: tl.constexpr,
    HW_: tl.constexpr,
    TOTAL_REDUCE_: tl.constexpr,
    AVGPOOL_SCALE_: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_TILES: tl.constexpr,
):
    c = tl.program_id(0)
    mean = tl.load(mean_ptr + c).to(tl.float32)
    acc1 = tl.full((), 0.0, tl.float32)
    acc2 = tl.full((), 0.0, tl.float32)

    for tile in range(0, NUM_TILES):
        k = tile * BLOCK_K + tl.arange(0, BLOCK_K)
        active = k < TOTAL_REDUCE_
        n = k // HW_
        hw = k - n * HW_
        offsets = n * (C_ * HW_) + c * HW_ + hw
        mm_offsets = n * C_ + c

        pooled_grad = tl.load(mm_ptr + mm_offsets, mask=active, other=0.0).to(tl.float32) * AVGPOOL_SCALE_
        mask_input = tl.load(mask_input_ptr + offsets, mask=active, other=0.0).to(tl.float32)
        where_self = tl.where(mask_input <= 0.0, 0.0, pooled_grad)
        where_self = tl.where(active, where_self, 0.0)

        source = tl.load(centered_source_ptr + offsets, mask=active, other=0.0).to(tl.float32)
        centered = source - mean
        acc1 += tl.sum(where_self, axis=0)
        acc2 += tl.sum(where_self * centered, axis=0)

    invstd = tl.load(invstd_ptr + c).to(tl.float32)

    tl.store(sum1_ptr + c, acc1)
    tl.store(sum2_ptr + c, acc2)
    tl.store(vector_out_ptr + c, acc2 * invstd)


@triton.jit
def _full_epilogue_kernel(
    mm_ptr,
    mask_input_ptr,
    centered_source_ptr,
    mean_ptr,
    invstd_ptr,
    affine_weight_ptr,
    sum1_ptr,
    sum2_ptr,
    out_ptr,
    NUMEL_: tl.constexpr,
    C_: tl.constexpr,
    HW_: tl.constexpr,
    AVGPOOL_SCALE_: tl.constexpr,
    BN_SCALE_: tl.constexpr,
    BLOCK_ELEMS: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_ELEMS + tl.arange(0, BLOCK_ELEMS)
    active = offsets < NUMEL_

    hw = offsets % HW_
    c = (offsets // HW_) % C_
    n = offsets // (C_ * HW_)
    input_offsets = n * (C_ * HW_) + c * HW_ + hw
    mm_offsets = n * C_ + c

    pooled_grad = tl.load(mm_ptr + mm_offsets, mask=active, other=0.0).to(tl.float32) * AVGPOOL_SCALE_
    mask_input = tl.load(mask_input_ptr + input_offsets, mask=active, other=0.0).to(tl.float32)
    where_self = tl.where(mask_input <= 0.0, 0.0, pooled_grad)

    source = tl.load(centered_source_ptr + input_offsets, mask=active, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c, mask=active, other=0.0).to(tl.float32)
    invstd = tl.load(invstd_ptr + c, mask=active, other=0.0).to(tl.float32)
    affine_weight = tl.load(affine_weight_ptr + c, mask=active, other=0.0).to(tl.float32)
    sum1 = tl.load(sum1_ptr + c, mask=active, other=0.0).to(tl.float32)
    sum2 = tl.load(sum2_ptr + c, mask=active, other=0.0).to(tl.float32)

    centered = source - mean
    mean_term = sum1 * BN_SCALE_
    variance_term = sum2 * BN_SCALE_ * invstd * invstd
    affine_term = invstd * affine_weight
    result = (where_self - centered * variance_term - mean_term) * affine_term
    tl.store(out_ptr + input_offsets, result, mask=active)


def _load_repro_module():
    spec = importlib.util.spec_from_file_location(f"{REPRO_ID}_repro", REPRO_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load repro module from {REPRO_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_inputs() -> tuple[object, ...]:
    module = _load_repro_module()
    return tuple(x.cuda() if isinstance(x, torch.Tensor) else x for x in module.make_inputs())


def oracle_fused(
    mm: torch.Tensor,
    arg102_1: torch.Tensor,
    arg100_1: torch.Tensor,
    arg105_1: torch.Tensor,
    arg101_1: torch.Tensor,
    arg42_1: torch.Tensor,
    _shape_param_0: object,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert mm.shape == (N, C)
    assert arg102_1.shape == (N, C, H, W)
    assert arg100_1.shape == (N, C, H, W)
    assert arg105_1.shape == (1, C, 1, 1)
    assert arg101_1.shape == (C,)
    assert arg42_1.shape == (C,)
    assert mm.is_contiguous()
    assert arg102_1.is_contiguous()
    assert arg100_1.is_contiguous()

    block_k = 1024
    num_tiles = triton.cdiv(TOTAL_REDUCE, block_k)
    sum1 = torch.empty((C,), device=mm.device, dtype=torch.float32)
    sum2 = torch.empty((C,), device=mm.device, dtype=torch.float32)
    vector_out = torch.empty((C,), device=mm.device, dtype=torch.float32)

    _dual_reduce_channel_kernel[(C,)](
        mm,
        arg102_1,
        arg100_1,
        arg105_1,
        arg101_1,
        sum1,
        sum2,
        vector_out,
        C_=C,
        HW_=HW,
        TOTAL_REDUCE_=TOTAL_REDUCE,
        AVGPOOL_SCALE_=AVGPOOL_SCALE,
        BLOCK_K=block_k,
        NUM_TILES=num_tiles,
        num_warps=4,
    )

    out = torch.empty((N, C, H, W), device=mm.device, dtype=torch.float32)
    block_elems = 512
    _full_epilogue_kernel[(triton.cdiv(NUMEL, block_elems),)](
        mm,
        arg102_1,
        arg100_1,
        arg105_1,
        arg101_1,
        arg42_1,
        sum1,
        sum2,
        out,
        NUMEL_=NUMEL,
        C_=C,
        HW_=HW,
        AVGPOOL_SCALE_=AVGPOOL_SCALE,
        BN_SCALE_=BN_SCALE,
        BLOCK_ELEMS=block_elems,
        num_warps=4,
    )

    return out, vector_out


def reference_outputs(inputs: tuple[object, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    module = _load_repro_module()
    model = module.Repro().cuda()
    with torch.no_grad():
        return model(*inputs)


def _max_diff(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    diff = (actual.float() - expected.float()).abs()
    rel = diff / (expected.float().abs() + 1e-8)
    return diff.max().item(), rel.max().item()


def run_check(rtol: float, atol: float) -> bool:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Triton oracle check")

    torch.manual_seed(0)
    inputs = make_inputs()
    with torch.no_grad():
        ref = reference_outputs(inputs)
        actual = oracle_fused(*inputs)
        torch.cuda.synchronize()

    ok = True
    for idx, (got, expected) in enumerate(zip(actual, ref)):
        max_abs, max_rel = _max_diff(got, expected)
        output_ok = torch.allclose(got.float(), expected.float(), rtol=rtol, atol=atol)
        stride_ok = got.stride() == expected.stride()
        ok = ok and output_ok and stride_ok
        print(
            f"output[{idx}]: shape={list(got.shape)} max_abs={max_abs:.6e} "
            f"max_rel={max_rel:.6e} allclose={output_ok} stride_match={stride_ok}"
        )

    print(f"Correctness: {'PASS' if ok else 'FAIL'}")
    return ok


def _compile_with_config(model: torch.nn.Module, inputs: tuple[object, ...], config: dict[str, object]):
    import torch._dynamo
    import torch._inductor.config as inductor_config

    torch._dynamo.reset()
    with inductor_config.patch(config):
        compiled = torch.compile(model)
        for _ in range(3):
            compiled(*inputs)
        torch.cuda.synchronize()
    return compiled


def run_bench(rep: int, warmup: int, no_compile: bool) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Triton oracle benchmark")

    torch.manual_seed(0)
    inputs = make_inputs()
    with torch.no_grad():
        oracle_fused(*inputs)
        torch.cuda.synchronize()
        oracle_us = triton.testing.do_bench(
            lambda: oracle_fused(*inputs),
            warmup=warmup,
            rep=rep,
            return_mode="min",
        ) * 1000.0
    print(f"oracle_fused full-scope masked dual reduction + epilogue: {oracle_us:.3f} us")

    if no_compile:
        return

    module = _load_repro_module()
    compile_configs = [
        ("coordinate_descent_tuning=True", {"coordinate_descent_tuning": True}),
        (
            (
                "combo_kernels=True,combo_kernel_per_subkernel_blocks=True,"
                "coordinate_descent_tuning=True,benchmark_combo_kernel=True,triton.multi_kernel=3"
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
    for label, config in compile_configs:
        model = module.Repro().cuda()
        with torch.no_grad():
            compiled = _compile_with_config(model, inputs, config)
            compiled_us = triton.testing.do_bench(
                lambda: compiled(*inputs),
                warmup=warmup,
                rep=rep,
                return_mode="min",
            ) * 1000.0
        print(f"torch.compile {label}: {compiled_us:.3f} us")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="run correctness check against repro.py")
    parser.add_argument("--bench", action="store_true", help="run timing benchmark")
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument("--atol", type=float, default=5e-2)
    parser.add_argument("--rep", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--no-compile", action="store_true", help="only benchmark the oracle")
    args = parser.parse_args()

    if not args.check and not args.bench:
        args.check = True
        args.bench = True

    if args.check and not run_check(rtol=args.rtol, atol=args.atol):
        sys.exit(1)
    if args.bench:
        run_bench(rep=args.rep, warmup=args.warmup, no_compile=args.no_compile)


if __name__ == "__main__":
    with torch.no_grad():
        main()
