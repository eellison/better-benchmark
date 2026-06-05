"""
Full-scope Triton oracle for sum_f403c03c9650.

Gap diagnosis (classification: BANDWIDTH_BOUND): the oracle consumes the same
two f32[131072, 288] tensors and shape parameters as repro.py, computes the
full sigmoid/SiLU-gradient pointwise producer, writes the backing storage for
the returned f32[288, 131072] transposed view with stride (1, 288), and
accumulates the returned f32[288] feature sum from the same pass. This differs
from a naive schedule by avoiding a second full read of the producer for the
sum, but it does not currently demonstrate an Inductor scheduling limitation:
the historical best compile time is already at the memory/exp throughput floor
for this full materialization plus reduction. The Inductor fix classification
is therefore BANDWIDTH_BOUND; no compiler change should be claimed unless a
full-scope Triton implementation beats both required local compile configs and
the historical best_compile_us gate.
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



REPRO_ID = "sum_f403c03c9650"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"

M = 131072
N = 288
HISTORICAL_BEST_COMPILE_US = 87.90399879217148
DEFAULT_BLOCK_M = 256
DEFAULT_BLOCK_N = 32

sys.path.insert(0, str(REPO_ROOT))


@triton.jit
def _pointwise_copy_reduce_partials_kernel(
    mm_ptr,
    addmm_ptr,
    layout_storage_ptr,
    partials_ptr,
    num_m_blocks: tl.constexpr,
    M_: tl.constexpr,
    N_: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offsets = m_offsets[:, None] * N_ + n_offsets[None, :]
    mask = (m_offsets[:, None] < M_) & (n_offsets[None, :] < N_)

    mm_values = tl.load(mm_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    addmm_values = tl.load(addmm_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    sigmoid = 1.0 / (tl.exp(-addmm_values) + 1.0)
    values = mm_values * sigmoid * (addmm_values * (1.0 - sigmoid) + 1.0)

    tl.store(layout_storage_ptr + offsets, values, mask=mask)
    partial = tl.sum(tl.where(mask, values, 0.0), axis=0)
    tl.store(
        partials_ptr + n_offsets * num_m_blocks + pid_m,
        partial,
        mask=n_offsets < N_,
    )


@triton.jit
def _finish_partials_kernel(
    partials_ptr,
    sum_out_ptr,
    num_m_blocks: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    feature = tl.program_id(0)
    row_blocks = tl.arange(0, BLOCK_R)
    mask = row_blocks < num_m_blocks
    values = tl.load(
        partials_ptr + feature * num_m_blocks + row_blocks,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    tl.store(sum_out_ptr + feature, tl.sum(values, axis=0))


def make_inputs() -> tuple[object, ...]:
    module = _load_repro_module()
    return tuple(x.cuda() if isinstance(x, torch.Tensor) else x for x in module.make_inputs())


def _num_m_blocks(block_m: int) -> int:
    return triton.cdiv(M, block_m)


def _is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def _make_workspace(
    device: torch.device,
    block_m: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_m_blocks = _num_m_blocks(block_m)
    layout_out = torch.empty_strided((N, M), (1, N), device=device, dtype=torch.float32)
    partials = torch.empty((N, num_m_blocks), device=device, dtype=torch.float32)
    sum_out = torch.empty((N,), device=device, dtype=torch.float32)
    return layout_out, partials, sum_out


def _oracle_into(
    mm_66: torch.Tensor,
    addmm_2: torch.Tensor,
    layout_out: torch.Tensor,
    partials: torch.Tensor,
    sum_out: torch.Tensor,
    block_m: int,
    block_n: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert mm_66.shape == (M, N)
    assert addmm_2.shape == (M, N)
    assert mm_66.dtype is torch.float32
    assert addmm_2.dtype is torch.float32
    assert mm_66.is_contiguous()
    assert addmm_2.is_contiguous()
    assert layout_out.shape == (N, M)
    assert layout_out.stride() == (1, N)
    assert partials.shape == (N, _num_m_blocks(block_m))
    assert sum_out.shape == (N,)

    num_m_blocks = _num_m_blocks(block_m)
    _pointwise_copy_reduce_partials_kernel[(num_m_blocks, triton.cdiv(N, block_n))](
        mm_66,
        addmm_2,
        layout_out,
        partials,
        num_m_blocks=num_m_blocks,
        M_=M,
        N_=N,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=8,
    )
    _finish_partials_kernel[(N,)](
        partials,
        sum_out,
        num_m_blocks=num_m_blocks,
        BLOCK_R=triton.next_power_of_2(num_m_blocks),
        num_warps=8,
    )
    return layout_out, sum_out


def oracle_fused(
    mm_66: torch.Tensor,
    addmm_2: torch.Tensor,
    _shape_param_0,
    _shape_param_1,
    _shape_param_2,
    _shape_param_3,
    *,
    block_m: int = DEFAULT_BLOCK_M,
    block_n: int = DEFAULT_BLOCK_N,
) -> tuple[torch.Tensor, torch.Tensor]:
    layout_out, partials, sum_out = _make_workspace(mm_66.device, block_m)
    return _oracle_into(mm_66, addmm_2, layout_out, partials, sum_out, block_m, block_n)


def reference_outputs(inputs: tuple[object, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    module = _load_repro_module()
    model = module.Repro().cuda()
    with torch.no_grad():
        return model(*inputs)


def _capture_cuda_graph(fn):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    torch.cuda.synchronize()
    return graph


def oracle_forward(inputs):
    return oracle_fused(*inputs)


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
