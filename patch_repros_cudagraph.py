"""
Patch all existing repro scripts to use CUDA Graph timing instead of wall-clock timing.
This eliminates Python/Dynamo dispatch overhead from measurements.
"""

import glob
import os
import re

CUDA_GRAPH_HELPER = '''def _cuda_graph_time(fn, inputs, n_warmup=10, n_iter=200):
    """Time a function using CUDA Graph replay (eliminates dispatch overhead)."""
    import time

    with torch.no_grad():
        # Warmup and compile
        for _ in range(n_warmup):
            fn(*inputs)
        torch.cuda.synchronize()

        # Capture CUDA Graph (must be in same no_grad context as warmup)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fn(*inputs)
        torch.cuda.synchronize()

        # Warmup replays
        for _ in range(5):
            g.replay()
        torch.cuda.synchronize()

        # Timed replays
        start = time.perf_counter()
        for _ in range(n_iter):
            g.replay()
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) / n_iter * 1e6
    return elapsed


'''

NEW_BENCHMARK_BODY = '''def benchmark(n_warmup=25, n_iter=200):
    mod = Repro()
    inputs = make_inputs()

    # Eager baseline
    with torch.no_grad():
        eager_out = mod(*inputs)

    total_bytes = _count_bytes(inputs, eager_out)

    # Compiled (default heuristics) - timed via CUDA Graph replay
    compiled = torch.compile(mod)
    compiled_time = _cuda_graph_time(compiled, inputs, n_warmup, n_iter)

    # Compiled with coordinate descent tuning
    inductor_config.coordinate_descent_tuning = True
    torch._dynamo.reset()
    compiled_cd = torch.compile(mod)
    cd_time = _cuda_graph_time(compiled_cd, inputs, n_warmup, n_iter)

    # Memcopy SOL at multiple sizes
    sol_results = _measure_memcopy_sol(total_bytes)
    kernel_sol = sol_results[total_bytes]

    print(f"\\nKernel data: {total_bytes / 1024:.1f} KB (read+write)")
    print(f"CUDA Graph (default):    {compiled_time:8.1f} us")
    print(f"CUDA Graph (coord desc): {cd_time:8.1f} us")
    print(f"Memcopy SOL (same size): {kernel_sol['time_us']:8.1f} us  ({kernel_sol['bw_gbps']:.1f} GB/s)")
    print(f"Gap (default / SOL):     {compiled_time / kernel_sol['time_us']:8.2f}x")
    print(f"Gap (CD / SOL):          {cd_time / kernel_sol['time_us']:8.2f}x")
    print(f"\\nMemcopy bandwidth curve:")
    for nbytes in sorted(sol_results):
        r = sol_results[nbytes]
        marker = " <-- kernel size" if r["is_kernel_size"] else ""
        print(f"  {r['label']:>10s}: {r['time_us']:8.1f} us  {r['bw_gbps']:.1f} GB/s{marker}")

    return {
        "compiled_us": compiled_time,
        "coord_descent_us": cd_time,
        "memcopy_sol_us": kernel_sol["time_us"],
        "total_bytes": total_bytes,
        "sol_curve": sol_results,
    }


if __name__ == "__main__":
    benchmark()
'''


def patch_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    if 'def benchmark(' not in content:
        return 'no_benchmark'

    # Find the benchmark function (or _cuda_graph_time if already patched) and replace everything from it to end
    # First check if it already has the correct version
    if 'g = torch.cuda.CUDAGraph()' in content and 'g.replay()' in content:
        return 'already_patched'

    # Find the start of the benchmark function (or _cuda_graph_time if partially patched)
    match = re.search(r'^def (_cuda_graph_time|benchmark)\(', content, re.MULTILINE)
    if not match:
        return 'no_match'

    # Keep everything before this function, insert helper + new benchmark
    before = content[:match.start()]
    new_content = before + CUDA_GRAPH_HELPER + NEW_BENCHMARK_BODY

    with open(filepath, 'w') as f:
        f.write(new_content)
    return 'patched'


def main():
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "aten_repros")
    repros = glob.glob(os.path.join(base_dir, "**", "region_*.py"), recursive=True)
    repros = [r for r in repros if not r.endswith('_investigation.txt')]

    print(f"Found {len(repros)} repro files")

    counts = {}
    for filepath in sorted(repros):
        result = patch_file(filepath)
        counts[result] = counts.get(result, 0) + 1

    for status, count in sorted(counts.items()):
        print(f"  {status}: {count}")


if __name__ == "__main__":
    main()
