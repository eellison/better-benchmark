"""Test the integrated pass against model repros."""
import sys
sys.path.insert(0, "/tmp/pytorch-work")
import torch
import torch._inductor.config as cfg
import importlib.util, os, time
cfg.force_disable_caches = True

def bench(fn, inputs, warmup=50, iters=500):
    with torch.no_grad():
        for _ in range(warmup): fn(*inputs)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g), torch.no_grad(): fn(*inputs)
    torch.cuda.synchronize()
    for _ in range(warmup): g.replay()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters): g.replay()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6

repro_dir = "repo/repros/multi_kernel_inference"
models = [
    ("Mistral-7B", "k03_vllm_mistralai_Mistral-7B-Instruct-v0.3_inference__region_011_pointwise_108892bc6e62_8f1de50a.py"),
    ("Qwen3-0.6B (r005)", "k03_Qwen3-0.6B_inference__region_005_mean_c0e8d7954fcb_fcbe8ae0.py"),
    ("gpt-oss-20b (r013)", "k03_vllm_openai_gpt-oss-20b_inference__region_013_pointwise_1b8e92f2de8a_3af1681a.py"),
]

print(f"{'Model':<25} {'Baseline':>10} {'Split':>10} {'Speedup':>8} {'MaxDiff':>8}")
print("-" * 70)

for name, filename in models:
    filepath = os.path.join(repro_dir, filename)
    if not os.path.exists(filepath):
        print(f"{name:<25} {'MISSING':>10}")
        continue

    spec = importlib.util.spec_from_file_location("repro_mod", filepath)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as e:
        print(f"{name:<25} {'LOAD ERR':>10} {str(e)[:40]}")
        continue

    inputs = mod.make_inputs()

    # Baseline (pass disabled)
    torch._dynamo.reset()
    with cfg.patch("split_cat_pointwise_pass", False):
        compiled = torch.compile(mod.Repro())
        with torch.no_grad():
            out_base = compiled(*inputs)
        t_base = bench(compiled, inputs)

    # With pass (enabled by default)
    torch._dynamo.reset()
    compiled2 = torch.compile(mod.Repro())
    with torch.no_grad():
        out_split = compiled2(*inputs)
    if isinstance(out_base, tuple):
        max_diff = max((a - b).abs().max().item() for a, b in zip(out_base, out_split))
    else:
        max_diff = (out_base - out_split).abs().max().item()
    t_split = bench(compiled2, inputs)
    speedup = t_base / t_split
    print(f"{name:<25} {t_base:>8.1f}us {t_split:>8.1f}us {speedup:>7.2f}x {max_diff:>7.3f}")
