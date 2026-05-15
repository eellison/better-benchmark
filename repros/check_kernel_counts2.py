"""Compare kernel counts: baseline vs force-split-all for Qwen3."""
import sys
sys.path.insert(0, "/tmp/pytorch-work")
import torch
import torch._inductor.config as cfg
import torch._inductor.ir as ir
from torch._inductor.graph import GraphLowering
from torch._inductor.loop_body import LoopBody
from torch._inductor.sizevars import statically_known_true
from torch.utils._sympy.functions import Identity
import sympy, copy as _copy, importlib.util, os
from torch.utils._sympy.value_ranges import ValueRanges
from functools import reduce
import operator
cfg.force_disable_caches = True

# Import core pass functions directly (without profitability check)
exec(open("repo/repros/test_all_models_v2.py").read().split("# === Benchmark helper ===")[0])

original_update = GraphLowering._update_scheduler

def force_split_update(self):
    from torch._inductor.scheduler import Scheduler
    n_split = 0
    split_any = True
    while split_any:
        split_any = False
        for op in list(self.operations):
            if isinstance(op, ir.ComputedBuffer) and isinstance(op.data, ir.Pointwise):
                if split_pointwise_cat(self, op):
                    split_any = True
                    n_split += 1
                    break
    print(f"  Force-split: {n_split} ops")
    with cfg.patch("triton.store_cubin", False):
        self.scheduler = Scheduler(self.operations)
    nodes = self.scheduler.nodes
    fused = [n for n in nodes if hasattr(n, 'snodes') and len(n.snodes) > 1]
    print(f"  Result: {len(nodes)} scheduler nodes ({len(fused)} fused, {len(nodes)-len(fused)} single)")
    for n in nodes:
        if hasattr(n, 'snodes'):
            names = [s.node.get_name() for s in n.snodes if hasattr(s, 'node')]
            print(f"    Fused: {names}")
        else:
            name = n.node.get_name() if hasattr(n, 'node') else '?'
            print(f"    Single: {name}")

# Also check Mistral baseline fusion
repro_dir = "repo/repros/multi_kernel_inference"
models = [
    ("Qwen3-0.6B", "k03_Qwen3-0.6B_inference__region_005_mean_c0e8d7954fcb_fcbe8ae0.py"),
    ("Mistral-7B", "k03_vllm_mistralai_Mistral-7B-Instruct-v0.3_inference__region_011_pointwise_108892bc6e62_8f1de50a.py"),
    ("gpt-oss-20b", "k03_vllm_openai_gpt-oss-20b_inference__region_013_pointwise_1b8e92f2de8a_3af1681a.py"),
]

for name, filename in models:
    filepath = os.path.join(repro_dir, filename)
    if not os.path.exists(filepath):
        continue
    print(f"\n=== {name} BASELINE ===")
    spec = importlib.util.spec_from_file_location("repro_mod", filepath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    inputs = mod.make_inputs()
    
    torch._dynamo.reset()
    def baseline_update(self):
        from torch._inductor.scheduler import Scheduler
        with cfg.patch("triton.store_cubin", False):
            self.scheduler = Scheduler(self.operations)
        nodes = self.scheduler.nodes
        fused = [n for n in nodes if hasattr(n, 'snodes') and len(n.snodes) > 1]
        print(f"  Baseline: {len(nodes)} scheduler nodes ({len(fused)} fused, {len(nodes)-len(fused)} single)")
        for n in nodes:
            if hasattr(n, 'snodes'):
                names = [s.node.get_name() for s in n.snodes if hasattr(s, 'node')]
                print(f"    Fused: {names}")
            else:
                nm = n.node.get_name() if hasattr(n, 'node') else '?'
                print(f"    Single: {nm}")
    GraphLowering._update_scheduler = baseline_update
    compiled = torch.compile(mod.Repro())
    with torch.no_grad():
        compiled(*inputs)
    
    print(f"\n=== {name} FORCE-SPLIT ===")
    torch._dynamo.reset()
    GraphLowering._update_scheduler = force_split_update
    try:
        compiled2 = torch.compile(mod.Repro())
        with torch.no_grad():
            compiled2(*inputs)
    except Exception as e:
        print(f"  ERROR: {e}")

GraphLowering._update_scheduler = original_update
