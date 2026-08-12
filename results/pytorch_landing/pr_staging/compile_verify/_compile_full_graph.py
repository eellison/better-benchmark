"""
Shadow compile-smoke for a genai full_graph_*.py repro. Loads the graph via
full_graph_harness, runs eager, then torch.compile, compares numerics.
NOT a timing harness (no do_bench) -- correctness/compile smoke only.

Usage: PYTHONPATH=<worktree> [TORCH_LOGS=output_code] \
    python _compile_full_graph.py <full_graph.py> <tag>
"""
import sys, os
from pathlib import Path
ROOT = Path("/tmp/scratch_space/better_benchmark"); sys.path.insert(0, str(ROOT))
graph_path, tag = sys.argv[1:3]
import torch
pp = os.environ.get("PYTHONPATH", "/__none__").split(":")[0]
assert torch.__file__.startswith(pp), f"NOT shadowed: {torch.__file__}"
sys.stderr.write(f"### TAG={tag} torch={torch.__file__}\n")
from full_graph_harness import load_full_graph
instance, inputs, definition = load_full_graph(graph_path, default_device="cuda")
sys.stderr.write(f"### TAG={tag} loaded graph, n_inputs={len(inputs)}\n")
with torch.no_grad():
    eager = instance(*inputs)
torch._dynamo.reset()
compiled = torch.compile(instance)
with torch.no_grad():
    comp = compiled(*inputs)
    torch.cuda.synchronize()
def first(x):
    return x[0] if isinstance(x, (tuple, list)) else x
eo, co = first(eager), first(comp)
d = (eo.float() - co.float()).abs()
den = eo.float().abs().clamp_min(1e-3)
sys.stderr.write(
    f"### TAG={tag} NUMERICS eager-vs-compiled: max_abs={d.max().item():.3e} "
    f"mean_abs={d.mean().item():.3e} max_rel={(d/den).max().item():.3e} "
    f"nan_eager={torch.isnan(eo).any().item()} nan_compiled={torch.isnan(co).any().item()} "
    f"shape={tuple(eo.shape)} dtype={co.dtype}\n")
sys.stderr.write(f"### TAG={tag} DONE\n")
