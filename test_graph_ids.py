"""Quick check: what are the 3 captured graphs for BERT?"""
import copy
import torch
import torch.fx as fx
import torch._inductor.config as inductor_config

inductor_config.force_disable_caches = True
inductor_config.split_reductions = False

captured_gms = []

def capture_post_grad(graph_or_gm):
    if isinstance(graph_or_gm, fx.GraphModule):
        captured_gms.append(copy.deepcopy(graph_or_gm))
    elif isinstance(graph_or_gm, fx.Graph):
        if hasattr(graph_or_gm, 'owning_module') and graph_or_gm.owning_module is not None:
            captured_gms.append(copy.deepcopy(graph_or_gm.owning_module))
    return graph_or_gm

from transformers import AutoConfig, BertForMaskedLM
config = AutoConfig.from_pretrained("bert-base-uncased")
config.use_cache = False

# Forward-only pass (no_grad)
print("=== Forward-only (no_grad) ===")
torch._dynamo.reset()
inductor_config.post_grad_custom_pre_pass = capture_post_grad
model = BertForMaskedLM(config).cuda().eval()
x = torch.randint(0, 30522, (8, 512), device="cuda")
compiled = torch.compile(model)
with torch.no_grad():
    compiled(x)
fwd_count = len(captured_gms)
print(f"  Captured {fwd_count} graphs")

# Training pass (fwd + bwd)
print("\n=== Training (fwd + bwd) ===")
torch._dynamo.reset()
inductor_config.post_grad_custom_pre_pass = capture_post_grad
model2 = BertForMaskedLM(config).cuda().train()
x2 = torch.randint(0, 30522, (8, 512), device="cuda")
compiled2 = torch.compile(model2)
out = compiled2(x2)
out.logits.sum().backward()
train_count = len(captured_gms) - fwd_count
print(f"  Captured {train_count} graphs")

inductor_config.post_grad_custom_pre_pass = None

print(f"\nTotal: {len(captured_gms)} graphs")
for i, gm in enumerate(captured_gms):
    n_ph = sum(1 for n in gm.graph.nodes if n.op == "placeholder")
    n_call = sum(1 for n in gm.graph.nodes if n.op == "call_function")
    n_out = 0
    for n in gm.graph.nodes:
        if n.op == "output":
            args = n.args[0] if isinstance(n.args[0], (tuple, list)) else [n.args[0]]
            n_out = len(args)
    source = "fwd-only" if i < fwd_count else "training"
    print(f"  Graph {i} ({source}): {n_ph} placeholders, {n_call} call_function, {n_out} outputs")
