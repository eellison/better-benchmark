"""Print the backward graph to see the exact node connectivity."""
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

class LNModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ln = torch.nn.LayerNorm(768)

    def forward(self, x):
        return self.ln(x)

model = LNModel().cuda()
x = torch.randn(8, 512, 768, device="cuda", requires_grad=True)

torch._dynamo.reset()
inductor_config.post_grad_custom_pre_pass = capture_post_grad

compiled = torch.compile(model)
out = compiled(x)
out.sum().backward()

inductor_config.post_grad_custom_pre_pass = None

# Print backward graph
bwd = captured_gms[1]
print("=== LayerNorm backward graph ===")
for node in bwd.graph.nodes:
    if node.op == "placeholder":
        val = node.meta.get("val", None)
        shape = list(val.shape) if isinstance(val, torch.Tensor) else "?"
        print(f"  PLACEHOLDER {node.name}: {shape}")
    elif node.op == "call_function":
        inputs = [a.name if isinstance(a, fx.Node) else str(a) for a in node.args]
        val = node.meta.get("val", None)
        shape = list(val.shape) if isinstance(val, torch.Tensor) else "?"
        is_red = ""
        if isinstance(node.target, torch._ops.OpOverload) and torch.Tag.reduction in node.target.tags:
            is_red = " [REDUCTION]"
        print(f"  {node.name} = {node.target}({', '.join(inputs)}) -> {shape}{is_red}")
    elif node.op == "output":
        outs = node.args[0] if isinstance(node.args[0], tuple) else node.args
        names = [a.name if isinstance(a, fx.Node) else str(a) for a in outs]
        print(f"  OUTPUT: ({', '.join(names)})")
