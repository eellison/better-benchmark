"""Test that shared-input merging gives LayerNorm backward as one partition."""
import copy
import collections
import torch
import torch.fx as fx
import torch._inductor.config as inductor_config

inductor_config.force_disable_caches = True
inductor_config.split_reductions = False

# Import the merge function from our extraction script
import sys
sys.path.insert(0, "/tmp/scratch_space/better_benchmark")
from extract_reductions import _has_reduction, _merge_shared_input_reductions

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

from torch._inductor.fx_passes.fusion_regions import is_fusible_node
from torch.fx.passes.infra.partitioner import CapabilityBasedPartitioner
from torch.fx.passes.operator_support import create_op_support

def _is_supported(_submodules, node):
    return is_fusible_node(node)

support = create_op_support(_is_supported)

# Test backward graph
bwd = captured_gms[1]

partitioner = CapabilityBasedPartitioner(
    bwd, support, allows_single_node_partition=True,
    skip_horizontal_fusion=True,
)
partitions = partitioner.propose_partitions()
components = [list(p.nodes.keys()) for p in partitions]

print(f"Before merge: {len(components)} partitions")
for i, comp in enumerate(components):
    n_ops = sum(1 for n in comp if n.op == "call_function")
    reds = sum(1 for n in comp if n.op == "call_function" and isinstance(n.target, torch._ops.OpOverload) and torch.Tag.reduction in n.target.tags)
    print(f"  Partition {i}: {n_ops} ops, {reds} reductions")

merged = _merge_shared_input_reductions(components)

print(f"\nAfter merge: {len(merged)} partitions")
for i, comp in enumerate(merged):
    n_ops = sum(1 for n in comp if n.op == "call_function")
    reds = []
    for n in comp:
        if n.op == "call_function" and isinstance(n.target, torch._ops.OpOverload):
            if torch.Tag.reduction in n.target.tags:
                val = n.meta.get("val", None)
                shape = list(val.shape) if isinstance(val, torch.Tensor) else "?"
                reds.append(f"{n.target.overloadpacket.__name__}->{shape}")
    has_red = " [HAS REDUCTION]" if reds else ""
    print(f"  Partition {i}: {n_ops} ops{has_red}")
    for r in reds:
        print(f"    {r}")

# Verify: backward should be exactly 1 partition with 4 reductions
reduction_partitions = [comp for comp in merged if _has_reduction(comp)]
assert len(reduction_partitions) == 1, f"Expected 1 reduction partition, got {len(reduction_partitions)}"
n_reds = sum(
    1 for n in reduction_partitions[0]
    if n.op == "call_function" and isinstance(n.target, torch._ops.OpOverload) and torch.Tag.reduction in n.target.tags
)
assert n_reds == 4, f"Expected 4 reductions in merged partition, got {n_reds}"
print("\nPASS: LayerNorm backward is one partition with 4 reductions")
