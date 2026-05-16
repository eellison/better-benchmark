"""Check what shared inputs are causing ALBERT's massive merges."""
import copy
import collections
import torch
import torch.fx as fx
import torch._inductor.config as inductor_config

inductor_config.force_disable_caches = True
inductor_config.split_reductions = False

from extract_reductions import _has_reduction

captured_gms = []

def capture_post_grad(graph_or_gm):
    if isinstance(graph_or_gm, fx.GraphModule):
        captured_gms.append(copy.deepcopy(graph_or_gm))
    elif isinstance(graph_or_gm, fx.Graph):
        if hasattr(graph_or_gm, 'owning_module') and graph_or_gm.owning_module is not None:
            captured_gms.append(copy.deepcopy(graph_or_gm.owning_module))
    return graph_or_gm

from transformers import AutoConfig, AlbertForMaskedLM
config = AutoConfig.from_pretrained("albert-base-v2")
config.use_cache = False

# Bwd only
bwd_gms = []
torch._dynamo.reset()
inductor_config.post_grad_custom_pre_pass = lambda g: (bwd_gms.append(copy.deepcopy(g.owning_module)) if isinstance(g, fx.Graph) and hasattr(g, 'owning_module') and g.owning_module else None) or g

model = AlbertForMaskedLM(config).cuda().train()
x = torch.randint(0, 30000, (8, 512), device="cuda")
compiled = torch.compile(model)
out = compiled(x)
out.logits.sum().backward()

inductor_config.post_grad_custom_pre_pass = None

# Use the last graph (bwd)
bwd = bwd_gms[-1]

from torch._inductor.fx_passes.fusion_regions import is_fusible_node
from torch.fx.passes.infra.partitioner import CapabilityBasedPartitioner
from torch.fx.passes.operator_support import create_op_support

support = create_op_support(lambda _s, n: is_fusible_node(n))
partitioner = CapabilityBasedPartitioner(
    bwd, support, allows_single_node_partition=True,
    skip_horizontal_fusion=True,
)
partitions = partitioner.propose_partitions()
components = [list(p.nodes.keys()) for p in partitions]

reduction_idxs = [i for i, c in enumerate(components) if _has_reduction(c)]
print(f"{len(components)} partitions, {len(reduction_idxs)} with reductions")

def get_ext_inputs(comp):
    node_set = set(comp)
    ext = set()
    for n in comp:
        for inp in n.all_input_nodes:
            if inp not in node_set:
                ext.add(inp)
    return ext

# Find shared inputs between reduction partitions
merge_groups = collections.defaultdict(list)
for i in reduction_idxs:
    for j in reduction_idxs:
        if i >= j:
            continue
        shared = get_ext_inputs(components[i]) & get_ext_inputs(components[j])
        sig_shared = []
        for inp in shared:
            val = inp.meta.get("val", None)
            if isinstance(val, torch.Tensor) and val.numel() >= 64:
                sig_shared.append(inp)
        if sig_shared:
            for inp in sig_shared:
                val = inp.meta.get("val", None)
                shape = list(val.shape) if isinstance(val, torch.Tensor) else "?"
                merge_groups[(inp.name, str(shape))].append((i, j, len(components[i]), len(components[j])))

print(f"\nShared inputs causing merges:")
for (name, shape), pairs in sorted(merge_groups.items(), key=lambda x: -len(x[1])):
    print(f"  {name} shape={shape}: merges {len(pairs)} partition pairs")
    # check if it's a placeholder
    for inp_node in bwd.graph.nodes:
        if inp_node.name == name and inp_node.op == "placeholder":
            val = inp_node.meta.get("val", None)
            print(f"    -> placeholder, numel={val.numel() if isinstance(val, torch.Tensor) else '?'}")
            break
