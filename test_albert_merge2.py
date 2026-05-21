"""Trace exactly which shared inputs cause the ALBERT mega-merges."""
import copy
import collections
import torch
import torch.fx as fx
import torch._inductor.config as inductor_config

inductor_config.force_disable_caches = True
inductor_config.split_reductions = False

from extract_reductions import _has_reduction, _merge_shared_input_reductions

bwd_gms = []

def capture(g):
    if isinstance(g, fx.Graph) and hasattr(g, 'owning_module') and g.owning_module:
        bwd_gms.append(copy.deepcopy(g.owning_module))
    return g

from transformers import AutoConfig, AlbertForMaskedLM
config = AutoConfig.from_pretrained("albert-base-v2")
config.use_cache = False

torch._dynamo.reset()
inductor_config.post_grad_custom_pre_pass = capture
model = AlbertForMaskedLM(config).cuda().train()
x = torch.randint(0, 30000, (8, 512), device="cuda")
compiled = torch.compile(model)
out = compiled(x)
out.logits.sum().backward()
inductor_config.post_grad_custom_pre_pass = None

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
print(f"Before merge: {len(components)} partitions, {len(reduction_idxs)} with reductions")
for i in reduction_idxs:
    print(f"  partition {i}: {len(components[i])} ops")

merged = _merge_shared_input_reductions(components)
red_merged = [c for c in merged if _has_reduction(c)]
print(f"\nAfter merge: {len(merged)} partitions, {len(red_merged)} with reductions")
for i, c in enumerate(merged):
    if _has_reduction(c):
        print(f"  partition {i}: {len(c)} ops")

# Now trace WHY the big ones merged — which shared activation inputs?
print("\n=== Tracing merge causes ===")

def get_ext_inputs(comp):
    node_set = set(comp)
    return {inp for n in comp for inp in n.all_input_nodes if inp not in node_set}

for i in range(len(reduction_idxs)):
    for j in range(i + 1, len(reduction_idxs)):
        ci, cj = reduction_idxs[i], reduction_idxs[j]
        shared = get_ext_inputs(components[ci]) & get_ext_inputs(components[cj])
        act_shared = []
        for inp in shared:
            val = inp.meta.get("val", None)
            if isinstance(val, torch.Tensor) and val.dim() >= 2 and val.numel() >= 64:
                act_shared.append(inp)
        if act_shared:
            sizes_i = len(components[ci])
            sizes_j = len(components[cj])
            for inp in act_shared:
                val = inp.meta.get("val", None)
                print(f"  partitions {ci}({sizes_i} ops) + {cj}({sizes_j} ops) share: "
                      f"{inp.name} shape={list(val.shape)} op={inp.op}")
