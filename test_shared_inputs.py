"""Check what shared inputs cause merges — find the problematic ones."""
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

# Check backward graph
bwd = captured_gms[1]
partitioner = CapabilityBasedPartitioner(
    bwd, support, allows_single_node_partition=True,
    skip_horizontal_fusion=True,
)
partitions = partitioner.propose_partitions()
components = [list(p.nodes.keys()) for p in partitions]

reduction_idxs = [i for i, comp in enumerate(components) if _has_reduction(comp)]
print(f"Backward: {len(components)} partitions, {len(reduction_idxs)} with reductions")

for i in reduction_idxs:
    comp = components[i]
    node_set = set(comp)
    ext_inputs = set()
    for n in comp:
        for inp in n.all_input_nodes:
            if inp not in node_set:
                ext_inputs.add(inp)
    print(f"\n  Partition {i} ({len(comp)} ops):")
    for inp in ext_inputs:
        val = inp.meta.get("val", None)
        if isinstance(val, torch.Tensor):
            print(f"    input: {inp.name} shape={list(val.shape)} numel={val.numel()}")
        else:
            print(f"    input: {inp.name} val_type={type(val).__name__}")

# Now check the HF BERT joint graph (graph 1) which has the 240-op problem
print("\n\n=== Now checking HF BERT ===")
captured_gms.clear()
torch._dynamo.reset()

from transformers import AutoConfig, BertForMaskedLM
config = AutoConfig.from_pretrained("bert-base-uncased")
config.use_cache = False
hf_model = BertForMaskedLM(config).cuda().train()
hf_x = torch.randint(0, 30522, (8, 512), device="cuda")

inductor_config.post_grad_custom_pre_pass = capture_post_grad
compiled_hf = torch.compile(hf_model)
out = compiled_hf(hf_x)
out.logits.sum().backward()
inductor_config.post_grad_custom_pre_pass = None

print(f"Captured {len(captured_gms)} graphs")

# Check the joint graph (graph 1) which is the problematic one
for gi in range(len(captured_gms)):
    gm = captured_gms[gi]
    partitioner = CapabilityBasedPartitioner(
        gm, support, allows_single_node_partition=True,
        skip_horizontal_fusion=True,
    )
    partitions = partitioner.propose_partitions()
    components = [list(p.nodes.keys()) for p in partitions]
    reduction_idxs = [i for i, comp in enumerate(components) if _has_reduction(comp)]

    if len(reduction_idxs) <= 1:
        continue

    print(f"\nGraph {gi}: {len(components)} partitions, {len(reduction_idxs)} with reductions")

    # Find shared inputs between reduction partitions
    def get_ext_inputs(comp):
        node_set = set(comp)
        ext = set()
        for n in comp:
            for inp in n.all_input_nodes:
                if inp not in node_set:
                    ext.add(inp)
        return ext

    for i in range(len(reduction_idxs)):
        for j in range(i + 1, len(reduction_idxs)):
            ci, cj = reduction_idxs[i], reduction_idxs[j]
            shared = get_ext_inputs(components[ci]) & get_ext_inputs(components[cj])
            if shared:
                print(f"  Partitions {ci} ({len(components[ci])} ops) & {cj} ({len(components[cj])} ops) share:")
                for inp in shared:
                    val = inp.meta.get("val", None)
                    if isinstance(val, torch.Tensor):
                        print(f"    {inp.name}: shape={list(val.shape)} numel={val.numel()}")
                    elif isinstance(val, (int, float)):
                        print(f"    {inp.name}: scalar={val}")
                    else:
                        print(f"    {inp.name}: type={type(val).__name__}")
