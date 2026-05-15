"""Inspect the LoopBody FX graph for a pointwise_cat consumer to understand structure."""
import sys
sys.path.insert(0, "/tmp/pytorch-work")
import torch
import torch._inductor.config as cfg
cfg.force_disable_caches = True
cfg.prefer_concat_kernel_shared_reads = False  # Force pointwise_cat

B, S, H, D = 4, 512, 32, 128
D_HALF = D // 2

class LlamaRoPE(torch.nn.Module):
    def forward(self, x, cos, sin):
        cos_e = cos.unsqueeze(0).unsqueeze(0)
        sin_e = sin.unsqueeze(0).unsqueeze(0)
        x2 = x[..., D_HALF:]
        x1 = x[..., :D_HALF]
        rotated = torch.cat([-x2, x1], dim=-1)
        return x * cos_e + rotated * sin_e

x = torch.randn(B, H, S, D, dtype=torch.bfloat16, device='cuda')
cos = torch.randn(S, D, dtype=torch.bfloat16, device='cuda')
sin = torch.randn(S, D, dtype=torch.bfloat16, device='cuda')

# Hook into Scheduler to inspect the SchedulerNodes
original_init = None
from torch._inductor import scheduler as sched_mod

class SchedulerHook:
    @staticmethod
    def inspect(nodes):
        for node in nodes:
            if not isinstance(node, sched_mod.SchedulerNode):
                continue
            if not hasattr(node, '_body'):
                continue
            body = node._body
            print(f"\n{'='*60}")
            print(f"Node: {node.get_name()}")
            print(f"Group: {node.group}")
            print(f"Sizes: {node._sizes}")
            print(f"Body graph:")
            print(body.root_block.graph)
            print(f"Indexing exprs: {body.indexing_exprs}")
            # Look for masked/where patterns
            for n in body.root_block.graph.nodes:
                if 'masked' in str(n.target) or 'where' in str(n.target):
                    print(f"  FOUND: {n.target} args={n.args}")

original_pre_fusion = cfg._pre_fusion_custom_pass
def my_pre_fusion_pass(nodes):
    SchedulerHook.inspect(nodes)
    if original_pre_fusion:
        return original_pre_fusion(nodes)
    return nodes

cfg._pre_fusion_custom_pass = my_pre_fusion_pass

torch._dynamo.reset()
compiled = torch.compile(LlamaRoPE().cuda())
with torch.no_grad():
    out = compiled(x, cos, sin)
