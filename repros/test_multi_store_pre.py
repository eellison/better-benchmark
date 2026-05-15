"""Look at PRE-fusion nodes for 2-store ConcatKernel."""
import sys
sys.path.insert(0, "/tmp/pytorch-work")
import torch
import torch._inductor.config as cfg
cfg.force_disable_caches = True
cfg.prefer_concat_kernel_shared_reads = True

B, S, H, D = 4, 512, 32, 128
D_HALF = D // 2

class TwoOutputRoPE(torch.nn.Module):
    def forward(self, x, cos, sin):
        cos_e = cos.unsqueeze(0).unsqueeze(0)
        sin_e = sin.unsqueeze(0).unsqueeze(0)
        x1 = x[..., :D_HALF]
        x2 = x[..., D_HALF:]
        o1 = x1 * cos_e - x2 * sin_e
        o2 = x1 * sin_e + x2 * cos_e
        return torch.cat([o1, o2], dim=-1)

x = torch.randn(B, H, S, D, dtype=torch.bfloat16, device='cuda')
cos = torch.randn(S, D_HALF, dtype=torch.bfloat16, device='cuda')
sin = torch.randn(S, D_HALF, dtype=torch.bfloat16, device='cuda')

from torch._inductor import scheduler as sched_mod

def my_pre_fusion_pass(nodes):
    print(f"\nPre-fusion nodes ({len(nodes)}):")
    for node in nodes:
        if isinstance(node, sched_mod.NopKernelSchedulerNode):
            print(f"  NOP: {node.get_name()}")
        elif isinstance(node, sched_mod.SchedulerNode):
            print(f"  SCHED: {node.get_name()}, group={node.group[1]}, sizes={node._sizes}")
            if hasattr(node, '_body'):
                body = node._body
                print(f"    store: {[n.args[1] for n in body.root_block.graph.nodes if n.target == 'store']}")
                print(f"    indexing: {body.indexing_exprs}")
    return nodes

cfg._pre_fusion_custom_pass = my_pre_fusion_pass

torch._dynamo.reset()
compiled = torch.compile(TwoOutputRoPE().cuda())
with torch.no_grad():
    out = compiled(x, cos, sin)
