"""Look at LoopBody of a ConcatKernel fusion (2-store kernel) to understand structure."""
import sys
sys.path.insert(0, "/tmp/pytorch-work")
import torch
import torch._inductor.config as cfg
cfg.force_disable_caches = True
cfg.prefer_concat_kernel_shared_reads = True  # Enable to get 2-store kernel

B, S, H, D = 4, 512, 32, 128
D_HALF = D // 2

# This produces the optimal 2-store kernel via ConcatKernel
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

def my_post_fusion_pass(nodes):
    print(f"\nPost-fusion nodes ({len(nodes)}):")
    for node in nodes:
        if isinstance(node, sched_mod.FusedSchedulerNode):
            print(f"  FUSED: {node.get_name()}, snodes={len(node.snodes)}")
            for sn in node.snodes:
                if hasattr(sn, '_body'):
                    print(f"    sub: {sn.get_name()}, sizes={sn._sizes}")
                    body = sn._body
                    print(f"    graph: {body.root_block.graph}")
                    print(f"    indexing: {body.indexing_exprs}")
        elif isinstance(node, sched_mod.SchedulerNode):
            if hasattr(node, '_body'):
                print(f"  SCHED: {node.get_name()}, sizes={node._sizes}")
                body = node._body
                print(f"    graph: {body.root_block.graph}")
                print(f"    indexing: {body.indexing_exprs}")
    return nodes

cfg._post_fusion_custom_pass = my_post_fusion_pass

torch._dynamo.reset()
compiled = torch.compile(TwoOutputRoPE().cuda())
with torch.no_grad():
    out = compiled(x, cos, sin)
