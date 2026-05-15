"""Check if inner_fn is still accessible from scheduler."""
import sys
sys.path.insert(0, "/tmp/pytorch-work")
import torch
import torch._inductor.config as cfg
import torch._inductor.ir as ir
cfg.force_disable_caches = True
cfg.prefer_concat_kernel_shared_reads = False

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

from torch._inductor import scheduler as sched_mod

def my_pre_fusion_pass(nodes):
    for node in nodes:
        if not isinstance(node, sched_mod.SchedulerNode):
            continue
        ir_node = node.node  # ComputedBuffer
        print(f"IR node type: {type(ir_node)}")
        if hasattr(ir_node, 'data'):
            pw = ir_node.data  # Pointwise
            print(f"Data type: {type(pw)}")
            print(f"Has inner_fn: {hasattr(pw, 'inner_fn')}")
            print(f"inner_fn: {pw.inner_fn}")
            print(f"ranges: {pw.ranges}")
    return nodes

cfg._pre_fusion_custom_pass = my_pre_fusion_pass

torch._dynamo.reset()
compiled = torch.compile(LlamaRoPE().cuda())
with torch.no_grad():
    out = compiled(x, cos, sin)
