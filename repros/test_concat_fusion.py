"""Trace what the scheduler does with ConcatKernel intermediate."""
import sys
sys.path.insert(0, "/tmp/pytorch-work")
import torch
import torch._inductor.config as cfg
import torch._inductor.metrics as m
cfg.force_disable_caches = True
cfg.max_pointwise_cat_inputs = 0  # Force ConcatKernel

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
    print(f"\nPre-fusion nodes ({len(nodes)}):")
    for node in nodes:
        if isinstance(node, sched_mod.NopKernelSchedulerNode):
            print(f"  NOP: {node.get_name()} (inputs: {[n.name for n in node.node.inputs] if hasattr(node.node, 'inputs') else 'N/A'})")
        elif isinstance(node, sched_mod.SchedulerNode):
            print(f"  SCHED: {node.get_name()}, group={node.group[1]}, sizes={node._sizes}")
            if hasattr(node, '_body'):
                body = node._body
                # Check for masked subblocks
                masked = [k for k in body.subblocks if 'masked' in k]
                if masked:
                    print(f"    HAS MASKED SUBBLOCKS: {masked}")
                print(f"    indexing: {body.indexing_exprs}")
    return nodes

cfg._pre_fusion_custom_pass = my_pre_fusion_pass

torch._dynamo.reset()
compiled = torch.compile(LlamaRoPE().cuda())
with torch.no_grad():
    out = compiled(x, cos, sin)
print(f"\nTotal kernels: {m.generated_kernel_count}")
