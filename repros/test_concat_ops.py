"""Check what operations are created when ConcatKernel is used for intermediate."""
import sys
sys.path.insert(0, "/tmp/pytorch-work")
import torch
import torch._inductor.config as cfg
import torch._inductor.ir as ir
cfg.force_disable_caches = True
cfg.max_pointwise_cat_inputs = 0

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

# Monkey-patch to see what operations get created
orig_update = None
from torch._inductor.graph import GraphLowering

original_update_scheduler = GraphLowering._update_scheduler
def patched_update_scheduler(self):
    print(f"\nOperations before scheduling ({len(self.operations)}):")
    for i, op in enumerate(self.operations):
        print(f"  [{i}] {type(op).__name__}: {op.get_name()}")
        if hasattr(op, 'data') and hasattr(op.data, 'ranges'):
            print(f"       ranges={op.data.ranges}")
        if hasattr(op, 'inputs'):
            print(f"       inputs={[getattr(inp, 'name', str(inp)[:30]) for inp in op.inputs]}")
    original_update_scheduler(self)

GraphLowering._update_scheduler = patched_update_scheduler

torch._dynamo.reset()
compiled = torch.compile(LlamaRoPE().cuda())
with torch.no_grad():
    out = compiled(x, cos, sin)
