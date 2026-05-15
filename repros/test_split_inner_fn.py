"""Test: can we call inner_fn with constrained indices to eliminate branches?"""
import sys
sys.path.insert(0, "/tmp/pytorch-work")
import torch
import torch._inductor.config as cfg
import torch._inductor.ir as ir
import sympy
from torch._inductor.virtualized import V
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

captured_inner_fn = [None]
captured_ranges = [None]

def my_pre_fusion_pass(nodes):
    for node in nodes:
        if not isinstance(node, sched_mod.SchedulerNode):
            continue
        ir_node = node.node
        if hasattr(ir_node, 'data') and hasattr(ir_node.data, 'inner_fn'):
            captured_inner_fn[0] = ir_node.data.inner_fn
            captured_ranges[0] = ir_node.data.ranges
    return nodes

cfg._pre_fusion_custom_pass = my_pre_fusion_pass

torch._dynamo.reset()
compiled = torch.compile(LlamaRoPE().cuda())
with torch.no_grad():
    out = compiled(x, cos, sin)

# Now let's call the inner_fn with a test to see what it produces
inner_fn = captured_inner_fn[0]
ranges = captured_ranges[0]
print(f"Ranges: {ranges}")

# The inner_fn uses ops.* calls. Let's trace it with KernelFormatterHandler
from torch._inductor.virtualized import V
result_str = V.KernelFormatterHandler.ir_to_string(inner_fn, [sympy.Symbol(f'i{k}') for k in range(4)])
print(f"\nFull inner_fn (i3 in [0,128)):\n{result_str}")

# Now let's try with i3 replaced by something < 64
# The inner_fn closure captures the loaders, which don't know about ranges.
# We need to check: does the inner_fn produce different code when i3 is provably < 64?
# Answer: No, because it uses ops.lt/ops.where which are symbolic at trace time.

# But what if we create a WRAPPER inner_fn that only evaluates one branch?
def make_split_fn(original_fn, split_dim, split_point, which_half):
    """Create an inner_fn for one half of a pointwise_cat split.
    
    which_half=0: evaluates the 'true' branch (idx[split_dim] < split_point)
    which_half=1: evaluates the 'false' branch (idx[split_dim] >= split_point)
    """
    def split_fn(idx):
        new_idx = list(idx)
        if which_half == 1:
            # Offset the inner dim by split_point
            new_idx[split_dim] = idx[split_dim] + split_point
        return original_fn(new_idx)
    return split_fn

# The problem: even with offset indices, the inner_fn still generates ops.where
# We need the ops handler to constant-fold the comparison.
# Let's check what happens:
split_fn_lo = make_split_fn(inner_fn, 3, 64, 0)
split_fn_hi = make_split_fn(inner_fn, 3, 64, 1)

result_lo = V.KernelFormatterHandler.ir_to_string(split_fn_lo, [sympy.Symbol(f'i{k}') for k in range(4)])
result_hi = V.KernelFormatterHandler.ir_to_string(split_fn_hi, [sympy.Symbol(f'i{k}') for k in range(4)])
print(f"\nLow half inner_fn (i3 in [0,64)):\n{result_lo}")
print(f"\nHigh half inner_fn (i3+64 in [64,128)):\n{result_hi}")
