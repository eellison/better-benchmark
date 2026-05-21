"""
Standalone repro captured via capture_hook.
Label: hf_OPTForCausalLM_train
Pattern hash: 693e171c3299
Shape hash: d3f3ead3
"""
_shapes_config = "(T([4, 2048], i64), T([4, 2048, 768], f32))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, add: "i64[4, 2048]", tangents_1: "f32[4, 2048, 768]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:70 in forward, code: return super().forward(position_ids + self.offset)
        eq_scalar: "b8[4, 2048]" = torch.ops.aten.eq.Scalar(add, -1)
        unsqueeze_default: "b8[4, 2048, 1]" = torch.ops.aten.unsqueeze.default(eq_scalar, -1);  eq_scalar = None
        full_default: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "f32[4, 2048, 768]" = torch.ops.aten.where.self(unsqueeze_default, full_default, tangents_1);  unsqueeze_default = full_default = tangents_1 = None
        full_default_1: "f32[2050, 768]" = torch.ops.aten.full.default([2050, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default: "f32[2050, 768]" = torch.ops.aten.index_put.default(full_default_1, [add], where_self, True);  full_default_1 = add = where_self = None
        return index_put_default



def make_inputs():
    return [
    torch.randint(0, 2050, [4, 2048], dtype=torch.int64, device='cuda'),
    torch.randn([4, 2048, 768], dtype=torch.float32, device='cuda'),
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
