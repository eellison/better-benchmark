"""
Standalone repro captured via capture_hook.
Label: hf_PLBartForCausalLM_train
Pattern hash: 5e67603d3f77
Shape hash: 021fba69
"""
_shapes_config = "(T([16, 1024, 768], f32), T([16, 1024], i64))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, tangents_1: "f32[16, 1024, 768]", primals_2: "i64[16, 1024]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/plbart/modeling_plbart.py:71 in forward, code: return super().forward(input_ids) * self.embed_scale
        mul_tensor: "f32[16, 1024, 768]" = torch.ops.aten.mul.Tensor(tangents_1, 27.712812921102035);  tangents_1 = None
        eq_scalar: "b8[16, 1024]" = torch.ops.aten.eq.Scalar(primals_2, 1)
        unsqueeze_default: "b8[16, 1024, 1]" = torch.ops.aten.unsqueeze.default(eq_scalar, -1);  eq_scalar = None
        full_default: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "f32[16, 1024, 768]" = torch.ops.aten.where.self(unsqueeze_default, full_default, mul_tensor);  unsqueeze_default = full_default = mul_tensor = None
        full_default_1: "f32[50005, 768]" = torch.ops.aten.full.default([50005, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default: "f32[50005, 768]" = torch.ops.aten.index_put.default(full_default_1, [primals_2], where_self, True);  full_default_1 = primals_2 = where_self = None
        return index_put_default



def make_inputs():
    return [
    torch.randn([16, 1024, 768], dtype=torch.float32, device='cuda'),
    torch.randint(0, 50005, [16, 1024], dtype=torch.int64, device='cuda'),
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
