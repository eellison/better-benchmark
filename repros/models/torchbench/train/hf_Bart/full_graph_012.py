import torch
from torch import device
from math import inf, nan

class GraphModule(torch.nn.Module):
    def forward(self, add: "i64[1, 512]", tangents_1: "f32[1, 512, 768]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:98 in forward, code: return super().forward(position_ids + self.offset)
        eq: "b8[1, 512]" = torch.ops.aten.eq.Scalar(add, -1)
        unsqueeze_1: "b8[1, 512, 1]" = torch.ops.aten.unsqueeze.default(eq, -1);  eq = None
        full_default: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "f32[1, 512, 768]" = torch.ops.aten.where.self(unsqueeze_1, full_default, tangents_1);  unsqueeze_1 = full_default = tangents_1 = None
        full_default_1: "f32[1026, 768]" = torch.ops.aten.full.default([1026, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put: "f32[1026, 768]" = torch.ops.aten.index_put.default(full_default_1, [add], where, True);  full_default_1 = add = where = None
        return (None, index_put)
