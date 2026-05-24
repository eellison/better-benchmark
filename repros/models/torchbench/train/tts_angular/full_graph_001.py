import torch
from torch import device
from math import inf, nan

class GraphModule(torch.nn.Module):
    def forward(self, select: "f32[64, 256]", pow_2: "f32[64, 1]", tangents_1: "f32[64, 256]"):
        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/tts_angular/model.py:75 in torch_dynamo_resume_in_forward_at_73, code: d = torch.nn.functional.normalize(d[:, -1], p=2, dim=1)
        neg: "f32[64, 256]" = torch.ops.aten.neg.default(tangents_1)
        clamp_min: "f32[64, 1]" = torch.ops.aten.clamp_min.default(pow_2, 1e-12)
        expand: "f32[64, 256]" = torch.ops.aten.expand.default(clamp_min, [64, 256]);  clamp_min = None
        div_1: "f32[64, 256]" = torch.ops.aten.div.Tensor(select, expand)
        div_2: "f32[64, 256]" = torch.ops.aten.div.Tensor(div_1, expand);  div_1 = None
        mul: "f32[64, 256]" = torch.ops.aten.mul.Tensor(neg, div_2);  neg = div_2 = None
        div_3: "f32[64, 256]" = torch.ops.aten.div.Tensor(tangents_1, expand);  tangents_1 = expand = None
        sum_2: "f32[64, 1]" = torch.ops.aten.sum.dim_IntList(mul, [1], True);  mul = None
        full_default: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        ge: "b8[64, 1]" = torch.ops.aten.ge.Scalar(pow_2, 1e-12)
        where: "f32[64, 1]" = torch.ops.aten.where.self(ge, sum_2, full_default);  ge = sum_2 = None
        div_4: "f32[64, 256]" = torch.ops.aten.div.Tensor(select, pow_2);  select = None
        eq: "b8[64, 1]" = torch.ops.aten.eq.Scalar(pow_2, 0);  pow_2 = None
        where_1: "f32[64, 256]" = torch.ops.aten.where.self(eq, full_default, div_4);  eq = full_default = div_4 = None
        mul_1: "f32[64, 256]" = torch.ops.aten.mul.Tensor(where, where_1);  where = where_1 = None
        add: "f32[64, 256]" = torch.ops.aten.add.Tensor(div_3, mul_1);  div_3 = mul_1 = None
        full_default_2: "f32[64, 50, 256]" = torch.ops.aten.full.default([64, 50, 256], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        select_scatter: "f32[64, 50, 256]" = torch.ops.aten.select_scatter.default(full_default_2, add, 1, -1);  full_default_2 = add = None
        return (select_scatter,)
