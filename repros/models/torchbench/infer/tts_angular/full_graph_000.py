import torch
from torch import device
from math import inf, nan

class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "f32[64, 50, 256]"):
        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/tts_angular/model.py:75 in torch_dynamo_resume_in_forward_at_73, code: d = torch.nn.functional.normalize(d[:, -1], p=2, dim=1)
        select: "f32[64, 256]" = torch.ops.aten.select.int(arg0_1, 1, -1);  arg0_1 = None
        pow_1: "f32[64, 256]" = torch.ops.aten.pow.Tensor_Scalar(select, 2)
        sum_1: "f32[64, 1]" = torch.ops.aten.sum.dim_IntList(pow_1, [1], True);  pow_1 = None
        pow_2: "f32[64, 1]" = torch.ops.aten.pow.Tensor_Scalar(sum_1, 0.5);  sum_1 = None
        clamp_min: "f32[64, 1]" = torch.ops.aten.clamp_min.default(pow_2, 1e-12);  pow_2 = None
        expand: "f32[64, 256]" = torch.ops.aten.expand.default(clamp_min, [64, 256]);  clamp_min = None
        div: "f32[64, 256]" = torch.ops.aten.div.Tensor(select, expand);  select = expand = None
        return (div,)
