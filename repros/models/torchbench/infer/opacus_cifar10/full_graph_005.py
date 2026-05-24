import torch
from torch import device
from math import inf, nan

class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "Sym(s89)", arg1_1: "Sym(s39)", arg2_1: "f32[64, 64, s89, s39]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:94 in torch_dynamo_resume_in_forward_at_93, code: out = self.relu(out)
        relu: "f32[64, 64, s89, s39]" = torch.ops.aten.relu.default(arg2_1)
        copy_: "f32[64, 64, s89, s39]" = torch.ops.aten.copy_.default(arg2_1, relu);  arg2_1 = relu = None
        return (copy_,)
