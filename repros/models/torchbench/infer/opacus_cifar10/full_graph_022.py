import torch
from torch import device
from math import inf, nan

class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "Sym(s49)", arg1_1: "f32[64, s49, 1, 1]", arg2_1: "f32[64, s49, 1, 1]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:102 in torch_dynamo_resume_in_forward_at_97, code: out += identity
        add_10: "f32[64, s49, 1, 1]" = torch.ops.aten.add.Tensor(arg1_1, arg2_1);  arg2_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:103 in torch_dynamo_resume_in_forward_at_97, code: out = self.relu(out)
        relu: "f32[64, s49, 1, 1]" = torch.ops.aten.relu.default(add_10);  add_10 = None
        copy_: "f32[64, s49, 1, 1]" = torch.ops.aten.copy_.default(arg1_1, relu);  arg1_1 = relu = None
        return (copy_,)
