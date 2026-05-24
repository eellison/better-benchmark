import torch
from torch import device
from math import inf, nan

class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "f32[128, 64, 1, 1]", arg1_1: "Sym(s16)", arg2_1: "Sym(s82)", arg3_1: "f32[64, 64, s16, s82]"):
        # No stacktrace found for following nodes
        convolution: "f32[64, 128, (((s16 - 1)//2)) + 1, (((s82 - 1)//2)) + 1]" = torch.ops.aten.convolution.default(arg3_1, arg0_1, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  arg3_1 = arg0_1 = None
        return (convolution,)
