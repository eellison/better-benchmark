import torch
from torch import device
from math import inf, nan

class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "f16[320, 512]", arg1_1: "f16[1, 4096, 512]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1825 in forward_chunk, code: hidden_states = self.decoder(hidden_states)
        view: "f16[4096, 512]" = torch.ops.aten.reshape.default(arg1_1, [4096, 512]);  arg1_1 = None
        permute: "f16[512, 320]" = torch.ops.aten.permute.default(arg0_1, [1, 0]);  arg0_1 = None
        mm: "f16[4096, 320]" = torch.ops.aten.mm.default(view, permute);  view = permute = None
        view_1: "f16[1, 4096, 320]" = torch.ops.aten.reshape.default(mm, [1, 4096, 320]);  mm = None
        return (view_1,)
