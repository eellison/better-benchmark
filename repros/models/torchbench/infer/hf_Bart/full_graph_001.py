import torch
from torch import device
from math import inf, nan

class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "f16[1026, 768]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:92 in forward, code: position_ids = torch.arange(
        iota: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:94 in forward, code: ).expand(bsz, -1)
        expand: "i64[1, 512]" = torch.ops.aten.expand.default(iota, [1, -1]);  iota = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:98 in forward, code: return super().forward(position_ids + self.offset)
        add: "i64[1, 512]" = torch.ops.aten.add.Tensor(expand, 2);  expand = None
        embedding: "f16[1, 512, 768]" = torch.ops.aten.embedding.default(arg0_1, add);  arg0_1 = add = None
        return (embedding,)
