import torch
from torch import device
from math import inf, nan

class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "i64[1024]", primals_2: "f32[1026, 1024]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:96 in forward, code: position_ids = position_ids.unsqueeze(0)
        unsqueeze: "i64[1, 1024]" = torch.ops.aten.unsqueeze.default(primals_1, 0);  primals_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:98 in forward, code: return super().forward(position_ids + self.offset)
        add: "i64[1, 1024]" = torch.ops.aten.add.Tensor(unsqueeze, 2);  unsqueeze = None
        embedding: "f32[1, 1024, 1024]" = torch.ops.aten.embedding.default(primals_2, add);  primals_2 = None
        return (embedding, add)
