import torch
from torch import device
from math import inf, nan

class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[128112, 1024]", primals_2: "i64[64, 128]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:77 in forward, code: return super().forward(input_ids) * self.embed_scale
        embedding: "f32[64, 128, 1024]" = torch.ops.aten.embedding.default(primals_1, primals_2, 1);  primals_1 = None
        mul: "f32[64, 128, 1024]" = torch.ops.aten.mul.Tensor(embedding, 32.0);  embedding = None
        return (mul, primals_2)
