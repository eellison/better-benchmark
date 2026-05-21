class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "i64[4, 2048]", primals_2: "f32[2050, 768]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:70 in forward, code: return super().forward(position_ids + self.offset)
        add_tensor: "i64[4, 2048]" = torch.ops.aten.add.Tensor(primals_1, 2);  primals_1 = None
        embedding_default: "f32[4, 2048, 768]" = torch.ops.aten.embedding.default(primals_2, add_tensor);  primals_2 = add_tensor = None
        return embedding_default
