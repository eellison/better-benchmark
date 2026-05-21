class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[50265, 768]", primals_2: "i64[4, 512]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:111 in forward, code: return super().forward(input_ids) * self.embed_scale
        embedding: "f32[4, 512, 768]" = torch.ops.aten.embedding.default(primals_1, primals_2, 1);  primals_1 = None
        mul: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(embedding, 1.0);  embedding = None
        return (mul, primals_2)
