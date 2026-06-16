class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[50265, 1024][1024, 1]cuda:0", primals_2: "i64[8, 1024][1024, 1]cuda:0"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:123 in forward, code: return super().forward(input_ids) * self.embed_scale
        embedding: "f32[8, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.embedding.default(primals_1, primals_2, 1);  primals_1 = None
        mul: "f32[8, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(embedding, 1.0);  embedding = None
        return (mul, primals_2)
