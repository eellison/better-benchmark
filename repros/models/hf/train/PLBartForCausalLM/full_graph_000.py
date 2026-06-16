class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[50005, 768][768, 1]cuda:0", primals_2: "i64[16, 1024][1024, 1]cuda:0"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/plbart/modeling_plbart.py:71 in forward, code: return super().forward(input_ids) * self.embed_scale
        embedding: "f32[16, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.embedding.default(primals_1, primals_2, 1);  primals_1 = None
        mul: "f32[16, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(embedding, 27.712812921102035);  embedding = None
        return (mul, primals_2)
