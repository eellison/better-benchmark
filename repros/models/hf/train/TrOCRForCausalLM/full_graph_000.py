class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[50265, 1024][1024, 1]cuda:0", primals_2: "i64[64, 256][256, 1]cuda:0"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:75 in forward, code: return super().forward(input_ids) * self.embed_scale
        embedding: "f32[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.embedding.default(primals_1, primals_2, 1);  primals_1 = None
        mul: "f32[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(embedding, 1.0);  embedding = None
        return (mul, primals_2)
