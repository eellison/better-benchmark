class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[256008, 1024][1024, 1]cuda:0", primals_2: "i64[32, 128][128, 1]cuda:0"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:50 in forward, code: return super().forward(input_ids) * self.embed_scale
        embedding: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.embedding.default(primals_1, primals_2, 1);  primals_1 = None
        mul: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(embedding, 32.0);  embedding = None
        return (mul, primals_2)
