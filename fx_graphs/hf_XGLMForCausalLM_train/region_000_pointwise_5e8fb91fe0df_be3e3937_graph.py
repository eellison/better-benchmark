class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[256008, 1024]", primals_2: "i64[32, 128]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:50 in forward, code: return super().forward(input_ids) * self.embed_scale
        embedding_default: "f32[32, 128, 1024]" = torch.ops.aten.embedding.default(primals_1, primals_2, 1);  primals_1 = primals_2 = None
        mul_tensor: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(embedding_default, 32.0);  embedding_default = None
        return mul_tensor
