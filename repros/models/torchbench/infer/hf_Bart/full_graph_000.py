class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "f16[50265, 768]", arg1_1: "i64[1, 512]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:111 in forward, code: return super().forward(input_ids) * self.embed_scale
        embedding: "f16[1, 512, 768]" = torch.ops.aten.embedding.default(arg0_1, arg1_1, 1);  arg0_1 = arg1_1 = None
        mul: "f16[1, 512, 768]" = torch.ops.aten.mul.Tensor(embedding, 1.0);  embedding = None
        return (mul,)
