class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "i64[4, 2048][2048, 1]cuda:0", primals_2: "f32[2050, 768][768, 1]cuda:0"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:70 in forward, code: return super().forward(position_ids + self.offset)
        add: "i64[4, 2048][2048, 1]cuda:0" = torch.ops.aten.add.Tensor(primals_1, 2);  primals_1 = None
        embedding: "f32[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.embedding.default(primals_2, add);  primals_2 = None
        return (embedding, add)
