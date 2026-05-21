class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "i64[4, 512]", arg1_1: "f16[2050, 768]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:70 in forward, code: return super().forward(position_ids + self.offset)
        add: "i64[4, 512]" = torch.ops.aten.add.Tensor(arg0_1, 2);  arg0_1 = None
        embedding: "f16[4, 512, 768]" = torch.ops.aten.embedding.default(arg1_1, add);  arg1_1 = add = None
        return (embedding,)
