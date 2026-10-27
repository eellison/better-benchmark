class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "i64[128][1]cuda:0", arg1_1: "f32[1024, 1024][1024, 1]cuda:0"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:105 in forward, code: return super().forward(position_ids)
        embedding: "f32[128, 1024][1024, 1]cuda:0" = torch.ops.aten.embedding.default(arg1_1, arg0_1);  arg1_1 = arg0_1 = None
        return (embedding,)
