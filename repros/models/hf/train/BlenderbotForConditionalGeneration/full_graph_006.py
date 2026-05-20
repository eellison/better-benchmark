class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "i64[128]", primals_2: "f32[128, 2560]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:82 in forward, code: return super().forward(position_ids)
        embedding: "f32[128, 2560]" = torch.ops.aten.embedding.default(primals_2, primals_1);  primals_2 = None
        return (embedding, primals_1)
