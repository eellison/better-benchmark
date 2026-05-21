class GraphModule(torch.nn.Module):
    def forward(self, primals_2: "f32[128, 2560]", primals_1: "i64[128]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:82 in forward, code: return super().forward(position_ids)
        embedding_default: "f32[128, 2560]" = torch.ops.aten.embedding.default(primals_2, primals_1);  primals_2 = primals_1 = None
        return embedding_default
