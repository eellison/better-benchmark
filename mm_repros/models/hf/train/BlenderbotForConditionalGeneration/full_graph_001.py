class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[128, 2560][2560, 1]cuda:0"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:79 in forward, code: position_ids = torch.arange(
        iota: "i64[128][1]cuda:0" = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:82 in forward, code: return super().forward(position_ids)
        embedding: "f32[128, 2560][2560, 1]cuda:0" = torch.ops.aten.embedding.default(primals_1, iota);  primals_1 = None
        return (embedding, iota)
