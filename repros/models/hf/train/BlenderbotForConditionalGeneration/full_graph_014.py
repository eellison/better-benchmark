class GraphModule(torch.nn.Module):
    def forward(self, iota: "i64[128][1]cuda:0", tangents_1: "f32[128, 2560][2560, 1]cuda:0"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:82 in forward, code: return super().forward(position_ids)
        full_default: "b8[128, 1][1, 1]cuda:0" = torch.ops.aten.full.default([128, 1], True, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_1: "f32[128, 2560][2560, 1]cuda:0" = torch.ops.aten.full.default([128, 2560], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        _unsafe_masked_index_put_accumulate: "f32[128, 2560][2560, 1]cuda:0" = torch.ops.aten._unsafe_masked_index_put_accumulate.default(full_default_1, full_default, [iota], tangents_1);  full_default_1 = full_default = iota = tangents_1 = None
        return (_unsafe_masked_index_put_accumulate,)
