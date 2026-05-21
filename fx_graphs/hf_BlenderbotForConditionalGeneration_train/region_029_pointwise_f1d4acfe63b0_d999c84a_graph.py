class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "i64[128]", tangents_1: "f32[128, 2560]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:82 in forward, code: return super().forward(position_ids)
        eq_scalar: "b8[128]" = torch.ops.aten.eq.Scalar(primals_1, -1)
        unsqueeze_default: "b8[128, 1]" = torch.ops.aten.unsqueeze.default(eq_scalar, -1);  eq_scalar = None
        full_default: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "f32[128, 2560]" = torch.ops.aten.where.self(unsqueeze_default, full_default, tangents_1);  unsqueeze_default = full_default = tangents_1 = None
        full_default_1: "f32[128, 2560]" = torch.ops.aten.full.default([128, 2560], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default: "f32[128, 2560]" = torch.ops.aten.index_put.default(full_default_1, [primals_1], where_self, True);  full_default_1 = primals_1 = where_self = None
        return index_put_default
