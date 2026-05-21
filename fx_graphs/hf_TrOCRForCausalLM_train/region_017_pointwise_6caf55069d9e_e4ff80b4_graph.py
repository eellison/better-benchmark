class GraphModule(torch.nn.Module):
    def forward(self, add: "i64[64, 256]", tangents_1: "f32[64, 256, 1024]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:61 in forward, code: return super().forward(position_ids + self.offset)
        eq_scalar: "b8[64, 256]" = torch.ops.aten.eq.Scalar(add, -1)
        unsqueeze_default: "b8[64, 256, 1]" = torch.ops.aten.unsqueeze.default(eq_scalar, -1);  eq_scalar = None
        full_default: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "f32[64, 256, 1024]" = torch.ops.aten.where.self(unsqueeze_default, full_default, tangents_1);  unsqueeze_default = full_default = tangents_1 = None
        full_default_1: "f32[514, 1024]" = torch.ops.aten.full.default([514, 1024], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default: "f32[514, 1024]" = torch.ops.aten.index_put.default(full_default_1, [add], where_self, True);  full_default_1 = add = where_self = None
        return index_put_default
