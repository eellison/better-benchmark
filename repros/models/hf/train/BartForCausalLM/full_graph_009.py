class GraphModule(torch.nn.Module):
    def forward(self, primals_2: "i64[8, 1024]", tangents_1: "f32[8, 1024, 1024]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:111 in forward, code: return super().forward(input_ids) * self.embed_scale
        mul_1: "f32[8, 1024, 1024]" = torch.ops.aten.mul.Tensor(tangents_1, 1.0);  tangents_1 = None
        eq: "b8[8, 1024]" = torch.ops.aten.eq.Scalar(primals_2, 1)
        unsqueeze: "b8[8, 1024, 1]" = torch.ops.aten.unsqueeze.default(eq, -1);  eq = None
        full_default: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "f32[8, 1024, 1024]" = torch.ops.aten.where.self(unsqueeze, full_default, mul_1);  unsqueeze = full_default = mul_1 = None
        full_default_1: "f32[50265, 1024]" = torch.ops.aten.full.default([50265, 1024], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put: "f32[50265, 1024]" = torch.ops.aten.index_put.default(full_default_1, [primals_2], where, True);  full_default_1 = primals_2 = where = None
        return (index_put, None)
