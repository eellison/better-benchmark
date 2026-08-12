class GraphModule(torch.nn.Module):
    def forward(self, add: "i64[1, 1024][1024, 1]cuda:0", tangents_1: "f32[1, 1024, 768][786432, 768, 1]cuda:0"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/plbart/modeling_plbart.py:114 in forward, code: return super().forward(position_ids + self.offset)
        ge: "b8[1, 1024][1024, 1]cuda:0" = torch.ops.aten.ge.Scalar(add, 0)
        lt: "b8[1, 1024][1024, 1]cuda:0" = torch.ops.aten.lt.Scalar(add, 1026)
        bitwise_and: "b8[1, 1024][1024, 1]cuda:0" = torch.ops.aten.bitwise_and.Tensor(ge, lt);  ge = lt = None
        ne: "b8[1, 1024][1024, 1]cuda:0" = torch.ops.aten.ne.Scalar(add, -1)
        bitwise_and_1: "b8[1, 1024][1024, 1]cuda:0" = torch.ops.aten.bitwise_and.Tensor(bitwise_and, ne);  bitwise_and = ne = None
        unsqueeze_1: "b8[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(bitwise_and_1, -1);  bitwise_and_1 = None
        full_default: "f32[1026, 768][768, 1]cuda:0" = torch.ops.aten.full.default([1026, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        _unsafe_masked_index_put_accumulate: "f32[1026, 768][768, 1]cuda:0" = torch.ops.aten._unsafe_masked_index_put_accumulate.default(full_default, unsqueeze_1, [add], tangents_1);  full_default = unsqueeze_1 = add = tangents_1 = None
        return (None, _unsafe_masked_index_put_accumulate)
