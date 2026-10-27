class GraphModule(torch.nn.Module):
    def forward(self, primals_2: "i64[8, 1024][1024, 1]cuda:0", tangents_1: "f32[8, 1024, 1024][1048576, 1024, 1]cuda:0"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:123 in forward, code: return super().forward(input_ids) * self.embed_scale
        mul_1: "f32[8, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(tangents_1, 1.0);  tangents_1 = None
        ge: "b8[8, 1024][1024, 1]cuda:0" = torch.ops.aten.ge.Scalar(primals_2, 0)
        lt: "b8[8, 1024][1024, 1]cuda:0" = torch.ops.aten.lt.Scalar(primals_2, 50265)
        bitwise_and: "b8[8, 1024][1024, 1]cuda:0" = torch.ops.aten.bitwise_and.Tensor(ge, lt);  ge = lt = None
        ne: "b8[8, 1024][1024, 1]cuda:0" = torch.ops.aten.ne.Scalar(primals_2, 1)
        bitwise_and_1: "b8[8, 1024][1024, 1]cuda:0" = torch.ops.aten.bitwise_and.Tensor(bitwise_and, ne);  bitwise_and = ne = None
        unsqueeze: "b8[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(bitwise_and_1, -1);  bitwise_and_1 = None
        full_default: "f32[50265, 1024][1024, 1]cuda:0" = torch.ops.aten.full.default([50265, 1024], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        _unsafe_masked_index_put_accumulate: "f32[50265, 1024][1024, 1]cuda:0" = torch.ops.aten._unsafe_masked_index_put_accumulate.default(full_default, unsqueeze, [primals_2], mul_1);  full_default = unsqueeze = primals_2 = mul_1 = None
        return (_unsafe_masked_index_put_accumulate, None)
