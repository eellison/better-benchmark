class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[512]", primals_3: "f32[8, 4096, 512]", getitem_1: "f32[8, 4096, 1]", rsqrt: "f32[8, 4096, 1]", gt: "b8[8, 4096, 512]", tangents_1: "f32[8, 4096, 512]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1799 in torch_dynamo_resume_in_forward_at_1781, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        convert_element_type: "f32[8, 4096, 512]" = torch.ops.prims.convert_element_type.default(gt, torch.float32);  gt = None
        mul_4: "f32[8, 4096, 512]" = torch.ops.aten.mul.Tensor(convert_element_type, 1.0526315789473684);  convert_element_type = None
        mul_5: "f32[8, 4096, 512]" = torch.ops.aten.mul.Tensor(tangents_1, mul_4);  tangents_1 = mul_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1796 in torch_dynamo_resume_in_forward_at_1781, code: hidden_states = self.layer_norm(hidden_states)
        mul_7: "f32[8, 4096, 512]" = torch.ops.aten.mul.Tensor(mul_5, primals_1);  primals_1 = None
        mul_8: "f32[8, 4096, 512]" = torch.ops.aten.mul.Tensor(mul_7, 512)
        sum_1: "f32[8, 4096, 1]" = torch.ops.aten.sum.dim_IntList(mul_7, [2], True)
        sub: "f32[8, 4096, 512]" = torch.ops.aten.sub.Tensor(primals_3, getitem_1);  primals_3 = getitem_1 = None
        mul: "f32[8, 4096, 512]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        mul_9: "f32[8, 4096, 512]" = torch.ops.aten.mul.Tensor(mul_7, mul);  mul_7 = None
        sum_2: "f32[8, 4096, 1]" = torch.ops.aten.sum.dim_IntList(mul_9, [2], True);  mul_9 = None
        mul_10: "f32[8, 4096, 512]" = torch.ops.aten.mul.Tensor(mul, sum_2);  sum_2 = None
        sub_2: "f32[8, 4096, 512]" = torch.ops.aten.sub.Tensor(mul_8, sum_1);  mul_8 = sum_1 = None
        sub_3: "f32[8, 4096, 512]" = torch.ops.aten.sub.Tensor(sub_2, mul_10);  sub_2 = mul_10 = None
        div: "f32[8, 4096, 1]" = torch.ops.aten.div.Tensor(rsqrt, 512);  rsqrt = None
        mul_11: "f32[8, 4096, 512]" = torch.ops.aten.mul.Tensor(div, sub_3);  div = sub_3 = None
        mul_12: "f32[8, 4096, 512]" = torch.ops.aten.mul.Tensor(mul_5, mul);  mul = None
        sum_3: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_12, [0, 1]);  mul_12 = None
        sum_4: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_5, [0, 1]);  mul_5 = None
        return (sum_3, sum_4, mul_11)
