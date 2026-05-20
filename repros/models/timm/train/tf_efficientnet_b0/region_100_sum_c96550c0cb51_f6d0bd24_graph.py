class GraphModule(torch.nn.Module):
    def forward(self, getitem_98: "f32[128, 320, 7, 7]", convolution_79: "f32[128, 320, 7, 7]", unsqueeze_210: "f32[1, 320, 1, 1]", squeeze_142: "f32[320]", primals_352: "f32[320]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sum_dim_int_list: "f32[320]" = torch.ops.aten.sum.dim_IntList(getitem_98, [0, 2, 3])
        sub_tensor: "f32[128, 320, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_79, unsqueeze_210);  convolution_79 = unsqueeze_210 = None
        mul_tensor: "f32[128, 320, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_98, sub_tensor)
        sum_dim_int_list_1: "f32[320]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [0, 2, 3]);  mul_tensor = None
        mul_tensor_1: "f32[320]" = torch.ops.aten.mul.Tensor(sum_dim_int_list, 0.00015943877551020407);  sum_dim_int_list = None
        unsqueeze_default: "f32[1, 320]" = torch.ops.aten.unsqueeze.default(mul_tensor_1, 0);  mul_tensor_1 = None
        unsqueeze_default_1: "f32[1, 320, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, 2);  unsqueeze_default = None
        unsqueeze_default_2: "f32[1, 320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_1, 3);  unsqueeze_default_1 = None
        mul_tensor_2: "f32[320]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, 0.00015943877551020407);  sum_dim_int_list_1 = None
        mul_tensor_3: "f32[320]" = torch.ops.aten.mul.Tensor(squeeze_142, squeeze_142)
        mul_tensor_4: "f32[320]" = torch.ops.aten.mul.Tensor(mul_tensor_2, mul_tensor_3);  mul_tensor_2 = mul_tensor_3 = None
        unsqueeze_default_3: "f32[1, 320]" = torch.ops.aten.unsqueeze.default(mul_tensor_4, 0);  mul_tensor_4 = None
        unsqueeze_default_4: "f32[1, 320, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_3, 2);  unsqueeze_default_3 = None
        unsqueeze_default_5: "f32[1, 320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 3);  unsqueeze_default_4 = None
        mul_tensor_5: "f32[320]" = torch.ops.aten.mul.Tensor(squeeze_142, primals_352);  squeeze_142 = primals_352 = None
        unsqueeze_default_6: "f32[1, 320]" = torch.ops.aten.unsqueeze.default(mul_tensor_5, 0);  mul_tensor_5 = None
        unsqueeze_default_7: "f32[1, 320, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, 2);  unsqueeze_default_6 = None
        unsqueeze_default_8: "f32[1, 320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_7, 3);  unsqueeze_default_7 = None
        mul_tensor_6: "f32[128, 320, 7, 7]" = torch.ops.aten.mul.Tensor(sub_tensor, unsqueeze_default_5);  sub_tensor = unsqueeze_default_5 = None
        sub_tensor_1: "f32[128, 320, 7, 7]" = torch.ops.aten.sub.Tensor(getitem_98, mul_tensor_6);  getitem_98 = mul_tensor_6 = None
        sub_tensor_2: "f32[128, 320, 7, 7]" = torch.ops.aten.sub.Tensor(sub_tensor_1, unsqueeze_default_2);  sub_tensor_1 = unsqueeze_default_2 = None
        mul_tensor_7: "f32[128, 320, 7, 7]" = torch.ops.aten.mul.Tensor(sub_tensor_2, unsqueeze_default_8);  sub_tensor_2 = unsqueeze_default_8 = None
        return mul_tensor_7
