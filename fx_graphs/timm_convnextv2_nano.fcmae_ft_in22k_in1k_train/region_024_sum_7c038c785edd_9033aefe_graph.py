class GraphModule(torch.nn.Module):
    def forward(self, getitem_125: "f32[128, 320, 14, 14]", primals_56: "f32[320]", convolution_15: "f32[128, 320, 14, 14]", getitem_15: "f32[128, 14, 14, 1]", rsqrt_7: "f32[128, 14, 14, 1]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:133 in forward, code: x = x.permute(0, 3, 1, 2)
        permute_default: "f32[128, 14, 14, 320]" = torch.ops.aten.permute.default(getitem_125, [0, 2, 3, 1]);  getitem_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_tensor: "f32[128, 14, 14, 320]" = torch.ops.aten.mul.Tensor(permute_default, primals_56);  permute_default = primals_56 = None
        mul_tensor_1: "f32[128, 14, 14, 320]" = torch.ops.aten.mul.Tensor(mul_tensor, 320)
        sum_dim_int_list: "f32[128, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [3], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:128 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_default_1: "f32[128, 14, 14, 320]" = torch.ops.aten.permute.default(convolution_15, [0, 2, 3, 1]);  convolution_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        sub_tensor: "f32[128, 14, 14, 320]" = torch.ops.aten.sub.Tensor(permute_default_1, getitem_15);  permute_default_1 = getitem_15 = None
        mul_tensor_2: "f32[128, 14, 14, 320]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_7);  sub_tensor = None
        mul_tensor_3: "f32[128, 14, 14, 320]" = torch.ops.aten.mul.Tensor(mul_tensor, mul_tensor_2);  mul_tensor = None
        sum_dim_int_list_1: "f32[128, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_3, [3], True);  mul_tensor_3 = None
        mul_tensor_4: "f32[128, 14, 14, 320]" = torch.ops.aten.mul.Tensor(mul_tensor_2, sum_dim_int_list_1);  mul_tensor_2 = sum_dim_int_list_1 = None
        sub_tensor_1: "f32[128, 14, 14, 320]" = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list);  mul_tensor_1 = sum_dim_int_list = None
        sub_tensor_2: "f32[128, 14, 14, 320]" = torch.ops.aten.sub.Tensor(sub_tensor_1, mul_tensor_4);  sub_tensor_1 = mul_tensor_4 = None
        div_tensor: "f32[128, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_7, 320);  rsqrt_7 = None
        mul_tensor_5: "f32[128, 14, 14, 320]" = torch.ops.aten.mul.Tensor(div_tensor, sub_tensor_2);  div_tensor = sub_tensor_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:128 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_default_2: "f32[128, 320, 14, 14]" = torch.ops.aten.permute.default(mul_tensor_5, [0, 3, 1, 2]);  mul_tensor_5 = None
        return permute_default_2
