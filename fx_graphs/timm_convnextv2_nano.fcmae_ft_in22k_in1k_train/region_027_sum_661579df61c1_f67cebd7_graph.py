class GraphModule(torch.nn.Module):
    def forward(self, getitem_50: "f32[128, 640, 7, 7]", primals_140: "f32[640]", convolution_40: "f32[128, 640, 7, 7]", getitem_33: "f32[128, 7, 7, 1]", rsqrt_16: "f32[128, 7, 7, 1]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:133 in forward, code: x = x.permute(0, 3, 1, 2)
        permute_default: "f32[128, 7, 7, 640]" = torch.ops.aten.permute.default(getitem_50, [0, 2, 3, 1]);  getitem_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_tensor: "f32[128, 7, 7, 640]" = torch.ops.aten.mul.Tensor(permute_default, primals_140);  permute_default = primals_140 = None
        mul_tensor_1: "f32[128, 7, 7, 640]" = torch.ops.aten.mul.Tensor(mul_tensor, 640)
        sum_dim_int_list: "f32[128, 7, 7, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [3], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:128 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_default_1: "f32[128, 7, 7, 640]" = torch.ops.aten.permute.default(convolution_40, [0, 2, 3, 1]);  convolution_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        sub_tensor: "f32[128, 7, 7, 640]" = torch.ops.aten.sub.Tensor(permute_default_1, getitem_33);  permute_default_1 = getitem_33 = None
        mul_tensor_2: "f32[128, 7, 7, 640]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_16);  sub_tensor = None
        mul_tensor_3: "f32[128, 7, 7, 640]" = torch.ops.aten.mul.Tensor(mul_tensor, mul_tensor_2);  mul_tensor = None
        sum_dim_int_list_1: "f32[128, 7, 7, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_3, [3], True);  mul_tensor_3 = None
        mul_tensor_4: "f32[128, 7, 7, 640]" = torch.ops.aten.mul.Tensor(mul_tensor_2, sum_dim_int_list_1);  mul_tensor_2 = sum_dim_int_list_1 = None
        sub_tensor_1: "f32[128, 7, 7, 640]" = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list);  mul_tensor_1 = sum_dim_int_list = None
        sub_tensor_2: "f32[128, 7, 7, 640]" = torch.ops.aten.sub.Tensor(sub_tensor_1, mul_tensor_4);  sub_tensor_1 = mul_tensor_4 = None
        div_tensor: "f32[128, 7, 7, 1]" = torch.ops.aten.div.Tensor(rsqrt_16, 640);  rsqrt_16 = None
        mul_tensor_5: "f32[128, 7, 7, 640]" = torch.ops.aten.mul.Tensor(div_tensor, sub_tensor_2);  div_tensor = sub_tensor_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:128 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_default_2: "f32[128, 640, 7, 7]" = torch.ops.aten.permute.default(mul_tensor_5, [0, 3, 1, 2]);  mul_tensor_5 = None
        return permute_default_2
