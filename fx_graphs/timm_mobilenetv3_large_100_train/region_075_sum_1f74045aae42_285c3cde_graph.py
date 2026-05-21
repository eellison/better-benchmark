class GraphModule(torch.nn.Module):
    def forward(self, convolution_42: "f32[512, 672, 14, 14]", getitem_69: "f32[1, 672, 1, 1]", rsqrt_34: "f32[1, 672, 1, 1]", primals_226: "f32[672]", primals_227: "f32[672]", getitem_143: "f32[512, 672, 14, 14]", convolution_44: "f32[512, 672, 1, 1]", full_default: "f32[]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_tensor: "f32[512, 672, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_42, getitem_69);  convolution_42 = getitem_69 = None
        mul_tensor: "f32[512, 672, 14, 14]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_34);  sub_tensor = rsqrt_34 = None
        unsqueeze_default: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(primals_226, -1);  primals_226 = None
        unsqueeze_default_1: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        mul_tensor_1: "f32[512, 672, 14, 14]" = torch.ops.aten.mul.Tensor(mul_tensor, unsqueeze_default_1);  mul_tensor = unsqueeze_default_1 = None
        unsqueeze_default_2: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(primals_227, -1);  primals_227 = None
        unsqueeze_default_3: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        add_tensor: "f32[512, 672, 14, 14]" = torch.ops.aten.add.Tensor(mul_tensor_1, unsqueeze_default_3);  mul_tensor_1 = unsqueeze_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        add_tensor_1: "f32[512, 672, 14, 14]" = torch.ops.aten.add.Tensor(add_tensor, 3)
        clamp_min_default: "f32[512, 672, 14, 14]" = torch.ops.aten.clamp_min.default(add_tensor_1, 0);  add_tensor_1 = None
        clamp_max_default: "f32[512, 672, 14, 14]" = torch.ops.aten.clamp_max.default(clamp_min_default, 6);  clamp_min_default = None
        mul_tensor_2: "f32[512, 672, 14, 14]" = torch.ops.aten.mul.Tensor(add_tensor, clamp_max_default);  add_tensor = clamp_max_default = None
        div_tensor: "f32[512, 672, 14, 14]" = torch.ops.aten.div.Tensor(mul_tensor_2, 6);  mul_tensor_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:83 in forward, code: return x * self.gate(x_se)
        mul_tensor_3: "f32[512, 672, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_143, div_tensor);  getitem_143 = div_tensor = None
        sum_dim_int_list: "f32[512, 672, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_3, [2, 3], True);  mul_tensor_3 = None
        gt_scalar: "b8[512, 672, 1, 1]" = torch.ops.aten.gt.Scalar(convolution_44, -3.0)
        lt_scalar: "b8[512, 672, 1, 1]" = torch.ops.aten.lt.Scalar(convolution_44, 3.0);  convolution_44 = None
        bitwise_and_tensor: "b8[512, 672, 1, 1]" = torch.ops.aten.bitwise_and.Tensor(gt_scalar, lt_scalar);  gt_scalar = lt_scalar = None
        mul_tensor_4: "f32[512, 672, 1, 1]" = torch.ops.aten.mul.Tensor(sum_dim_int_list, 0.16666666666666666);  sum_dim_int_list = None
        where_self: "f32[512, 672, 1, 1]" = torch.ops.aten.where.self(bitwise_and_tensor, mul_tensor_4, full_default);  bitwise_and_tensor = mul_tensor_4 = full_default = None
        return where_self
