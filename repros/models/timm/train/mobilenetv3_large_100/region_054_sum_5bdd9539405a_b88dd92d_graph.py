class GraphModule(torch.nn.Module):
    def forward(self, convolution_10: "f32[512, 72, 28, 28]", getitem_21: "f32[1, 72, 1, 1]", rsqrt_10: "f32[1, 72, 1, 1]", primals_66: "f32[72]", primals_67: "f32[72]", getitem_239: "f32[512, 72, 28, 28]", convolution_12: "f32[512, 72, 1, 1]", full_default: "f32[]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_tensor: "f32[512, 72, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_10, getitem_21);  convolution_10 = getitem_21 = None
        mul_tensor: "f32[512, 72, 28, 28]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_10);  sub_tensor = rsqrt_10 = None
        unsqueeze_default: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(primals_66, -1);  primals_66 = None
        unsqueeze_default_1: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        mul_tensor_1: "f32[512, 72, 28, 28]" = torch.ops.aten.mul.Tensor(mul_tensor, unsqueeze_default_1);  mul_tensor = unsqueeze_default_1 = None
        unsqueeze_default_2: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(primals_67, -1);  primals_67 = None
        unsqueeze_default_3: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        add_tensor: "f32[512, 72, 28, 28]" = torch.ops.aten.add.Tensor(mul_tensor_1, unsqueeze_default_3);  mul_tensor_1 = unsqueeze_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_default: "f32[512, 72, 28, 28]" = torch.ops.aten.relu.default(add_tensor);  add_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:83 in forward, code: return x * self.gate(x_se)
        mul_tensor_2: "f32[512, 72, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_239, relu_default);  getitem_239 = relu_default = None
        sum_dim_int_list: "f32[512, 72, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [2, 3], True);  mul_tensor_2 = None
        gt_scalar: "b8[512, 72, 1, 1]" = torch.ops.aten.gt.Scalar(convolution_12, -3.0)
        lt_scalar: "b8[512, 72, 1, 1]" = torch.ops.aten.lt.Scalar(convolution_12, 3.0);  convolution_12 = None
        bitwise_and_tensor: "b8[512, 72, 1, 1]" = torch.ops.aten.bitwise_and.Tensor(gt_scalar, lt_scalar);  gt_scalar = lt_scalar = None
        mul_tensor_3: "f32[512, 72, 1, 1]" = torch.ops.aten.mul.Tensor(sum_dim_int_list, 0.16666666666666666);  sum_dim_int_list = None
        where_self: "f32[512, 72, 1, 1]" = torch.ops.aten.where.self(bitwise_and_tensor, mul_tensor_3, full_default);  bitwise_and_tensor = mul_tensor_3 = full_default = None
        return where_self
