class GraphModule(torch.nn.Module):
    def forward(self, convolution_66: "f32[512, 672, 7, 7]", getitem_117: "f32[1, 672, 1, 1]", rsqrt_58: "f32[1, 672, 1, 1]", primals_370: "f32[672]", primals_371: "f32[672]", getitem_235: "f32[512, 672, 7, 7]", convolution_68: "f32[512, 672, 1, 1]", full_default: "f32[]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:436 in forward, code: x = self.bn_dw(x)
        sub_tensor: "f32[512, 672, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_66, getitem_117);  convolution_66 = getitem_117 = None
        mul_tensor: "f32[512, 672, 7, 7]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_58);  sub_tensor = rsqrt_58 = None
        unsqueeze_default: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(primals_370, -1);  primals_370 = None
        unsqueeze_default_1: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        mul_tensor_1: "f32[512, 672, 7, 7]" = torch.ops.aten.mul.Tensor(mul_tensor, unsqueeze_default_1);  mul_tensor = unsqueeze_default_1 = None
        unsqueeze_default_2: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(primals_371, -1);  primals_371 = None
        unsqueeze_default_3: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        add_tensor: "f32[512, 672, 7, 7]" = torch.ops.aten.add.Tensor(mul_tensor_1, unsqueeze_default_3);  mul_tensor_1 = unsqueeze_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:83 in forward, code: return x * self.gate(x_se)
        mul_tensor_2: "f32[512, 672, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_235, add_tensor);  getitem_235 = add_tensor = None
        sum_dim_int_list: "f32[512, 672, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [2, 3], True);  mul_tensor_2 = None
        gt_scalar: "b8[512, 672, 1, 1]" = torch.ops.aten.gt.Scalar(convolution_68, -3.0)
        lt_scalar: "b8[512, 672, 1, 1]" = torch.ops.aten.lt.Scalar(convolution_68, 3.0);  convolution_68 = None
        bitwise_and_tensor: "b8[512, 672, 1, 1]" = torch.ops.aten.bitwise_and.Tensor(gt_scalar, lt_scalar);  gt_scalar = lt_scalar = None
        mul_tensor_3: "f32[512, 672, 1, 1]" = torch.ops.aten.mul.Tensor(sum_dim_int_list, 0.16666666666666666);  sum_dim_int_list = None
        where_self: "f32[512, 672, 1, 1]" = torch.ops.aten.where.self(bitwise_and_tensor, mul_tensor_3, full_default);  bitwise_and_tensor = mul_tensor_3 = full_default = None
        return where_self
