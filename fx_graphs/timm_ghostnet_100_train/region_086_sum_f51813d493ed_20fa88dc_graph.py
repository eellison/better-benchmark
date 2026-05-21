class GraphModule(torch.nn.Module):
    def forward(self, getitem_355: "f32[512, 120, 28, 28]", cat_8: "f32[512, 120, 28, 28]", convolution_28: "f32[512, 120, 1, 1]", full_default: "f32[]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:83 in forward, code: return x * self.gate(x_se)
        mul_tensor: "f32[512, 120, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_355, cat_8);  getitem_355 = cat_8 = None
        sum_dim_int_list: "f32[512, 120, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [2, 3], True);  mul_tensor = None
        gt_scalar: "b8[512, 120, 1, 1]" = torch.ops.aten.gt.Scalar(convolution_28, -3.0)
        lt_scalar: "b8[512, 120, 1, 1]" = torch.ops.aten.lt.Scalar(convolution_28, 3.0);  convolution_28 = None
        bitwise_and_tensor: "b8[512, 120, 1, 1]" = torch.ops.aten.bitwise_and.Tensor(gt_scalar, lt_scalar);  gt_scalar = lt_scalar = None
        mul_tensor_1: "f32[512, 120, 1, 1]" = torch.ops.aten.mul.Tensor(sum_dim_int_list, 0.16666666666666666);  sum_dim_int_list = None
        where_self: "f32[512, 120, 1, 1]" = torch.ops.aten.where.self(bitwise_and_tensor, mul_tensor_1, full_default);  bitwise_and_tensor = mul_tensor_1 = full_default = None
        return where_self
