class GraphModule(torch.nn.Module):
    def forward(self, convolution_32: "f32[128, 20, 1, 1]", getitem_239: "f32[128, 20, 1, 1]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:81 in forward, code: x_se = self.act1(x_se)
        neg_default: "f32[128, 20, 1, 1]" = torch.ops.aten.neg.default(convolution_32)
        exp_default: "f32[128, 20, 1, 1]" = torch.ops.aten.exp.default(neg_default);  neg_default = None
        add_tensor: "f32[128, 20, 1, 1]" = torch.ops.aten.add.Tensor(exp_default, 1);  exp_default = None
        reciprocal_default: "f32[128, 20, 1, 1]" = torch.ops.aten.reciprocal.default(add_tensor);  add_tensor = None
        mul_tensor: "f32[128, 20, 1, 1]" = torch.ops.aten.mul.Tensor(reciprocal_default, 1);  reciprocal_default = None
        mul_tensor_1: "f32[128, 20, 1, 1]" = torch.ops.aten.mul.Tensor(getitem_239, mul_tensor);  getitem_239 = None
        sub_tensor: "f32[128, 20, 1, 1]" = torch.ops.aten.sub.Tensor(1, mul_tensor);  mul_tensor = None
        mul_tensor_2: "f32[128, 20, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_32, sub_tensor);  convolution_32 = sub_tensor = None
        add_tensor_1: "f32[128, 20, 1, 1]" = torch.ops.aten.add.Tensor(mul_tensor_2, 1);  mul_tensor_2 = None
        mul_tensor_3: "f32[128, 20, 1, 1]" = torch.ops.aten.mul.Tensor(mul_tensor_1, add_tensor_1);  mul_tensor_1 = add_tensor_1 = None
        return mul_tensor_3
