class GraphModule(torch.nn.Module):
    def forward(self, convolution_12: "f32[128, 128, 56, 56]", getitem_315: "f32[128, 128, 56, 56]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:269 in forward, code: out = self.conv2(self.act2(out))
        neg_default: "f32[128, 128, 56, 56]" = torch.ops.aten.neg.default(convolution_12)
        exp_default: "f32[128, 128, 56, 56]" = torch.ops.aten.exp.default(neg_default);  neg_default = None
        add_tensor: "f32[128, 128, 56, 56]" = torch.ops.aten.add.Tensor(exp_default, 1);  exp_default = None
        reciprocal_default: "f32[128, 128, 56, 56]" = torch.ops.aten.reciprocal.default(add_tensor);  add_tensor = None
        mul_tensor: "f32[128, 128, 56, 56]" = torch.ops.aten.mul.Tensor(reciprocal_default, 1);  reciprocal_default = None
        mul_tensor_1: "f32[128, 128, 56, 56]" = torch.ops.aten.mul.Tensor(getitem_315, mul_tensor);  getitem_315 = None
        sub_tensor: "f32[128, 128, 56, 56]" = torch.ops.aten.sub.Tensor(1, mul_tensor);  mul_tensor = None
        mul_tensor_2: "f32[128, 128, 56, 56]" = torch.ops.aten.mul.Tensor(convolution_12, sub_tensor);  convolution_12 = sub_tensor = None
        add_tensor_1: "f32[128, 128, 56, 56]" = torch.ops.aten.add.Tensor(mul_tensor_2, 1);  mul_tensor_2 = None
        mul_tensor_3: "f32[128, 128, 56, 56]" = torch.ops.aten.mul.Tensor(mul_tensor_1, add_tensor_1);  mul_tensor_1 = add_tensor_1 = None
        return mul_tensor_3
