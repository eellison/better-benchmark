class GraphModule(torch.nn.Module):
    def forward(self, convolution_26: "f32[128, 1536, 14, 14]", getitem_167: "f32[128, 1536, 14, 14]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:68 in forward, code: x = self.act1(x)
        mul_tensor: "f32[128, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_26, 0.7071067811865476)
        erf_default: "f32[128, 1536, 14, 14]" = torch.ops.aten.erf.default(mul_tensor);  mul_tensor = None
        add_tensor: "f32[128, 1536, 14, 14]" = torch.ops.aten.add.Tensor(erf_default, 1);  erf_default = None
        mul_tensor_1: "f32[128, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(add_tensor, 0.5);  add_tensor = None
        mul_tensor_2: "f32[128, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_26, convolution_26)
        mul_tensor_3: "f32[128, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_tensor_2, -0.5);  mul_tensor_2 = None
        exp_default: "f32[128, 1536, 14, 14]" = torch.ops.aten.exp.default(mul_tensor_3);  mul_tensor_3 = None
        mul_tensor_4: "f32[128, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(exp_default, 0.3989422804014327);  exp_default = None
        mul_tensor_5: "f32[128, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_26, mul_tensor_4);  convolution_26 = mul_tensor_4 = None
        add_tensor_1: "f32[128, 1536, 14, 14]" = torch.ops.aten.add.Tensor(mul_tensor_1, mul_tensor_5);  mul_tensor_1 = mul_tensor_5 = None
        mul_tensor_6: "f32[128, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_167, add_tensor_1);  getitem_167 = add_tensor_1 = None
        return mul_tensor_6
