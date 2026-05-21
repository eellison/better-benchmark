class GraphModule(torch.nn.Module):
    def forward(self, getitem_345: "f32[128, 64, 97, 97]", convolution_2: "f32[128, 64, 96, 96]"):
        # File: /tmp/pytorch-work/torch/nn/functional.py:5461 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_default: "f32[128, 64, 96, 96]" = torch.ops.aten.constant_pad_nd.default(getitem_345, [0, -1, 0, -1]);  getitem_345 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:89 in forward, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
        mul_tensor: "f32[128, 64, 96, 96]" = torch.ops.aten.mul.Tensor(constant_pad_nd_default, 1.7015043497085571);  constant_pad_nd_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:135 in gelu, code: return F.gelu(x)
        mul_tensor_1: "f32[128, 64, 96, 96]" = torch.ops.aten.mul.Tensor(convolution_2, 0.7071067811865476)
        erf_default: "f32[128, 64, 96, 96]" = torch.ops.aten.erf.default(mul_tensor_1);  mul_tensor_1 = None
        add_tensor: "f32[128, 64, 96, 96]" = torch.ops.aten.add.Tensor(erf_default, 1);  erf_default = None
        mul_tensor_2: "f32[128, 64, 96, 96]" = torch.ops.aten.mul.Tensor(add_tensor, 0.5);  add_tensor = None
        mul_tensor_3: "f32[128, 64, 96, 96]" = torch.ops.aten.mul.Tensor(convolution_2, convolution_2)
        mul_tensor_4: "f32[128, 64, 96, 96]" = torch.ops.aten.mul.Tensor(mul_tensor_3, -0.5);  mul_tensor_3 = None
        exp_default: "f32[128, 64, 96, 96]" = torch.ops.aten.exp.default(mul_tensor_4);  mul_tensor_4 = None
        mul_tensor_5: "f32[128, 64, 96, 96]" = torch.ops.aten.mul.Tensor(exp_default, 0.3989422804014327);  exp_default = None
        mul_tensor_6: "f32[128, 64, 96, 96]" = torch.ops.aten.mul.Tensor(convolution_2, mul_tensor_5);  convolution_2 = mul_tensor_5 = None
        add_tensor_1: "f32[128, 64, 96, 96]" = torch.ops.aten.add.Tensor(mul_tensor_2, mul_tensor_6);  mul_tensor_2 = mul_tensor_6 = None
        mul_tensor_7: "f32[128, 64, 96, 96]" = torch.ops.aten.mul.Tensor(mul_tensor, add_tensor_1);  mul_tensor = add_tensor_1 = None
        return mul_tensor_7
