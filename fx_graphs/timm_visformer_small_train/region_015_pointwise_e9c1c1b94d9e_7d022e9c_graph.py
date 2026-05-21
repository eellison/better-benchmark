class GraphModule(torch.nn.Module):
    def forward(self, convolution_21: "f32[128, 384, 28, 28]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:72 in forward, code: x = self.act2(x)
        mul_tensor: "f32[128, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_21, 0.5)
        mul_tensor_1: "f32[128, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_21, 0.7071067811865476);  convolution_21 = None
        erf_default: "f32[128, 384, 28, 28]" = torch.ops.aten.erf.default(mul_tensor_1);  mul_tensor_1 = None
        add_tensor: "f32[128, 384, 28, 28]" = torch.ops.aten.add.Tensor(erf_default, 1);  erf_default = None
        mul_tensor_2: "f32[128, 384, 28, 28]" = torch.ops.aten.mul.Tensor(mul_tensor, add_tensor);  mul_tensor = add_tensor = None
        return mul_tensor_2
