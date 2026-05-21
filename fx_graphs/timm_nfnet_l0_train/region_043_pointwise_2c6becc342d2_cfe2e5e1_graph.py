class GraphModule(torch.nn.Module):
    def forward(self, mul_884: "f32[128, 512, 28, 28]", sigmoid_1: "f32[128, 512, 1, 1]", getitem_306: "f32[128, 512, 1, 1]", _shape_param_0):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_tensor: "f32[128, 512, 28, 28]" = torch.ops.aten.mul.Tensor(mul_884, sigmoid_1);  mul_884 = sigmoid_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:56 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        expand_default: "f32[128, 512, 28, 28]" = torch.ops.aten.expand.default(getitem_306, _shape_param_0);  getitem_306 = _shape_param_0 = None
        div_scalar: "f32[128, 512, 28, 28]" = torch.ops.aten.div.Scalar(expand_default, 784);  expand_default = None
        add_tensor: "f32[128, 512, 28, 28]" = torch.ops.aten.add.Tensor(mul_tensor, div_scalar);  mul_tensor = div_scalar = None
        return add_tensor
