class GraphModule(torch.nn.Module):
    def forward(self, mul_1381: "f32[128, 256, 48, 48]", sigmoid: "f32[128, 256, 1, 1]", getitem_327: "f32[128, 256, 1, 1]", _shape_param_0):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_tensor: "f32[128, 256, 48, 48]" = torch.ops.aten.mul.Tensor(mul_1381, sigmoid);  mul_1381 = sigmoid = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:56 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        expand_default: "f32[128, 256, 48, 48]" = torch.ops.aten.expand.default(getitem_327, _shape_param_0);  getitem_327 = _shape_param_0 = None
        div_scalar: "f32[128, 256, 48, 48]" = torch.ops.aten.div.Scalar(expand_default, 2304);  expand_default = None
        add_tensor: "f32[128, 256, 48, 48]" = torch.ops.aten.add.Tensor(mul_tensor, div_scalar);  mul_tensor = div_scalar = None
        return add_tensor
