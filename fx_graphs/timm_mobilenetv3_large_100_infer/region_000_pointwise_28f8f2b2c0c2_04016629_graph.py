class GraphModule(torch.nn.Module):
    def forward(self, convolution_62: "f32[512, 1280, 1, 1]", arg265_1: "f32[1000, 1280]", _shape_param_0):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/mobilenetv3.py:325 in forward_head, code: x = self.act2(x)
        add_tensor: "f32[512, 1280, 1, 1]" = torch.ops.aten.add.Tensor(convolution_62, 3)
        clamp_min_default: "f32[512, 1280, 1, 1]" = torch.ops.aten.clamp_min.default(add_tensor, 0);  add_tensor = None
        clamp_max_default: "f32[512, 1280, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_default, 6);  clamp_min_default = None
        mul_tensor: "f32[512, 1280, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_62, clamp_max_default);  convolution_62 = clamp_max_default = None
        div_tensor: "f32[512, 1280, 1, 1]" = torch.ops.aten.div.Tensor(mul_tensor, 6);  mul_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/mobilenetv3.py:326 in forward_head, code: x = self.flatten(x)
        reshape_default: "f32[512, 1280]" = torch.ops.aten.reshape.default(div_tensor, _shape_param_0);  div_tensor = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/linear.py:19 in forward, code: return F.linear(input, self.weight, self.bias)
        permute_default: "f32[1280, 1000]" = torch.ops.aten.permute.default(arg265_1, [1, 0]);  arg265_1 = None
        return (reshape_default, permute_default)
