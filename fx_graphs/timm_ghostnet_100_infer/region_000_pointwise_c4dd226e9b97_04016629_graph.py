class GraphModule(torch.nn.Module):
    def forward(self, convolution_94: "f32[512, 1280, 1, 1]", arg431_1: "f32[1000, 1280]", _shape_param_0):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:833 in forward_head, code: x = self.act2(x)
        relu_default: "f32[512, 1280, 1, 1]" = torch.ops.aten.relu.default(convolution_94);  convolution_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:834 in forward_head, code: x = self.flatten(x)
        reshape_default: "f32[512, 1280]" = torch.ops.aten.reshape.default(relu_default, _shape_param_0);  relu_default = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/linear.py:19 in forward, code: return F.linear(input, self.weight, self.bias)
        permute_default: "f32[1280, 1000]" = torch.ops.aten.permute.default(arg431_1, [1, 0]);  arg431_1 = None
        return (reshape_default, permute_default)
