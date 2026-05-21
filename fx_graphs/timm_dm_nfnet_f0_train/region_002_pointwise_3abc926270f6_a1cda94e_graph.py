class GraphModule(torch.nn.Module):
    def forward(self, convolution_78: "f32[128, 768, 1, 1]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:61 in forward, code: x_se = self.act(self.bn(x_se))
        relu_default: "f32[128, 768, 1, 1]" = torch.ops.aten.relu.default(convolution_78);  convolution_78 = None
        return relu_default
