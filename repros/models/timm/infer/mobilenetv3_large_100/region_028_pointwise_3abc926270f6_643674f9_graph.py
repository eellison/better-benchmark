class GraphModule(torch.nn.Module):
    def forward(self, convolution_21: "f32[512, 32, 1, 1]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:81 in forward, code: x_se = self.act1(x_se)
        relu_default: "f32[512, 32, 1, 1]" = torch.ops.aten.relu.default(convolution_21);  convolution_21 = None
        return relu_default
