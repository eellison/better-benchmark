class GraphModule(torch.nn.Module):
    def forward(self, convolution_23: "f32[128, 240, 1, 1]", div_13: "f32[128, 240, 28, 28]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:83 in forward, code: return x * self.gate(x_se)
        sigmoid_default: "f32[128, 240, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_23);  convolution_23 = None
        mul_tensor: "f32[128, 240, 28, 28]" = torch.ops.aten.mul.Tensor(div_13, sigmoid_default);  div_13 = sigmoid_default = None
        return mul_tensor
