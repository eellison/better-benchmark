class GraphModule(torch.nn.Module):
    def forward(self, convolution_28: "f32[128, 240, 1, 1]", div_16: "f32[128, 240, 14, 14]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:83 in forward, code: return x * self.gate(x_se)
        sigmoid_default: "f32[128, 240, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_28);  convolution_28 = None
        mul_tensor: "f32[128, 240, 14, 14]" = torch.ops.aten.mul.Tensor(div_16, sigmoid_default);  div_16 = sigmoid_default = None
        return mul_tensor
