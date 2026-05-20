class GraphModule(torch.nn.Module):
    def forward(self, convolution_18: "f32[128, 144, 1, 1]", div_10: "f32[128, 144, 28, 28]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:83 in forward, code: return x * self.gate(x_se)
        sigmoid_default: "f32[128, 144, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_18);  convolution_18 = None
        mul_tensor: "f32[128, 144, 28, 28]" = torch.ops.aten.mul.Tensor(div_10, sigmoid_default);  div_10 = sigmoid_default = None
        return mul_tensor
