class GraphModule(torch.nn.Module):
    def forward(self, convolution_13: "f32[128, 144, 1, 1]", div_7: "f32[128, 144, 56, 56]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:83 in forward, code: return x * self.gate(x_se)
        sigmoid_default: "f32[128, 144, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_13);  convolution_13 = None
        mul_tensor: "f32[128, 144, 56, 56]" = torch.ops.aten.mul.Tensor(div_7, sigmoid_default);  div_7 = sigmoid_default = None
        return mul_tensor
