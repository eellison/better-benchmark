class GraphModule(torch.nn.Module):
    def forward(self, convolution_8: "f32[128, 96, 1, 1]", div_4: "f32[128, 96, 56, 56]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:83 in forward, code: return x * self.gate(x_se)
        sigmoid_default: "f32[128, 96, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_8);  convolution_8 = None
        mul_tensor: "f32[128, 96, 56, 56]" = torch.ops.aten.mul.Tensor(div_4, sigmoid_default);  div_4 = sigmoid_default = None
        return mul_tensor
