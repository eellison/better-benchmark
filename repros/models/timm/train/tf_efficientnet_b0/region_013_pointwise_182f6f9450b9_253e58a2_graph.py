class GraphModule(torch.nn.Module):
    def forward(self, convolution_53: "f32[128, 672, 1, 1]", div_31: "f32[128, 672, 14, 14]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:83 in forward, code: return x * self.gate(x_se)
        sigmoid_default: "f32[128, 672, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_53);  convolution_53 = None
        mul_tensor: "f32[128, 672, 14, 14]" = torch.ops.aten.mul.Tensor(div_31, sigmoid_default);  div_31 = sigmoid_default = None
        return mul_tensor
