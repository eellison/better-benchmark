class GraphModule(torch.nn.Module):
    def forward(self, convolution_43: "f32[128, 480, 1, 1]", div_25: "f32[128, 480, 14, 14]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:83 in forward, code: return x * self.gate(x_se)
        sigmoid_default: "f32[128, 480, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_43);  convolution_43 = None
        mul_tensor: "f32[128, 480, 14, 14]" = torch.ops.aten.mul.Tensor(div_25, sigmoid_default);  div_25 = sigmoid_default = None
        return mul_tensor
