class GraphModule(torch.nn.Module):
    def forward(self, convolution_3: "f32[128, 32, 1, 1]", div_1: "f32[128, 32, 112, 112]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:83 in forward, code: return x * self.gate(x_se)
        sigmoid_default: "f32[128, 32, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_3);  convolution_3 = None
        mul_tensor: "f32[128, 32, 112, 112]" = torch.ops.aten.mul.Tensor(div_1, sigmoid_default);  div_1 = sigmoid_default = None
        return mul_tensor
