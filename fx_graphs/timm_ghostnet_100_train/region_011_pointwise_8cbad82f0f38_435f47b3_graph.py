class GraphModule(torch.nn.Module):
    def forward(self, convolution_68: "f32[512, 672, 1, 1]", add_309: "f32[512, 672, 7, 7]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:83 in forward, code: return x * self.gate(x_se)
        add_tensor: "f32[512, 672, 1, 1]" = torch.ops.aten.add.Tensor(convolution_68, 3);  convolution_68 = None
        clamp_min_default: "f32[512, 672, 1, 1]" = torch.ops.aten.clamp_min.default(add_tensor, 0);  add_tensor = None
        clamp_max_default: "f32[512, 672, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_default, 6);  clamp_min_default = None
        div_tensor: "f32[512, 672, 1, 1]" = torch.ops.aten.div.Tensor(clamp_max_default, 6);  clamp_max_default = None
        mul_tensor: "f32[512, 672, 7, 7]" = torch.ops.aten.mul.Tensor(add_309, div_tensor);  add_309 = div_tensor = None
        return mul_tensor
