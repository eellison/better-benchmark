class GraphModule(torch.nn.Module):
    def forward(self, convolution_28: "f32[512, 120, 1, 1]", cat_8: "f32[512, 120, 28, 28]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:83 in forward, code: return x * self.gate(x_se)
        add_tensor: "f32[512, 120, 1, 1]" = torch.ops.aten.add.Tensor(convolution_28, 3);  convolution_28 = None
        clamp_min_default: "f32[512, 120, 1, 1]" = torch.ops.aten.clamp_min.default(add_tensor, 0);  add_tensor = None
        clamp_max_default: "f32[512, 120, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_default, 6);  clamp_min_default = None
        div_tensor: "f32[512, 120, 1, 1]" = torch.ops.aten.div.Tensor(clamp_max_default, 6);  clamp_max_default = None
        mul_tensor: "f32[512, 120, 28, 28]" = torch.ops.aten.mul.Tensor(cat_8, div_tensor);  cat_8 = div_tensor = None
        return mul_tensor
