class GraphModule(torch.nn.Module):
    def forward(self, convolution_53: "f32[512, 480, 1, 1]", cat_18: "f32[512, 480, 14, 14]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:83 in forward, code: return x * self.gate(x_se)
        add_tensor: "f32[512, 480, 1, 1]" = torch.ops.aten.add.Tensor(convolution_53, 3);  convolution_53 = None
        clamp_min_default: "f32[512, 480, 1, 1]" = torch.ops.aten.clamp_min.default(add_tensor, 0);  add_tensor = None
        clamp_max_default: "f32[512, 480, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_default, 6);  clamp_min_default = None
        div_tensor: "f32[512, 480, 1, 1]" = torch.ops.aten.div.Tensor(clamp_max_default, 6);  clamp_max_default = None
        mul_tensor: "f32[512, 480, 14, 14]" = torch.ops.aten.mul.Tensor(cat_18, div_tensor);  cat_18 = div_tensor = None
        return mul_tensor
