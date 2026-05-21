class GraphModule(torch.nn.Module):
    def forward(self, convolution_10: "f32[128, 160, 28, 28]", convolution_7: "f32[128, 160, 28, 28]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/convnext.py:212 in forward, code: x = self.drop_path(x) + self.shortcut(shortcut)
        add_tensor: "f32[128, 160, 28, 28]" = torch.ops.aten.add.Tensor(convolution_10, convolution_7);  convolution_10 = convolution_7 = None
        return add_tensor
