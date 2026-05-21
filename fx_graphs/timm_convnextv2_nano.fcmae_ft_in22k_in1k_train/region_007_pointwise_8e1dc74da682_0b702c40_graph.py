class GraphModule(torch.nn.Module):
    def forward(self, convolution_35: "f32[128, 320, 14, 14]", add_65: "f32[128, 320, 14, 14]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/convnext.py:212 in forward, code: x = self.drop_path(x) + self.shortcut(shortcut)
        add_tensor: "f32[128, 320, 14, 14]" = torch.ops.aten.add.Tensor(convolution_35, add_65);  convolution_35 = add_65 = None
        return add_tensor
