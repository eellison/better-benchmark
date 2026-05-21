class GraphModule(torch.nn.Module):
    def forward(self, convolution_42: "f32[128, 640, 7, 7]", convolution_39: "f32[128, 640, 7, 7]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/convnext.py:212 in forward, code: x = self.drop_path(x) + self.shortcut(shortcut)
        add_tensor: "f32[128, 640, 7, 7]" = torch.ops.aten.add.Tensor(convolution_42, convolution_39);  convolution_42 = convolution_39 = None
        return add_tensor
