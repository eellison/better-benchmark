class GraphModule(torch.nn.Module):
    def forward(self, convolution_3: "f32[128, 80, 56, 56]", permute_1: "f32[128, 80, 56, 56]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/convnext.py:212 in forward, code: x = self.drop_path(x) + self.shortcut(shortcut)
        add_tensor: "f32[128, 80, 56, 56]" = torch.ops.aten.add.Tensor(convolution_3, permute_1);  convolution_3 = permute_1 = None
        return add_tensor
