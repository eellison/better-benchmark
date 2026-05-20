class GraphModule(torch.nn.Module):
    def forward(self, add_117: "f32[128, 384, 14, 14]", convolution_39: "f32[128, 384, 14, 14]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:177 in forward, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
        add_tensor: "f32[128, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_117, convolution_39);  add_117 = convolution_39 = None
        return add_tensor
