class GraphModule(torch.nn.Module):
    def forward(self, convolution_58: "f32[128, 1536, 14, 14]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:56 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_dim: "f32[128, 1536, 1, 1]" = torch.ops.aten.mean.dim(convolution_58, [2, 3], True);  convolution_58 = None
        return mean_dim
