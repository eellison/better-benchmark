class GraphModule(torch.nn.Module):
    def forward(self, relu_14: "f32[512, 120, 1, 1]", full_default: "f32[]", getitem_161: "f32[512, 120, 1, 1]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:81 in forward, code: x_se = self.act1(x_se)
        le_scalar: "b8[512, 120, 1, 1]" = torch.ops.aten.le.Scalar(relu_14, 0);  relu_14 = None
        where_self: "f32[512, 120, 1, 1]" = torch.ops.aten.where.self(le_scalar, full_default, getitem_161);  le_scalar = full_default = getitem_161 = None
        return where_self
