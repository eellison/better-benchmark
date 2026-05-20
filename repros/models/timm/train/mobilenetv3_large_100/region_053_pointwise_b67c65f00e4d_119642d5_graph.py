class GraphModule(torch.nn.Module):
    def forward(self, relu_7: "f32[512, 24, 1, 1]", full_default: "f32[]", getitem_242: "f32[512, 24, 1, 1]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:81 in forward, code: x_se = self.act1(x_se)
        le_scalar: "b8[512, 24, 1, 1]" = torch.ops.aten.le.Scalar(relu_7, 0);  relu_7 = None
        where_self: "f32[512, 24, 1, 1]" = torch.ops.aten.where.self(le_scalar, full_default, getitem_242);  le_scalar = full_default = getitem_242 = None
        return where_self
