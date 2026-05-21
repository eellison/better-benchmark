class GraphModule(torch.nn.Module):
    def forward(self, relu: "f32[128, 64, 1, 1]", full_default: "f32[]", getitem_324: "f32[128, 64, 1, 1]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:61 in forward, code: x_se = self.act(self.bn(x_se))
        le_scalar: "b8[128, 64, 1, 1]" = torch.ops.aten.le.Scalar(relu, 0);  relu = None
        where_self: "f32[128, 64, 1, 1]" = torch.ops.aten.where.self(le_scalar, full_default, getitem_324);  le_scalar = full_default = getitem_324 = None
        return where_self
