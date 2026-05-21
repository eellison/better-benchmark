class GraphModule(torch.nn.Module):
    def forward(self, relu_1: "f32[128, 128, 1, 1]", full_default: "f32[]", getitem_303: "f32[128, 128, 1, 1]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:61 in forward, code: x_se = self.act(self.bn(x_se))
        le_scalar: "b8[128, 128, 1, 1]" = torch.ops.aten.le.Scalar(relu_1, 0);  relu_1 = None
        where_self: "f32[128, 128, 1, 1]" = torch.ops.aten.where.self(le_scalar, full_default, getitem_303);  le_scalar = full_default = getitem_303 = None
        return where_self
