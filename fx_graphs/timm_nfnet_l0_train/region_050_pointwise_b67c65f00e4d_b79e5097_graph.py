class GraphModule(torch.nn.Module):
    def forward(self, relu_3: "f32[128, 384, 1, 1]", full_default: "f32[]", getitem_264: "f32[128, 384, 1, 1]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:61 in forward, code: x_se = self.act(self.bn(x_se))
        le_scalar: "b8[128, 384, 1, 1]" = torch.ops.aten.le.Scalar(relu_3, 0);  relu_3 = None
        where_self: "f32[128, 384, 1, 1]" = torch.ops.aten.where.self(le_scalar, full_default, getitem_264);  le_scalar = full_default = getitem_264 = None
        return where_self
