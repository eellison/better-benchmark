class GraphModule(torch.nn.Module):
    def forward(self, relu_11: "f32[128, 384, 1, 1]", getitem_117: "f32[128, 384, 1, 1]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:61 in forward, code: x_se = self.act(self.bn(x_se))
        le_scalar: "b8[128, 384, 1, 1]" = torch.ops.aten.le.Scalar(relu_11, 0);  relu_11 = None
        full_default: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "f32[128, 384, 1, 1]" = torch.ops.aten.where.self(le_scalar, full_default, getitem_117);  le_scalar = full_default = getitem_117 = None
        return where_self
