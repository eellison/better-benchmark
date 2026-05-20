class GraphModule(torch.nn.Module):
    def forward(self, getitem_173: "f32[128, 384, 14, 14]", _shape_param_0, _shape_param_1):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:121 in forward, code: x = x.permute(0, 1, 3, 2).reshape(B, -1, H, W)
        reshape_default: "f32[128, 6, 64, 196]" = torch.ops.aten.reshape.default(getitem_173, _shape_param_0);  getitem_173 = _shape_param_0 = None
        permute_default: "f32[128, 6, 196, 64]" = torch.ops.aten.permute.default(reshape_default, [0, 1, 3, 2]);  reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:119 in forward, code: x = attn @ v
        clone_default: "f32[128, 6, 196, 64]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        reshape_default_1: "f32[768, 196, 64]" = torch.ops.aten.reshape.default(clone_default, _shape_param_1);  clone_default = _shape_param_1 = None
        return reshape_default_1
