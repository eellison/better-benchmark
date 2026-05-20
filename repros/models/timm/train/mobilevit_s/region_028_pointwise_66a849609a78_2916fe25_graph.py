class GraphModule(torch.nn.Module):
    def forward(self, addmm_4: "f32[131072, 432]", _shape_param_0, _shape_param_1):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        reshape_default: "f32[512, 256, 432]" = torch.ops.aten.reshape.default(addmm_4, _shape_param_0);  addmm_4 = _shape_param_0 = None
        reshape_default_1: "f32[512, 256, 3, 4, 36]" = torch.ops.aten.reshape.default(reshape_default, _shape_param_1);  reshape_default = _shape_param_1 = None
        permute_default: "f32[3, 512, 4, 256, 36]" = torch.ops.aten.permute.default(reshape_default_1, [2, 0, 3, 1, 4]);  reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        unbind_int = torch.ops.aten.unbind.int(permute_default);  permute_default = None
        getitem: "f32[512, 4, 256, 36]" = unbind_int[0]
        getitem_1: "f32[512, 4, 256, 36]" = unbind_int[1]
        getitem_2: "f32[512, 4, 256, 36]" = unbind_int[2];  unbind_int = None
        return (getitem, getitem_1, getitem_2)
