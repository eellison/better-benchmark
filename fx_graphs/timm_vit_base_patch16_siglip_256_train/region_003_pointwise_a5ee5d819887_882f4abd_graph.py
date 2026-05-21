class GraphModule(torch.nn.Module):
    def forward(self, mm: "f32[128, 768]", primals_153: "f32[768]", addmm_48: "f32[32768, 1536]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention_pool.py:102 in forward, code: q = self.q(q_latent).reshape(B, self.latent_len, self.num_heads, self.head_dim).transpose(1, 2)
        reshape_default: "f32[128, 1, 768]" = torch.ops.aten.reshape.default(mm, _shape_param_0);  mm = _shape_param_0 = None
        add_tensor: "f32[128, 1, 768]" = torch.ops.aten.add.Tensor(reshape_default, primals_153);  reshape_default = primals_153 = None
        reshape_default_1: "f32[128, 1, 12, 64]" = torch.ops.aten.reshape.default(add_tensor, _shape_param_1);  add_tensor = _shape_param_1 = None
        permute_default: "f32[128, 12, 1, 64]" = torch.ops.aten.permute.default(reshape_default_1, [0, 2, 1, 3]);  reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention_pool.py:104 in forward, code: kv = self.kv(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        reshape_default_2: "f32[128, 256, 1536]" = torch.ops.aten.reshape.default(addmm_48, _shape_param_2);  addmm_48 = _shape_param_2 = None
        reshape_default_3: "f32[128, 256, 2, 12, 64]" = torch.ops.aten.reshape.default(reshape_default_2, _shape_param_3);  reshape_default_2 = _shape_param_3 = None
        permute_default_1: "f32[2, 128, 12, 256, 64]" = torch.ops.aten.permute.default(reshape_default_3, [2, 0, 3, 1, 4]);  reshape_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention_pool.py:105 in forward, code: k, v = kv.unbind(0)
        unbind_int = torch.ops.aten.unbind.int(permute_default_1);  permute_default_1 = None
        getitem: "f32[128, 12, 256, 64]" = unbind_int[0]
        getitem_1: "f32[128, 12, 256, 64]" = unbind_int[1];  unbind_int = None
        return (permute_default, getitem, getitem_1)
