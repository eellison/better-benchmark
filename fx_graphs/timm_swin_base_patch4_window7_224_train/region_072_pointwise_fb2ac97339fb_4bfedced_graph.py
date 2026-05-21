class GraphModule(torch.nn.Module):
    def forward(self, bmm_52: "f32[4096, 49, 32]", _shape_param_0, bmm_54: "f32[4096, 32, 49]", _shape_param_1, bmm_55: "f32[4096, 49, 32]", _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, primals_336: "f32[3072, 1024]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        reshape_default: "f32[128, 32, 49, 32]" = torch.ops.aten.reshape.default(bmm_52, _shape_param_0);  bmm_52 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        reshape_default_1: "f32[128, 32, 32, 49]" = torch.ops.aten.reshape.default(bmm_54, _shape_param_1);  bmm_54 = _shape_param_1 = None
        reshape_default_2: "f32[128, 32, 49, 32]" = torch.ops.aten.reshape.default(bmm_55, _shape_param_2);  bmm_55 = _shape_param_2 = None
        permute_default: "f32[128, 32, 49, 32]" = torch.ops.aten.permute.default(reshape_default_1, [0, 1, 3, 2]);  reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:234 in forward, code: q = q * self.scale
        mul_tensor: "f32[128, 32, 49, 32]" = torch.ops.aten.mul.Tensor(reshape_default_2, 0.1767766952966369);  reshape_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:220 in forward, code: q, k, v = qkv.unbind(0)
        cat_default: "f32[384, 32, 49, 32]" = torch.ops.aten.cat.default([mul_tensor, permute_default, reshape_default]);  mul_tensor = permute_default = reshape_default = None
        reshape_default_3: "f32[3, 128, 32, 49, 32]" = torch.ops.aten.reshape.default(cat_default, _shape_param_3);  cat_default = _shape_param_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        permute_default_1: "f32[128, 49, 3, 32, 32]" = torch.ops.aten.permute.default(reshape_default_3, [1, 3, 0, 2, 4]);  reshape_default_3 = None
        clone_default: "f32[128, 49, 3, 32, 32]" = torch.ops.aten.clone.default(permute_default_1, memory_format = torch.contiguous_format);  permute_default_1 = None
        reshape_default_4: "f32[128, 49, 3072]" = torch.ops.aten.reshape.default(clone_default, _shape_param_4);  clone_default = _shape_param_4 = None
        reshape_default_5: "f32[6272, 3072]" = torch.ops.aten.reshape.default(reshape_default_4, _shape_param_5);  reshape_default_4 = _shape_param_5 = None
        permute_default_2: "f32[1024, 3072]" = torch.ops.aten.permute.default(primals_336, [1, 0]);  primals_336 = None
        permute_default_3: "f32[3072, 1024]" = torch.ops.aten.permute.default(permute_default_2, [1, 0]);  permute_default_2 = None
        return (reshape_default_5, permute_default_3)
