class GraphModule(torch.nn.Module):
    def forward(self, addmm_19: "f32[32768, 192]", add_166: "f32[512, 64, 192]", primals_204: "f32[192]", primals_205: "f32[192]", primals_206: "f32[576, 192]", _shape_param_0, _shape_param_1):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        reshape_default: "f32[512, 64, 192]" = torch.ops.aten.reshape.default(addmm_19, _shape_param_0);  addmm_19 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        add_tensor: "f32[512, 64, 192]" = torch.ops.aten.add.Tensor(add_166, reshape_default);  add_166 = reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        var_mean_correction = torch.ops.aten.var_mean.correction(add_tensor, [2], correction = 0, keepdim = True)
        getitem: "f32[512, 64, 1]" = var_mean_correction[0]
        getitem_1: "f32[512, 64, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor_1: "f32[512, 64, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt_default: "f32[512, 64, 1]" = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        sub_tensor: "f32[512, 64, 192]" = torch.ops.aten.sub.Tensor(add_tensor, getitem_1);  add_tensor = getitem_1 = None
        mul_tensor: "f32[512, 64, 192]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[512, 64, 192]" = torch.ops.aten.mul.Tensor(mul_tensor, primals_204);  mul_tensor = primals_204 = None
        add_tensor_2: "f32[512, 64, 192]" = torch.ops.aten.add.Tensor(mul_tensor_1, primals_205);  mul_tensor_1 = primals_205 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        reshape_default_1: "f32[32768, 192]" = torch.ops.aten.reshape.default(add_tensor_2, _shape_param_1);  add_tensor_2 = _shape_param_1 = None
        permute_default: "f32[192, 576]" = torch.ops.aten.permute.default(primals_206, [1, 0]);  primals_206 = None
        return (reshape_default_1, permute_default)
