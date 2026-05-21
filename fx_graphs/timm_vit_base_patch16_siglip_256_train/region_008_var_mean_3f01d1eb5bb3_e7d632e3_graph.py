class GraphModule(torch.nn.Module):
    def forward(self, addmm_43: "f32[32768, 768]", _shape_param_0, add_73: "f32[128, 256, 768]", primals_137: "f32[768]", primals_138: "f32[768]", _shape_param_1, primals_139: "f32[2304, 768]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        reshape_default: "f32[128, 256, 768]" = torch.ops.aten.reshape.default(addmm_43, _shape_param_0);  addmm_43 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        add_tensor: "f32[128, 256, 768]" = torch.ops.aten.add.Tensor(add_73, reshape_default);  add_73 = reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_correction = torch.ops.aten.var_mean.correction(add_tensor, [2], correction = 0, keepdim = True)
        getitem: "f32[128, 256, 1]" = var_mean_correction[0]
        getitem_1: "f32[128, 256, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor_1: "f32[128, 256, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-06);  getitem = None
        rsqrt_default: "f32[128, 256, 1]" = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        sub_tensor: "f32[128, 256, 768]" = torch.ops.aten.sub.Tensor(add_tensor, getitem_1);  add_tensor = getitem_1 = None
        mul_tensor: "f32[128, 256, 768]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[128, 256, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, primals_137);  mul_tensor = primals_137 = None
        add_tensor_2: "f32[128, 256, 768]" = torch.ops.aten.add.Tensor(mul_tensor_1, primals_138);  mul_tensor_1 = primals_138 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        reshape_default_1: "f32[32768, 768]" = torch.ops.aten.reshape.default(add_tensor_2, _shape_param_1);  add_tensor_2 = _shape_param_1 = None
        permute_default: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_139, [1, 0]);  primals_139 = None
        return (reshape_default_1, permute_default)
