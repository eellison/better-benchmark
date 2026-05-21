class GraphModule(torch.nn.Module):
    def forward(self, addmm_13: "f32[100352, 256]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, iota_4: "i64[28]", inductor_seeds_default: "i64[46]", view_83: "f32[128, 28, 28, 256]", _shape_param_4, primals_61: "f32[256]", primals_62: "f32[256]", _shape_param_5, primals_63: "f32[1024, 256]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        reshape_default: "f32[2048, 49, 256]" = torch.ops.aten.reshape.default(addmm_13, _shape_param_0);  addmm_13 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:465 in _attn, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        reshape_default_1: "f32[2048, 7, 7, 256]" = torch.ops.aten.reshape.default(reshape_default, _shape_param_1);  reshape_default = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:75 in window_reverse, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
        reshape_default_2: "f32[128, 4, 4, 7, 7, 256]" = torch.ops.aten.reshape.default(reshape_default_1, _shape_param_2);  reshape_default_1 = _shape_param_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:76 in window_reverse, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
        permute_default: "f32[128, 4, 7, 4, 7, 256]" = torch.ops.aten.permute.default(reshape_default_2, [0, 1, 3, 2, 4, 5]);  reshape_default_2 = None
        clone_default: "f32[128, 4, 7, 4, 7, 256]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        reshape_default_3: "f32[128, 28, 28, 256]" = torch.ops.aten.reshape.default(clone_default, _shape_param_3);  clone_default = _shape_param_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:471 in _attn, code: x = torch.roll(shifted_x, shifts=self.shift_size, dims=(1, 2))
        add_tensor: "i64[28]" = torch.ops.aten.add.Tensor(iota_4, 25);  iota_4 = None
        fmod_scalar: "i64[28]" = torch.ops.aten.fmod.Scalar(add_tensor, 28);  add_tensor = None
        index_tensor: "f32[128, 28, 28, 256]" = torch.ops.aten.index.Tensor(reshape_default_3, [None, fmod_scalar]);  reshape_default_3 = None
        index_tensor_1: "f32[128, 28, 28, 256]" = torch.ops.aten.index.Tensor(index_tensor, [None, None, fmod_scalar]);  index_tensor = fmod_scalar = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        inductor_lookup_seed_default: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 4);  inductor_seeds_default = None
        inductor_random_default: "f32[128, 1, 1, 1]" = torch.ops.prims.inductor_random.default([128, 1, 1, 1], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        lt_scalar: "b8[128, 1, 1, 1]" = torch.ops.aten.lt.Scalar(inductor_random_default, 0.9869565209373832);  inductor_random_default = None
        convert_element_type_default: "f32[128, 1, 1, 1]" = torch.ops.prims.convert_element_type.default(lt_scalar, torch.float32);  lt_scalar = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_tensor: "f32[128, 1, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_default, 0.9869565209373832);  convert_element_type_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_tensor: "f32[128, 28, 28, 256]" = torch.ops.aten.mul.Tensor(index_tensor_1, div_tensor);  index_tensor_1 = div_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        add_tensor_1: "f32[128, 28, 28, 256]" = torch.ops.aten.add.Tensor(view_83, mul_tensor);  view_83 = mul_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:487 in forward, code: x = x.reshape(B, -1, C)
        reshape_default_4: "f32[128, 784, 256]" = torch.ops.aten.reshape.default(add_tensor_1, _shape_param_4);  add_tensor_1 = _shape_param_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        var_mean_correction = torch.ops.aten.var_mean.correction(reshape_default_4, [2], correction = 0, keepdim = True)
        getitem: "f32[128, 784, 1]" = var_mean_correction[0]
        getitem_1: "f32[128, 784, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor_2: "f32[128, 784, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt_default: "f32[128, 784, 1]" = torch.ops.aten.rsqrt.default(add_tensor_2);  add_tensor_2 = None
        sub_tensor: "f32[128, 784, 256]" = torch.ops.aten.sub.Tensor(reshape_default_4, getitem_1);  reshape_default_4 = getitem_1 = None
        mul_tensor_1: "f32[128, 784, 256]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_2: "f32[128, 784, 256]" = torch.ops.aten.mul.Tensor(mul_tensor_1, primals_61);  mul_tensor_1 = primals_61 = None
        add_tensor_3: "f32[128, 784, 256]" = torch.ops.aten.add.Tensor(mul_tensor_2, primals_62);  mul_tensor_2 = primals_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        reshape_default_5: "f32[100352, 256]" = torch.ops.aten.reshape.default(add_tensor_3, _shape_param_5);  add_tensor_3 = _shape_param_5 = None
        permute_default_1: "f32[256, 1024]" = torch.ops.aten.permute.default(primals_63, [1, 0]);  primals_63 = None
        return (reshape_default_5, permute_default_1)
