class GraphModule(torch.nn.Module):
    def forward(self, addmm_95: "f32[6272, 1024]", inductor_seeds_default: "i64[46]", view_652: "f32[128, 49, 1024]", primals_362: "f32[1024]", primals_363: "f32[1024]", primals_364: "f32[1000, 1024]", rsqrt_51: "f32[128, 49, 1]", view_644: "f32[4096, 49, 32]", view_638: "f32[4096, 49, 32]", view_639: "f32[4096, 32, 49]", rsqrt_50: "f32[128, 7, 7, 1]", rsqrt_49: "f32[128, 49, 1]", view_618: "f32[4096, 49, 32]", view_612: "f32[4096, 49, 32]", view_613: "f32[4096, 32, 49]", rsqrt_47: "f32[128, 7, 7, 1]", rsqrt_46: "f32[128, 196, 1]", view_588: "f32[8192, 49, 32]", view_580: "f32[8192, 49, 32]", view_581: "f32[8192, 32, 49]", rsqrt_45: "f32[128, 14, 14, 1]", rsqrt_44: "f32[128, 196, 1]", view_560: "f32[8192, 49, 32]", view_554: "f32[8192, 49, 32]", view_555: "f32[8192, 32, 49]", rsqrt_43: "f32[128, 14, 14, 1]", rsqrt_42: "f32[128, 196, 1]", view_534: "f32[8192, 49, 32]", view_526: "f32[8192, 49, 32]", view_527: "f32[8192, 32, 49]", rsqrt_41: "f32[128, 14, 14, 1]", rsqrt_40: "f32[128, 196, 1]", view_506: "f32[8192, 49, 32]", view_500: "f32[8192, 49, 32]", view_501: "f32[8192, 32, 49]", rsqrt_39: "f32[128, 14, 14, 1]", rsqrt_38: "f32[128, 196, 1]", view_480: "f32[8192, 49, 32]", view_472: "f32[8192, 49, 32]", view_473: "f32[8192, 32, 49]", rsqrt_37: "f32[128, 14, 14, 1]", rsqrt_36: "f32[128, 196, 1]", view_452: "f32[8192, 49, 32]", view_446: "f32[8192, 49, 32]", view_447: "f32[8192, 32, 49]", rsqrt_35: "f32[128, 14, 14, 1]", rsqrt_34: "f32[128, 196, 1]", view_426: "f32[8192, 49, 32]", view_418: "f32[8192, 49, 32]", view_419: "f32[8192, 32, 49]", rsqrt_33: "f32[128, 14, 14, 1]", rsqrt_32: "f32[128, 196, 1]", view_398: "f32[8192, 49, 32]", view_392: "f32[8192, 49, 32]", view_393: "f32[8192, 32, 49]", rsqrt_31: "f32[128, 14, 14, 1]", rsqrt_30: "f32[128, 196, 1]", view_372: "f32[8192, 49, 32]", view_364: "f32[8192, 49, 32]", view_365: "f32[8192, 32, 49]", rsqrt_29: "f32[128, 14, 14, 1]", rsqrt_28: "f32[128, 196, 1]", view_344: "f32[8192, 49, 32]", view_338: "f32[8192, 49, 32]", view_339: "f32[8192, 32, 49]", rsqrt_27: "f32[128, 14, 14, 1]", rsqrt_26: "f32[128, 196, 1]", view_318: "f32[8192, 49, 32]", view_310: "f32[8192, 49, 32]", view_311: "f32[8192, 32, 49]", rsqrt_25: "f32[128, 14, 14, 1]", rsqrt_24: "f32[128, 196, 1]", view_290: "f32[8192, 49, 32]", view_284: "f32[8192, 49, 32]", view_285: "f32[8192, 32, 49]", rsqrt_23: "f32[128, 14, 14, 1]", rsqrt_22: "f32[128, 196, 1]", view_264: "f32[8192, 49, 32]", view_256: "f32[8192, 49, 32]", view_257: "f32[8192, 32, 49]", rsqrt_21: "f32[128, 14, 14, 1]", rsqrt_20: "f32[128, 196, 1]", view_236: "f32[8192, 49, 32]", view_230: "f32[8192, 49, 32]", view_231: "f32[8192, 32, 49]", rsqrt_19: "f32[128, 14, 14, 1]", rsqrt_18: "f32[128, 196, 1]", view_210: "f32[8192, 49, 32]", view_202: "f32[8192, 49, 32]", view_203: "f32[8192, 32, 49]", rsqrt_17: "f32[128, 14, 14, 1]", rsqrt_16: "f32[128, 196, 1]", view_182: "f32[8192, 49, 32]", view_176: "f32[8192, 49, 32]", view_177: "f32[8192, 32, 49]", rsqrt_15: "f32[128, 14, 14, 1]", rsqrt_14: "f32[128, 196, 1]", view_156: "f32[8192, 49, 32]", view_148: "f32[8192, 49, 32]", view_149: "f32[8192, 32, 49]", rsqrt_13: "f32[128, 14, 14, 1]", rsqrt_12: "f32[128, 196, 1]", view_128: "f32[8192, 49, 32]", view_122: "f32[8192, 49, 32]", view_123: "f32[8192, 32, 49]", rsqrt_10: "f32[128, 14, 14, 1]", rsqrt_9: "f32[128, 784, 1]", view_98: "f32[16384, 49, 32]", view_90: "f32[16384, 49, 32]", view_91: "f32[16384, 32, 49]", rsqrt_8: "f32[128, 28, 28, 1]", rsqrt_7: "f32[128, 784, 1]", view_70: "f32[16384, 49, 32]", view_64: "f32[16384, 49, 32]", view_65: "f32[16384, 32, 49]", rsqrt_5: "f32[128, 28, 28, 1]", rsqrt_4: "f32[128, 3136, 1]", view_40: "f32[32768, 49, 32]", view_32: "f32[32768, 49, 32]", view_33: "f32[32768, 32, 49]", rsqrt_3: "f32[128, 56, 56, 1]", rsqrt_2: "f32[128, 3136, 1]", view_12: "f32[32768, 49, 32]", view_6: "f32[32768, 49, 32]", view_7: "f32[32768, 32, 49]", _shape_param_0, _shape_param_1):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        reshape_default: "f32[128, 49, 1024]" = torch.ops.aten.reshape.default(addmm_95, _shape_param_0);  addmm_95 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        inductor_lookup_seed_default: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 45);  inductor_seeds_default = None
        inductor_random_default: "f32[128, 1, 1]" = torch.ops.prims.inductor_random.default([128, 1, 1], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        lt_scalar: "b8[128, 1, 1]" = torch.ops.aten.lt.Scalar(inductor_random_default, 0.8999999985098839);  inductor_random_default = None
        convert_element_type_default: "f32[128, 1, 1]" = torch.ops.prims.convert_element_type.default(lt_scalar, torch.float32);  lt_scalar = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_tensor: "f32[128, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_default, 0.8999999985098839);  convert_element_type_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_tensor: "f32[128, 49, 1024]" = torch.ops.aten.mul.Tensor(reshape_default, div_tensor);  reshape_default = div_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        add_tensor: "f32[128, 49, 1024]" = torch.ops.aten.add.Tensor(view_652, mul_tensor);  view_652 = mul_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:489 in forward, code: x = x.reshape(B, H, W, C)
        reshape_default_1: "f32[128, 7, 7, 1024]" = torch.ops.aten.reshape.default(add_tensor, _shape_param_1);  add_tensor = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:981 in forward_features, code: x = self.norm(x)
        var_mean_correction = torch.ops.aten.var_mean.correction(reshape_default_1, [3], correction = 0, keepdim = True)
        getitem: "f32[128, 7, 7, 1]" = var_mean_correction[0]
        getitem_1: "f32[128, 7, 7, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor_1: "f32[128, 7, 7, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt_default: "f32[128, 7, 7, 1]" = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        sub_tensor: "f32[128, 7, 7, 1024]" = torch.ops.aten.sub.Tensor(reshape_default_1, getitem_1);  reshape_default_1 = getitem_1 = None
        mul_tensor_1: "f32[128, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = None
        mul_tensor_2: "f32[128, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor_1, primals_362);  mul_tensor_1 = primals_362 = None
        add_tensor_2: "f32[128, 7, 7, 1024]" = torch.ops.aten.add.Tensor(mul_tensor_2, primals_363);  mul_tensor_2 = primals_363 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/adaptive_avgmax_pool.py:65 in forward, code: return x.mean(self.dim, keepdim=not self.flatten)
        mean_dim: "f32[128, 1024]" = torch.ops.aten.mean.dim(add_tensor_2, [1, 2]);  add_tensor_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/classifier.py:141 in forward, code: x = self.fc(x)
        permute_default: "f32[1024, 1000]" = torch.ops.aten.permute.default(primals_364, [1, 0]);  primals_364 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:981 in forward_features, code: x = self.norm(x)
        div_tensor_1: "f32[128, 7, 7, 1]" = torch.ops.aten.div.Tensor(rsqrt_default, 1024);  rsqrt_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        div_tensor_2: "f32[128, 49, 1]" = torch.ops.aten.div.Tensor(rsqrt_51, 1024);  rsqrt_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        permute_default_1: "f32[4096, 32, 49]" = torch.ops.aten.permute.default(view_644, [0, 2, 1]);  view_644 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_default_2: "f32[4096, 32, 49]" = torch.ops.aten.permute.default(view_638, [0, 2, 1]);  view_638 = None
        permute_default_3: "f32[4096, 49, 32]" = torch.ops.aten.permute.default(view_639, [0, 2, 1]);  view_639 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        div_tensor_3: "f32[128, 7, 7, 1]" = torch.ops.aten.div.Tensor(rsqrt_50, 1024);  rsqrt_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        div_tensor_4: "f32[128, 49, 1]" = torch.ops.aten.div.Tensor(rsqrt_49, 1024);  rsqrt_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        permute_default_4: "f32[4096, 32, 49]" = torch.ops.aten.permute.default(view_618, [0, 2, 1]);  view_618 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_default_5: "f32[4096, 32, 49]" = torch.ops.aten.permute.default(view_612, [0, 2, 1]);  view_612 = None
        permute_default_6: "f32[4096, 49, 32]" = torch.ops.aten.permute.default(view_613, [0, 2, 1]);  view_613 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:540 in forward, code: x = self.norm(x)
        div_tensor_5: "f32[128, 7, 7, 1]" = torch.ops.aten.div.Tensor(rsqrt_47, 2048);  rsqrt_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        div_tensor_6: "f32[128, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_46, 512);  rsqrt_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        permute_default_7: "f32[8192, 32, 49]" = torch.ops.aten.permute.default(view_588, [0, 2, 1]);  view_588 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_default_8: "f32[8192, 32, 49]" = torch.ops.aten.permute.default(view_580, [0, 2, 1]);  view_580 = None
        permute_default_9: "f32[8192, 49, 32]" = torch.ops.aten.permute.default(view_581, [0, 2, 1]);  view_581 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        div_tensor_7: "f32[128, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_45, 512);  rsqrt_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        div_tensor_8: "f32[128, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_44, 512);  rsqrt_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        permute_default_10: "f32[8192, 32, 49]" = torch.ops.aten.permute.default(view_560, [0, 2, 1]);  view_560 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_default_11: "f32[8192, 32, 49]" = torch.ops.aten.permute.default(view_554, [0, 2, 1]);  view_554 = None
        permute_default_12: "f32[8192, 49, 32]" = torch.ops.aten.permute.default(view_555, [0, 2, 1]);  view_555 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        div_tensor_9: "f32[128, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_43, 512);  rsqrt_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        div_tensor_10: "f32[128, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_42, 512);  rsqrt_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        permute_default_13: "f32[8192, 32, 49]" = torch.ops.aten.permute.default(view_534, [0, 2, 1]);  view_534 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_default_14: "f32[8192, 32, 49]" = torch.ops.aten.permute.default(view_526, [0, 2, 1]);  view_526 = None
        permute_default_15: "f32[8192, 49, 32]" = torch.ops.aten.permute.default(view_527, [0, 2, 1]);  view_527 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        div_tensor_11: "f32[128, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_41, 512);  rsqrt_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        div_tensor_12: "f32[128, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_40, 512);  rsqrt_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        permute_default_16: "f32[8192, 32, 49]" = torch.ops.aten.permute.default(view_506, [0, 2, 1]);  view_506 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_default_17: "f32[8192, 32, 49]" = torch.ops.aten.permute.default(view_500, [0, 2, 1]);  view_500 = None
        permute_default_18: "f32[8192, 49, 32]" = torch.ops.aten.permute.default(view_501, [0, 2, 1]);  view_501 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        div_tensor_13: "f32[128, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_39, 512);  rsqrt_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        div_tensor_14: "f32[128, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_38, 512);  rsqrt_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        permute_default_19: "f32[8192, 32, 49]" = torch.ops.aten.permute.default(view_480, [0, 2, 1]);  view_480 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_default_20: "f32[8192, 32, 49]" = torch.ops.aten.permute.default(view_472, [0, 2, 1]);  view_472 = None
        permute_default_21: "f32[8192, 49, 32]" = torch.ops.aten.permute.default(view_473, [0, 2, 1]);  view_473 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        div_tensor_15: "f32[128, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_37, 512);  rsqrt_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        div_tensor_16: "f32[128, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_36, 512);  rsqrt_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        permute_default_22: "f32[8192, 32, 49]" = torch.ops.aten.permute.default(view_452, [0, 2, 1]);  view_452 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_default_23: "f32[8192, 32, 49]" = torch.ops.aten.permute.default(view_446, [0, 2, 1]);  view_446 = None
        permute_default_24: "f32[8192, 49, 32]" = torch.ops.aten.permute.default(view_447, [0, 2, 1]);  view_447 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        div_tensor_17: "f32[128, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_35, 512);  rsqrt_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        div_tensor_18: "f32[128, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_34, 512);  rsqrt_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        permute_default_25: "f32[8192, 32, 49]" = torch.ops.aten.permute.default(view_426, [0, 2, 1]);  view_426 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_default_26: "f32[8192, 32, 49]" = torch.ops.aten.permute.default(view_418, [0, 2, 1]);  view_418 = None
        permute_default_27: "f32[8192, 49, 32]" = torch.ops.aten.permute.default(view_419, [0, 2, 1]);  view_419 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        div_tensor_19: "f32[128, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_33, 512);  rsqrt_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        div_tensor_20: "f32[128, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_32, 512);  rsqrt_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        permute_default_28: "f32[8192, 32, 49]" = torch.ops.aten.permute.default(view_398, [0, 2, 1]);  view_398 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_default_29: "f32[8192, 32, 49]" = torch.ops.aten.permute.default(view_392, [0, 2, 1]);  view_392 = None
        permute_default_30: "f32[8192, 49, 32]" = torch.ops.aten.permute.default(view_393, [0, 2, 1]);  view_393 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        div_tensor_21: "f32[128, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_31, 512);  rsqrt_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        div_tensor_22: "f32[128, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_30, 512);  rsqrt_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        permute_default_31: "f32[8192, 32, 49]" = torch.ops.aten.permute.default(view_372, [0, 2, 1]);  view_372 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_default_32: "f32[8192, 32, 49]" = torch.ops.aten.permute.default(view_364, [0, 2, 1]);  view_364 = None
        permute_default_33: "f32[8192, 49, 32]" = torch.ops.aten.permute.default(view_365, [0, 2, 1]);  view_365 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        div_tensor_23: "f32[128, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_29, 512);  rsqrt_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        div_tensor_24: "f32[128, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_28, 512);  rsqrt_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        permute_default_34: "f32[8192, 32, 49]" = torch.ops.aten.permute.default(view_344, [0, 2, 1]);  view_344 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_default_35: "f32[8192, 32, 49]" = torch.ops.aten.permute.default(view_338, [0, 2, 1]);  view_338 = None
        permute_default_36: "f32[8192, 49, 32]" = torch.ops.aten.permute.default(view_339, [0, 2, 1]);  view_339 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        div_tensor_25: "f32[128, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_27, 512);  rsqrt_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        div_tensor_26: "f32[128, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_26, 512);  rsqrt_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        permute_default_37: "f32[8192, 32, 49]" = torch.ops.aten.permute.default(view_318, [0, 2, 1]);  view_318 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_default_38: "f32[8192, 32, 49]" = torch.ops.aten.permute.default(view_310, [0, 2, 1]);  view_310 = None
        permute_default_39: "f32[8192, 49, 32]" = torch.ops.aten.permute.default(view_311, [0, 2, 1]);  view_311 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        div_tensor_27: "f32[128, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_25, 512);  rsqrt_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        div_tensor_28: "f32[128, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_24, 512);  rsqrt_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        permute_default_40: "f32[8192, 32, 49]" = torch.ops.aten.permute.default(view_290, [0, 2, 1]);  view_290 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_default_41: "f32[8192, 32, 49]" = torch.ops.aten.permute.default(view_284, [0, 2, 1]);  view_284 = None
        permute_default_42: "f32[8192, 49, 32]" = torch.ops.aten.permute.default(view_285, [0, 2, 1]);  view_285 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        div_tensor_29: "f32[128, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_23, 512);  rsqrt_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        div_tensor_30: "f32[128, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_22, 512);  rsqrt_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        permute_default_43: "f32[8192, 32, 49]" = torch.ops.aten.permute.default(view_264, [0, 2, 1]);  view_264 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_default_44: "f32[8192, 32, 49]" = torch.ops.aten.permute.default(view_256, [0, 2, 1]);  view_256 = None
        permute_default_45: "f32[8192, 49, 32]" = torch.ops.aten.permute.default(view_257, [0, 2, 1]);  view_257 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        div_tensor_31: "f32[128, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_21, 512);  rsqrt_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        div_tensor_32: "f32[128, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_20, 512);  rsqrt_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        permute_default_46: "f32[8192, 32, 49]" = torch.ops.aten.permute.default(view_236, [0, 2, 1]);  view_236 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_default_47: "f32[8192, 32, 49]" = torch.ops.aten.permute.default(view_230, [0, 2, 1]);  view_230 = None
        permute_default_48: "f32[8192, 49, 32]" = torch.ops.aten.permute.default(view_231, [0, 2, 1]);  view_231 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        div_tensor_33: "f32[128, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_19, 512);  rsqrt_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        div_tensor_34: "f32[128, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_18, 512);  rsqrt_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        permute_default_49: "f32[8192, 32, 49]" = torch.ops.aten.permute.default(view_210, [0, 2, 1]);  view_210 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_default_50: "f32[8192, 32, 49]" = torch.ops.aten.permute.default(view_202, [0, 2, 1]);  view_202 = None
        permute_default_51: "f32[8192, 49, 32]" = torch.ops.aten.permute.default(view_203, [0, 2, 1]);  view_203 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        div_tensor_35: "f32[128, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_17, 512);  rsqrt_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        div_tensor_36: "f32[128, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_16, 512);  rsqrt_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        permute_default_52: "f32[8192, 32, 49]" = torch.ops.aten.permute.default(view_182, [0, 2, 1]);  view_182 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_default_53: "f32[8192, 32, 49]" = torch.ops.aten.permute.default(view_176, [0, 2, 1]);  view_176 = None
        permute_default_54: "f32[8192, 49, 32]" = torch.ops.aten.permute.default(view_177, [0, 2, 1]);  view_177 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        div_tensor_37: "f32[128, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_15, 512);  rsqrt_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        div_tensor_38: "f32[128, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_14, 512);  rsqrt_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        permute_default_55: "f32[8192, 32, 49]" = torch.ops.aten.permute.default(view_156, [0, 2, 1]);  view_156 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_default_56: "f32[8192, 32, 49]" = torch.ops.aten.permute.default(view_148, [0, 2, 1]);  view_148 = None
        permute_default_57: "f32[8192, 49, 32]" = torch.ops.aten.permute.default(view_149, [0, 2, 1]);  view_149 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        div_tensor_39: "f32[128, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_13, 512);  rsqrt_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        div_tensor_40: "f32[128, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_12, 512);  rsqrt_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        permute_default_58: "f32[8192, 32, 49]" = torch.ops.aten.permute.default(view_128, [0, 2, 1]);  view_128 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_default_59: "f32[8192, 32, 49]" = torch.ops.aten.permute.default(view_122, [0, 2, 1]);  view_122 = None
        permute_default_60: "f32[8192, 49, 32]" = torch.ops.aten.permute.default(view_123, [0, 2, 1]);  view_123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:540 in forward, code: x = self.norm(x)
        div_tensor_41: "f32[128, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_10, 1024);  rsqrt_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        div_tensor_42: "f32[128, 784, 1]" = torch.ops.aten.div.Tensor(rsqrt_9, 256);  rsqrt_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        permute_default_61: "f32[16384, 32, 49]" = torch.ops.aten.permute.default(view_98, [0, 2, 1]);  view_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_default_62: "f32[16384, 32, 49]" = torch.ops.aten.permute.default(view_90, [0, 2, 1]);  view_90 = None
        permute_default_63: "f32[16384, 49, 32]" = torch.ops.aten.permute.default(view_91, [0, 2, 1]);  view_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        div_tensor_43: "f32[128, 28, 28, 1]" = torch.ops.aten.div.Tensor(rsqrt_8, 256);  rsqrt_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        div_tensor_44: "f32[128, 784, 1]" = torch.ops.aten.div.Tensor(rsqrt_7, 256);  rsqrt_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        permute_default_64: "f32[16384, 32, 49]" = torch.ops.aten.permute.default(view_70, [0, 2, 1]);  view_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_default_65: "f32[16384, 32, 49]" = torch.ops.aten.permute.default(view_64, [0, 2, 1]);  view_64 = None
        permute_default_66: "f32[16384, 49, 32]" = torch.ops.aten.permute.default(view_65, [0, 2, 1]);  view_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:540 in forward, code: x = self.norm(x)
        div_tensor_45: "f32[128, 28, 28, 1]" = torch.ops.aten.div.Tensor(rsqrt_5, 512);  rsqrt_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        div_tensor_46: "f32[128, 3136, 1]" = torch.ops.aten.div.Tensor(rsqrt_4, 128);  rsqrt_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        permute_default_67: "f32[32768, 32, 49]" = torch.ops.aten.permute.default(view_40, [0, 2, 1]);  view_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_default_68: "f32[32768, 32, 49]" = torch.ops.aten.permute.default(view_32, [0, 2, 1]);  view_32 = None
        permute_default_69: "f32[32768, 49, 32]" = torch.ops.aten.permute.default(view_33, [0, 2, 1]);  view_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        div_tensor_47: "f32[128, 56, 56, 1]" = torch.ops.aten.div.Tensor(rsqrt_3, 128);  rsqrt_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        div_tensor_48: "f32[128, 3136, 1]" = torch.ops.aten.div.Tensor(rsqrt_2, 128);  rsqrt_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        permute_default_70: "f32[32768, 32, 49]" = torch.ops.aten.permute.default(view_12, [0, 2, 1]);  view_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_default_71: "f32[32768, 32, 49]" = torch.ops.aten.permute.default(view_6, [0, 2, 1]);  view_6 = None
        permute_default_72: "f32[32768, 49, 32]" = torch.ops.aten.permute.default(view_7, [0, 2, 1]);  view_7 = None
        return (mean_dim, permute_default, div_tensor_1, div_tensor_2, permute_default_1, permute_default_2, permute_default_3, div_tensor_3, div_tensor_4, permute_default_4, permute_default_5, permute_default_6, div_tensor_5, div_tensor_6, permute_default_7, permute_default_8, permute_default_9, div_tensor_7, div_tensor_8, permute_default_10, permute_default_11, permute_default_12, div_tensor_9, div_tensor_10, permute_default_13, permute_default_14, permute_default_15, div_tensor_11, div_tensor_12, permute_default_16, permute_default_17, permute_default_18, div_tensor_13, div_tensor_14, permute_default_19, permute_default_20, permute_default_21, div_tensor_15, div_tensor_16, permute_default_22, permute_default_23, permute_default_24, div_tensor_17, div_tensor_18, permute_default_25, permute_default_26, permute_default_27, div_tensor_19, div_tensor_20, permute_default_28, permute_default_29, permute_default_30, div_tensor_21, div_tensor_22, permute_default_31, permute_default_32, permute_default_33, div_tensor_23, div_tensor_24, permute_default_34, permute_default_35, permute_default_36, div_tensor_25, div_tensor_26, permute_default_37, permute_default_38, permute_default_39, div_tensor_27, div_tensor_28, permute_default_40, permute_default_41, permute_default_42, div_tensor_29, div_tensor_30, permute_default_43, permute_default_44, permute_default_45, div_tensor_31, div_tensor_32, permute_default_46, permute_default_47, permute_default_48, div_tensor_33, div_tensor_34, permute_default_49, permute_default_50, permute_default_51, div_tensor_35, div_tensor_36, permute_default_52, permute_default_53, permute_default_54, div_tensor_37, div_tensor_38, permute_default_55, permute_default_56, permute_default_57, div_tensor_39, div_tensor_40, permute_default_58, permute_default_59, permute_default_60, div_tensor_41, div_tensor_42, permute_default_61, permute_default_62, permute_default_63, div_tensor_43, div_tensor_44, permute_default_64, permute_default_65, permute_default_66, div_tensor_45, div_tensor_46, permute_default_67, permute_default_68, permute_default_69, div_tensor_47, div_tensor_48, permute_default_70, permute_default_71, permute_default_72)
