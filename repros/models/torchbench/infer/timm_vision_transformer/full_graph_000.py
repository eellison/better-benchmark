class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "f16[32, 3, 224, 224]", arg1_1: "f16[384, 3, 16, 16]", arg2_1: "f16[384]", arg3_1: "f16[1, 1, 384]", arg4_1: "f16[1, 197, 384]", arg5_1: "f16[384]", arg6_1: "f16[384]", arg7_1: "f16[1152, 384]", arg8_1: "f16[1152]", arg9_1: "f16[384, 384]", arg10_1: "f16[384]", arg11_1: "f16[384]", arg12_1: "f16[384]", arg13_1: "f16[1536, 384]", arg14_1: "f16[1536]", arg15_1: "f16[384, 1536]", arg16_1: "f16[384]", arg17_1: "f16[384]", arg18_1: "f16[384]", arg19_1: "f16[1152, 384]", arg20_1: "f16[1152]", arg21_1: "f16[384, 384]", arg22_1: "f16[384]", arg23_1: "f16[384]", arg24_1: "f16[384]", arg25_1: "f16[1536, 384]", arg26_1: "f16[1536]", arg27_1: "f16[384, 1536]", arg28_1: "f16[384]", arg29_1: "f16[384]", arg30_1: "f16[384]", arg31_1: "f16[1152, 384]", arg32_1: "f16[1152]", arg33_1: "f16[384, 384]", arg34_1: "f16[384]", arg35_1: "f16[384]", arg36_1: "f16[384]", arg37_1: "f16[1536, 384]", arg38_1: "f16[1536]", arg39_1: "f16[384, 1536]", arg40_1: "f16[384]", arg41_1: "f16[384]", arg42_1: "f16[384]", arg43_1: "f16[1152, 384]", arg44_1: "f16[1152]", arg45_1: "f16[384, 384]", arg46_1: "f16[384]", arg47_1: "f16[384]", arg48_1: "f16[384]", arg49_1: "f16[1536, 384]", arg50_1: "f16[1536]", arg51_1: "f16[384, 1536]", arg52_1: "f16[384]", arg53_1: "f16[384]", arg54_1: "f16[384]", arg55_1: "f16[1152, 384]", arg56_1: "f16[1152]", arg57_1: "f16[384, 384]", arg58_1: "f16[384]", arg59_1: "f16[384]", arg60_1: "f16[384]", arg61_1: "f16[1536, 384]", arg62_1: "f16[1536]", arg63_1: "f16[384, 1536]", arg64_1: "f16[384]", arg65_1: "f16[384]", arg66_1: "f16[384]", arg67_1: "f16[1152, 384]", arg68_1: "f16[1152]", arg69_1: "f16[384, 384]", arg70_1: "f16[384]", arg71_1: "f16[384]", arg72_1: "f16[384]", arg73_1: "f16[1536, 384]", arg74_1: "f16[1536]", arg75_1: "f16[384, 1536]", arg76_1: "f16[384]", arg77_1: "f16[384]", arg78_1: "f16[384]", arg79_1: "f16[1152, 384]", arg80_1: "f16[1152]", arg81_1: "f16[384, 384]", arg82_1: "f16[384]", arg83_1: "f16[384]", arg84_1: "f16[384]", arg85_1: "f16[1536, 384]", arg86_1: "f16[1536]", arg87_1: "f16[384, 1536]", arg88_1: "f16[384]", arg89_1: "f16[384]", arg90_1: "f16[384]", arg91_1: "f16[1152, 384]", arg92_1: "f16[1152]", arg93_1: "f16[384, 384]", arg94_1: "f16[384]", arg95_1: "f16[384]", arg96_1: "f16[384]", arg97_1: "f16[1536, 384]", arg98_1: "f16[1536]", arg99_1: "f16[384, 1536]", arg100_1: "f16[384]", arg101_1: "f16[384]", arg102_1: "f16[384]", arg103_1: "f16[1152, 384]", arg104_1: "f16[1152]", arg105_1: "f16[384, 384]", arg106_1: "f16[384]", arg107_1: "f16[384]", arg108_1: "f16[384]", arg109_1: "f16[1536, 384]", arg110_1: "f16[1536]", arg111_1: "f16[384, 1536]", arg112_1: "f16[384]", arg113_1: "f16[384]", arg114_1: "f16[384]", arg115_1: "f16[1152, 384]", arg116_1: "f16[1152]", arg117_1: "f16[384, 384]", arg118_1: "f16[384]", arg119_1: "f16[384]", arg120_1: "f16[384]", arg121_1: "f16[1536, 384]", arg122_1: "f16[1536]", arg123_1: "f16[384, 1536]", arg124_1: "f16[384]", arg125_1: "f16[384]", arg126_1: "f16[384]", arg127_1: "f16[1152, 384]", arg128_1: "f16[1152]", arg129_1: "f16[384, 384]", arg130_1: "f16[384]", arg131_1: "f16[384]", arg132_1: "f16[384]", arg133_1: "f16[1536, 384]", arg134_1: "f16[1536]", arg135_1: "f16[384, 1536]", arg136_1: "f16[384]", arg137_1: "f16[384]", arg138_1: "f16[384]", arg139_1: "f16[1152, 384]", arg140_1: "f16[1152]", arg141_1: "f16[384, 384]", arg142_1: "f16[384]", arg143_1: "f16[384]", arg144_1: "f16[384]", arg145_1: "f16[1536, 384]", arg146_1: "f16[1536]", arg147_1: "f16[384, 1536]", arg148_1: "f16[384]", arg149_1: "f16[384]", arg150_1: "f16[384]", arg151_1: "f16[1000, 384]", arg152_1: "f16[1000]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:1042 in _pos_embed, code: to_cat.append(self.cls_token.expand(x.shape[0], -1, -1))
        expand: "f16[32, 1, 384]" = torch.ops.aten.expand.default(arg3_1, [32, -1, -1]);  arg3_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/patch_embed.py:136 in forward, code: x = self.proj(x)
        convolution: "f16[32, 384, 14, 14]" = torch.ops.aten.convolution.default(arg0_1, arg1_1, arg2_1, [16, 16], [0, 0], [1, 1], False, [0, 0], 1);  arg0_1 = arg1_1 = arg2_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/patch_embed.py:138 in forward, code: x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
        view: "f16[32, 384, 196]" = torch.ops.aten.reshape.default(convolution, [32, 384, 196]);  convolution = None
        permute: "f16[32, 196, 384]" = torch.ops.aten.permute.default(view, [0, 2, 1]);  view = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:1072 in _pos_embed, code: x = torch.cat(to_cat + [x], dim=1)
        cat: "f16[32, 197, 384]" = torch.ops.aten.cat.default([expand, permute], 1);  expand = permute = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:1073 in _pos_embed, code: x = x + pos_embed
        add: "f16[32, 197, 384]" = torch.ops.aten.add.Tensor(cat, arg4_1);  cat = arg4_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type: "f32[32, 197, 384]" = torch.ops.prims.convert_element_type.default(add, torch.float32)
        var_mean = torch.ops.aten.var_mean.correction(convert_element_type, [2], correction = 0, keepdim = True)
        getitem: "f32[32, 197, 1]" = var_mean[0]
        getitem_1: "f32[32, 197, 1]" = var_mean[1];  var_mean = None
        sub: "f32[32, 197, 384]" = torch.ops.aten.sub.Tensor(convert_element_type, getitem_1);  convert_element_type = getitem_1 = None
        add_1: "f32[32, 197, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-06);  getitem = None
        rsqrt: "f32[32, 197, 1]" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
        mul: "f32[32, 197, 384]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
        mul_1: "f32[32, 197, 384]" = torch.ops.aten.mul.Tensor(mul, arg5_1);  mul = arg5_1 = None
        add_2: "f32[32, 197, 384]" = torch.ops.aten.add.Tensor(mul_1, arg6_1);  mul_1 = arg6_1 = None
        convert_element_type_1: "f16[32, 197, 384]" = torch.ops.prims.convert_element_type.default(add_2, torch.float16);  add_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        view_1: "f16[6304, 384]" = torch.ops.aten.reshape.default(convert_element_type_1, [6304, 384]);  convert_element_type_1 = None
        permute_1: "f16[384, 1152]" = torch.ops.aten.permute.default(arg7_1, [1, 0]);  arg7_1 = None
        addmm: "f16[6304, 1152]" = torch.ops.aten.addmm.default(arg8_1, view_1, permute_1);  arg8_1 = view_1 = permute_1 = None
        view_2: "f16[32, 197, 1152]" = torch.ops.aten.reshape.default(addmm, [32, 197, 1152]);  addmm = None
        view_3: "f16[32, 197, 3, 6, 64]" = torch.ops.aten.reshape.default(view_2, [32, 197, 3, 6, 64]);  view_2 = None
        permute_2: "f16[3, 32, 6, 197, 64]" = torch.ops.aten.permute.default(view_3, [2, 0, 3, 1, 4]);  view_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        unbind = torch.ops.aten.unbind.int(permute_2);  permute_2 = None
        getitem_2: "f16[32, 6, 197, 64]" = unbind[0]
        getitem_3: "f16[32, 6, 197, 64]" = unbind[1]
        getitem_4: "f16[32, 6, 197, 64]" = unbind[2];  unbind = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:120 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention = torch.ops.aten._scaled_dot_product_cudnn_attention.default(getitem_2, getitem_3, getitem_4, None, False);  getitem_2 = getitem_3 = getitem_4 = None
        getitem_5: "f16[32, 6, 197, 64]" = _scaled_dot_product_cudnn_attention[0];  _scaled_dot_product_cudnn_attention = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_3: "f16[32, 197, 6, 64]" = torch.ops.aten.permute.default(getitem_5, [0, 2, 1, 3]);  getitem_5 = None
        view_4: "f16[32, 197, 384]" = torch.ops.aten.reshape.default(permute_3, [32, 197, 384]);  permute_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_5: "f16[6304, 384]" = torch.ops.aten.reshape.default(view_4, [6304, 384]);  view_4 = None
        permute_4: "f16[384, 384]" = torch.ops.aten.permute.default(arg9_1, [1, 0]);  arg9_1 = None
        addmm_1: "f16[6304, 384]" = torch.ops.aten.addmm.default(arg10_1, view_5, permute_4);  arg10_1 = view_5 = permute_4 = None
        view_6: "f16[32, 197, 384]" = torch.ops.aten.reshape.default(addmm_1, [32, 197, 384]);  addmm_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        add_3: "f16[32, 197, 384]" = torch.ops.aten.add.Tensor(add, view_6);  add = view_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_8: "f32[32, 197, 384]" = torch.ops.prims.convert_element_type.default(add_3, torch.float32)
        var_mean_1 = torch.ops.aten.var_mean.correction(convert_element_type_8, [2], correction = 0, keepdim = True)
        getitem_14: "f32[32, 197, 1]" = var_mean_1[0]
        getitem_15: "f32[32, 197, 1]" = var_mean_1[1];  var_mean_1 = None
        sub_1: "f32[32, 197, 384]" = torch.ops.aten.sub.Tensor(convert_element_type_8, getitem_15);  convert_element_type_8 = getitem_15 = None
        add_4: "f32[32, 197, 1]" = torch.ops.aten.add.Tensor(getitem_14, 1e-06);  getitem_14 = None
        rsqrt_1: "f32[32, 197, 1]" = torch.ops.aten.rsqrt.default(add_4);  add_4 = None
        mul_2: "f32[32, 197, 384]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = rsqrt_1 = None
        mul_3: "f32[32, 197, 384]" = torch.ops.aten.mul.Tensor(mul_2, arg11_1);  mul_2 = arg11_1 = None
        add_5: "f32[32, 197, 384]" = torch.ops.aten.add.Tensor(mul_3, arg12_1);  mul_3 = arg12_1 = None
        convert_element_type_9: "f16[32, 197, 384]" = torch.ops.prims.convert_element_type.default(add_5, torch.float16);  add_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_7: "f16[6304, 384]" = torch.ops.aten.reshape.default(convert_element_type_9, [6304, 384]);  convert_element_type_9 = None
        permute_5: "f16[384, 1536]" = torch.ops.aten.permute.default(arg13_1, [1, 0]);  arg13_1 = None
        addmm_2: "f16[6304, 1536]" = torch.ops.aten.addmm.default(arg14_1, view_7, permute_5);  arg14_1 = view_7 = permute_5 = None
        view_8: "f16[32, 197, 1536]" = torch.ops.aten.reshape.default(addmm_2, [32, 197, 1536]);  addmm_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_13: "f32[32, 197, 1536]" = torch.ops.prims.convert_element_type.default(view_8, torch.float32);  view_8 = None
        mul_4: "f32[32, 197, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_13, 0.5)
        mul_5: "f32[32, 197, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_13, 0.7071067811865476);  convert_element_type_13 = None
        erf: "f32[32, 197, 1536]" = torch.ops.aten.erf.default(mul_5);  mul_5 = None
        add_6: "f32[32, 197, 1536]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_6: "f32[32, 197, 1536]" = torch.ops.aten.mul.Tensor(mul_4, add_6);  mul_4 = add_6 = None
        convert_element_type_14: "f16[32, 197, 1536]" = torch.ops.prims.convert_element_type.default(mul_6, torch.float16);  mul_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_9: "f16[6304, 1536]" = torch.ops.aten.reshape.default(convert_element_type_14, [6304, 1536]);  convert_element_type_14 = None
        permute_6: "f16[1536, 384]" = torch.ops.aten.permute.default(arg15_1, [1, 0]);  arg15_1 = None
        addmm_3: "f16[6304, 384]" = torch.ops.aten.addmm.default(arg16_1, view_9, permute_6);  arg16_1 = view_9 = permute_6 = None
        view_10: "f16[32, 197, 384]" = torch.ops.aten.reshape.default(addmm_3, [32, 197, 384]);  addmm_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        add_7: "f16[32, 197, 384]" = torch.ops.aten.add.Tensor(add_3, view_10);  add_3 = view_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_18: "f32[32, 197, 384]" = torch.ops.prims.convert_element_type.default(add_7, torch.float32)
        var_mean_2 = torch.ops.aten.var_mean.correction(convert_element_type_18, [2], correction = 0, keepdim = True)
        getitem_16: "f32[32, 197, 1]" = var_mean_2[0]
        getitem_17: "f32[32, 197, 1]" = var_mean_2[1];  var_mean_2 = None
        sub_2: "f32[32, 197, 384]" = torch.ops.aten.sub.Tensor(convert_element_type_18, getitem_17);  convert_element_type_18 = getitem_17 = None
        add_8: "f32[32, 197, 1]" = torch.ops.aten.add.Tensor(getitem_16, 1e-06);  getitem_16 = None
        rsqrt_2: "f32[32, 197, 1]" = torch.ops.aten.rsqrt.default(add_8);  add_8 = None
        mul_7: "f32[32, 197, 384]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_2);  sub_2 = rsqrt_2 = None
        mul_8: "f32[32, 197, 384]" = torch.ops.aten.mul.Tensor(mul_7, arg17_1);  mul_7 = arg17_1 = None
        add_9: "f32[32, 197, 384]" = torch.ops.aten.add.Tensor(mul_8, arg18_1);  mul_8 = arg18_1 = None
        convert_element_type_19: "f16[32, 197, 384]" = torch.ops.prims.convert_element_type.default(add_9, torch.float16);  add_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        view_11: "f16[6304, 384]" = torch.ops.aten.reshape.default(convert_element_type_19, [6304, 384]);  convert_element_type_19 = None
        permute_7: "f16[384, 1152]" = torch.ops.aten.permute.default(arg19_1, [1, 0]);  arg19_1 = None
        addmm_4: "f16[6304, 1152]" = torch.ops.aten.addmm.default(arg20_1, view_11, permute_7);  arg20_1 = view_11 = permute_7 = None
        view_12: "f16[32, 197, 1152]" = torch.ops.aten.reshape.default(addmm_4, [32, 197, 1152]);  addmm_4 = None
        view_13: "f16[32, 197, 3, 6, 64]" = torch.ops.aten.reshape.default(view_12, [32, 197, 3, 6, 64]);  view_12 = None
        permute_8: "f16[3, 32, 6, 197, 64]" = torch.ops.aten.permute.default(view_13, [2, 0, 3, 1, 4]);  view_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        unbind_1 = torch.ops.aten.unbind.int(permute_8);  permute_8 = None
        getitem_18: "f16[32, 6, 197, 64]" = unbind_1[0]
        getitem_19: "f16[32, 6, 197, 64]" = unbind_1[1]
        getitem_20: "f16[32, 6, 197, 64]" = unbind_1[2];  unbind_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:120 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention_1 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(getitem_18, getitem_19, getitem_20, None, False);  getitem_18 = getitem_19 = getitem_20 = None
        getitem_21: "f16[32, 6, 197, 64]" = _scaled_dot_product_cudnn_attention_1[0];  _scaled_dot_product_cudnn_attention_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_9: "f16[32, 197, 6, 64]" = torch.ops.aten.permute.default(getitem_21, [0, 2, 1, 3]);  getitem_21 = None
        view_14: "f16[32, 197, 384]" = torch.ops.aten.reshape.default(permute_9, [32, 197, 384]);  permute_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_15: "f16[6304, 384]" = torch.ops.aten.reshape.default(view_14, [6304, 384]);  view_14 = None
        permute_10: "f16[384, 384]" = torch.ops.aten.permute.default(arg21_1, [1, 0]);  arg21_1 = None
        addmm_5: "f16[6304, 384]" = torch.ops.aten.addmm.default(arg22_1, view_15, permute_10);  arg22_1 = view_15 = permute_10 = None
        view_16: "f16[32, 197, 384]" = torch.ops.aten.reshape.default(addmm_5, [32, 197, 384]);  addmm_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        add_10: "f16[32, 197, 384]" = torch.ops.aten.add.Tensor(add_7, view_16);  add_7 = view_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_26: "f32[32, 197, 384]" = torch.ops.prims.convert_element_type.default(add_10, torch.float32)
        var_mean_3 = torch.ops.aten.var_mean.correction(convert_element_type_26, [2], correction = 0, keepdim = True)
        getitem_30: "f32[32, 197, 1]" = var_mean_3[0]
        getitem_31: "f32[32, 197, 1]" = var_mean_3[1];  var_mean_3 = None
        sub_3: "f32[32, 197, 384]" = torch.ops.aten.sub.Tensor(convert_element_type_26, getitem_31);  convert_element_type_26 = getitem_31 = None
        add_11: "f32[32, 197, 1]" = torch.ops.aten.add.Tensor(getitem_30, 1e-06);  getitem_30 = None
        rsqrt_3: "f32[32, 197, 1]" = torch.ops.aten.rsqrt.default(add_11);  add_11 = None
        mul_9: "f32[32, 197, 384]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_3);  sub_3 = rsqrt_3 = None
        mul_10: "f32[32, 197, 384]" = torch.ops.aten.mul.Tensor(mul_9, arg23_1);  mul_9 = arg23_1 = None
        add_12: "f32[32, 197, 384]" = torch.ops.aten.add.Tensor(mul_10, arg24_1);  mul_10 = arg24_1 = None
        convert_element_type_27: "f16[32, 197, 384]" = torch.ops.prims.convert_element_type.default(add_12, torch.float16);  add_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_17: "f16[6304, 384]" = torch.ops.aten.reshape.default(convert_element_type_27, [6304, 384]);  convert_element_type_27 = None
        permute_11: "f16[384, 1536]" = torch.ops.aten.permute.default(arg25_1, [1, 0]);  arg25_1 = None
        addmm_6: "f16[6304, 1536]" = torch.ops.aten.addmm.default(arg26_1, view_17, permute_11);  arg26_1 = view_17 = permute_11 = None
        view_18: "f16[32, 197, 1536]" = torch.ops.aten.reshape.default(addmm_6, [32, 197, 1536]);  addmm_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_31: "f32[32, 197, 1536]" = torch.ops.prims.convert_element_type.default(view_18, torch.float32);  view_18 = None
        mul_11: "f32[32, 197, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_31, 0.5)
        mul_12: "f32[32, 197, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_31, 0.7071067811865476);  convert_element_type_31 = None
        erf_1: "f32[32, 197, 1536]" = torch.ops.aten.erf.default(mul_12);  mul_12 = None
        add_13: "f32[32, 197, 1536]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
        mul_13: "f32[32, 197, 1536]" = torch.ops.aten.mul.Tensor(mul_11, add_13);  mul_11 = add_13 = None
        convert_element_type_32: "f16[32, 197, 1536]" = torch.ops.prims.convert_element_type.default(mul_13, torch.float16);  mul_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_19: "f16[6304, 1536]" = torch.ops.aten.reshape.default(convert_element_type_32, [6304, 1536]);  convert_element_type_32 = None
        permute_12: "f16[1536, 384]" = torch.ops.aten.permute.default(arg27_1, [1, 0]);  arg27_1 = None
        addmm_7: "f16[6304, 384]" = torch.ops.aten.addmm.default(arg28_1, view_19, permute_12);  arg28_1 = view_19 = permute_12 = None
        view_20: "f16[32, 197, 384]" = torch.ops.aten.reshape.default(addmm_7, [32, 197, 384]);  addmm_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        add_14: "f16[32, 197, 384]" = torch.ops.aten.add.Tensor(add_10, view_20);  add_10 = view_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_36: "f32[32, 197, 384]" = torch.ops.prims.convert_element_type.default(add_14, torch.float32)
        var_mean_4 = torch.ops.aten.var_mean.correction(convert_element_type_36, [2], correction = 0, keepdim = True)
        getitem_32: "f32[32, 197, 1]" = var_mean_4[0]
        getitem_33: "f32[32, 197, 1]" = var_mean_4[1];  var_mean_4 = None
        sub_4: "f32[32, 197, 384]" = torch.ops.aten.sub.Tensor(convert_element_type_36, getitem_33);  convert_element_type_36 = getitem_33 = None
        add_15: "f32[32, 197, 1]" = torch.ops.aten.add.Tensor(getitem_32, 1e-06);  getitem_32 = None
        rsqrt_4: "f32[32, 197, 1]" = torch.ops.aten.rsqrt.default(add_15);  add_15 = None
        mul_14: "f32[32, 197, 384]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_4);  sub_4 = rsqrt_4 = None
        mul_15: "f32[32, 197, 384]" = torch.ops.aten.mul.Tensor(mul_14, arg29_1);  mul_14 = arg29_1 = None
        add_16: "f32[32, 197, 384]" = torch.ops.aten.add.Tensor(mul_15, arg30_1);  mul_15 = arg30_1 = None
        convert_element_type_37: "f16[32, 197, 384]" = torch.ops.prims.convert_element_type.default(add_16, torch.float16);  add_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        view_21: "f16[6304, 384]" = torch.ops.aten.reshape.default(convert_element_type_37, [6304, 384]);  convert_element_type_37 = None
        permute_13: "f16[384, 1152]" = torch.ops.aten.permute.default(arg31_1, [1, 0]);  arg31_1 = None
        addmm_8: "f16[6304, 1152]" = torch.ops.aten.addmm.default(arg32_1, view_21, permute_13);  arg32_1 = view_21 = permute_13 = None
        view_22: "f16[32, 197, 1152]" = torch.ops.aten.reshape.default(addmm_8, [32, 197, 1152]);  addmm_8 = None
        view_23: "f16[32, 197, 3, 6, 64]" = torch.ops.aten.reshape.default(view_22, [32, 197, 3, 6, 64]);  view_22 = None
        permute_14: "f16[3, 32, 6, 197, 64]" = torch.ops.aten.permute.default(view_23, [2, 0, 3, 1, 4]);  view_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        unbind_2 = torch.ops.aten.unbind.int(permute_14);  permute_14 = None
        getitem_34: "f16[32, 6, 197, 64]" = unbind_2[0]
        getitem_35: "f16[32, 6, 197, 64]" = unbind_2[1]
        getitem_36: "f16[32, 6, 197, 64]" = unbind_2[2];  unbind_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:120 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention_2 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(getitem_34, getitem_35, getitem_36, None, False);  getitem_34 = getitem_35 = getitem_36 = None
        getitem_37: "f16[32, 6, 197, 64]" = _scaled_dot_product_cudnn_attention_2[0];  _scaled_dot_product_cudnn_attention_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_15: "f16[32, 197, 6, 64]" = torch.ops.aten.permute.default(getitem_37, [0, 2, 1, 3]);  getitem_37 = None
        view_24: "f16[32, 197, 384]" = torch.ops.aten.reshape.default(permute_15, [32, 197, 384]);  permute_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_25: "f16[6304, 384]" = torch.ops.aten.reshape.default(view_24, [6304, 384]);  view_24 = None
        permute_16: "f16[384, 384]" = torch.ops.aten.permute.default(arg33_1, [1, 0]);  arg33_1 = None
        addmm_9: "f16[6304, 384]" = torch.ops.aten.addmm.default(arg34_1, view_25, permute_16);  arg34_1 = view_25 = permute_16 = None
        view_26: "f16[32, 197, 384]" = torch.ops.aten.reshape.default(addmm_9, [32, 197, 384]);  addmm_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        add_17: "f16[32, 197, 384]" = torch.ops.aten.add.Tensor(add_14, view_26);  add_14 = view_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_44: "f32[32, 197, 384]" = torch.ops.prims.convert_element_type.default(add_17, torch.float32)
        var_mean_5 = torch.ops.aten.var_mean.correction(convert_element_type_44, [2], correction = 0, keepdim = True)
        getitem_46: "f32[32, 197, 1]" = var_mean_5[0]
        getitem_47: "f32[32, 197, 1]" = var_mean_5[1];  var_mean_5 = None
        sub_5: "f32[32, 197, 384]" = torch.ops.aten.sub.Tensor(convert_element_type_44, getitem_47);  convert_element_type_44 = getitem_47 = None
        add_18: "f32[32, 197, 1]" = torch.ops.aten.add.Tensor(getitem_46, 1e-06);  getitem_46 = None
        rsqrt_5: "f32[32, 197, 1]" = torch.ops.aten.rsqrt.default(add_18);  add_18 = None
        mul_16: "f32[32, 197, 384]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_5);  sub_5 = rsqrt_5 = None
        mul_17: "f32[32, 197, 384]" = torch.ops.aten.mul.Tensor(mul_16, arg35_1);  mul_16 = arg35_1 = None
        add_19: "f32[32, 197, 384]" = torch.ops.aten.add.Tensor(mul_17, arg36_1);  mul_17 = arg36_1 = None
        convert_element_type_45: "f16[32, 197, 384]" = torch.ops.prims.convert_element_type.default(add_19, torch.float16);  add_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_27: "f16[6304, 384]" = torch.ops.aten.reshape.default(convert_element_type_45, [6304, 384]);  convert_element_type_45 = None
        permute_17: "f16[384, 1536]" = torch.ops.aten.permute.default(arg37_1, [1, 0]);  arg37_1 = None
        addmm_10: "f16[6304, 1536]" = torch.ops.aten.addmm.default(arg38_1, view_27, permute_17);  arg38_1 = view_27 = permute_17 = None
        view_28: "f16[32, 197, 1536]" = torch.ops.aten.reshape.default(addmm_10, [32, 197, 1536]);  addmm_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_49: "f32[32, 197, 1536]" = torch.ops.prims.convert_element_type.default(view_28, torch.float32);  view_28 = None
        mul_18: "f32[32, 197, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_49, 0.5)
        mul_19: "f32[32, 197, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_49, 0.7071067811865476);  convert_element_type_49 = None
        erf_2: "f32[32, 197, 1536]" = torch.ops.aten.erf.default(mul_19);  mul_19 = None
        add_20: "f32[32, 197, 1536]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
        mul_20: "f32[32, 197, 1536]" = torch.ops.aten.mul.Tensor(mul_18, add_20);  mul_18 = add_20 = None
        convert_element_type_50: "f16[32, 197, 1536]" = torch.ops.prims.convert_element_type.default(mul_20, torch.float16);  mul_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_29: "f16[6304, 1536]" = torch.ops.aten.reshape.default(convert_element_type_50, [6304, 1536]);  convert_element_type_50 = None
        permute_18: "f16[1536, 384]" = torch.ops.aten.permute.default(arg39_1, [1, 0]);  arg39_1 = None
        addmm_11: "f16[6304, 384]" = torch.ops.aten.addmm.default(arg40_1, view_29, permute_18);  arg40_1 = view_29 = permute_18 = None
        view_30: "f16[32, 197, 384]" = torch.ops.aten.reshape.default(addmm_11, [32, 197, 384]);  addmm_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        add_21: "f16[32, 197, 384]" = torch.ops.aten.add.Tensor(add_17, view_30);  add_17 = view_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_54: "f32[32, 197, 384]" = torch.ops.prims.convert_element_type.default(add_21, torch.float32)
        var_mean_6 = torch.ops.aten.var_mean.correction(convert_element_type_54, [2], correction = 0, keepdim = True)
        getitem_48: "f32[32, 197, 1]" = var_mean_6[0]
        getitem_49: "f32[32, 197, 1]" = var_mean_6[1];  var_mean_6 = None
        sub_6: "f32[32, 197, 384]" = torch.ops.aten.sub.Tensor(convert_element_type_54, getitem_49);  convert_element_type_54 = getitem_49 = None
        add_22: "f32[32, 197, 1]" = torch.ops.aten.add.Tensor(getitem_48, 1e-06);  getitem_48 = None
        rsqrt_6: "f32[32, 197, 1]" = torch.ops.aten.rsqrt.default(add_22);  add_22 = None
        mul_21: "f32[32, 197, 384]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_6);  sub_6 = rsqrt_6 = None
        mul_22: "f32[32, 197, 384]" = torch.ops.aten.mul.Tensor(mul_21, arg41_1);  mul_21 = arg41_1 = None
        add_23: "f32[32, 197, 384]" = torch.ops.aten.add.Tensor(mul_22, arg42_1);  mul_22 = arg42_1 = None
        convert_element_type_55: "f16[32, 197, 384]" = torch.ops.prims.convert_element_type.default(add_23, torch.float16);  add_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        view_31: "f16[6304, 384]" = torch.ops.aten.reshape.default(convert_element_type_55, [6304, 384]);  convert_element_type_55 = None
        permute_19: "f16[384, 1152]" = torch.ops.aten.permute.default(arg43_1, [1, 0]);  arg43_1 = None
        addmm_12: "f16[6304, 1152]" = torch.ops.aten.addmm.default(arg44_1, view_31, permute_19);  arg44_1 = view_31 = permute_19 = None
        view_32: "f16[32, 197, 1152]" = torch.ops.aten.reshape.default(addmm_12, [32, 197, 1152]);  addmm_12 = None
        view_33: "f16[32, 197, 3, 6, 64]" = torch.ops.aten.reshape.default(view_32, [32, 197, 3, 6, 64]);  view_32 = None
        permute_20: "f16[3, 32, 6, 197, 64]" = torch.ops.aten.permute.default(view_33, [2, 0, 3, 1, 4]);  view_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        unbind_3 = torch.ops.aten.unbind.int(permute_20);  permute_20 = None
        getitem_50: "f16[32, 6, 197, 64]" = unbind_3[0]
        getitem_51: "f16[32, 6, 197, 64]" = unbind_3[1]
        getitem_52: "f16[32, 6, 197, 64]" = unbind_3[2];  unbind_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:120 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention_3 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(getitem_50, getitem_51, getitem_52, None, False);  getitem_50 = getitem_51 = getitem_52 = None
        getitem_53: "f16[32, 6, 197, 64]" = _scaled_dot_product_cudnn_attention_3[0];  _scaled_dot_product_cudnn_attention_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_21: "f16[32, 197, 6, 64]" = torch.ops.aten.permute.default(getitem_53, [0, 2, 1, 3]);  getitem_53 = None
        view_34: "f16[32, 197, 384]" = torch.ops.aten.reshape.default(permute_21, [32, 197, 384]);  permute_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_35: "f16[6304, 384]" = torch.ops.aten.reshape.default(view_34, [6304, 384]);  view_34 = None
        permute_22: "f16[384, 384]" = torch.ops.aten.permute.default(arg45_1, [1, 0]);  arg45_1 = None
        addmm_13: "f16[6304, 384]" = torch.ops.aten.addmm.default(arg46_1, view_35, permute_22);  arg46_1 = view_35 = permute_22 = None
        view_36: "f16[32, 197, 384]" = torch.ops.aten.reshape.default(addmm_13, [32, 197, 384]);  addmm_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        add_24: "f16[32, 197, 384]" = torch.ops.aten.add.Tensor(add_21, view_36);  add_21 = view_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_62: "f32[32, 197, 384]" = torch.ops.prims.convert_element_type.default(add_24, torch.float32)
        var_mean_7 = torch.ops.aten.var_mean.correction(convert_element_type_62, [2], correction = 0, keepdim = True)
        getitem_62: "f32[32, 197, 1]" = var_mean_7[0]
        getitem_63: "f32[32, 197, 1]" = var_mean_7[1];  var_mean_7 = None
        sub_7: "f32[32, 197, 384]" = torch.ops.aten.sub.Tensor(convert_element_type_62, getitem_63);  convert_element_type_62 = getitem_63 = None
        add_25: "f32[32, 197, 1]" = torch.ops.aten.add.Tensor(getitem_62, 1e-06);  getitem_62 = None
        rsqrt_7: "f32[32, 197, 1]" = torch.ops.aten.rsqrt.default(add_25);  add_25 = None
        mul_23: "f32[32, 197, 384]" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_7);  sub_7 = rsqrt_7 = None
        mul_24: "f32[32, 197, 384]" = torch.ops.aten.mul.Tensor(mul_23, arg47_1);  mul_23 = arg47_1 = None
        add_26: "f32[32, 197, 384]" = torch.ops.aten.add.Tensor(mul_24, arg48_1);  mul_24 = arg48_1 = None
        convert_element_type_63: "f16[32, 197, 384]" = torch.ops.prims.convert_element_type.default(add_26, torch.float16);  add_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_37: "f16[6304, 384]" = torch.ops.aten.reshape.default(convert_element_type_63, [6304, 384]);  convert_element_type_63 = None
        permute_23: "f16[384, 1536]" = torch.ops.aten.permute.default(arg49_1, [1, 0]);  arg49_1 = None
        addmm_14: "f16[6304, 1536]" = torch.ops.aten.addmm.default(arg50_1, view_37, permute_23);  arg50_1 = view_37 = permute_23 = None
        view_38: "f16[32, 197, 1536]" = torch.ops.aten.reshape.default(addmm_14, [32, 197, 1536]);  addmm_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_67: "f32[32, 197, 1536]" = torch.ops.prims.convert_element_type.default(view_38, torch.float32);  view_38 = None
        mul_25: "f32[32, 197, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_67, 0.5)
        mul_26: "f32[32, 197, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_67, 0.7071067811865476);  convert_element_type_67 = None
        erf_3: "f32[32, 197, 1536]" = torch.ops.aten.erf.default(mul_26);  mul_26 = None
        add_27: "f32[32, 197, 1536]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
        mul_27: "f32[32, 197, 1536]" = torch.ops.aten.mul.Tensor(mul_25, add_27);  mul_25 = add_27 = None
        convert_element_type_68: "f16[32, 197, 1536]" = torch.ops.prims.convert_element_type.default(mul_27, torch.float16);  mul_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_39: "f16[6304, 1536]" = torch.ops.aten.reshape.default(convert_element_type_68, [6304, 1536]);  convert_element_type_68 = None
        permute_24: "f16[1536, 384]" = torch.ops.aten.permute.default(arg51_1, [1, 0]);  arg51_1 = None
        addmm_15: "f16[6304, 384]" = torch.ops.aten.addmm.default(arg52_1, view_39, permute_24);  arg52_1 = view_39 = permute_24 = None
        view_40: "f16[32, 197, 384]" = torch.ops.aten.reshape.default(addmm_15, [32, 197, 384]);  addmm_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        add_28: "f16[32, 197, 384]" = torch.ops.aten.add.Tensor(add_24, view_40);  add_24 = view_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_72: "f32[32, 197, 384]" = torch.ops.prims.convert_element_type.default(add_28, torch.float32)
        var_mean_8 = torch.ops.aten.var_mean.correction(convert_element_type_72, [2], correction = 0, keepdim = True)
        getitem_64: "f32[32, 197, 1]" = var_mean_8[0]
        getitem_65: "f32[32, 197, 1]" = var_mean_8[1];  var_mean_8 = None
        sub_8: "f32[32, 197, 384]" = torch.ops.aten.sub.Tensor(convert_element_type_72, getitem_65);  convert_element_type_72 = getitem_65 = None
        add_29: "f32[32, 197, 1]" = torch.ops.aten.add.Tensor(getitem_64, 1e-06);  getitem_64 = None
        rsqrt_8: "f32[32, 197, 1]" = torch.ops.aten.rsqrt.default(add_29);  add_29 = None
        mul_28: "f32[32, 197, 384]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_8);  sub_8 = rsqrt_8 = None
        mul_29: "f32[32, 197, 384]" = torch.ops.aten.mul.Tensor(mul_28, arg53_1);  mul_28 = arg53_1 = None
        add_30: "f32[32, 197, 384]" = torch.ops.aten.add.Tensor(mul_29, arg54_1);  mul_29 = arg54_1 = None
        convert_element_type_73: "f16[32, 197, 384]" = torch.ops.prims.convert_element_type.default(add_30, torch.float16);  add_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        view_41: "f16[6304, 384]" = torch.ops.aten.reshape.default(convert_element_type_73, [6304, 384]);  convert_element_type_73 = None
        permute_25: "f16[384, 1152]" = torch.ops.aten.permute.default(arg55_1, [1, 0]);  arg55_1 = None
        addmm_16: "f16[6304, 1152]" = torch.ops.aten.addmm.default(arg56_1, view_41, permute_25);  arg56_1 = view_41 = permute_25 = None
        view_42: "f16[32, 197, 1152]" = torch.ops.aten.reshape.default(addmm_16, [32, 197, 1152]);  addmm_16 = None
        view_43: "f16[32, 197, 3, 6, 64]" = torch.ops.aten.reshape.default(view_42, [32, 197, 3, 6, 64]);  view_42 = None
        permute_26: "f16[3, 32, 6, 197, 64]" = torch.ops.aten.permute.default(view_43, [2, 0, 3, 1, 4]);  view_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        unbind_4 = torch.ops.aten.unbind.int(permute_26);  permute_26 = None
        getitem_66: "f16[32, 6, 197, 64]" = unbind_4[0]
        getitem_67: "f16[32, 6, 197, 64]" = unbind_4[1]
        getitem_68: "f16[32, 6, 197, 64]" = unbind_4[2];  unbind_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:120 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention_4 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(getitem_66, getitem_67, getitem_68, None, False);  getitem_66 = getitem_67 = getitem_68 = None
        getitem_69: "f16[32, 6, 197, 64]" = _scaled_dot_product_cudnn_attention_4[0];  _scaled_dot_product_cudnn_attention_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_27: "f16[32, 197, 6, 64]" = torch.ops.aten.permute.default(getitem_69, [0, 2, 1, 3]);  getitem_69 = None
        view_44: "f16[32, 197, 384]" = torch.ops.aten.reshape.default(permute_27, [32, 197, 384]);  permute_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_45: "f16[6304, 384]" = torch.ops.aten.reshape.default(view_44, [6304, 384]);  view_44 = None
        permute_28: "f16[384, 384]" = torch.ops.aten.permute.default(arg57_1, [1, 0]);  arg57_1 = None
        addmm_17: "f16[6304, 384]" = torch.ops.aten.addmm.default(arg58_1, view_45, permute_28);  arg58_1 = view_45 = permute_28 = None
        view_46: "f16[32, 197, 384]" = torch.ops.aten.reshape.default(addmm_17, [32, 197, 384]);  addmm_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        add_31: "f16[32, 197, 384]" = torch.ops.aten.add.Tensor(add_28, view_46);  add_28 = view_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_80: "f32[32, 197, 384]" = torch.ops.prims.convert_element_type.default(add_31, torch.float32)
        var_mean_9 = torch.ops.aten.var_mean.correction(convert_element_type_80, [2], correction = 0, keepdim = True)
        getitem_78: "f32[32, 197, 1]" = var_mean_9[0]
        getitem_79: "f32[32, 197, 1]" = var_mean_9[1];  var_mean_9 = None
        sub_9: "f32[32, 197, 384]" = torch.ops.aten.sub.Tensor(convert_element_type_80, getitem_79);  convert_element_type_80 = getitem_79 = None
        add_32: "f32[32, 197, 1]" = torch.ops.aten.add.Tensor(getitem_78, 1e-06);  getitem_78 = None
        rsqrt_9: "f32[32, 197, 1]" = torch.ops.aten.rsqrt.default(add_32);  add_32 = None
        mul_30: "f32[32, 197, 384]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_9);  sub_9 = rsqrt_9 = None
        mul_31: "f32[32, 197, 384]" = torch.ops.aten.mul.Tensor(mul_30, arg59_1);  mul_30 = arg59_1 = None
        add_33: "f32[32, 197, 384]" = torch.ops.aten.add.Tensor(mul_31, arg60_1);  mul_31 = arg60_1 = None
        convert_element_type_81: "f16[32, 197, 384]" = torch.ops.prims.convert_element_type.default(add_33, torch.float16);  add_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_47: "f16[6304, 384]" = torch.ops.aten.reshape.default(convert_element_type_81, [6304, 384]);  convert_element_type_81 = None
        permute_29: "f16[384, 1536]" = torch.ops.aten.permute.default(arg61_1, [1, 0]);  arg61_1 = None
        addmm_18: "f16[6304, 1536]" = torch.ops.aten.addmm.default(arg62_1, view_47, permute_29);  arg62_1 = view_47 = permute_29 = None
        view_48: "f16[32, 197, 1536]" = torch.ops.aten.reshape.default(addmm_18, [32, 197, 1536]);  addmm_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_85: "f32[32, 197, 1536]" = torch.ops.prims.convert_element_type.default(view_48, torch.float32);  view_48 = None
        mul_32: "f32[32, 197, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_85, 0.5)
        mul_33: "f32[32, 197, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_85, 0.7071067811865476);  convert_element_type_85 = None
        erf_4: "f32[32, 197, 1536]" = torch.ops.aten.erf.default(mul_33);  mul_33 = None
        add_34: "f32[32, 197, 1536]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
        mul_34: "f32[32, 197, 1536]" = torch.ops.aten.mul.Tensor(mul_32, add_34);  mul_32 = add_34 = None
        convert_element_type_86: "f16[32, 197, 1536]" = torch.ops.prims.convert_element_type.default(mul_34, torch.float16);  mul_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_49: "f16[6304, 1536]" = torch.ops.aten.reshape.default(convert_element_type_86, [6304, 1536]);  convert_element_type_86 = None
        permute_30: "f16[1536, 384]" = torch.ops.aten.permute.default(arg63_1, [1, 0]);  arg63_1 = None
        addmm_19: "f16[6304, 384]" = torch.ops.aten.addmm.default(arg64_1, view_49, permute_30);  arg64_1 = view_49 = permute_30 = None
        view_50: "f16[32, 197, 384]" = torch.ops.aten.reshape.default(addmm_19, [32, 197, 384]);  addmm_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        add_35: "f16[32, 197, 384]" = torch.ops.aten.add.Tensor(add_31, view_50);  add_31 = view_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_90: "f32[32, 197, 384]" = torch.ops.prims.convert_element_type.default(add_35, torch.float32)
        var_mean_10 = torch.ops.aten.var_mean.correction(convert_element_type_90, [2], correction = 0, keepdim = True)
        getitem_80: "f32[32, 197, 1]" = var_mean_10[0]
        getitem_81: "f32[32, 197, 1]" = var_mean_10[1];  var_mean_10 = None
        sub_10: "f32[32, 197, 384]" = torch.ops.aten.sub.Tensor(convert_element_type_90, getitem_81);  convert_element_type_90 = getitem_81 = None
        add_36: "f32[32, 197, 1]" = torch.ops.aten.add.Tensor(getitem_80, 1e-06);  getitem_80 = None
        rsqrt_10: "f32[32, 197, 1]" = torch.ops.aten.rsqrt.default(add_36);  add_36 = None
        mul_35: "f32[32, 197, 384]" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_10);  sub_10 = rsqrt_10 = None
        mul_36: "f32[32, 197, 384]" = torch.ops.aten.mul.Tensor(mul_35, arg65_1);  mul_35 = arg65_1 = None
        add_37: "f32[32, 197, 384]" = torch.ops.aten.add.Tensor(mul_36, arg66_1);  mul_36 = arg66_1 = None
        convert_element_type_91: "f16[32, 197, 384]" = torch.ops.prims.convert_element_type.default(add_37, torch.float16);  add_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        view_51: "f16[6304, 384]" = torch.ops.aten.reshape.default(convert_element_type_91, [6304, 384]);  convert_element_type_91 = None
        permute_31: "f16[384, 1152]" = torch.ops.aten.permute.default(arg67_1, [1, 0]);  arg67_1 = None
        addmm_20: "f16[6304, 1152]" = torch.ops.aten.addmm.default(arg68_1, view_51, permute_31);  arg68_1 = view_51 = permute_31 = None
        view_52: "f16[32, 197, 1152]" = torch.ops.aten.reshape.default(addmm_20, [32, 197, 1152]);  addmm_20 = None
        view_53: "f16[32, 197, 3, 6, 64]" = torch.ops.aten.reshape.default(view_52, [32, 197, 3, 6, 64]);  view_52 = None
        permute_32: "f16[3, 32, 6, 197, 64]" = torch.ops.aten.permute.default(view_53, [2, 0, 3, 1, 4]);  view_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        unbind_5 = torch.ops.aten.unbind.int(permute_32);  permute_32 = None
        getitem_82: "f16[32, 6, 197, 64]" = unbind_5[0]
        getitem_83: "f16[32, 6, 197, 64]" = unbind_5[1]
        getitem_84: "f16[32, 6, 197, 64]" = unbind_5[2];  unbind_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:120 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention_5 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(getitem_82, getitem_83, getitem_84, None, False);  getitem_82 = getitem_83 = getitem_84 = None
        getitem_85: "f16[32, 6, 197, 64]" = _scaled_dot_product_cudnn_attention_5[0];  _scaled_dot_product_cudnn_attention_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_33: "f16[32, 197, 6, 64]" = torch.ops.aten.permute.default(getitem_85, [0, 2, 1, 3]);  getitem_85 = None
        view_54: "f16[32, 197, 384]" = torch.ops.aten.reshape.default(permute_33, [32, 197, 384]);  permute_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_55: "f16[6304, 384]" = torch.ops.aten.reshape.default(view_54, [6304, 384]);  view_54 = None
        permute_34: "f16[384, 384]" = torch.ops.aten.permute.default(arg69_1, [1, 0]);  arg69_1 = None
        addmm_21: "f16[6304, 384]" = torch.ops.aten.addmm.default(arg70_1, view_55, permute_34);  arg70_1 = view_55 = permute_34 = None
        view_56: "f16[32, 197, 384]" = torch.ops.aten.reshape.default(addmm_21, [32, 197, 384]);  addmm_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        add_38: "f16[32, 197, 384]" = torch.ops.aten.add.Tensor(add_35, view_56);  add_35 = view_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_98: "f32[32, 197, 384]" = torch.ops.prims.convert_element_type.default(add_38, torch.float32)
        var_mean_11 = torch.ops.aten.var_mean.correction(convert_element_type_98, [2], correction = 0, keepdim = True)
        getitem_94: "f32[32, 197, 1]" = var_mean_11[0]
        getitem_95: "f32[32, 197, 1]" = var_mean_11[1];  var_mean_11 = None
        sub_11: "f32[32, 197, 384]" = torch.ops.aten.sub.Tensor(convert_element_type_98, getitem_95);  convert_element_type_98 = getitem_95 = None
        add_39: "f32[32, 197, 1]" = torch.ops.aten.add.Tensor(getitem_94, 1e-06);  getitem_94 = None
        rsqrt_11: "f32[32, 197, 1]" = torch.ops.aten.rsqrt.default(add_39);  add_39 = None
        mul_37: "f32[32, 197, 384]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_11);  sub_11 = rsqrt_11 = None
        mul_38: "f32[32, 197, 384]" = torch.ops.aten.mul.Tensor(mul_37, arg71_1);  mul_37 = arg71_1 = None
        add_40: "f32[32, 197, 384]" = torch.ops.aten.add.Tensor(mul_38, arg72_1);  mul_38 = arg72_1 = None
        convert_element_type_99: "f16[32, 197, 384]" = torch.ops.prims.convert_element_type.default(add_40, torch.float16);  add_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_57: "f16[6304, 384]" = torch.ops.aten.reshape.default(convert_element_type_99, [6304, 384]);  convert_element_type_99 = None
        permute_35: "f16[384, 1536]" = torch.ops.aten.permute.default(arg73_1, [1, 0]);  arg73_1 = None
        addmm_22: "f16[6304, 1536]" = torch.ops.aten.addmm.default(arg74_1, view_57, permute_35);  arg74_1 = view_57 = permute_35 = None
        view_58: "f16[32, 197, 1536]" = torch.ops.aten.reshape.default(addmm_22, [32, 197, 1536]);  addmm_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_103: "f32[32, 197, 1536]" = torch.ops.prims.convert_element_type.default(view_58, torch.float32);  view_58 = None
        mul_39: "f32[32, 197, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_103, 0.5)
        mul_40: "f32[32, 197, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_103, 0.7071067811865476);  convert_element_type_103 = None
        erf_5: "f32[32, 197, 1536]" = torch.ops.aten.erf.default(mul_40);  mul_40 = None
        add_41: "f32[32, 197, 1536]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
        mul_41: "f32[32, 197, 1536]" = torch.ops.aten.mul.Tensor(mul_39, add_41);  mul_39 = add_41 = None
        convert_element_type_104: "f16[32, 197, 1536]" = torch.ops.prims.convert_element_type.default(mul_41, torch.float16);  mul_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_59: "f16[6304, 1536]" = torch.ops.aten.reshape.default(convert_element_type_104, [6304, 1536]);  convert_element_type_104 = None
        permute_36: "f16[1536, 384]" = torch.ops.aten.permute.default(arg75_1, [1, 0]);  arg75_1 = None
        addmm_23: "f16[6304, 384]" = torch.ops.aten.addmm.default(arg76_1, view_59, permute_36);  arg76_1 = view_59 = permute_36 = None
        view_60: "f16[32, 197, 384]" = torch.ops.aten.reshape.default(addmm_23, [32, 197, 384]);  addmm_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        add_42: "f16[32, 197, 384]" = torch.ops.aten.add.Tensor(add_38, view_60);  add_38 = view_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_108: "f32[32, 197, 384]" = torch.ops.prims.convert_element_type.default(add_42, torch.float32)
        var_mean_12 = torch.ops.aten.var_mean.correction(convert_element_type_108, [2], correction = 0, keepdim = True)
        getitem_96: "f32[32, 197, 1]" = var_mean_12[0]
        getitem_97: "f32[32, 197, 1]" = var_mean_12[1];  var_mean_12 = None
        sub_12: "f32[32, 197, 384]" = torch.ops.aten.sub.Tensor(convert_element_type_108, getitem_97);  convert_element_type_108 = getitem_97 = None
        add_43: "f32[32, 197, 1]" = torch.ops.aten.add.Tensor(getitem_96, 1e-06);  getitem_96 = None
        rsqrt_12: "f32[32, 197, 1]" = torch.ops.aten.rsqrt.default(add_43);  add_43 = None
        mul_42: "f32[32, 197, 384]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_12);  sub_12 = rsqrt_12 = None
        mul_43: "f32[32, 197, 384]" = torch.ops.aten.mul.Tensor(mul_42, arg77_1);  mul_42 = arg77_1 = None
        add_44: "f32[32, 197, 384]" = torch.ops.aten.add.Tensor(mul_43, arg78_1);  mul_43 = arg78_1 = None
        convert_element_type_109: "f16[32, 197, 384]" = torch.ops.prims.convert_element_type.default(add_44, torch.float16);  add_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        view_61: "f16[6304, 384]" = torch.ops.aten.reshape.default(convert_element_type_109, [6304, 384]);  convert_element_type_109 = None
        permute_37: "f16[384, 1152]" = torch.ops.aten.permute.default(arg79_1, [1, 0]);  arg79_1 = None
        addmm_24: "f16[6304, 1152]" = torch.ops.aten.addmm.default(arg80_1, view_61, permute_37);  arg80_1 = view_61 = permute_37 = None
        view_62: "f16[32, 197, 1152]" = torch.ops.aten.reshape.default(addmm_24, [32, 197, 1152]);  addmm_24 = None
        view_63: "f16[32, 197, 3, 6, 64]" = torch.ops.aten.reshape.default(view_62, [32, 197, 3, 6, 64]);  view_62 = None
        permute_38: "f16[3, 32, 6, 197, 64]" = torch.ops.aten.permute.default(view_63, [2, 0, 3, 1, 4]);  view_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        unbind_6 = torch.ops.aten.unbind.int(permute_38);  permute_38 = None
        getitem_98: "f16[32, 6, 197, 64]" = unbind_6[0]
        getitem_99: "f16[32, 6, 197, 64]" = unbind_6[1]
        getitem_100: "f16[32, 6, 197, 64]" = unbind_6[2];  unbind_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:120 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention_6 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(getitem_98, getitem_99, getitem_100, None, False);  getitem_98 = getitem_99 = getitem_100 = None
        getitem_101: "f16[32, 6, 197, 64]" = _scaled_dot_product_cudnn_attention_6[0];  _scaled_dot_product_cudnn_attention_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_39: "f16[32, 197, 6, 64]" = torch.ops.aten.permute.default(getitem_101, [0, 2, 1, 3]);  getitem_101 = None
        view_64: "f16[32, 197, 384]" = torch.ops.aten.reshape.default(permute_39, [32, 197, 384]);  permute_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_65: "f16[6304, 384]" = torch.ops.aten.reshape.default(view_64, [6304, 384]);  view_64 = None
        permute_40: "f16[384, 384]" = torch.ops.aten.permute.default(arg81_1, [1, 0]);  arg81_1 = None
        addmm_25: "f16[6304, 384]" = torch.ops.aten.addmm.default(arg82_1, view_65, permute_40);  arg82_1 = view_65 = permute_40 = None
        view_66: "f16[32, 197, 384]" = torch.ops.aten.reshape.default(addmm_25, [32, 197, 384]);  addmm_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        add_45: "f16[32, 197, 384]" = torch.ops.aten.add.Tensor(add_42, view_66);  add_42 = view_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_116: "f32[32, 197, 384]" = torch.ops.prims.convert_element_type.default(add_45, torch.float32)
        var_mean_13 = torch.ops.aten.var_mean.correction(convert_element_type_116, [2], correction = 0, keepdim = True)
        getitem_110: "f32[32, 197, 1]" = var_mean_13[0]
        getitem_111: "f32[32, 197, 1]" = var_mean_13[1];  var_mean_13 = None
        sub_13: "f32[32, 197, 384]" = torch.ops.aten.sub.Tensor(convert_element_type_116, getitem_111);  convert_element_type_116 = getitem_111 = None
        add_46: "f32[32, 197, 1]" = torch.ops.aten.add.Tensor(getitem_110, 1e-06);  getitem_110 = None
        rsqrt_13: "f32[32, 197, 1]" = torch.ops.aten.rsqrt.default(add_46);  add_46 = None
        mul_44: "f32[32, 197, 384]" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_13);  sub_13 = rsqrt_13 = None
        mul_45: "f32[32, 197, 384]" = torch.ops.aten.mul.Tensor(mul_44, arg83_1);  mul_44 = arg83_1 = None
        add_47: "f32[32, 197, 384]" = torch.ops.aten.add.Tensor(mul_45, arg84_1);  mul_45 = arg84_1 = None
        convert_element_type_117: "f16[32, 197, 384]" = torch.ops.prims.convert_element_type.default(add_47, torch.float16);  add_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_67: "f16[6304, 384]" = torch.ops.aten.reshape.default(convert_element_type_117, [6304, 384]);  convert_element_type_117 = None
        permute_41: "f16[384, 1536]" = torch.ops.aten.permute.default(arg85_1, [1, 0]);  arg85_1 = None
        addmm_26: "f16[6304, 1536]" = torch.ops.aten.addmm.default(arg86_1, view_67, permute_41);  arg86_1 = view_67 = permute_41 = None
        view_68: "f16[32, 197, 1536]" = torch.ops.aten.reshape.default(addmm_26, [32, 197, 1536]);  addmm_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_121: "f32[32, 197, 1536]" = torch.ops.prims.convert_element_type.default(view_68, torch.float32);  view_68 = None
        mul_46: "f32[32, 197, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_121, 0.5)
        mul_47: "f32[32, 197, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_121, 0.7071067811865476);  convert_element_type_121 = None
        erf_6: "f32[32, 197, 1536]" = torch.ops.aten.erf.default(mul_47);  mul_47 = None
        add_48: "f32[32, 197, 1536]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
        mul_48: "f32[32, 197, 1536]" = torch.ops.aten.mul.Tensor(mul_46, add_48);  mul_46 = add_48 = None
        convert_element_type_122: "f16[32, 197, 1536]" = torch.ops.prims.convert_element_type.default(mul_48, torch.float16);  mul_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_69: "f16[6304, 1536]" = torch.ops.aten.reshape.default(convert_element_type_122, [6304, 1536]);  convert_element_type_122 = None
        permute_42: "f16[1536, 384]" = torch.ops.aten.permute.default(arg87_1, [1, 0]);  arg87_1 = None
        addmm_27: "f16[6304, 384]" = torch.ops.aten.addmm.default(arg88_1, view_69, permute_42);  arg88_1 = view_69 = permute_42 = None
        view_70: "f16[32, 197, 384]" = torch.ops.aten.reshape.default(addmm_27, [32, 197, 384]);  addmm_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        add_49: "f16[32, 197, 384]" = torch.ops.aten.add.Tensor(add_45, view_70);  add_45 = view_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_126: "f32[32, 197, 384]" = torch.ops.prims.convert_element_type.default(add_49, torch.float32)
        var_mean_14 = torch.ops.aten.var_mean.correction(convert_element_type_126, [2], correction = 0, keepdim = True)
        getitem_112: "f32[32, 197, 1]" = var_mean_14[0]
        getitem_113: "f32[32, 197, 1]" = var_mean_14[1];  var_mean_14 = None
        sub_14: "f32[32, 197, 384]" = torch.ops.aten.sub.Tensor(convert_element_type_126, getitem_113);  convert_element_type_126 = getitem_113 = None
        add_50: "f32[32, 197, 1]" = torch.ops.aten.add.Tensor(getitem_112, 1e-06);  getitem_112 = None
        rsqrt_14: "f32[32, 197, 1]" = torch.ops.aten.rsqrt.default(add_50);  add_50 = None
        mul_49: "f32[32, 197, 384]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_14);  sub_14 = rsqrt_14 = None
        mul_50: "f32[32, 197, 384]" = torch.ops.aten.mul.Tensor(mul_49, arg89_1);  mul_49 = arg89_1 = None
        add_51: "f32[32, 197, 384]" = torch.ops.aten.add.Tensor(mul_50, arg90_1);  mul_50 = arg90_1 = None
        convert_element_type_127: "f16[32, 197, 384]" = torch.ops.prims.convert_element_type.default(add_51, torch.float16);  add_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        view_71: "f16[6304, 384]" = torch.ops.aten.reshape.default(convert_element_type_127, [6304, 384]);  convert_element_type_127 = None
        permute_43: "f16[384, 1152]" = torch.ops.aten.permute.default(arg91_1, [1, 0]);  arg91_1 = None
        addmm_28: "f16[6304, 1152]" = torch.ops.aten.addmm.default(arg92_1, view_71, permute_43);  arg92_1 = view_71 = permute_43 = None
        view_72: "f16[32, 197, 1152]" = torch.ops.aten.reshape.default(addmm_28, [32, 197, 1152]);  addmm_28 = None
        view_73: "f16[32, 197, 3, 6, 64]" = torch.ops.aten.reshape.default(view_72, [32, 197, 3, 6, 64]);  view_72 = None
        permute_44: "f16[3, 32, 6, 197, 64]" = torch.ops.aten.permute.default(view_73, [2, 0, 3, 1, 4]);  view_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        unbind_7 = torch.ops.aten.unbind.int(permute_44);  permute_44 = None
        getitem_114: "f16[32, 6, 197, 64]" = unbind_7[0]
        getitem_115: "f16[32, 6, 197, 64]" = unbind_7[1]
        getitem_116: "f16[32, 6, 197, 64]" = unbind_7[2];  unbind_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:120 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention_7 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(getitem_114, getitem_115, getitem_116, None, False);  getitem_114 = getitem_115 = getitem_116 = None
        getitem_117: "f16[32, 6, 197, 64]" = _scaled_dot_product_cudnn_attention_7[0];  _scaled_dot_product_cudnn_attention_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_45: "f16[32, 197, 6, 64]" = torch.ops.aten.permute.default(getitem_117, [0, 2, 1, 3]);  getitem_117 = None
        view_74: "f16[32, 197, 384]" = torch.ops.aten.reshape.default(permute_45, [32, 197, 384]);  permute_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_75: "f16[6304, 384]" = torch.ops.aten.reshape.default(view_74, [6304, 384]);  view_74 = None
        permute_46: "f16[384, 384]" = torch.ops.aten.permute.default(arg93_1, [1, 0]);  arg93_1 = None
        addmm_29: "f16[6304, 384]" = torch.ops.aten.addmm.default(arg94_1, view_75, permute_46);  arg94_1 = view_75 = permute_46 = None
        view_76: "f16[32, 197, 384]" = torch.ops.aten.reshape.default(addmm_29, [32, 197, 384]);  addmm_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        add_52: "f16[32, 197, 384]" = torch.ops.aten.add.Tensor(add_49, view_76);  add_49 = view_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_134: "f32[32, 197, 384]" = torch.ops.prims.convert_element_type.default(add_52, torch.float32)
        var_mean_15 = torch.ops.aten.var_mean.correction(convert_element_type_134, [2], correction = 0, keepdim = True)
        getitem_126: "f32[32, 197, 1]" = var_mean_15[0]
        getitem_127: "f32[32, 197, 1]" = var_mean_15[1];  var_mean_15 = None
        sub_15: "f32[32, 197, 384]" = torch.ops.aten.sub.Tensor(convert_element_type_134, getitem_127);  convert_element_type_134 = getitem_127 = None
        add_53: "f32[32, 197, 1]" = torch.ops.aten.add.Tensor(getitem_126, 1e-06);  getitem_126 = None
        rsqrt_15: "f32[32, 197, 1]" = torch.ops.aten.rsqrt.default(add_53);  add_53 = None
        mul_51: "f32[32, 197, 384]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_15);  sub_15 = rsqrt_15 = None
        mul_52: "f32[32, 197, 384]" = torch.ops.aten.mul.Tensor(mul_51, arg95_1);  mul_51 = arg95_1 = None
        add_54: "f32[32, 197, 384]" = torch.ops.aten.add.Tensor(mul_52, arg96_1);  mul_52 = arg96_1 = None
        convert_element_type_135: "f16[32, 197, 384]" = torch.ops.prims.convert_element_type.default(add_54, torch.float16);  add_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_77: "f16[6304, 384]" = torch.ops.aten.reshape.default(convert_element_type_135, [6304, 384]);  convert_element_type_135 = None
        permute_47: "f16[384, 1536]" = torch.ops.aten.permute.default(arg97_1, [1, 0]);  arg97_1 = None
        addmm_30: "f16[6304, 1536]" = torch.ops.aten.addmm.default(arg98_1, view_77, permute_47);  arg98_1 = view_77 = permute_47 = None
        view_78: "f16[32, 197, 1536]" = torch.ops.aten.reshape.default(addmm_30, [32, 197, 1536]);  addmm_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_139: "f32[32, 197, 1536]" = torch.ops.prims.convert_element_type.default(view_78, torch.float32);  view_78 = None
        mul_53: "f32[32, 197, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_139, 0.5)
        mul_54: "f32[32, 197, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_139, 0.7071067811865476);  convert_element_type_139 = None
        erf_7: "f32[32, 197, 1536]" = torch.ops.aten.erf.default(mul_54);  mul_54 = None
        add_55: "f32[32, 197, 1536]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
        mul_55: "f32[32, 197, 1536]" = torch.ops.aten.mul.Tensor(mul_53, add_55);  mul_53 = add_55 = None
        convert_element_type_140: "f16[32, 197, 1536]" = torch.ops.prims.convert_element_type.default(mul_55, torch.float16);  mul_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_79: "f16[6304, 1536]" = torch.ops.aten.reshape.default(convert_element_type_140, [6304, 1536]);  convert_element_type_140 = None
        permute_48: "f16[1536, 384]" = torch.ops.aten.permute.default(arg99_1, [1, 0]);  arg99_1 = None
        addmm_31: "f16[6304, 384]" = torch.ops.aten.addmm.default(arg100_1, view_79, permute_48);  arg100_1 = view_79 = permute_48 = None
        view_80: "f16[32, 197, 384]" = torch.ops.aten.reshape.default(addmm_31, [32, 197, 384]);  addmm_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        add_56: "f16[32, 197, 384]" = torch.ops.aten.add.Tensor(add_52, view_80);  add_52 = view_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_144: "f32[32, 197, 384]" = torch.ops.prims.convert_element_type.default(add_56, torch.float32)
        var_mean_16 = torch.ops.aten.var_mean.correction(convert_element_type_144, [2], correction = 0, keepdim = True)
        getitem_128: "f32[32, 197, 1]" = var_mean_16[0]
        getitem_129: "f32[32, 197, 1]" = var_mean_16[1];  var_mean_16 = None
        sub_16: "f32[32, 197, 384]" = torch.ops.aten.sub.Tensor(convert_element_type_144, getitem_129);  convert_element_type_144 = getitem_129 = None
        add_57: "f32[32, 197, 1]" = torch.ops.aten.add.Tensor(getitem_128, 1e-06);  getitem_128 = None
        rsqrt_16: "f32[32, 197, 1]" = torch.ops.aten.rsqrt.default(add_57);  add_57 = None
        mul_56: "f32[32, 197, 384]" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_16);  sub_16 = rsqrt_16 = None
        mul_57: "f32[32, 197, 384]" = torch.ops.aten.mul.Tensor(mul_56, arg101_1);  mul_56 = arg101_1 = None
        add_58: "f32[32, 197, 384]" = torch.ops.aten.add.Tensor(mul_57, arg102_1);  mul_57 = arg102_1 = None
        convert_element_type_145: "f16[32, 197, 384]" = torch.ops.prims.convert_element_type.default(add_58, torch.float16);  add_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        view_81: "f16[6304, 384]" = torch.ops.aten.reshape.default(convert_element_type_145, [6304, 384]);  convert_element_type_145 = None
        permute_49: "f16[384, 1152]" = torch.ops.aten.permute.default(arg103_1, [1, 0]);  arg103_1 = None
        addmm_32: "f16[6304, 1152]" = torch.ops.aten.addmm.default(arg104_1, view_81, permute_49);  arg104_1 = view_81 = permute_49 = None
        view_82: "f16[32, 197, 1152]" = torch.ops.aten.reshape.default(addmm_32, [32, 197, 1152]);  addmm_32 = None
        view_83: "f16[32, 197, 3, 6, 64]" = torch.ops.aten.reshape.default(view_82, [32, 197, 3, 6, 64]);  view_82 = None
        permute_50: "f16[3, 32, 6, 197, 64]" = torch.ops.aten.permute.default(view_83, [2, 0, 3, 1, 4]);  view_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        unbind_8 = torch.ops.aten.unbind.int(permute_50);  permute_50 = None
        getitem_130: "f16[32, 6, 197, 64]" = unbind_8[0]
        getitem_131: "f16[32, 6, 197, 64]" = unbind_8[1]
        getitem_132: "f16[32, 6, 197, 64]" = unbind_8[2];  unbind_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:120 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention_8 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(getitem_130, getitem_131, getitem_132, None, False);  getitem_130 = getitem_131 = getitem_132 = None
        getitem_133: "f16[32, 6, 197, 64]" = _scaled_dot_product_cudnn_attention_8[0];  _scaled_dot_product_cudnn_attention_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_51: "f16[32, 197, 6, 64]" = torch.ops.aten.permute.default(getitem_133, [0, 2, 1, 3]);  getitem_133 = None
        view_84: "f16[32, 197, 384]" = torch.ops.aten.reshape.default(permute_51, [32, 197, 384]);  permute_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_85: "f16[6304, 384]" = torch.ops.aten.reshape.default(view_84, [6304, 384]);  view_84 = None
        permute_52: "f16[384, 384]" = torch.ops.aten.permute.default(arg105_1, [1, 0]);  arg105_1 = None
        addmm_33: "f16[6304, 384]" = torch.ops.aten.addmm.default(arg106_1, view_85, permute_52);  arg106_1 = view_85 = permute_52 = None
        view_86: "f16[32, 197, 384]" = torch.ops.aten.reshape.default(addmm_33, [32, 197, 384]);  addmm_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        add_59: "f16[32, 197, 384]" = torch.ops.aten.add.Tensor(add_56, view_86);  add_56 = view_86 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_152: "f32[32, 197, 384]" = torch.ops.prims.convert_element_type.default(add_59, torch.float32)
        var_mean_17 = torch.ops.aten.var_mean.correction(convert_element_type_152, [2], correction = 0, keepdim = True)
        getitem_142: "f32[32, 197, 1]" = var_mean_17[0]
        getitem_143: "f32[32, 197, 1]" = var_mean_17[1];  var_mean_17 = None
        sub_17: "f32[32, 197, 384]" = torch.ops.aten.sub.Tensor(convert_element_type_152, getitem_143);  convert_element_type_152 = getitem_143 = None
        add_60: "f32[32, 197, 1]" = torch.ops.aten.add.Tensor(getitem_142, 1e-06);  getitem_142 = None
        rsqrt_17: "f32[32, 197, 1]" = torch.ops.aten.rsqrt.default(add_60);  add_60 = None
        mul_58: "f32[32, 197, 384]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_17);  sub_17 = rsqrt_17 = None
        mul_59: "f32[32, 197, 384]" = torch.ops.aten.mul.Tensor(mul_58, arg107_1);  mul_58 = arg107_1 = None
        add_61: "f32[32, 197, 384]" = torch.ops.aten.add.Tensor(mul_59, arg108_1);  mul_59 = arg108_1 = None
        convert_element_type_153: "f16[32, 197, 384]" = torch.ops.prims.convert_element_type.default(add_61, torch.float16);  add_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_87: "f16[6304, 384]" = torch.ops.aten.reshape.default(convert_element_type_153, [6304, 384]);  convert_element_type_153 = None
        permute_53: "f16[384, 1536]" = torch.ops.aten.permute.default(arg109_1, [1, 0]);  arg109_1 = None
        addmm_34: "f16[6304, 1536]" = torch.ops.aten.addmm.default(arg110_1, view_87, permute_53);  arg110_1 = view_87 = permute_53 = None
        view_88: "f16[32, 197, 1536]" = torch.ops.aten.reshape.default(addmm_34, [32, 197, 1536]);  addmm_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_157: "f32[32, 197, 1536]" = torch.ops.prims.convert_element_type.default(view_88, torch.float32);  view_88 = None
        mul_60: "f32[32, 197, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_157, 0.5)
        mul_61: "f32[32, 197, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_157, 0.7071067811865476);  convert_element_type_157 = None
        erf_8: "f32[32, 197, 1536]" = torch.ops.aten.erf.default(mul_61);  mul_61 = None
        add_62: "f32[32, 197, 1536]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
        mul_62: "f32[32, 197, 1536]" = torch.ops.aten.mul.Tensor(mul_60, add_62);  mul_60 = add_62 = None
        convert_element_type_158: "f16[32, 197, 1536]" = torch.ops.prims.convert_element_type.default(mul_62, torch.float16);  mul_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_89: "f16[6304, 1536]" = torch.ops.aten.reshape.default(convert_element_type_158, [6304, 1536]);  convert_element_type_158 = None
        permute_54: "f16[1536, 384]" = torch.ops.aten.permute.default(arg111_1, [1, 0]);  arg111_1 = None
        addmm_35: "f16[6304, 384]" = torch.ops.aten.addmm.default(arg112_1, view_89, permute_54);  arg112_1 = view_89 = permute_54 = None
        view_90: "f16[32, 197, 384]" = torch.ops.aten.reshape.default(addmm_35, [32, 197, 384]);  addmm_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        add_63: "f16[32, 197, 384]" = torch.ops.aten.add.Tensor(add_59, view_90);  add_59 = view_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_162: "f32[32, 197, 384]" = torch.ops.prims.convert_element_type.default(add_63, torch.float32)
        var_mean_18 = torch.ops.aten.var_mean.correction(convert_element_type_162, [2], correction = 0, keepdim = True)
        getitem_144: "f32[32, 197, 1]" = var_mean_18[0]
        getitem_145: "f32[32, 197, 1]" = var_mean_18[1];  var_mean_18 = None
        sub_18: "f32[32, 197, 384]" = torch.ops.aten.sub.Tensor(convert_element_type_162, getitem_145);  convert_element_type_162 = getitem_145 = None
        add_64: "f32[32, 197, 1]" = torch.ops.aten.add.Tensor(getitem_144, 1e-06);  getitem_144 = None
        rsqrt_18: "f32[32, 197, 1]" = torch.ops.aten.rsqrt.default(add_64);  add_64 = None
        mul_63: "f32[32, 197, 384]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_18);  sub_18 = rsqrt_18 = None
        mul_64: "f32[32, 197, 384]" = torch.ops.aten.mul.Tensor(mul_63, arg113_1);  mul_63 = arg113_1 = None
        add_65: "f32[32, 197, 384]" = torch.ops.aten.add.Tensor(mul_64, arg114_1);  mul_64 = arg114_1 = None
        convert_element_type_163: "f16[32, 197, 384]" = torch.ops.prims.convert_element_type.default(add_65, torch.float16);  add_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        view_91: "f16[6304, 384]" = torch.ops.aten.reshape.default(convert_element_type_163, [6304, 384]);  convert_element_type_163 = None
        permute_55: "f16[384, 1152]" = torch.ops.aten.permute.default(arg115_1, [1, 0]);  arg115_1 = None
        addmm_36: "f16[6304, 1152]" = torch.ops.aten.addmm.default(arg116_1, view_91, permute_55);  arg116_1 = view_91 = permute_55 = None
        view_92: "f16[32, 197, 1152]" = torch.ops.aten.reshape.default(addmm_36, [32, 197, 1152]);  addmm_36 = None
        view_93: "f16[32, 197, 3, 6, 64]" = torch.ops.aten.reshape.default(view_92, [32, 197, 3, 6, 64]);  view_92 = None
        permute_56: "f16[3, 32, 6, 197, 64]" = torch.ops.aten.permute.default(view_93, [2, 0, 3, 1, 4]);  view_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        unbind_9 = torch.ops.aten.unbind.int(permute_56);  permute_56 = None
        getitem_146: "f16[32, 6, 197, 64]" = unbind_9[0]
        getitem_147: "f16[32, 6, 197, 64]" = unbind_9[1]
        getitem_148: "f16[32, 6, 197, 64]" = unbind_9[2];  unbind_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:120 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention_9 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(getitem_146, getitem_147, getitem_148, None, False);  getitem_146 = getitem_147 = getitem_148 = None
        getitem_149: "f16[32, 6, 197, 64]" = _scaled_dot_product_cudnn_attention_9[0];  _scaled_dot_product_cudnn_attention_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_57: "f16[32, 197, 6, 64]" = torch.ops.aten.permute.default(getitem_149, [0, 2, 1, 3]);  getitem_149 = None
        view_94: "f16[32, 197, 384]" = torch.ops.aten.reshape.default(permute_57, [32, 197, 384]);  permute_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_95: "f16[6304, 384]" = torch.ops.aten.reshape.default(view_94, [6304, 384]);  view_94 = None
        permute_58: "f16[384, 384]" = torch.ops.aten.permute.default(arg117_1, [1, 0]);  arg117_1 = None
        addmm_37: "f16[6304, 384]" = torch.ops.aten.addmm.default(arg118_1, view_95, permute_58);  arg118_1 = view_95 = permute_58 = None
        view_96: "f16[32, 197, 384]" = torch.ops.aten.reshape.default(addmm_37, [32, 197, 384]);  addmm_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        add_66: "f16[32, 197, 384]" = torch.ops.aten.add.Tensor(add_63, view_96);  add_63 = view_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_170: "f32[32, 197, 384]" = torch.ops.prims.convert_element_type.default(add_66, torch.float32)
        var_mean_19 = torch.ops.aten.var_mean.correction(convert_element_type_170, [2], correction = 0, keepdim = True)
        getitem_158: "f32[32, 197, 1]" = var_mean_19[0]
        getitem_159: "f32[32, 197, 1]" = var_mean_19[1];  var_mean_19 = None
        sub_19: "f32[32, 197, 384]" = torch.ops.aten.sub.Tensor(convert_element_type_170, getitem_159);  convert_element_type_170 = getitem_159 = None
        add_67: "f32[32, 197, 1]" = torch.ops.aten.add.Tensor(getitem_158, 1e-06);  getitem_158 = None
        rsqrt_19: "f32[32, 197, 1]" = torch.ops.aten.rsqrt.default(add_67);  add_67 = None
        mul_65: "f32[32, 197, 384]" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_19);  sub_19 = rsqrt_19 = None
        mul_66: "f32[32, 197, 384]" = torch.ops.aten.mul.Tensor(mul_65, arg119_1);  mul_65 = arg119_1 = None
        add_68: "f32[32, 197, 384]" = torch.ops.aten.add.Tensor(mul_66, arg120_1);  mul_66 = arg120_1 = None
        convert_element_type_171: "f16[32, 197, 384]" = torch.ops.prims.convert_element_type.default(add_68, torch.float16);  add_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_97: "f16[6304, 384]" = torch.ops.aten.reshape.default(convert_element_type_171, [6304, 384]);  convert_element_type_171 = None
        permute_59: "f16[384, 1536]" = torch.ops.aten.permute.default(arg121_1, [1, 0]);  arg121_1 = None
        addmm_38: "f16[6304, 1536]" = torch.ops.aten.addmm.default(arg122_1, view_97, permute_59);  arg122_1 = view_97 = permute_59 = None
        view_98: "f16[32, 197, 1536]" = torch.ops.aten.reshape.default(addmm_38, [32, 197, 1536]);  addmm_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_175: "f32[32, 197, 1536]" = torch.ops.prims.convert_element_type.default(view_98, torch.float32);  view_98 = None
        mul_67: "f32[32, 197, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_175, 0.5)
        mul_68: "f32[32, 197, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_175, 0.7071067811865476);  convert_element_type_175 = None
        erf_9: "f32[32, 197, 1536]" = torch.ops.aten.erf.default(mul_68);  mul_68 = None
        add_69: "f32[32, 197, 1536]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
        mul_69: "f32[32, 197, 1536]" = torch.ops.aten.mul.Tensor(mul_67, add_69);  mul_67 = add_69 = None
        convert_element_type_176: "f16[32, 197, 1536]" = torch.ops.prims.convert_element_type.default(mul_69, torch.float16);  mul_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_99: "f16[6304, 1536]" = torch.ops.aten.reshape.default(convert_element_type_176, [6304, 1536]);  convert_element_type_176 = None
        permute_60: "f16[1536, 384]" = torch.ops.aten.permute.default(arg123_1, [1, 0]);  arg123_1 = None
        addmm_39: "f16[6304, 384]" = torch.ops.aten.addmm.default(arg124_1, view_99, permute_60);  arg124_1 = view_99 = permute_60 = None
        view_100: "f16[32, 197, 384]" = torch.ops.aten.reshape.default(addmm_39, [32, 197, 384]);  addmm_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        add_70: "f16[32, 197, 384]" = torch.ops.aten.add.Tensor(add_66, view_100);  add_66 = view_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_180: "f32[32, 197, 384]" = torch.ops.prims.convert_element_type.default(add_70, torch.float32)
        var_mean_20 = torch.ops.aten.var_mean.correction(convert_element_type_180, [2], correction = 0, keepdim = True)
        getitem_160: "f32[32, 197, 1]" = var_mean_20[0]
        getitem_161: "f32[32, 197, 1]" = var_mean_20[1];  var_mean_20 = None
        sub_20: "f32[32, 197, 384]" = torch.ops.aten.sub.Tensor(convert_element_type_180, getitem_161);  convert_element_type_180 = getitem_161 = None
        add_71: "f32[32, 197, 1]" = torch.ops.aten.add.Tensor(getitem_160, 1e-06);  getitem_160 = None
        rsqrt_20: "f32[32, 197, 1]" = torch.ops.aten.rsqrt.default(add_71);  add_71 = None
        mul_70: "f32[32, 197, 384]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_20);  sub_20 = rsqrt_20 = None
        mul_71: "f32[32, 197, 384]" = torch.ops.aten.mul.Tensor(mul_70, arg125_1);  mul_70 = arg125_1 = None
        add_72: "f32[32, 197, 384]" = torch.ops.aten.add.Tensor(mul_71, arg126_1);  mul_71 = arg126_1 = None
        convert_element_type_181: "f16[32, 197, 384]" = torch.ops.prims.convert_element_type.default(add_72, torch.float16);  add_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        view_101: "f16[6304, 384]" = torch.ops.aten.reshape.default(convert_element_type_181, [6304, 384]);  convert_element_type_181 = None
        permute_61: "f16[384, 1152]" = torch.ops.aten.permute.default(arg127_1, [1, 0]);  arg127_1 = None
        addmm_40: "f16[6304, 1152]" = torch.ops.aten.addmm.default(arg128_1, view_101, permute_61);  arg128_1 = view_101 = permute_61 = None
        view_102: "f16[32, 197, 1152]" = torch.ops.aten.reshape.default(addmm_40, [32, 197, 1152]);  addmm_40 = None
        view_103: "f16[32, 197, 3, 6, 64]" = torch.ops.aten.reshape.default(view_102, [32, 197, 3, 6, 64]);  view_102 = None
        permute_62: "f16[3, 32, 6, 197, 64]" = torch.ops.aten.permute.default(view_103, [2, 0, 3, 1, 4]);  view_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        unbind_10 = torch.ops.aten.unbind.int(permute_62);  permute_62 = None
        getitem_162: "f16[32, 6, 197, 64]" = unbind_10[0]
        getitem_163: "f16[32, 6, 197, 64]" = unbind_10[1]
        getitem_164: "f16[32, 6, 197, 64]" = unbind_10[2];  unbind_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:120 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention_10 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(getitem_162, getitem_163, getitem_164, None, False);  getitem_162 = getitem_163 = getitem_164 = None
        getitem_165: "f16[32, 6, 197, 64]" = _scaled_dot_product_cudnn_attention_10[0];  _scaled_dot_product_cudnn_attention_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_63: "f16[32, 197, 6, 64]" = torch.ops.aten.permute.default(getitem_165, [0, 2, 1, 3]);  getitem_165 = None
        view_104: "f16[32, 197, 384]" = torch.ops.aten.reshape.default(permute_63, [32, 197, 384]);  permute_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_105: "f16[6304, 384]" = torch.ops.aten.reshape.default(view_104, [6304, 384]);  view_104 = None
        permute_64: "f16[384, 384]" = torch.ops.aten.permute.default(arg129_1, [1, 0]);  arg129_1 = None
        addmm_41: "f16[6304, 384]" = torch.ops.aten.addmm.default(arg130_1, view_105, permute_64);  arg130_1 = view_105 = permute_64 = None
        view_106: "f16[32, 197, 384]" = torch.ops.aten.reshape.default(addmm_41, [32, 197, 384]);  addmm_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        add_73: "f16[32, 197, 384]" = torch.ops.aten.add.Tensor(add_70, view_106);  add_70 = view_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_188: "f32[32, 197, 384]" = torch.ops.prims.convert_element_type.default(add_73, torch.float32)
        var_mean_21 = torch.ops.aten.var_mean.correction(convert_element_type_188, [2], correction = 0, keepdim = True)
        getitem_174: "f32[32, 197, 1]" = var_mean_21[0]
        getitem_175: "f32[32, 197, 1]" = var_mean_21[1];  var_mean_21 = None
        sub_21: "f32[32, 197, 384]" = torch.ops.aten.sub.Tensor(convert_element_type_188, getitem_175);  convert_element_type_188 = getitem_175 = None
        add_74: "f32[32, 197, 1]" = torch.ops.aten.add.Tensor(getitem_174, 1e-06);  getitem_174 = None
        rsqrt_21: "f32[32, 197, 1]" = torch.ops.aten.rsqrt.default(add_74);  add_74 = None
        mul_72: "f32[32, 197, 384]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_21);  sub_21 = rsqrt_21 = None
        mul_73: "f32[32, 197, 384]" = torch.ops.aten.mul.Tensor(mul_72, arg131_1);  mul_72 = arg131_1 = None
        add_75: "f32[32, 197, 384]" = torch.ops.aten.add.Tensor(mul_73, arg132_1);  mul_73 = arg132_1 = None
        convert_element_type_189: "f16[32, 197, 384]" = torch.ops.prims.convert_element_type.default(add_75, torch.float16);  add_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_107: "f16[6304, 384]" = torch.ops.aten.reshape.default(convert_element_type_189, [6304, 384]);  convert_element_type_189 = None
        permute_65: "f16[384, 1536]" = torch.ops.aten.permute.default(arg133_1, [1, 0]);  arg133_1 = None
        addmm_42: "f16[6304, 1536]" = torch.ops.aten.addmm.default(arg134_1, view_107, permute_65);  arg134_1 = view_107 = permute_65 = None
        view_108: "f16[32, 197, 1536]" = torch.ops.aten.reshape.default(addmm_42, [32, 197, 1536]);  addmm_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_193: "f32[32, 197, 1536]" = torch.ops.prims.convert_element_type.default(view_108, torch.float32);  view_108 = None
        mul_74: "f32[32, 197, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_193, 0.5)
        mul_75: "f32[32, 197, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_193, 0.7071067811865476);  convert_element_type_193 = None
        erf_10: "f32[32, 197, 1536]" = torch.ops.aten.erf.default(mul_75);  mul_75 = None
        add_76: "f32[32, 197, 1536]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
        mul_76: "f32[32, 197, 1536]" = torch.ops.aten.mul.Tensor(mul_74, add_76);  mul_74 = add_76 = None
        convert_element_type_194: "f16[32, 197, 1536]" = torch.ops.prims.convert_element_type.default(mul_76, torch.float16);  mul_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_109: "f16[6304, 1536]" = torch.ops.aten.reshape.default(convert_element_type_194, [6304, 1536]);  convert_element_type_194 = None
        permute_66: "f16[1536, 384]" = torch.ops.aten.permute.default(arg135_1, [1, 0]);  arg135_1 = None
        addmm_43: "f16[6304, 384]" = torch.ops.aten.addmm.default(arg136_1, view_109, permute_66);  arg136_1 = view_109 = permute_66 = None
        view_110: "f16[32, 197, 384]" = torch.ops.aten.reshape.default(addmm_43, [32, 197, 384]);  addmm_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        add_77: "f16[32, 197, 384]" = torch.ops.aten.add.Tensor(add_73, view_110);  add_73 = view_110 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_198: "f32[32, 197, 384]" = torch.ops.prims.convert_element_type.default(add_77, torch.float32)
        var_mean_22 = torch.ops.aten.var_mean.correction(convert_element_type_198, [2], correction = 0, keepdim = True)
        getitem_176: "f32[32, 197, 1]" = var_mean_22[0]
        getitem_177: "f32[32, 197, 1]" = var_mean_22[1];  var_mean_22 = None
        sub_22: "f32[32, 197, 384]" = torch.ops.aten.sub.Tensor(convert_element_type_198, getitem_177);  convert_element_type_198 = getitem_177 = None
        add_78: "f32[32, 197, 1]" = torch.ops.aten.add.Tensor(getitem_176, 1e-06);  getitem_176 = None
        rsqrt_22: "f32[32, 197, 1]" = torch.ops.aten.rsqrt.default(add_78);  add_78 = None
        mul_77: "f32[32, 197, 384]" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_22);  sub_22 = rsqrt_22 = None
        mul_78: "f32[32, 197, 384]" = torch.ops.aten.mul.Tensor(mul_77, arg137_1);  mul_77 = arg137_1 = None
        add_79: "f32[32, 197, 384]" = torch.ops.aten.add.Tensor(mul_78, arg138_1);  mul_78 = arg138_1 = None
        convert_element_type_199: "f16[32, 197, 384]" = torch.ops.prims.convert_element_type.default(add_79, torch.float16);  add_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        view_111: "f16[6304, 384]" = torch.ops.aten.reshape.default(convert_element_type_199, [6304, 384]);  convert_element_type_199 = None
        permute_67: "f16[384, 1152]" = torch.ops.aten.permute.default(arg139_1, [1, 0]);  arg139_1 = None
        addmm_44: "f16[6304, 1152]" = torch.ops.aten.addmm.default(arg140_1, view_111, permute_67);  arg140_1 = view_111 = permute_67 = None
        view_112: "f16[32, 197, 1152]" = torch.ops.aten.reshape.default(addmm_44, [32, 197, 1152]);  addmm_44 = None
        view_113: "f16[32, 197, 3, 6, 64]" = torch.ops.aten.reshape.default(view_112, [32, 197, 3, 6, 64]);  view_112 = None
        permute_68: "f16[3, 32, 6, 197, 64]" = torch.ops.aten.permute.default(view_113, [2, 0, 3, 1, 4]);  view_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        unbind_11 = torch.ops.aten.unbind.int(permute_68);  permute_68 = None
        getitem_178: "f16[32, 6, 197, 64]" = unbind_11[0]
        getitem_179: "f16[32, 6, 197, 64]" = unbind_11[1]
        getitem_180: "f16[32, 6, 197, 64]" = unbind_11[2];  unbind_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:120 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention_11 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(getitem_178, getitem_179, getitem_180, None, False);  getitem_178 = getitem_179 = getitem_180 = None
        getitem_181: "f16[32, 6, 197, 64]" = _scaled_dot_product_cudnn_attention_11[0];  _scaled_dot_product_cudnn_attention_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_69: "f16[32, 197, 6, 64]" = torch.ops.aten.permute.default(getitem_181, [0, 2, 1, 3]);  getitem_181 = None
        view_114: "f16[32, 197, 384]" = torch.ops.aten.reshape.default(permute_69, [32, 197, 384]);  permute_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_115: "f16[6304, 384]" = torch.ops.aten.reshape.default(view_114, [6304, 384]);  view_114 = None
        permute_70: "f16[384, 384]" = torch.ops.aten.permute.default(arg141_1, [1, 0]);  arg141_1 = None
        addmm_45: "f16[6304, 384]" = torch.ops.aten.addmm.default(arg142_1, view_115, permute_70);  arg142_1 = view_115 = permute_70 = None
        view_116: "f16[32, 197, 384]" = torch.ops.aten.reshape.default(addmm_45, [32, 197, 384]);  addmm_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        add_80: "f16[32, 197, 384]" = torch.ops.aten.add.Tensor(add_77, view_116);  add_77 = view_116 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_206: "f32[32, 197, 384]" = torch.ops.prims.convert_element_type.default(add_80, torch.float32)
        var_mean_23 = torch.ops.aten.var_mean.correction(convert_element_type_206, [2], correction = 0, keepdim = True)
        getitem_190: "f32[32, 197, 1]" = var_mean_23[0]
        getitem_191: "f32[32, 197, 1]" = var_mean_23[1];  var_mean_23 = None
        sub_23: "f32[32, 197, 384]" = torch.ops.aten.sub.Tensor(convert_element_type_206, getitem_191);  convert_element_type_206 = getitem_191 = None
        add_81: "f32[32, 197, 1]" = torch.ops.aten.add.Tensor(getitem_190, 1e-06);  getitem_190 = None
        rsqrt_23: "f32[32, 197, 1]" = torch.ops.aten.rsqrt.default(add_81);  add_81 = None
        mul_79: "f32[32, 197, 384]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_23);  sub_23 = rsqrt_23 = None
        mul_80: "f32[32, 197, 384]" = torch.ops.aten.mul.Tensor(mul_79, arg143_1);  mul_79 = arg143_1 = None
        add_82: "f32[32, 197, 384]" = torch.ops.aten.add.Tensor(mul_80, arg144_1);  mul_80 = arg144_1 = None
        convert_element_type_207: "f16[32, 197, 384]" = torch.ops.prims.convert_element_type.default(add_82, torch.float16);  add_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_117: "f16[6304, 384]" = torch.ops.aten.reshape.default(convert_element_type_207, [6304, 384]);  convert_element_type_207 = None
        permute_71: "f16[384, 1536]" = torch.ops.aten.permute.default(arg145_1, [1, 0]);  arg145_1 = None
        addmm_46: "f16[6304, 1536]" = torch.ops.aten.addmm.default(arg146_1, view_117, permute_71);  arg146_1 = view_117 = permute_71 = None
        view_118: "f16[32, 197, 1536]" = torch.ops.aten.reshape.default(addmm_46, [32, 197, 1536]);  addmm_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_211: "f32[32, 197, 1536]" = torch.ops.prims.convert_element_type.default(view_118, torch.float32);  view_118 = None
        mul_81: "f32[32, 197, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_211, 0.5)
        mul_82: "f32[32, 197, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_211, 0.7071067811865476);  convert_element_type_211 = None
        erf_11: "f32[32, 197, 1536]" = torch.ops.aten.erf.default(mul_82);  mul_82 = None
        add_83: "f32[32, 197, 1536]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
        mul_83: "f32[32, 197, 1536]" = torch.ops.aten.mul.Tensor(mul_81, add_83);  mul_81 = add_83 = None
        convert_element_type_212: "f16[32, 197, 1536]" = torch.ops.prims.convert_element_type.default(mul_83, torch.float16);  mul_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_119: "f16[6304, 1536]" = torch.ops.aten.reshape.default(convert_element_type_212, [6304, 1536]);  convert_element_type_212 = None
        permute_72: "f16[1536, 384]" = torch.ops.aten.permute.default(arg147_1, [1, 0]);  arg147_1 = None
        addmm_47: "f16[6304, 384]" = torch.ops.aten.addmm.default(arg148_1, view_119, permute_72);  arg148_1 = view_119 = permute_72 = None
        view_120: "f16[32, 197, 384]" = torch.ops.aten.reshape.default(addmm_47, [32, 197, 384]);  addmm_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        add_84: "f16[32, 197, 384]" = torch.ops.aten.add.Tensor(add_80, view_120);  add_80 = view_120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_216: "f32[32, 197, 384]" = torch.ops.prims.convert_element_type.default(add_84, torch.float32);  add_84 = None
        var_mean_24 = torch.ops.aten.var_mean.correction(convert_element_type_216, [2], correction = 0, keepdim = True)
        getitem_192: "f32[32, 197, 1]" = var_mean_24[0]
        getitem_193: "f32[32, 197, 1]" = var_mean_24[1];  var_mean_24 = None
        sub_24: "f32[32, 197, 384]" = torch.ops.aten.sub.Tensor(convert_element_type_216, getitem_193);  convert_element_type_216 = getitem_193 = None
        add_85: "f32[32, 197, 1]" = torch.ops.aten.add.Tensor(getitem_192, 1e-06);  getitem_192 = None
        rsqrt_24: "f32[32, 197, 1]" = torch.ops.aten.rsqrt.default(add_85);  add_85 = None
        mul_84: "f32[32, 197, 384]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_24);  sub_24 = rsqrt_24 = None
        mul_85: "f32[32, 197, 384]" = torch.ops.aten.mul.Tensor(mul_84, arg149_1);  mul_84 = arg149_1 = None
        add_86: "f32[32, 197, 384]" = torch.ops.aten.add.Tensor(mul_85, arg150_1);  mul_85 = arg150_1 = None
        convert_element_type_217: "f16[32, 197, 384]" = torch.ops.prims.convert_element_type.default(add_86, torch.float16);  add_86 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:696 in global_pool_nlc, code: x = x[:, 0]  # class token
        select: "f16[32, 384]" = torch.ops.aten.select.int(convert_element_type_217, 1, 0);  convert_element_type_217 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:1291 in forward_head, code: x = self.head_drop(x)
        clone_37: "f16[32, 384]" = torch.ops.aten.clone.default(select);  select = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:1292 in forward_head, code: return x if pre_logits else self.head(x)
        permute_73: "f16[384, 1000]" = torch.ops.aten.permute.default(arg151_1, [1, 0]);  arg151_1 = None
        addmm_48: "f16[32, 1000]" = torch.ops.aten.addmm.default(arg152_1, clone_37, permute_73);  arg152_1 = clone_37 = permute_73 = None
        return (addmm_48,)
