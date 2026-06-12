class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[128, 3, 518, 518][804972, 1, 1554, 3]cuda:0", primals_2: "f32[768, 3, 14, 14][588, 1, 42, 3]cuda:0", primals_3: "f32[768][1]cuda:0", primals_4: "f32[1, 1, 768][768, 768, 1]cuda:0", primals_5: "f32[1, 1370, 768][1052160, 768, 1]cuda:0", primals_6: "f32[768][1]cuda:0", primals_7: "f32[768][1]cuda:0", primals_8: "f32[2304, 768][768, 1]cuda:0", primals_9: "f32[2304][1]cuda:0", primals_10: "f32[768, 768][768, 1]cuda:0", primals_11: "f32[768][1]cuda:0", primals_12: "f32[768][1]cuda:0", primals_13: "f32[768][1]cuda:0", primals_14: "f32[768][1]cuda:0", primals_15: "f32[3072, 768][768, 1]cuda:0", primals_16: "f32[3072][1]cuda:0", primals_17: "f32[768, 3072][3072, 1]cuda:0", primals_18: "f32[768][1]cuda:0", primals_19: "f32[768][1]cuda:0", primals_20: "f32[768][1]cuda:0", primals_21: "f32[768][1]cuda:0", primals_22: "f32[2304, 768][768, 1]cuda:0", primals_23: "f32[2304][1]cuda:0", primals_24: "f32[768, 768][768, 1]cuda:0", primals_25: "f32[768][1]cuda:0", primals_26: "f32[768][1]cuda:0", primals_27: "f32[768][1]cuda:0", primals_28: "f32[768][1]cuda:0", primals_29: "f32[3072, 768][768, 1]cuda:0", primals_30: "f32[3072][1]cuda:0", primals_31: "f32[768, 3072][3072, 1]cuda:0", primals_32: "f32[768][1]cuda:0", primals_33: "f32[768][1]cuda:0", primals_34: "f32[768][1]cuda:0", primals_35: "f32[768][1]cuda:0", primals_36: "f32[2304, 768][768, 1]cuda:0", primals_37: "f32[2304][1]cuda:0", primals_38: "f32[768, 768][768, 1]cuda:0", primals_39: "f32[768][1]cuda:0", primals_40: "f32[768][1]cuda:0", primals_41: "f32[768][1]cuda:0", primals_42: "f32[768][1]cuda:0", primals_43: "f32[3072, 768][768, 1]cuda:0", primals_44: "f32[3072][1]cuda:0", primals_45: "f32[768, 3072][3072, 1]cuda:0", primals_46: "f32[768][1]cuda:0", primals_47: "f32[768][1]cuda:0", primals_48: "f32[768][1]cuda:0", primals_49: "f32[768][1]cuda:0", primals_50: "f32[2304, 768][768, 1]cuda:0", primals_51: "f32[2304][1]cuda:0", primals_52: "f32[768, 768][768, 1]cuda:0", primals_53: "f32[768][1]cuda:0", primals_54: "f32[768][1]cuda:0", primals_55: "f32[768][1]cuda:0", primals_56: "f32[768][1]cuda:0", primals_57: "f32[3072, 768][768, 1]cuda:0", primals_58: "f32[3072][1]cuda:0", primals_59: "f32[768, 3072][3072, 1]cuda:0", primals_60: "f32[768][1]cuda:0", primals_61: "f32[768][1]cuda:0", primals_62: "f32[768][1]cuda:0", primals_63: "f32[768][1]cuda:0", primals_64: "f32[2304, 768][768, 1]cuda:0", primals_65: "f32[2304][1]cuda:0", primals_66: "f32[768, 768][768, 1]cuda:0", primals_67: "f32[768][1]cuda:0", primals_68: "f32[768][1]cuda:0", primals_69: "f32[768][1]cuda:0", primals_70: "f32[768][1]cuda:0", primals_71: "f32[3072, 768][768, 1]cuda:0", primals_72: "f32[3072][1]cuda:0", primals_73: "f32[768, 3072][3072, 1]cuda:0", primals_74: "f32[768][1]cuda:0", primals_75: "f32[768][1]cuda:0", primals_76: "f32[768][1]cuda:0", primals_77: "f32[768][1]cuda:0", primals_78: "f32[2304, 768][768, 1]cuda:0", primals_79: "f32[2304][1]cuda:0", primals_80: "f32[768, 768][768, 1]cuda:0", primals_81: "f32[768][1]cuda:0", primals_82: "f32[768][1]cuda:0", primals_83: "f32[768][1]cuda:0", primals_84: "f32[768][1]cuda:0", primals_85: "f32[3072, 768][768, 1]cuda:0", primals_86: "f32[3072][1]cuda:0", primals_87: "f32[768, 3072][3072, 1]cuda:0", primals_88: "f32[768][1]cuda:0", primals_89: "f32[768][1]cuda:0", primals_90: "f32[768][1]cuda:0", primals_91: "f32[768][1]cuda:0", primals_92: "f32[2304, 768][768, 1]cuda:0", primals_93: "f32[2304][1]cuda:0", primals_94: "f32[768, 768][768, 1]cuda:0", primals_95: "f32[768][1]cuda:0", primals_96: "f32[768][1]cuda:0", primals_97: "f32[768][1]cuda:0", primals_98: "f32[768][1]cuda:0", primals_99: "f32[3072, 768][768, 1]cuda:0", primals_100: "f32[3072][1]cuda:0", primals_101: "f32[768, 3072][3072, 1]cuda:0", primals_102: "f32[768][1]cuda:0", primals_103: "f32[768][1]cuda:0", primals_104: "f32[768][1]cuda:0", primals_105: "f32[768][1]cuda:0", primals_106: "f32[2304, 768][768, 1]cuda:0", primals_107: "f32[2304][1]cuda:0", primals_108: "f32[768, 768][768, 1]cuda:0", primals_109: "f32[768][1]cuda:0", primals_110: "f32[768][1]cuda:0", primals_111: "f32[768][1]cuda:0", primals_112: "f32[768][1]cuda:0", primals_113: "f32[3072, 768][768, 1]cuda:0", primals_114: "f32[3072][1]cuda:0", primals_115: "f32[768, 3072][3072, 1]cuda:0", primals_116: "f32[768][1]cuda:0", primals_117: "f32[768][1]cuda:0", primals_118: "f32[768][1]cuda:0", primals_119: "f32[768][1]cuda:0", primals_120: "f32[2304, 768][768, 1]cuda:0", primals_121: "f32[2304][1]cuda:0", primals_122: "f32[768, 768][768, 1]cuda:0", primals_123: "f32[768][1]cuda:0", primals_124: "f32[768][1]cuda:0", primals_125: "f32[768][1]cuda:0", primals_126: "f32[768][1]cuda:0", primals_127: "f32[3072, 768][768, 1]cuda:0", primals_128: "f32[3072][1]cuda:0", primals_129: "f32[768, 3072][3072, 1]cuda:0", primals_130: "f32[768][1]cuda:0", primals_131: "f32[768][1]cuda:0", primals_132: "f32[768][1]cuda:0", primals_133: "f32[768][1]cuda:0", primals_134: "f32[2304, 768][768, 1]cuda:0", primals_135: "f32[2304][1]cuda:0", primals_136: "f32[768, 768][768, 1]cuda:0", primals_137: "f32[768][1]cuda:0", primals_138: "f32[768][1]cuda:0", primals_139: "f32[768][1]cuda:0", primals_140: "f32[768][1]cuda:0", primals_141: "f32[3072, 768][768, 1]cuda:0", primals_142: "f32[3072][1]cuda:0", primals_143: "f32[768, 3072][3072, 1]cuda:0", primals_144: "f32[768][1]cuda:0", primals_145: "f32[768][1]cuda:0", primals_146: "f32[768][1]cuda:0", primals_147: "f32[768][1]cuda:0", primals_148: "f32[2304, 768][768, 1]cuda:0", primals_149: "f32[2304][1]cuda:0", primals_150: "f32[768, 768][768, 1]cuda:0", primals_151: "f32[768][1]cuda:0", primals_152: "f32[768][1]cuda:0", primals_153: "f32[768][1]cuda:0", primals_154: "f32[768][1]cuda:0", primals_155: "f32[3072, 768][768, 1]cuda:0", primals_156: "f32[3072][1]cuda:0", primals_157: "f32[768, 3072][3072, 1]cuda:0", primals_158: "f32[768][1]cuda:0", primals_159: "f32[768][1]cuda:0", primals_160: "f32[768][1]cuda:0", primals_161: "f32[768][1]cuda:0", primals_162: "f32[2304, 768][768, 1]cuda:0", primals_163: "f32[2304][1]cuda:0", primals_164: "f32[768, 768][768, 1]cuda:0", primals_165: "f32[768][1]cuda:0", primals_166: "f32[768][1]cuda:0", primals_167: "f32[768][1]cuda:0", primals_168: "f32[768][1]cuda:0", primals_169: "f32[3072, 768][768, 1]cuda:0", primals_170: "f32[3072][1]cuda:0", primals_171: "f32[768, 3072][3072, 1]cuda:0", primals_172: "f32[768][1]cuda:0", primals_173: "f32[768][1]cuda:0", primals_174: "f32[768][1]cuda:0", primals_175: "f32[768][1]cuda:0"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/patch_embed.py:136 in forward, code: x = self.proj(x)
        convert_element_type: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_3, torch.bfloat16);  primals_3 = None
        convert_element_type_1: "bf16[768, 3, 14, 14][588, 1, 42, 3]cuda:0" = torch.ops.prims.convert_element_type.default(primals_2, torch.bfloat16);  primals_2 = None
        convert_element_type_2: "bf16[128, 3, 518, 518][804972, 1, 1554, 3]cuda:0" = torch.ops.prims.convert_element_type.default(primals_1, torch.bfloat16);  primals_1 = None
        convolution: "bf16[128, 768, 37, 37][1051392, 1, 28416, 768]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_2, convert_element_type_1, convert_element_type, [14, 14], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/patch_embed.py:138 in forward, code: x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
        view: "bf16[128, 768, 1369][1051392, 1, 768]cuda:0" = torch.ops.aten.reshape.default(convolution, [128, 768, 1369]);  convolution = None
        permute: "bf16[128, 1369, 768][1051392, 768, 1]cuda:0" = torch.ops.aten.permute.default(view, [0, 2, 1]);  view = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:1042 in _pos_embed, code: to_cat.append(self.cls_token.expand(x.shape[0], -1, -1))
        expand: "f32[128, 1, 768][0, 768, 1]cuda:0" = torch.ops.aten.expand.default(primals_4, [128, -1, -1]);  primals_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:1072 in _pos_embed, code: x = torch.cat(to_cat + [x], dim=1)
        cat: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.cat.default([expand, permute], 1);  expand = permute = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:1073 in _pos_embed, code: x = x + pos_embed
        add: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(cat, primals_5)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean = torch.ops.aten.var_mean.correction(add, [2], correction = 0, keepdim = True)
        getitem: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = var_mean[0]
        getitem_1: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = var_mean[1];  var_mean = None
        add_1: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem, 1e-06);  getitem = None
        rsqrt: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
        sub: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add, getitem_1)
        mul: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        mul_1: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul, primals_6);  mul = None
        add_2: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_1, primals_7);  mul_1 = primals_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        convert_element_type_3: "bf16[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_9, torch.bfloat16);  primals_9 = None
        convert_element_type_4: "bf16[2304, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_8, torch.bfloat16);  primals_8 = None
        convert_element_type_5: "bf16[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_2, torch.bfloat16);  add_2 = None
        view_1: "bf16[175360, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_5, [175360, 768]);  convert_element_type_5 = None
        permute_1: "bf16[768, 2304][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_4, [1, 0]);  convert_element_type_4 = None
        addmm: "bf16[175360, 2304][2304, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_3, view_1, permute_1);  convert_element_type_3 = None
        view_2: "bf16[128, 1370, 2304][3156480, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(addmm, [128, 1370, 2304]);  addmm = None
        view_3: "bf16[128, 1370, 3, 12, 64][3156480, 2304, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_2, [128, 1370, 3, 12, 64]);  view_2 = None
        permute_2: "bf16[3, 128, 12, 1370, 64][768, 3156480, 64, 2304, 1]cuda:0" = torch.ops.aten.permute.default(view_3, [2, 0, 3, 1, 4]);  view_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        unbind = torch.ops.aten.unbind.int(permute_2);  permute_2 = None
        getitem_2: "bf16[128, 12, 1370, 64][3156480, 64, 2304, 1]cuda:0" = unbind[0]
        getitem_3: "bf16[128, 12, 1370, 64][3156480, 64, 2304, 1]cuda:0" = unbind[1]
        getitem_4: "bf16[128, 12, 1370, 64][3156480, 64, 2304, 1]cuda:0" = unbind[2];  unbind = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:120 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention = torch.ops.aten._scaled_dot_product_cudnn_attention.default(getitem_2, getitem_3, getitem_4, None, True)
        getitem_5: "bf16[128, 12, 1370, 64][1052160, 64, 768, 1]cuda:0" = _scaled_dot_product_cudnn_attention[0]
        getitem_6: "f32[128, 12, 1370, 1][16440, 1370, 1, 1]cuda:0" = _scaled_dot_product_cudnn_attention[1]
        getitem_11: "i64[][]cuda:0" = _scaled_dot_product_cudnn_attention[6]
        getitem_12: "i64[][]cuda:0" = _scaled_dot_product_cudnn_attention[7];  _scaled_dot_product_cudnn_attention = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_3: "bf16[128, 1370, 12, 64][1052160, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_5, [0, 2, 1, 3])
        view_4: "bf16[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_3, [128, 1370, 768]);  permute_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        convert_element_type_9: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_11, torch.bfloat16);  primals_11 = None
        convert_element_type_10: "bf16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_10, torch.bfloat16);  primals_10 = None
        view_5: "bf16[175360, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_4, [175360, 768]);  view_4 = None
        permute_4: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_10, [1, 0]);  convert_element_type_10 = None
        addmm_1: "bf16[175360, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_9, view_5, permute_4);  convert_element_type_9 = view_5 = None
        view_6: "bf16[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_1, [128, 1370, 768])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/layer_scale.py:27 in forward, code: return x.mul_(self.gamma) if self.inplace else x * self.gamma
        mul_2: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_6, primals_12);  view_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        add_3: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add, mul_2);  add = mul_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_1 = torch.ops.aten.var_mean.correction(add_3, [2], correction = 0, keepdim = True)
        getitem_14: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = var_mean_1[0]
        getitem_15: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = var_mean_1[1];  var_mean_1 = None
        add_4: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_14, 1e-06);  getitem_14 = None
        rsqrt_1: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_4);  add_4 = None
        sub_1: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_3, getitem_15);  getitem_15 = None
        mul_3: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = None
        mul_4: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_3, primals_13)
        add_5: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_4, primals_14);  mul_4 = primals_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        convert_element_type_14: "bf16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_16, torch.bfloat16);  primals_16 = None
        convert_element_type_15: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_15, torch.bfloat16);  primals_15 = None
        convert_element_type_16: "bf16[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_5, torch.bfloat16);  add_5 = None
        view_7: "bf16[175360, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_16, [175360, 768]);  convert_element_type_16 = None
        permute_5: "bf16[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_15, [1, 0]);  convert_element_type_15 = None
        addmm_2: "bf16[175360, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_14, view_7, permute_5);  convert_element_type_14 = None
        view_8: "bf16[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_2, [128, 1370, 3072])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_20: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_8, torch.float32);  view_8 = None
        mul_5: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_20, 0.5)
        mul_6: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_20, 0.7071067811865476);  convert_element_type_20 = None
        erf: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_6);  mul_6 = None
        add_6: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_7: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_5, add_6);  mul_5 = add_6 = None
        convert_element_type_21: "bf16[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_7, torch.bfloat16);  mul_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        convert_element_type_22: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_18, torch.bfloat16);  primals_18 = None
        convert_element_type_23: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_17, torch.bfloat16);  primals_17 = None
        view_9: "bf16[175360, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_21, [175360, 3072]);  convert_element_type_21 = None
        permute_6: "bf16[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(convert_element_type_23, [1, 0]);  convert_element_type_23 = None
        addmm_3: "bf16[175360, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_22, view_9, permute_6);  convert_element_type_22 = None
        view_10: "bf16[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_3, [128, 1370, 768])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/layer_scale.py:27 in forward, code: return x.mul_(self.gamma) if self.inplace else x * self.gamma
        mul_8: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_10, primals_19);  view_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        add_7: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_3, mul_8);  add_3 = mul_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_2 = torch.ops.aten.var_mean.correction(add_7, [2], correction = 0, keepdim = True)
        getitem_16: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = var_mean_2[0]
        getitem_17: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = var_mean_2[1];  var_mean_2 = None
        add_8: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_16, 1e-06);  getitem_16 = None
        rsqrt_2: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_8);  add_8 = None
        sub_2: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_7, getitem_17);  getitem_17 = None
        mul_9: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_2);  sub_2 = None
        mul_10: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_9, primals_20)
        add_9: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_10, primals_21);  mul_10 = primals_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        convert_element_type_27: "bf16[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_23, torch.bfloat16);  primals_23 = None
        convert_element_type_28: "bf16[2304, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_22, torch.bfloat16);  primals_22 = None
        convert_element_type_29: "bf16[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_9, torch.bfloat16);  add_9 = None
        view_11: "bf16[175360, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_29, [175360, 768]);  convert_element_type_29 = None
        permute_7: "bf16[768, 2304][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_28, [1, 0]);  convert_element_type_28 = None
        addmm_4: "bf16[175360, 2304][2304, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_27, view_11, permute_7);  convert_element_type_27 = None
        view_12: "bf16[128, 1370, 2304][3156480, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_4, [128, 1370, 2304]);  addmm_4 = None
        view_13: "bf16[128, 1370, 3, 12, 64][3156480, 2304, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_12, [128, 1370, 3, 12, 64]);  view_12 = None
        permute_8: "bf16[3, 128, 12, 1370, 64][768, 3156480, 64, 2304, 1]cuda:0" = torch.ops.aten.permute.default(view_13, [2, 0, 3, 1, 4]);  view_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        unbind_1 = torch.ops.aten.unbind.int(permute_8);  permute_8 = None
        getitem_18: "bf16[128, 12, 1370, 64][3156480, 64, 2304, 1]cuda:0" = unbind_1[0]
        getitem_19: "bf16[128, 12, 1370, 64][3156480, 64, 2304, 1]cuda:0" = unbind_1[1]
        getitem_20: "bf16[128, 12, 1370, 64][3156480, 64, 2304, 1]cuda:0" = unbind_1[2];  unbind_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:120 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention_1 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(getitem_18, getitem_19, getitem_20, None, True)
        getitem_21: "bf16[128, 12, 1370, 64][1052160, 64, 768, 1]cuda:0" = _scaled_dot_product_cudnn_attention_1[0]
        getitem_22: "f32[128, 12, 1370, 1][16440, 1370, 1, 1]cuda:0" = _scaled_dot_product_cudnn_attention_1[1]
        getitem_27: "i64[][]cuda:0" = _scaled_dot_product_cudnn_attention_1[6]
        getitem_28: "i64[][]cuda:0" = _scaled_dot_product_cudnn_attention_1[7];  _scaled_dot_product_cudnn_attention_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_9: "bf16[128, 1370, 12, 64][1052160, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_21, [0, 2, 1, 3])
        view_14: "bf16[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_9, [128, 1370, 768]);  permute_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        convert_element_type_33: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_25, torch.bfloat16);  primals_25 = None
        convert_element_type_34: "bf16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_24, torch.bfloat16);  primals_24 = None
        view_15: "bf16[175360, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_14, [175360, 768]);  view_14 = None
        permute_10: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_34, [1, 0]);  convert_element_type_34 = None
        addmm_5: "bf16[175360, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_33, view_15, permute_10);  convert_element_type_33 = view_15 = None
        view_16: "bf16[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_5, [128, 1370, 768])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/layer_scale.py:27 in forward, code: return x.mul_(self.gamma) if self.inplace else x * self.gamma
        mul_11: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_16, primals_26);  view_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        add_10: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_7, mul_11);  add_7 = mul_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_3 = torch.ops.aten.var_mean.correction(add_10, [2], correction = 0, keepdim = True)
        getitem_30: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = var_mean_3[0]
        getitem_31: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = var_mean_3[1];  var_mean_3 = None
        add_11: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_30, 1e-06);  getitem_30 = None
        rsqrt_3: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_11);  add_11 = None
        sub_3: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_10, getitem_31);  getitem_31 = None
        mul_12: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_3);  sub_3 = None
        mul_13: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_12, primals_27)
        add_12: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_13, primals_28);  mul_13 = primals_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        convert_element_type_38: "bf16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_30, torch.bfloat16);  primals_30 = None
        convert_element_type_39: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_29, torch.bfloat16);  primals_29 = None
        convert_element_type_40: "bf16[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_12, torch.bfloat16);  add_12 = None
        view_17: "bf16[175360, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_40, [175360, 768]);  convert_element_type_40 = None
        permute_11: "bf16[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_39, [1, 0]);  convert_element_type_39 = None
        addmm_6: "bf16[175360, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_38, view_17, permute_11);  convert_element_type_38 = None
        view_18: "bf16[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_6, [128, 1370, 3072])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_44: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_18, torch.float32);  view_18 = None
        mul_14: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_44, 0.5)
        mul_15: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_44, 0.7071067811865476);  convert_element_type_44 = None
        erf_1: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_15);  mul_15 = None
        add_13: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
        mul_16: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_14, add_13);  mul_14 = add_13 = None
        convert_element_type_45: "bf16[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_16, torch.bfloat16);  mul_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        convert_element_type_46: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_32, torch.bfloat16);  primals_32 = None
        convert_element_type_47: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_31, torch.bfloat16);  primals_31 = None
        view_19: "bf16[175360, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_45, [175360, 3072]);  convert_element_type_45 = None
        permute_12: "bf16[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(convert_element_type_47, [1, 0]);  convert_element_type_47 = None
        addmm_7: "bf16[175360, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_46, view_19, permute_12);  convert_element_type_46 = None
        view_20: "bf16[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_7, [128, 1370, 768])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/layer_scale.py:27 in forward, code: return x.mul_(self.gamma) if self.inplace else x * self.gamma
        mul_17: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_20, primals_33);  view_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        add_14: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_10, mul_17);  add_10 = mul_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_4 = torch.ops.aten.var_mean.correction(add_14, [2], correction = 0, keepdim = True)
        getitem_32: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = var_mean_4[0]
        getitem_33: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = var_mean_4[1];  var_mean_4 = None
        add_15: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_32, 1e-06);  getitem_32 = None
        rsqrt_4: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_15);  add_15 = None
        sub_4: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_14, getitem_33);  getitem_33 = None
        mul_18: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_4);  sub_4 = None
        mul_19: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_18, primals_34)
        add_16: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_19, primals_35);  mul_19 = primals_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        convert_element_type_51: "bf16[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_37, torch.bfloat16);  primals_37 = None
        convert_element_type_52: "bf16[2304, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_36, torch.bfloat16);  primals_36 = None
        convert_element_type_53: "bf16[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_16, torch.bfloat16);  add_16 = None
        view_21: "bf16[175360, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_53, [175360, 768]);  convert_element_type_53 = None
        permute_13: "bf16[768, 2304][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_52, [1, 0]);  convert_element_type_52 = None
        addmm_8: "bf16[175360, 2304][2304, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_51, view_21, permute_13);  convert_element_type_51 = None
        view_22: "bf16[128, 1370, 2304][3156480, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_8, [128, 1370, 2304]);  addmm_8 = None
        view_23: "bf16[128, 1370, 3, 12, 64][3156480, 2304, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_22, [128, 1370, 3, 12, 64]);  view_22 = None
        permute_14: "bf16[3, 128, 12, 1370, 64][768, 3156480, 64, 2304, 1]cuda:0" = torch.ops.aten.permute.default(view_23, [2, 0, 3, 1, 4]);  view_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        unbind_2 = torch.ops.aten.unbind.int(permute_14);  permute_14 = None
        getitem_34: "bf16[128, 12, 1370, 64][3156480, 64, 2304, 1]cuda:0" = unbind_2[0]
        getitem_35: "bf16[128, 12, 1370, 64][3156480, 64, 2304, 1]cuda:0" = unbind_2[1]
        getitem_36: "bf16[128, 12, 1370, 64][3156480, 64, 2304, 1]cuda:0" = unbind_2[2];  unbind_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:120 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention_2 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(getitem_34, getitem_35, getitem_36, None, True)
        getitem_37: "bf16[128, 12, 1370, 64][1052160, 64, 768, 1]cuda:0" = _scaled_dot_product_cudnn_attention_2[0]
        getitem_38: "f32[128, 12, 1370, 1][16440, 1370, 1, 1]cuda:0" = _scaled_dot_product_cudnn_attention_2[1]
        getitem_43: "i64[][]cuda:0" = _scaled_dot_product_cudnn_attention_2[6]
        getitem_44: "i64[][]cuda:0" = _scaled_dot_product_cudnn_attention_2[7];  _scaled_dot_product_cudnn_attention_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_15: "bf16[128, 1370, 12, 64][1052160, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_37, [0, 2, 1, 3])
        view_24: "bf16[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_15, [128, 1370, 768]);  permute_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        convert_element_type_57: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_39, torch.bfloat16);  primals_39 = None
        convert_element_type_58: "bf16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_38, torch.bfloat16);  primals_38 = None
        view_25: "bf16[175360, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_24, [175360, 768]);  view_24 = None
        permute_16: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_58, [1, 0]);  convert_element_type_58 = None
        addmm_9: "bf16[175360, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_57, view_25, permute_16);  convert_element_type_57 = view_25 = None
        view_26: "bf16[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_9, [128, 1370, 768])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/layer_scale.py:27 in forward, code: return x.mul_(self.gamma) if self.inplace else x * self.gamma
        mul_20: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_26, primals_40);  view_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        add_17: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_14, mul_20);  add_14 = mul_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_5 = torch.ops.aten.var_mean.correction(add_17, [2], correction = 0, keepdim = True)
        getitem_46: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = var_mean_5[0]
        getitem_47: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = var_mean_5[1];  var_mean_5 = None
        add_18: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_46, 1e-06);  getitem_46 = None
        rsqrt_5: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_18);  add_18 = None
        sub_5: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_17, getitem_47);  getitem_47 = None
        mul_21: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_5);  sub_5 = None
        mul_22: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_21, primals_41)
        add_19: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_22, primals_42);  mul_22 = primals_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        convert_element_type_62: "bf16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_44, torch.bfloat16);  primals_44 = None
        convert_element_type_63: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_43, torch.bfloat16);  primals_43 = None
        convert_element_type_64: "bf16[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_19, torch.bfloat16);  add_19 = None
        view_27: "bf16[175360, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_64, [175360, 768]);  convert_element_type_64 = None
        permute_17: "bf16[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_63, [1, 0]);  convert_element_type_63 = None
        addmm_10: "bf16[175360, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_62, view_27, permute_17);  convert_element_type_62 = None
        view_28: "bf16[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_10, [128, 1370, 3072])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_68: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_28, torch.float32);  view_28 = None
        mul_23: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_68, 0.5)
        mul_24: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_68, 0.7071067811865476);  convert_element_type_68 = None
        erf_2: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_24);  mul_24 = None
        add_20: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
        mul_25: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_23, add_20);  mul_23 = add_20 = None
        convert_element_type_69: "bf16[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_25, torch.bfloat16);  mul_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        convert_element_type_70: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_46, torch.bfloat16);  primals_46 = None
        convert_element_type_71: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_45, torch.bfloat16);  primals_45 = None
        view_29: "bf16[175360, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_69, [175360, 3072]);  convert_element_type_69 = None
        permute_18: "bf16[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(convert_element_type_71, [1, 0]);  convert_element_type_71 = None
        addmm_11: "bf16[175360, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_70, view_29, permute_18);  convert_element_type_70 = None
        view_30: "bf16[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_11, [128, 1370, 768])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/layer_scale.py:27 in forward, code: return x.mul_(self.gamma) if self.inplace else x * self.gamma
        mul_26: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_30, primals_47);  view_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        add_21: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_17, mul_26);  add_17 = mul_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_6 = torch.ops.aten.var_mean.correction(add_21, [2], correction = 0, keepdim = True)
        getitem_48: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = var_mean_6[0]
        getitem_49: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = var_mean_6[1];  var_mean_6 = None
        add_22: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_48, 1e-06);  getitem_48 = None
        rsqrt_6: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_22);  add_22 = None
        sub_6: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_21, getitem_49);  getitem_49 = None
        mul_27: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_6);  sub_6 = None
        mul_28: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_27, primals_48)
        add_23: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_28, primals_49);  mul_28 = primals_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        convert_element_type_75: "bf16[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_51, torch.bfloat16);  primals_51 = None
        convert_element_type_76: "bf16[2304, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_50, torch.bfloat16);  primals_50 = None
        convert_element_type_77: "bf16[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_23, torch.bfloat16);  add_23 = None
        view_31: "bf16[175360, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_77, [175360, 768]);  convert_element_type_77 = None
        permute_19: "bf16[768, 2304][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_76, [1, 0]);  convert_element_type_76 = None
        addmm_12: "bf16[175360, 2304][2304, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_75, view_31, permute_19);  convert_element_type_75 = None
        view_32: "bf16[128, 1370, 2304][3156480, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_12, [128, 1370, 2304]);  addmm_12 = None
        view_33: "bf16[128, 1370, 3, 12, 64][3156480, 2304, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_32, [128, 1370, 3, 12, 64]);  view_32 = None
        permute_20: "bf16[3, 128, 12, 1370, 64][768, 3156480, 64, 2304, 1]cuda:0" = torch.ops.aten.permute.default(view_33, [2, 0, 3, 1, 4]);  view_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        unbind_3 = torch.ops.aten.unbind.int(permute_20);  permute_20 = None
        getitem_50: "bf16[128, 12, 1370, 64][3156480, 64, 2304, 1]cuda:0" = unbind_3[0]
        getitem_51: "bf16[128, 12, 1370, 64][3156480, 64, 2304, 1]cuda:0" = unbind_3[1]
        getitem_52: "bf16[128, 12, 1370, 64][3156480, 64, 2304, 1]cuda:0" = unbind_3[2];  unbind_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:120 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention_3 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(getitem_50, getitem_51, getitem_52, None, True)
        getitem_53: "bf16[128, 12, 1370, 64][1052160, 64, 768, 1]cuda:0" = _scaled_dot_product_cudnn_attention_3[0]
        getitem_54: "f32[128, 12, 1370, 1][16440, 1370, 1, 1]cuda:0" = _scaled_dot_product_cudnn_attention_3[1]
        getitem_59: "i64[][]cuda:0" = _scaled_dot_product_cudnn_attention_3[6]
        getitem_60: "i64[][]cuda:0" = _scaled_dot_product_cudnn_attention_3[7];  _scaled_dot_product_cudnn_attention_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_21: "bf16[128, 1370, 12, 64][1052160, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_53, [0, 2, 1, 3])
        view_34: "bf16[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_21, [128, 1370, 768]);  permute_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        convert_element_type_81: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_53, torch.bfloat16);  primals_53 = None
        convert_element_type_82: "bf16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_52, torch.bfloat16);  primals_52 = None
        view_35: "bf16[175360, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_34, [175360, 768]);  view_34 = None
        permute_22: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_82, [1, 0]);  convert_element_type_82 = None
        addmm_13: "bf16[175360, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_81, view_35, permute_22);  convert_element_type_81 = view_35 = None
        view_36: "bf16[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_13, [128, 1370, 768])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/layer_scale.py:27 in forward, code: return x.mul_(self.gamma) if self.inplace else x * self.gamma
        mul_29: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_36, primals_54);  view_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        add_24: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_21, mul_29);  add_21 = mul_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_7 = torch.ops.aten.var_mean.correction(add_24, [2], correction = 0, keepdim = True)
        getitem_62: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = var_mean_7[0]
        getitem_63: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = var_mean_7[1];  var_mean_7 = None
        add_25: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_62, 1e-06);  getitem_62 = None
        rsqrt_7: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_25);  add_25 = None
        sub_7: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_24, getitem_63);  getitem_63 = None
        mul_30: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_7);  sub_7 = None
        mul_31: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_30, primals_55)
        add_26: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_31, primals_56);  mul_31 = primals_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        convert_element_type_86: "bf16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_58, torch.bfloat16);  primals_58 = None
        convert_element_type_87: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_57, torch.bfloat16);  primals_57 = None
        convert_element_type_88: "bf16[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_26, torch.bfloat16);  add_26 = None
        view_37: "bf16[175360, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_88, [175360, 768]);  convert_element_type_88 = None
        permute_23: "bf16[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_87, [1, 0]);  convert_element_type_87 = None
        addmm_14: "bf16[175360, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_86, view_37, permute_23);  convert_element_type_86 = None
        view_38: "bf16[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_14, [128, 1370, 3072])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_92: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_38, torch.float32);  view_38 = None
        mul_32: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_92, 0.5)
        mul_33: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_92, 0.7071067811865476);  convert_element_type_92 = None
        erf_3: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_33);  mul_33 = None
        add_27: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
        mul_34: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_32, add_27);  mul_32 = add_27 = None
        convert_element_type_93: "bf16[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_34, torch.bfloat16);  mul_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        convert_element_type_94: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_60, torch.bfloat16);  primals_60 = None
        convert_element_type_95: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_59, torch.bfloat16);  primals_59 = None
        view_39: "bf16[175360, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_93, [175360, 3072]);  convert_element_type_93 = None
        permute_24: "bf16[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(convert_element_type_95, [1, 0]);  convert_element_type_95 = None
        addmm_15: "bf16[175360, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_94, view_39, permute_24);  convert_element_type_94 = None
        view_40: "bf16[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_15, [128, 1370, 768])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/layer_scale.py:27 in forward, code: return x.mul_(self.gamma) if self.inplace else x * self.gamma
        mul_35: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_40, primals_61);  view_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        add_28: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_24, mul_35);  add_24 = mul_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_8 = torch.ops.aten.var_mean.correction(add_28, [2], correction = 0, keepdim = True)
        getitem_64: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = var_mean_8[0]
        getitem_65: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = var_mean_8[1];  var_mean_8 = None
        add_29: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_64, 1e-06);  getitem_64 = None
        rsqrt_8: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_29);  add_29 = None
        sub_8: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_28, getitem_65);  getitem_65 = None
        mul_36: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_8);  sub_8 = None
        mul_37: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_36, primals_62)
        add_30: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_37, primals_63);  mul_37 = primals_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        convert_element_type_99: "bf16[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_65, torch.bfloat16);  primals_65 = None
        convert_element_type_100: "bf16[2304, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_64, torch.bfloat16);  primals_64 = None
        convert_element_type_101: "bf16[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_30, torch.bfloat16);  add_30 = None
        view_41: "bf16[175360, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_101, [175360, 768]);  convert_element_type_101 = None
        permute_25: "bf16[768, 2304][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_100, [1, 0]);  convert_element_type_100 = None
        addmm_16: "bf16[175360, 2304][2304, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_99, view_41, permute_25);  convert_element_type_99 = None
        view_42: "bf16[128, 1370, 2304][3156480, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_16, [128, 1370, 2304]);  addmm_16 = None
        view_43: "bf16[128, 1370, 3, 12, 64][3156480, 2304, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_42, [128, 1370, 3, 12, 64]);  view_42 = None
        permute_26: "bf16[3, 128, 12, 1370, 64][768, 3156480, 64, 2304, 1]cuda:0" = torch.ops.aten.permute.default(view_43, [2, 0, 3, 1, 4]);  view_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        unbind_4 = torch.ops.aten.unbind.int(permute_26);  permute_26 = None
        getitem_66: "bf16[128, 12, 1370, 64][3156480, 64, 2304, 1]cuda:0" = unbind_4[0]
        getitem_67: "bf16[128, 12, 1370, 64][3156480, 64, 2304, 1]cuda:0" = unbind_4[1]
        getitem_68: "bf16[128, 12, 1370, 64][3156480, 64, 2304, 1]cuda:0" = unbind_4[2];  unbind_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:120 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention_4 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(getitem_66, getitem_67, getitem_68, None, True)
        getitem_69: "bf16[128, 12, 1370, 64][1052160, 64, 768, 1]cuda:0" = _scaled_dot_product_cudnn_attention_4[0]
        getitem_70: "f32[128, 12, 1370, 1][16440, 1370, 1, 1]cuda:0" = _scaled_dot_product_cudnn_attention_4[1]
        getitem_75: "i64[][]cuda:0" = _scaled_dot_product_cudnn_attention_4[6]
        getitem_76: "i64[][]cuda:0" = _scaled_dot_product_cudnn_attention_4[7];  _scaled_dot_product_cudnn_attention_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_27: "bf16[128, 1370, 12, 64][1052160, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_69, [0, 2, 1, 3])
        view_44: "bf16[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_27, [128, 1370, 768]);  permute_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        convert_element_type_105: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_67, torch.bfloat16);  primals_67 = None
        convert_element_type_106: "bf16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_66, torch.bfloat16);  primals_66 = None
        view_45: "bf16[175360, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_44, [175360, 768]);  view_44 = None
        permute_28: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_106, [1, 0]);  convert_element_type_106 = None
        addmm_17: "bf16[175360, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_105, view_45, permute_28);  convert_element_type_105 = view_45 = None
        view_46: "bf16[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_17, [128, 1370, 768])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/layer_scale.py:27 in forward, code: return x.mul_(self.gamma) if self.inplace else x * self.gamma
        mul_38: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_46, primals_68);  view_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        add_31: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_28, mul_38);  add_28 = mul_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_9 = torch.ops.aten.var_mean.correction(add_31, [2], correction = 0, keepdim = True)
        getitem_78: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = var_mean_9[0]
        getitem_79: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = var_mean_9[1];  var_mean_9 = None
        add_32: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_78, 1e-06);  getitem_78 = None
        rsqrt_9: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_32);  add_32 = None
        sub_9: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_31, getitem_79);  getitem_79 = None
        mul_39: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_9);  sub_9 = None
        mul_40: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_39, primals_69)
        add_33: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_40, primals_70);  mul_40 = primals_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        convert_element_type_110: "bf16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_72, torch.bfloat16);  primals_72 = None
        convert_element_type_111: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_71, torch.bfloat16);  primals_71 = None
        convert_element_type_112: "bf16[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_33, torch.bfloat16);  add_33 = None
        view_47: "bf16[175360, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_112, [175360, 768]);  convert_element_type_112 = None
        permute_29: "bf16[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_111, [1, 0]);  convert_element_type_111 = None
        addmm_18: "bf16[175360, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_110, view_47, permute_29);  convert_element_type_110 = None
        view_48: "bf16[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_18, [128, 1370, 3072])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_116: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_48, torch.float32);  view_48 = None
        mul_41: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_116, 0.5)
        mul_42: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_116, 0.7071067811865476);  convert_element_type_116 = None
        erf_4: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_42);  mul_42 = None
        add_34: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
        mul_43: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_41, add_34);  mul_41 = add_34 = None
        convert_element_type_117: "bf16[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_43, torch.bfloat16);  mul_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        convert_element_type_118: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_74, torch.bfloat16);  primals_74 = None
        convert_element_type_119: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_73, torch.bfloat16);  primals_73 = None
        view_49: "bf16[175360, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_117, [175360, 3072]);  convert_element_type_117 = None
        permute_30: "bf16[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(convert_element_type_119, [1, 0]);  convert_element_type_119 = None
        addmm_19: "bf16[175360, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_118, view_49, permute_30);  convert_element_type_118 = None
        view_50: "bf16[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_19, [128, 1370, 768])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/layer_scale.py:27 in forward, code: return x.mul_(self.gamma) if self.inplace else x * self.gamma
        mul_44: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_50, primals_75);  view_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        add_35: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_31, mul_44);  add_31 = mul_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_10 = torch.ops.aten.var_mean.correction(add_35, [2], correction = 0, keepdim = True)
        getitem_80: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = var_mean_10[0]
        getitem_81: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = var_mean_10[1];  var_mean_10 = None
        add_36: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_80, 1e-06);  getitem_80 = None
        rsqrt_10: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_36);  add_36 = None
        sub_10: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_35, getitem_81);  getitem_81 = None
        mul_45: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_10);  sub_10 = None
        mul_46: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_45, primals_76)
        add_37: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_46, primals_77);  mul_46 = primals_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        convert_element_type_123: "bf16[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_79, torch.bfloat16);  primals_79 = None
        convert_element_type_124: "bf16[2304, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_78, torch.bfloat16);  primals_78 = None
        convert_element_type_125: "bf16[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_37, torch.bfloat16);  add_37 = None
        view_51: "bf16[175360, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_125, [175360, 768]);  convert_element_type_125 = None
        permute_31: "bf16[768, 2304][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_124, [1, 0]);  convert_element_type_124 = None
        addmm_20: "bf16[175360, 2304][2304, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_123, view_51, permute_31);  convert_element_type_123 = None
        view_52: "bf16[128, 1370, 2304][3156480, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_20, [128, 1370, 2304]);  addmm_20 = None
        view_53: "bf16[128, 1370, 3, 12, 64][3156480, 2304, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_52, [128, 1370, 3, 12, 64]);  view_52 = None
        permute_32: "bf16[3, 128, 12, 1370, 64][768, 3156480, 64, 2304, 1]cuda:0" = torch.ops.aten.permute.default(view_53, [2, 0, 3, 1, 4]);  view_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        unbind_5 = torch.ops.aten.unbind.int(permute_32);  permute_32 = None
        getitem_82: "bf16[128, 12, 1370, 64][3156480, 64, 2304, 1]cuda:0" = unbind_5[0]
        getitem_83: "bf16[128, 12, 1370, 64][3156480, 64, 2304, 1]cuda:0" = unbind_5[1]
        getitem_84: "bf16[128, 12, 1370, 64][3156480, 64, 2304, 1]cuda:0" = unbind_5[2];  unbind_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:120 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention_5 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(getitem_82, getitem_83, getitem_84, None, True)
        getitem_85: "bf16[128, 12, 1370, 64][1052160, 64, 768, 1]cuda:0" = _scaled_dot_product_cudnn_attention_5[0]
        getitem_86: "f32[128, 12, 1370, 1][16440, 1370, 1, 1]cuda:0" = _scaled_dot_product_cudnn_attention_5[1]
        getitem_91: "i64[][]cuda:0" = _scaled_dot_product_cudnn_attention_5[6]
        getitem_92: "i64[][]cuda:0" = _scaled_dot_product_cudnn_attention_5[7];  _scaled_dot_product_cudnn_attention_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_33: "bf16[128, 1370, 12, 64][1052160, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_85, [0, 2, 1, 3])
        view_54: "bf16[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_33, [128, 1370, 768]);  permute_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        convert_element_type_129: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_81, torch.bfloat16);  primals_81 = None
        convert_element_type_130: "bf16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_80, torch.bfloat16);  primals_80 = None
        view_55: "bf16[175360, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_54, [175360, 768]);  view_54 = None
        permute_34: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_130, [1, 0]);  convert_element_type_130 = None
        addmm_21: "bf16[175360, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_129, view_55, permute_34);  convert_element_type_129 = view_55 = None
        view_56: "bf16[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_21, [128, 1370, 768])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/layer_scale.py:27 in forward, code: return x.mul_(self.gamma) if self.inplace else x * self.gamma
        mul_47: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_56, primals_82);  view_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        add_38: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_35, mul_47);  add_35 = mul_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_11 = torch.ops.aten.var_mean.correction(add_38, [2], correction = 0, keepdim = True)
        getitem_94: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = var_mean_11[0]
        getitem_95: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = var_mean_11[1];  var_mean_11 = None
        add_39: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_94, 1e-06);  getitem_94 = None
        rsqrt_11: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_39);  add_39 = None
        sub_11: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_38, getitem_95);  getitem_95 = None
        mul_48: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_11);  sub_11 = None
        mul_49: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_48, primals_83)
        add_40: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_49, primals_84);  mul_49 = primals_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        convert_element_type_134: "bf16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_86, torch.bfloat16);  primals_86 = None
        convert_element_type_135: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_85, torch.bfloat16);  primals_85 = None
        convert_element_type_136: "bf16[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_40, torch.bfloat16);  add_40 = None
        view_57: "bf16[175360, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_136, [175360, 768]);  convert_element_type_136 = None
        permute_35: "bf16[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_135, [1, 0]);  convert_element_type_135 = None
        addmm_22: "bf16[175360, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_134, view_57, permute_35);  convert_element_type_134 = None
        view_58: "bf16[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_22, [128, 1370, 3072])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_140: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_58, torch.float32);  view_58 = None
        mul_50: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_140, 0.5)
        mul_51: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_140, 0.7071067811865476);  convert_element_type_140 = None
        erf_5: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_51);  mul_51 = None
        add_41: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
        mul_52: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_50, add_41);  mul_50 = add_41 = None
        convert_element_type_141: "bf16[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_52, torch.bfloat16);  mul_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        convert_element_type_142: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_88, torch.bfloat16);  primals_88 = None
        convert_element_type_143: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_87, torch.bfloat16);  primals_87 = None
        view_59: "bf16[175360, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_141, [175360, 3072]);  convert_element_type_141 = None
        permute_36: "bf16[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(convert_element_type_143, [1, 0]);  convert_element_type_143 = None
        addmm_23: "bf16[175360, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_142, view_59, permute_36);  convert_element_type_142 = None
        view_60: "bf16[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_23, [128, 1370, 768])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/layer_scale.py:27 in forward, code: return x.mul_(self.gamma) if self.inplace else x * self.gamma
        mul_53: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_60, primals_89);  view_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        add_42: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_38, mul_53);  add_38 = mul_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_12 = torch.ops.aten.var_mean.correction(add_42, [2], correction = 0, keepdim = True)
        getitem_96: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = var_mean_12[0]
        getitem_97: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = var_mean_12[1];  var_mean_12 = None
        add_43: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_96, 1e-06);  getitem_96 = None
        rsqrt_12: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_43);  add_43 = None
        sub_12: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_42, getitem_97);  getitem_97 = None
        mul_54: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_12);  sub_12 = None
        mul_55: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_54, primals_90)
        add_44: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_55, primals_91);  mul_55 = primals_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        convert_element_type_147: "bf16[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_93, torch.bfloat16);  primals_93 = None
        convert_element_type_148: "bf16[2304, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_92, torch.bfloat16);  primals_92 = None
        convert_element_type_149: "bf16[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_44, torch.bfloat16);  add_44 = None
        view_61: "bf16[175360, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_149, [175360, 768]);  convert_element_type_149 = None
        permute_37: "bf16[768, 2304][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_148, [1, 0]);  convert_element_type_148 = None
        addmm_24: "bf16[175360, 2304][2304, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_147, view_61, permute_37);  convert_element_type_147 = None
        view_62: "bf16[128, 1370, 2304][3156480, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_24, [128, 1370, 2304]);  addmm_24 = None
        view_63: "bf16[128, 1370, 3, 12, 64][3156480, 2304, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_62, [128, 1370, 3, 12, 64]);  view_62 = None
        permute_38: "bf16[3, 128, 12, 1370, 64][768, 3156480, 64, 2304, 1]cuda:0" = torch.ops.aten.permute.default(view_63, [2, 0, 3, 1, 4]);  view_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        unbind_6 = torch.ops.aten.unbind.int(permute_38);  permute_38 = None
        getitem_98: "bf16[128, 12, 1370, 64][3156480, 64, 2304, 1]cuda:0" = unbind_6[0]
        getitem_99: "bf16[128, 12, 1370, 64][3156480, 64, 2304, 1]cuda:0" = unbind_6[1]
        getitem_100: "bf16[128, 12, 1370, 64][3156480, 64, 2304, 1]cuda:0" = unbind_6[2];  unbind_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:120 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention_6 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(getitem_98, getitem_99, getitem_100, None, True)
        getitem_101: "bf16[128, 12, 1370, 64][1052160, 64, 768, 1]cuda:0" = _scaled_dot_product_cudnn_attention_6[0]
        getitem_102: "f32[128, 12, 1370, 1][16440, 1370, 1, 1]cuda:0" = _scaled_dot_product_cudnn_attention_6[1]
        getitem_107: "i64[][]cuda:0" = _scaled_dot_product_cudnn_attention_6[6]
        getitem_108: "i64[][]cuda:0" = _scaled_dot_product_cudnn_attention_6[7];  _scaled_dot_product_cudnn_attention_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_39: "bf16[128, 1370, 12, 64][1052160, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_101, [0, 2, 1, 3])
        view_64: "bf16[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_39, [128, 1370, 768]);  permute_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        convert_element_type_153: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_95, torch.bfloat16);  primals_95 = None
        convert_element_type_154: "bf16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_94, torch.bfloat16);  primals_94 = None
        view_65: "bf16[175360, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_64, [175360, 768]);  view_64 = None
        permute_40: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_154, [1, 0]);  convert_element_type_154 = None
        addmm_25: "bf16[175360, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_153, view_65, permute_40);  convert_element_type_153 = view_65 = None
        view_66: "bf16[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_25, [128, 1370, 768])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/layer_scale.py:27 in forward, code: return x.mul_(self.gamma) if self.inplace else x * self.gamma
        mul_56: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_66, primals_96);  view_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        add_45: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_42, mul_56);  add_42 = mul_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_13 = torch.ops.aten.var_mean.correction(add_45, [2], correction = 0, keepdim = True)
        getitem_110: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = var_mean_13[0]
        getitem_111: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = var_mean_13[1];  var_mean_13 = None
        add_46: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_110, 1e-06);  getitem_110 = None
        rsqrt_13: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_46);  add_46 = None
        sub_13: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_45, getitem_111);  getitem_111 = None
        mul_57: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_13);  sub_13 = None
        mul_58: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_57, primals_97)
        add_47: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_58, primals_98);  mul_58 = primals_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        convert_element_type_158: "bf16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_100, torch.bfloat16);  primals_100 = None
        convert_element_type_159: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_99, torch.bfloat16);  primals_99 = None
        convert_element_type_160: "bf16[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_47, torch.bfloat16);  add_47 = None
        view_67: "bf16[175360, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_160, [175360, 768]);  convert_element_type_160 = None
        permute_41: "bf16[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_159, [1, 0]);  convert_element_type_159 = None
        addmm_26: "bf16[175360, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_158, view_67, permute_41);  convert_element_type_158 = None
        view_68: "bf16[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_26, [128, 1370, 3072])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_164: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_68, torch.float32);  view_68 = None
        mul_59: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_164, 0.5)
        mul_60: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_164, 0.7071067811865476);  convert_element_type_164 = None
        erf_6: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_60);  mul_60 = None
        add_48: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
        mul_61: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_59, add_48);  mul_59 = add_48 = None
        convert_element_type_165: "bf16[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_61, torch.bfloat16);  mul_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        convert_element_type_166: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_102, torch.bfloat16);  primals_102 = None
        convert_element_type_167: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_101, torch.bfloat16);  primals_101 = None
        view_69: "bf16[175360, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_165, [175360, 3072]);  convert_element_type_165 = None
        permute_42: "bf16[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(convert_element_type_167, [1, 0]);  convert_element_type_167 = None
        addmm_27: "bf16[175360, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_166, view_69, permute_42);  convert_element_type_166 = None
        view_70: "bf16[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_27, [128, 1370, 768])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/layer_scale.py:27 in forward, code: return x.mul_(self.gamma) if self.inplace else x * self.gamma
        mul_62: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_70, primals_103);  view_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        add_49: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_45, mul_62);  add_45 = mul_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_14 = torch.ops.aten.var_mean.correction(add_49, [2], correction = 0, keepdim = True)
        getitem_112: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = var_mean_14[0]
        getitem_113: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = var_mean_14[1];  var_mean_14 = None
        add_50: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_112, 1e-06);  getitem_112 = None
        rsqrt_14: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_50);  add_50 = None
        sub_14: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_49, getitem_113);  getitem_113 = None
        mul_63: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_14);  sub_14 = None
        mul_64: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_63, primals_104)
        add_51: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_64, primals_105);  mul_64 = primals_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        convert_element_type_171: "bf16[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_107, torch.bfloat16);  primals_107 = None
        convert_element_type_172: "bf16[2304, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_106, torch.bfloat16);  primals_106 = None
        convert_element_type_173: "bf16[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_51, torch.bfloat16);  add_51 = None
        view_71: "bf16[175360, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_173, [175360, 768]);  convert_element_type_173 = None
        permute_43: "bf16[768, 2304][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_172, [1, 0]);  convert_element_type_172 = None
        addmm_28: "bf16[175360, 2304][2304, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_171, view_71, permute_43);  convert_element_type_171 = None
        view_72: "bf16[128, 1370, 2304][3156480, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_28, [128, 1370, 2304]);  addmm_28 = None
        view_73: "bf16[128, 1370, 3, 12, 64][3156480, 2304, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_72, [128, 1370, 3, 12, 64]);  view_72 = None
        permute_44: "bf16[3, 128, 12, 1370, 64][768, 3156480, 64, 2304, 1]cuda:0" = torch.ops.aten.permute.default(view_73, [2, 0, 3, 1, 4]);  view_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        unbind_7 = torch.ops.aten.unbind.int(permute_44);  permute_44 = None
        getitem_114: "bf16[128, 12, 1370, 64][3156480, 64, 2304, 1]cuda:0" = unbind_7[0]
        getitem_115: "bf16[128, 12, 1370, 64][3156480, 64, 2304, 1]cuda:0" = unbind_7[1]
        getitem_116: "bf16[128, 12, 1370, 64][3156480, 64, 2304, 1]cuda:0" = unbind_7[2];  unbind_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:120 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention_7 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(getitem_114, getitem_115, getitem_116, None, True)
        getitem_117: "bf16[128, 12, 1370, 64][1052160, 64, 768, 1]cuda:0" = _scaled_dot_product_cudnn_attention_7[0]
        getitem_118: "f32[128, 12, 1370, 1][16440, 1370, 1, 1]cuda:0" = _scaled_dot_product_cudnn_attention_7[1]
        getitem_123: "i64[][]cuda:0" = _scaled_dot_product_cudnn_attention_7[6]
        getitem_124: "i64[][]cuda:0" = _scaled_dot_product_cudnn_attention_7[7];  _scaled_dot_product_cudnn_attention_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_45: "bf16[128, 1370, 12, 64][1052160, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_117, [0, 2, 1, 3])
        view_74: "bf16[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_45, [128, 1370, 768]);  permute_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        convert_element_type_177: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_109, torch.bfloat16);  primals_109 = None
        convert_element_type_178: "bf16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_108, torch.bfloat16);  primals_108 = None
        view_75: "bf16[175360, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_74, [175360, 768]);  view_74 = None
        permute_46: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_178, [1, 0]);  convert_element_type_178 = None
        addmm_29: "bf16[175360, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_177, view_75, permute_46);  convert_element_type_177 = view_75 = None
        view_76: "bf16[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_29, [128, 1370, 768])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/layer_scale.py:27 in forward, code: return x.mul_(self.gamma) if self.inplace else x * self.gamma
        mul_65: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_76, primals_110);  view_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        add_52: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_49, mul_65);  add_49 = mul_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_15 = torch.ops.aten.var_mean.correction(add_52, [2], correction = 0, keepdim = True)
        getitem_126: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = var_mean_15[0]
        getitem_127: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = var_mean_15[1];  var_mean_15 = None
        add_53: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_126, 1e-06);  getitem_126 = None
        rsqrt_15: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_53);  add_53 = None
        sub_15: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_52, getitem_127);  getitem_127 = None
        mul_66: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_15);  sub_15 = None
        mul_67: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_66, primals_111)
        add_54: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_67, primals_112);  mul_67 = primals_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        convert_element_type_182: "bf16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_114, torch.bfloat16);  primals_114 = None
        convert_element_type_183: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_113, torch.bfloat16);  primals_113 = None
        convert_element_type_184: "bf16[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_54, torch.bfloat16);  add_54 = None
        view_77: "bf16[175360, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_184, [175360, 768]);  convert_element_type_184 = None
        permute_47: "bf16[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_183, [1, 0]);  convert_element_type_183 = None
        addmm_30: "bf16[175360, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_182, view_77, permute_47);  convert_element_type_182 = None
        view_78: "bf16[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_30, [128, 1370, 3072])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_188: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_78, torch.float32);  view_78 = None
        mul_68: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_188, 0.5)
        mul_69: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_188, 0.7071067811865476);  convert_element_type_188 = None
        erf_7: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_69);  mul_69 = None
        add_55: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
        mul_70: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_68, add_55);  mul_68 = add_55 = None
        convert_element_type_189: "bf16[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_70, torch.bfloat16);  mul_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        convert_element_type_190: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_116, torch.bfloat16);  primals_116 = None
        convert_element_type_191: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_115, torch.bfloat16);  primals_115 = None
        view_79: "bf16[175360, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_189, [175360, 3072]);  convert_element_type_189 = None
        permute_48: "bf16[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(convert_element_type_191, [1, 0]);  convert_element_type_191 = None
        addmm_31: "bf16[175360, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_190, view_79, permute_48);  convert_element_type_190 = None
        view_80: "bf16[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_31, [128, 1370, 768])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/layer_scale.py:27 in forward, code: return x.mul_(self.gamma) if self.inplace else x * self.gamma
        mul_71: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_80, primals_117);  view_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        add_56: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_52, mul_71);  add_52 = mul_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_16 = torch.ops.aten.var_mean.correction(add_56, [2], correction = 0, keepdim = True)
        getitem_128: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = var_mean_16[0]
        getitem_129: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = var_mean_16[1];  var_mean_16 = None
        add_57: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_128, 1e-06);  getitem_128 = None
        rsqrt_16: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_57);  add_57 = None
        sub_16: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_56, getitem_129);  getitem_129 = None
        mul_72: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_16);  sub_16 = None
        mul_73: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_72, primals_118)
        add_58: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_73, primals_119);  mul_73 = primals_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        convert_element_type_195: "bf16[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_121, torch.bfloat16);  primals_121 = None
        convert_element_type_196: "bf16[2304, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_120, torch.bfloat16);  primals_120 = None
        convert_element_type_197: "bf16[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_58, torch.bfloat16);  add_58 = None
        view_81: "bf16[175360, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_197, [175360, 768]);  convert_element_type_197 = None
        permute_49: "bf16[768, 2304][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_196, [1, 0]);  convert_element_type_196 = None
        addmm_32: "bf16[175360, 2304][2304, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_195, view_81, permute_49);  convert_element_type_195 = None
        view_82: "bf16[128, 1370, 2304][3156480, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_32, [128, 1370, 2304]);  addmm_32 = None
        view_83: "bf16[128, 1370, 3, 12, 64][3156480, 2304, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_82, [128, 1370, 3, 12, 64]);  view_82 = None
        permute_50: "bf16[3, 128, 12, 1370, 64][768, 3156480, 64, 2304, 1]cuda:0" = torch.ops.aten.permute.default(view_83, [2, 0, 3, 1, 4]);  view_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        unbind_8 = torch.ops.aten.unbind.int(permute_50);  permute_50 = None
        getitem_130: "bf16[128, 12, 1370, 64][3156480, 64, 2304, 1]cuda:0" = unbind_8[0]
        getitem_131: "bf16[128, 12, 1370, 64][3156480, 64, 2304, 1]cuda:0" = unbind_8[1]
        getitem_132: "bf16[128, 12, 1370, 64][3156480, 64, 2304, 1]cuda:0" = unbind_8[2];  unbind_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:120 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention_8 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(getitem_130, getitem_131, getitem_132, None, True)
        getitem_133: "bf16[128, 12, 1370, 64][1052160, 64, 768, 1]cuda:0" = _scaled_dot_product_cudnn_attention_8[0]
        getitem_134: "f32[128, 12, 1370, 1][16440, 1370, 1, 1]cuda:0" = _scaled_dot_product_cudnn_attention_8[1]
        getitem_139: "i64[][]cuda:0" = _scaled_dot_product_cudnn_attention_8[6]
        getitem_140: "i64[][]cuda:0" = _scaled_dot_product_cudnn_attention_8[7];  _scaled_dot_product_cudnn_attention_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_51: "bf16[128, 1370, 12, 64][1052160, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_133, [0, 2, 1, 3])
        view_84: "bf16[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_51, [128, 1370, 768]);  permute_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        convert_element_type_201: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_123, torch.bfloat16);  primals_123 = None
        convert_element_type_202: "bf16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_122, torch.bfloat16);  primals_122 = None
        view_85: "bf16[175360, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_84, [175360, 768]);  view_84 = None
        permute_52: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_202, [1, 0]);  convert_element_type_202 = None
        addmm_33: "bf16[175360, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_201, view_85, permute_52);  convert_element_type_201 = view_85 = None
        view_86: "bf16[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_33, [128, 1370, 768])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/layer_scale.py:27 in forward, code: return x.mul_(self.gamma) if self.inplace else x * self.gamma
        mul_74: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_86, primals_124);  view_86 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        add_59: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_56, mul_74);  add_56 = mul_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_17 = torch.ops.aten.var_mean.correction(add_59, [2], correction = 0, keepdim = True)
        getitem_142: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = var_mean_17[0]
        getitem_143: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = var_mean_17[1];  var_mean_17 = None
        add_60: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_142, 1e-06);  getitem_142 = None
        rsqrt_17: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_60);  add_60 = None
        sub_17: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_59, getitem_143);  getitem_143 = None
        mul_75: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_17);  sub_17 = None
        mul_76: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_75, primals_125)
        add_61: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_76, primals_126);  mul_76 = primals_126 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        convert_element_type_206: "bf16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_128, torch.bfloat16);  primals_128 = None
        convert_element_type_207: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_127, torch.bfloat16);  primals_127 = None
        convert_element_type_208: "bf16[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_61, torch.bfloat16);  add_61 = None
        view_87: "bf16[175360, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_208, [175360, 768]);  convert_element_type_208 = None
        permute_53: "bf16[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_207, [1, 0]);  convert_element_type_207 = None
        addmm_34: "bf16[175360, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_206, view_87, permute_53);  convert_element_type_206 = None
        view_88: "bf16[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_34, [128, 1370, 3072])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_212: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_88, torch.float32);  view_88 = None
        mul_77: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_212, 0.5)
        mul_78: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_212, 0.7071067811865476);  convert_element_type_212 = None
        erf_8: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_78);  mul_78 = None
        add_62: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
        mul_79: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_77, add_62);  mul_77 = add_62 = None
        convert_element_type_213: "bf16[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_79, torch.bfloat16);  mul_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        convert_element_type_214: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_130, torch.bfloat16);  primals_130 = None
        convert_element_type_215: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_129, torch.bfloat16);  primals_129 = None
        view_89: "bf16[175360, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_213, [175360, 3072]);  convert_element_type_213 = None
        permute_54: "bf16[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(convert_element_type_215, [1, 0]);  convert_element_type_215 = None
        addmm_35: "bf16[175360, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_214, view_89, permute_54);  convert_element_type_214 = None
        view_90: "bf16[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_35, [128, 1370, 768])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/layer_scale.py:27 in forward, code: return x.mul_(self.gamma) if self.inplace else x * self.gamma
        mul_80: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_90, primals_131);  view_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        add_63: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_59, mul_80);  add_59 = mul_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_18 = torch.ops.aten.var_mean.correction(add_63, [2], correction = 0, keepdim = True)
        getitem_144: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = var_mean_18[0]
        getitem_145: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = var_mean_18[1];  var_mean_18 = None
        add_64: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_144, 1e-06);  getitem_144 = None
        rsqrt_18: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_64);  add_64 = None
        sub_18: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_63, getitem_145);  getitem_145 = None
        mul_81: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_18);  sub_18 = None
        mul_82: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_81, primals_132)
        add_65: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_82, primals_133);  mul_82 = primals_133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        convert_element_type_219: "bf16[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_135, torch.bfloat16);  primals_135 = None
        convert_element_type_220: "bf16[2304, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_134, torch.bfloat16);  primals_134 = None
        convert_element_type_221: "bf16[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_65, torch.bfloat16);  add_65 = None
        view_91: "bf16[175360, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_221, [175360, 768]);  convert_element_type_221 = None
        permute_55: "bf16[768, 2304][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_220, [1, 0]);  convert_element_type_220 = None
        addmm_36: "bf16[175360, 2304][2304, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_219, view_91, permute_55);  convert_element_type_219 = None
        view_92: "bf16[128, 1370, 2304][3156480, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_36, [128, 1370, 2304]);  addmm_36 = None
        view_93: "bf16[128, 1370, 3, 12, 64][3156480, 2304, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_92, [128, 1370, 3, 12, 64]);  view_92 = None
        permute_56: "bf16[3, 128, 12, 1370, 64][768, 3156480, 64, 2304, 1]cuda:0" = torch.ops.aten.permute.default(view_93, [2, 0, 3, 1, 4]);  view_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        unbind_9 = torch.ops.aten.unbind.int(permute_56);  permute_56 = None
        getitem_146: "bf16[128, 12, 1370, 64][3156480, 64, 2304, 1]cuda:0" = unbind_9[0]
        getitem_147: "bf16[128, 12, 1370, 64][3156480, 64, 2304, 1]cuda:0" = unbind_9[1]
        getitem_148: "bf16[128, 12, 1370, 64][3156480, 64, 2304, 1]cuda:0" = unbind_9[2];  unbind_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:120 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention_9 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(getitem_146, getitem_147, getitem_148, None, True)
        getitem_149: "bf16[128, 12, 1370, 64][1052160, 64, 768, 1]cuda:0" = _scaled_dot_product_cudnn_attention_9[0]
        getitem_150: "f32[128, 12, 1370, 1][16440, 1370, 1, 1]cuda:0" = _scaled_dot_product_cudnn_attention_9[1]
        getitem_155: "i64[][]cuda:0" = _scaled_dot_product_cudnn_attention_9[6]
        getitem_156: "i64[][]cuda:0" = _scaled_dot_product_cudnn_attention_9[7];  _scaled_dot_product_cudnn_attention_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_57: "bf16[128, 1370, 12, 64][1052160, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_149, [0, 2, 1, 3])
        view_94: "bf16[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_57, [128, 1370, 768]);  permute_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        convert_element_type_225: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_137, torch.bfloat16);  primals_137 = None
        convert_element_type_226: "bf16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_136, torch.bfloat16);  primals_136 = None
        view_95: "bf16[175360, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_94, [175360, 768]);  view_94 = None
        permute_58: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_226, [1, 0]);  convert_element_type_226 = None
        addmm_37: "bf16[175360, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_225, view_95, permute_58);  convert_element_type_225 = view_95 = None
        view_96: "bf16[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_37, [128, 1370, 768])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/layer_scale.py:27 in forward, code: return x.mul_(self.gamma) if self.inplace else x * self.gamma
        mul_83: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_96, primals_138);  view_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        add_66: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_63, mul_83);  add_63 = mul_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_19 = torch.ops.aten.var_mean.correction(add_66, [2], correction = 0, keepdim = True)
        getitem_158: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = var_mean_19[0]
        getitem_159: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = var_mean_19[1];  var_mean_19 = None
        add_67: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_158, 1e-06);  getitem_158 = None
        rsqrt_19: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_67);  add_67 = None
        sub_19: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_66, getitem_159);  getitem_159 = None
        mul_84: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_19);  sub_19 = None
        mul_85: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_84, primals_139)
        add_68: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_85, primals_140);  mul_85 = primals_140 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        convert_element_type_230: "bf16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_142, torch.bfloat16);  primals_142 = None
        convert_element_type_231: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_141, torch.bfloat16);  primals_141 = None
        convert_element_type_232: "bf16[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_68, torch.bfloat16);  add_68 = None
        view_97: "bf16[175360, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_232, [175360, 768]);  convert_element_type_232 = None
        permute_59: "bf16[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_231, [1, 0]);  convert_element_type_231 = None
        addmm_38: "bf16[175360, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_230, view_97, permute_59);  convert_element_type_230 = None
        view_98: "bf16[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_38, [128, 1370, 3072])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_236: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_98, torch.float32);  view_98 = None
        mul_86: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_236, 0.5)
        mul_87: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_236, 0.7071067811865476);  convert_element_type_236 = None
        erf_9: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_87);  mul_87 = None
        add_69: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
        mul_88: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_86, add_69);  mul_86 = add_69 = None
        convert_element_type_237: "bf16[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_88, torch.bfloat16);  mul_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        convert_element_type_238: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_144, torch.bfloat16);  primals_144 = None
        convert_element_type_239: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_143, torch.bfloat16);  primals_143 = None
        view_99: "bf16[175360, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_237, [175360, 3072]);  convert_element_type_237 = None
        permute_60: "bf16[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(convert_element_type_239, [1, 0]);  convert_element_type_239 = None
        addmm_39: "bf16[175360, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_238, view_99, permute_60);  convert_element_type_238 = None
        view_100: "bf16[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_39, [128, 1370, 768])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/layer_scale.py:27 in forward, code: return x.mul_(self.gamma) if self.inplace else x * self.gamma
        mul_89: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_100, primals_145);  view_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        add_70: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_66, mul_89);  add_66 = mul_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_20 = torch.ops.aten.var_mean.correction(add_70, [2], correction = 0, keepdim = True)
        getitem_160: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = var_mean_20[0]
        getitem_161: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = var_mean_20[1];  var_mean_20 = None
        add_71: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_160, 1e-06);  getitem_160 = None
        rsqrt_20: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_71);  add_71 = None
        sub_20: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_70, getitem_161);  getitem_161 = None
        mul_90: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_20);  sub_20 = None
        mul_91: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_90, primals_146)
        add_72: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_91, primals_147);  mul_91 = primals_147 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        convert_element_type_243: "bf16[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_149, torch.bfloat16);  primals_149 = None
        convert_element_type_244: "bf16[2304, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_148, torch.bfloat16);  primals_148 = None
        convert_element_type_245: "bf16[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_72, torch.bfloat16);  add_72 = None
        view_101: "bf16[175360, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_245, [175360, 768]);  convert_element_type_245 = None
        permute_61: "bf16[768, 2304][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_244, [1, 0]);  convert_element_type_244 = None
        addmm_40: "bf16[175360, 2304][2304, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_243, view_101, permute_61);  convert_element_type_243 = None
        view_102: "bf16[128, 1370, 2304][3156480, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_40, [128, 1370, 2304]);  addmm_40 = None
        view_103: "bf16[128, 1370, 3, 12, 64][3156480, 2304, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_102, [128, 1370, 3, 12, 64]);  view_102 = None
        permute_62: "bf16[3, 128, 12, 1370, 64][768, 3156480, 64, 2304, 1]cuda:0" = torch.ops.aten.permute.default(view_103, [2, 0, 3, 1, 4]);  view_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        unbind_10 = torch.ops.aten.unbind.int(permute_62);  permute_62 = None
        getitem_162: "bf16[128, 12, 1370, 64][3156480, 64, 2304, 1]cuda:0" = unbind_10[0]
        getitem_163: "bf16[128, 12, 1370, 64][3156480, 64, 2304, 1]cuda:0" = unbind_10[1]
        getitem_164: "bf16[128, 12, 1370, 64][3156480, 64, 2304, 1]cuda:0" = unbind_10[2];  unbind_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:120 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention_10 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(getitem_162, getitem_163, getitem_164, None, True)
        getitem_165: "bf16[128, 12, 1370, 64][1052160, 64, 768, 1]cuda:0" = _scaled_dot_product_cudnn_attention_10[0]
        getitem_166: "f32[128, 12, 1370, 1][16440, 1370, 1, 1]cuda:0" = _scaled_dot_product_cudnn_attention_10[1]
        getitem_171: "i64[][]cuda:0" = _scaled_dot_product_cudnn_attention_10[6]
        getitem_172: "i64[][]cuda:0" = _scaled_dot_product_cudnn_attention_10[7];  _scaled_dot_product_cudnn_attention_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_63: "bf16[128, 1370, 12, 64][1052160, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_165, [0, 2, 1, 3])
        view_104: "bf16[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_63, [128, 1370, 768]);  permute_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        convert_element_type_249: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_151, torch.bfloat16);  primals_151 = None
        convert_element_type_250: "bf16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_150, torch.bfloat16);  primals_150 = None
        view_105: "bf16[175360, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_104, [175360, 768]);  view_104 = None
        permute_64: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_250, [1, 0]);  convert_element_type_250 = None
        addmm_41: "bf16[175360, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_249, view_105, permute_64);  convert_element_type_249 = view_105 = None
        view_106: "bf16[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_41, [128, 1370, 768])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/layer_scale.py:27 in forward, code: return x.mul_(self.gamma) if self.inplace else x * self.gamma
        mul_92: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_106, primals_152);  view_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        add_73: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_70, mul_92);  add_70 = mul_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_21 = torch.ops.aten.var_mean.correction(add_73, [2], correction = 0, keepdim = True)
        getitem_174: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = var_mean_21[0]
        getitem_175: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = var_mean_21[1];  var_mean_21 = None
        add_74: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_174, 1e-06);  getitem_174 = None
        rsqrt_21: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_74);  add_74 = None
        sub_21: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_73, getitem_175);  getitem_175 = None
        mul_93: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_21);  sub_21 = None
        mul_94: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_93, primals_153)
        add_75: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_94, primals_154);  mul_94 = primals_154 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        convert_element_type_254: "bf16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_156, torch.bfloat16);  primals_156 = None
        convert_element_type_255: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_155, torch.bfloat16);  primals_155 = None
        convert_element_type_256: "bf16[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_75, torch.bfloat16);  add_75 = None
        view_107: "bf16[175360, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_256, [175360, 768]);  convert_element_type_256 = None
        permute_65: "bf16[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_255, [1, 0]);  convert_element_type_255 = None
        addmm_42: "bf16[175360, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_254, view_107, permute_65);  convert_element_type_254 = None
        view_108: "bf16[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_42, [128, 1370, 3072])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_260: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_108, torch.float32);  view_108 = None
        mul_95: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_260, 0.5)
        mul_96: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_260, 0.7071067811865476);  convert_element_type_260 = None
        erf_10: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_96);  mul_96 = None
        add_76: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
        mul_97: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_95, add_76);  mul_95 = add_76 = None
        convert_element_type_261: "bf16[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_97, torch.bfloat16);  mul_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        convert_element_type_262: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_158, torch.bfloat16);  primals_158 = None
        convert_element_type_263: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_157, torch.bfloat16);  primals_157 = None
        view_109: "bf16[175360, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_261, [175360, 3072]);  convert_element_type_261 = None
        permute_66: "bf16[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(convert_element_type_263, [1, 0]);  convert_element_type_263 = None
        addmm_43: "bf16[175360, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_262, view_109, permute_66);  convert_element_type_262 = None
        view_110: "bf16[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_43, [128, 1370, 768])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/layer_scale.py:27 in forward, code: return x.mul_(self.gamma) if self.inplace else x * self.gamma
        mul_98: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_110, primals_159);  view_110 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        add_77: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_73, mul_98);  add_73 = mul_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_22 = torch.ops.aten.var_mean.correction(add_77, [2], correction = 0, keepdim = True)
        getitem_176: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = var_mean_22[0]
        getitem_177: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = var_mean_22[1];  var_mean_22 = None
        add_78: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_176, 1e-06);  getitem_176 = None
        rsqrt_22: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_78);  add_78 = None
        sub_22: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_77, getitem_177);  getitem_177 = None
        mul_99: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_22);  sub_22 = None
        mul_100: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_99, primals_160)
        add_79: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_100, primals_161);  mul_100 = primals_161 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        convert_element_type_267: "bf16[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_163, torch.bfloat16);  primals_163 = None
        convert_element_type_268: "bf16[2304, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_162, torch.bfloat16);  primals_162 = None
        convert_element_type_269: "bf16[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_79, torch.bfloat16);  add_79 = None
        view_111: "bf16[175360, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_269, [175360, 768]);  convert_element_type_269 = None
        permute_67: "bf16[768, 2304][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_268, [1, 0]);  convert_element_type_268 = None
        addmm_44: "bf16[175360, 2304][2304, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_267, view_111, permute_67);  convert_element_type_267 = None
        view_112: "bf16[128, 1370, 2304][3156480, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_44, [128, 1370, 2304]);  addmm_44 = None
        view_113: "bf16[128, 1370, 3, 12, 64][3156480, 2304, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_112, [128, 1370, 3, 12, 64]);  view_112 = None
        permute_68: "bf16[3, 128, 12, 1370, 64][768, 3156480, 64, 2304, 1]cuda:0" = torch.ops.aten.permute.default(view_113, [2, 0, 3, 1, 4]);  view_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        unbind_11 = torch.ops.aten.unbind.int(permute_68);  permute_68 = None
        getitem_178: "bf16[128, 12, 1370, 64][3156480, 64, 2304, 1]cuda:0" = unbind_11[0]
        getitem_179: "bf16[128, 12, 1370, 64][3156480, 64, 2304, 1]cuda:0" = unbind_11[1]
        getitem_180: "bf16[128, 12, 1370, 64][3156480, 64, 2304, 1]cuda:0" = unbind_11[2];  unbind_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:120 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention_11 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(getitem_178, getitem_179, getitem_180, None, True)
        getitem_181: "bf16[128, 12, 1370, 64][1052160, 64, 768, 1]cuda:0" = _scaled_dot_product_cudnn_attention_11[0]
        getitem_182: "f32[128, 12, 1370, 1][16440, 1370, 1, 1]cuda:0" = _scaled_dot_product_cudnn_attention_11[1]
        getitem_187: "i64[][]cuda:0" = _scaled_dot_product_cudnn_attention_11[6]
        getitem_188: "i64[][]cuda:0" = _scaled_dot_product_cudnn_attention_11[7];  _scaled_dot_product_cudnn_attention_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_69: "bf16[128, 1370, 12, 64][1052160, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_181, [0, 2, 1, 3])
        view_114: "bf16[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_69, [128, 1370, 768]);  permute_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        convert_element_type_273: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_165, torch.bfloat16);  primals_165 = None
        convert_element_type_274: "bf16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_164, torch.bfloat16);  primals_164 = None
        view_115: "bf16[175360, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_114, [175360, 768]);  view_114 = None
        permute_70: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_274, [1, 0]);  convert_element_type_274 = None
        addmm_45: "bf16[175360, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_273, view_115, permute_70);  convert_element_type_273 = view_115 = None
        view_116: "bf16[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_45, [128, 1370, 768])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/layer_scale.py:27 in forward, code: return x.mul_(self.gamma) if self.inplace else x * self.gamma
        mul_101: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_116, primals_166);  view_116 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        add_80: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_77, mul_101);  add_77 = mul_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_23 = torch.ops.aten.var_mean.correction(add_80, [2], correction = 0, keepdim = True)
        getitem_190: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = var_mean_23[0]
        getitem_191: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = var_mean_23[1];  var_mean_23 = None
        add_81: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_190, 1e-06);  getitem_190 = None
        rsqrt_23: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_81);  add_81 = None
        sub_23: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_80, getitem_191);  getitem_191 = None
        mul_102: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_23);  sub_23 = None
        mul_103: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_102, primals_167)
        add_82: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_103, primals_168);  mul_103 = primals_168 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        convert_element_type_278: "bf16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_170, torch.bfloat16);  primals_170 = None
        convert_element_type_279: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_169, torch.bfloat16);  primals_169 = None
        convert_element_type_280: "bf16[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_82, torch.bfloat16);  add_82 = None
        view_117: "bf16[175360, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_280, [175360, 768]);  convert_element_type_280 = None
        permute_71: "bf16[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_279, [1, 0]);  convert_element_type_279 = None
        addmm_46: "bf16[175360, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_278, view_117, permute_71);  convert_element_type_278 = None
        view_118: "bf16[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_46, [128, 1370, 3072])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_284: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_118, torch.float32);  view_118 = None
        mul_104: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_284, 0.5)
        mul_105: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_284, 0.7071067811865476);  convert_element_type_284 = None
        erf_11: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_105);  mul_105 = None
        add_83: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
        mul_106: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_104, add_83);  mul_104 = add_83 = None
        convert_element_type_285: "bf16[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_106, torch.bfloat16);  mul_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        convert_element_type_286: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_172, torch.bfloat16);  primals_172 = None
        convert_element_type_287: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_171, torch.bfloat16);  primals_171 = None
        view_119: "bf16[175360, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_285, [175360, 3072]);  convert_element_type_285 = None
        permute_72: "bf16[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(convert_element_type_287, [1, 0]);  convert_element_type_287 = None
        addmm_47: "bf16[175360, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_286, view_119, permute_72);  convert_element_type_286 = None
        view_120: "bf16[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_47, [128, 1370, 768])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/layer_scale.py:27 in forward, code: return x.mul_(self.gamma) if self.inplace else x * self.gamma
        mul_107: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_120, primals_173);  view_120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        add_84: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_80, mul_107);  add_80 = mul_107 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_24 = torch.ops.aten.var_mean.correction(add_84, [2], correction = 0, keepdim = True)
        getitem_192: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = var_mean_24[0]
        getitem_193: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = var_mean_24[1];  var_mean_24 = None
        add_85: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_192, 1e-06);  getitem_192 = None
        rsqrt_24: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_85);  add_85 = None
        sub_24: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_84, getitem_193);  add_84 = getitem_193 = None
        mul_108: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_24);  sub_24 = None
        mul_109: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_108, primals_174)
        add_86: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_109, primals_175);  mul_109 = primals_175 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:696 in global_pool_nlc, code: x = x[:, 0]  # class token
        select: "f32[128, 768][1052160, 1]cuda:0" = torch.ops.aten.select.int(add_86, 1, 0);  add_86 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:1291 in forward_head, code: x = self.head_drop(x)
        clone_37: "f32[128, 768][768, 1]cuda:0" = torch.ops.aten.clone.default(select);  select = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        div: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_24, 768);  rsqrt_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        permute_73: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.permute.default(permute_72, [1, 0]);  permute_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        permute_77: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_71, [1, 0]);  permute_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        div_1: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_23, 768);  rsqrt_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        permute_81: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_70, [1, 0]);  permute_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        permute_87: "bf16[2304, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_67, [1, 0]);  permute_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        div_2: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_22, 768);  rsqrt_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        permute_91: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.permute.default(permute_66, [1, 0]);  permute_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        permute_95: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_65, [1, 0]);  permute_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        div_3: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_21, 768);  rsqrt_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        permute_99: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_64, [1, 0]);  permute_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        permute_105: "bf16[2304, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_61, [1, 0]);  permute_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        div_4: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_20, 768);  rsqrt_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        permute_109: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.permute.default(permute_60, [1, 0]);  permute_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        permute_113: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_59, [1, 0]);  permute_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        div_5: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_19, 768);  rsqrt_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        permute_117: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_58, [1, 0]);  permute_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        permute_123: "bf16[2304, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_55, [1, 0]);  permute_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        div_6: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_18, 768);  rsqrt_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        permute_127: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.permute.default(permute_54, [1, 0]);  permute_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        permute_131: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_53, [1, 0]);  permute_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        div_7: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_17, 768);  rsqrt_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        permute_135: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_52, [1, 0]);  permute_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        permute_141: "bf16[2304, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_49, [1, 0]);  permute_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        div_8: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_16, 768);  rsqrt_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        permute_145: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.permute.default(permute_48, [1, 0]);  permute_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        permute_149: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_47, [1, 0]);  permute_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        div_9: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_15, 768);  rsqrt_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        permute_153: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_46, [1, 0]);  permute_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        permute_159: "bf16[2304, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_43, [1, 0]);  permute_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        div_10: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_14, 768);  rsqrt_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        permute_163: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.permute.default(permute_42, [1, 0]);  permute_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        permute_167: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_41, [1, 0]);  permute_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        div_11: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_13, 768);  rsqrt_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        permute_171: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_40, [1, 0]);  permute_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        permute_177: "bf16[2304, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_37, [1, 0]);  permute_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        div_12: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_12, 768);  rsqrt_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        permute_181: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.permute.default(permute_36, [1, 0]);  permute_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        permute_185: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_35, [1, 0]);  permute_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        div_13: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_11, 768);  rsqrt_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        permute_189: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_34, [1, 0]);  permute_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        permute_195: "bf16[2304, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_31, [1, 0]);  permute_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        div_14: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_10, 768);  rsqrt_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        permute_199: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.permute.default(permute_30, [1, 0]);  permute_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        permute_203: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_29, [1, 0]);  permute_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        div_15: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_9, 768);  rsqrt_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        permute_207: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_28, [1, 0]);  permute_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        permute_213: "bf16[2304, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_25, [1, 0]);  permute_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        div_16: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_8, 768);  rsqrt_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        permute_217: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.permute.default(permute_24, [1, 0]);  permute_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        permute_221: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_23, [1, 0]);  permute_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        div_17: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_7, 768);  rsqrt_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        permute_225: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_22, [1, 0]);  permute_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        permute_231: "bf16[2304, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_19, [1, 0]);  permute_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        div_18: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_6, 768);  rsqrt_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        permute_235: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.permute.default(permute_18, [1, 0]);  permute_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        permute_239: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_17, [1, 0]);  permute_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        div_19: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_5, 768);  rsqrt_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        permute_243: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_16, [1, 0]);  permute_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        permute_249: "bf16[2304, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_13, [1, 0]);  permute_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        div_20: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_4, 768);  rsqrt_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        permute_253: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.permute.default(permute_12, [1, 0]);  permute_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        permute_257: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_11, [1, 0]);  permute_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        div_21: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_3, 768);  rsqrt_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        permute_261: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_10, [1, 0]);  permute_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        permute_267: "bf16[2304, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_7, [1, 0]);  permute_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        div_22: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_2, 768);  rsqrt_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        permute_271: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.permute.default(permute_6, [1, 0]);  permute_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        permute_275: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_5, [1, 0]);  permute_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        div_23: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_1, 768);  rsqrt_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        permute_279: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_4, [1, 0]);  permute_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        permute_285: "bf16[2304, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_1, [1, 0]);  permute_1 = None
        return (clone_37, primals_5, primals_6, primals_12, primals_13, primals_19, primals_20, primals_26, primals_27, primals_33, primals_34, primals_40, primals_41, primals_47, primals_48, primals_54, primals_55, primals_61, primals_62, primals_68, primals_69, primals_75, primals_76, primals_82, primals_83, primals_89, primals_90, primals_96, primals_97, primals_103, primals_104, primals_110, primals_111, primals_117, primals_118, primals_124, primals_125, primals_131, primals_132, primals_138, primals_139, primals_145, primals_146, primals_152, primals_153, primals_159, primals_160, primals_166, primals_167, primals_173, primals_174, convert_element_type_1, convert_element_type_2, cat, getitem_1, rsqrt, view_1, getitem_2, getitem_3, getitem_4, getitem_5, getitem_6, getitem_11, getitem_12, addmm_1, mul_3, view_7, addmm_2, view_9, addmm_3, mul_9, view_11, getitem_18, getitem_19, getitem_20, getitem_21, getitem_22, getitem_27, getitem_28, addmm_5, mul_12, view_17, addmm_6, view_19, addmm_7, mul_18, view_21, getitem_34, getitem_35, getitem_36, getitem_37, getitem_38, getitem_43, getitem_44, addmm_9, mul_21, view_27, addmm_10, view_29, addmm_11, mul_27, view_31, getitem_50, getitem_51, getitem_52, getitem_53, getitem_54, getitem_59, getitem_60, addmm_13, mul_30, view_37, addmm_14, view_39, addmm_15, mul_36, view_41, getitem_66, getitem_67, getitem_68, getitem_69, getitem_70, getitem_75, getitem_76, addmm_17, mul_39, view_47, addmm_18, view_49, addmm_19, mul_45, view_51, getitem_82, getitem_83, getitem_84, getitem_85, getitem_86, getitem_91, getitem_92, addmm_21, mul_48, view_57, addmm_22, view_59, addmm_23, mul_54, view_61, getitem_98, getitem_99, getitem_100, getitem_101, getitem_102, getitem_107, getitem_108, addmm_25, mul_57, view_67, addmm_26, view_69, addmm_27, mul_63, view_71, getitem_114, getitem_115, getitem_116, getitem_117, getitem_118, getitem_123, getitem_124, addmm_29, mul_66, view_77, addmm_30, view_79, addmm_31, mul_72, view_81, getitem_130, getitem_131, getitem_132, getitem_133, getitem_134, getitem_139, getitem_140, addmm_33, mul_75, view_87, addmm_34, view_89, addmm_35, mul_81, view_91, getitem_146, getitem_147, getitem_148, getitem_149, getitem_150, getitem_155, getitem_156, addmm_37, mul_84, view_97, addmm_38, view_99, addmm_39, mul_90, view_101, getitem_162, getitem_163, getitem_164, getitem_165, getitem_166, getitem_171, getitem_172, addmm_41, mul_93, view_107, addmm_42, view_109, addmm_43, mul_99, view_111, getitem_178, getitem_179, getitem_180, getitem_181, getitem_182, getitem_187, getitem_188, addmm_45, mul_102, view_117, addmm_46, view_119, addmm_47, mul_108, div, permute_73, permute_77, div_1, permute_81, permute_87, div_2, permute_91, permute_95, div_3, permute_99, permute_105, div_4, permute_109, permute_113, div_5, permute_117, permute_123, div_6, permute_127, permute_131, div_7, permute_135, permute_141, div_8, permute_145, permute_149, div_9, permute_153, permute_159, div_10, permute_163, permute_167, div_11, permute_171, permute_177, div_12, permute_181, permute_185, div_13, permute_189, permute_195, div_14, permute_199, permute_203, div_15, permute_207, permute_213, div_16, permute_217, permute_221, div_17, permute_225, permute_231, div_18, permute_235, permute_239, div_19, permute_243, permute_249, div_20, permute_253, permute_257, div_21, permute_261, permute_267, div_22, permute_271, permute_275, div_23, permute_279, permute_285)
