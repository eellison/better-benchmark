import torch
from torch import device
from math import inf, nan

class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[128, 3, 224, 224][150528, 1, 672, 3]cuda:0", primals_2: "f32[768, 3, 16, 16][768, 1, 48, 3]cuda:0", primals_3: "f32[768][1]cuda:0", primals_4: "f32[1, 198, 768][152064, 768, 1]cuda:0", primals_5: "f32[1, 1, 768][768, 768, 1]cuda:0", primals_6: "f32[1, 1, 768][768, 768, 1]cuda:0", primals_7: "f32[768][1]cuda:0", primals_8: "f32[768][1]cuda:0", primals_9: "f32[2304, 768][768, 1]cuda:0", primals_10: "f32[2304][1]cuda:0", primals_11: "f32[768, 768][768, 1]cuda:0", primals_12: "f32[768][1]cuda:0", primals_13: "f32[768][1]cuda:0", primals_14: "f32[768][1]cuda:0", primals_15: "f32[3072, 768][768, 1]cuda:0", primals_16: "f32[3072][1]cuda:0", primals_17: "f32[768, 3072][3072, 1]cuda:0", primals_18: "f32[768][1]cuda:0", primals_19: "f32[768][1]cuda:0", primals_20: "f32[768][1]cuda:0", primals_21: "f32[2304, 768][768, 1]cuda:0", primals_22: "f32[2304][1]cuda:0", primals_23: "f32[768, 768][768, 1]cuda:0", primals_24: "f32[768][1]cuda:0", primals_25: "f32[768][1]cuda:0", primals_26: "f32[768][1]cuda:0", primals_27: "f32[3072, 768][768, 1]cuda:0", primals_28: "f32[3072][1]cuda:0", primals_29: "f32[768, 3072][3072, 1]cuda:0", primals_30: "f32[768][1]cuda:0", primals_31: "f32[768][1]cuda:0", primals_32: "f32[768][1]cuda:0", primals_33: "f32[2304, 768][768, 1]cuda:0", primals_34: "f32[2304][1]cuda:0", primals_35: "f32[768, 768][768, 1]cuda:0", primals_36: "f32[768][1]cuda:0", primals_37: "f32[768][1]cuda:0", primals_38: "f32[768][1]cuda:0", primals_39: "f32[3072, 768][768, 1]cuda:0", primals_40: "f32[3072][1]cuda:0", primals_41: "f32[768, 3072][3072, 1]cuda:0", primals_42: "f32[768][1]cuda:0", primals_43: "f32[768][1]cuda:0", primals_44: "f32[768][1]cuda:0", primals_45: "f32[2304, 768][768, 1]cuda:0", primals_46: "f32[2304][1]cuda:0", primals_47: "f32[768, 768][768, 1]cuda:0", primals_48: "f32[768][1]cuda:0", primals_49: "f32[768][1]cuda:0", primals_50: "f32[768][1]cuda:0", primals_51: "f32[3072, 768][768, 1]cuda:0", primals_52: "f32[3072][1]cuda:0", primals_53: "f32[768, 3072][3072, 1]cuda:0", primals_54: "f32[768][1]cuda:0", primals_55: "f32[768][1]cuda:0", primals_56: "f32[768][1]cuda:0", primals_57: "f32[2304, 768][768, 1]cuda:0", primals_58: "f32[2304][1]cuda:0", primals_59: "f32[768, 768][768, 1]cuda:0", primals_60: "f32[768][1]cuda:0", primals_61: "f32[768][1]cuda:0", primals_62: "f32[768][1]cuda:0", primals_63: "f32[3072, 768][768, 1]cuda:0", primals_64: "f32[3072][1]cuda:0", primals_65: "f32[768, 3072][3072, 1]cuda:0", primals_66: "f32[768][1]cuda:0", primals_67: "f32[768][1]cuda:0", primals_68: "f32[768][1]cuda:0", primals_69: "f32[2304, 768][768, 1]cuda:0", primals_70: "f32[2304][1]cuda:0", primals_71: "f32[768, 768][768, 1]cuda:0", primals_72: "f32[768][1]cuda:0", primals_73: "f32[768][1]cuda:0", primals_74: "f32[768][1]cuda:0", primals_75: "f32[3072, 768][768, 1]cuda:0", primals_76: "f32[3072][1]cuda:0", primals_77: "f32[768, 3072][3072, 1]cuda:0", primals_78: "f32[768][1]cuda:0", primals_79: "f32[768][1]cuda:0", primals_80: "f32[768][1]cuda:0", primals_81: "f32[2304, 768][768, 1]cuda:0", primals_82: "f32[2304][1]cuda:0", primals_83: "f32[768, 768][768, 1]cuda:0", primals_84: "f32[768][1]cuda:0", primals_85: "f32[768][1]cuda:0", primals_86: "f32[768][1]cuda:0", primals_87: "f32[3072, 768][768, 1]cuda:0", primals_88: "f32[3072][1]cuda:0", primals_89: "f32[768, 3072][3072, 1]cuda:0", primals_90: "f32[768][1]cuda:0", primals_91: "f32[768][1]cuda:0", primals_92: "f32[768][1]cuda:0", primals_93: "f32[2304, 768][768, 1]cuda:0", primals_94: "f32[2304][1]cuda:0", primals_95: "f32[768, 768][768, 1]cuda:0", primals_96: "f32[768][1]cuda:0", primals_97: "f32[768][1]cuda:0", primals_98: "f32[768][1]cuda:0", primals_99: "f32[3072, 768][768, 1]cuda:0", primals_100: "f32[3072][1]cuda:0", primals_101: "f32[768, 3072][3072, 1]cuda:0", primals_102: "f32[768][1]cuda:0", primals_103: "f32[768][1]cuda:0", primals_104: "f32[768][1]cuda:0", primals_105: "f32[2304, 768][768, 1]cuda:0", primals_106: "f32[2304][1]cuda:0", primals_107: "f32[768, 768][768, 1]cuda:0", primals_108: "f32[768][1]cuda:0", primals_109: "f32[768][1]cuda:0", primals_110: "f32[768][1]cuda:0", primals_111: "f32[3072, 768][768, 1]cuda:0", primals_112: "f32[3072][1]cuda:0", primals_113: "f32[768, 3072][3072, 1]cuda:0", primals_114: "f32[768][1]cuda:0", primals_115: "f32[768][1]cuda:0", primals_116: "f32[768][1]cuda:0", primals_117: "f32[2304, 768][768, 1]cuda:0", primals_118: "f32[2304][1]cuda:0", primals_119: "f32[768, 768][768, 1]cuda:0", primals_120: "f32[768][1]cuda:0", primals_121: "f32[768][1]cuda:0", primals_122: "f32[768][1]cuda:0", primals_123: "f32[3072, 768][768, 1]cuda:0", primals_124: "f32[3072][1]cuda:0", primals_125: "f32[768, 3072][3072, 1]cuda:0", primals_126: "f32[768][1]cuda:0", primals_127: "f32[768][1]cuda:0", primals_128: "f32[768][1]cuda:0", primals_129: "f32[2304, 768][768, 1]cuda:0", primals_130: "f32[2304][1]cuda:0", primals_131: "f32[768, 768][768, 1]cuda:0", primals_132: "f32[768][1]cuda:0", primals_133: "f32[768][1]cuda:0", primals_134: "f32[768][1]cuda:0", primals_135: "f32[3072, 768][768, 1]cuda:0", primals_136: "f32[3072][1]cuda:0", primals_137: "f32[768, 3072][3072, 1]cuda:0", primals_138: "f32[768][1]cuda:0", primals_139: "f32[768][1]cuda:0", primals_140: "f32[768][1]cuda:0", primals_141: "f32[2304, 768][768, 1]cuda:0", primals_142: "f32[2304][1]cuda:0", primals_143: "f32[768, 768][768, 1]cuda:0", primals_144: "f32[768][1]cuda:0", primals_145: "f32[768][1]cuda:0", primals_146: "f32[768][1]cuda:0", primals_147: "f32[3072, 768][768, 1]cuda:0", primals_148: "f32[3072][1]cuda:0", primals_149: "f32[768, 3072][3072, 1]cuda:0", primals_150: "f32[768][1]cuda:0", primals_151: "f32[768][1]cuda:0", primals_152: "f32[768][1]cuda:0", primals_153: "f32[1000, 768][768, 1]cuda:0", primals_154: "f32[1000][1]cuda:0", primals_155: "f32[1000, 768][768, 1]cuda:0", primals_156: "f32[1000][1]cuda:0"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/patch_embed.py:136 in forward, code: x = self.proj(x)
        convolution: "f32[128, 768, 14, 14][150528, 1, 10752, 768]cuda:0" = torch.ops.aten.convolution.default(primals_1, primals_2, primals_3, [16, 16], [0, 0], [1, 1], False, [0, 0], 1);  primals_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/patch_embed.py:138 in forward, code: x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
        view: "f32[128, 768, 196][150528, 1, 768]cuda:0" = torch.ops.aten.reshape.default(convolution, [128, 768, 196]);  convolution = None
        permute: "f32[128, 196, 768][150528, 768, 1]cuda:0" = torch.ops.aten.permute.default(view, [0, 2, 1]);  view = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/deit.py:106 in _pos_embed, code: self.cls_token.expand(x.shape[0], -1, -1),
        expand: "f32[128, 1, 768][0, 768, 1]cuda:0" = torch.ops.aten.expand.default(primals_5, [128, -1, -1]);  primals_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/deit.py:107 in _pos_embed, code: self.dist_token.expand(x.shape[0], -1, -1),
        expand_1: "f32[128, 1, 768][0, 768, 1]cuda:0" = torch.ops.aten.expand.default(primals_6, [128, -1, -1]);  primals_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/deit.py:105 in _pos_embed, code: x = torch.cat((
        cat: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.cat.default([expand, expand_1, permute], 1);  expand = expand_1 = permute = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/deit.py:110 in _pos_embed, code: x = x + pos_embed
        add: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(cat, primals_4)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean = torch.ops.aten.var_mean.correction(add, [2], correction = 0, keepdim = True)
        getitem: "f32[128, 198, 1][198, 1, 1]cuda:0" = var_mean[0]
        getitem_1: "f32[128, 198, 1][198, 1, 1]cuda:0" = var_mean[1];  var_mean = None
        add_1: "f32[128, 198, 1][198, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem, 1e-06);  getitem = None
        rsqrt: "f32[128, 198, 1][198, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
        sub: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add, getitem_1)
        mul: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        mul_1: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul, primals_7);  mul = None
        add_2: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_1, primals_8);  mul_1 = primals_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        view_1: "f32[25344, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(add_2, [25344, 768]);  add_2 = None
        permute_1: "f32[768, 2304][1, 768]cuda:0" = torch.ops.aten.permute.default(primals_9, [1, 0])
        addmm: "f32[25344, 2304][2304, 1]cuda:0" = torch.ops.aten.addmm.default(primals_10, view_1, permute_1);  primals_10 = permute_1 = None
        view_2: "f32[128, 198, 2304][456192, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(addmm, [128, 198, 2304]);  addmm = None
        view_3: "f32[128, 198, 3, 12, 64][456192, 2304, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_2, [128, 198, 3, 12, 64]);  view_2 = None
        permute_2: "f32[3, 128, 12, 198, 64][768, 456192, 64, 2304, 1]cuda:0" = torch.ops.aten.permute.default(view_3, [2, 0, 3, 1, 4]);  view_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        unbind = torch.ops.aten.unbind.int(permute_2);  permute_2 = None
        getitem_2: "f32[128, 12, 198, 64][456192, 64, 2304, 1]cuda:0" = unbind[0]
        getitem_3: "f32[128, 12, 198, 64][456192, 64, 2304, 1]cuda:0" = unbind[1]
        getitem_4: "f32[128, 12, 198, 64][456192, 64, 2304, 1]cuda:0" = unbind[2];  unbind = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:120 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_2, getitem_3, getitem_4, None, True)
        getitem_5: "f32[128, 12, 198, 64][152064, 64, 768, 1]cuda:0" = _scaled_dot_product_efficient_attention[0]
        getitem_6: "f32[128, 12, 224][2688, 224, 1]cuda:0" = _scaled_dot_product_efficient_attention[1]
        getitem_7: "i64[][]cuda:0" = _scaled_dot_product_efficient_attention[2]
        getitem_8: "i64[][]cuda:0" = _scaled_dot_product_efficient_attention[3];  _scaled_dot_product_efficient_attention = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_3: "f32[128, 198, 12, 64][152064, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_5, [0, 2, 1, 3])
        view_4: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_3, [128, 198, 768]);  permute_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_5: "f32[25344, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_4, [25344, 768]);  view_4 = None
        permute_4: "f32[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(primals_11, [1, 0])
        addmm_1: "f32[25344, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(primals_12, view_5, permute_4);  primals_12 = view_5 = permute_4 = None
        view_6: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_1, [128, 198, 768]);  addmm_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        add_3: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add, view_6);  add = view_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_1 = torch.ops.aten.var_mean.correction(add_3, [2], correction = 0, keepdim = True)
        getitem_9: "f32[128, 198, 1][198, 1, 1]cuda:0" = var_mean_1[0]
        getitem_10: "f32[128, 198, 1][198, 1, 1]cuda:0" = var_mean_1[1];  var_mean_1 = None
        add_4: "f32[128, 198, 1][198, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_9, 1e-06);  getitem_9 = None
        rsqrt_1: "f32[128, 198, 1][198, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_4);  add_4 = None
        sub_1: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_3, getitem_10);  getitem_10 = None
        mul_2: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = None
        mul_3: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2, primals_13)
        add_5: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_3, primals_14);  mul_3 = primals_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_7: "f32[25344, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(add_5, [25344, 768]);  add_5 = None
        permute_5: "f32[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(primals_15, [1, 0])
        addmm_2: "f32[25344, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(primals_16, view_7, permute_5);  primals_16 = permute_5 = None
        view_8: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_2, [128, 198, 3072])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        mul_4: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_8, 0.5)
        mul_5: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_8, 0.7071067811865476);  view_8 = None
        erf: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_5);  mul_5 = None
        add_6: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_6: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_4, add_6);  mul_4 = add_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_9: "f32[25344, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(mul_6, [25344, 3072]);  mul_6 = None
        permute_6: "f32[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(primals_17, [1, 0])
        addmm_3: "f32[25344, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(primals_18, view_9, permute_6);  primals_18 = permute_6 = None
        view_10: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_3, [128, 198, 768]);  addmm_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        add_7: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_3, view_10);  add_3 = view_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_2 = torch.ops.aten.var_mean.correction(add_7, [2], correction = 0, keepdim = True)
        getitem_11: "f32[128, 198, 1][198, 1, 1]cuda:0" = var_mean_2[0]
        getitem_12: "f32[128, 198, 1][198, 1, 1]cuda:0" = var_mean_2[1];  var_mean_2 = None
        add_8: "f32[128, 198, 1][198, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_11, 1e-06);  getitem_11 = None
        rsqrt_2: "f32[128, 198, 1][198, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_8);  add_8 = None
        sub_2: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_7, getitem_12);  getitem_12 = None
        mul_7: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_2);  sub_2 = None
        mul_8: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_7, primals_19)
        add_9: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_8, primals_20);  mul_8 = primals_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        view_11: "f32[25344, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(add_9, [25344, 768]);  add_9 = None
        permute_7: "f32[768, 2304][1, 768]cuda:0" = torch.ops.aten.permute.default(primals_21, [1, 0])
        addmm_4: "f32[25344, 2304][2304, 1]cuda:0" = torch.ops.aten.addmm.default(primals_22, view_11, permute_7);  primals_22 = permute_7 = None
        view_12: "f32[128, 198, 2304][456192, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_4, [128, 198, 2304]);  addmm_4 = None
        view_13: "f32[128, 198, 3, 12, 64][456192, 2304, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_12, [128, 198, 3, 12, 64]);  view_12 = None
        permute_8: "f32[3, 128, 12, 198, 64][768, 456192, 64, 2304, 1]cuda:0" = torch.ops.aten.permute.default(view_13, [2, 0, 3, 1, 4]);  view_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        unbind_1 = torch.ops.aten.unbind.int(permute_8);  permute_8 = None
        getitem_13: "f32[128, 12, 198, 64][456192, 64, 2304, 1]cuda:0" = unbind_1[0]
        getitem_14: "f32[128, 12, 198, 64][456192, 64, 2304, 1]cuda:0" = unbind_1[1]
        getitem_15: "f32[128, 12, 198, 64][456192, 64, 2304, 1]cuda:0" = unbind_1[2];  unbind_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:120 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_1 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_13, getitem_14, getitem_15, None, True)
        getitem_16: "f32[128, 12, 198, 64][152064, 64, 768, 1]cuda:0" = _scaled_dot_product_efficient_attention_1[0]
        getitem_17: "f32[128, 12, 224][2688, 224, 1]cuda:0" = _scaled_dot_product_efficient_attention_1[1]
        getitem_18: "i64[][]cuda:0" = _scaled_dot_product_efficient_attention_1[2]
        getitem_19: "i64[][]cuda:0" = _scaled_dot_product_efficient_attention_1[3];  _scaled_dot_product_efficient_attention_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_9: "f32[128, 198, 12, 64][152064, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_16, [0, 2, 1, 3])
        view_14: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_9, [128, 198, 768]);  permute_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_15: "f32[25344, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_14, [25344, 768]);  view_14 = None
        permute_10: "f32[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(primals_23, [1, 0])
        addmm_5: "f32[25344, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(primals_24, view_15, permute_10);  primals_24 = view_15 = permute_10 = None
        view_16: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_5, [128, 198, 768]);  addmm_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        add_10: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_7, view_16);  add_7 = view_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_3 = torch.ops.aten.var_mean.correction(add_10, [2], correction = 0, keepdim = True)
        getitem_20: "f32[128, 198, 1][198, 1, 1]cuda:0" = var_mean_3[0]
        getitem_21: "f32[128, 198, 1][198, 1, 1]cuda:0" = var_mean_3[1];  var_mean_3 = None
        add_11: "f32[128, 198, 1][198, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_20, 1e-06);  getitem_20 = None
        rsqrt_3: "f32[128, 198, 1][198, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_11);  add_11 = None
        sub_3: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_10, getitem_21);  getitem_21 = None
        mul_9: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_3);  sub_3 = None
        mul_10: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_9, primals_25)
        add_12: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_10, primals_26);  mul_10 = primals_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_17: "f32[25344, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(add_12, [25344, 768]);  add_12 = None
        permute_11: "f32[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(primals_27, [1, 0])
        addmm_6: "f32[25344, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(primals_28, view_17, permute_11);  primals_28 = permute_11 = None
        view_18: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_6, [128, 198, 3072])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        mul_11: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_18, 0.5)
        mul_12: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_18, 0.7071067811865476);  view_18 = None
        erf_1: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_12);  mul_12 = None
        add_13: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
        mul_13: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_11, add_13);  mul_11 = add_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_19: "f32[25344, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(mul_13, [25344, 3072]);  mul_13 = None
        permute_12: "f32[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(primals_29, [1, 0])
        addmm_7: "f32[25344, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(primals_30, view_19, permute_12);  primals_30 = permute_12 = None
        view_20: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_7, [128, 198, 768]);  addmm_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        add_14: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_10, view_20);  add_10 = view_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_4 = torch.ops.aten.var_mean.correction(add_14, [2], correction = 0, keepdim = True)
        getitem_22: "f32[128, 198, 1][198, 1, 1]cuda:0" = var_mean_4[0]
        getitem_23: "f32[128, 198, 1][198, 1, 1]cuda:0" = var_mean_4[1];  var_mean_4 = None
        add_15: "f32[128, 198, 1][198, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_22, 1e-06);  getitem_22 = None
        rsqrt_4: "f32[128, 198, 1][198, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_15);  add_15 = None
        sub_4: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_14, getitem_23);  getitem_23 = None
        mul_14: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_4);  sub_4 = None
        mul_15: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_14, primals_31)
        add_16: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_15, primals_32);  mul_15 = primals_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        view_21: "f32[25344, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(add_16, [25344, 768]);  add_16 = None
        permute_13: "f32[768, 2304][1, 768]cuda:0" = torch.ops.aten.permute.default(primals_33, [1, 0])
        addmm_8: "f32[25344, 2304][2304, 1]cuda:0" = torch.ops.aten.addmm.default(primals_34, view_21, permute_13);  primals_34 = permute_13 = None
        view_22: "f32[128, 198, 2304][456192, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_8, [128, 198, 2304]);  addmm_8 = None
        view_23: "f32[128, 198, 3, 12, 64][456192, 2304, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_22, [128, 198, 3, 12, 64]);  view_22 = None
        permute_14: "f32[3, 128, 12, 198, 64][768, 456192, 64, 2304, 1]cuda:0" = torch.ops.aten.permute.default(view_23, [2, 0, 3, 1, 4]);  view_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        unbind_2 = torch.ops.aten.unbind.int(permute_14);  permute_14 = None
        getitem_24: "f32[128, 12, 198, 64][456192, 64, 2304, 1]cuda:0" = unbind_2[0]
        getitem_25: "f32[128, 12, 198, 64][456192, 64, 2304, 1]cuda:0" = unbind_2[1]
        getitem_26: "f32[128, 12, 198, 64][456192, 64, 2304, 1]cuda:0" = unbind_2[2];  unbind_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:120 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_2 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_24, getitem_25, getitem_26, None, True)
        getitem_27: "f32[128, 12, 198, 64][152064, 64, 768, 1]cuda:0" = _scaled_dot_product_efficient_attention_2[0]
        getitem_28: "f32[128, 12, 224][2688, 224, 1]cuda:0" = _scaled_dot_product_efficient_attention_2[1]
        getitem_29: "i64[][]cuda:0" = _scaled_dot_product_efficient_attention_2[2]
        getitem_30: "i64[][]cuda:0" = _scaled_dot_product_efficient_attention_2[3];  _scaled_dot_product_efficient_attention_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_15: "f32[128, 198, 12, 64][152064, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_27, [0, 2, 1, 3])
        view_24: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_15, [128, 198, 768]);  permute_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_25: "f32[25344, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_24, [25344, 768]);  view_24 = None
        permute_16: "f32[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(primals_35, [1, 0])
        addmm_9: "f32[25344, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(primals_36, view_25, permute_16);  primals_36 = view_25 = permute_16 = None
        view_26: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_9, [128, 198, 768]);  addmm_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        add_17: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_14, view_26);  add_14 = view_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_5 = torch.ops.aten.var_mean.correction(add_17, [2], correction = 0, keepdim = True)
        getitem_31: "f32[128, 198, 1][198, 1, 1]cuda:0" = var_mean_5[0]
        getitem_32: "f32[128, 198, 1][198, 1, 1]cuda:0" = var_mean_5[1];  var_mean_5 = None
        add_18: "f32[128, 198, 1][198, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_31, 1e-06);  getitem_31 = None
        rsqrt_5: "f32[128, 198, 1][198, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_18);  add_18 = None
        sub_5: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_17, getitem_32);  getitem_32 = None
        mul_16: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_5);  sub_5 = None
        mul_17: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_16, primals_37)
        add_19: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_17, primals_38);  mul_17 = primals_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_27: "f32[25344, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(add_19, [25344, 768]);  add_19 = None
        permute_17: "f32[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(primals_39, [1, 0])
        addmm_10: "f32[25344, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(primals_40, view_27, permute_17);  primals_40 = permute_17 = None
        view_28: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_10, [128, 198, 3072])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        mul_18: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_28, 0.5)
        mul_19: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_28, 0.7071067811865476);  view_28 = None
        erf_2: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_19);  mul_19 = None
        add_20: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
        mul_20: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_18, add_20);  mul_18 = add_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_29: "f32[25344, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(mul_20, [25344, 3072]);  mul_20 = None
        permute_18: "f32[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(primals_41, [1, 0])
        addmm_11: "f32[25344, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(primals_42, view_29, permute_18);  primals_42 = permute_18 = None
        view_30: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_11, [128, 198, 768]);  addmm_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        add_21: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_17, view_30);  add_17 = view_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_6 = torch.ops.aten.var_mean.correction(add_21, [2], correction = 0, keepdim = True)
        getitem_33: "f32[128, 198, 1][198, 1, 1]cuda:0" = var_mean_6[0]
        getitem_34: "f32[128, 198, 1][198, 1, 1]cuda:0" = var_mean_6[1];  var_mean_6 = None
        add_22: "f32[128, 198, 1][198, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_33, 1e-06);  getitem_33 = None
        rsqrt_6: "f32[128, 198, 1][198, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_22);  add_22 = None
        sub_6: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_21, getitem_34);  getitem_34 = None
        mul_21: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_6);  sub_6 = None
        mul_22: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_21, primals_43)
        add_23: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_22, primals_44);  mul_22 = primals_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        view_31: "f32[25344, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(add_23, [25344, 768]);  add_23 = None
        permute_19: "f32[768, 2304][1, 768]cuda:0" = torch.ops.aten.permute.default(primals_45, [1, 0])
        addmm_12: "f32[25344, 2304][2304, 1]cuda:0" = torch.ops.aten.addmm.default(primals_46, view_31, permute_19);  primals_46 = permute_19 = None
        view_32: "f32[128, 198, 2304][456192, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_12, [128, 198, 2304]);  addmm_12 = None
        view_33: "f32[128, 198, 3, 12, 64][456192, 2304, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_32, [128, 198, 3, 12, 64]);  view_32 = None
        permute_20: "f32[3, 128, 12, 198, 64][768, 456192, 64, 2304, 1]cuda:0" = torch.ops.aten.permute.default(view_33, [2, 0, 3, 1, 4]);  view_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        unbind_3 = torch.ops.aten.unbind.int(permute_20);  permute_20 = None
        getitem_35: "f32[128, 12, 198, 64][456192, 64, 2304, 1]cuda:0" = unbind_3[0]
        getitem_36: "f32[128, 12, 198, 64][456192, 64, 2304, 1]cuda:0" = unbind_3[1]
        getitem_37: "f32[128, 12, 198, 64][456192, 64, 2304, 1]cuda:0" = unbind_3[2];  unbind_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:120 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_3 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_35, getitem_36, getitem_37, None, True)
        getitem_38: "f32[128, 12, 198, 64][152064, 64, 768, 1]cuda:0" = _scaled_dot_product_efficient_attention_3[0]
        getitem_39: "f32[128, 12, 224][2688, 224, 1]cuda:0" = _scaled_dot_product_efficient_attention_3[1]
        getitem_40: "i64[][]cuda:0" = _scaled_dot_product_efficient_attention_3[2]
        getitem_41: "i64[][]cuda:0" = _scaled_dot_product_efficient_attention_3[3];  _scaled_dot_product_efficient_attention_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_21: "f32[128, 198, 12, 64][152064, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_38, [0, 2, 1, 3])
        view_34: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_21, [128, 198, 768]);  permute_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_35: "f32[25344, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_34, [25344, 768]);  view_34 = None
        permute_22: "f32[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(primals_47, [1, 0])
        addmm_13: "f32[25344, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(primals_48, view_35, permute_22);  primals_48 = view_35 = permute_22 = None
        view_36: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_13, [128, 198, 768]);  addmm_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        add_24: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_21, view_36);  add_21 = view_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_7 = torch.ops.aten.var_mean.correction(add_24, [2], correction = 0, keepdim = True)
        getitem_42: "f32[128, 198, 1][198, 1, 1]cuda:0" = var_mean_7[0]
        getitem_43: "f32[128, 198, 1][198, 1, 1]cuda:0" = var_mean_7[1];  var_mean_7 = None
        add_25: "f32[128, 198, 1][198, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_42, 1e-06);  getitem_42 = None
        rsqrt_7: "f32[128, 198, 1][198, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_25);  add_25 = None
        sub_7: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_24, getitem_43);  getitem_43 = None
        mul_23: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_7);  sub_7 = None
        mul_24: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_23, primals_49)
        add_26: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_24, primals_50);  mul_24 = primals_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_37: "f32[25344, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(add_26, [25344, 768]);  add_26 = None
        permute_23: "f32[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(primals_51, [1, 0])
        addmm_14: "f32[25344, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(primals_52, view_37, permute_23);  primals_52 = permute_23 = None
        view_38: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_14, [128, 198, 3072])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        mul_25: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_38, 0.5)
        mul_26: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_38, 0.7071067811865476);  view_38 = None
        erf_3: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_26);  mul_26 = None
        add_27: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
        mul_27: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_25, add_27);  mul_25 = add_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_39: "f32[25344, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(mul_27, [25344, 3072]);  mul_27 = None
        permute_24: "f32[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(primals_53, [1, 0])
        addmm_15: "f32[25344, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(primals_54, view_39, permute_24);  primals_54 = permute_24 = None
        view_40: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_15, [128, 198, 768]);  addmm_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        add_28: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_24, view_40);  add_24 = view_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_8 = torch.ops.aten.var_mean.correction(add_28, [2], correction = 0, keepdim = True)
        getitem_44: "f32[128, 198, 1][198, 1, 1]cuda:0" = var_mean_8[0]
        getitem_45: "f32[128, 198, 1][198, 1, 1]cuda:0" = var_mean_8[1];  var_mean_8 = None
        add_29: "f32[128, 198, 1][198, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_44, 1e-06);  getitem_44 = None
        rsqrt_8: "f32[128, 198, 1][198, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_29);  add_29 = None
        sub_8: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_28, getitem_45);  getitem_45 = None
        mul_28: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_8);  sub_8 = None
        mul_29: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_28, primals_55)
        add_30: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_29, primals_56);  mul_29 = primals_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        view_41: "f32[25344, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(add_30, [25344, 768]);  add_30 = None
        permute_25: "f32[768, 2304][1, 768]cuda:0" = torch.ops.aten.permute.default(primals_57, [1, 0])
        addmm_16: "f32[25344, 2304][2304, 1]cuda:0" = torch.ops.aten.addmm.default(primals_58, view_41, permute_25);  primals_58 = permute_25 = None
        view_42: "f32[128, 198, 2304][456192, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_16, [128, 198, 2304]);  addmm_16 = None
        view_43: "f32[128, 198, 3, 12, 64][456192, 2304, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_42, [128, 198, 3, 12, 64]);  view_42 = None
        permute_26: "f32[3, 128, 12, 198, 64][768, 456192, 64, 2304, 1]cuda:0" = torch.ops.aten.permute.default(view_43, [2, 0, 3, 1, 4]);  view_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        unbind_4 = torch.ops.aten.unbind.int(permute_26);  permute_26 = None
        getitem_46: "f32[128, 12, 198, 64][456192, 64, 2304, 1]cuda:0" = unbind_4[0]
        getitem_47: "f32[128, 12, 198, 64][456192, 64, 2304, 1]cuda:0" = unbind_4[1]
        getitem_48: "f32[128, 12, 198, 64][456192, 64, 2304, 1]cuda:0" = unbind_4[2];  unbind_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:120 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_4 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_46, getitem_47, getitem_48, None, True)
        getitem_49: "f32[128, 12, 198, 64][152064, 64, 768, 1]cuda:0" = _scaled_dot_product_efficient_attention_4[0]
        getitem_50: "f32[128, 12, 224][2688, 224, 1]cuda:0" = _scaled_dot_product_efficient_attention_4[1]
        getitem_51: "i64[][]cuda:0" = _scaled_dot_product_efficient_attention_4[2]
        getitem_52: "i64[][]cuda:0" = _scaled_dot_product_efficient_attention_4[3];  _scaled_dot_product_efficient_attention_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_27: "f32[128, 198, 12, 64][152064, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_49, [0, 2, 1, 3])
        view_44: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_27, [128, 198, 768]);  permute_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_45: "f32[25344, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_44, [25344, 768]);  view_44 = None
        permute_28: "f32[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(primals_59, [1, 0])
        addmm_17: "f32[25344, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(primals_60, view_45, permute_28);  primals_60 = view_45 = permute_28 = None
        view_46: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_17, [128, 198, 768]);  addmm_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        add_31: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_28, view_46);  add_28 = view_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_9 = torch.ops.aten.var_mean.correction(add_31, [2], correction = 0, keepdim = True)
        getitem_53: "f32[128, 198, 1][198, 1, 1]cuda:0" = var_mean_9[0]
        getitem_54: "f32[128, 198, 1][198, 1, 1]cuda:0" = var_mean_9[1];  var_mean_9 = None
        add_32: "f32[128, 198, 1][198, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_53, 1e-06);  getitem_53 = None
        rsqrt_9: "f32[128, 198, 1][198, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_32);  add_32 = None
        sub_9: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_31, getitem_54);  getitem_54 = None
        mul_30: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_9);  sub_9 = None
        mul_31: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_30, primals_61)
        add_33: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_31, primals_62);  mul_31 = primals_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_47: "f32[25344, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(add_33, [25344, 768]);  add_33 = None
        permute_29: "f32[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(primals_63, [1, 0])
        addmm_18: "f32[25344, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(primals_64, view_47, permute_29);  primals_64 = permute_29 = None
        view_48: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_18, [128, 198, 3072])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        mul_32: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_48, 0.5)
        mul_33: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_48, 0.7071067811865476);  view_48 = None
        erf_4: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_33);  mul_33 = None
        add_34: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
        mul_34: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_32, add_34);  mul_32 = add_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_49: "f32[25344, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(mul_34, [25344, 3072]);  mul_34 = None
        permute_30: "f32[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(primals_65, [1, 0])
        addmm_19: "f32[25344, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(primals_66, view_49, permute_30);  primals_66 = permute_30 = None
        view_50: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_19, [128, 198, 768]);  addmm_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        add_35: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_31, view_50);  add_31 = view_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_10 = torch.ops.aten.var_mean.correction(add_35, [2], correction = 0, keepdim = True)
        getitem_55: "f32[128, 198, 1][198, 1, 1]cuda:0" = var_mean_10[0]
        getitem_56: "f32[128, 198, 1][198, 1, 1]cuda:0" = var_mean_10[1];  var_mean_10 = None
        add_36: "f32[128, 198, 1][198, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_55, 1e-06);  getitem_55 = None
        rsqrt_10: "f32[128, 198, 1][198, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_36);  add_36 = None
        sub_10: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_35, getitem_56);  getitem_56 = None
        mul_35: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_10);  sub_10 = None
        mul_36: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_35, primals_67)
        add_37: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_36, primals_68);  mul_36 = primals_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        view_51: "f32[25344, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(add_37, [25344, 768]);  add_37 = None
        permute_31: "f32[768, 2304][1, 768]cuda:0" = torch.ops.aten.permute.default(primals_69, [1, 0])
        addmm_20: "f32[25344, 2304][2304, 1]cuda:0" = torch.ops.aten.addmm.default(primals_70, view_51, permute_31);  primals_70 = permute_31 = None
        view_52: "f32[128, 198, 2304][456192, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_20, [128, 198, 2304]);  addmm_20 = None
        view_53: "f32[128, 198, 3, 12, 64][456192, 2304, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_52, [128, 198, 3, 12, 64]);  view_52 = None
        permute_32: "f32[3, 128, 12, 198, 64][768, 456192, 64, 2304, 1]cuda:0" = torch.ops.aten.permute.default(view_53, [2, 0, 3, 1, 4]);  view_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        unbind_5 = torch.ops.aten.unbind.int(permute_32);  permute_32 = None
        getitem_57: "f32[128, 12, 198, 64][456192, 64, 2304, 1]cuda:0" = unbind_5[0]
        getitem_58: "f32[128, 12, 198, 64][456192, 64, 2304, 1]cuda:0" = unbind_5[1]
        getitem_59: "f32[128, 12, 198, 64][456192, 64, 2304, 1]cuda:0" = unbind_5[2];  unbind_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:120 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_5 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_57, getitem_58, getitem_59, None, True)
        getitem_60: "f32[128, 12, 198, 64][152064, 64, 768, 1]cuda:0" = _scaled_dot_product_efficient_attention_5[0]
        getitem_61: "f32[128, 12, 224][2688, 224, 1]cuda:0" = _scaled_dot_product_efficient_attention_5[1]
        getitem_62: "i64[][]cuda:0" = _scaled_dot_product_efficient_attention_5[2]
        getitem_63: "i64[][]cuda:0" = _scaled_dot_product_efficient_attention_5[3];  _scaled_dot_product_efficient_attention_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_33: "f32[128, 198, 12, 64][152064, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_60, [0, 2, 1, 3])
        view_54: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_33, [128, 198, 768]);  permute_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_55: "f32[25344, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_54, [25344, 768]);  view_54 = None
        permute_34: "f32[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(primals_71, [1, 0])
        addmm_21: "f32[25344, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(primals_72, view_55, permute_34);  primals_72 = view_55 = permute_34 = None
        view_56: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_21, [128, 198, 768]);  addmm_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        add_38: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_35, view_56);  add_35 = view_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_11 = torch.ops.aten.var_mean.correction(add_38, [2], correction = 0, keepdim = True)
        getitem_64: "f32[128, 198, 1][198, 1, 1]cuda:0" = var_mean_11[0]
        getitem_65: "f32[128, 198, 1][198, 1, 1]cuda:0" = var_mean_11[1];  var_mean_11 = None
        add_39: "f32[128, 198, 1][198, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_64, 1e-06);  getitem_64 = None
        rsqrt_11: "f32[128, 198, 1][198, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_39);  add_39 = None
        sub_11: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_38, getitem_65);  getitem_65 = None
        mul_37: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_11);  sub_11 = None
        mul_38: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_37, primals_73)
        add_40: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_38, primals_74);  mul_38 = primals_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_57: "f32[25344, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(add_40, [25344, 768]);  add_40 = None
        permute_35: "f32[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(primals_75, [1, 0])
        addmm_22: "f32[25344, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(primals_76, view_57, permute_35);  primals_76 = permute_35 = None
        view_58: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_22, [128, 198, 3072])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        mul_39: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_58, 0.5)
        mul_40: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_58, 0.7071067811865476);  view_58 = None
        erf_5: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_40);  mul_40 = None
        add_41: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
        mul_41: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_39, add_41);  mul_39 = add_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_59: "f32[25344, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(mul_41, [25344, 3072]);  mul_41 = None
        permute_36: "f32[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(primals_77, [1, 0])
        addmm_23: "f32[25344, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(primals_78, view_59, permute_36);  primals_78 = permute_36 = None
        view_60: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_23, [128, 198, 768]);  addmm_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        add_42: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_38, view_60);  add_38 = view_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_12 = torch.ops.aten.var_mean.correction(add_42, [2], correction = 0, keepdim = True)
        getitem_66: "f32[128, 198, 1][198, 1, 1]cuda:0" = var_mean_12[0]
        getitem_67: "f32[128, 198, 1][198, 1, 1]cuda:0" = var_mean_12[1];  var_mean_12 = None
        add_43: "f32[128, 198, 1][198, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_66, 1e-06);  getitem_66 = None
        rsqrt_12: "f32[128, 198, 1][198, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_43);  add_43 = None
        sub_12: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_42, getitem_67);  getitem_67 = None
        mul_42: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_12);  sub_12 = None
        mul_43: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_42, primals_79)
        add_44: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_43, primals_80);  mul_43 = primals_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        view_61: "f32[25344, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(add_44, [25344, 768]);  add_44 = None
        permute_37: "f32[768, 2304][1, 768]cuda:0" = torch.ops.aten.permute.default(primals_81, [1, 0])
        addmm_24: "f32[25344, 2304][2304, 1]cuda:0" = torch.ops.aten.addmm.default(primals_82, view_61, permute_37);  primals_82 = permute_37 = None
        view_62: "f32[128, 198, 2304][456192, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_24, [128, 198, 2304]);  addmm_24 = None
        view_63: "f32[128, 198, 3, 12, 64][456192, 2304, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_62, [128, 198, 3, 12, 64]);  view_62 = None
        permute_38: "f32[3, 128, 12, 198, 64][768, 456192, 64, 2304, 1]cuda:0" = torch.ops.aten.permute.default(view_63, [2, 0, 3, 1, 4]);  view_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        unbind_6 = torch.ops.aten.unbind.int(permute_38);  permute_38 = None
        getitem_68: "f32[128, 12, 198, 64][456192, 64, 2304, 1]cuda:0" = unbind_6[0]
        getitem_69: "f32[128, 12, 198, 64][456192, 64, 2304, 1]cuda:0" = unbind_6[1]
        getitem_70: "f32[128, 12, 198, 64][456192, 64, 2304, 1]cuda:0" = unbind_6[2];  unbind_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:120 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_6 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_68, getitem_69, getitem_70, None, True)
        getitem_71: "f32[128, 12, 198, 64][152064, 64, 768, 1]cuda:0" = _scaled_dot_product_efficient_attention_6[0]
        getitem_72: "f32[128, 12, 224][2688, 224, 1]cuda:0" = _scaled_dot_product_efficient_attention_6[1]
        getitem_73: "i64[][]cuda:0" = _scaled_dot_product_efficient_attention_6[2]
        getitem_74: "i64[][]cuda:0" = _scaled_dot_product_efficient_attention_6[3];  _scaled_dot_product_efficient_attention_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_39: "f32[128, 198, 12, 64][152064, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_71, [0, 2, 1, 3])
        view_64: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_39, [128, 198, 768]);  permute_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_65: "f32[25344, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_64, [25344, 768]);  view_64 = None
        permute_40: "f32[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(primals_83, [1, 0])
        addmm_25: "f32[25344, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(primals_84, view_65, permute_40);  primals_84 = view_65 = permute_40 = None
        view_66: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_25, [128, 198, 768]);  addmm_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        add_45: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_42, view_66);  add_42 = view_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_13 = torch.ops.aten.var_mean.correction(add_45, [2], correction = 0, keepdim = True)
        getitem_75: "f32[128, 198, 1][198, 1, 1]cuda:0" = var_mean_13[0]
        getitem_76: "f32[128, 198, 1][198, 1, 1]cuda:0" = var_mean_13[1];  var_mean_13 = None
        add_46: "f32[128, 198, 1][198, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_75, 1e-06);  getitem_75 = None
        rsqrt_13: "f32[128, 198, 1][198, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_46);  add_46 = None
        sub_13: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_45, getitem_76);  getitem_76 = None
        mul_44: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_13);  sub_13 = None
        mul_45: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_44, primals_85)
        add_47: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_45, primals_86);  mul_45 = primals_86 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_67: "f32[25344, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(add_47, [25344, 768]);  add_47 = None
        permute_41: "f32[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(primals_87, [1, 0])
        addmm_26: "f32[25344, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(primals_88, view_67, permute_41);  primals_88 = permute_41 = None
        view_68: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_26, [128, 198, 3072])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        mul_46: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_68, 0.5)
        mul_47: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_68, 0.7071067811865476);  view_68 = None
        erf_6: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_47);  mul_47 = None
        add_48: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
        mul_48: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_46, add_48);  mul_46 = add_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_69: "f32[25344, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(mul_48, [25344, 3072]);  mul_48 = None
        permute_42: "f32[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(primals_89, [1, 0])
        addmm_27: "f32[25344, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(primals_90, view_69, permute_42);  primals_90 = permute_42 = None
        view_70: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_27, [128, 198, 768]);  addmm_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        add_49: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_45, view_70);  add_45 = view_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_14 = torch.ops.aten.var_mean.correction(add_49, [2], correction = 0, keepdim = True)
        getitem_77: "f32[128, 198, 1][198, 1, 1]cuda:0" = var_mean_14[0]
        getitem_78: "f32[128, 198, 1][198, 1, 1]cuda:0" = var_mean_14[1];  var_mean_14 = None
        add_50: "f32[128, 198, 1][198, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_77, 1e-06);  getitem_77 = None
        rsqrt_14: "f32[128, 198, 1][198, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_50);  add_50 = None
        sub_14: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_49, getitem_78);  getitem_78 = None
        mul_49: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_14);  sub_14 = None
        mul_50: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_49, primals_91)
        add_51: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_50, primals_92);  mul_50 = primals_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        view_71: "f32[25344, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(add_51, [25344, 768]);  add_51 = None
        permute_43: "f32[768, 2304][1, 768]cuda:0" = torch.ops.aten.permute.default(primals_93, [1, 0])
        addmm_28: "f32[25344, 2304][2304, 1]cuda:0" = torch.ops.aten.addmm.default(primals_94, view_71, permute_43);  primals_94 = permute_43 = None
        view_72: "f32[128, 198, 2304][456192, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_28, [128, 198, 2304]);  addmm_28 = None
        view_73: "f32[128, 198, 3, 12, 64][456192, 2304, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_72, [128, 198, 3, 12, 64]);  view_72 = None
        permute_44: "f32[3, 128, 12, 198, 64][768, 456192, 64, 2304, 1]cuda:0" = torch.ops.aten.permute.default(view_73, [2, 0, 3, 1, 4]);  view_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        unbind_7 = torch.ops.aten.unbind.int(permute_44);  permute_44 = None
        getitem_79: "f32[128, 12, 198, 64][456192, 64, 2304, 1]cuda:0" = unbind_7[0]
        getitem_80: "f32[128, 12, 198, 64][456192, 64, 2304, 1]cuda:0" = unbind_7[1]
        getitem_81: "f32[128, 12, 198, 64][456192, 64, 2304, 1]cuda:0" = unbind_7[2];  unbind_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:120 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_7 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_79, getitem_80, getitem_81, None, True)
        getitem_82: "f32[128, 12, 198, 64][152064, 64, 768, 1]cuda:0" = _scaled_dot_product_efficient_attention_7[0]
        getitem_83: "f32[128, 12, 224][2688, 224, 1]cuda:0" = _scaled_dot_product_efficient_attention_7[1]
        getitem_84: "i64[][]cuda:0" = _scaled_dot_product_efficient_attention_7[2]
        getitem_85: "i64[][]cuda:0" = _scaled_dot_product_efficient_attention_7[3];  _scaled_dot_product_efficient_attention_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_45: "f32[128, 198, 12, 64][152064, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_82, [0, 2, 1, 3])
        view_74: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_45, [128, 198, 768]);  permute_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_75: "f32[25344, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_74, [25344, 768]);  view_74 = None
        permute_46: "f32[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(primals_95, [1, 0])
        addmm_29: "f32[25344, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(primals_96, view_75, permute_46);  primals_96 = view_75 = permute_46 = None
        view_76: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_29, [128, 198, 768]);  addmm_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        add_52: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_49, view_76);  add_49 = view_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_15 = torch.ops.aten.var_mean.correction(add_52, [2], correction = 0, keepdim = True)
        getitem_86: "f32[128, 198, 1][198, 1, 1]cuda:0" = var_mean_15[0]
        getitem_87: "f32[128, 198, 1][198, 1, 1]cuda:0" = var_mean_15[1];  var_mean_15 = None
        add_53: "f32[128, 198, 1][198, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_86, 1e-06);  getitem_86 = None
        rsqrt_15: "f32[128, 198, 1][198, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_53);  add_53 = None
        sub_15: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_52, getitem_87);  getitem_87 = None
        mul_51: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_15);  sub_15 = None
        mul_52: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_51, primals_97)
        add_54: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_52, primals_98);  mul_52 = primals_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_77: "f32[25344, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(add_54, [25344, 768]);  add_54 = None
        permute_47: "f32[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(primals_99, [1, 0])
        addmm_30: "f32[25344, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(primals_100, view_77, permute_47);  primals_100 = permute_47 = None
        view_78: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_30, [128, 198, 3072])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        mul_53: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_78, 0.5)
        mul_54: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_78, 0.7071067811865476);  view_78 = None
        erf_7: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_54);  mul_54 = None
        add_55: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
        mul_55: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_53, add_55);  mul_53 = add_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_79: "f32[25344, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(mul_55, [25344, 3072]);  mul_55 = None
        permute_48: "f32[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(primals_101, [1, 0])
        addmm_31: "f32[25344, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(primals_102, view_79, permute_48);  primals_102 = permute_48 = None
        view_80: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_31, [128, 198, 768]);  addmm_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        add_56: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_52, view_80);  add_52 = view_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_16 = torch.ops.aten.var_mean.correction(add_56, [2], correction = 0, keepdim = True)
        getitem_88: "f32[128, 198, 1][198, 1, 1]cuda:0" = var_mean_16[0]
        getitem_89: "f32[128, 198, 1][198, 1, 1]cuda:0" = var_mean_16[1];  var_mean_16 = None
        add_57: "f32[128, 198, 1][198, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_88, 1e-06);  getitem_88 = None
        rsqrt_16: "f32[128, 198, 1][198, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_57);  add_57 = None
        sub_16: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_56, getitem_89);  getitem_89 = None
        mul_56: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_16);  sub_16 = None
        mul_57: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_56, primals_103)
        add_58: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_57, primals_104);  mul_57 = primals_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        view_81: "f32[25344, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(add_58, [25344, 768]);  add_58 = None
        permute_49: "f32[768, 2304][1, 768]cuda:0" = torch.ops.aten.permute.default(primals_105, [1, 0])
        addmm_32: "f32[25344, 2304][2304, 1]cuda:0" = torch.ops.aten.addmm.default(primals_106, view_81, permute_49);  primals_106 = permute_49 = None
        view_82: "f32[128, 198, 2304][456192, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_32, [128, 198, 2304]);  addmm_32 = None
        view_83: "f32[128, 198, 3, 12, 64][456192, 2304, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_82, [128, 198, 3, 12, 64]);  view_82 = None
        permute_50: "f32[3, 128, 12, 198, 64][768, 456192, 64, 2304, 1]cuda:0" = torch.ops.aten.permute.default(view_83, [2, 0, 3, 1, 4]);  view_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        unbind_8 = torch.ops.aten.unbind.int(permute_50);  permute_50 = None
        getitem_90: "f32[128, 12, 198, 64][456192, 64, 2304, 1]cuda:0" = unbind_8[0]
        getitem_91: "f32[128, 12, 198, 64][456192, 64, 2304, 1]cuda:0" = unbind_8[1]
        getitem_92: "f32[128, 12, 198, 64][456192, 64, 2304, 1]cuda:0" = unbind_8[2];  unbind_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:120 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_8 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_90, getitem_91, getitem_92, None, True)
        getitem_93: "f32[128, 12, 198, 64][152064, 64, 768, 1]cuda:0" = _scaled_dot_product_efficient_attention_8[0]
        getitem_94: "f32[128, 12, 224][2688, 224, 1]cuda:0" = _scaled_dot_product_efficient_attention_8[1]
        getitem_95: "i64[][]cuda:0" = _scaled_dot_product_efficient_attention_8[2]
        getitem_96: "i64[][]cuda:0" = _scaled_dot_product_efficient_attention_8[3];  _scaled_dot_product_efficient_attention_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_51: "f32[128, 198, 12, 64][152064, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_93, [0, 2, 1, 3])
        view_84: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_51, [128, 198, 768]);  permute_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_85: "f32[25344, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_84, [25344, 768]);  view_84 = None
        permute_52: "f32[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(primals_107, [1, 0])
        addmm_33: "f32[25344, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(primals_108, view_85, permute_52);  primals_108 = view_85 = permute_52 = None
        view_86: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_33, [128, 198, 768]);  addmm_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        add_59: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_56, view_86);  add_56 = view_86 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_17 = torch.ops.aten.var_mean.correction(add_59, [2], correction = 0, keepdim = True)
        getitem_97: "f32[128, 198, 1][198, 1, 1]cuda:0" = var_mean_17[0]
        getitem_98: "f32[128, 198, 1][198, 1, 1]cuda:0" = var_mean_17[1];  var_mean_17 = None
        add_60: "f32[128, 198, 1][198, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_97, 1e-06);  getitem_97 = None
        rsqrt_17: "f32[128, 198, 1][198, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_60);  add_60 = None
        sub_17: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_59, getitem_98);  getitem_98 = None
        mul_58: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_17);  sub_17 = None
        mul_59: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_58, primals_109)
        add_61: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_59, primals_110);  mul_59 = primals_110 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_87: "f32[25344, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(add_61, [25344, 768]);  add_61 = None
        permute_53: "f32[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(primals_111, [1, 0])
        addmm_34: "f32[25344, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(primals_112, view_87, permute_53);  primals_112 = permute_53 = None
        view_88: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_34, [128, 198, 3072])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        mul_60: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_88, 0.5)
        mul_61: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_88, 0.7071067811865476);  view_88 = None
        erf_8: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_61);  mul_61 = None
        add_62: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
        mul_62: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_60, add_62);  mul_60 = add_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_89: "f32[25344, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(mul_62, [25344, 3072]);  mul_62 = None
        permute_54: "f32[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(primals_113, [1, 0])
        addmm_35: "f32[25344, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(primals_114, view_89, permute_54);  primals_114 = permute_54 = None
        view_90: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_35, [128, 198, 768]);  addmm_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        add_63: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_59, view_90);  add_59 = view_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_18 = torch.ops.aten.var_mean.correction(add_63, [2], correction = 0, keepdim = True)
        getitem_99: "f32[128, 198, 1][198, 1, 1]cuda:0" = var_mean_18[0]
        getitem_100: "f32[128, 198, 1][198, 1, 1]cuda:0" = var_mean_18[1];  var_mean_18 = None
        add_64: "f32[128, 198, 1][198, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_99, 1e-06);  getitem_99 = None
        rsqrt_18: "f32[128, 198, 1][198, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_64);  add_64 = None
        sub_18: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_63, getitem_100);  getitem_100 = None
        mul_63: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_18);  sub_18 = None
        mul_64: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_63, primals_115)
        add_65: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_64, primals_116);  mul_64 = primals_116 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        view_91: "f32[25344, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(add_65, [25344, 768]);  add_65 = None
        permute_55: "f32[768, 2304][1, 768]cuda:0" = torch.ops.aten.permute.default(primals_117, [1, 0])
        addmm_36: "f32[25344, 2304][2304, 1]cuda:0" = torch.ops.aten.addmm.default(primals_118, view_91, permute_55);  primals_118 = permute_55 = None
        view_92: "f32[128, 198, 2304][456192, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_36, [128, 198, 2304]);  addmm_36 = None
        view_93: "f32[128, 198, 3, 12, 64][456192, 2304, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_92, [128, 198, 3, 12, 64]);  view_92 = None
        permute_56: "f32[3, 128, 12, 198, 64][768, 456192, 64, 2304, 1]cuda:0" = torch.ops.aten.permute.default(view_93, [2, 0, 3, 1, 4]);  view_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        unbind_9 = torch.ops.aten.unbind.int(permute_56);  permute_56 = None
        getitem_101: "f32[128, 12, 198, 64][456192, 64, 2304, 1]cuda:0" = unbind_9[0]
        getitem_102: "f32[128, 12, 198, 64][456192, 64, 2304, 1]cuda:0" = unbind_9[1]
        getitem_103: "f32[128, 12, 198, 64][456192, 64, 2304, 1]cuda:0" = unbind_9[2];  unbind_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:120 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_9 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_101, getitem_102, getitem_103, None, True)
        getitem_104: "f32[128, 12, 198, 64][152064, 64, 768, 1]cuda:0" = _scaled_dot_product_efficient_attention_9[0]
        getitem_105: "f32[128, 12, 224][2688, 224, 1]cuda:0" = _scaled_dot_product_efficient_attention_9[1]
        getitem_106: "i64[][]cuda:0" = _scaled_dot_product_efficient_attention_9[2]
        getitem_107: "i64[][]cuda:0" = _scaled_dot_product_efficient_attention_9[3];  _scaled_dot_product_efficient_attention_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_57: "f32[128, 198, 12, 64][152064, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_104, [0, 2, 1, 3])
        view_94: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_57, [128, 198, 768]);  permute_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_95: "f32[25344, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_94, [25344, 768]);  view_94 = None
        permute_58: "f32[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(primals_119, [1, 0])
        addmm_37: "f32[25344, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(primals_120, view_95, permute_58);  primals_120 = view_95 = permute_58 = None
        view_96: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_37, [128, 198, 768]);  addmm_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        add_66: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_63, view_96);  add_63 = view_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_19 = torch.ops.aten.var_mean.correction(add_66, [2], correction = 0, keepdim = True)
        getitem_108: "f32[128, 198, 1][198, 1, 1]cuda:0" = var_mean_19[0]
        getitem_109: "f32[128, 198, 1][198, 1, 1]cuda:0" = var_mean_19[1];  var_mean_19 = None
        add_67: "f32[128, 198, 1][198, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_108, 1e-06);  getitem_108 = None
        rsqrt_19: "f32[128, 198, 1][198, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_67);  add_67 = None
        sub_19: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_66, getitem_109);  getitem_109 = None
        mul_65: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_19);  sub_19 = None
        mul_66: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_65, primals_121)
        add_68: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_66, primals_122);  mul_66 = primals_122 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_97: "f32[25344, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(add_68, [25344, 768]);  add_68 = None
        permute_59: "f32[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(primals_123, [1, 0])
        addmm_38: "f32[25344, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(primals_124, view_97, permute_59);  primals_124 = permute_59 = None
        view_98: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_38, [128, 198, 3072])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        mul_67: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_98, 0.5)
        mul_68: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_98, 0.7071067811865476);  view_98 = None
        erf_9: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_68);  mul_68 = None
        add_69: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
        mul_69: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_67, add_69);  mul_67 = add_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_99: "f32[25344, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(mul_69, [25344, 3072]);  mul_69 = None
        permute_60: "f32[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(primals_125, [1, 0])
        addmm_39: "f32[25344, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(primals_126, view_99, permute_60);  primals_126 = permute_60 = None
        view_100: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_39, [128, 198, 768]);  addmm_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        add_70: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_66, view_100);  add_66 = view_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_20 = torch.ops.aten.var_mean.correction(add_70, [2], correction = 0, keepdim = True)
        getitem_110: "f32[128, 198, 1][198, 1, 1]cuda:0" = var_mean_20[0]
        getitem_111: "f32[128, 198, 1][198, 1, 1]cuda:0" = var_mean_20[1];  var_mean_20 = None
        add_71: "f32[128, 198, 1][198, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_110, 1e-06);  getitem_110 = None
        rsqrt_20: "f32[128, 198, 1][198, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_71);  add_71 = None
        sub_20: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_70, getitem_111);  getitem_111 = None
        mul_70: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_20);  sub_20 = None
        mul_71: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_70, primals_127)
        add_72: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_71, primals_128);  mul_71 = primals_128 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        view_101: "f32[25344, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(add_72, [25344, 768]);  add_72 = None
        permute_61: "f32[768, 2304][1, 768]cuda:0" = torch.ops.aten.permute.default(primals_129, [1, 0])
        addmm_40: "f32[25344, 2304][2304, 1]cuda:0" = torch.ops.aten.addmm.default(primals_130, view_101, permute_61);  primals_130 = permute_61 = None
        view_102: "f32[128, 198, 2304][456192, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_40, [128, 198, 2304]);  addmm_40 = None
        view_103: "f32[128, 198, 3, 12, 64][456192, 2304, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_102, [128, 198, 3, 12, 64]);  view_102 = None
        permute_62: "f32[3, 128, 12, 198, 64][768, 456192, 64, 2304, 1]cuda:0" = torch.ops.aten.permute.default(view_103, [2, 0, 3, 1, 4]);  view_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        unbind_10 = torch.ops.aten.unbind.int(permute_62);  permute_62 = None
        getitem_112: "f32[128, 12, 198, 64][456192, 64, 2304, 1]cuda:0" = unbind_10[0]
        getitem_113: "f32[128, 12, 198, 64][456192, 64, 2304, 1]cuda:0" = unbind_10[1]
        getitem_114: "f32[128, 12, 198, 64][456192, 64, 2304, 1]cuda:0" = unbind_10[2];  unbind_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:120 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_10 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_112, getitem_113, getitem_114, None, True)
        getitem_115: "f32[128, 12, 198, 64][152064, 64, 768, 1]cuda:0" = _scaled_dot_product_efficient_attention_10[0]
        getitem_116: "f32[128, 12, 224][2688, 224, 1]cuda:0" = _scaled_dot_product_efficient_attention_10[1]
        getitem_117: "i64[][]cuda:0" = _scaled_dot_product_efficient_attention_10[2]
        getitem_118: "i64[][]cuda:0" = _scaled_dot_product_efficient_attention_10[3];  _scaled_dot_product_efficient_attention_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_63: "f32[128, 198, 12, 64][152064, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_115, [0, 2, 1, 3])
        view_104: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_63, [128, 198, 768]);  permute_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_105: "f32[25344, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_104, [25344, 768]);  view_104 = None
        permute_64: "f32[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(primals_131, [1, 0])
        addmm_41: "f32[25344, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(primals_132, view_105, permute_64);  primals_132 = view_105 = permute_64 = None
        view_106: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_41, [128, 198, 768]);  addmm_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        add_73: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_70, view_106);  add_70 = view_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_21 = torch.ops.aten.var_mean.correction(add_73, [2], correction = 0, keepdim = True)
        getitem_119: "f32[128, 198, 1][198, 1, 1]cuda:0" = var_mean_21[0]
        getitem_120: "f32[128, 198, 1][198, 1, 1]cuda:0" = var_mean_21[1];  var_mean_21 = None
        add_74: "f32[128, 198, 1][198, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_119, 1e-06);  getitem_119 = None
        rsqrt_21: "f32[128, 198, 1][198, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_74);  add_74 = None
        sub_21: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_73, getitem_120);  getitem_120 = None
        mul_72: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_21);  sub_21 = None
        mul_73: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_72, primals_133)
        add_75: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_73, primals_134);  mul_73 = primals_134 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_107: "f32[25344, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(add_75, [25344, 768]);  add_75 = None
        permute_65: "f32[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(primals_135, [1, 0])
        addmm_42: "f32[25344, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(primals_136, view_107, permute_65);  primals_136 = permute_65 = None
        view_108: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_42, [128, 198, 3072])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        mul_74: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_108, 0.5)
        mul_75: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_108, 0.7071067811865476);  view_108 = None
        erf_10: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_75);  mul_75 = None
        add_76: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
        mul_76: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_74, add_76);  mul_74 = add_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_109: "f32[25344, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(mul_76, [25344, 3072]);  mul_76 = None
        permute_66: "f32[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(primals_137, [1, 0])
        addmm_43: "f32[25344, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(primals_138, view_109, permute_66);  primals_138 = permute_66 = None
        view_110: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_43, [128, 198, 768]);  addmm_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        add_77: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_73, view_110);  add_73 = view_110 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_22 = torch.ops.aten.var_mean.correction(add_77, [2], correction = 0, keepdim = True)
        getitem_121: "f32[128, 198, 1][198, 1, 1]cuda:0" = var_mean_22[0]
        getitem_122: "f32[128, 198, 1][198, 1, 1]cuda:0" = var_mean_22[1];  var_mean_22 = None
        add_78: "f32[128, 198, 1][198, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_121, 1e-06);  getitem_121 = None
        rsqrt_22: "f32[128, 198, 1][198, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_78);  add_78 = None
        sub_22: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_77, getitem_122);  getitem_122 = None
        mul_77: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_22);  sub_22 = None
        mul_78: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_77, primals_139)
        add_79: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_78, primals_140);  mul_78 = primals_140 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        view_111: "f32[25344, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(add_79, [25344, 768]);  add_79 = None
        permute_67: "f32[768, 2304][1, 768]cuda:0" = torch.ops.aten.permute.default(primals_141, [1, 0])
        addmm_44: "f32[25344, 2304][2304, 1]cuda:0" = torch.ops.aten.addmm.default(primals_142, view_111, permute_67);  primals_142 = permute_67 = None
        view_112: "f32[128, 198, 2304][456192, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_44, [128, 198, 2304]);  addmm_44 = None
        view_113: "f32[128, 198, 3, 12, 64][456192, 2304, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_112, [128, 198, 3, 12, 64]);  view_112 = None
        permute_68: "f32[3, 128, 12, 198, 64][768, 456192, 64, 2304, 1]cuda:0" = torch.ops.aten.permute.default(view_113, [2, 0, 3, 1, 4]);  view_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        unbind_11 = torch.ops.aten.unbind.int(permute_68);  permute_68 = None
        getitem_123: "f32[128, 12, 198, 64][456192, 64, 2304, 1]cuda:0" = unbind_11[0]
        getitem_124: "f32[128, 12, 198, 64][456192, 64, 2304, 1]cuda:0" = unbind_11[1]
        getitem_125: "f32[128, 12, 198, 64][456192, 64, 2304, 1]cuda:0" = unbind_11[2];  unbind_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:120 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_11 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_123, getitem_124, getitem_125, None, True)
        getitem_126: "f32[128, 12, 198, 64][152064, 64, 768, 1]cuda:0" = _scaled_dot_product_efficient_attention_11[0]
        getitem_127: "f32[128, 12, 224][2688, 224, 1]cuda:0" = _scaled_dot_product_efficient_attention_11[1]
        getitem_128: "i64[][]cuda:0" = _scaled_dot_product_efficient_attention_11[2]
        getitem_129: "i64[][]cuda:0" = _scaled_dot_product_efficient_attention_11[3];  _scaled_dot_product_efficient_attention_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_69: "f32[128, 198, 12, 64][152064, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_126, [0, 2, 1, 3])
        view_114: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_69, [128, 198, 768]);  permute_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_115: "f32[25344, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_114, [25344, 768]);  view_114 = None
        permute_70: "f32[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(primals_143, [1, 0])
        addmm_45: "f32[25344, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(primals_144, view_115, permute_70);  primals_144 = view_115 = permute_70 = None
        view_116: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_45, [128, 198, 768]);  addmm_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        add_80: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_77, view_116);  add_77 = view_116 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_23 = torch.ops.aten.var_mean.correction(add_80, [2], correction = 0, keepdim = True)
        getitem_130: "f32[128, 198, 1][198, 1, 1]cuda:0" = var_mean_23[0]
        getitem_131: "f32[128, 198, 1][198, 1, 1]cuda:0" = var_mean_23[1];  var_mean_23 = None
        add_81: "f32[128, 198, 1][198, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_130, 1e-06);  getitem_130 = None
        rsqrt_23: "f32[128, 198, 1][198, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_81);  add_81 = None
        sub_23: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_80, getitem_131);  getitem_131 = None
        mul_79: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_23);  sub_23 = None
        mul_80: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_79, primals_145)
        add_82: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_80, primals_146);  mul_80 = primals_146 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_117: "f32[25344, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(add_82, [25344, 768]);  add_82 = None
        permute_71: "f32[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(primals_147, [1, 0])
        addmm_46: "f32[25344, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(primals_148, view_117, permute_71);  primals_148 = permute_71 = None
        view_118: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_46, [128, 198, 3072])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        mul_81: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_118, 0.5)
        mul_82: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_118, 0.7071067811865476);  view_118 = None
        erf_11: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_82);  mul_82 = None
        add_83: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
        mul_83: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_81, add_83);  mul_81 = add_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_119: "f32[25344, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(mul_83, [25344, 3072]);  mul_83 = None
        permute_72: "f32[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(primals_149, [1, 0])
        addmm_47: "f32[25344, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(primals_150, view_119, permute_72);  primals_150 = permute_72 = None
        view_120: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_47, [128, 198, 768]);  addmm_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        add_84: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_80, view_120);  add_80 = view_120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_24 = torch.ops.aten.var_mean.correction(add_84, [2], correction = 0, keepdim = True)
        getitem_132: "f32[128, 198, 1][198, 1, 1]cuda:0" = var_mean_24[0]
        getitem_133: "f32[128, 198, 1][198, 1, 1]cuda:0" = var_mean_24[1];  var_mean_24 = None
        add_85: "f32[128, 198, 1][198, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_132, 1e-06);  getitem_132 = None
        rsqrt_24: "f32[128, 198, 1][198, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_85);  add_85 = None
        sub_24: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_84, getitem_133);  add_84 = getitem_133 = None
        mul_84: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_24);  sub_24 = None
        mul_85: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_84, primals_151)
        add_86: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_85, primals_152);  mul_85 = primals_152 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/deit.py:114 in forward_head, code: x, x_dist = x[:, 0], x[:, 1]
        select: "f32[128, 768][152064, 1]cuda:0" = torch.ops.aten.select.int(add_86, 1, 0)
        select_1: "f32[128, 768][152064, 1]cuda:0" = torch.ops.aten.select.int(add_86, 1, 1);  add_86 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/deit.py:117 in forward_head, code: x = self.head(x)
        permute_73: "f32[768, 1000][1, 768]cuda:0" = torch.ops.aten.permute.default(primals_153, [1, 0])
        addmm_48: "f32[128, 1000][1000, 1]cuda:0" = torch.ops.aten.addmm.default(primals_154, select, permute_73);  primals_154 = permute_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/deit.py:118 in forward_head, code: x_dist = self.head_dist(x_dist)
        permute_74: "f32[768, 1000][1, 768]cuda:0" = torch.ops.aten.permute.default(primals_155, [1, 0])
        addmm_49: "f32[128, 1000][1000, 1]cuda:0" = torch.ops.aten.addmm.default(primals_156, select_1, permute_74);  primals_156 = permute_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/deit.py:124 in forward_head, code: return (x + x_dist) / 2
        add_87: "f32[128, 1000][1000, 1]cuda:0" = torch.ops.aten.add.Tensor(addmm_48, addmm_49);  addmm_48 = addmm_49 = None
        div: "f32[128, 1000][1000, 1]cuda:0" = torch.ops.aten.div.Tensor(add_87, 2);  add_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        div_2: "f32[128, 198, 1][198, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_24, 768);  rsqrt_24 = None
        div_3: "f32[128, 198, 1][198, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_23, 768);  rsqrt_23 = None
        div_4: "f32[128, 198, 1][198, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_22, 768);  rsqrt_22 = None
        div_5: "f32[128, 198, 1][198, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_21, 768);  rsqrt_21 = None
        div_6: "f32[128, 198, 1][198, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_20, 768);  rsqrt_20 = None
        div_7: "f32[128, 198, 1][198, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_19, 768);  rsqrt_19 = None
        div_8: "f32[128, 198, 1][198, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_18, 768);  rsqrt_18 = None
        div_9: "f32[128, 198, 1][198, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_17, 768);  rsqrt_17 = None
        div_10: "f32[128, 198, 1][198, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_16, 768);  rsqrt_16 = None
        div_11: "f32[128, 198, 1][198, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_15, 768);  rsqrt_15 = None
        div_12: "f32[128, 198, 1][198, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_14, 768);  rsqrt_14 = None
        div_13: "f32[128, 198, 1][198, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_13, 768);  rsqrt_13 = None
        div_14: "f32[128, 198, 1][198, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_12, 768);  rsqrt_12 = None
        div_15: "f32[128, 198, 1][198, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_11, 768);  rsqrt_11 = None
        div_16: "f32[128, 198, 1][198, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_10, 768);  rsqrt_10 = None
        div_17: "f32[128, 198, 1][198, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_9, 768);  rsqrt_9 = None
        div_18: "f32[128, 198, 1][198, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_8, 768);  rsqrt_8 = None
        div_19: "f32[128, 198, 1][198, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_7, 768);  rsqrt_7 = None
        div_20: "f32[128, 198, 1][198, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_6, 768);  rsqrt_6 = None
        div_21: "f32[128, 198, 1][198, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_5, 768);  rsqrt_5 = None
        div_22: "f32[128, 198, 1][198, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_4, 768);  rsqrt_4 = None
        div_23: "f32[128, 198, 1][198, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_3, 768);  rsqrt_3 = None
        div_24: "f32[128, 198, 1][198, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_2, 768);  rsqrt_2 = None
        div_25: "f32[128, 198, 1][198, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_1, 768);  rsqrt_1 = None
        return (div, primals_1, primals_2, primals_4, primals_7, primals_9, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, primals_27, primals_29, primals_31, primals_33, primals_35, primals_37, primals_39, primals_41, primals_43, primals_45, primals_47, primals_49, primals_51, primals_53, primals_55, primals_57, primals_59, primals_61, primals_63, primals_65, primals_67, primals_69, primals_71, primals_73, primals_75, primals_77, primals_79, primals_81, primals_83, primals_85, primals_87, primals_89, primals_91, primals_93, primals_95, primals_97, primals_99, primals_101, primals_103, primals_105, primals_107, primals_109, primals_111, primals_113, primals_115, primals_117, primals_119, primals_121, primals_123, primals_125, primals_127, primals_129, primals_131, primals_133, primals_135, primals_137, primals_139, primals_141, primals_143, primals_145, primals_147, primals_149, primals_151, primals_153, primals_155, cat, getitem_1, rsqrt, view_1, getitem_2, getitem_3, getitem_4, getitem_5, getitem_6, getitem_7, getitem_8, mul_2, view_7, addmm_2, view_9, mul_7, view_11, getitem_13, getitem_14, getitem_15, getitem_16, getitem_17, getitem_18, getitem_19, mul_9, view_17, addmm_6, view_19, mul_14, view_21, getitem_24, getitem_25, getitem_26, getitem_27, getitem_28, getitem_29, getitem_30, mul_16, view_27, addmm_10, view_29, mul_21, view_31, getitem_35, getitem_36, getitem_37, getitem_38, getitem_39, getitem_40, getitem_41, mul_23, view_37, addmm_14, view_39, mul_28, view_41, getitem_46, getitem_47, getitem_48, getitem_49, getitem_50, getitem_51, getitem_52, mul_30, view_47, addmm_18, view_49, mul_35, view_51, getitem_57, getitem_58, getitem_59, getitem_60, getitem_61, getitem_62, getitem_63, mul_37, view_57, addmm_22, view_59, mul_42, view_61, getitem_68, getitem_69, getitem_70, getitem_71, getitem_72, getitem_73, getitem_74, mul_44, view_67, addmm_26, view_69, mul_49, view_71, getitem_79, getitem_80, getitem_81, getitem_82, getitem_83, getitem_84, getitem_85, mul_51, view_77, addmm_30, view_79, mul_56, view_81, getitem_90, getitem_91, getitem_92, getitem_93, getitem_94, getitem_95, getitem_96, mul_58, view_87, addmm_34, view_89, mul_63, view_91, getitem_101, getitem_102, getitem_103, getitem_104, getitem_105, getitem_106, getitem_107, mul_65, view_97, addmm_38, view_99, mul_70, view_101, getitem_112, getitem_113, getitem_114, getitem_115, getitem_116, getitem_117, getitem_118, mul_72, view_107, addmm_42, view_109, mul_77, view_111, getitem_123, getitem_124, getitem_125, getitem_126, getitem_127, getitem_128, getitem_129, mul_79, view_117, addmm_46, view_119, mul_84, select, select_1, div_2, div_3, div_4, div_5, div_6, div_7, div_8, div_9, div_10, div_11, div_12, div_13, div_14, div_15, div_16, div_17, div_18, div_19, div_20, div_21, div_22, div_23, div_24, div_25)
