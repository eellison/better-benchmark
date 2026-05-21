class GraphModule(torch.nn.Module):
    def forward(self, tangents_1: "f32[128, 1000]", _shape_param_0, sum_3: "f32[640]", squeeze_94: "f32[640]", sum_5: "f32[160]", squeeze_91: "f32[160]", sum_7: "f32[160]", squeeze_88: "f32[160]", view_113: "f32[512, 16, 240]", mul_243: "f32[512, 16, 240]", view_114: "f32[8192, 240]", _shape_param_1, view_117: "f32[8192, 480]", _shape_param_2, view_119: "f32[512, 16, 240]", mul_241: "f32[512, 16, 240]", view_120: "f32[8192, 240]", getitem_155: "f32[512, 4, 16, 60]", _shape_param_3, _shape_param_4, _shape_param_5, view_126: "f32[8192, 720]", _shape_param_6, view_128: "f32[512, 16, 240]", mul_239: "f32[512, 16, 240]", view_129: "f32[8192, 240]", _shape_param_7, view_132: "f32[8192, 480]", _shape_param_8, view_134: "f32[512, 16, 240]", mul_237: "f32[512, 16, 240]", view_135: "f32[8192, 240]", getitem_144: "f32[512, 4, 16, 60]", _shape_param_9, _shape_param_10, _shape_param_11, view_141: "f32[8192, 720]", _shape_param_12, view_143: "f32[512, 16, 240]", mul_235: "f32[512, 16, 240]", view_144: "f32[8192, 240]", _shape_param_13, view_147: "f32[8192, 480]", _shape_param_14, view_149: "f32[512, 16, 240]", mul_233: "f32[512, 16, 240]", view_150: "f32[8192, 240]", getitem_133: "f32[512, 4, 16, 60]", _shape_param_15, _shape_param_16, _shape_param_17, view_156: "f32[8192, 720]", _shape_param_18, view_158: "f32[512, 16, 240]", mul_231: "f32[512, 16, 240]", sum_49: "f32[160]", squeeze_85: "f32[160]", sum_51: "f32[160]", squeeze_82: "f32[160]", sum_53: "f32[512]", squeeze_79: "f32[512]", sum_55: "f32[512]", squeeze_76: "f32[512]", sum_57: "f32[128]", squeeze_73: "f32[128]", sum_59: "f32[128]", squeeze_70: "f32[128]", view_164: "f32[512, 64, 192]", mul_187: "f32[512, 64, 192]", view_165: "f32[32768, 192]", _shape_param_19, view_168: "f32[32768, 384]", _shape_param_20, view_170: "f32[512, 64, 192]", mul_185: "f32[512, 64, 192]", view_171: "f32[32768, 192]", getitem_108: "f32[512, 4, 64, 48]", _shape_param_21, _shape_param_22, _shape_param_23, view_177: "f32[32768, 576]", _shape_param_24, view_179: "f32[512, 64, 192]", mul_183: "f32[512, 64, 192]", view_180: "f32[32768, 192]", _shape_param_25, view_183: "f32[32768, 384]", _shape_param_26, view_185: "f32[512, 64, 192]", mul_181: "f32[512, 64, 192]", view_186: "f32[32768, 192]", getitem_97: "f32[512, 4, 64, 48]", _shape_param_27, _shape_param_28, _shape_param_29, view_192: "f32[32768, 576]", _shape_param_30, view_194: "f32[512, 64, 192]", mul_179: "f32[512, 64, 192]", view_195: "f32[32768, 192]", _shape_param_31, view_198: "f32[32768, 384]", _shape_param_32, view_200: "f32[512, 64, 192]", mul_177: "f32[512, 64, 192]", view_201: "f32[32768, 192]", getitem_86: "f32[512, 4, 64, 48]", _shape_param_33, _shape_param_34, _shape_param_35, view_207: "f32[32768, 576]", _shape_param_36, view_209: "f32[512, 64, 192]", mul_175: "f32[512, 64, 192]", view_210: "f32[32768, 192]", _shape_param_37, view_213: "f32[32768, 384]", _shape_param_38, view_215: "f32[512, 64, 192]", mul_173: "f32[512, 64, 192]", view_216: "f32[32768, 192]", getitem_75: "f32[512, 4, 64, 48]", _shape_param_39, _shape_param_40, _shape_param_41, view_222: "f32[32768, 576]", _shape_param_42, view_224: "f32[512, 64, 192]", mul_171: "f32[512, 64, 192]", sum_113: "f32[128]", squeeze_67: "f32[128]", sum_115: "f32[128]", squeeze_64: "f32[128]", sum_117: "f32[384]", squeeze_61: "f32[384]", sum_119: "f32[384]", squeeze_58: "f32[384]", sum_121: "f32[96]", squeeze_55: "f32[96]", sum_123: "f32[96]", squeeze_52: "f32[96]", view_230: "f32[512, 256, 144]", mul_127: "f32[512, 256, 144]", view_231: "f32[131072, 144]", _shape_param_43, view_234: "f32[131072, 288]", _shape_param_44, view_236: "f32[512, 256, 144]", mul_125: "f32[512, 256, 144]", view_237: "f32[131072, 144]", getitem_50: "f32[512, 4, 256, 36]", _shape_param_45, _shape_param_46, _shape_param_47, view_243: "f32[131072, 432]", _shape_param_48, view_245: "f32[512, 256, 144]", mul_123: "f32[512, 256, 144]", view_246: "f32[131072, 144]", _shape_param_49, view_249: "f32[131072, 288]", _shape_param_50, view_251: "f32[512, 256, 144]", mul_121: "f32[512, 256, 144]", view_252: "f32[131072, 144]", getitem_39: "f32[512, 4, 256, 36]", _shape_param_51, _shape_param_52, _shape_param_53, view_258: "f32[131072, 432]", _shape_param_54, view_260: "f32[512, 256, 144]", mul_119: "f32[512, 256, 144]", sum_153: "f32[96]", squeeze_49: "f32[96]", sum_155: "f32[96]", squeeze_46: "f32[96]", sum_157: "f32[256]", squeeze_43: "f32[256]", sum_159: "f32[256]", squeeze_40: "f32[256]", sum_161: "f32[64]", squeeze_37: "f32[64]", sum_163: "f32[256]", squeeze_34: "f32[256]", sum_165: "f32[256]", squeeze_31: "f32[256]", sum_167: "f32[64]", squeeze_28: "f32[64]", sum_169: "f32[256]", squeeze_25: "f32[256]", sum_171: "f32[256]", squeeze_22: "f32[256]", sum_173: "f32[64]", squeeze_19: "f32[64]", sum_175: "f32[128]", squeeze_16: "f32[128]", sum_177: "f32[128]", squeeze_13: "f32[128]", sum_179: "f32[32]", squeeze_10: "f32[32]", sum_181: "f32[64]", squeeze_7: "f32[64]", sum_183: "f32[64]", squeeze_4: "f32[64]", sum_185: "f32[16]", squeeze_1: "f32[16]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/classifier.py:141 in forward, code: x = self.fc(x)
        permute_default: "f32[1000, 128]" = torch.ops.aten.permute.default(tangents_1, [1, 0])
        sum_dim_int_list: "f32[1, 1000]" = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True);  tangents_1 = None
        reshape_default: "f32[1000]" = torch.ops.aten.reshape.default(sum_dim_int_list, _shape_param_0);  sum_dim_int_list = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        mul_tensor: "f32[640]" = torch.ops.aten.mul.Tensor(sum_3, squeeze_94);  sum_3 = squeeze_94 = None
        mul_tensor_1: "f32[160]" = torch.ops.aten.mul.Tensor(sum_5, squeeze_91);  sum_5 = squeeze_91 = None
        mul_tensor_2: "f32[160]" = torch.ops.aten.mul.Tensor(sum_7, squeeze_88);  sum_7 = squeeze_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/mobilevit.py:264 in forward, code: x = self.norm(x)
        mul_tensor_3: "f32[512, 16, 240]" = torch.ops.aten.mul.Tensor(view_113, mul_243);  mul_243 = None
        sum_dim_int_list_1: "f32[240]" = torch.ops.aten.sum.dim_IntList(mul_tensor_3, [0, 1]);  mul_tensor_3 = None
        sum_dim_int_list_2: "f32[240]" = torch.ops.aten.sum.dim_IntList(view_113, [0, 1]);  view_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        permute_default_1: "f32[240, 8192]" = torch.ops.aten.permute.default(view_114, [1, 0])
        sum_dim_int_list_3: "f32[1, 240]" = torch.ops.aten.sum.dim_IntList(view_114, [0], True);  view_114 = None
        reshape_default_1: "f32[240]" = torch.ops.aten.reshape.default(sum_dim_int_list_3, _shape_param_1);  sum_dim_int_list_3 = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        permute_default_2: "f32[480, 8192]" = torch.ops.aten.permute.default(view_117, [1, 0])
        sum_dim_int_list_4: "f32[1, 480]" = torch.ops.aten.sum.dim_IntList(view_117, [0], True);  view_117 = None
        reshape_default_2: "f32[480]" = torch.ops.aten.reshape.default(sum_dim_int_list_4, _shape_param_2);  sum_dim_int_list_4 = _shape_param_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        mul_tensor_4: "f32[512, 16, 240]" = torch.ops.aten.mul.Tensor(view_119, mul_241);  mul_241 = None
        sum_dim_int_list_5: "f32[240]" = torch.ops.aten.sum.dim_IntList(mul_tensor_4, [0, 1]);  mul_tensor_4 = None
        sum_dim_int_list_6: "f32[240]" = torch.ops.aten.sum.dim_IntList(view_119, [0, 1]);  view_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        permute_default_3: "f32[240, 8192]" = torch.ops.aten.permute.default(view_120, [1, 0])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_default_4: "f32[512, 16, 4, 60]" = torch.ops.aten.permute.default(getitem_155, [0, 2, 1, 3]);  getitem_155 = None
        reshape_default_3: "f32[512, 16, 240]" = torch.ops.aten.reshape.default(permute_default_4, _shape_param_3);  permute_default_4 = _shape_param_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        reshape_default_4: "f32[8192, 240]" = torch.ops.aten.reshape.default(reshape_default_3, _shape_param_4);  reshape_default_3 = _shape_param_4 = None
        sum_dim_int_list_7: "f32[1, 240]" = torch.ops.aten.sum.dim_IntList(view_120, [0], True);  view_120 = None
        reshape_default_5: "f32[240]" = torch.ops.aten.reshape.default(sum_dim_int_list_7, _shape_param_5);  sum_dim_int_list_7 = _shape_param_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        permute_default_5: "f32[720, 8192]" = torch.ops.aten.permute.default(view_126, [1, 0])
        sum_dim_int_list_8: "f32[1, 720]" = torch.ops.aten.sum.dim_IntList(view_126, [0], True);  view_126 = None
        reshape_default_6: "f32[720]" = torch.ops.aten.reshape.default(sum_dim_int_list_8, _shape_param_6);  sum_dim_int_list_8 = _shape_param_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        mul_tensor_5: "f32[512, 16, 240]" = torch.ops.aten.mul.Tensor(view_128, mul_239);  mul_239 = None
        sum_dim_int_list_9: "f32[240]" = torch.ops.aten.sum.dim_IntList(mul_tensor_5, [0, 1]);  mul_tensor_5 = None
        sum_dim_int_list_10: "f32[240]" = torch.ops.aten.sum.dim_IntList(view_128, [0, 1]);  view_128 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        permute_default_6: "f32[240, 8192]" = torch.ops.aten.permute.default(view_129, [1, 0])
        sum_dim_int_list_11: "f32[1, 240]" = torch.ops.aten.sum.dim_IntList(view_129, [0], True);  view_129 = None
        reshape_default_7: "f32[240]" = torch.ops.aten.reshape.default(sum_dim_int_list_11, _shape_param_7);  sum_dim_int_list_11 = _shape_param_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        permute_default_7: "f32[480, 8192]" = torch.ops.aten.permute.default(view_132, [1, 0])
        sum_dim_int_list_12: "f32[1, 480]" = torch.ops.aten.sum.dim_IntList(view_132, [0], True);  view_132 = None
        reshape_default_8: "f32[480]" = torch.ops.aten.reshape.default(sum_dim_int_list_12, _shape_param_8);  sum_dim_int_list_12 = _shape_param_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        mul_tensor_6: "f32[512, 16, 240]" = torch.ops.aten.mul.Tensor(view_134, mul_237);  mul_237 = None
        sum_dim_int_list_13: "f32[240]" = torch.ops.aten.sum.dim_IntList(mul_tensor_6, [0, 1]);  mul_tensor_6 = None
        sum_dim_int_list_14: "f32[240]" = torch.ops.aten.sum.dim_IntList(view_134, [0, 1]);  view_134 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        permute_default_8: "f32[240, 8192]" = torch.ops.aten.permute.default(view_135, [1, 0])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_default_9: "f32[512, 16, 4, 60]" = torch.ops.aten.permute.default(getitem_144, [0, 2, 1, 3]);  getitem_144 = None
        reshape_default_9: "f32[512, 16, 240]" = torch.ops.aten.reshape.default(permute_default_9, _shape_param_9);  permute_default_9 = _shape_param_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        reshape_default_10: "f32[8192, 240]" = torch.ops.aten.reshape.default(reshape_default_9, _shape_param_10);  reshape_default_9 = _shape_param_10 = None
        sum_dim_int_list_15: "f32[1, 240]" = torch.ops.aten.sum.dim_IntList(view_135, [0], True);  view_135 = None
        reshape_default_11: "f32[240]" = torch.ops.aten.reshape.default(sum_dim_int_list_15, _shape_param_11);  sum_dim_int_list_15 = _shape_param_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        permute_default_10: "f32[720, 8192]" = torch.ops.aten.permute.default(view_141, [1, 0])
        sum_dim_int_list_16: "f32[1, 720]" = torch.ops.aten.sum.dim_IntList(view_141, [0], True);  view_141 = None
        reshape_default_12: "f32[720]" = torch.ops.aten.reshape.default(sum_dim_int_list_16, _shape_param_12);  sum_dim_int_list_16 = _shape_param_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        mul_tensor_7: "f32[512, 16, 240]" = torch.ops.aten.mul.Tensor(view_143, mul_235);  mul_235 = None
        sum_dim_int_list_17: "f32[240]" = torch.ops.aten.sum.dim_IntList(mul_tensor_7, [0, 1]);  mul_tensor_7 = None
        sum_dim_int_list_18: "f32[240]" = torch.ops.aten.sum.dim_IntList(view_143, [0, 1]);  view_143 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        permute_default_11: "f32[240, 8192]" = torch.ops.aten.permute.default(view_144, [1, 0])
        sum_dim_int_list_19: "f32[1, 240]" = torch.ops.aten.sum.dim_IntList(view_144, [0], True);  view_144 = None
        reshape_default_13: "f32[240]" = torch.ops.aten.reshape.default(sum_dim_int_list_19, _shape_param_13);  sum_dim_int_list_19 = _shape_param_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        permute_default_12: "f32[480, 8192]" = torch.ops.aten.permute.default(view_147, [1, 0])
        sum_dim_int_list_20: "f32[1, 480]" = torch.ops.aten.sum.dim_IntList(view_147, [0], True);  view_147 = None
        reshape_default_14: "f32[480]" = torch.ops.aten.reshape.default(sum_dim_int_list_20, _shape_param_14);  sum_dim_int_list_20 = _shape_param_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        mul_tensor_8: "f32[512, 16, 240]" = torch.ops.aten.mul.Tensor(view_149, mul_233);  mul_233 = None
        sum_dim_int_list_21: "f32[240]" = torch.ops.aten.sum.dim_IntList(mul_tensor_8, [0, 1]);  mul_tensor_8 = None
        sum_dim_int_list_22: "f32[240]" = torch.ops.aten.sum.dim_IntList(view_149, [0, 1]);  view_149 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        permute_default_13: "f32[240, 8192]" = torch.ops.aten.permute.default(view_150, [1, 0])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_default_14: "f32[512, 16, 4, 60]" = torch.ops.aten.permute.default(getitem_133, [0, 2, 1, 3]);  getitem_133 = None
        reshape_default_15: "f32[512, 16, 240]" = torch.ops.aten.reshape.default(permute_default_14, _shape_param_15);  permute_default_14 = _shape_param_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        reshape_default_16: "f32[8192, 240]" = torch.ops.aten.reshape.default(reshape_default_15, _shape_param_16);  reshape_default_15 = _shape_param_16 = None
        sum_dim_int_list_23: "f32[1, 240]" = torch.ops.aten.sum.dim_IntList(view_150, [0], True);  view_150 = None
        reshape_default_17: "f32[240]" = torch.ops.aten.reshape.default(sum_dim_int_list_23, _shape_param_17);  sum_dim_int_list_23 = _shape_param_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        permute_default_15: "f32[720, 8192]" = torch.ops.aten.permute.default(view_156, [1, 0])
        sum_dim_int_list_24: "f32[1, 720]" = torch.ops.aten.sum.dim_IntList(view_156, [0], True);  view_156 = None
        reshape_default_18: "f32[720]" = torch.ops.aten.reshape.default(sum_dim_int_list_24, _shape_param_18);  sum_dim_int_list_24 = _shape_param_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        mul_tensor_9: "f32[512, 16, 240]" = torch.ops.aten.mul.Tensor(view_158, mul_231);  mul_231 = None
        sum_dim_int_list_25: "f32[240]" = torch.ops.aten.sum.dim_IntList(mul_tensor_9, [0, 1]);  mul_tensor_9 = None
        sum_dim_int_list_26: "f32[240]" = torch.ops.aten.sum.dim_IntList(view_158, [0, 1]);  view_158 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        mul_tensor_10: "f32[160]" = torch.ops.aten.mul.Tensor(sum_49, squeeze_85);  sum_49 = squeeze_85 = None
        mul_tensor_11: "f32[160]" = torch.ops.aten.mul.Tensor(sum_51, squeeze_82);  sum_51 = squeeze_82 = None
        mul_tensor_12: "f32[512]" = torch.ops.aten.mul.Tensor(sum_53, squeeze_79);  sum_53 = squeeze_79 = None
        mul_tensor_13: "f32[512]" = torch.ops.aten.mul.Tensor(sum_55, squeeze_76);  sum_55 = squeeze_76 = None
        mul_tensor_14: "f32[128]" = torch.ops.aten.mul.Tensor(sum_57, squeeze_73);  sum_57 = squeeze_73 = None
        mul_tensor_15: "f32[128]" = torch.ops.aten.mul.Tensor(sum_59, squeeze_70);  sum_59 = squeeze_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/mobilevit.py:264 in forward, code: x = self.norm(x)
        mul_tensor_16: "f32[512, 64, 192]" = torch.ops.aten.mul.Tensor(view_164, mul_187);  mul_187 = None
        sum_dim_int_list_27: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_tensor_16, [0, 1]);  mul_tensor_16 = None
        sum_dim_int_list_28: "f32[192]" = torch.ops.aten.sum.dim_IntList(view_164, [0, 1]);  view_164 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        permute_default_16: "f32[192, 32768]" = torch.ops.aten.permute.default(view_165, [1, 0])
        sum_dim_int_list_29: "f32[1, 192]" = torch.ops.aten.sum.dim_IntList(view_165, [0], True);  view_165 = None
        reshape_default_19: "f32[192]" = torch.ops.aten.reshape.default(sum_dim_int_list_29, _shape_param_19);  sum_dim_int_list_29 = _shape_param_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        permute_default_17: "f32[384, 32768]" = torch.ops.aten.permute.default(view_168, [1, 0])
        sum_dim_int_list_30: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_168, [0], True);  view_168 = None
        reshape_default_20: "f32[384]" = torch.ops.aten.reshape.default(sum_dim_int_list_30, _shape_param_20);  sum_dim_int_list_30 = _shape_param_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        mul_tensor_17: "f32[512, 64, 192]" = torch.ops.aten.mul.Tensor(view_170, mul_185);  mul_185 = None
        sum_dim_int_list_31: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_tensor_17, [0, 1]);  mul_tensor_17 = None
        sum_dim_int_list_32: "f32[192]" = torch.ops.aten.sum.dim_IntList(view_170, [0, 1]);  view_170 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        permute_default_18: "f32[192, 32768]" = torch.ops.aten.permute.default(view_171, [1, 0])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_default_19: "f32[512, 64, 4, 48]" = torch.ops.aten.permute.default(getitem_108, [0, 2, 1, 3]);  getitem_108 = None
        reshape_default_21: "f32[512, 64, 192]" = torch.ops.aten.reshape.default(permute_default_19, _shape_param_21);  permute_default_19 = _shape_param_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        reshape_default_22: "f32[32768, 192]" = torch.ops.aten.reshape.default(reshape_default_21, _shape_param_22);  reshape_default_21 = _shape_param_22 = None
        sum_dim_int_list_33: "f32[1, 192]" = torch.ops.aten.sum.dim_IntList(view_171, [0], True);  view_171 = None
        reshape_default_23: "f32[192]" = torch.ops.aten.reshape.default(sum_dim_int_list_33, _shape_param_23);  sum_dim_int_list_33 = _shape_param_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        permute_default_20: "f32[576, 32768]" = torch.ops.aten.permute.default(view_177, [1, 0])
        sum_dim_int_list_34: "f32[1, 576]" = torch.ops.aten.sum.dim_IntList(view_177, [0], True);  view_177 = None
        reshape_default_24: "f32[576]" = torch.ops.aten.reshape.default(sum_dim_int_list_34, _shape_param_24);  sum_dim_int_list_34 = _shape_param_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        mul_tensor_18: "f32[512, 64, 192]" = torch.ops.aten.mul.Tensor(view_179, mul_183);  mul_183 = None
        sum_dim_int_list_35: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_tensor_18, [0, 1]);  mul_tensor_18 = None
        sum_dim_int_list_36: "f32[192]" = torch.ops.aten.sum.dim_IntList(view_179, [0, 1]);  view_179 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        permute_default_21: "f32[192, 32768]" = torch.ops.aten.permute.default(view_180, [1, 0])
        sum_dim_int_list_37: "f32[1, 192]" = torch.ops.aten.sum.dim_IntList(view_180, [0], True);  view_180 = None
        reshape_default_25: "f32[192]" = torch.ops.aten.reshape.default(sum_dim_int_list_37, _shape_param_25);  sum_dim_int_list_37 = _shape_param_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        permute_default_22: "f32[384, 32768]" = torch.ops.aten.permute.default(view_183, [1, 0])
        sum_dim_int_list_38: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_183, [0], True);  view_183 = None
        reshape_default_26: "f32[384]" = torch.ops.aten.reshape.default(sum_dim_int_list_38, _shape_param_26);  sum_dim_int_list_38 = _shape_param_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        mul_tensor_19: "f32[512, 64, 192]" = torch.ops.aten.mul.Tensor(view_185, mul_181);  mul_181 = None
        sum_dim_int_list_39: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_tensor_19, [0, 1]);  mul_tensor_19 = None
        sum_dim_int_list_40: "f32[192]" = torch.ops.aten.sum.dim_IntList(view_185, [0, 1]);  view_185 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        permute_default_23: "f32[192, 32768]" = torch.ops.aten.permute.default(view_186, [1, 0])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_default_24: "f32[512, 64, 4, 48]" = torch.ops.aten.permute.default(getitem_97, [0, 2, 1, 3]);  getitem_97 = None
        reshape_default_27: "f32[512, 64, 192]" = torch.ops.aten.reshape.default(permute_default_24, _shape_param_27);  permute_default_24 = _shape_param_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        reshape_default_28: "f32[32768, 192]" = torch.ops.aten.reshape.default(reshape_default_27, _shape_param_28);  reshape_default_27 = _shape_param_28 = None
        sum_dim_int_list_41: "f32[1, 192]" = torch.ops.aten.sum.dim_IntList(view_186, [0], True);  view_186 = None
        reshape_default_29: "f32[192]" = torch.ops.aten.reshape.default(sum_dim_int_list_41, _shape_param_29);  sum_dim_int_list_41 = _shape_param_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        permute_default_25: "f32[576, 32768]" = torch.ops.aten.permute.default(view_192, [1, 0])
        sum_dim_int_list_42: "f32[1, 576]" = torch.ops.aten.sum.dim_IntList(view_192, [0], True);  view_192 = None
        reshape_default_30: "f32[576]" = torch.ops.aten.reshape.default(sum_dim_int_list_42, _shape_param_30);  sum_dim_int_list_42 = _shape_param_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        mul_tensor_20: "f32[512, 64, 192]" = torch.ops.aten.mul.Tensor(view_194, mul_179);  mul_179 = None
        sum_dim_int_list_43: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_tensor_20, [0, 1]);  mul_tensor_20 = None
        sum_dim_int_list_44: "f32[192]" = torch.ops.aten.sum.dim_IntList(view_194, [0, 1]);  view_194 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        permute_default_26: "f32[192, 32768]" = torch.ops.aten.permute.default(view_195, [1, 0])
        sum_dim_int_list_45: "f32[1, 192]" = torch.ops.aten.sum.dim_IntList(view_195, [0], True);  view_195 = None
        reshape_default_31: "f32[192]" = torch.ops.aten.reshape.default(sum_dim_int_list_45, _shape_param_31);  sum_dim_int_list_45 = _shape_param_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        permute_default_27: "f32[384, 32768]" = torch.ops.aten.permute.default(view_198, [1, 0])
        sum_dim_int_list_46: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_198, [0], True);  view_198 = None
        reshape_default_32: "f32[384]" = torch.ops.aten.reshape.default(sum_dim_int_list_46, _shape_param_32);  sum_dim_int_list_46 = _shape_param_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        mul_tensor_21: "f32[512, 64, 192]" = torch.ops.aten.mul.Tensor(view_200, mul_177);  mul_177 = None
        sum_dim_int_list_47: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_tensor_21, [0, 1]);  mul_tensor_21 = None
        sum_dim_int_list_48: "f32[192]" = torch.ops.aten.sum.dim_IntList(view_200, [0, 1]);  view_200 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        permute_default_28: "f32[192, 32768]" = torch.ops.aten.permute.default(view_201, [1, 0])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_default_29: "f32[512, 64, 4, 48]" = torch.ops.aten.permute.default(getitem_86, [0, 2, 1, 3]);  getitem_86 = None
        reshape_default_33: "f32[512, 64, 192]" = torch.ops.aten.reshape.default(permute_default_29, _shape_param_33);  permute_default_29 = _shape_param_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        reshape_default_34: "f32[32768, 192]" = torch.ops.aten.reshape.default(reshape_default_33, _shape_param_34);  reshape_default_33 = _shape_param_34 = None
        sum_dim_int_list_49: "f32[1, 192]" = torch.ops.aten.sum.dim_IntList(view_201, [0], True);  view_201 = None
        reshape_default_35: "f32[192]" = torch.ops.aten.reshape.default(sum_dim_int_list_49, _shape_param_35);  sum_dim_int_list_49 = _shape_param_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        permute_default_30: "f32[576, 32768]" = torch.ops.aten.permute.default(view_207, [1, 0])
        sum_dim_int_list_50: "f32[1, 576]" = torch.ops.aten.sum.dim_IntList(view_207, [0], True);  view_207 = None
        reshape_default_36: "f32[576]" = torch.ops.aten.reshape.default(sum_dim_int_list_50, _shape_param_36);  sum_dim_int_list_50 = _shape_param_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        mul_tensor_22: "f32[512, 64, 192]" = torch.ops.aten.mul.Tensor(view_209, mul_175);  mul_175 = None
        sum_dim_int_list_51: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_tensor_22, [0, 1]);  mul_tensor_22 = None
        sum_dim_int_list_52: "f32[192]" = torch.ops.aten.sum.dim_IntList(view_209, [0, 1]);  view_209 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        permute_default_31: "f32[192, 32768]" = torch.ops.aten.permute.default(view_210, [1, 0])
        sum_dim_int_list_53: "f32[1, 192]" = torch.ops.aten.sum.dim_IntList(view_210, [0], True);  view_210 = None
        reshape_default_37: "f32[192]" = torch.ops.aten.reshape.default(sum_dim_int_list_53, _shape_param_37);  sum_dim_int_list_53 = _shape_param_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        permute_default_32: "f32[384, 32768]" = torch.ops.aten.permute.default(view_213, [1, 0])
        sum_dim_int_list_54: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_213, [0], True);  view_213 = None
        reshape_default_38: "f32[384]" = torch.ops.aten.reshape.default(sum_dim_int_list_54, _shape_param_38);  sum_dim_int_list_54 = _shape_param_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        mul_tensor_23: "f32[512, 64, 192]" = torch.ops.aten.mul.Tensor(view_215, mul_173);  mul_173 = None
        sum_dim_int_list_55: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_tensor_23, [0, 1]);  mul_tensor_23 = None
        sum_dim_int_list_56: "f32[192]" = torch.ops.aten.sum.dim_IntList(view_215, [0, 1]);  view_215 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        permute_default_33: "f32[192, 32768]" = torch.ops.aten.permute.default(view_216, [1, 0])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_default_34: "f32[512, 64, 4, 48]" = torch.ops.aten.permute.default(getitem_75, [0, 2, 1, 3]);  getitem_75 = None
        reshape_default_39: "f32[512, 64, 192]" = torch.ops.aten.reshape.default(permute_default_34, _shape_param_39);  permute_default_34 = _shape_param_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        reshape_default_40: "f32[32768, 192]" = torch.ops.aten.reshape.default(reshape_default_39, _shape_param_40);  reshape_default_39 = _shape_param_40 = None
        sum_dim_int_list_57: "f32[1, 192]" = torch.ops.aten.sum.dim_IntList(view_216, [0], True);  view_216 = None
        reshape_default_41: "f32[192]" = torch.ops.aten.reshape.default(sum_dim_int_list_57, _shape_param_41);  sum_dim_int_list_57 = _shape_param_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        permute_default_35: "f32[576, 32768]" = torch.ops.aten.permute.default(view_222, [1, 0])
        sum_dim_int_list_58: "f32[1, 576]" = torch.ops.aten.sum.dim_IntList(view_222, [0], True);  view_222 = None
        reshape_default_42: "f32[576]" = torch.ops.aten.reshape.default(sum_dim_int_list_58, _shape_param_42);  sum_dim_int_list_58 = _shape_param_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        mul_tensor_24: "f32[512, 64, 192]" = torch.ops.aten.mul.Tensor(view_224, mul_171);  mul_171 = None
        sum_dim_int_list_59: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_tensor_24, [0, 1]);  mul_tensor_24 = None
        sum_dim_int_list_60: "f32[192]" = torch.ops.aten.sum.dim_IntList(view_224, [0, 1]);  view_224 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        mul_tensor_25: "f32[128]" = torch.ops.aten.mul.Tensor(sum_113, squeeze_67);  sum_113 = squeeze_67 = None
        mul_tensor_26: "f32[128]" = torch.ops.aten.mul.Tensor(sum_115, squeeze_64);  sum_115 = squeeze_64 = None
        mul_tensor_27: "f32[384]" = torch.ops.aten.mul.Tensor(sum_117, squeeze_61);  sum_117 = squeeze_61 = None
        mul_tensor_28: "f32[384]" = torch.ops.aten.mul.Tensor(sum_119, squeeze_58);  sum_119 = squeeze_58 = None
        mul_tensor_29: "f32[96]" = torch.ops.aten.mul.Tensor(sum_121, squeeze_55);  sum_121 = squeeze_55 = None
        mul_tensor_30: "f32[96]" = torch.ops.aten.mul.Tensor(sum_123, squeeze_52);  sum_123 = squeeze_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/mobilevit.py:264 in forward, code: x = self.norm(x)
        mul_tensor_31: "f32[512, 256, 144]" = torch.ops.aten.mul.Tensor(view_230, mul_127);  mul_127 = None
        sum_dim_int_list_61: "f32[144]" = torch.ops.aten.sum.dim_IntList(mul_tensor_31, [0, 1]);  mul_tensor_31 = None
        sum_dim_int_list_62: "f32[144]" = torch.ops.aten.sum.dim_IntList(view_230, [0, 1]);  view_230 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        permute_default_36: "f32[144, 131072]" = torch.ops.aten.permute.default(view_231, [1, 0])
        sum_dim_int_list_63: "f32[1, 144]" = torch.ops.aten.sum.dim_IntList(view_231, [0], True);  view_231 = None
        reshape_default_43: "f32[144]" = torch.ops.aten.reshape.default(sum_dim_int_list_63, _shape_param_43);  sum_dim_int_list_63 = _shape_param_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        permute_default_37: "f32[288, 131072]" = torch.ops.aten.permute.default(view_234, [1, 0])
        sum_dim_int_list_64: "f32[1, 288]" = torch.ops.aten.sum.dim_IntList(view_234, [0], True);  view_234 = None
        reshape_default_44: "f32[288]" = torch.ops.aten.reshape.default(sum_dim_int_list_64, _shape_param_44);  sum_dim_int_list_64 = _shape_param_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        mul_tensor_32: "f32[512, 256, 144]" = torch.ops.aten.mul.Tensor(view_236, mul_125);  mul_125 = None
        sum_dim_int_list_65: "f32[144]" = torch.ops.aten.sum.dim_IntList(mul_tensor_32, [0, 1]);  mul_tensor_32 = None
        sum_dim_int_list_66: "f32[144]" = torch.ops.aten.sum.dim_IntList(view_236, [0, 1]);  view_236 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        permute_default_38: "f32[144, 131072]" = torch.ops.aten.permute.default(view_237, [1, 0])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_default_39: "f32[512, 256, 4, 36]" = torch.ops.aten.permute.default(getitem_50, [0, 2, 1, 3]);  getitem_50 = None
        reshape_default_45: "f32[512, 256, 144]" = torch.ops.aten.reshape.default(permute_default_39, _shape_param_45);  permute_default_39 = _shape_param_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        reshape_default_46: "f32[131072, 144]" = torch.ops.aten.reshape.default(reshape_default_45, _shape_param_46);  reshape_default_45 = _shape_param_46 = None
        sum_dim_int_list_67: "f32[1, 144]" = torch.ops.aten.sum.dim_IntList(view_237, [0], True);  view_237 = None
        reshape_default_47: "f32[144]" = torch.ops.aten.reshape.default(sum_dim_int_list_67, _shape_param_47);  sum_dim_int_list_67 = _shape_param_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        permute_default_40: "f32[432, 131072]" = torch.ops.aten.permute.default(view_243, [1, 0])
        sum_dim_int_list_68: "f32[1, 432]" = torch.ops.aten.sum.dim_IntList(view_243, [0], True);  view_243 = None
        reshape_default_48: "f32[432]" = torch.ops.aten.reshape.default(sum_dim_int_list_68, _shape_param_48);  sum_dim_int_list_68 = _shape_param_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        mul_tensor_33: "f32[512, 256, 144]" = torch.ops.aten.mul.Tensor(view_245, mul_123);  mul_123 = None
        sum_dim_int_list_69: "f32[144]" = torch.ops.aten.sum.dim_IntList(mul_tensor_33, [0, 1]);  mul_tensor_33 = None
        sum_dim_int_list_70: "f32[144]" = torch.ops.aten.sum.dim_IntList(view_245, [0, 1]);  view_245 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        permute_default_41: "f32[144, 131072]" = torch.ops.aten.permute.default(view_246, [1, 0])
        sum_dim_int_list_71: "f32[1, 144]" = torch.ops.aten.sum.dim_IntList(view_246, [0], True);  view_246 = None
        reshape_default_49: "f32[144]" = torch.ops.aten.reshape.default(sum_dim_int_list_71, _shape_param_49);  sum_dim_int_list_71 = _shape_param_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        permute_default_42: "f32[288, 131072]" = torch.ops.aten.permute.default(view_249, [1, 0])
        sum_dim_int_list_72: "f32[1, 288]" = torch.ops.aten.sum.dim_IntList(view_249, [0], True);  view_249 = None
        reshape_default_50: "f32[288]" = torch.ops.aten.reshape.default(sum_dim_int_list_72, _shape_param_50);  sum_dim_int_list_72 = _shape_param_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        mul_tensor_34: "f32[512, 256, 144]" = torch.ops.aten.mul.Tensor(view_251, mul_121);  mul_121 = None
        sum_dim_int_list_73: "f32[144]" = torch.ops.aten.sum.dim_IntList(mul_tensor_34, [0, 1]);  mul_tensor_34 = None
        sum_dim_int_list_74: "f32[144]" = torch.ops.aten.sum.dim_IntList(view_251, [0, 1]);  view_251 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        permute_default_43: "f32[144, 131072]" = torch.ops.aten.permute.default(view_252, [1, 0])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_default_44: "f32[512, 256, 4, 36]" = torch.ops.aten.permute.default(getitem_39, [0, 2, 1, 3]);  getitem_39 = None
        reshape_default_51: "f32[512, 256, 144]" = torch.ops.aten.reshape.default(permute_default_44, _shape_param_51);  permute_default_44 = _shape_param_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        reshape_default_52: "f32[131072, 144]" = torch.ops.aten.reshape.default(reshape_default_51, _shape_param_52);  reshape_default_51 = _shape_param_52 = None
        sum_dim_int_list_75: "f32[1, 144]" = torch.ops.aten.sum.dim_IntList(view_252, [0], True);  view_252 = None
        reshape_default_53: "f32[144]" = torch.ops.aten.reshape.default(sum_dim_int_list_75, _shape_param_53);  sum_dim_int_list_75 = _shape_param_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        permute_default_45: "f32[432, 131072]" = torch.ops.aten.permute.default(view_258, [1, 0])
        sum_dim_int_list_76: "f32[1, 432]" = torch.ops.aten.sum.dim_IntList(view_258, [0], True);  view_258 = None
        reshape_default_54: "f32[432]" = torch.ops.aten.reshape.default(sum_dim_int_list_76, _shape_param_54);  sum_dim_int_list_76 = _shape_param_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        mul_tensor_35: "f32[512, 256, 144]" = torch.ops.aten.mul.Tensor(view_260, mul_119);  mul_119 = None
        sum_dim_int_list_77: "f32[144]" = torch.ops.aten.sum.dim_IntList(mul_tensor_35, [0, 1]);  mul_tensor_35 = None
        sum_dim_int_list_78: "f32[144]" = torch.ops.aten.sum.dim_IntList(view_260, [0, 1]);  view_260 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        mul_tensor_36: "f32[96]" = torch.ops.aten.mul.Tensor(sum_153, squeeze_49);  sum_153 = squeeze_49 = None
        mul_tensor_37: "f32[96]" = torch.ops.aten.mul.Tensor(sum_155, squeeze_46);  sum_155 = squeeze_46 = None
        mul_tensor_38: "f32[256]" = torch.ops.aten.mul.Tensor(sum_157, squeeze_43);  sum_157 = squeeze_43 = None
        mul_tensor_39: "f32[256]" = torch.ops.aten.mul.Tensor(sum_159, squeeze_40);  sum_159 = squeeze_40 = None
        mul_tensor_40: "f32[64]" = torch.ops.aten.mul.Tensor(sum_161, squeeze_37);  sum_161 = squeeze_37 = None
        mul_tensor_41: "f32[256]" = torch.ops.aten.mul.Tensor(sum_163, squeeze_34);  sum_163 = squeeze_34 = None
        mul_tensor_42: "f32[256]" = torch.ops.aten.mul.Tensor(sum_165, squeeze_31);  sum_165 = squeeze_31 = None
        mul_tensor_43: "f32[64]" = torch.ops.aten.mul.Tensor(sum_167, squeeze_28);  sum_167 = squeeze_28 = None
        mul_tensor_44: "f32[256]" = torch.ops.aten.mul.Tensor(sum_169, squeeze_25);  sum_169 = squeeze_25 = None
        mul_tensor_45: "f32[256]" = torch.ops.aten.mul.Tensor(sum_171, squeeze_22);  sum_171 = squeeze_22 = None
        mul_tensor_46: "f32[64]" = torch.ops.aten.mul.Tensor(sum_173, squeeze_19);  sum_173 = squeeze_19 = None
        mul_tensor_47: "f32[128]" = torch.ops.aten.mul.Tensor(sum_175, squeeze_16);  sum_175 = squeeze_16 = None
        mul_tensor_48: "f32[128]" = torch.ops.aten.mul.Tensor(sum_177, squeeze_13);  sum_177 = squeeze_13 = None
        mul_tensor_49: "f32[32]" = torch.ops.aten.mul.Tensor(sum_179, squeeze_10);  sum_179 = squeeze_10 = None
        mul_tensor_50: "f32[64]" = torch.ops.aten.mul.Tensor(sum_181, squeeze_7);  sum_181 = squeeze_7 = None
        mul_tensor_51: "f32[64]" = torch.ops.aten.mul.Tensor(sum_183, squeeze_4);  sum_183 = squeeze_4 = None
        mul_tensor_52: "f32[16]" = torch.ops.aten.mul.Tensor(sum_185, squeeze_1);  sum_185 = squeeze_1 = None
        return (permute_default, reshape_default, mul_tensor, mul_tensor_1, mul_tensor_2, sum_dim_int_list_1, sum_dim_int_list_2, permute_default_1, reshape_default_1, permute_default_2, reshape_default_2, sum_dim_int_list_5, sum_dim_int_list_6, permute_default_3, reshape_default_4, reshape_default_5, permute_default_5, reshape_default_6, sum_dim_int_list_9, sum_dim_int_list_10, permute_default_6, reshape_default_7, permute_default_7, reshape_default_8, sum_dim_int_list_13, sum_dim_int_list_14, permute_default_8, reshape_default_10, reshape_default_11, permute_default_10, reshape_default_12, sum_dim_int_list_17, sum_dim_int_list_18, permute_default_11, reshape_default_13, permute_default_12, reshape_default_14, sum_dim_int_list_21, sum_dim_int_list_22, permute_default_13, reshape_default_16, reshape_default_17, permute_default_15, reshape_default_18, sum_dim_int_list_25, sum_dim_int_list_26, mul_tensor_10, mul_tensor_11, mul_tensor_12, mul_tensor_13, mul_tensor_14, mul_tensor_15, sum_dim_int_list_27, sum_dim_int_list_28, permute_default_16, reshape_default_19, permute_default_17, reshape_default_20, sum_dim_int_list_31, sum_dim_int_list_32, permute_default_18, reshape_default_22, reshape_default_23, permute_default_20, reshape_default_24, sum_dim_int_list_35, sum_dim_int_list_36, permute_default_21, reshape_default_25, permute_default_22, reshape_default_26, sum_dim_int_list_39, sum_dim_int_list_40, permute_default_23, reshape_default_28, reshape_default_29, permute_default_25, reshape_default_30, sum_dim_int_list_43, sum_dim_int_list_44, permute_default_26, reshape_default_31, permute_default_27, reshape_default_32, sum_dim_int_list_47, sum_dim_int_list_48, permute_default_28, reshape_default_34, reshape_default_35, permute_default_30, reshape_default_36, sum_dim_int_list_51, sum_dim_int_list_52, permute_default_31, reshape_default_37, permute_default_32, reshape_default_38, sum_dim_int_list_55, sum_dim_int_list_56, permute_default_33, reshape_default_40, reshape_default_41, permute_default_35, reshape_default_42, sum_dim_int_list_59, sum_dim_int_list_60, mul_tensor_25, mul_tensor_26, mul_tensor_27, mul_tensor_28, mul_tensor_29, mul_tensor_30, sum_dim_int_list_61, sum_dim_int_list_62, permute_default_36, reshape_default_43, permute_default_37, reshape_default_44, sum_dim_int_list_65, sum_dim_int_list_66, permute_default_38, reshape_default_46, reshape_default_47, permute_default_40, reshape_default_48, sum_dim_int_list_69, sum_dim_int_list_70, permute_default_41, reshape_default_49, permute_default_42, reshape_default_50, sum_dim_int_list_73, sum_dim_int_list_74, permute_default_43, reshape_default_52, reshape_default_53, permute_default_45, reshape_default_54, sum_dim_int_list_77, sum_dim_int_list_78, mul_tensor_36, mul_tensor_37, mul_tensor_38, mul_tensor_39, mul_tensor_40, mul_tensor_41, mul_tensor_42, mul_tensor_43, mul_tensor_44, mul_tensor_45, mul_tensor_46, mul_tensor_47, mul_tensor_48, mul_tensor_49, mul_tensor_50, mul_tensor_51, mul_tensor_52)
