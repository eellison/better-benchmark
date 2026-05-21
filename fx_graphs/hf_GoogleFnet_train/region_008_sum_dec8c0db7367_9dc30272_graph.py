class GraphModule(torch.nn.Module):
    def forward(self, view_57: "f32[16384, 32000]", view_59: "f32[32, 512, 768]", mul_128: "f32[32, 512, 768]", view_60: "f32[16384, 768]", view_62: "f32[32, 512, 768]", mul_122: "f32[32, 512, 768]", view_63: "f32[16384, 768]", view_66: "f32[16384, 3072]", add_109: "f32[32, 512, 768]", mul_114: "f32[32, 512, 768]", add_110: "f32[32, 512, 768]", mul_112: "f32[32, 512, 768]", view_69: "f32[16384, 768]", view_72: "f32[16384, 3072]", add_113: "f32[32, 512, 768]", mul_104: "f32[32, 512, 768]", add_114: "f32[32, 512, 768]", mul_102: "f32[32, 512, 768]", view_75: "f32[16384, 768]", view_78: "f32[16384, 3072]", add_117: "f32[32, 512, 768]", mul_94: "f32[32, 512, 768]", add_118: "f32[32, 512, 768]", mul_92: "f32[32, 512, 768]", view_81: "f32[16384, 768]", view_84: "f32[16384, 3072]", add_121: "f32[32, 512, 768]", mul_84: "f32[32, 512, 768]", add_122: "f32[32, 512, 768]", mul_82: "f32[32, 512, 768]", view_87: "f32[16384, 768]", view_90: "f32[16384, 3072]", add_125: "f32[32, 512, 768]", mul_74: "f32[32, 512, 768]", add_126: "f32[32, 512, 768]", mul_72: "f32[32, 512, 768]", view_93: "f32[16384, 768]", view_96: "f32[16384, 3072]", add_129: "f32[32, 512, 768]", mul_64: "f32[32, 512, 768]", add_130: "f32[32, 512, 768]", mul_62: "f32[32, 512, 768]", view_99: "f32[16384, 768]", view_102: "f32[16384, 3072]", add_133: "f32[32, 512, 768]", mul_54: "f32[32, 512, 768]", add_134: "f32[32, 512, 768]", mul_52: "f32[32, 512, 768]", view_105: "f32[16384, 768]", view_108: "f32[16384, 3072]", add_137: "f32[32, 512, 768]", mul_44: "f32[32, 512, 768]", add_138: "f32[32, 512, 768]", mul_42: "f32[32, 512, 768]", view_111: "f32[16384, 768]", view_114: "f32[16384, 3072]", add_141: "f32[32, 512, 768]", mul_34: "f32[32, 512, 768]", add_142: "f32[32, 512, 768]", mul_32: "f32[32, 512, 768]", view_117: "f32[16384, 768]", view_120: "f32[16384, 3072]", add_145: "f32[32, 512, 768]", mul_24: "f32[32, 512, 768]", add_146: "f32[32, 512, 768]", mul_22: "f32[32, 512, 768]", view_123: "f32[16384, 768]", view_126: "f32[16384, 3072]", add_149: "f32[32, 512, 768]", mul_14: "f32[32, 512, 768]", add_150: "f32[32, 512, 768]", mul_12: "f32[32, 512, 768]", view_129: "f32[16384, 768]", view_132: "f32[16384, 3072]", add_153: "f32[32, 512, 768]", mul_4: "f32[32, 512, 768]", view_135: "f32[16384, 768]", mm_52: "f32[16384, 768]", primals_7: "f32[768]", mul: "f32[32, 512, 768]", div_27: "f32[32, 512, 1]", primals_3: "i64[1, 512]", full_default_1: "f32[]", primals_2: "i64[1, 512]", primals_1: "i64[32, 512]", mm_1: "f32[32000, 768]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12, _shape_param_13, _shape_param_14, _shape_param_15, _shape_param_16, _shape_param_17, _shape_param_18, _shape_param_19, _shape_param_20, _shape_param_21, _shape_param_22, _shape_param_23, _shape_param_24, _shape_param_25, _shape_param_26, _shape_param_27, _shape_param_28):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:331 in forward, code: hidden_states = self.decoder(hidden_states)
        sum_dim_int_list: "f32[1, 32000]" = torch.ops.aten.sum.dim_IntList(view_57, [0], True);  view_57 = None
        reshape_default: "f32[32000]" = torch.ops.aten.reshape.default(sum_dim_int_list, _shape_param_0);  sum_dim_int_list = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:318 in forward, code: hidden_states = self.LayerNorm(hidden_states)
        mul_tensor: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(view_59, mul_128);  mul_128 = None
        sum_dim_int_list_1: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [0, 1]);  mul_tensor = None
        sum_dim_int_list_2: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_59, [0, 1]);  view_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:316 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default: "f32[768, 16384]" = torch.ops.aten.permute.default(view_60, [1, 0])
        sum_dim_int_list_3: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_60, [0], True);  view_60 = None
        reshape_default_1: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_3, _shape_param_1);  sum_dim_int_list_3 = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:230 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_tensor_1: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(view_62, mul_122);  mul_122 = None
        sum_dim_int_list_4: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_1, [0, 1]);  mul_tensor_1 = None
        sum_dim_int_list_5: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_62, [0, 1]);  view_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:228 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_1: "f32[768, 16384]" = torch.ops.aten.permute.default(view_63, [1, 0])
        sum_dim_int_list_6: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_63, [0], True);  view_63 = None
        reshape_default_2: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_6, _shape_param_2);  sum_dim_int_list_6 = _shape_param_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:214 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_2: "f32[3072, 16384]" = torch.ops.aten.permute.default(view_66, [1, 0])
        sum_dim_int_list_7: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_66, [0], True);  view_66 = None
        reshape_default_3: "f32[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_7, _shape_param_3);  sum_dim_int_list_7 = _shape_param_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:186 in forward, code: hidden_states = self.LayerNorm(input_tensor + hidden_states)
        mul_tensor_2: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_109, mul_114);  mul_114 = None
        sum_dim_int_list_8: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [0, 1]);  mul_tensor_2 = None
        sum_dim_int_list_9: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_109, [0, 1]);  add_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:230 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_tensor_3: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_110, mul_112);  mul_112 = None
        sum_dim_int_list_10: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_3, [0, 1]);  mul_tensor_3 = None
        sum_dim_int_list_11: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_110, [0, 1]);  add_110 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:228 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_3: "f32[768, 16384]" = torch.ops.aten.permute.default(view_69, [1, 0])
        sum_dim_int_list_12: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_69, [0], True);  view_69 = None
        reshape_default_4: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_12, _shape_param_4);  sum_dim_int_list_12 = _shape_param_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:214 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_4: "f32[3072, 16384]" = torch.ops.aten.permute.default(view_72, [1, 0])
        sum_dim_int_list_13: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_72, [0], True);  view_72 = None
        reshape_default_5: "f32[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_13, _shape_param_5);  sum_dim_int_list_13 = _shape_param_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:186 in forward, code: hidden_states = self.LayerNorm(input_tensor + hidden_states)
        mul_tensor_4: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_113, mul_104);  mul_104 = None
        sum_dim_int_list_14: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_4, [0, 1]);  mul_tensor_4 = None
        sum_dim_int_list_15: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_113, [0, 1]);  add_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:230 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_tensor_5: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_114, mul_102);  mul_102 = None
        sum_dim_int_list_16: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_5, [0, 1]);  mul_tensor_5 = None
        sum_dim_int_list_17: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_114, [0, 1]);  add_114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:228 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_5: "f32[768, 16384]" = torch.ops.aten.permute.default(view_75, [1, 0])
        sum_dim_int_list_18: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_75, [0], True);  view_75 = None
        reshape_default_6: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_18, _shape_param_6);  sum_dim_int_list_18 = _shape_param_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:214 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_6: "f32[3072, 16384]" = torch.ops.aten.permute.default(view_78, [1, 0])
        sum_dim_int_list_19: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_78, [0], True);  view_78 = None
        reshape_default_7: "f32[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_19, _shape_param_7);  sum_dim_int_list_19 = _shape_param_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:186 in forward, code: hidden_states = self.LayerNorm(input_tensor + hidden_states)
        mul_tensor_6: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_117, mul_94);  mul_94 = None
        sum_dim_int_list_20: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_6, [0, 1]);  mul_tensor_6 = None
        sum_dim_int_list_21: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_117, [0, 1]);  add_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:230 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_tensor_7: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_118, mul_92);  mul_92 = None
        sum_dim_int_list_22: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_7, [0, 1]);  mul_tensor_7 = None
        sum_dim_int_list_23: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_118, [0, 1]);  add_118 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:228 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_7: "f32[768, 16384]" = torch.ops.aten.permute.default(view_81, [1, 0])
        sum_dim_int_list_24: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_81, [0], True);  view_81 = None
        reshape_default_8: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_24, _shape_param_8);  sum_dim_int_list_24 = _shape_param_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:214 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_8: "f32[3072, 16384]" = torch.ops.aten.permute.default(view_84, [1, 0])
        sum_dim_int_list_25: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_84, [0], True);  view_84 = None
        reshape_default_9: "f32[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_25, _shape_param_9);  sum_dim_int_list_25 = _shape_param_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:186 in forward, code: hidden_states = self.LayerNorm(input_tensor + hidden_states)
        mul_tensor_8: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_121, mul_84);  mul_84 = None
        sum_dim_int_list_26: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_8, [0, 1]);  mul_tensor_8 = None
        sum_dim_int_list_27: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_121, [0, 1]);  add_121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:230 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_tensor_9: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_122, mul_82);  mul_82 = None
        sum_dim_int_list_28: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_9, [0, 1]);  mul_tensor_9 = None
        sum_dim_int_list_29: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_122, [0, 1]);  add_122 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:228 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_9: "f32[768, 16384]" = torch.ops.aten.permute.default(view_87, [1, 0])
        sum_dim_int_list_30: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_87, [0], True);  view_87 = None
        reshape_default_10: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_30, _shape_param_10);  sum_dim_int_list_30 = _shape_param_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:214 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_10: "f32[3072, 16384]" = torch.ops.aten.permute.default(view_90, [1, 0])
        sum_dim_int_list_31: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_90, [0], True);  view_90 = None
        reshape_default_11: "f32[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_31, _shape_param_11);  sum_dim_int_list_31 = _shape_param_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:186 in forward, code: hidden_states = self.LayerNorm(input_tensor + hidden_states)
        mul_tensor_10: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_125, mul_74);  mul_74 = None
        sum_dim_int_list_32: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_10, [0, 1]);  mul_tensor_10 = None
        sum_dim_int_list_33: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_125, [0, 1]);  add_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:230 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_tensor_11: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_126, mul_72);  mul_72 = None
        sum_dim_int_list_34: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_11, [0, 1]);  mul_tensor_11 = None
        sum_dim_int_list_35: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_126, [0, 1]);  add_126 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:228 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_11: "f32[768, 16384]" = torch.ops.aten.permute.default(view_93, [1, 0])
        sum_dim_int_list_36: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_93, [0], True);  view_93 = None
        reshape_default_12: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_36, _shape_param_12);  sum_dim_int_list_36 = _shape_param_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:214 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_12: "f32[3072, 16384]" = torch.ops.aten.permute.default(view_96, [1, 0])
        sum_dim_int_list_37: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_96, [0], True);  view_96 = None
        reshape_default_13: "f32[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_37, _shape_param_13);  sum_dim_int_list_37 = _shape_param_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:186 in forward, code: hidden_states = self.LayerNorm(input_tensor + hidden_states)
        mul_tensor_12: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_129, mul_64);  mul_64 = None
        sum_dim_int_list_38: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_12, [0, 1]);  mul_tensor_12 = None
        sum_dim_int_list_39: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_129, [0, 1]);  add_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:230 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_tensor_13: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_130, mul_62);  mul_62 = None
        sum_dim_int_list_40: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_13, [0, 1]);  mul_tensor_13 = None
        sum_dim_int_list_41: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_130, [0, 1]);  add_130 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:228 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_13: "f32[768, 16384]" = torch.ops.aten.permute.default(view_99, [1, 0])
        sum_dim_int_list_42: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_99, [0], True);  view_99 = None
        reshape_default_14: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_42, _shape_param_14);  sum_dim_int_list_42 = _shape_param_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:214 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_14: "f32[3072, 16384]" = torch.ops.aten.permute.default(view_102, [1, 0])
        sum_dim_int_list_43: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_102, [0], True);  view_102 = None
        reshape_default_15: "f32[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_43, _shape_param_15);  sum_dim_int_list_43 = _shape_param_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:186 in forward, code: hidden_states = self.LayerNorm(input_tensor + hidden_states)
        mul_tensor_14: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_133, mul_54);  mul_54 = None
        sum_dim_int_list_44: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_14, [0, 1]);  mul_tensor_14 = None
        sum_dim_int_list_45: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_133, [0, 1]);  add_133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:230 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_tensor_15: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_134, mul_52);  mul_52 = None
        sum_dim_int_list_46: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_15, [0, 1]);  mul_tensor_15 = None
        sum_dim_int_list_47: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_134, [0, 1]);  add_134 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:228 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_15: "f32[768, 16384]" = torch.ops.aten.permute.default(view_105, [1, 0])
        sum_dim_int_list_48: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_105, [0], True);  view_105 = None
        reshape_default_16: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_48, _shape_param_16);  sum_dim_int_list_48 = _shape_param_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:214 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_16: "f32[3072, 16384]" = torch.ops.aten.permute.default(view_108, [1, 0])
        sum_dim_int_list_49: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_108, [0], True);  view_108 = None
        reshape_default_17: "f32[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_49, _shape_param_17);  sum_dim_int_list_49 = _shape_param_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:186 in forward, code: hidden_states = self.LayerNorm(input_tensor + hidden_states)
        mul_tensor_16: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_137, mul_44);  mul_44 = None
        sum_dim_int_list_50: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_16, [0, 1]);  mul_tensor_16 = None
        sum_dim_int_list_51: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_137, [0, 1]);  add_137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:230 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_tensor_17: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_138, mul_42);  mul_42 = None
        sum_dim_int_list_52: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_17, [0, 1]);  mul_tensor_17 = None
        sum_dim_int_list_53: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_138, [0, 1]);  add_138 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:228 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_17: "f32[768, 16384]" = torch.ops.aten.permute.default(view_111, [1, 0])
        sum_dim_int_list_54: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_111, [0], True);  view_111 = None
        reshape_default_18: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_54, _shape_param_18);  sum_dim_int_list_54 = _shape_param_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:214 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_18: "f32[3072, 16384]" = torch.ops.aten.permute.default(view_114, [1, 0])
        sum_dim_int_list_55: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_114, [0], True);  view_114 = None
        reshape_default_19: "f32[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_55, _shape_param_19);  sum_dim_int_list_55 = _shape_param_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:186 in forward, code: hidden_states = self.LayerNorm(input_tensor + hidden_states)
        mul_tensor_18: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_141, mul_34);  mul_34 = None
        sum_dim_int_list_56: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_18, [0, 1]);  mul_tensor_18 = None
        sum_dim_int_list_57: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_141, [0, 1]);  add_141 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:230 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_tensor_19: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_142, mul_32);  mul_32 = None
        sum_dim_int_list_58: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_19, [0, 1]);  mul_tensor_19 = None
        sum_dim_int_list_59: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_142, [0, 1]);  add_142 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:228 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_19: "f32[768, 16384]" = torch.ops.aten.permute.default(view_117, [1, 0])
        sum_dim_int_list_60: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_117, [0], True);  view_117 = None
        reshape_default_20: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_60, _shape_param_20);  sum_dim_int_list_60 = _shape_param_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:214 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_20: "f32[3072, 16384]" = torch.ops.aten.permute.default(view_120, [1, 0])
        sum_dim_int_list_61: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_120, [0], True);  view_120 = None
        reshape_default_21: "f32[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_61, _shape_param_21);  sum_dim_int_list_61 = _shape_param_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:186 in forward, code: hidden_states = self.LayerNorm(input_tensor + hidden_states)
        mul_tensor_20: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_145, mul_24);  mul_24 = None
        sum_dim_int_list_62: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_20, [0, 1]);  mul_tensor_20 = None
        sum_dim_int_list_63: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_145, [0, 1]);  add_145 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:230 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_tensor_21: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_146, mul_22);  mul_22 = None
        sum_dim_int_list_64: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_21, [0, 1]);  mul_tensor_21 = None
        sum_dim_int_list_65: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_146, [0, 1]);  add_146 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:228 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_21: "f32[768, 16384]" = torch.ops.aten.permute.default(view_123, [1, 0])
        sum_dim_int_list_66: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_123, [0], True);  view_123 = None
        reshape_default_22: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_66, _shape_param_22);  sum_dim_int_list_66 = _shape_param_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:214 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_22: "f32[3072, 16384]" = torch.ops.aten.permute.default(view_126, [1, 0])
        sum_dim_int_list_67: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_126, [0], True);  view_126 = None
        reshape_default_23: "f32[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_67, _shape_param_23);  sum_dim_int_list_67 = _shape_param_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:186 in forward, code: hidden_states = self.LayerNorm(input_tensor + hidden_states)
        mul_tensor_22: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_149, mul_14);  mul_14 = None
        sum_dim_int_list_68: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_22, [0, 1]);  mul_tensor_22 = None
        sum_dim_int_list_69: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_149, [0, 1]);  add_149 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:230 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_tensor_23: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_150, mul_12);  mul_12 = None
        sum_dim_int_list_70: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_23, [0, 1]);  mul_tensor_23 = None
        sum_dim_int_list_71: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_150, [0, 1]);  add_150 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:228 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_23: "f32[768, 16384]" = torch.ops.aten.permute.default(view_129, [1, 0])
        sum_dim_int_list_72: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_129, [0], True);  view_129 = None
        reshape_default_24: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_72, _shape_param_24);  sum_dim_int_list_72 = _shape_param_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:214 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_24: "f32[3072, 16384]" = torch.ops.aten.permute.default(view_132, [1, 0])
        sum_dim_int_list_73: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_132, [0], True);  view_132 = None
        reshape_default_25: "f32[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_73, _shape_param_25);  sum_dim_int_list_73 = _shape_param_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:186 in forward, code: hidden_states = self.LayerNorm(input_tensor + hidden_states)
        mul_tensor_24: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_153, mul_4);  mul_4 = None
        sum_dim_int_list_74: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_24, [0, 1]);  mul_tensor_24 = None
        sum_dim_int_list_75: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_153, [0, 1]);  add_153 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:137 in forward, code: embeddings = self.projection(embeddings)
        permute_default_25: "f32[768, 16384]" = torch.ops.aten.permute.default(view_135, [1, 0])
        sum_dim_int_list_76: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_135, [0], True);  view_135 = None
        reshape_default_26: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_76, _shape_param_26);  sum_dim_int_list_76 = _shape_param_26 = None
        reshape_default_27: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_52, _shape_param_27);  mm_52 = _shape_param_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:136 in forward, code: embeddings = self.LayerNorm(embeddings)
        mul_tensor_25: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(reshape_default_27, primals_7);  primals_7 = None
        mul_tensor_26: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_25, 768)
        sum_dim_int_list_77: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_25, [2], True)
        mul_tensor_27: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_25, mul);  mul_tensor_25 = None
        sum_dim_int_list_78: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_27, [2], True);  mul_tensor_27 = None
        mul_tensor_28: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul, sum_dim_int_list_78);  sum_dim_int_list_78 = None
        sub_tensor: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_tensor_26, sum_dim_int_list_77);  mul_tensor_26 = sum_dim_int_list_77 = None
        sub_tensor_1: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub_tensor, mul_tensor_28);  sub_tensor = mul_tensor_28 = None
        mul_tensor_29: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(div_27, sub_tensor_1);  div_27 = sub_tensor_1 = None
        mul_tensor_30: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(reshape_default_27, mul);  mul = None
        sum_dim_int_list_79: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_30, [0, 1]);  mul_tensor_30 = None
        sum_dim_int_list_80: "f32[768]" = torch.ops.aten.sum.dim_IntList(reshape_default_27, [0, 1]);  reshape_default_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:135 in forward, code: embeddings += position_embeddings
        sum_dim_int_list_81: "f32[1, 512, 768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_29, [0], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:134 in forward, code: position_embeddings = self.position_embeddings(position_ids)
        eq_scalar: "b8[1, 512]" = torch.ops.aten.eq.Scalar(primals_3, -1)
        unsqueeze_default: "b8[1, 512, 1]" = torch.ops.aten.unsqueeze.default(eq_scalar, -1);  eq_scalar = None
        where_self: "f32[1, 512, 768]" = torch.ops.aten.where.self(unsqueeze_default, full_default_1, sum_dim_int_list_81);  unsqueeze_default = sum_dim_int_list_81 = None
        full_default: "f32[512, 768]" = torch.ops.aten.full.default([512, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default: "f32[512, 768]" = torch.ops.aten.index_put.default(full_default, [primals_3], where_self, True);  full_default = primals_3 = where_self = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:479 in forward, code: buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
        expand_default: "i64[32, 512]" = torch.ops.aten.expand.default(primals_2, _shape_param_28);  primals_2 = _shape_param_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:130 in forward, code: token_type_embeddings = self.token_type_embeddings(token_type_ids)
        eq_scalar_1: "b8[32, 512]" = torch.ops.aten.eq.Scalar(expand_default, -1)
        unsqueeze_default_1: "b8[32, 512, 1]" = torch.ops.aten.unsqueeze.default(eq_scalar_1, -1);  eq_scalar_1 = None
        where_self_1: "f32[32, 512, 768]" = torch.ops.aten.where.self(unsqueeze_default_1, full_default_1, mul_tensor_29);  unsqueeze_default_1 = None
        full_default_2: "f32[4, 768]" = torch.ops.aten.full.default([4, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default_1: "f32[4, 768]" = torch.ops.aten.index_put.default(full_default_2, [expand_default], where_self_1, True);  full_default_2 = expand_default = where_self_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:129 in forward, code: inputs_embeds = self.word_embeddings(input_ids)
        eq_scalar_2: "b8[32, 512]" = torch.ops.aten.eq.Scalar(primals_1, 3)
        unsqueeze_default_2: "b8[32, 512, 1]" = torch.ops.aten.unsqueeze.default(eq_scalar_2, -1);  eq_scalar_2 = None
        where_self_2: "f32[32, 512, 768]" = torch.ops.aten.where.self(unsqueeze_default_2, full_default_1, mul_tensor_29);  unsqueeze_default_2 = full_default_1 = mul_tensor_29 = None
        full_default_3: "f32[32000, 768]" = torch.ops.aten.full.default([32000, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default_2: "f32[32000, 768]" = torch.ops.aten.index_put.default(full_default_3, [primals_1], where_self_2, True);  full_default_3 = primals_1 = where_self_2 = None
        add_tensor: "f32[32000, 768]" = torch.ops.aten.add.Tensor(mm_1, index_put_default_2);  mm_1 = index_put_default_2 = None
        return (reshape_default, sum_dim_int_list_1, sum_dim_int_list_2, permute_default, reshape_default_1, sum_dim_int_list_4, sum_dim_int_list_5, permute_default_1, reshape_default_2, permute_default_2, reshape_default_3, sum_dim_int_list_8, sum_dim_int_list_9, sum_dim_int_list_10, sum_dim_int_list_11, permute_default_3, reshape_default_4, permute_default_4, reshape_default_5, sum_dim_int_list_14, sum_dim_int_list_15, sum_dim_int_list_16, sum_dim_int_list_17, permute_default_5, reshape_default_6, permute_default_6, reshape_default_7, sum_dim_int_list_20, sum_dim_int_list_21, sum_dim_int_list_22, sum_dim_int_list_23, permute_default_7, reshape_default_8, permute_default_8, reshape_default_9, sum_dim_int_list_26, sum_dim_int_list_27, sum_dim_int_list_28, sum_dim_int_list_29, permute_default_9, reshape_default_10, permute_default_10, reshape_default_11, sum_dim_int_list_32, sum_dim_int_list_33, sum_dim_int_list_34, sum_dim_int_list_35, permute_default_11, reshape_default_12, permute_default_12, reshape_default_13, sum_dim_int_list_38, sum_dim_int_list_39, sum_dim_int_list_40, sum_dim_int_list_41, permute_default_13, reshape_default_14, permute_default_14, reshape_default_15, sum_dim_int_list_44, sum_dim_int_list_45, sum_dim_int_list_46, sum_dim_int_list_47, permute_default_15, reshape_default_16, permute_default_16, reshape_default_17, sum_dim_int_list_50, sum_dim_int_list_51, sum_dim_int_list_52, sum_dim_int_list_53, permute_default_17, reshape_default_18, permute_default_18, reshape_default_19, sum_dim_int_list_56, sum_dim_int_list_57, sum_dim_int_list_58, sum_dim_int_list_59, permute_default_19, reshape_default_20, permute_default_20, reshape_default_21, sum_dim_int_list_62, sum_dim_int_list_63, sum_dim_int_list_64, sum_dim_int_list_65, permute_default_21, reshape_default_22, permute_default_22, reshape_default_23, sum_dim_int_list_68, sum_dim_int_list_69, sum_dim_int_list_70, sum_dim_int_list_71, permute_default_23, reshape_default_24, permute_default_24, reshape_default_25, sum_dim_int_list_74, sum_dim_int_list_75, permute_default_25, reshape_default_26, sum_dim_int_list_79, sum_dim_int_list_80, index_put_default, index_put_default_1, add_tensor)
