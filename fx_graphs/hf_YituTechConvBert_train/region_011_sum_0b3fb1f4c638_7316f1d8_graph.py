class GraphModule(torch.nn.Module):
    def forward(self, mm_default: "f32[30524, 768]", view_439: "f32[16384, 30522]", view_441: "f32[32, 512, 768]", mul_176: "f32[32, 512, 768]", view_442: "f32[16384, 768]", view_444: "f32[32, 512, 768]", mul_171: "f32[32, 512, 768]", view_445: "f32[16384, 768]", view_448: "f32[16384, 3072]", add_156: "f32[32, 512, 768]", mul_164: "f32[32, 512, 768]", view_451: "f32[16384, 768]", view_470: "f32[16384, 384]", view_473: "f32[32, 512, 54]", view_475: "f32[16384, 54]", view_480: "f32[16384, 384]", permute_276: "f32[32, 384, 512]", view_484: "f32[16384, 384]", view_487: "f32[16384, 384]", add_164: "f32[32, 512, 768]", mul_157: "f32[32, 512, 768]", view_490: "f32[16384, 768]", view_493: "f32[16384, 3072]", add_167: "f32[32, 512, 768]", mul_150: "f32[32, 512, 768]", view_496: "f32[16384, 768]", view_515: "f32[16384, 384]", view_518: "f32[32, 512, 54]", view_520: "f32[16384, 54]", view_525: "f32[16384, 384]", permute_324: "f32[32, 384, 512]", view_529: "f32[16384, 384]", view_532: "f32[16384, 384]", add_175: "f32[32, 512, 768]", mul_143: "f32[32, 512, 768]", view_535: "f32[16384, 768]", view_538: "f32[16384, 3072]", add_178: "f32[32, 512, 768]", mul_136: "f32[32, 512, 768]", view_541: "f32[16384, 768]", view_560: "f32[16384, 384]", view_563: "f32[32, 512, 54]", view_565: "f32[16384, 54]", view_570: "f32[16384, 384]", permute_372: "f32[32, 384, 512]", view_574: "f32[16384, 384]", view_577: "f32[16384, 384]", add_186: "f32[32, 512, 768]", mul_129: "f32[32, 512, 768]", view_580: "f32[16384, 768]", view_583: "f32[16384, 3072]", add_189: "f32[32, 512, 768]", mul_122: "f32[32, 512, 768]", view_586: "f32[16384, 768]", view_605: "f32[16384, 384]", view_608: "f32[32, 512, 54]", view_610: "f32[16384, 54]", view_615: "f32[16384, 384]", permute_420: "f32[32, 384, 512]", view_619: "f32[16384, 384]", view_622: "f32[16384, 384]", add_197: "f32[32, 512, 768]", mul_115: "f32[32, 512, 768]", view_625: "f32[16384, 768]", view_628: "f32[16384, 3072]", add_200: "f32[32, 512, 768]", mul_108: "f32[32, 512, 768]", view_631: "f32[16384, 768]", view_650: "f32[16384, 384]", view_653: "f32[32, 512, 54]", view_655: "f32[16384, 54]", view_660: "f32[16384, 384]", permute_468: "f32[32, 384, 512]", view_664: "f32[16384, 384]", view_667: "f32[16384, 384]", add_208: "f32[32, 512, 768]", mul_101: "f32[32, 512, 768]", view_670: "f32[16384, 768]", view_673: "f32[16384, 3072]", add_211: "f32[32, 512, 768]", mul_94: "f32[32, 512, 768]", view_676: "f32[16384, 768]", view_695: "f32[16384, 384]", view_698: "f32[32, 512, 54]", view_700: "f32[16384, 54]", view_705: "f32[16384, 384]", permute_516: "f32[32, 384, 512]", view_709: "f32[16384, 384]", view_712: "f32[16384, 384]", add_219: "f32[32, 512, 768]", mul_87: "f32[32, 512, 768]", view_715: "f32[16384, 768]", view_718: "f32[16384, 3072]", add_222: "f32[32, 512, 768]", mul_80: "f32[32, 512, 768]", view_721: "f32[16384, 768]", view_740: "f32[16384, 384]", view_743: "f32[32, 512, 54]", view_745: "f32[16384, 54]", view_750: "f32[16384, 384]", permute_564: "f32[32, 384, 512]", view_754: "f32[16384, 384]", view_757: "f32[16384, 384]", add_230: "f32[32, 512, 768]", mul_73: "f32[32, 512, 768]", view_760: "f32[16384, 768]", view_763: "f32[16384, 3072]", add_233: "f32[32, 512, 768]", mul_66: "f32[32, 512, 768]", view_766: "f32[16384, 768]", view_785: "f32[16384, 384]", view_788: "f32[32, 512, 54]", view_790: "f32[16384, 54]", view_795: "f32[16384, 384]", permute_612: "f32[32, 384, 512]", view_799: "f32[16384, 384]", view_802: "f32[16384, 384]", add_241: "f32[32, 512, 768]", mul_59: "f32[32, 512, 768]", view_805: "f32[16384, 768]", view_808: "f32[16384, 3072]", add_244: "f32[32, 512, 768]", mul_52: "f32[32, 512, 768]", view_811: "f32[16384, 768]", view_830: "f32[16384, 384]", view_833: "f32[32, 512, 54]", view_835: "f32[16384, 54]", view_840: "f32[16384, 384]", permute_660: "f32[32, 384, 512]", view_844: "f32[16384, 384]", view_847: "f32[16384, 384]", add_252: "f32[32, 512, 768]", mul_45: "f32[32, 512, 768]", view_850: "f32[16384, 768]", view_853: "f32[16384, 3072]", add_255: "f32[32, 512, 768]", mul_38: "f32[32, 512, 768]", view_856: "f32[16384, 768]", view_875: "f32[16384, 384]", view_878: "f32[32, 512, 54]", view_880: "f32[16384, 54]", view_885: "f32[16384, 384]", permute_708: "f32[32, 384, 512]", view_889: "f32[16384, 384]", view_892: "f32[16384, 384]", add_263: "f32[32, 512, 768]", mul_31: "f32[32, 512, 768]", view_895: "f32[16384, 768]", view_898: "f32[16384, 3072]", add_266: "f32[32, 512, 768]", mul_24: "f32[32, 512, 768]", view_901: "f32[16384, 768]", view_920: "f32[16384, 384]", view_923: "f32[32, 512, 54]", view_925: "f32[16384, 54]", view_930: "f32[16384, 384]", permute_756: "f32[32, 384, 512]", view_934: "f32[16384, 384]", view_937: "f32[16384, 384]", add_274: "f32[32, 512, 768]", mul_17: "f32[32, 512, 768]", view_940: "f32[16384, 768]", view_943: "f32[16384, 3072]", add_277: "f32[32, 512, 768]", mul_10: "f32[32, 512, 768]", view_946: "f32[16384, 768]", view_965: "f32[16384, 384]", mm_198: "f32[16384, 768]", mul_567: "f32[32, 512, 768]", view_968: "f32[32, 512, 54]", view_970: "f32[16384, 54]", view_975: "f32[16384, 384]", mm_202: "f32[16384, 768]", permute_804: "f32[32, 384, 512]", getitem_121: "f32[32, 768, 512]", view_979: "f32[16384, 384]", mm_204: "f32[16384, 768]", view_982: "f32[16384, 384]", mm_206: "f32[16384, 768]", gt: "b8[32, 512, 768]", primals_7: "f32[768]", mul_1: "f32[32, 512, 768]", div_75: "f32[32, 512, 1]", primals_2: "i64[1, 512]", full_default_2: "f32[]", primals_4: "i64[1, 512]", primals_1: "i64[32, 512]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12, _shape_param_13, _shape_param_14, _shape_param_15, _shape_param_16, _shape_param_17, _shape_param_18, _shape_param_19, _shape_param_20, _shape_param_21, _shape_param_22, _shape_param_23, _shape_param_24, _shape_param_25, _shape_param_26, _shape_param_27, _shape_param_28, _shape_param_29, _shape_param_30, _shape_param_31, _shape_param_32, _shape_param_33, _shape_param_34, _shape_param_35, _shape_param_36, _shape_param_37, _shape_param_38, _shape_param_39, _shape_param_40, _shape_param_41, _shape_param_42, _shape_param_43, _shape_param_44, _shape_param_45, _shape_param_46, _shape_param_47, _shape_param_48, _shape_param_49, _shape_param_50, _shape_param_51, _shape_param_52, _shape_param_53, _shape_param_54, _shape_param_55, _shape_param_56, _shape_param_57, _shape_param_58, _shape_param_59, _shape_param_60, _shape_param_61, _shape_param_62, _shape_param_63, _shape_param_64, _shape_param_65, _shape_param_66, _shape_param_67, _shape_param_68, _shape_param_69, _shape_param_70, _shape_param_71, _shape_param_72, _shape_param_73, _shape_param_74, _shape_param_75, _shape_param_76, _shape_param_77, _shape_param_78, _shape_param_79, _shape_param_80, _shape_param_81, _shape_param_82, _shape_param_83, _shape_param_84, _shape_param_85, _shape_param_86, _shape_param_87, _shape_param_88, _shape_param_89, _shape_param_90, _shape_param_91, _shape_param_92, _shape_param_93, _shape_param_94, _shape_param_95, _shape_param_96, _shape_param_97, _shape_param_98, _shape_param_99, _shape_param_100, _shape_param_101, _shape_param_102, _shape_param_103, _shape_param_104, _shape_param_105, _shape_param_106, _shape_param_107, _shape_param_108, _shape_param_109, _shape_param_110, _shape_param_111, _shape_param_112, _shape_param_113, _shape_param_114):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:718 in forward, code: prediction_scores = self.generator_lm_head(prediction_scores)
        slice_tensor: "f32[30522, 768]" = torch.ops.aten.slice.Tensor(mm_default, 0, 0, -2);  mm_default = None
        sum_dim_int_list: "f32[1, 30522]" = torch.ops.aten.sum.dim_IntList(view_439, [0], True);  view_439 = None
        reshape_default: "f32[30522]" = torch.ops.aten.reshape.default(sum_dim_int_list, _shape_param_0);  sum_dim_int_list = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:664 in forward, code: hidden_states = self.LayerNorm(hidden_states)
        mul_tensor: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(view_441, mul_176);  mul_176 = None
        sum_dim_int_list_1: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [0, 1]);  mul_tensor = None
        sum_dim_int_list_2: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_441, [0, 1]);  view_441 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:662 in forward, code: hidden_states = self.dense(generator_hidden_states)
        permute_default: "f32[768, 16384]" = torch.ops.aten.permute.default(view_442, [1, 0])
        sum_dim_int_list_3: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_442, [0], True);  view_442 = None
        reshape_default_1: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_3, _shape_param_1);  sum_dim_int_list_3 = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:351 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_tensor_1: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(view_444, mul_171);  mul_171 = None
        sum_dim_int_list_4: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_1, [0, 1]);  mul_tensor_1 = None
        sum_dim_int_list_5: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_444, [0, 1]);  view_444 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:349 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_1: "f32[768, 16384]" = torch.ops.aten.permute.default(view_445, [1, 0])
        sum_dim_int_list_6: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_445, [0], True);  view_445 = None
        reshape_default_2: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_6, _shape_param_2);  sum_dim_int_list_6 = _shape_param_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:331 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_2: "f32[3072, 16384]" = torch.ops.aten.permute.default(view_448, [1, 0])
        sum_dim_int_list_7: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_448, [0], True);  view_448 = None
        reshape_default_3: "f32[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_7, _shape_param_3);  sum_dim_int_list_7 = _shape_param_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:267 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_tensor_2: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_156, mul_164);  mul_164 = None
        sum_dim_int_list_8: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [0, 1]);  mul_tensor_2 = None
        sum_dim_int_list_9: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_156, [0, 1]);  add_156 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:265 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_3: "f32[768, 16384]" = torch.ops.aten.permute.default(view_451, [1, 0])
        sum_dim_int_list_10: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_451, [0], True);  view_451 = None
        reshape_default_4: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_10, _shape_param_4);  sum_dim_int_list_10 = _shape_param_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:209 in forward, code: conv_out_layer = self.conv_out_layer(hidden_states)
        permute_default_4: "f32[384, 16384]" = torch.ops.aten.permute.default(view_470, [1, 0])
        sum_dim_int_list_11: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_470, [0], True);  view_470 = None
        reshape_default_5: "f32[384]" = torch.ops.aten.reshape.default(sum_dim_int_list_11, _shape_param_5);  sum_dim_int_list_11 = _shape_param_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:205 in forward, code: conv_kernel_layer = self.conv_kernel_layer(conv_attn_layer)
        sum_dim_int_list_12: "f32[1, 1, 54]" = torch.ops.aten.sum.dim_IntList(view_473, [0, 1], True);  view_473 = None
        reshape_default_6: "f32[54]" = torch.ops.aten.reshape.default(sum_dim_int_list_12, _shape_param_6);  sum_dim_int_list_12 = _shape_param_6 = None
        permute_default_5: "f32[54, 16384]" = torch.ops.aten.permute.default(view_475, [1, 0]);  view_475 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:197 in forward, code: mixed_query_layer = self.query(hidden_states)
        permute_default_6: "f32[384, 16384]" = torch.ops.aten.permute.default(view_480, [1, 0])
        sum_dim_int_list_13: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_480, [0], True);  view_480 = None
        reshape_default_7: "f32[384]" = torch.ops.aten.reshape.default(sum_dim_int_list_13, _shape_param_7);  sum_dim_int_list_13 = _shape_param_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:131 in forward, code: x += self.bias
        sum_dim_int_list_14: "f32[1, 384, 1]" = torch.ops.aten.sum.dim_IntList(permute_276, [0, 2], True);  permute_276 = None
        reshape_default_8: "f32[384, 1]" = torch.ops.aten.reshape.default(sum_dim_int_list_14, _shape_param_8);  sum_dim_int_list_14 = _shape_param_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:192 in forward, code: mixed_value_layer = self.value(hidden_states)
        permute_default_7: "f32[384, 16384]" = torch.ops.aten.permute.default(view_484, [1, 0])
        sum_dim_int_list_15: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_484, [0], True);  view_484 = None
        reshape_default_9: "f32[384]" = torch.ops.aten.reshape.default(sum_dim_int_list_15, _shape_param_9);  sum_dim_int_list_15 = _shape_param_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:191 in forward, code: mixed_key_layer = self.key(hidden_states)
        permute_default_8: "f32[384, 16384]" = torch.ops.aten.permute.default(view_487, [1, 0])
        sum_dim_int_list_16: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_487, [0], True);  view_487 = None
        reshape_default_10: "f32[384]" = torch.ops.aten.reshape.default(sum_dim_int_list_16, _shape_param_10);  sum_dim_int_list_16 = _shape_param_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:351 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_tensor_3: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_164, mul_157);  mul_157 = None
        sum_dim_int_list_17: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_3, [0, 1]);  mul_tensor_3 = None
        sum_dim_int_list_18: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_164, [0, 1]);  add_164 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:349 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_9: "f32[768, 16384]" = torch.ops.aten.permute.default(view_490, [1, 0])
        sum_dim_int_list_19: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_490, [0], True);  view_490 = None
        reshape_default_11: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_19, _shape_param_11);  sum_dim_int_list_19 = _shape_param_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:331 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_10: "f32[3072, 16384]" = torch.ops.aten.permute.default(view_493, [1, 0])
        sum_dim_int_list_20: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_493, [0], True);  view_493 = None
        reshape_default_12: "f32[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_20, _shape_param_12);  sum_dim_int_list_20 = _shape_param_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:267 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_tensor_4: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_167, mul_150);  mul_150 = None
        sum_dim_int_list_21: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_4, [0, 1]);  mul_tensor_4 = None
        sum_dim_int_list_22: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_167, [0, 1]);  add_167 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:265 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_11: "f32[768, 16384]" = torch.ops.aten.permute.default(view_496, [1, 0])
        sum_dim_int_list_23: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_496, [0], True);  view_496 = None
        reshape_default_13: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_23, _shape_param_13);  sum_dim_int_list_23 = _shape_param_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:209 in forward, code: conv_out_layer = self.conv_out_layer(hidden_states)
        permute_default_12: "f32[384, 16384]" = torch.ops.aten.permute.default(view_515, [1, 0])
        sum_dim_int_list_24: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_515, [0], True);  view_515 = None
        reshape_default_14: "f32[384]" = torch.ops.aten.reshape.default(sum_dim_int_list_24, _shape_param_14);  sum_dim_int_list_24 = _shape_param_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:205 in forward, code: conv_kernel_layer = self.conv_kernel_layer(conv_attn_layer)
        sum_dim_int_list_25: "f32[1, 1, 54]" = torch.ops.aten.sum.dim_IntList(view_518, [0, 1], True);  view_518 = None
        reshape_default_15: "f32[54]" = torch.ops.aten.reshape.default(sum_dim_int_list_25, _shape_param_15);  sum_dim_int_list_25 = _shape_param_15 = None
        permute_default_13: "f32[54, 16384]" = torch.ops.aten.permute.default(view_520, [1, 0]);  view_520 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:197 in forward, code: mixed_query_layer = self.query(hidden_states)
        permute_default_14: "f32[384, 16384]" = torch.ops.aten.permute.default(view_525, [1, 0])
        sum_dim_int_list_26: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_525, [0], True);  view_525 = None
        reshape_default_16: "f32[384]" = torch.ops.aten.reshape.default(sum_dim_int_list_26, _shape_param_16);  sum_dim_int_list_26 = _shape_param_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:131 in forward, code: x += self.bias
        sum_dim_int_list_27: "f32[1, 384, 1]" = torch.ops.aten.sum.dim_IntList(permute_324, [0, 2], True);  permute_324 = None
        reshape_default_17: "f32[384, 1]" = torch.ops.aten.reshape.default(sum_dim_int_list_27, _shape_param_17);  sum_dim_int_list_27 = _shape_param_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:192 in forward, code: mixed_value_layer = self.value(hidden_states)
        permute_default_15: "f32[384, 16384]" = torch.ops.aten.permute.default(view_529, [1, 0])
        sum_dim_int_list_28: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_529, [0], True);  view_529 = None
        reshape_default_18: "f32[384]" = torch.ops.aten.reshape.default(sum_dim_int_list_28, _shape_param_18);  sum_dim_int_list_28 = _shape_param_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:191 in forward, code: mixed_key_layer = self.key(hidden_states)
        permute_default_16: "f32[384, 16384]" = torch.ops.aten.permute.default(view_532, [1, 0])
        sum_dim_int_list_29: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_532, [0], True);  view_532 = None
        reshape_default_19: "f32[384]" = torch.ops.aten.reshape.default(sum_dim_int_list_29, _shape_param_19);  sum_dim_int_list_29 = _shape_param_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:351 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_tensor_5: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_175, mul_143);  mul_143 = None
        sum_dim_int_list_30: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_5, [0, 1]);  mul_tensor_5 = None
        sum_dim_int_list_31: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_175, [0, 1]);  add_175 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:349 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_17: "f32[768, 16384]" = torch.ops.aten.permute.default(view_535, [1, 0])
        sum_dim_int_list_32: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_535, [0], True);  view_535 = None
        reshape_default_20: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_32, _shape_param_20);  sum_dim_int_list_32 = _shape_param_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:331 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_18: "f32[3072, 16384]" = torch.ops.aten.permute.default(view_538, [1, 0])
        sum_dim_int_list_33: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_538, [0], True);  view_538 = None
        reshape_default_21: "f32[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_33, _shape_param_21);  sum_dim_int_list_33 = _shape_param_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:267 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_tensor_6: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_178, mul_136);  mul_136 = None
        sum_dim_int_list_34: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_6, [0, 1]);  mul_tensor_6 = None
        sum_dim_int_list_35: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_178, [0, 1]);  add_178 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:265 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_19: "f32[768, 16384]" = torch.ops.aten.permute.default(view_541, [1, 0])
        sum_dim_int_list_36: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_541, [0], True);  view_541 = None
        reshape_default_22: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_36, _shape_param_22);  sum_dim_int_list_36 = _shape_param_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:209 in forward, code: conv_out_layer = self.conv_out_layer(hidden_states)
        permute_default_20: "f32[384, 16384]" = torch.ops.aten.permute.default(view_560, [1, 0])
        sum_dim_int_list_37: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_560, [0], True);  view_560 = None
        reshape_default_23: "f32[384]" = torch.ops.aten.reshape.default(sum_dim_int_list_37, _shape_param_23);  sum_dim_int_list_37 = _shape_param_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:205 in forward, code: conv_kernel_layer = self.conv_kernel_layer(conv_attn_layer)
        sum_dim_int_list_38: "f32[1, 1, 54]" = torch.ops.aten.sum.dim_IntList(view_563, [0, 1], True);  view_563 = None
        reshape_default_24: "f32[54]" = torch.ops.aten.reshape.default(sum_dim_int_list_38, _shape_param_24);  sum_dim_int_list_38 = _shape_param_24 = None
        permute_default_21: "f32[54, 16384]" = torch.ops.aten.permute.default(view_565, [1, 0]);  view_565 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:197 in forward, code: mixed_query_layer = self.query(hidden_states)
        permute_default_22: "f32[384, 16384]" = torch.ops.aten.permute.default(view_570, [1, 0])
        sum_dim_int_list_39: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_570, [0], True);  view_570 = None
        reshape_default_25: "f32[384]" = torch.ops.aten.reshape.default(sum_dim_int_list_39, _shape_param_25);  sum_dim_int_list_39 = _shape_param_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:131 in forward, code: x += self.bias
        sum_dim_int_list_40: "f32[1, 384, 1]" = torch.ops.aten.sum.dim_IntList(permute_372, [0, 2], True);  permute_372 = None
        reshape_default_26: "f32[384, 1]" = torch.ops.aten.reshape.default(sum_dim_int_list_40, _shape_param_26);  sum_dim_int_list_40 = _shape_param_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:192 in forward, code: mixed_value_layer = self.value(hidden_states)
        permute_default_23: "f32[384, 16384]" = torch.ops.aten.permute.default(view_574, [1, 0])
        sum_dim_int_list_41: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_574, [0], True);  view_574 = None
        reshape_default_27: "f32[384]" = torch.ops.aten.reshape.default(sum_dim_int_list_41, _shape_param_27);  sum_dim_int_list_41 = _shape_param_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:191 in forward, code: mixed_key_layer = self.key(hidden_states)
        permute_default_24: "f32[384, 16384]" = torch.ops.aten.permute.default(view_577, [1, 0])
        sum_dim_int_list_42: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_577, [0], True);  view_577 = None
        reshape_default_28: "f32[384]" = torch.ops.aten.reshape.default(sum_dim_int_list_42, _shape_param_28);  sum_dim_int_list_42 = _shape_param_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:351 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_tensor_7: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_186, mul_129);  mul_129 = None
        sum_dim_int_list_43: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_7, [0, 1]);  mul_tensor_7 = None
        sum_dim_int_list_44: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_186, [0, 1]);  add_186 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:349 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_25: "f32[768, 16384]" = torch.ops.aten.permute.default(view_580, [1, 0])
        sum_dim_int_list_45: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_580, [0], True);  view_580 = None
        reshape_default_29: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_45, _shape_param_29);  sum_dim_int_list_45 = _shape_param_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:331 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_26: "f32[3072, 16384]" = torch.ops.aten.permute.default(view_583, [1, 0])
        sum_dim_int_list_46: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_583, [0], True);  view_583 = None
        reshape_default_30: "f32[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_46, _shape_param_30);  sum_dim_int_list_46 = _shape_param_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:267 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_tensor_8: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_189, mul_122);  mul_122 = None
        sum_dim_int_list_47: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_8, [0, 1]);  mul_tensor_8 = None
        sum_dim_int_list_48: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_189, [0, 1]);  add_189 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:265 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_27: "f32[768, 16384]" = torch.ops.aten.permute.default(view_586, [1, 0])
        sum_dim_int_list_49: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_586, [0], True);  view_586 = None
        reshape_default_31: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_49, _shape_param_31);  sum_dim_int_list_49 = _shape_param_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:209 in forward, code: conv_out_layer = self.conv_out_layer(hidden_states)
        permute_default_28: "f32[384, 16384]" = torch.ops.aten.permute.default(view_605, [1, 0])
        sum_dim_int_list_50: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_605, [0], True);  view_605 = None
        reshape_default_32: "f32[384]" = torch.ops.aten.reshape.default(sum_dim_int_list_50, _shape_param_32);  sum_dim_int_list_50 = _shape_param_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:205 in forward, code: conv_kernel_layer = self.conv_kernel_layer(conv_attn_layer)
        sum_dim_int_list_51: "f32[1, 1, 54]" = torch.ops.aten.sum.dim_IntList(view_608, [0, 1], True);  view_608 = None
        reshape_default_33: "f32[54]" = torch.ops.aten.reshape.default(sum_dim_int_list_51, _shape_param_33);  sum_dim_int_list_51 = _shape_param_33 = None
        permute_default_29: "f32[54, 16384]" = torch.ops.aten.permute.default(view_610, [1, 0]);  view_610 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:197 in forward, code: mixed_query_layer = self.query(hidden_states)
        permute_default_30: "f32[384, 16384]" = torch.ops.aten.permute.default(view_615, [1, 0])
        sum_dim_int_list_52: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_615, [0], True);  view_615 = None
        reshape_default_34: "f32[384]" = torch.ops.aten.reshape.default(sum_dim_int_list_52, _shape_param_34);  sum_dim_int_list_52 = _shape_param_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:131 in forward, code: x += self.bias
        sum_dim_int_list_53: "f32[1, 384, 1]" = torch.ops.aten.sum.dim_IntList(permute_420, [0, 2], True);  permute_420 = None
        reshape_default_35: "f32[384, 1]" = torch.ops.aten.reshape.default(sum_dim_int_list_53, _shape_param_35);  sum_dim_int_list_53 = _shape_param_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:192 in forward, code: mixed_value_layer = self.value(hidden_states)
        permute_default_31: "f32[384, 16384]" = torch.ops.aten.permute.default(view_619, [1, 0])
        sum_dim_int_list_54: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_619, [0], True);  view_619 = None
        reshape_default_36: "f32[384]" = torch.ops.aten.reshape.default(sum_dim_int_list_54, _shape_param_36);  sum_dim_int_list_54 = _shape_param_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:191 in forward, code: mixed_key_layer = self.key(hidden_states)
        permute_default_32: "f32[384, 16384]" = torch.ops.aten.permute.default(view_622, [1, 0])
        sum_dim_int_list_55: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_622, [0], True);  view_622 = None
        reshape_default_37: "f32[384]" = torch.ops.aten.reshape.default(sum_dim_int_list_55, _shape_param_37);  sum_dim_int_list_55 = _shape_param_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:351 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_tensor_9: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_197, mul_115);  mul_115 = None
        sum_dim_int_list_56: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_9, [0, 1]);  mul_tensor_9 = None
        sum_dim_int_list_57: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_197, [0, 1]);  add_197 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:349 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_33: "f32[768, 16384]" = torch.ops.aten.permute.default(view_625, [1, 0])
        sum_dim_int_list_58: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_625, [0], True);  view_625 = None
        reshape_default_38: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_58, _shape_param_38);  sum_dim_int_list_58 = _shape_param_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:331 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_34: "f32[3072, 16384]" = torch.ops.aten.permute.default(view_628, [1, 0])
        sum_dim_int_list_59: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_628, [0], True);  view_628 = None
        reshape_default_39: "f32[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_59, _shape_param_39);  sum_dim_int_list_59 = _shape_param_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:267 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_tensor_10: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_200, mul_108);  mul_108 = None
        sum_dim_int_list_60: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_10, [0, 1]);  mul_tensor_10 = None
        sum_dim_int_list_61: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_200, [0, 1]);  add_200 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:265 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_35: "f32[768, 16384]" = torch.ops.aten.permute.default(view_631, [1, 0])
        sum_dim_int_list_62: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_631, [0], True);  view_631 = None
        reshape_default_40: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_62, _shape_param_40);  sum_dim_int_list_62 = _shape_param_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:209 in forward, code: conv_out_layer = self.conv_out_layer(hidden_states)
        permute_default_36: "f32[384, 16384]" = torch.ops.aten.permute.default(view_650, [1, 0])
        sum_dim_int_list_63: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_650, [0], True);  view_650 = None
        reshape_default_41: "f32[384]" = torch.ops.aten.reshape.default(sum_dim_int_list_63, _shape_param_41);  sum_dim_int_list_63 = _shape_param_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:205 in forward, code: conv_kernel_layer = self.conv_kernel_layer(conv_attn_layer)
        sum_dim_int_list_64: "f32[1, 1, 54]" = torch.ops.aten.sum.dim_IntList(view_653, [0, 1], True);  view_653 = None
        reshape_default_42: "f32[54]" = torch.ops.aten.reshape.default(sum_dim_int_list_64, _shape_param_42);  sum_dim_int_list_64 = _shape_param_42 = None
        permute_default_37: "f32[54, 16384]" = torch.ops.aten.permute.default(view_655, [1, 0]);  view_655 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:197 in forward, code: mixed_query_layer = self.query(hidden_states)
        permute_default_38: "f32[384, 16384]" = torch.ops.aten.permute.default(view_660, [1, 0])
        sum_dim_int_list_65: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_660, [0], True);  view_660 = None
        reshape_default_43: "f32[384]" = torch.ops.aten.reshape.default(sum_dim_int_list_65, _shape_param_43);  sum_dim_int_list_65 = _shape_param_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:131 in forward, code: x += self.bias
        sum_dim_int_list_66: "f32[1, 384, 1]" = torch.ops.aten.sum.dim_IntList(permute_468, [0, 2], True);  permute_468 = None
        reshape_default_44: "f32[384, 1]" = torch.ops.aten.reshape.default(sum_dim_int_list_66, _shape_param_44);  sum_dim_int_list_66 = _shape_param_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:192 in forward, code: mixed_value_layer = self.value(hidden_states)
        permute_default_39: "f32[384, 16384]" = torch.ops.aten.permute.default(view_664, [1, 0])
        sum_dim_int_list_67: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_664, [0], True);  view_664 = None
        reshape_default_45: "f32[384]" = torch.ops.aten.reshape.default(sum_dim_int_list_67, _shape_param_45);  sum_dim_int_list_67 = _shape_param_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:191 in forward, code: mixed_key_layer = self.key(hidden_states)
        permute_default_40: "f32[384, 16384]" = torch.ops.aten.permute.default(view_667, [1, 0])
        sum_dim_int_list_68: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_667, [0], True);  view_667 = None
        reshape_default_46: "f32[384]" = torch.ops.aten.reshape.default(sum_dim_int_list_68, _shape_param_46);  sum_dim_int_list_68 = _shape_param_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:351 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_tensor_11: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_208, mul_101);  mul_101 = None
        sum_dim_int_list_69: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_11, [0, 1]);  mul_tensor_11 = None
        sum_dim_int_list_70: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_208, [0, 1]);  add_208 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:349 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_41: "f32[768, 16384]" = torch.ops.aten.permute.default(view_670, [1, 0])
        sum_dim_int_list_71: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_670, [0], True);  view_670 = None
        reshape_default_47: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_71, _shape_param_47);  sum_dim_int_list_71 = _shape_param_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:331 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_42: "f32[3072, 16384]" = torch.ops.aten.permute.default(view_673, [1, 0])
        sum_dim_int_list_72: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_673, [0], True);  view_673 = None
        reshape_default_48: "f32[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_72, _shape_param_48);  sum_dim_int_list_72 = _shape_param_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:267 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_tensor_12: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_211, mul_94);  mul_94 = None
        sum_dim_int_list_73: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_12, [0, 1]);  mul_tensor_12 = None
        sum_dim_int_list_74: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_211, [0, 1]);  add_211 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:265 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_43: "f32[768, 16384]" = torch.ops.aten.permute.default(view_676, [1, 0])
        sum_dim_int_list_75: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_676, [0], True);  view_676 = None
        reshape_default_49: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_75, _shape_param_49);  sum_dim_int_list_75 = _shape_param_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:209 in forward, code: conv_out_layer = self.conv_out_layer(hidden_states)
        permute_default_44: "f32[384, 16384]" = torch.ops.aten.permute.default(view_695, [1, 0])
        sum_dim_int_list_76: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_695, [0], True);  view_695 = None
        reshape_default_50: "f32[384]" = torch.ops.aten.reshape.default(sum_dim_int_list_76, _shape_param_50);  sum_dim_int_list_76 = _shape_param_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:205 in forward, code: conv_kernel_layer = self.conv_kernel_layer(conv_attn_layer)
        sum_dim_int_list_77: "f32[1, 1, 54]" = torch.ops.aten.sum.dim_IntList(view_698, [0, 1], True);  view_698 = None
        reshape_default_51: "f32[54]" = torch.ops.aten.reshape.default(sum_dim_int_list_77, _shape_param_51);  sum_dim_int_list_77 = _shape_param_51 = None
        permute_default_45: "f32[54, 16384]" = torch.ops.aten.permute.default(view_700, [1, 0]);  view_700 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:197 in forward, code: mixed_query_layer = self.query(hidden_states)
        permute_default_46: "f32[384, 16384]" = torch.ops.aten.permute.default(view_705, [1, 0])
        sum_dim_int_list_78: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_705, [0], True);  view_705 = None
        reshape_default_52: "f32[384]" = torch.ops.aten.reshape.default(sum_dim_int_list_78, _shape_param_52);  sum_dim_int_list_78 = _shape_param_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:131 in forward, code: x += self.bias
        sum_dim_int_list_79: "f32[1, 384, 1]" = torch.ops.aten.sum.dim_IntList(permute_516, [0, 2], True);  permute_516 = None
        reshape_default_53: "f32[384, 1]" = torch.ops.aten.reshape.default(sum_dim_int_list_79, _shape_param_53);  sum_dim_int_list_79 = _shape_param_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:192 in forward, code: mixed_value_layer = self.value(hidden_states)
        permute_default_47: "f32[384, 16384]" = torch.ops.aten.permute.default(view_709, [1, 0])
        sum_dim_int_list_80: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_709, [0], True);  view_709 = None
        reshape_default_54: "f32[384]" = torch.ops.aten.reshape.default(sum_dim_int_list_80, _shape_param_54);  sum_dim_int_list_80 = _shape_param_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:191 in forward, code: mixed_key_layer = self.key(hidden_states)
        permute_default_48: "f32[384, 16384]" = torch.ops.aten.permute.default(view_712, [1, 0])
        sum_dim_int_list_81: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_712, [0], True);  view_712 = None
        reshape_default_55: "f32[384]" = torch.ops.aten.reshape.default(sum_dim_int_list_81, _shape_param_55);  sum_dim_int_list_81 = _shape_param_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:351 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_tensor_13: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_219, mul_87);  mul_87 = None
        sum_dim_int_list_82: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_13, [0, 1]);  mul_tensor_13 = None
        sum_dim_int_list_83: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_219, [0, 1]);  add_219 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:349 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_49: "f32[768, 16384]" = torch.ops.aten.permute.default(view_715, [1, 0])
        sum_dim_int_list_84: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_715, [0], True);  view_715 = None
        reshape_default_56: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_84, _shape_param_56);  sum_dim_int_list_84 = _shape_param_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:331 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_50: "f32[3072, 16384]" = torch.ops.aten.permute.default(view_718, [1, 0])
        sum_dim_int_list_85: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_718, [0], True);  view_718 = None
        reshape_default_57: "f32[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_85, _shape_param_57);  sum_dim_int_list_85 = _shape_param_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:267 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_tensor_14: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_222, mul_80);  mul_80 = None
        sum_dim_int_list_86: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_14, [0, 1]);  mul_tensor_14 = None
        sum_dim_int_list_87: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_222, [0, 1]);  add_222 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:265 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_51: "f32[768, 16384]" = torch.ops.aten.permute.default(view_721, [1, 0])
        sum_dim_int_list_88: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_721, [0], True);  view_721 = None
        reshape_default_58: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_88, _shape_param_58);  sum_dim_int_list_88 = _shape_param_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:209 in forward, code: conv_out_layer = self.conv_out_layer(hidden_states)
        permute_default_52: "f32[384, 16384]" = torch.ops.aten.permute.default(view_740, [1, 0])
        sum_dim_int_list_89: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_740, [0], True);  view_740 = None
        reshape_default_59: "f32[384]" = torch.ops.aten.reshape.default(sum_dim_int_list_89, _shape_param_59);  sum_dim_int_list_89 = _shape_param_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:205 in forward, code: conv_kernel_layer = self.conv_kernel_layer(conv_attn_layer)
        sum_dim_int_list_90: "f32[1, 1, 54]" = torch.ops.aten.sum.dim_IntList(view_743, [0, 1], True);  view_743 = None
        reshape_default_60: "f32[54]" = torch.ops.aten.reshape.default(sum_dim_int_list_90, _shape_param_60);  sum_dim_int_list_90 = _shape_param_60 = None
        permute_default_53: "f32[54, 16384]" = torch.ops.aten.permute.default(view_745, [1, 0]);  view_745 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:197 in forward, code: mixed_query_layer = self.query(hidden_states)
        permute_default_54: "f32[384, 16384]" = torch.ops.aten.permute.default(view_750, [1, 0])
        sum_dim_int_list_91: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_750, [0], True);  view_750 = None
        reshape_default_61: "f32[384]" = torch.ops.aten.reshape.default(sum_dim_int_list_91, _shape_param_61);  sum_dim_int_list_91 = _shape_param_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:131 in forward, code: x += self.bias
        sum_dim_int_list_92: "f32[1, 384, 1]" = torch.ops.aten.sum.dim_IntList(permute_564, [0, 2], True);  permute_564 = None
        reshape_default_62: "f32[384, 1]" = torch.ops.aten.reshape.default(sum_dim_int_list_92, _shape_param_62);  sum_dim_int_list_92 = _shape_param_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:192 in forward, code: mixed_value_layer = self.value(hidden_states)
        permute_default_55: "f32[384, 16384]" = torch.ops.aten.permute.default(view_754, [1, 0])
        sum_dim_int_list_93: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_754, [0], True);  view_754 = None
        reshape_default_63: "f32[384]" = torch.ops.aten.reshape.default(sum_dim_int_list_93, _shape_param_63);  sum_dim_int_list_93 = _shape_param_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:191 in forward, code: mixed_key_layer = self.key(hidden_states)
        permute_default_56: "f32[384, 16384]" = torch.ops.aten.permute.default(view_757, [1, 0])
        sum_dim_int_list_94: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_757, [0], True);  view_757 = None
        reshape_default_64: "f32[384]" = torch.ops.aten.reshape.default(sum_dim_int_list_94, _shape_param_64);  sum_dim_int_list_94 = _shape_param_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:351 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_tensor_15: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_230, mul_73);  mul_73 = None
        sum_dim_int_list_95: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_15, [0, 1]);  mul_tensor_15 = None
        sum_dim_int_list_96: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_230, [0, 1]);  add_230 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:349 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_57: "f32[768, 16384]" = torch.ops.aten.permute.default(view_760, [1, 0])
        sum_dim_int_list_97: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_760, [0], True);  view_760 = None
        reshape_default_65: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_97, _shape_param_65);  sum_dim_int_list_97 = _shape_param_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:331 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_58: "f32[3072, 16384]" = torch.ops.aten.permute.default(view_763, [1, 0])
        sum_dim_int_list_98: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_763, [0], True);  view_763 = None
        reshape_default_66: "f32[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_98, _shape_param_66);  sum_dim_int_list_98 = _shape_param_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:267 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_tensor_16: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_233, mul_66);  mul_66 = None
        sum_dim_int_list_99: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_16, [0, 1]);  mul_tensor_16 = None
        sum_dim_int_list_100: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_233, [0, 1]);  add_233 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:265 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_59: "f32[768, 16384]" = torch.ops.aten.permute.default(view_766, [1, 0])
        sum_dim_int_list_101: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_766, [0], True);  view_766 = None
        reshape_default_67: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_101, _shape_param_67);  sum_dim_int_list_101 = _shape_param_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:209 in forward, code: conv_out_layer = self.conv_out_layer(hidden_states)
        permute_default_60: "f32[384, 16384]" = torch.ops.aten.permute.default(view_785, [1, 0])
        sum_dim_int_list_102: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_785, [0], True);  view_785 = None
        reshape_default_68: "f32[384]" = torch.ops.aten.reshape.default(sum_dim_int_list_102, _shape_param_68);  sum_dim_int_list_102 = _shape_param_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:205 in forward, code: conv_kernel_layer = self.conv_kernel_layer(conv_attn_layer)
        sum_dim_int_list_103: "f32[1, 1, 54]" = torch.ops.aten.sum.dim_IntList(view_788, [0, 1], True);  view_788 = None
        reshape_default_69: "f32[54]" = torch.ops.aten.reshape.default(sum_dim_int_list_103, _shape_param_69);  sum_dim_int_list_103 = _shape_param_69 = None
        permute_default_61: "f32[54, 16384]" = torch.ops.aten.permute.default(view_790, [1, 0]);  view_790 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:197 in forward, code: mixed_query_layer = self.query(hidden_states)
        permute_default_62: "f32[384, 16384]" = torch.ops.aten.permute.default(view_795, [1, 0])
        sum_dim_int_list_104: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_795, [0], True);  view_795 = None
        reshape_default_70: "f32[384]" = torch.ops.aten.reshape.default(sum_dim_int_list_104, _shape_param_70);  sum_dim_int_list_104 = _shape_param_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:131 in forward, code: x += self.bias
        sum_dim_int_list_105: "f32[1, 384, 1]" = torch.ops.aten.sum.dim_IntList(permute_612, [0, 2], True);  permute_612 = None
        reshape_default_71: "f32[384, 1]" = torch.ops.aten.reshape.default(sum_dim_int_list_105, _shape_param_71);  sum_dim_int_list_105 = _shape_param_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:192 in forward, code: mixed_value_layer = self.value(hidden_states)
        permute_default_63: "f32[384, 16384]" = torch.ops.aten.permute.default(view_799, [1, 0])
        sum_dim_int_list_106: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_799, [0], True);  view_799 = None
        reshape_default_72: "f32[384]" = torch.ops.aten.reshape.default(sum_dim_int_list_106, _shape_param_72);  sum_dim_int_list_106 = _shape_param_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:191 in forward, code: mixed_key_layer = self.key(hidden_states)
        permute_default_64: "f32[384, 16384]" = torch.ops.aten.permute.default(view_802, [1, 0])
        sum_dim_int_list_107: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_802, [0], True);  view_802 = None
        reshape_default_73: "f32[384]" = torch.ops.aten.reshape.default(sum_dim_int_list_107, _shape_param_73);  sum_dim_int_list_107 = _shape_param_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:351 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_tensor_17: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_241, mul_59);  mul_59 = None
        sum_dim_int_list_108: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_17, [0, 1]);  mul_tensor_17 = None
        sum_dim_int_list_109: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_241, [0, 1]);  add_241 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:349 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_65: "f32[768, 16384]" = torch.ops.aten.permute.default(view_805, [1, 0])
        sum_dim_int_list_110: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_805, [0], True);  view_805 = None
        reshape_default_74: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_110, _shape_param_74);  sum_dim_int_list_110 = _shape_param_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:331 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_66: "f32[3072, 16384]" = torch.ops.aten.permute.default(view_808, [1, 0])
        sum_dim_int_list_111: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_808, [0], True);  view_808 = None
        reshape_default_75: "f32[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_111, _shape_param_75);  sum_dim_int_list_111 = _shape_param_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:267 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_tensor_18: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_244, mul_52);  mul_52 = None
        sum_dim_int_list_112: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_18, [0, 1]);  mul_tensor_18 = None
        sum_dim_int_list_113: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_244, [0, 1]);  add_244 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:265 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_67: "f32[768, 16384]" = torch.ops.aten.permute.default(view_811, [1, 0])
        sum_dim_int_list_114: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_811, [0], True);  view_811 = None
        reshape_default_76: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_114, _shape_param_76);  sum_dim_int_list_114 = _shape_param_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:209 in forward, code: conv_out_layer = self.conv_out_layer(hidden_states)
        permute_default_68: "f32[384, 16384]" = torch.ops.aten.permute.default(view_830, [1, 0])
        sum_dim_int_list_115: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_830, [0], True);  view_830 = None
        reshape_default_77: "f32[384]" = torch.ops.aten.reshape.default(sum_dim_int_list_115, _shape_param_77);  sum_dim_int_list_115 = _shape_param_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:205 in forward, code: conv_kernel_layer = self.conv_kernel_layer(conv_attn_layer)
        sum_dim_int_list_116: "f32[1, 1, 54]" = torch.ops.aten.sum.dim_IntList(view_833, [0, 1], True);  view_833 = None
        reshape_default_78: "f32[54]" = torch.ops.aten.reshape.default(sum_dim_int_list_116, _shape_param_78);  sum_dim_int_list_116 = _shape_param_78 = None
        permute_default_69: "f32[54, 16384]" = torch.ops.aten.permute.default(view_835, [1, 0]);  view_835 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:197 in forward, code: mixed_query_layer = self.query(hidden_states)
        permute_default_70: "f32[384, 16384]" = torch.ops.aten.permute.default(view_840, [1, 0])
        sum_dim_int_list_117: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_840, [0], True);  view_840 = None
        reshape_default_79: "f32[384]" = torch.ops.aten.reshape.default(sum_dim_int_list_117, _shape_param_79);  sum_dim_int_list_117 = _shape_param_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:131 in forward, code: x += self.bias
        sum_dim_int_list_118: "f32[1, 384, 1]" = torch.ops.aten.sum.dim_IntList(permute_660, [0, 2], True);  permute_660 = None
        reshape_default_80: "f32[384, 1]" = torch.ops.aten.reshape.default(sum_dim_int_list_118, _shape_param_80);  sum_dim_int_list_118 = _shape_param_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:192 in forward, code: mixed_value_layer = self.value(hidden_states)
        permute_default_71: "f32[384, 16384]" = torch.ops.aten.permute.default(view_844, [1, 0])
        sum_dim_int_list_119: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_844, [0], True);  view_844 = None
        reshape_default_81: "f32[384]" = torch.ops.aten.reshape.default(sum_dim_int_list_119, _shape_param_81);  sum_dim_int_list_119 = _shape_param_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:191 in forward, code: mixed_key_layer = self.key(hidden_states)
        permute_default_72: "f32[384, 16384]" = torch.ops.aten.permute.default(view_847, [1, 0])
        sum_dim_int_list_120: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_847, [0], True);  view_847 = None
        reshape_default_82: "f32[384]" = torch.ops.aten.reshape.default(sum_dim_int_list_120, _shape_param_82);  sum_dim_int_list_120 = _shape_param_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:351 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_tensor_19: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_252, mul_45);  mul_45 = None
        sum_dim_int_list_121: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_19, [0, 1]);  mul_tensor_19 = None
        sum_dim_int_list_122: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_252, [0, 1]);  add_252 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:349 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_73: "f32[768, 16384]" = torch.ops.aten.permute.default(view_850, [1, 0])
        sum_dim_int_list_123: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_850, [0], True);  view_850 = None
        reshape_default_83: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_123, _shape_param_83);  sum_dim_int_list_123 = _shape_param_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:331 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_74: "f32[3072, 16384]" = torch.ops.aten.permute.default(view_853, [1, 0])
        sum_dim_int_list_124: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_853, [0], True);  view_853 = None
        reshape_default_84: "f32[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_124, _shape_param_84);  sum_dim_int_list_124 = _shape_param_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:267 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_tensor_20: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_255, mul_38);  mul_38 = None
        sum_dim_int_list_125: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_20, [0, 1]);  mul_tensor_20 = None
        sum_dim_int_list_126: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_255, [0, 1]);  add_255 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:265 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_75: "f32[768, 16384]" = torch.ops.aten.permute.default(view_856, [1, 0])
        sum_dim_int_list_127: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_856, [0], True);  view_856 = None
        reshape_default_85: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_127, _shape_param_85);  sum_dim_int_list_127 = _shape_param_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:209 in forward, code: conv_out_layer = self.conv_out_layer(hidden_states)
        permute_default_76: "f32[384, 16384]" = torch.ops.aten.permute.default(view_875, [1, 0])
        sum_dim_int_list_128: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_875, [0], True);  view_875 = None
        reshape_default_86: "f32[384]" = torch.ops.aten.reshape.default(sum_dim_int_list_128, _shape_param_86);  sum_dim_int_list_128 = _shape_param_86 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:205 in forward, code: conv_kernel_layer = self.conv_kernel_layer(conv_attn_layer)
        sum_dim_int_list_129: "f32[1, 1, 54]" = torch.ops.aten.sum.dim_IntList(view_878, [0, 1], True);  view_878 = None
        reshape_default_87: "f32[54]" = torch.ops.aten.reshape.default(sum_dim_int_list_129, _shape_param_87);  sum_dim_int_list_129 = _shape_param_87 = None
        permute_default_77: "f32[54, 16384]" = torch.ops.aten.permute.default(view_880, [1, 0]);  view_880 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:197 in forward, code: mixed_query_layer = self.query(hidden_states)
        permute_default_78: "f32[384, 16384]" = torch.ops.aten.permute.default(view_885, [1, 0])
        sum_dim_int_list_130: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_885, [0], True);  view_885 = None
        reshape_default_88: "f32[384]" = torch.ops.aten.reshape.default(sum_dim_int_list_130, _shape_param_88);  sum_dim_int_list_130 = _shape_param_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:131 in forward, code: x += self.bias
        sum_dim_int_list_131: "f32[1, 384, 1]" = torch.ops.aten.sum.dim_IntList(permute_708, [0, 2], True);  permute_708 = None
        reshape_default_89: "f32[384, 1]" = torch.ops.aten.reshape.default(sum_dim_int_list_131, _shape_param_89);  sum_dim_int_list_131 = _shape_param_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:192 in forward, code: mixed_value_layer = self.value(hidden_states)
        permute_default_79: "f32[384, 16384]" = torch.ops.aten.permute.default(view_889, [1, 0])
        sum_dim_int_list_132: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_889, [0], True);  view_889 = None
        reshape_default_90: "f32[384]" = torch.ops.aten.reshape.default(sum_dim_int_list_132, _shape_param_90);  sum_dim_int_list_132 = _shape_param_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:191 in forward, code: mixed_key_layer = self.key(hidden_states)
        permute_default_80: "f32[384, 16384]" = torch.ops.aten.permute.default(view_892, [1, 0])
        sum_dim_int_list_133: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_892, [0], True);  view_892 = None
        reshape_default_91: "f32[384]" = torch.ops.aten.reshape.default(sum_dim_int_list_133, _shape_param_91);  sum_dim_int_list_133 = _shape_param_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:351 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_tensor_21: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_263, mul_31);  mul_31 = None
        sum_dim_int_list_134: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_21, [0, 1]);  mul_tensor_21 = None
        sum_dim_int_list_135: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_263, [0, 1]);  add_263 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:349 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_81: "f32[768, 16384]" = torch.ops.aten.permute.default(view_895, [1, 0])
        sum_dim_int_list_136: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_895, [0], True);  view_895 = None
        reshape_default_92: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_136, _shape_param_92);  sum_dim_int_list_136 = _shape_param_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:331 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_82: "f32[3072, 16384]" = torch.ops.aten.permute.default(view_898, [1, 0])
        sum_dim_int_list_137: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_898, [0], True);  view_898 = None
        reshape_default_93: "f32[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_137, _shape_param_93);  sum_dim_int_list_137 = _shape_param_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:267 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_tensor_22: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_266, mul_24);  mul_24 = None
        sum_dim_int_list_138: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_22, [0, 1]);  mul_tensor_22 = None
        sum_dim_int_list_139: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_266, [0, 1]);  add_266 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:265 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_83: "f32[768, 16384]" = torch.ops.aten.permute.default(view_901, [1, 0])
        sum_dim_int_list_140: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_901, [0], True);  view_901 = None
        reshape_default_94: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_140, _shape_param_94);  sum_dim_int_list_140 = _shape_param_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:209 in forward, code: conv_out_layer = self.conv_out_layer(hidden_states)
        permute_default_84: "f32[384, 16384]" = torch.ops.aten.permute.default(view_920, [1, 0])
        sum_dim_int_list_141: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_920, [0], True);  view_920 = None
        reshape_default_95: "f32[384]" = torch.ops.aten.reshape.default(sum_dim_int_list_141, _shape_param_95);  sum_dim_int_list_141 = _shape_param_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:205 in forward, code: conv_kernel_layer = self.conv_kernel_layer(conv_attn_layer)
        sum_dim_int_list_142: "f32[1, 1, 54]" = torch.ops.aten.sum.dim_IntList(view_923, [0, 1], True);  view_923 = None
        reshape_default_96: "f32[54]" = torch.ops.aten.reshape.default(sum_dim_int_list_142, _shape_param_96);  sum_dim_int_list_142 = _shape_param_96 = None
        permute_default_85: "f32[54, 16384]" = torch.ops.aten.permute.default(view_925, [1, 0]);  view_925 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:197 in forward, code: mixed_query_layer = self.query(hidden_states)
        permute_default_86: "f32[384, 16384]" = torch.ops.aten.permute.default(view_930, [1, 0])
        sum_dim_int_list_143: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_930, [0], True);  view_930 = None
        reshape_default_97: "f32[384]" = torch.ops.aten.reshape.default(sum_dim_int_list_143, _shape_param_97);  sum_dim_int_list_143 = _shape_param_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:131 in forward, code: x += self.bias
        sum_dim_int_list_144: "f32[1, 384, 1]" = torch.ops.aten.sum.dim_IntList(permute_756, [0, 2], True);  permute_756 = None
        reshape_default_98: "f32[384, 1]" = torch.ops.aten.reshape.default(sum_dim_int_list_144, _shape_param_98);  sum_dim_int_list_144 = _shape_param_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:192 in forward, code: mixed_value_layer = self.value(hidden_states)
        permute_default_87: "f32[384, 16384]" = torch.ops.aten.permute.default(view_934, [1, 0])
        sum_dim_int_list_145: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_934, [0], True);  view_934 = None
        reshape_default_99: "f32[384]" = torch.ops.aten.reshape.default(sum_dim_int_list_145, _shape_param_99);  sum_dim_int_list_145 = _shape_param_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:191 in forward, code: mixed_key_layer = self.key(hidden_states)
        permute_default_88: "f32[384, 16384]" = torch.ops.aten.permute.default(view_937, [1, 0])
        sum_dim_int_list_146: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_937, [0], True);  view_937 = None
        reshape_default_100: "f32[384]" = torch.ops.aten.reshape.default(sum_dim_int_list_146, _shape_param_100);  sum_dim_int_list_146 = _shape_param_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:351 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_tensor_23: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_274, mul_17);  mul_17 = None
        sum_dim_int_list_147: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_23, [0, 1]);  mul_tensor_23 = None
        sum_dim_int_list_148: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_274, [0, 1]);  add_274 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:349 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_89: "f32[768, 16384]" = torch.ops.aten.permute.default(view_940, [1, 0])
        sum_dim_int_list_149: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_940, [0], True);  view_940 = None
        reshape_default_101: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_149, _shape_param_101);  sum_dim_int_list_149 = _shape_param_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:331 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_90: "f32[3072, 16384]" = torch.ops.aten.permute.default(view_943, [1, 0])
        sum_dim_int_list_150: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_943, [0], True);  view_943 = None
        reshape_default_102: "f32[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_150, _shape_param_102);  sum_dim_int_list_150 = _shape_param_102 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:267 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_tensor_24: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_277, mul_10);  mul_10 = None
        sum_dim_int_list_151: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_24, [0, 1]);  mul_tensor_24 = None
        sum_dim_int_list_152: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_277, [0, 1]);  add_277 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:265 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_91: "f32[768, 16384]" = torch.ops.aten.permute.default(view_946, [1, 0])
        sum_dim_int_list_153: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_946, [0], True);  view_946 = None
        reshape_default_103: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_153, _shape_param_103);  sum_dim_int_list_153 = _shape_param_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:209 in forward, code: conv_out_layer = self.conv_out_layer(hidden_states)
        permute_default_92: "f32[384, 16384]" = torch.ops.aten.permute.default(view_965, [1, 0])
        sum_dim_int_list_154: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_965, [0], True);  view_965 = None
        reshape_default_104: "f32[384]" = torch.ops.aten.reshape.default(sum_dim_int_list_154, _shape_param_104);  sum_dim_int_list_154 = _shape_param_104 = None
        reshape_default_105: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_198, _shape_param_105);  mm_198 = _shape_param_105 = None
        add_tensor: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_567, reshape_default_105);  mul_567 = reshape_default_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:205 in forward, code: conv_kernel_layer = self.conv_kernel_layer(conv_attn_layer)
        sum_dim_int_list_155: "f32[1, 1, 54]" = torch.ops.aten.sum.dim_IntList(view_968, [0, 1], True);  view_968 = None
        reshape_default_106: "f32[54]" = torch.ops.aten.reshape.default(sum_dim_int_list_155, _shape_param_106);  sum_dim_int_list_155 = _shape_param_106 = None
        permute_default_93: "f32[54, 16384]" = torch.ops.aten.permute.default(view_970, [1, 0]);  view_970 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:197 in forward, code: mixed_query_layer = self.query(hidden_states)
        permute_default_94: "f32[384, 16384]" = torch.ops.aten.permute.default(view_975, [1, 0])
        sum_dim_int_list_156: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_975, [0], True);  view_975 = None
        reshape_default_107: "f32[384]" = torch.ops.aten.reshape.default(sum_dim_int_list_156, _shape_param_107);  sum_dim_int_list_156 = _shape_param_107 = None
        reshape_default_108: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_202, _shape_param_108);  mm_202 = _shape_param_108 = None
        add_tensor_1: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_tensor, reshape_default_108);  add_tensor = reshape_default_108 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:131 in forward, code: x += self.bias
        sum_dim_int_list_157: "f32[1, 384, 1]" = torch.ops.aten.sum.dim_IntList(permute_804, [0, 2], True);  permute_804 = None
        reshape_default_109: "f32[384, 1]" = torch.ops.aten.reshape.default(sum_dim_int_list_157, _shape_param_109);  sum_dim_int_list_157 = _shape_param_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:194 in forward, code: mixed_key_conv_attn_layer = self.key_conv_attn_layer(hidden_states.transpose(1, 2))
        permute_default_95: "f32[32, 512, 768]" = torch.ops.aten.permute.default(getitem_121, [0, 2, 1]);  getitem_121 = None
        add_tensor_2: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_tensor_1, permute_default_95);  add_tensor_1 = permute_default_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:192 in forward, code: mixed_value_layer = self.value(hidden_states)
        permute_default_96: "f32[384, 16384]" = torch.ops.aten.permute.default(view_979, [1, 0])
        sum_dim_int_list_158: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_979, [0], True);  view_979 = None
        reshape_default_110: "f32[384]" = torch.ops.aten.reshape.default(sum_dim_int_list_158, _shape_param_110);  sum_dim_int_list_158 = _shape_param_110 = None
        reshape_default_111: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_204, _shape_param_111);  mm_204 = _shape_param_111 = None
        add_tensor_3: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_tensor_2, reshape_default_111);  add_tensor_2 = reshape_default_111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:191 in forward, code: mixed_key_layer = self.key(hidden_states)
        permute_default_97: "f32[384, 16384]" = torch.ops.aten.permute.default(view_982, [1, 0])
        sum_dim_int_list_159: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_982, [0], True);  view_982 = None
        reshape_default_112: "f32[384]" = torch.ops.aten.reshape.default(sum_dim_int_list_159, _shape_param_112);  sum_dim_int_list_159 = _shape_param_112 = None
        reshape_default_113: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_206, _shape_param_113);  mm_206 = _shape_param_113 = None
        add_tensor_4: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_tensor_3, reshape_default_113);  add_tensor_3 = reshape_default_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:105 in forward, code: embeddings = self.dropout(embeddings)
        convert_element_type_default: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(gt, torch.float32);  gt = None
        mul_tensor_25: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 1.0);  convert_element_type_default = None
        mul_tensor_26: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_tensor_4, mul_tensor_25);  add_tensor_4 = mul_tensor_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:104 in forward, code: embeddings = self.LayerNorm(embeddings)
        mul_tensor_27: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_26, primals_7);  primals_7 = None
        mul_tensor_28: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_27, 768)
        sum_dim_int_list_160: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_27, [2], True)
        mul_tensor_29: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_27, mul_1);  mul_tensor_27 = None
        sum_dim_int_list_161: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_29, [2], True);  mul_tensor_29 = None
        mul_tensor_30: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_1, sum_dim_int_list_161);  sum_dim_int_list_161 = None
        sub_tensor: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_tensor_28, sum_dim_int_list_160);  mul_tensor_28 = sum_dim_int_list_160 = None
        sub_tensor_1: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub_tensor, mul_tensor_30);  sub_tensor = mul_tensor_30 = None
        mul_tensor_31: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(div_75, sub_tensor_1);  div_75 = sub_tensor_1 = None
        mul_tensor_32: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_26, mul_1);  mul_1 = None
        sum_dim_int_list_162: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_32, [0, 1]);  mul_tensor_32 = None
        sum_dim_int_list_163: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_26, [0, 1]);  mul_tensor_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:103 in forward, code: embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        sum_dim_int_list_164: "f32[1, 512, 768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_31, [0], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:628 in forward, code: buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
        expand_default: "i64[32, 512]" = torch.ops.aten.expand.default(primals_2, _shape_param_114);  primals_2 = _shape_param_114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:101 in forward, code: token_type_embeddings = self.token_type_embeddings(token_type_ids)
        eq_scalar: "b8[32, 512]" = torch.ops.aten.eq.Scalar(expand_default, -1)
        unsqueeze_default: "b8[32, 512, 1]" = torch.ops.aten.unsqueeze.default(eq_scalar, -1);  eq_scalar = None
        where_self: "f32[32, 512, 768]" = torch.ops.aten.where.self(unsqueeze_default, full_default_2, mul_tensor_31);  unsqueeze_default = None
        full_default: "f32[2, 768]" = torch.ops.aten.full.default([2, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default: "f32[2, 768]" = torch.ops.aten.index_put.default(full_default, [expand_default], where_self, True);  full_default = expand_default = where_self = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:100 in forward, code: position_embeddings = self.position_embeddings(position_ids)
        eq_scalar_1: "b8[1, 512]" = torch.ops.aten.eq.Scalar(primals_4, -1)
        unsqueeze_default_1: "b8[1, 512, 1]" = torch.ops.aten.unsqueeze.default(eq_scalar_1, -1);  eq_scalar_1 = None
        where_self_1: "f32[1, 512, 768]" = torch.ops.aten.where.self(unsqueeze_default_1, full_default_2, sum_dim_int_list_164);  unsqueeze_default_1 = sum_dim_int_list_164 = None
        full_default_3: "f32[512, 768]" = torch.ops.aten.full.default([512, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default_1: "f32[512, 768]" = torch.ops.aten.index_put.default(full_default_3, [primals_4], where_self_1, True);  full_default_3 = primals_4 = where_self_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:99 in forward, code: inputs_embeds = self.word_embeddings(input_ids)
        eq_scalar_2: "b8[32, 512]" = torch.ops.aten.eq.Scalar(primals_1, 0)
        unsqueeze_default_2: "b8[32, 512, 1]" = torch.ops.aten.unsqueeze.default(eq_scalar_2, -1);  eq_scalar_2 = None
        where_self_2: "f32[32, 512, 768]" = torch.ops.aten.where.self(unsqueeze_default_2, full_default_2, mul_tensor_31);  unsqueeze_default_2 = full_default_2 = mul_tensor_31 = None
        full_default_4: "f32[30522, 768]" = torch.ops.aten.full.default([30522, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default_2: "f32[30522, 768]" = torch.ops.aten.index_put.default(full_default_4, [primals_1], where_self_2, True);  full_default_4 = primals_1 = where_self_2 = None
        add_tensor_5: "f32[30522, 768]" = torch.ops.aten.add.Tensor(slice_tensor, index_put_default_2);  slice_tensor = index_put_default_2 = None
        return (reshape_default, sum_dim_int_list_1, sum_dim_int_list_2, permute_default, reshape_default_1, sum_dim_int_list_4, sum_dim_int_list_5, permute_default_1, reshape_default_2, permute_default_2, reshape_default_3, sum_dim_int_list_8, sum_dim_int_list_9, permute_default_3, reshape_default_4, permute_default_4, reshape_default_5, reshape_default_6, permute_default_5, permute_default_6, reshape_default_7, reshape_default_8, permute_default_7, reshape_default_9, permute_default_8, reshape_default_10, sum_dim_int_list_17, sum_dim_int_list_18, permute_default_9, reshape_default_11, permute_default_10, reshape_default_12, sum_dim_int_list_21, sum_dim_int_list_22, permute_default_11, reshape_default_13, permute_default_12, reshape_default_14, reshape_default_15, permute_default_13, permute_default_14, reshape_default_16, reshape_default_17, permute_default_15, reshape_default_18, permute_default_16, reshape_default_19, sum_dim_int_list_30, sum_dim_int_list_31, permute_default_17, reshape_default_20, permute_default_18, reshape_default_21, sum_dim_int_list_34, sum_dim_int_list_35, permute_default_19, reshape_default_22, permute_default_20, reshape_default_23, reshape_default_24, permute_default_21, permute_default_22, reshape_default_25, reshape_default_26, permute_default_23, reshape_default_27, permute_default_24, reshape_default_28, sum_dim_int_list_43, sum_dim_int_list_44, permute_default_25, reshape_default_29, permute_default_26, reshape_default_30, sum_dim_int_list_47, sum_dim_int_list_48, permute_default_27, reshape_default_31, permute_default_28, reshape_default_32, reshape_default_33, permute_default_29, permute_default_30, reshape_default_34, reshape_default_35, permute_default_31, reshape_default_36, permute_default_32, reshape_default_37, sum_dim_int_list_56, sum_dim_int_list_57, permute_default_33, reshape_default_38, permute_default_34, reshape_default_39, sum_dim_int_list_60, sum_dim_int_list_61, permute_default_35, reshape_default_40, permute_default_36, reshape_default_41, reshape_default_42, permute_default_37, permute_default_38, reshape_default_43, reshape_default_44, permute_default_39, reshape_default_45, permute_default_40, reshape_default_46, sum_dim_int_list_69, sum_dim_int_list_70, permute_default_41, reshape_default_47, permute_default_42, reshape_default_48, sum_dim_int_list_73, sum_dim_int_list_74, permute_default_43, reshape_default_49, permute_default_44, reshape_default_50, reshape_default_51, permute_default_45, permute_default_46, reshape_default_52, reshape_default_53, permute_default_47, reshape_default_54, permute_default_48, reshape_default_55, sum_dim_int_list_82, sum_dim_int_list_83, permute_default_49, reshape_default_56, permute_default_50, reshape_default_57, sum_dim_int_list_86, sum_dim_int_list_87, permute_default_51, reshape_default_58, permute_default_52, reshape_default_59, reshape_default_60, permute_default_53, permute_default_54, reshape_default_61, reshape_default_62, permute_default_55, reshape_default_63, permute_default_56, reshape_default_64, sum_dim_int_list_95, sum_dim_int_list_96, permute_default_57, reshape_default_65, permute_default_58, reshape_default_66, sum_dim_int_list_99, sum_dim_int_list_100, permute_default_59, reshape_default_67, permute_default_60, reshape_default_68, reshape_default_69, permute_default_61, permute_default_62, reshape_default_70, reshape_default_71, permute_default_63, reshape_default_72, permute_default_64, reshape_default_73, sum_dim_int_list_108, sum_dim_int_list_109, permute_default_65, reshape_default_74, permute_default_66, reshape_default_75, sum_dim_int_list_112, sum_dim_int_list_113, permute_default_67, reshape_default_76, permute_default_68, reshape_default_77, reshape_default_78, permute_default_69, permute_default_70, reshape_default_79, reshape_default_80, permute_default_71, reshape_default_81, permute_default_72, reshape_default_82, sum_dim_int_list_121, sum_dim_int_list_122, permute_default_73, reshape_default_83, permute_default_74, reshape_default_84, sum_dim_int_list_125, sum_dim_int_list_126, permute_default_75, reshape_default_85, permute_default_76, reshape_default_86, reshape_default_87, permute_default_77, permute_default_78, reshape_default_88, reshape_default_89, permute_default_79, reshape_default_90, permute_default_80, reshape_default_91, sum_dim_int_list_134, sum_dim_int_list_135, permute_default_81, reshape_default_92, permute_default_82, reshape_default_93, sum_dim_int_list_138, sum_dim_int_list_139, permute_default_83, reshape_default_94, permute_default_84, reshape_default_95, reshape_default_96, permute_default_85, permute_default_86, reshape_default_97, reshape_default_98, permute_default_87, reshape_default_99, permute_default_88, reshape_default_100, sum_dim_int_list_147, sum_dim_int_list_148, permute_default_89, reshape_default_101, permute_default_90, reshape_default_102, sum_dim_int_list_151, sum_dim_int_list_152, permute_default_91, reshape_default_103, permute_default_92, reshape_default_104, reshape_default_106, permute_default_93, permute_default_94, reshape_default_107, reshape_default_109, permute_default_96, reshape_default_110, permute_default_97, reshape_default_112, sum_dim_int_list_162, sum_dim_int_list_163, index_put_default, index_put_default_1, add_tensor_5)
