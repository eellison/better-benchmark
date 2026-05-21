class GraphModule(torch.nn.Module):
    def forward(self, mm_default_1: "f32[50260, 2048]", view_535: "f32[32, 128, 2048]", mul_192: "f32[32, 128, 2048]", view_537: "f32[4096, 2048]", view_540: "f32[4096, 8192]", view_542: "f32[32, 128, 2048]", mul_186: "f32[32, 128, 2048]", view_543: "f32[4096, 2048]", view_556: "f32[4096, 2048]", view_558: "f32[4096, 2048]", view_560: "f32[4096, 2048]", add_227: "f32[32, 128, 2048]", mul_184: "f32[32, 128, 2048]", view_562: "f32[4096, 2048]", view_565: "f32[4096, 8192]", view_567: "f32[32, 128, 2048]", mul_178: "f32[32, 128, 2048]", view_568: "f32[4096, 2048]", view_581: "f32[4096, 2048]", view_583: "f32[4096, 2048]", view_585: "f32[4096, 2048]", add_233: "f32[32, 128, 2048]", mul_176: "f32[32, 128, 2048]", view_587: "f32[4096, 2048]", view_590: "f32[4096, 8192]", view_592: "f32[32, 128, 2048]", mul_170: "f32[32, 128, 2048]", view_593: "f32[4096, 2048]", view_606: "f32[4096, 2048]", view_608: "f32[4096, 2048]", view_610: "f32[4096, 2048]", add_239: "f32[32, 128, 2048]", mul_168: "f32[32, 128, 2048]", view_612: "f32[4096, 2048]", view_615: "f32[4096, 8192]", view_617: "f32[32, 128, 2048]", mul_162: "f32[32, 128, 2048]", view_618: "f32[4096, 2048]", view_631: "f32[4096, 2048]", view_633: "f32[4096, 2048]", view_635: "f32[4096, 2048]", add_245: "f32[32, 128, 2048]", mul_160: "f32[32, 128, 2048]", view_637: "f32[4096, 2048]", view_640: "f32[4096, 8192]", view_642: "f32[32, 128, 2048]", mul_154: "f32[32, 128, 2048]", view_643: "f32[4096, 2048]", view_656: "f32[4096, 2048]", view_658: "f32[4096, 2048]", view_660: "f32[4096, 2048]", add_251: "f32[32, 128, 2048]", mul_152: "f32[32, 128, 2048]", view_662: "f32[4096, 2048]", view_665: "f32[4096, 8192]", view_667: "f32[32, 128, 2048]", mul_146: "f32[32, 128, 2048]", view_668: "f32[4096, 2048]", view_681: "f32[4096, 2048]", view_683: "f32[4096, 2048]", view_685: "f32[4096, 2048]", add_257: "f32[32, 128, 2048]", mul_144: "f32[32, 128, 2048]", view_687: "f32[4096, 2048]", view_690: "f32[4096, 8192]", view_692: "f32[32, 128, 2048]", mul_138: "f32[32, 128, 2048]", view_693: "f32[4096, 2048]", view_706: "f32[4096, 2048]", view_708: "f32[4096, 2048]", view_710: "f32[4096, 2048]", add_263: "f32[32, 128, 2048]", mul_136: "f32[32, 128, 2048]", view_712: "f32[4096, 2048]", view_715: "f32[4096, 8192]", view_717: "f32[32, 128, 2048]", mul_130: "f32[32, 128, 2048]", view_718: "f32[4096, 2048]", view_731: "f32[4096, 2048]", view_733: "f32[4096, 2048]", view_735: "f32[4096, 2048]", add_269: "f32[32, 128, 2048]", mul_128: "f32[32, 128, 2048]", view_737: "f32[4096, 2048]", view_740: "f32[4096, 8192]", view_742: "f32[32, 128, 2048]", mul_122: "f32[32, 128, 2048]", view_743: "f32[4096, 2048]", view_756: "f32[4096, 2048]", view_758: "f32[4096, 2048]", view_760: "f32[4096, 2048]", add_275: "f32[32, 128, 2048]", mul_120: "f32[32, 128, 2048]", view_762: "f32[4096, 2048]", view_765: "f32[4096, 8192]", view_767: "f32[32, 128, 2048]", mul_114: "f32[32, 128, 2048]", view_768: "f32[4096, 2048]", view_781: "f32[4096, 2048]", view_783: "f32[4096, 2048]", view_785: "f32[4096, 2048]", add_281: "f32[32, 128, 2048]", mul_112: "f32[32, 128, 2048]", view_787: "f32[4096, 2048]", view_790: "f32[4096, 8192]", view_792: "f32[32, 128, 2048]", mul_106: "f32[32, 128, 2048]", view_793: "f32[4096, 2048]", view_806: "f32[4096, 2048]", view_808: "f32[4096, 2048]", view_810: "f32[4096, 2048]", add_287: "f32[32, 128, 2048]", mul_104: "f32[32, 128, 2048]", view_812: "f32[4096, 2048]", view_815: "f32[4096, 8192]", view_817: "f32[32, 128, 2048]", mul_98: "f32[32, 128, 2048]", view_818: "f32[4096, 2048]", view_831: "f32[4096, 2048]", view_833: "f32[4096, 2048]", view_835: "f32[4096, 2048]", add_293: "f32[32, 128, 2048]", mul_96: "f32[32, 128, 2048]", view_837: "f32[4096, 2048]", view_840: "f32[4096, 8192]", view_842: "f32[32, 128, 2048]", mul_90: "f32[32, 128, 2048]", view_843: "f32[4096, 2048]", view_856: "f32[4096, 2048]", view_858: "f32[4096, 2048]", view_860: "f32[4096, 2048]", add_299: "f32[32, 128, 2048]", mul_88: "f32[32, 128, 2048]", view_862: "f32[4096, 2048]", view_865: "f32[4096, 8192]", view_867: "f32[32, 128, 2048]", mul_82: "f32[32, 128, 2048]", view_868: "f32[4096, 2048]", view_881: "f32[4096, 2048]", view_883: "f32[4096, 2048]", view_885: "f32[4096, 2048]", add_305: "f32[32, 128, 2048]", mul_80: "f32[32, 128, 2048]", view_887: "f32[4096, 2048]", view_890: "f32[4096, 8192]", view_892: "f32[32, 128, 2048]", mul_74: "f32[32, 128, 2048]", view_893: "f32[4096, 2048]", view_906: "f32[4096, 2048]", view_908: "f32[4096, 2048]", view_910: "f32[4096, 2048]", add_311: "f32[32, 128, 2048]", mul_72: "f32[32, 128, 2048]", view_912: "f32[4096, 2048]", view_915: "f32[4096, 8192]", view_917: "f32[32, 128, 2048]", mul_66: "f32[32, 128, 2048]", view_918: "f32[4096, 2048]", view_931: "f32[4096, 2048]", view_933: "f32[4096, 2048]", view_935: "f32[4096, 2048]", add_317: "f32[32, 128, 2048]", mul_64: "f32[32, 128, 2048]", view_937: "f32[4096, 2048]", view_940: "f32[4096, 8192]", view_942: "f32[32, 128, 2048]", mul_58: "f32[32, 128, 2048]", view_943: "f32[4096, 2048]", view_956: "f32[4096, 2048]", view_958: "f32[4096, 2048]", view_960: "f32[4096, 2048]", add_323: "f32[32, 128, 2048]", mul_56: "f32[32, 128, 2048]", view_962: "f32[4096, 2048]", view_965: "f32[4096, 8192]", view_967: "f32[32, 128, 2048]", mul_50: "f32[32, 128, 2048]", view_968: "f32[4096, 2048]", view_981: "f32[4096, 2048]", view_983: "f32[4096, 2048]", view_985: "f32[4096, 2048]", add_329: "f32[32, 128, 2048]", mul_48: "f32[32, 128, 2048]", view_987: "f32[4096, 2048]", view_990: "f32[4096, 8192]", view_992: "f32[32, 128, 2048]", mul_42: "f32[32, 128, 2048]", view_993: "f32[4096, 2048]", view_1006: "f32[4096, 2048]", view_1008: "f32[4096, 2048]", view_1010: "f32[4096, 2048]", add_335: "f32[32, 128, 2048]", mul_40: "f32[32, 128, 2048]", view_1012: "f32[4096, 2048]", view_1015: "f32[4096, 8192]", view_1017: "f32[32, 128, 2048]", mul_34: "f32[32, 128, 2048]", view_1018: "f32[4096, 2048]", view_1031: "f32[4096, 2048]", view_1033: "f32[4096, 2048]", view_1035: "f32[4096, 2048]", add_341: "f32[32, 128, 2048]", mul_32: "f32[32, 128, 2048]", view_1037: "f32[4096, 2048]", view_1040: "f32[4096, 8192]", view_1042: "f32[32, 128, 2048]", mul_26: "f32[32, 128, 2048]", view_1043: "f32[4096, 2048]", view_1056: "f32[4096, 2048]", view_1058: "f32[4096, 2048]", view_1060: "f32[4096, 2048]", add_347: "f32[32, 128, 2048]", mul_24: "f32[32, 128, 2048]", view_1062: "f32[4096, 2048]", view_1065: "f32[4096, 8192]", view_1067: "f32[32, 128, 2048]", mul_18: "f32[32, 128, 2048]", view_1068: "f32[4096, 2048]", view_1081: "f32[4096, 2048]", view_1083: "f32[4096, 2048]", view_1085: "f32[4096, 2048]", add_353: "f32[32, 128, 2048]", mul_16: "f32[32, 128, 2048]", view_1087: "f32[4096, 2048]", view_1090: "f32[4096, 8192]", view_1092: "f32[32, 128, 2048]", mul_10: "f32[32, 128, 2048]", view_1093: "f32[4096, 2048]", view_1106: "f32[4096, 2048]", view_1108: "f32[4096, 2048]", view_1110: "f32[4096, 2048]", add_359: "f32[32, 128, 2048]", mul_8: "f32[32, 128, 2048]", view_1112: "f32[4096, 2048]", view_1115: "f32[4096, 8192]", view_1117: "f32[32, 128, 2048]", mul_2: "f32[32, 128, 2048]", view_1118: "f32[4096, 2048]", view_1131: "f32[4096, 2048]", mm_358: "f32[4096, 2048]", view_1133: "f32[4096, 2048]", mm_360: "f32[4096, 2048]", view_1135: "f32[4096, 2048]", mm_362: "f32[4096, 2048]", primals_4: "f32[2048]", embedding: "f32[32, 128, 2048]", embedding_1: "f32[1, 128, 2048]", getitem_1: "f32[32, 128, 1]", rsqrt: "f32[32, 128, 1]", add_363: "f32[32, 128, 2048]", unsqueeze: "i64[1, 128]", full_default_1: "f32[]", primals_1: "i64[32, 128]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12, _shape_param_13, _shape_param_14, _shape_param_15, _shape_param_16, _shape_param_17, _shape_param_18, _shape_param_19, _shape_param_20, _shape_param_21, _shape_param_22, _shape_param_23, _shape_param_24, _shape_param_25, _shape_param_26, _shape_param_27, _shape_param_28, _shape_param_29, _shape_param_30, _shape_param_31, _shape_param_32, _shape_param_33, _shape_param_34, _shape_param_35, _shape_param_36, _shape_param_37, _shape_param_38, _shape_param_39, _shape_param_40, _shape_param_41, _shape_param_42, _shape_param_43, _shape_param_44, _shape_param_45, _shape_param_46, _shape_param_47, _shape_param_48, _shape_param_49, _shape_param_50, _shape_param_51, _shape_param_52, _shape_param_53, _shape_param_54, _shape_param_55, _shape_param_56, _shape_param_57, _shape_param_58, _shape_param_59, _shape_param_60, _shape_param_61, _shape_param_62, _shape_param_63, _shape_param_64, _shape_param_65, _shape_param_66, _shape_param_67, _shape_param_68, _shape_param_69, _shape_param_70, _shape_param_71, _shape_param_72, _shape_param_73, _shape_param_74):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:583 in forward, code: logits = self.lm_head(hidden_states[:, slice_indices, :])
        slice_tensor: "f32[50257, 2048]" = torch.ops.aten.slice.Tensor(mm_default_1, 0, 0, -3);  mm_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:492 in forward, code: hidden_states = self.ln_f(hidden_states)
        mul_tensor: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(view_535, mul_192);  mul_192 = None
        sum_dim_int_list: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [0, 1]);  mul_tensor = None
        sum_dim_int_list_1: "f32[2048]" = torch.ops.aten.sum.dim_IntList(view_535, [0, 1]);  view_535 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:307 in forward, code: hidden_states = self.c_proj(hidden_states)
        permute_default: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_537, [1, 0])
        sum_dim_int_list_2: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_537, [0], True);  view_537 = None
        reshape_default: "f32[2048]" = torch.ops.aten.reshape.default(sum_dim_int_list_2, _shape_param_0);  sum_dim_int_list_2 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        permute_default_1: "f32[8192, 4096]" = torch.ops.aten.permute.default(view_540, [1, 0])
        sum_dim_int_list_3: "f32[1, 8192]" = torch.ops.aten.sum.dim_IntList(view_540, [0], True);  view_540 = None
        reshape_default_1: "f32[8192]" = torch.ops.aten.reshape.default(sum_dim_int_list_3, _shape_param_1);  sum_dim_int_list_3 = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_tensor_1: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(view_542, mul_186);  mul_186 = None
        sum_dim_int_list_4: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_tensor_1, [0, 1]);  mul_tensor_1 = None
        sum_dim_int_list_5: "f32[2048]" = torch.ops.aten.sum.dim_IntList(view_542, [0, 1]);  view_542 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:155 in forward, code: attn_output = self.out_proj(attn_output)
        permute_default_2: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_543, [1, 0])
        sum_dim_int_list_6: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_543, [0], True);  view_543 = None
        reshape_default_2: "f32[2048]" = torch.ops.aten.reshape.default(sum_dim_int_list_6, _shape_param_2);  sum_dim_int_list_6 = _shape_param_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:143 in forward, code: value = self.v_proj(hidden_states)
        permute_default_3: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_556, [1, 0]);  view_556 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:142 in forward, code: key = self.k_proj(hidden_states)
        permute_default_4: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_558, [1, 0]);  view_558 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:141 in forward, code: query = self.q_proj(hidden_states)
        permute_default_5: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_560, [1, 0]);  view_560 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_tensor_2: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_227, mul_184);  mul_184 = None
        sum_dim_int_list_7: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [0, 1]);  mul_tensor_2 = None
        sum_dim_int_list_8: "f32[2048]" = torch.ops.aten.sum.dim_IntList(add_227, [0, 1]);  add_227 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:307 in forward, code: hidden_states = self.c_proj(hidden_states)
        permute_default_6: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_562, [1, 0])
        sum_dim_int_list_9: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_562, [0], True);  view_562 = None
        reshape_default_3: "f32[2048]" = torch.ops.aten.reshape.default(sum_dim_int_list_9, _shape_param_3);  sum_dim_int_list_9 = _shape_param_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        permute_default_7: "f32[8192, 4096]" = torch.ops.aten.permute.default(view_565, [1, 0])
        sum_dim_int_list_10: "f32[1, 8192]" = torch.ops.aten.sum.dim_IntList(view_565, [0], True);  view_565 = None
        reshape_default_4: "f32[8192]" = torch.ops.aten.reshape.default(sum_dim_int_list_10, _shape_param_4);  sum_dim_int_list_10 = _shape_param_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_tensor_3: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(view_567, mul_178);  mul_178 = None
        sum_dim_int_list_11: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_tensor_3, [0, 1]);  mul_tensor_3 = None
        sum_dim_int_list_12: "f32[2048]" = torch.ops.aten.sum.dim_IntList(view_567, [0, 1]);  view_567 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:155 in forward, code: attn_output = self.out_proj(attn_output)
        permute_default_8: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_568, [1, 0])
        sum_dim_int_list_13: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_568, [0], True);  view_568 = None
        reshape_default_5: "f32[2048]" = torch.ops.aten.reshape.default(sum_dim_int_list_13, _shape_param_5);  sum_dim_int_list_13 = _shape_param_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:143 in forward, code: value = self.v_proj(hidden_states)
        permute_default_9: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_581, [1, 0]);  view_581 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:142 in forward, code: key = self.k_proj(hidden_states)
        permute_default_10: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_583, [1, 0]);  view_583 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:141 in forward, code: query = self.q_proj(hidden_states)
        permute_default_11: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_585, [1, 0]);  view_585 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_tensor_4: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_233, mul_176);  mul_176 = None
        sum_dim_int_list_14: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_tensor_4, [0, 1]);  mul_tensor_4 = None
        sum_dim_int_list_15: "f32[2048]" = torch.ops.aten.sum.dim_IntList(add_233, [0, 1]);  add_233 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:307 in forward, code: hidden_states = self.c_proj(hidden_states)
        permute_default_12: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_587, [1, 0])
        sum_dim_int_list_16: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_587, [0], True);  view_587 = None
        reshape_default_6: "f32[2048]" = torch.ops.aten.reshape.default(sum_dim_int_list_16, _shape_param_6);  sum_dim_int_list_16 = _shape_param_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        permute_default_13: "f32[8192, 4096]" = torch.ops.aten.permute.default(view_590, [1, 0])
        sum_dim_int_list_17: "f32[1, 8192]" = torch.ops.aten.sum.dim_IntList(view_590, [0], True);  view_590 = None
        reshape_default_7: "f32[8192]" = torch.ops.aten.reshape.default(sum_dim_int_list_17, _shape_param_7);  sum_dim_int_list_17 = _shape_param_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_tensor_5: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(view_592, mul_170);  mul_170 = None
        sum_dim_int_list_18: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_tensor_5, [0, 1]);  mul_tensor_5 = None
        sum_dim_int_list_19: "f32[2048]" = torch.ops.aten.sum.dim_IntList(view_592, [0, 1]);  view_592 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:155 in forward, code: attn_output = self.out_proj(attn_output)
        permute_default_14: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_593, [1, 0])
        sum_dim_int_list_20: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_593, [0], True);  view_593 = None
        reshape_default_8: "f32[2048]" = torch.ops.aten.reshape.default(sum_dim_int_list_20, _shape_param_8);  sum_dim_int_list_20 = _shape_param_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:143 in forward, code: value = self.v_proj(hidden_states)
        permute_default_15: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_606, [1, 0]);  view_606 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:142 in forward, code: key = self.k_proj(hidden_states)
        permute_default_16: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_608, [1, 0]);  view_608 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:141 in forward, code: query = self.q_proj(hidden_states)
        permute_default_17: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_610, [1, 0]);  view_610 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_tensor_6: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_239, mul_168);  mul_168 = None
        sum_dim_int_list_21: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_tensor_6, [0, 1]);  mul_tensor_6 = None
        sum_dim_int_list_22: "f32[2048]" = torch.ops.aten.sum.dim_IntList(add_239, [0, 1]);  add_239 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:307 in forward, code: hidden_states = self.c_proj(hidden_states)
        permute_default_18: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_612, [1, 0])
        sum_dim_int_list_23: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_612, [0], True);  view_612 = None
        reshape_default_9: "f32[2048]" = torch.ops.aten.reshape.default(sum_dim_int_list_23, _shape_param_9);  sum_dim_int_list_23 = _shape_param_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        permute_default_19: "f32[8192, 4096]" = torch.ops.aten.permute.default(view_615, [1, 0])
        sum_dim_int_list_24: "f32[1, 8192]" = torch.ops.aten.sum.dim_IntList(view_615, [0], True);  view_615 = None
        reshape_default_10: "f32[8192]" = torch.ops.aten.reshape.default(sum_dim_int_list_24, _shape_param_10);  sum_dim_int_list_24 = _shape_param_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_tensor_7: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(view_617, mul_162);  mul_162 = None
        sum_dim_int_list_25: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_tensor_7, [0, 1]);  mul_tensor_7 = None
        sum_dim_int_list_26: "f32[2048]" = torch.ops.aten.sum.dim_IntList(view_617, [0, 1]);  view_617 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:155 in forward, code: attn_output = self.out_proj(attn_output)
        permute_default_20: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_618, [1, 0])
        sum_dim_int_list_27: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_618, [0], True);  view_618 = None
        reshape_default_11: "f32[2048]" = torch.ops.aten.reshape.default(sum_dim_int_list_27, _shape_param_11);  sum_dim_int_list_27 = _shape_param_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:143 in forward, code: value = self.v_proj(hidden_states)
        permute_default_21: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_631, [1, 0]);  view_631 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:142 in forward, code: key = self.k_proj(hidden_states)
        permute_default_22: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_633, [1, 0]);  view_633 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:141 in forward, code: query = self.q_proj(hidden_states)
        permute_default_23: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_635, [1, 0]);  view_635 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_tensor_8: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_245, mul_160);  mul_160 = None
        sum_dim_int_list_28: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_tensor_8, [0, 1]);  mul_tensor_8 = None
        sum_dim_int_list_29: "f32[2048]" = torch.ops.aten.sum.dim_IntList(add_245, [0, 1]);  add_245 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:307 in forward, code: hidden_states = self.c_proj(hidden_states)
        permute_default_24: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_637, [1, 0])
        sum_dim_int_list_30: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_637, [0], True);  view_637 = None
        reshape_default_12: "f32[2048]" = torch.ops.aten.reshape.default(sum_dim_int_list_30, _shape_param_12);  sum_dim_int_list_30 = _shape_param_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        permute_default_25: "f32[8192, 4096]" = torch.ops.aten.permute.default(view_640, [1, 0])
        sum_dim_int_list_31: "f32[1, 8192]" = torch.ops.aten.sum.dim_IntList(view_640, [0], True);  view_640 = None
        reshape_default_13: "f32[8192]" = torch.ops.aten.reshape.default(sum_dim_int_list_31, _shape_param_13);  sum_dim_int_list_31 = _shape_param_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_tensor_9: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(view_642, mul_154);  mul_154 = None
        sum_dim_int_list_32: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_tensor_9, [0, 1]);  mul_tensor_9 = None
        sum_dim_int_list_33: "f32[2048]" = torch.ops.aten.sum.dim_IntList(view_642, [0, 1]);  view_642 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:155 in forward, code: attn_output = self.out_proj(attn_output)
        permute_default_26: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_643, [1, 0])
        sum_dim_int_list_34: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_643, [0], True);  view_643 = None
        reshape_default_14: "f32[2048]" = torch.ops.aten.reshape.default(sum_dim_int_list_34, _shape_param_14);  sum_dim_int_list_34 = _shape_param_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:143 in forward, code: value = self.v_proj(hidden_states)
        permute_default_27: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_656, [1, 0]);  view_656 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:142 in forward, code: key = self.k_proj(hidden_states)
        permute_default_28: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_658, [1, 0]);  view_658 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:141 in forward, code: query = self.q_proj(hidden_states)
        permute_default_29: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_660, [1, 0]);  view_660 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_tensor_10: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_251, mul_152);  mul_152 = None
        sum_dim_int_list_35: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_tensor_10, [0, 1]);  mul_tensor_10 = None
        sum_dim_int_list_36: "f32[2048]" = torch.ops.aten.sum.dim_IntList(add_251, [0, 1]);  add_251 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:307 in forward, code: hidden_states = self.c_proj(hidden_states)
        permute_default_30: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_662, [1, 0])
        sum_dim_int_list_37: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_662, [0], True);  view_662 = None
        reshape_default_15: "f32[2048]" = torch.ops.aten.reshape.default(sum_dim_int_list_37, _shape_param_15);  sum_dim_int_list_37 = _shape_param_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        permute_default_31: "f32[8192, 4096]" = torch.ops.aten.permute.default(view_665, [1, 0])
        sum_dim_int_list_38: "f32[1, 8192]" = torch.ops.aten.sum.dim_IntList(view_665, [0], True);  view_665 = None
        reshape_default_16: "f32[8192]" = torch.ops.aten.reshape.default(sum_dim_int_list_38, _shape_param_16);  sum_dim_int_list_38 = _shape_param_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_tensor_11: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(view_667, mul_146);  mul_146 = None
        sum_dim_int_list_39: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_tensor_11, [0, 1]);  mul_tensor_11 = None
        sum_dim_int_list_40: "f32[2048]" = torch.ops.aten.sum.dim_IntList(view_667, [0, 1]);  view_667 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:155 in forward, code: attn_output = self.out_proj(attn_output)
        permute_default_32: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_668, [1, 0])
        sum_dim_int_list_41: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_668, [0], True);  view_668 = None
        reshape_default_17: "f32[2048]" = torch.ops.aten.reshape.default(sum_dim_int_list_41, _shape_param_17);  sum_dim_int_list_41 = _shape_param_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:143 in forward, code: value = self.v_proj(hidden_states)
        permute_default_33: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_681, [1, 0]);  view_681 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:142 in forward, code: key = self.k_proj(hidden_states)
        permute_default_34: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_683, [1, 0]);  view_683 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:141 in forward, code: query = self.q_proj(hidden_states)
        permute_default_35: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_685, [1, 0]);  view_685 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_tensor_12: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_257, mul_144);  mul_144 = None
        sum_dim_int_list_42: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_tensor_12, [0, 1]);  mul_tensor_12 = None
        sum_dim_int_list_43: "f32[2048]" = torch.ops.aten.sum.dim_IntList(add_257, [0, 1]);  add_257 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:307 in forward, code: hidden_states = self.c_proj(hidden_states)
        permute_default_36: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_687, [1, 0])
        sum_dim_int_list_44: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_687, [0], True);  view_687 = None
        reshape_default_18: "f32[2048]" = torch.ops.aten.reshape.default(sum_dim_int_list_44, _shape_param_18);  sum_dim_int_list_44 = _shape_param_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        permute_default_37: "f32[8192, 4096]" = torch.ops.aten.permute.default(view_690, [1, 0])
        sum_dim_int_list_45: "f32[1, 8192]" = torch.ops.aten.sum.dim_IntList(view_690, [0], True);  view_690 = None
        reshape_default_19: "f32[8192]" = torch.ops.aten.reshape.default(sum_dim_int_list_45, _shape_param_19);  sum_dim_int_list_45 = _shape_param_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_tensor_13: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(view_692, mul_138);  mul_138 = None
        sum_dim_int_list_46: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_tensor_13, [0, 1]);  mul_tensor_13 = None
        sum_dim_int_list_47: "f32[2048]" = torch.ops.aten.sum.dim_IntList(view_692, [0, 1]);  view_692 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:155 in forward, code: attn_output = self.out_proj(attn_output)
        permute_default_38: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_693, [1, 0])
        sum_dim_int_list_48: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_693, [0], True);  view_693 = None
        reshape_default_20: "f32[2048]" = torch.ops.aten.reshape.default(sum_dim_int_list_48, _shape_param_20);  sum_dim_int_list_48 = _shape_param_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:143 in forward, code: value = self.v_proj(hidden_states)
        permute_default_39: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_706, [1, 0]);  view_706 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:142 in forward, code: key = self.k_proj(hidden_states)
        permute_default_40: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_708, [1, 0]);  view_708 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:141 in forward, code: query = self.q_proj(hidden_states)
        permute_default_41: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_710, [1, 0]);  view_710 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_tensor_14: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_263, mul_136);  mul_136 = None
        sum_dim_int_list_49: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_tensor_14, [0, 1]);  mul_tensor_14 = None
        sum_dim_int_list_50: "f32[2048]" = torch.ops.aten.sum.dim_IntList(add_263, [0, 1]);  add_263 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:307 in forward, code: hidden_states = self.c_proj(hidden_states)
        permute_default_42: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_712, [1, 0])
        sum_dim_int_list_51: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_712, [0], True);  view_712 = None
        reshape_default_21: "f32[2048]" = torch.ops.aten.reshape.default(sum_dim_int_list_51, _shape_param_21);  sum_dim_int_list_51 = _shape_param_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        permute_default_43: "f32[8192, 4096]" = torch.ops.aten.permute.default(view_715, [1, 0])
        sum_dim_int_list_52: "f32[1, 8192]" = torch.ops.aten.sum.dim_IntList(view_715, [0], True);  view_715 = None
        reshape_default_22: "f32[8192]" = torch.ops.aten.reshape.default(sum_dim_int_list_52, _shape_param_22);  sum_dim_int_list_52 = _shape_param_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_tensor_15: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(view_717, mul_130);  mul_130 = None
        sum_dim_int_list_53: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_tensor_15, [0, 1]);  mul_tensor_15 = None
        sum_dim_int_list_54: "f32[2048]" = torch.ops.aten.sum.dim_IntList(view_717, [0, 1]);  view_717 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:155 in forward, code: attn_output = self.out_proj(attn_output)
        permute_default_44: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_718, [1, 0])
        sum_dim_int_list_55: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_718, [0], True);  view_718 = None
        reshape_default_23: "f32[2048]" = torch.ops.aten.reshape.default(sum_dim_int_list_55, _shape_param_23);  sum_dim_int_list_55 = _shape_param_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:143 in forward, code: value = self.v_proj(hidden_states)
        permute_default_45: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_731, [1, 0]);  view_731 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:142 in forward, code: key = self.k_proj(hidden_states)
        permute_default_46: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_733, [1, 0]);  view_733 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:141 in forward, code: query = self.q_proj(hidden_states)
        permute_default_47: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_735, [1, 0]);  view_735 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_tensor_16: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_269, mul_128);  mul_128 = None
        sum_dim_int_list_56: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_tensor_16, [0, 1]);  mul_tensor_16 = None
        sum_dim_int_list_57: "f32[2048]" = torch.ops.aten.sum.dim_IntList(add_269, [0, 1]);  add_269 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:307 in forward, code: hidden_states = self.c_proj(hidden_states)
        permute_default_48: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_737, [1, 0])
        sum_dim_int_list_58: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_737, [0], True);  view_737 = None
        reshape_default_24: "f32[2048]" = torch.ops.aten.reshape.default(sum_dim_int_list_58, _shape_param_24);  sum_dim_int_list_58 = _shape_param_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        permute_default_49: "f32[8192, 4096]" = torch.ops.aten.permute.default(view_740, [1, 0])
        sum_dim_int_list_59: "f32[1, 8192]" = torch.ops.aten.sum.dim_IntList(view_740, [0], True);  view_740 = None
        reshape_default_25: "f32[8192]" = torch.ops.aten.reshape.default(sum_dim_int_list_59, _shape_param_25);  sum_dim_int_list_59 = _shape_param_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_tensor_17: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(view_742, mul_122);  mul_122 = None
        sum_dim_int_list_60: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_tensor_17, [0, 1]);  mul_tensor_17 = None
        sum_dim_int_list_61: "f32[2048]" = torch.ops.aten.sum.dim_IntList(view_742, [0, 1]);  view_742 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:155 in forward, code: attn_output = self.out_proj(attn_output)
        permute_default_50: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_743, [1, 0])
        sum_dim_int_list_62: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_743, [0], True);  view_743 = None
        reshape_default_26: "f32[2048]" = torch.ops.aten.reshape.default(sum_dim_int_list_62, _shape_param_26);  sum_dim_int_list_62 = _shape_param_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:143 in forward, code: value = self.v_proj(hidden_states)
        permute_default_51: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_756, [1, 0]);  view_756 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:142 in forward, code: key = self.k_proj(hidden_states)
        permute_default_52: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_758, [1, 0]);  view_758 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:141 in forward, code: query = self.q_proj(hidden_states)
        permute_default_53: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_760, [1, 0]);  view_760 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_tensor_18: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_275, mul_120);  mul_120 = None
        sum_dim_int_list_63: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_tensor_18, [0, 1]);  mul_tensor_18 = None
        sum_dim_int_list_64: "f32[2048]" = torch.ops.aten.sum.dim_IntList(add_275, [0, 1]);  add_275 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:307 in forward, code: hidden_states = self.c_proj(hidden_states)
        permute_default_54: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_762, [1, 0])
        sum_dim_int_list_65: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_762, [0], True);  view_762 = None
        reshape_default_27: "f32[2048]" = torch.ops.aten.reshape.default(sum_dim_int_list_65, _shape_param_27);  sum_dim_int_list_65 = _shape_param_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        permute_default_55: "f32[8192, 4096]" = torch.ops.aten.permute.default(view_765, [1, 0])
        sum_dim_int_list_66: "f32[1, 8192]" = torch.ops.aten.sum.dim_IntList(view_765, [0], True);  view_765 = None
        reshape_default_28: "f32[8192]" = torch.ops.aten.reshape.default(sum_dim_int_list_66, _shape_param_28);  sum_dim_int_list_66 = _shape_param_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_tensor_19: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(view_767, mul_114);  mul_114 = None
        sum_dim_int_list_67: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_tensor_19, [0, 1]);  mul_tensor_19 = None
        sum_dim_int_list_68: "f32[2048]" = torch.ops.aten.sum.dim_IntList(view_767, [0, 1]);  view_767 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:155 in forward, code: attn_output = self.out_proj(attn_output)
        permute_default_56: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_768, [1, 0])
        sum_dim_int_list_69: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_768, [0], True);  view_768 = None
        reshape_default_29: "f32[2048]" = torch.ops.aten.reshape.default(sum_dim_int_list_69, _shape_param_29);  sum_dim_int_list_69 = _shape_param_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:143 in forward, code: value = self.v_proj(hidden_states)
        permute_default_57: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_781, [1, 0]);  view_781 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:142 in forward, code: key = self.k_proj(hidden_states)
        permute_default_58: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_783, [1, 0]);  view_783 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:141 in forward, code: query = self.q_proj(hidden_states)
        permute_default_59: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_785, [1, 0]);  view_785 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_tensor_20: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_281, mul_112);  mul_112 = None
        sum_dim_int_list_70: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_tensor_20, [0, 1]);  mul_tensor_20 = None
        sum_dim_int_list_71: "f32[2048]" = torch.ops.aten.sum.dim_IntList(add_281, [0, 1]);  add_281 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:307 in forward, code: hidden_states = self.c_proj(hidden_states)
        permute_default_60: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_787, [1, 0])
        sum_dim_int_list_72: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_787, [0], True);  view_787 = None
        reshape_default_30: "f32[2048]" = torch.ops.aten.reshape.default(sum_dim_int_list_72, _shape_param_30);  sum_dim_int_list_72 = _shape_param_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        permute_default_61: "f32[8192, 4096]" = torch.ops.aten.permute.default(view_790, [1, 0])
        sum_dim_int_list_73: "f32[1, 8192]" = torch.ops.aten.sum.dim_IntList(view_790, [0], True);  view_790 = None
        reshape_default_31: "f32[8192]" = torch.ops.aten.reshape.default(sum_dim_int_list_73, _shape_param_31);  sum_dim_int_list_73 = _shape_param_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_tensor_21: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(view_792, mul_106);  mul_106 = None
        sum_dim_int_list_74: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_tensor_21, [0, 1]);  mul_tensor_21 = None
        sum_dim_int_list_75: "f32[2048]" = torch.ops.aten.sum.dim_IntList(view_792, [0, 1]);  view_792 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:155 in forward, code: attn_output = self.out_proj(attn_output)
        permute_default_62: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_793, [1, 0])
        sum_dim_int_list_76: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_793, [0], True);  view_793 = None
        reshape_default_32: "f32[2048]" = torch.ops.aten.reshape.default(sum_dim_int_list_76, _shape_param_32);  sum_dim_int_list_76 = _shape_param_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:143 in forward, code: value = self.v_proj(hidden_states)
        permute_default_63: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_806, [1, 0]);  view_806 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:142 in forward, code: key = self.k_proj(hidden_states)
        permute_default_64: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_808, [1, 0]);  view_808 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:141 in forward, code: query = self.q_proj(hidden_states)
        permute_default_65: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_810, [1, 0]);  view_810 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_tensor_22: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_287, mul_104);  mul_104 = None
        sum_dim_int_list_77: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_tensor_22, [0, 1]);  mul_tensor_22 = None
        sum_dim_int_list_78: "f32[2048]" = torch.ops.aten.sum.dim_IntList(add_287, [0, 1]);  add_287 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:307 in forward, code: hidden_states = self.c_proj(hidden_states)
        permute_default_66: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_812, [1, 0])
        sum_dim_int_list_79: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_812, [0], True);  view_812 = None
        reshape_default_33: "f32[2048]" = torch.ops.aten.reshape.default(sum_dim_int_list_79, _shape_param_33);  sum_dim_int_list_79 = _shape_param_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        permute_default_67: "f32[8192, 4096]" = torch.ops.aten.permute.default(view_815, [1, 0])
        sum_dim_int_list_80: "f32[1, 8192]" = torch.ops.aten.sum.dim_IntList(view_815, [0], True);  view_815 = None
        reshape_default_34: "f32[8192]" = torch.ops.aten.reshape.default(sum_dim_int_list_80, _shape_param_34);  sum_dim_int_list_80 = _shape_param_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_tensor_23: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(view_817, mul_98);  mul_98 = None
        sum_dim_int_list_81: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_tensor_23, [0, 1]);  mul_tensor_23 = None
        sum_dim_int_list_82: "f32[2048]" = torch.ops.aten.sum.dim_IntList(view_817, [0, 1]);  view_817 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:155 in forward, code: attn_output = self.out_proj(attn_output)
        permute_default_68: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_818, [1, 0])
        sum_dim_int_list_83: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_818, [0], True);  view_818 = None
        reshape_default_35: "f32[2048]" = torch.ops.aten.reshape.default(sum_dim_int_list_83, _shape_param_35);  sum_dim_int_list_83 = _shape_param_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:143 in forward, code: value = self.v_proj(hidden_states)
        permute_default_69: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_831, [1, 0]);  view_831 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:142 in forward, code: key = self.k_proj(hidden_states)
        permute_default_70: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_833, [1, 0]);  view_833 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:141 in forward, code: query = self.q_proj(hidden_states)
        permute_default_71: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_835, [1, 0]);  view_835 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_tensor_24: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_293, mul_96);  mul_96 = None
        sum_dim_int_list_84: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_tensor_24, [0, 1]);  mul_tensor_24 = None
        sum_dim_int_list_85: "f32[2048]" = torch.ops.aten.sum.dim_IntList(add_293, [0, 1]);  add_293 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:307 in forward, code: hidden_states = self.c_proj(hidden_states)
        permute_default_72: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_837, [1, 0])
        sum_dim_int_list_86: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_837, [0], True);  view_837 = None
        reshape_default_36: "f32[2048]" = torch.ops.aten.reshape.default(sum_dim_int_list_86, _shape_param_36);  sum_dim_int_list_86 = _shape_param_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        permute_default_73: "f32[8192, 4096]" = torch.ops.aten.permute.default(view_840, [1, 0])
        sum_dim_int_list_87: "f32[1, 8192]" = torch.ops.aten.sum.dim_IntList(view_840, [0], True);  view_840 = None
        reshape_default_37: "f32[8192]" = torch.ops.aten.reshape.default(sum_dim_int_list_87, _shape_param_37);  sum_dim_int_list_87 = _shape_param_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_tensor_25: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(view_842, mul_90);  mul_90 = None
        sum_dim_int_list_88: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_tensor_25, [0, 1]);  mul_tensor_25 = None
        sum_dim_int_list_89: "f32[2048]" = torch.ops.aten.sum.dim_IntList(view_842, [0, 1]);  view_842 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:155 in forward, code: attn_output = self.out_proj(attn_output)
        permute_default_74: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_843, [1, 0])
        sum_dim_int_list_90: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_843, [0], True);  view_843 = None
        reshape_default_38: "f32[2048]" = torch.ops.aten.reshape.default(sum_dim_int_list_90, _shape_param_38);  sum_dim_int_list_90 = _shape_param_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:143 in forward, code: value = self.v_proj(hidden_states)
        permute_default_75: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_856, [1, 0]);  view_856 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:142 in forward, code: key = self.k_proj(hidden_states)
        permute_default_76: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_858, [1, 0]);  view_858 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:141 in forward, code: query = self.q_proj(hidden_states)
        permute_default_77: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_860, [1, 0]);  view_860 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_tensor_26: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_299, mul_88);  mul_88 = None
        sum_dim_int_list_91: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_tensor_26, [0, 1]);  mul_tensor_26 = None
        sum_dim_int_list_92: "f32[2048]" = torch.ops.aten.sum.dim_IntList(add_299, [0, 1]);  add_299 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:307 in forward, code: hidden_states = self.c_proj(hidden_states)
        permute_default_78: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_862, [1, 0])
        sum_dim_int_list_93: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_862, [0], True);  view_862 = None
        reshape_default_39: "f32[2048]" = torch.ops.aten.reshape.default(sum_dim_int_list_93, _shape_param_39);  sum_dim_int_list_93 = _shape_param_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        permute_default_79: "f32[8192, 4096]" = torch.ops.aten.permute.default(view_865, [1, 0])
        sum_dim_int_list_94: "f32[1, 8192]" = torch.ops.aten.sum.dim_IntList(view_865, [0], True);  view_865 = None
        reshape_default_40: "f32[8192]" = torch.ops.aten.reshape.default(sum_dim_int_list_94, _shape_param_40);  sum_dim_int_list_94 = _shape_param_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_tensor_27: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(view_867, mul_82);  mul_82 = None
        sum_dim_int_list_95: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_tensor_27, [0, 1]);  mul_tensor_27 = None
        sum_dim_int_list_96: "f32[2048]" = torch.ops.aten.sum.dim_IntList(view_867, [0, 1]);  view_867 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:155 in forward, code: attn_output = self.out_proj(attn_output)
        permute_default_80: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_868, [1, 0])
        sum_dim_int_list_97: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_868, [0], True);  view_868 = None
        reshape_default_41: "f32[2048]" = torch.ops.aten.reshape.default(sum_dim_int_list_97, _shape_param_41);  sum_dim_int_list_97 = _shape_param_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:143 in forward, code: value = self.v_proj(hidden_states)
        permute_default_81: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_881, [1, 0]);  view_881 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:142 in forward, code: key = self.k_proj(hidden_states)
        permute_default_82: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_883, [1, 0]);  view_883 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:141 in forward, code: query = self.q_proj(hidden_states)
        permute_default_83: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_885, [1, 0]);  view_885 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_tensor_28: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_305, mul_80);  mul_80 = None
        sum_dim_int_list_98: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_tensor_28, [0, 1]);  mul_tensor_28 = None
        sum_dim_int_list_99: "f32[2048]" = torch.ops.aten.sum.dim_IntList(add_305, [0, 1]);  add_305 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:307 in forward, code: hidden_states = self.c_proj(hidden_states)
        permute_default_84: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_887, [1, 0])
        sum_dim_int_list_100: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_887, [0], True);  view_887 = None
        reshape_default_42: "f32[2048]" = torch.ops.aten.reshape.default(sum_dim_int_list_100, _shape_param_42);  sum_dim_int_list_100 = _shape_param_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        permute_default_85: "f32[8192, 4096]" = torch.ops.aten.permute.default(view_890, [1, 0])
        sum_dim_int_list_101: "f32[1, 8192]" = torch.ops.aten.sum.dim_IntList(view_890, [0], True);  view_890 = None
        reshape_default_43: "f32[8192]" = torch.ops.aten.reshape.default(sum_dim_int_list_101, _shape_param_43);  sum_dim_int_list_101 = _shape_param_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_tensor_29: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(view_892, mul_74);  mul_74 = None
        sum_dim_int_list_102: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_tensor_29, [0, 1]);  mul_tensor_29 = None
        sum_dim_int_list_103: "f32[2048]" = torch.ops.aten.sum.dim_IntList(view_892, [0, 1]);  view_892 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:155 in forward, code: attn_output = self.out_proj(attn_output)
        permute_default_86: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_893, [1, 0])
        sum_dim_int_list_104: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_893, [0], True);  view_893 = None
        reshape_default_44: "f32[2048]" = torch.ops.aten.reshape.default(sum_dim_int_list_104, _shape_param_44);  sum_dim_int_list_104 = _shape_param_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:143 in forward, code: value = self.v_proj(hidden_states)
        permute_default_87: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_906, [1, 0]);  view_906 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:142 in forward, code: key = self.k_proj(hidden_states)
        permute_default_88: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_908, [1, 0]);  view_908 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:141 in forward, code: query = self.q_proj(hidden_states)
        permute_default_89: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_910, [1, 0]);  view_910 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_tensor_30: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_311, mul_72);  mul_72 = None
        sum_dim_int_list_105: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_tensor_30, [0, 1]);  mul_tensor_30 = None
        sum_dim_int_list_106: "f32[2048]" = torch.ops.aten.sum.dim_IntList(add_311, [0, 1]);  add_311 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:307 in forward, code: hidden_states = self.c_proj(hidden_states)
        permute_default_90: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_912, [1, 0])
        sum_dim_int_list_107: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_912, [0], True);  view_912 = None
        reshape_default_45: "f32[2048]" = torch.ops.aten.reshape.default(sum_dim_int_list_107, _shape_param_45);  sum_dim_int_list_107 = _shape_param_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        permute_default_91: "f32[8192, 4096]" = torch.ops.aten.permute.default(view_915, [1, 0])
        sum_dim_int_list_108: "f32[1, 8192]" = torch.ops.aten.sum.dim_IntList(view_915, [0], True);  view_915 = None
        reshape_default_46: "f32[8192]" = torch.ops.aten.reshape.default(sum_dim_int_list_108, _shape_param_46);  sum_dim_int_list_108 = _shape_param_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_tensor_31: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(view_917, mul_66);  mul_66 = None
        sum_dim_int_list_109: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_tensor_31, [0, 1]);  mul_tensor_31 = None
        sum_dim_int_list_110: "f32[2048]" = torch.ops.aten.sum.dim_IntList(view_917, [0, 1]);  view_917 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:155 in forward, code: attn_output = self.out_proj(attn_output)
        permute_default_92: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_918, [1, 0])
        sum_dim_int_list_111: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_918, [0], True);  view_918 = None
        reshape_default_47: "f32[2048]" = torch.ops.aten.reshape.default(sum_dim_int_list_111, _shape_param_47);  sum_dim_int_list_111 = _shape_param_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:143 in forward, code: value = self.v_proj(hidden_states)
        permute_default_93: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_931, [1, 0]);  view_931 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:142 in forward, code: key = self.k_proj(hidden_states)
        permute_default_94: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_933, [1, 0]);  view_933 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:141 in forward, code: query = self.q_proj(hidden_states)
        permute_default_95: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_935, [1, 0]);  view_935 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_tensor_32: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_317, mul_64);  mul_64 = None
        sum_dim_int_list_112: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_tensor_32, [0, 1]);  mul_tensor_32 = None
        sum_dim_int_list_113: "f32[2048]" = torch.ops.aten.sum.dim_IntList(add_317, [0, 1]);  add_317 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:307 in forward, code: hidden_states = self.c_proj(hidden_states)
        permute_default_96: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_937, [1, 0])
        sum_dim_int_list_114: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_937, [0], True);  view_937 = None
        reshape_default_48: "f32[2048]" = torch.ops.aten.reshape.default(sum_dim_int_list_114, _shape_param_48);  sum_dim_int_list_114 = _shape_param_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        permute_default_97: "f32[8192, 4096]" = torch.ops.aten.permute.default(view_940, [1, 0])
        sum_dim_int_list_115: "f32[1, 8192]" = torch.ops.aten.sum.dim_IntList(view_940, [0], True);  view_940 = None
        reshape_default_49: "f32[8192]" = torch.ops.aten.reshape.default(sum_dim_int_list_115, _shape_param_49);  sum_dim_int_list_115 = _shape_param_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_tensor_33: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(view_942, mul_58);  mul_58 = None
        sum_dim_int_list_116: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_tensor_33, [0, 1]);  mul_tensor_33 = None
        sum_dim_int_list_117: "f32[2048]" = torch.ops.aten.sum.dim_IntList(view_942, [0, 1]);  view_942 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:155 in forward, code: attn_output = self.out_proj(attn_output)
        permute_default_98: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_943, [1, 0])
        sum_dim_int_list_118: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_943, [0], True);  view_943 = None
        reshape_default_50: "f32[2048]" = torch.ops.aten.reshape.default(sum_dim_int_list_118, _shape_param_50);  sum_dim_int_list_118 = _shape_param_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:143 in forward, code: value = self.v_proj(hidden_states)
        permute_default_99: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_956, [1, 0]);  view_956 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:142 in forward, code: key = self.k_proj(hidden_states)
        permute_default_100: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_958, [1, 0]);  view_958 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:141 in forward, code: query = self.q_proj(hidden_states)
        permute_default_101: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_960, [1, 0]);  view_960 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_tensor_34: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_323, mul_56);  mul_56 = None
        sum_dim_int_list_119: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_tensor_34, [0, 1]);  mul_tensor_34 = None
        sum_dim_int_list_120: "f32[2048]" = torch.ops.aten.sum.dim_IntList(add_323, [0, 1]);  add_323 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:307 in forward, code: hidden_states = self.c_proj(hidden_states)
        permute_default_102: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_962, [1, 0])
        sum_dim_int_list_121: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_962, [0], True);  view_962 = None
        reshape_default_51: "f32[2048]" = torch.ops.aten.reshape.default(sum_dim_int_list_121, _shape_param_51);  sum_dim_int_list_121 = _shape_param_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        permute_default_103: "f32[8192, 4096]" = torch.ops.aten.permute.default(view_965, [1, 0])
        sum_dim_int_list_122: "f32[1, 8192]" = torch.ops.aten.sum.dim_IntList(view_965, [0], True);  view_965 = None
        reshape_default_52: "f32[8192]" = torch.ops.aten.reshape.default(sum_dim_int_list_122, _shape_param_52);  sum_dim_int_list_122 = _shape_param_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_tensor_35: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(view_967, mul_50);  mul_50 = None
        sum_dim_int_list_123: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_tensor_35, [0, 1]);  mul_tensor_35 = None
        sum_dim_int_list_124: "f32[2048]" = torch.ops.aten.sum.dim_IntList(view_967, [0, 1]);  view_967 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:155 in forward, code: attn_output = self.out_proj(attn_output)
        permute_default_104: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_968, [1, 0])
        sum_dim_int_list_125: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_968, [0], True);  view_968 = None
        reshape_default_53: "f32[2048]" = torch.ops.aten.reshape.default(sum_dim_int_list_125, _shape_param_53);  sum_dim_int_list_125 = _shape_param_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:143 in forward, code: value = self.v_proj(hidden_states)
        permute_default_105: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_981, [1, 0]);  view_981 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:142 in forward, code: key = self.k_proj(hidden_states)
        permute_default_106: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_983, [1, 0]);  view_983 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:141 in forward, code: query = self.q_proj(hidden_states)
        permute_default_107: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_985, [1, 0]);  view_985 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_tensor_36: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_329, mul_48);  mul_48 = None
        sum_dim_int_list_126: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_tensor_36, [0, 1]);  mul_tensor_36 = None
        sum_dim_int_list_127: "f32[2048]" = torch.ops.aten.sum.dim_IntList(add_329, [0, 1]);  add_329 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:307 in forward, code: hidden_states = self.c_proj(hidden_states)
        permute_default_108: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_987, [1, 0])
        sum_dim_int_list_128: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_987, [0], True);  view_987 = None
        reshape_default_54: "f32[2048]" = torch.ops.aten.reshape.default(sum_dim_int_list_128, _shape_param_54);  sum_dim_int_list_128 = _shape_param_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        permute_default_109: "f32[8192, 4096]" = torch.ops.aten.permute.default(view_990, [1, 0])
        sum_dim_int_list_129: "f32[1, 8192]" = torch.ops.aten.sum.dim_IntList(view_990, [0], True);  view_990 = None
        reshape_default_55: "f32[8192]" = torch.ops.aten.reshape.default(sum_dim_int_list_129, _shape_param_55);  sum_dim_int_list_129 = _shape_param_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_tensor_37: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(view_992, mul_42);  mul_42 = None
        sum_dim_int_list_130: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_tensor_37, [0, 1]);  mul_tensor_37 = None
        sum_dim_int_list_131: "f32[2048]" = torch.ops.aten.sum.dim_IntList(view_992, [0, 1]);  view_992 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:155 in forward, code: attn_output = self.out_proj(attn_output)
        permute_default_110: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_993, [1, 0])
        sum_dim_int_list_132: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_993, [0], True);  view_993 = None
        reshape_default_56: "f32[2048]" = torch.ops.aten.reshape.default(sum_dim_int_list_132, _shape_param_56);  sum_dim_int_list_132 = _shape_param_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:143 in forward, code: value = self.v_proj(hidden_states)
        permute_default_111: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_1006, [1, 0]);  view_1006 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:142 in forward, code: key = self.k_proj(hidden_states)
        permute_default_112: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_1008, [1, 0]);  view_1008 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:141 in forward, code: query = self.q_proj(hidden_states)
        permute_default_113: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_1010, [1, 0]);  view_1010 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_tensor_38: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_335, mul_40);  mul_40 = None
        sum_dim_int_list_133: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_tensor_38, [0, 1]);  mul_tensor_38 = None
        sum_dim_int_list_134: "f32[2048]" = torch.ops.aten.sum.dim_IntList(add_335, [0, 1]);  add_335 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:307 in forward, code: hidden_states = self.c_proj(hidden_states)
        permute_default_114: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_1012, [1, 0])
        sum_dim_int_list_135: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_1012, [0], True);  view_1012 = None
        reshape_default_57: "f32[2048]" = torch.ops.aten.reshape.default(sum_dim_int_list_135, _shape_param_57);  sum_dim_int_list_135 = _shape_param_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        permute_default_115: "f32[8192, 4096]" = torch.ops.aten.permute.default(view_1015, [1, 0])
        sum_dim_int_list_136: "f32[1, 8192]" = torch.ops.aten.sum.dim_IntList(view_1015, [0], True);  view_1015 = None
        reshape_default_58: "f32[8192]" = torch.ops.aten.reshape.default(sum_dim_int_list_136, _shape_param_58);  sum_dim_int_list_136 = _shape_param_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_tensor_39: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(view_1017, mul_34);  mul_34 = None
        sum_dim_int_list_137: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_tensor_39, [0, 1]);  mul_tensor_39 = None
        sum_dim_int_list_138: "f32[2048]" = torch.ops.aten.sum.dim_IntList(view_1017, [0, 1]);  view_1017 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:155 in forward, code: attn_output = self.out_proj(attn_output)
        permute_default_116: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_1018, [1, 0])
        sum_dim_int_list_139: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_1018, [0], True);  view_1018 = None
        reshape_default_59: "f32[2048]" = torch.ops.aten.reshape.default(sum_dim_int_list_139, _shape_param_59);  sum_dim_int_list_139 = _shape_param_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:143 in forward, code: value = self.v_proj(hidden_states)
        permute_default_117: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_1031, [1, 0]);  view_1031 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:142 in forward, code: key = self.k_proj(hidden_states)
        permute_default_118: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_1033, [1, 0]);  view_1033 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:141 in forward, code: query = self.q_proj(hidden_states)
        permute_default_119: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_1035, [1, 0]);  view_1035 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_tensor_40: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_341, mul_32);  mul_32 = None
        sum_dim_int_list_140: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_tensor_40, [0, 1]);  mul_tensor_40 = None
        sum_dim_int_list_141: "f32[2048]" = torch.ops.aten.sum.dim_IntList(add_341, [0, 1]);  add_341 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:307 in forward, code: hidden_states = self.c_proj(hidden_states)
        permute_default_120: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_1037, [1, 0])
        sum_dim_int_list_142: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_1037, [0], True);  view_1037 = None
        reshape_default_60: "f32[2048]" = torch.ops.aten.reshape.default(sum_dim_int_list_142, _shape_param_60);  sum_dim_int_list_142 = _shape_param_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        permute_default_121: "f32[8192, 4096]" = torch.ops.aten.permute.default(view_1040, [1, 0])
        sum_dim_int_list_143: "f32[1, 8192]" = torch.ops.aten.sum.dim_IntList(view_1040, [0], True);  view_1040 = None
        reshape_default_61: "f32[8192]" = torch.ops.aten.reshape.default(sum_dim_int_list_143, _shape_param_61);  sum_dim_int_list_143 = _shape_param_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_tensor_41: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(view_1042, mul_26);  mul_26 = None
        sum_dim_int_list_144: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_tensor_41, [0, 1]);  mul_tensor_41 = None
        sum_dim_int_list_145: "f32[2048]" = torch.ops.aten.sum.dim_IntList(view_1042, [0, 1]);  view_1042 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:155 in forward, code: attn_output = self.out_proj(attn_output)
        permute_default_122: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_1043, [1, 0])
        sum_dim_int_list_146: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_1043, [0], True);  view_1043 = None
        reshape_default_62: "f32[2048]" = torch.ops.aten.reshape.default(sum_dim_int_list_146, _shape_param_62);  sum_dim_int_list_146 = _shape_param_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:143 in forward, code: value = self.v_proj(hidden_states)
        permute_default_123: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_1056, [1, 0]);  view_1056 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:142 in forward, code: key = self.k_proj(hidden_states)
        permute_default_124: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_1058, [1, 0]);  view_1058 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:141 in forward, code: query = self.q_proj(hidden_states)
        permute_default_125: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_1060, [1, 0]);  view_1060 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_tensor_42: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_347, mul_24);  mul_24 = None
        sum_dim_int_list_147: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_tensor_42, [0, 1]);  mul_tensor_42 = None
        sum_dim_int_list_148: "f32[2048]" = torch.ops.aten.sum.dim_IntList(add_347, [0, 1]);  add_347 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:307 in forward, code: hidden_states = self.c_proj(hidden_states)
        permute_default_126: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_1062, [1, 0])
        sum_dim_int_list_149: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_1062, [0], True);  view_1062 = None
        reshape_default_63: "f32[2048]" = torch.ops.aten.reshape.default(sum_dim_int_list_149, _shape_param_63);  sum_dim_int_list_149 = _shape_param_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        permute_default_127: "f32[8192, 4096]" = torch.ops.aten.permute.default(view_1065, [1, 0])
        sum_dim_int_list_150: "f32[1, 8192]" = torch.ops.aten.sum.dim_IntList(view_1065, [0], True);  view_1065 = None
        reshape_default_64: "f32[8192]" = torch.ops.aten.reshape.default(sum_dim_int_list_150, _shape_param_64);  sum_dim_int_list_150 = _shape_param_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_tensor_43: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(view_1067, mul_18);  mul_18 = None
        sum_dim_int_list_151: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_tensor_43, [0, 1]);  mul_tensor_43 = None
        sum_dim_int_list_152: "f32[2048]" = torch.ops.aten.sum.dim_IntList(view_1067, [0, 1]);  view_1067 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:155 in forward, code: attn_output = self.out_proj(attn_output)
        permute_default_128: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_1068, [1, 0])
        sum_dim_int_list_153: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_1068, [0], True);  view_1068 = None
        reshape_default_65: "f32[2048]" = torch.ops.aten.reshape.default(sum_dim_int_list_153, _shape_param_65);  sum_dim_int_list_153 = _shape_param_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:143 in forward, code: value = self.v_proj(hidden_states)
        permute_default_129: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_1081, [1, 0]);  view_1081 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:142 in forward, code: key = self.k_proj(hidden_states)
        permute_default_130: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_1083, [1, 0]);  view_1083 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:141 in forward, code: query = self.q_proj(hidden_states)
        permute_default_131: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_1085, [1, 0]);  view_1085 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_tensor_44: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_353, mul_16);  mul_16 = None
        sum_dim_int_list_154: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_tensor_44, [0, 1]);  mul_tensor_44 = None
        sum_dim_int_list_155: "f32[2048]" = torch.ops.aten.sum.dim_IntList(add_353, [0, 1]);  add_353 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:307 in forward, code: hidden_states = self.c_proj(hidden_states)
        permute_default_132: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_1087, [1, 0])
        sum_dim_int_list_156: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_1087, [0], True);  view_1087 = None
        reshape_default_66: "f32[2048]" = torch.ops.aten.reshape.default(sum_dim_int_list_156, _shape_param_66);  sum_dim_int_list_156 = _shape_param_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        permute_default_133: "f32[8192, 4096]" = torch.ops.aten.permute.default(view_1090, [1, 0])
        sum_dim_int_list_157: "f32[1, 8192]" = torch.ops.aten.sum.dim_IntList(view_1090, [0], True);  view_1090 = None
        reshape_default_67: "f32[8192]" = torch.ops.aten.reshape.default(sum_dim_int_list_157, _shape_param_67);  sum_dim_int_list_157 = _shape_param_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_tensor_45: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(view_1092, mul_10);  mul_10 = None
        sum_dim_int_list_158: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_tensor_45, [0, 1]);  mul_tensor_45 = None
        sum_dim_int_list_159: "f32[2048]" = torch.ops.aten.sum.dim_IntList(view_1092, [0, 1]);  view_1092 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:155 in forward, code: attn_output = self.out_proj(attn_output)
        permute_default_134: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_1093, [1, 0])
        sum_dim_int_list_160: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_1093, [0], True);  view_1093 = None
        reshape_default_68: "f32[2048]" = torch.ops.aten.reshape.default(sum_dim_int_list_160, _shape_param_68);  sum_dim_int_list_160 = _shape_param_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:143 in forward, code: value = self.v_proj(hidden_states)
        permute_default_135: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_1106, [1, 0]);  view_1106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:142 in forward, code: key = self.k_proj(hidden_states)
        permute_default_136: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_1108, [1, 0]);  view_1108 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:141 in forward, code: query = self.q_proj(hidden_states)
        permute_default_137: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_1110, [1, 0]);  view_1110 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_tensor_46: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_359, mul_8);  mul_8 = None
        sum_dim_int_list_161: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_tensor_46, [0, 1]);  mul_tensor_46 = None
        sum_dim_int_list_162: "f32[2048]" = torch.ops.aten.sum.dim_IntList(add_359, [0, 1]);  add_359 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:307 in forward, code: hidden_states = self.c_proj(hidden_states)
        permute_default_138: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_1112, [1, 0])
        sum_dim_int_list_163: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_1112, [0], True);  view_1112 = None
        reshape_default_69: "f32[2048]" = torch.ops.aten.reshape.default(sum_dim_int_list_163, _shape_param_69);  sum_dim_int_list_163 = _shape_param_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        permute_default_139: "f32[8192, 4096]" = torch.ops.aten.permute.default(view_1115, [1, 0])
        sum_dim_int_list_164: "f32[1, 8192]" = torch.ops.aten.sum.dim_IntList(view_1115, [0], True);  view_1115 = None
        reshape_default_70: "f32[8192]" = torch.ops.aten.reshape.default(sum_dim_int_list_164, _shape_param_70);  sum_dim_int_list_164 = _shape_param_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_tensor_47: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(view_1117, mul_2);  mul_2 = None
        sum_dim_int_list_165: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_tensor_47, [0, 1]);  mul_tensor_47 = None
        sum_dim_int_list_166: "f32[2048]" = torch.ops.aten.sum.dim_IntList(view_1117, [0, 1]);  view_1117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:155 in forward, code: attn_output = self.out_proj(attn_output)
        permute_default_140: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_1118, [1, 0])
        sum_dim_int_list_167: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_1118, [0], True);  view_1118 = None
        reshape_default_71: "f32[2048]" = torch.ops.aten.reshape.default(sum_dim_int_list_167, _shape_param_71);  sum_dim_int_list_167 = _shape_param_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:143 in forward, code: value = self.v_proj(hidden_states)
        permute_default_141: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_1131, [1, 0]);  view_1131 = None
        reshape_default_72: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_358, _shape_param_72);  mm_358 = _shape_param_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:142 in forward, code: key = self.k_proj(hidden_states)
        permute_default_142: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_1133, [1, 0]);  view_1133 = None
        reshape_default_73: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_360, _shape_param_73);  mm_360 = _shape_param_73 = None
        add_tensor: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(reshape_default_72, reshape_default_73);  reshape_default_72 = reshape_default_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:141 in forward, code: query = self.q_proj(hidden_states)
        permute_default_143: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_1135, [1, 0]);  view_1135 = None
        reshape_default_74: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_362, _shape_param_74);  mm_362 = _shape_param_74 = None
        add_tensor_1: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_tensor, reshape_default_74);  add_tensor = reshape_default_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_tensor_48: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_tensor_1, primals_4);  primals_4 = None
        mul_tensor_49: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_tensor_48, 2048)
        sum_dim_int_list_168: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_48, [2], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:463 in forward, code: hidden_states = inputs_embeds + position_embeds
        add_tensor_2: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(embedding, embedding_1);  embedding = embedding_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        sub_tensor: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(add_tensor_2, getitem_1);  add_tensor_2 = getitem_1 = None
        mul_tensor_50: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt);  sub_tensor = None
        mul_tensor_51: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_tensor_48, mul_tensor_50);  mul_tensor_48 = None
        sum_dim_int_list_169: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_51, [2], True);  mul_tensor_51 = None
        mul_tensor_52: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_tensor_50, sum_dim_int_list_169);  sum_dim_int_list_169 = None
        sub_tensor_1: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_tensor_49, sum_dim_int_list_168);  mul_tensor_49 = sum_dim_int_list_168 = None
        sub_tensor_2: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_tensor_1, mul_tensor_52);  sub_tensor_1 = mul_tensor_52 = None
        div_tensor: "f32[32, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt, 2048);  rsqrt = None
        mul_tensor_53: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(div_tensor, sub_tensor_2);  div_tensor = sub_tensor_2 = None
        mul_tensor_54: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_tensor_1, mul_tensor_50);  mul_tensor_50 = None
        sum_dim_int_list_170: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_tensor_54, [0, 1]);  mul_tensor_54 = None
        sum_dim_int_list_171: "f32[2048]" = torch.ops.aten.sum.dim_IntList(add_tensor_1, [0, 1]);  add_tensor_1 = None
        add_tensor_3: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_363, mul_tensor_53);  add_363 = mul_tensor_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:463 in forward, code: hidden_states = inputs_embeds + position_embeds
        sum_dim_int_list_172: "f32[1, 128, 2048]" = torch.ops.aten.sum.dim_IntList(add_tensor_3, [0], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:462 in forward, code: position_embeds = self.wpe(position_ids)
        eq_scalar: "b8[1, 128]" = torch.ops.aten.eq.Scalar(unsqueeze, -1)
        unsqueeze_default: "b8[1, 128, 1]" = torch.ops.aten.unsqueeze.default(eq_scalar, -1);  eq_scalar = None
        where_self: "f32[1, 128, 2048]" = torch.ops.aten.where.self(unsqueeze_default, full_default_1, sum_dim_int_list_172);  unsqueeze_default = sum_dim_int_list_172 = None
        full_default: "f32[2048, 2048]" = torch.ops.aten.full.default([2048, 2048], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default: "f32[2048, 2048]" = torch.ops.aten.index_put.default(full_default, [unsqueeze], where_self, True);  full_default = unsqueeze = where_self = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:444 in forward, code: inputs_embeds = self.wte(input_ids)
        eq_scalar_1: "b8[32, 128]" = torch.ops.aten.eq.Scalar(primals_1, -1)
        unsqueeze_default_1: "b8[32, 128, 1]" = torch.ops.aten.unsqueeze.default(eq_scalar_1, -1);  eq_scalar_1 = None
        where_self_1: "f32[32, 128, 2048]" = torch.ops.aten.where.self(unsqueeze_default_1, full_default_1, add_tensor_3);  unsqueeze_default_1 = full_default_1 = add_tensor_3 = None
        full_default_2: "f32[50257, 2048]" = torch.ops.aten.full.default([50257, 2048], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default_1: "f32[50257, 2048]" = torch.ops.aten.index_put.default(full_default_2, [primals_1], where_self_1, True);  full_default_2 = primals_1 = where_self_1 = None
        add_tensor_4: "f32[50257, 2048]" = torch.ops.aten.add.Tensor(slice_tensor, index_put_default_1);  slice_tensor = index_put_default_1 = None
        return (sum_dim_int_list, sum_dim_int_list_1, permute_default, reshape_default, permute_default_1, reshape_default_1, sum_dim_int_list_4, sum_dim_int_list_5, permute_default_2, reshape_default_2, permute_default_3, permute_default_4, permute_default_5, sum_dim_int_list_7, sum_dim_int_list_8, permute_default_6, reshape_default_3, permute_default_7, reshape_default_4, sum_dim_int_list_11, sum_dim_int_list_12, permute_default_8, reshape_default_5, permute_default_9, permute_default_10, permute_default_11, sum_dim_int_list_14, sum_dim_int_list_15, permute_default_12, reshape_default_6, permute_default_13, reshape_default_7, sum_dim_int_list_18, sum_dim_int_list_19, permute_default_14, reshape_default_8, permute_default_15, permute_default_16, permute_default_17, sum_dim_int_list_21, sum_dim_int_list_22, permute_default_18, reshape_default_9, permute_default_19, reshape_default_10, sum_dim_int_list_25, sum_dim_int_list_26, permute_default_20, reshape_default_11, permute_default_21, permute_default_22, permute_default_23, sum_dim_int_list_28, sum_dim_int_list_29, permute_default_24, reshape_default_12, permute_default_25, reshape_default_13, sum_dim_int_list_32, sum_dim_int_list_33, permute_default_26, reshape_default_14, permute_default_27, permute_default_28, permute_default_29, sum_dim_int_list_35, sum_dim_int_list_36, permute_default_30, reshape_default_15, permute_default_31, reshape_default_16, sum_dim_int_list_39, sum_dim_int_list_40, permute_default_32, reshape_default_17, permute_default_33, permute_default_34, permute_default_35, sum_dim_int_list_42, sum_dim_int_list_43, permute_default_36, reshape_default_18, permute_default_37, reshape_default_19, sum_dim_int_list_46, sum_dim_int_list_47, permute_default_38, reshape_default_20, permute_default_39, permute_default_40, permute_default_41, sum_dim_int_list_49, sum_dim_int_list_50, permute_default_42, reshape_default_21, permute_default_43, reshape_default_22, sum_dim_int_list_53, sum_dim_int_list_54, permute_default_44, reshape_default_23, permute_default_45, permute_default_46, permute_default_47, sum_dim_int_list_56, sum_dim_int_list_57, permute_default_48, reshape_default_24, permute_default_49, reshape_default_25, sum_dim_int_list_60, sum_dim_int_list_61, permute_default_50, reshape_default_26, permute_default_51, permute_default_52, permute_default_53, sum_dim_int_list_63, sum_dim_int_list_64, permute_default_54, reshape_default_27, permute_default_55, reshape_default_28, sum_dim_int_list_67, sum_dim_int_list_68, permute_default_56, reshape_default_29, permute_default_57, permute_default_58, permute_default_59, sum_dim_int_list_70, sum_dim_int_list_71, permute_default_60, reshape_default_30, permute_default_61, reshape_default_31, sum_dim_int_list_74, sum_dim_int_list_75, permute_default_62, reshape_default_32, permute_default_63, permute_default_64, permute_default_65, sum_dim_int_list_77, sum_dim_int_list_78, permute_default_66, reshape_default_33, permute_default_67, reshape_default_34, sum_dim_int_list_81, sum_dim_int_list_82, permute_default_68, reshape_default_35, permute_default_69, permute_default_70, permute_default_71, sum_dim_int_list_84, sum_dim_int_list_85, permute_default_72, reshape_default_36, permute_default_73, reshape_default_37, sum_dim_int_list_88, sum_dim_int_list_89, permute_default_74, reshape_default_38, permute_default_75, permute_default_76, permute_default_77, sum_dim_int_list_91, sum_dim_int_list_92, permute_default_78, reshape_default_39, permute_default_79, reshape_default_40, sum_dim_int_list_95, sum_dim_int_list_96, permute_default_80, reshape_default_41, permute_default_81, permute_default_82, permute_default_83, sum_dim_int_list_98, sum_dim_int_list_99, permute_default_84, reshape_default_42, permute_default_85, reshape_default_43, sum_dim_int_list_102, sum_dim_int_list_103, permute_default_86, reshape_default_44, permute_default_87, permute_default_88, permute_default_89, sum_dim_int_list_105, sum_dim_int_list_106, permute_default_90, reshape_default_45, permute_default_91, reshape_default_46, sum_dim_int_list_109, sum_dim_int_list_110, permute_default_92, reshape_default_47, permute_default_93, permute_default_94, permute_default_95, sum_dim_int_list_112, sum_dim_int_list_113, permute_default_96, reshape_default_48, permute_default_97, reshape_default_49, sum_dim_int_list_116, sum_dim_int_list_117, permute_default_98, reshape_default_50, permute_default_99, permute_default_100, permute_default_101, sum_dim_int_list_119, sum_dim_int_list_120, permute_default_102, reshape_default_51, permute_default_103, reshape_default_52, sum_dim_int_list_123, sum_dim_int_list_124, permute_default_104, reshape_default_53, permute_default_105, permute_default_106, permute_default_107, sum_dim_int_list_126, sum_dim_int_list_127, permute_default_108, reshape_default_54, permute_default_109, reshape_default_55, sum_dim_int_list_130, sum_dim_int_list_131, permute_default_110, reshape_default_56, permute_default_111, permute_default_112, permute_default_113, sum_dim_int_list_133, sum_dim_int_list_134, permute_default_114, reshape_default_57, permute_default_115, reshape_default_58, sum_dim_int_list_137, sum_dim_int_list_138, permute_default_116, reshape_default_59, permute_default_117, permute_default_118, permute_default_119, sum_dim_int_list_140, sum_dim_int_list_141, permute_default_120, reshape_default_60, permute_default_121, reshape_default_61, sum_dim_int_list_144, sum_dim_int_list_145, permute_default_122, reshape_default_62, permute_default_123, permute_default_124, permute_default_125, sum_dim_int_list_147, sum_dim_int_list_148, permute_default_126, reshape_default_63, permute_default_127, reshape_default_64, sum_dim_int_list_151, sum_dim_int_list_152, permute_default_128, reshape_default_65, permute_default_129, permute_default_130, permute_default_131, sum_dim_int_list_154, sum_dim_int_list_155, permute_default_132, reshape_default_66, permute_default_133, reshape_default_67, sum_dim_int_list_158, sum_dim_int_list_159, permute_default_134, reshape_default_68, permute_default_135, permute_default_136, permute_default_137, sum_dim_int_list_161, sum_dim_int_list_162, permute_default_138, reshape_default_69, permute_default_139, reshape_default_70, sum_dim_int_list_165, sum_dim_int_list_166, permute_default_140, reshape_default_71, permute_default_141, permute_default_142, permute_default_143, sum_dim_int_list_170, sum_dim_int_list_171, index_put_default, add_tensor_4)
