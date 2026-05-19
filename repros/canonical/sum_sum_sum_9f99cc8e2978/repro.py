"""
Standalone repro captured via capture_hook.
Label: hf_MegatronBertForCausalLM_train
Pattern hash: 9f99cc8e2978
Shape hash: f765240d
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, view_535: "f32[8192, 29056]", view_537: "f32[16, 512, 1024]", mul_320: "f32[16, 512, 1024]", view_538: "f32[8192, 1024]", view_540: "f32[16, 512, 1024]", mul_315: "f32[16, 512, 1024]", view_541: "f32[8192, 1024]", view_544: "f32[8192, 4096]", view_546: "f32[16, 512, 1024]", mul_308: "f32[16, 512, 1024]", view_547: "f32[8192, 1024]", view_558: "f32[8192, 1024]", view_562: "f32[8192, 1024]", view_566: "f32[8192, 1024]", add_206: "f32[16, 512, 1024]", mul_302: "f32[16, 512, 1024]", view_569: "f32[8192, 1024]", view_572: "f32[8192, 4096]", view_574: "f32[16, 512, 1024]", mul_295: "f32[16, 512, 1024]", view_575: "f32[8192, 1024]", view_586: "f32[8192, 1024]", view_590: "f32[8192, 1024]", view_594: "f32[8192, 1024]", add_212: "f32[16, 512, 1024]", mul_289: "f32[16, 512, 1024]", view_597: "f32[8192, 1024]", view_600: "f32[8192, 4096]", view_602: "f32[16, 512, 1024]", mul_282: "f32[16, 512, 1024]", view_603: "f32[8192, 1024]", view_614: "f32[8192, 1024]", view_618: "f32[8192, 1024]", view_622: "f32[8192, 1024]", add_218: "f32[16, 512, 1024]", mul_276: "f32[16, 512, 1024]", view_625: "f32[8192, 1024]", view_628: "f32[8192, 4096]", view_630: "f32[16, 512, 1024]", mul_269: "f32[16, 512, 1024]", view_631: "f32[8192, 1024]", view_642: "f32[8192, 1024]", view_646: "f32[8192, 1024]", view_650: "f32[8192, 1024]", add_224: "f32[16, 512, 1024]", mul_263: "f32[16, 512, 1024]", view_653: "f32[8192, 1024]", view_656: "f32[8192, 4096]", view_658: "f32[16, 512, 1024]", mul_256: "f32[16, 512, 1024]", view_659: "f32[8192, 1024]", view_670: "f32[8192, 1024]", view_674: "f32[8192, 1024]", view_678: "f32[8192, 1024]", add_230: "f32[16, 512, 1024]", mul_250: "f32[16, 512, 1024]", view_681: "f32[8192, 1024]", view_684: "f32[8192, 4096]", view_686: "f32[16, 512, 1024]", mul_243: "f32[16, 512, 1024]", view_687: "f32[8192, 1024]", view_698: "f32[8192, 1024]", view_702: "f32[8192, 1024]", view_706: "f32[8192, 1024]", add_236: "f32[16, 512, 1024]", mul_237: "f32[16, 512, 1024]", view_709: "f32[8192, 1024]", view_712: "f32[8192, 4096]", view_714: "f32[16, 512, 1024]", mul_230: "f32[16, 512, 1024]", view_715: "f32[8192, 1024]", view_726: "f32[8192, 1024]", view_730: "f32[8192, 1024]", view_734: "f32[8192, 1024]", add_242: "f32[16, 512, 1024]", mul_224: "f32[16, 512, 1024]", view_737: "f32[8192, 1024]", view_740: "f32[8192, 4096]", view_742: "f32[16, 512, 1024]", mul_217: "f32[16, 512, 1024]", view_743: "f32[8192, 1024]", view_754: "f32[8192, 1024]", view_758: "f32[8192, 1024]", view_762: "f32[8192, 1024]", add_248: "f32[16, 512, 1024]", mul_211: "f32[16, 512, 1024]", view_765: "f32[8192, 1024]", view_768: "f32[8192, 4096]", view_770: "f32[16, 512, 1024]", mul_204: "f32[16, 512, 1024]", view_771: "f32[8192, 1024]", view_782: "f32[8192, 1024]", view_786: "f32[8192, 1024]", view_790: "f32[8192, 1024]", add_254: "f32[16, 512, 1024]", mul_198: "f32[16, 512, 1024]", view_793: "f32[8192, 1024]", view_796: "f32[8192, 4096]", view_798: "f32[16, 512, 1024]", mul_191: "f32[16, 512, 1024]", view_799: "f32[8192, 1024]", view_810: "f32[8192, 1024]", view_814: "f32[8192, 1024]", view_818: "f32[8192, 1024]", add_260: "f32[16, 512, 1024]", mul_185: "f32[16, 512, 1024]", view_821: "f32[8192, 1024]", view_824: "f32[8192, 4096]", view_826: "f32[16, 512, 1024]", mul_178: "f32[16, 512, 1024]", view_827: "f32[8192, 1024]", view_838: "f32[8192, 1024]", view_842: "f32[8192, 1024]", view_846: "f32[8192, 1024]", add_266: "f32[16, 512, 1024]", mul_172: "f32[16, 512, 1024]", view_849: "f32[8192, 1024]", view_852: "f32[8192, 4096]", view_854: "f32[16, 512, 1024]", mul_165: "f32[16, 512, 1024]", view_855: "f32[8192, 1024]", view_866: "f32[8192, 1024]", view_870: "f32[8192, 1024]", view_874: "f32[8192, 1024]", add_272: "f32[16, 512, 1024]", mul_159: "f32[16, 512, 1024]", view_877: "f32[8192, 1024]", view_880: "f32[8192, 4096]", view_882: "f32[16, 512, 1024]", mul_152: "f32[16, 512, 1024]", view_883: "f32[8192, 1024]", view_894: "f32[8192, 1024]", view_898: "f32[8192, 1024]", view_902: "f32[8192, 1024]", add_278: "f32[16, 512, 1024]", mul_146: "f32[16, 512, 1024]", view_905: "f32[8192, 1024]", view_908: "f32[8192, 4096]", view_910: "f32[16, 512, 1024]", mul_139: "f32[16, 512, 1024]", view_911: "f32[8192, 1024]", view_922: "f32[8192, 1024]", view_926: "f32[8192, 1024]", view_930: "f32[8192, 1024]", add_284: "f32[16, 512, 1024]", mul_133: "f32[16, 512, 1024]", view_933: "f32[8192, 1024]", view_936: "f32[8192, 4096]", view_938: "f32[16, 512, 1024]", mul_126: "f32[16, 512, 1024]", view_939: "f32[8192, 1024]", view_950: "f32[8192, 1024]", view_954: "f32[8192, 1024]", view_958: "f32[8192, 1024]", add_290: "f32[16, 512, 1024]", mul_120: "f32[16, 512, 1024]", view_961: "f32[8192, 1024]", view_964: "f32[8192, 4096]", view_966: "f32[16, 512, 1024]", mul_113: "f32[16, 512, 1024]", view_967: "f32[8192, 1024]", view_978: "f32[8192, 1024]", view_982: "f32[8192, 1024]", view_986: "f32[8192, 1024]", add_296: "f32[16, 512, 1024]", mul_107: "f32[16, 512, 1024]", view_989: "f32[8192, 1024]", view_992: "f32[8192, 4096]", view_994: "f32[16, 512, 1024]", mul_100: "f32[16, 512, 1024]", view_995: "f32[8192, 1024]", view_1006: "f32[8192, 1024]", view_1010: "f32[8192, 1024]", view_1014: "f32[8192, 1024]", add_302: "f32[16, 512, 1024]", mul_94: "f32[16, 512, 1024]", view_1017: "f32[8192, 1024]", view_1020: "f32[8192, 4096]", view_1022: "f32[16, 512, 1024]", mul_87: "f32[16, 512, 1024]", view_1023: "f32[8192, 1024]", view_1034: "f32[8192, 1024]", view_1038: "f32[8192, 1024]", view_1042: "f32[8192, 1024]", add_308: "f32[16, 512, 1024]", mul_81: "f32[16, 512, 1024]", view_1045: "f32[8192, 1024]", view_1048: "f32[8192, 4096]", view_1050: "f32[16, 512, 1024]", mul_74: "f32[16, 512, 1024]", view_1051: "f32[8192, 1024]", view_1062: "f32[8192, 1024]", view_1066: "f32[8192, 1024]", view_1070: "f32[8192, 1024]", add_314: "f32[16, 512, 1024]", mul_68: "f32[16, 512, 1024]", view_1073: "f32[8192, 1024]", view_1076: "f32[8192, 4096]", view_1078: "f32[16, 512, 1024]", mul_61: "f32[16, 512, 1024]", view_1079: "f32[8192, 1024]", view_1090: "f32[8192, 1024]", view_1094: "f32[8192, 1024]", view_1098: "f32[8192, 1024]", add_320: "f32[16, 512, 1024]", mul_55: "f32[16, 512, 1024]", view_1101: "f32[8192, 1024]", view_1104: "f32[8192, 4096]", view_1106: "f32[16, 512, 1024]", mul_48: "f32[16, 512, 1024]", view_1107: "f32[8192, 1024]", view_1118: "f32[8192, 1024]", view_1122: "f32[8192, 1024]", view_1126: "f32[8192, 1024]", add_326: "f32[16, 512, 1024]", mul_42: "f32[16, 512, 1024]", view_1129: "f32[8192, 1024]", view_1132: "f32[8192, 4096]", view_1134: "f32[16, 512, 1024]", mul_35: "f32[16, 512, 1024]", view_1135: "f32[8192, 1024]", view_1146: "f32[8192, 1024]", view_1150: "f32[8192, 1024]", view_1154: "f32[8192, 1024]", add_332: "f32[16, 512, 1024]", mul_29: "f32[16, 512, 1024]", view_1157: "f32[8192, 1024]", view_1160: "f32[8192, 4096]", view_1162: "f32[16, 512, 1024]", mul_22: "f32[16, 512, 1024]", view_1163: "f32[8192, 1024]", view_1174: "f32[8192, 1024]", view_1178: "f32[8192, 1024]", view_1182: "f32[8192, 1024]", add_338: "f32[16, 512, 1024]", mul_16: "f32[16, 512, 1024]", view_1185: "f32[8192, 1024]", view_1188: "f32[8192, 4096]", view_1190: "f32[16, 512, 1024]", mul_9: "f32[16, 512, 1024]", view_1191: "f32[8192, 1024]", view_1202: "f32[8192, 1024]", mm_286: "f32[8192, 1024]", view_1206: "f32[8192, 1024]", mm_288: "f32[8192, 1024]", view_1210: "f32[8192, 1024]", mm_290: "f32[8192, 1024]", primals_7: "f32[1024]", mul_3: "f32[16, 512, 1024]", div_123: "f32[16, 512, 1]", add_342: "f32[16, 512, 1024]", gt: "b8[16, 512, 1024]", primals_4: "i64[1, 512]", full_default_3: "f32[]", full_default: "i64[16, 512]", primals_2: "i64[16, 512]", mm_1: "f32[29056, 1024]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12, _shape_param_13, _shape_param_14, _shape_param_15, _shape_param_16, _shape_param_17, _shape_param_18, _shape_param_19, _shape_param_20, _shape_param_21, _shape_param_22, _shape_param_23, _shape_param_24, _shape_param_25, _shape_param_26, _shape_param_27, _shape_param_28, _shape_param_29, _shape_param_30, _shape_param_31, _shape_param_32, _shape_param_33, _shape_param_34, _shape_param_35, _shape_param_36, _shape_param_37, _shape_param_38, _shape_param_39, _shape_param_40, _shape_param_41, _shape_param_42, _shape_param_43, _shape_param_44, _shape_param_45, _shape_param_46, _shape_param_47, _shape_param_48, _shape_param_49, _shape_param_50, _shape_param_51, _shape_param_52, _shape_param_53, _shape_param_54, _shape_param_55, _shape_param_56, _shape_param_57, _shape_param_58, _shape_param_59, _shape_param_60, _shape_param_61, _shape_param_62, _shape_param_63, _shape_param_64, _shape_param_65, _shape_param_66, _shape_param_67, _shape_param_68, _shape_param_69, _shape_param_70, _shape_param_71, _shape_param_72, _shape_param_73, _shape_param_74, _shape_param_75, _shape_param_76, _shape_param_77, _shape_param_78, _shape_param_79, _shape_param_80, _shape_param_81, _shape_param_82, _shape_param_83, _shape_param_84, _shape_param_85, _shape_param_86, _shape_param_87, _shape_param_88, _shape_param_89, _shape_param_90, _shape_param_91, _shape_param_92, _shape_param_93, _shape_param_94, _shape_param_95, _shape_param_96, _shape_param_97, _shape_param_98, _shape_param_99, _shape_param_100, _shape_param_101, _shape_param_102, _shape_param_103, _shape_param_104, _shape_param_105, _shape_param_106, _shape_param_107, _shape_param_108, _shape_param_109, _shape_param_110, _shape_param_111, _shape_param_112, _shape_param_113, _shape_param_114, _shape_param_115, _shape_param_116, _shape_param_117, _shape_param_118, _shape_param_119, _shape_param_120, _shape_param_121, _shape_param_122, _shape_param_123, _shape_param_124, _shape_param_125, _shape_param_126, _shape_param_127, _shape_param_128, _shape_param_129, _shape_param_130, _shape_param_131, _shape_param_132, _shape_param_133, _shape_param_134, _shape_param_135, _shape_param_136, _shape_param_137, _shape_param_138, _shape_param_139, _shape_param_140, _shape_param_141, _shape_param_142, _shape_param_143, _shape_param_144, _shape_param_145, _shape_param_146, _shape_param_147, _shape_param_148):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:465 in forward, code: hidden_states = self.decoder(hidden_states)
        sum_dim_int_list: "f32[1, 29056]" = torch.ops.aten.sum.dim_IntList(view_535, [0], True);  view_535 = None
        reshape_default: "f32[29056]" = torch.ops.aten.reshape.default(sum_dim_int_list, _shape_param_0);  sum_dim_int_list = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:448 in forward, code: hidden_states = self.LayerNorm(hidden_states)
        mul_tensor: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(view_537, mul_320);  mul_320 = None
        sum_dim_int_list_1: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [0, 1]);  mul_tensor = None
        sum_dim_int_list_2: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_537, [0, 1]);  view_537 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:446 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_538, [1, 0])
        sum_dim_int_list_3: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_538, [0], True);  view_538 = None
        reshape_default_1: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_3, _shape_param_1);  sum_dim_int_list_3 = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:392 in forward, code: hidden_states = self.ln(hidden_states)
        mul_tensor_1: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(view_540, mul_315);  mul_315 = None
        sum_dim_int_list_4: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_1, [0, 1]);  mul_tensor_1 = None
        sum_dim_int_list_5: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_540, [0, 1]);  view_540 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:261 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_1: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_541, [1, 0])
        sum_dim_int_list_6: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_541, [0], True);  view_541 = None
        reshape_default_2: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_6, _shape_param_2);  sum_dim_int_list_6 = _shape_param_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_2: "f32[4096, 8192]" = torch.ops.aten.permute.default(view_544, [1, 0])
        sum_dim_int_list_7: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_544, [0], True);  view_544 = None
        reshape_default_3: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_7, _shape_param_3);  sum_dim_int_list_7 = _shape_param_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        mul_tensor_2: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(view_546, mul_308);  mul_308 = None
        sum_dim_int_list_8: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [0, 1]);  mul_tensor_2 = None
        sum_dim_int_list_9: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_546, [0, 1]);  view_546 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:202 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_3: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_547, [1, 0])
        sum_dim_int_list_10: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_547, [0], True);  view_547 = None
        reshape_default_4: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_10, _shape_param_4);  sum_dim_int_list_10 = _shape_param_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:160 in forward, code: value_layer = self.value(current_states)
        permute_default_4: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_558, [1, 0])
        sum_dim_int_list_11: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_558, [0], True);  view_558 = None
        reshape_default_5: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_11, _shape_param_5);  sum_dim_int_list_11 = _shape_param_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:158 in forward, code: key_layer = self.key(current_states)
        permute_default_5: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_562, [1, 0])
        sum_dim_int_list_12: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_562, [0], True);  view_562 = None
        reshape_default_6: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_12, _shape_param_6);  sum_dim_int_list_12 = _shape_param_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:136 in forward, code: query_layer = self.query(hidden_states)
        permute_default_6: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_566, [1, 0])
        sum_dim_int_list_13: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_566, [0], True);  view_566 = None
        reshape_default_7: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_13, _shape_param_7);  sum_dim_int_list_13 = _shape_param_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        mul_tensor_3: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(add_206, mul_302);  mul_302 = None
        sum_dim_int_list_14: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_3, [0, 1]);  mul_tensor_3 = None
        sum_dim_int_list_15: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_206, [0, 1]);  add_206 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:261 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_7: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_569, [1, 0])
        sum_dim_int_list_16: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_569, [0], True);  view_569 = None
        reshape_default_8: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_16, _shape_param_8);  sum_dim_int_list_16 = _shape_param_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_8: "f32[4096, 8192]" = torch.ops.aten.permute.default(view_572, [1, 0])
        sum_dim_int_list_17: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_572, [0], True);  view_572 = None
        reshape_default_9: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_17, _shape_param_9);  sum_dim_int_list_17 = _shape_param_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        mul_tensor_4: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(view_574, mul_295);  mul_295 = None
        sum_dim_int_list_18: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_4, [0, 1]);  mul_tensor_4 = None
        sum_dim_int_list_19: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_574, [0, 1]);  view_574 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:202 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_9: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_575, [1, 0])
        sum_dim_int_list_20: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_575, [0], True);  view_575 = None
        reshape_default_10: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_20, _shape_param_10);  sum_dim_int_list_20 = _shape_param_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:160 in forward, code: value_layer = self.value(current_states)
        permute_default_10: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_586, [1, 0])
        sum_dim_int_list_21: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_586, [0], True);  view_586 = None
        reshape_default_11: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_21, _shape_param_11);  sum_dim_int_list_21 = _shape_param_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:158 in forward, code: key_layer = self.key(current_states)
        permute_default_11: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_590, [1, 0])
        sum_dim_int_list_22: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_590, [0], True);  view_590 = None
        reshape_default_12: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_22, _shape_param_12);  sum_dim_int_list_22 = _shape_param_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:136 in forward, code: query_layer = self.query(hidden_states)
        permute_default_12: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_594, [1, 0])
        sum_dim_int_list_23: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_594, [0], True);  view_594 = None
        reshape_default_13: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_23, _shape_param_13);  sum_dim_int_list_23 = _shape_param_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        mul_tensor_5: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(add_212, mul_289);  mul_289 = None
        sum_dim_int_list_24: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_5, [0, 1]);  mul_tensor_5 = None
        sum_dim_int_list_25: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_212, [0, 1]);  add_212 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:261 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_13: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_597, [1, 0])
        sum_dim_int_list_26: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_597, [0], True);  view_597 = None
        reshape_default_14: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_26, _shape_param_14);  sum_dim_int_list_26 = _shape_param_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_14: "f32[4096, 8192]" = torch.ops.aten.permute.default(view_600, [1, 0])
        sum_dim_int_list_27: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_600, [0], True);  view_600 = None
        reshape_default_15: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_27, _shape_param_15);  sum_dim_int_list_27 = _shape_param_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        mul_tensor_6: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(view_602, mul_282);  mul_282 = None
        sum_dim_int_list_28: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_6, [0, 1]);  mul_tensor_6 = None
        sum_dim_int_list_29: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_602, [0, 1]);  view_602 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:202 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_15: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_603, [1, 0])
        sum_dim_int_list_30: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_603, [0], True);  view_603 = None
        reshape_default_16: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_30, _shape_param_16);  sum_dim_int_list_30 = _shape_param_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:160 in forward, code: value_layer = self.value(current_states)
        permute_default_16: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_614, [1, 0])
        sum_dim_int_list_31: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_614, [0], True);  view_614 = None
        reshape_default_17: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_31, _shape_param_17);  sum_dim_int_list_31 = _shape_param_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:158 in forward, code: key_layer = self.key(current_states)
        permute_default_17: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_618, [1, 0])
        sum_dim_int_list_32: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_618, [0], True);  view_618 = None
        reshape_default_18: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_32, _shape_param_18);  sum_dim_int_list_32 = _shape_param_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:136 in forward, code: query_layer = self.query(hidden_states)
        permute_default_18: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_622, [1, 0])
        sum_dim_int_list_33: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_622, [0], True);  view_622 = None
        reshape_default_19: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_33, _shape_param_19);  sum_dim_int_list_33 = _shape_param_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        mul_tensor_7: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(add_218, mul_276);  mul_276 = None
        sum_dim_int_list_34: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_7, [0, 1]);  mul_tensor_7 = None
        sum_dim_int_list_35: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_218, [0, 1]);  add_218 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:261 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_19: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_625, [1, 0])
        sum_dim_int_list_36: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_625, [0], True);  view_625 = None
        reshape_default_20: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_36, _shape_param_20);  sum_dim_int_list_36 = _shape_param_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_20: "f32[4096, 8192]" = torch.ops.aten.permute.default(view_628, [1, 0])
        sum_dim_int_list_37: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_628, [0], True);  view_628 = None
        reshape_default_21: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_37, _shape_param_21);  sum_dim_int_list_37 = _shape_param_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        mul_tensor_8: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(view_630, mul_269);  mul_269 = None
        sum_dim_int_list_38: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_8, [0, 1]);  mul_tensor_8 = None
        sum_dim_int_list_39: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_630, [0, 1]);  view_630 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:202 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_21: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_631, [1, 0])
        sum_dim_int_list_40: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_631, [0], True);  view_631 = None
        reshape_default_22: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_40, _shape_param_22);  sum_dim_int_list_40 = _shape_param_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:160 in forward, code: value_layer = self.value(current_states)
        permute_default_22: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_642, [1, 0])
        sum_dim_int_list_41: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_642, [0], True);  view_642 = None
        reshape_default_23: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_41, _shape_param_23);  sum_dim_int_list_41 = _shape_param_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:158 in forward, code: key_layer = self.key(current_states)
        permute_default_23: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_646, [1, 0])
        sum_dim_int_list_42: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_646, [0], True);  view_646 = None
        reshape_default_24: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_42, _shape_param_24);  sum_dim_int_list_42 = _shape_param_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:136 in forward, code: query_layer = self.query(hidden_states)
        permute_default_24: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_650, [1, 0])
        sum_dim_int_list_43: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_650, [0], True);  view_650 = None
        reshape_default_25: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_43, _shape_param_25);  sum_dim_int_list_43 = _shape_param_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        mul_tensor_9: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(add_224, mul_263);  mul_263 = None
        sum_dim_int_list_44: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_9, [0, 1]);  mul_tensor_9 = None
        sum_dim_int_list_45: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_224, [0, 1]);  add_224 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:261 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_25: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_653, [1, 0])
        sum_dim_int_list_46: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_653, [0], True);  view_653 = None
        reshape_default_26: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_46, _shape_param_26);  sum_dim_int_list_46 = _shape_param_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_26: "f32[4096, 8192]" = torch.ops.aten.permute.default(view_656, [1, 0])
        sum_dim_int_list_47: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_656, [0], True);  view_656 = None
        reshape_default_27: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_47, _shape_param_27);  sum_dim_int_list_47 = _shape_param_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        mul_tensor_10: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(view_658, mul_256);  mul_256 = None
        sum_dim_int_list_48: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_10, [0, 1]);  mul_tensor_10 = None
        sum_dim_int_list_49: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_658, [0, 1]);  view_658 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:202 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_27: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_659, [1, 0])
        sum_dim_int_list_50: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_659, [0], True);  view_659 = None
        reshape_default_28: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_50, _shape_param_28);  sum_dim_int_list_50 = _shape_param_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:160 in forward, code: value_layer = self.value(current_states)
        permute_default_28: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_670, [1, 0])
        sum_dim_int_list_51: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_670, [0], True);  view_670 = None
        reshape_default_29: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_51, _shape_param_29);  sum_dim_int_list_51 = _shape_param_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:158 in forward, code: key_layer = self.key(current_states)
        permute_default_29: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_674, [1, 0])
        sum_dim_int_list_52: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_674, [0], True);  view_674 = None
        reshape_default_30: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_52, _shape_param_30);  sum_dim_int_list_52 = _shape_param_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:136 in forward, code: query_layer = self.query(hidden_states)
        permute_default_30: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_678, [1, 0])
        sum_dim_int_list_53: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_678, [0], True);  view_678 = None
        reshape_default_31: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_53, _shape_param_31);  sum_dim_int_list_53 = _shape_param_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        mul_tensor_11: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(add_230, mul_250);  mul_250 = None
        sum_dim_int_list_54: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_11, [0, 1]);  mul_tensor_11 = None
        sum_dim_int_list_55: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_230, [0, 1]);  add_230 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:261 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_31: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_681, [1, 0])
        sum_dim_int_list_56: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_681, [0], True);  view_681 = None
        reshape_default_32: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_56, _shape_param_32);  sum_dim_int_list_56 = _shape_param_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_32: "f32[4096, 8192]" = torch.ops.aten.permute.default(view_684, [1, 0])
        sum_dim_int_list_57: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_684, [0], True);  view_684 = None
        reshape_default_33: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_57, _shape_param_33);  sum_dim_int_list_57 = _shape_param_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        mul_tensor_12: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(view_686, mul_243);  mul_243 = None
        sum_dim_int_list_58: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_12, [0, 1]);  mul_tensor_12 = None
        sum_dim_int_list_59: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_686, [0, 1]);  view_686 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:202 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_33: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_687, [1, 0])
        sum_dim_int_list_60: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_687, [0], True);  view_687 = None
        reshape_default_34: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_60, _shape_param_34);  sum_dim_int_list_60 = _shape_param_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:160 in forward, code: value_layer = self.value(current_states)
        permute_default_34: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_698, [1, 0])
        sum_dim_int_list_61: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_698, [0], True);  view_698 = None
        reshape_default_35: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_61, _shape_param_35);  sum_dim_int_list_61 = _shape_param_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:158 in forward, code: key_layer = self.key(current_states)
        permute_default_35: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_702, [1, 0])
        sum_dim_int_list_62: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_702, [0], True);  view_702 = None
        reshape_default_36: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_62, _shape_param_36);  sum_dim_int_list_62 = _shape_param_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:136 in forward, code: query_layer = self.query(hidden_states)
        permute_default_36: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_706, [1, 0])
        sum_dim_int_list_63: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_706, [0], True);  view_706 = None
        reshape_default_37: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_63, _shape_param_37);  sum_dim_int_list_63 = _shape_param_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        mul_tensor_13: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(add_236, mul_237);  mul_237 = None
        sum_dim_int_list_64: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_13, [0, 1]);  mul_tensor_13 = None
        sum_dim_int_list_65: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_236, [0, 1]);  add_236 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:261 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_37: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_709, [1, 0])
        sum_dim_int_list_66: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_709, [0], True);  view_709 = None
        reshape_default_38: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_66, _shape_param_38);  sum_dim_int_list_66 = _shape_param_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_38: "f32[4096, 8192]" = torch.ops.aten.permute.default(view_712, [1, 0])
        sum_dim_int_list_67: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_712, [0], True);  view_712 = None
        reshape_default_39: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_67, _shape_param_39);  sum_dim_int_list_67 = _shape_param_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        mul_tensor_14: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(view_714, mul_230);  mul_230 = None
        sum_dim_int_list_68: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_14, [0, 1]);  mul_tensor_14 = None
        sum_dim_int_list_69: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_714, [0, 1]);  view_714 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:202 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_39: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_715, [1, 0])
        sum_dim_int_list_70: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_715, [0], True);  view_715 = None
        reshape_default_40: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_70, _shape_param_40);  sum_dim_int_list_70 = _shape_param_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:160 in forward, code: value_layer = self.value(current_states)
        permute_default_40: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_726, [1, 0])
        sum_dim_int_list_71: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_726, [0], True);  view_726 = None
        reshape_default_41: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_71, _shape_param_41);  sum_dim_int_list_71 = _shape_param_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:158 in forward, code: key_layer = self.key(current_states)
        permute_default_41: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_730, [1, 0])
        sum_dim_int_list_72: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_730, [0], True);  view_730 = None
        reshape_default_42: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_72, _shape_param_42);  sum_dim_int_list_72 = _shape_param_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:136 in forward, code: query_layer = self.query(hidden_states)
        permute_default_42: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_734, [1, 0])
        sum_dim_int_list_73: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_734, [0], True);  view_734 = None
        reshape_default_43: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_73, _shape_param_43);  sum_dim_int_list_73 = _shape_param_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        mul_tensor_15: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(add_242, mul_224);  mul_224 = None
        sum_dim_int_list_74: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_15, [0, 1]);  mul_tensor_15 = None
        sum_dim_int_list_75: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_242, [0, 1]);  add_242 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:261 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_43: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_737, [1, 0])
        sum_dim_int_list_76: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_737, [0], True);  view_737 = None
        reshape_default_44: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_76, _shape_param_44);  sum_dim_int_list_76 = _shape_param_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_44: "f32[4096, 8192]" = torch.ops.aten.permute.default(view_740, [1, 0])
        sum_dim_int_list_77: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_740, [0], True);  view_740 = None
        reshape_default_45: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_77, _shape_param_45);  sum_dim_int_list_77 = _shape_param_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        mul_tensor_16: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(view_742, mul_217);  mul_217 = None
        sum_dim_int_list_78: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_16, [0, 1]);  mul_tensor_16 = None
        sum_dim_int_list_79: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_742, [0, 1]);  view_742 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:202 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_45: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_743, [1, 0])
        sum_dim_int_list_80: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_743, [0], True);  view_743 = None
        reshape_default_46: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_80, _shape_param_46);  sum_dim_int_list_80 = _shape_param_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:160 in forward, code: value_layer = self.value(current_states)
        permute_default_46: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_754, [1, 0])
        sum_dim_int_list_81: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_754, [0], True);  view_754 = None
        reshape_default_47: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_81, _shape_param_47);  sum_dim_int_list_81 = _shape_param_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:158 in forward, code: key_layer = self.key(current_states)
        permute_default_47: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_758, [1, 0])
        sum_dim_int_list_82: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_758, [0], True);  view_758 = None
        reshape_default_48: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_82, _shape_param_48);  sum_dim_int_list_82 = _shape_param_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:136 in forward, code: query_layer = self.query(hidden_states)
        permute_default_48: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_762, [1, 0])
        sum_dim_int_list_83: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_762, [0], True);  view_762 = None
        reshape_default_49: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_83, _shape_param_49);  sum_dim_int_list_83 = _shape_param_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        mul_tensor_17: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(add_248, mul_211);  mul_211 = None
        sum_dim_int_list_84: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_17, [0, 1]);  mul_tensor_17 = None
        sum_dim_int_list_85: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_248, [0, 1]);  add_248 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:261 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_49: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_765, [1, 0])
        sum_dim_int_list_86: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_765, [0], True);  view_765 = None
        reshape_default_50: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_86, _shape_param_50);  sum_dim_int_list_86 = _shape_param_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_50: "f32[4096, 8192]" = torch.ops.aten.permute.default(view_768, [1, 0])
        sum_dim_int_list_87: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_768, [0], True);  view_768 = None
        reshape_default_51: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_87, _shape_param_51);  sum_dim_int_list_87 = _shape_param_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        mul_tensor_18: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(view_770, mul_204);  mul_204 = None
        sum_dim_int_list_88: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_18, [0, 1]);  mul_tensor_18 = None
        sum_dim_int_list_89: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_770, [0, 1]);  view_770 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:202 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_51: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_771, [1, 0])
        sum_dim_int_list_90: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_771, [0], True);  view_771 = None
        reshape_default_52: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_90, _shape_param_52);  sum_dim_int_list_90 = _shape_param_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:160 in forward, code: value_layer = self.value(current_states)
        permute_default_52: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_782, [1, 0])
        sum_dim_int_list_91: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_782, [0], True);  view_782 = None
        reshape_default_53: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_91, _shape_param_53);  sum_dim_int_list_91 = _shape_param_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:158 in forward, code: key_layer = self.key(current_states)
        permute_default_53: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_786, [1, 0])
        sum_dim_int_list_92: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_786, [0], True);  view_786 = None
        reshape_default_54: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_92, _shape_param_54);  sum_dim_int_list_92 = _shape_param_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:136 in forward, code: query_layer = self.query(hidden_states)
        permute_default_54: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_790, [1, 0])
        sum_dim_int_list_93: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_790, [0], True);  view_790 = None
        reshape_default_55: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_93, _shape_param_55);  sum_dim_int_list_93 = _shape_param_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        mul_tensor_19: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(add_254, mul_198);  mul_198 = None
        sum_dim_int_list_94: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_19, [0, 1]);  mul_tensor_19 = None
        sum_dim_int_list_95: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_254, [0, 1]);  add_254 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:261 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_55: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_793, [1, 0])
        sum_dim_int_list_96: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_793, [0], True);  view_793 = None
        reshape_default_56: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_96, _shape_param_56);  sum_dim_int_list_96 = _shape_param_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_56: "f32[4096, 8192]" = torch.ops.aten.permute.default(view_796, [1, 0])
        sum_dim_int_list_97: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_796, [0], True);  view_796 = None
        reshape_default_57: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_97, _shape_param_57);  sum_dim_int_list_97 = _shape_param_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        mul_tensor_20: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(view_798, mul_191);  mul_191 = None
        sum_dim_int_list_98: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_20, [0, 1]);  mul_tensor_20 = None
        sum_dim_int_list_99: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_798, [0, 1]);  view_798 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:202 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_57: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_799, [1, 0])
        sum_dim_int_list_100: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_799, [0], True);  view_799 = None
        reshape_default_58: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_100, _shape_param_58);  sum_dim_int_list_100 = _shape_param_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:160 in forward, code: value_layer = self.value(current_states)
        permute_default_58: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_810, [1, 0])
        sum_dim_int_list_101: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_810, [0], True);  view_810 = None
        reshape_default_59: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_101, _shape_param_59);  sum_dim_int_list_101 = _shape_param_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:158 in forward, code: key_layer = self.key(current_states)
        permute_default_59: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_814, [1, 0])
        sum_dim_int_list_102: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_814, [0], True);  view_814 = None
        reshape_default_60: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_102, _shape_param_60);  sum_dim_int_list_102 = _shape_param_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:136 in forward, code: query_layer = self.query(hidden_states)
        permute_default_60: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_818, [1, 0])
        sum_dim_int_list_103: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_818, [0], True);  view_818 = None
        reshape_default_61: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_103, _shape_param_61);  sum_dim_int_list_103 = _shape_param_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        mul_tensor_21: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(add_260, mul_185);  mul_185 = None
        sum_dim_int_list_104: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_21, [0, 1]);  mul_tensor_21 = None
        sum_dim_int_list_105: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_260, [0, 1]);  add_260 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:261 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_61: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_821, [1, 0])
        sum_dim_int_list_106: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_821, [0], True);  view_821 = None
        reshape_default_62: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_106, _shape_param_62);  sum_dim_int_list_106 = _shape_param_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_62: "f32[4096, 8192]" = torch.ops.aten.permute.default(view_824, [1, 0])
        sum_dim_int_list_107: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_824, [0], True);  view_824 = None
        reshape_default_63: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_107, _shape_param_63);  sum_dim_int_list_107 = _shape_param_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        mul_tensor_22: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(view_826, mul_178);  mul_178 = None
        sum_dim_int_list_108: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_22, [0, 1]);  mul_tensor_22 = None
        sum_dim_int_list_109: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_826, [0, 1]);  view_826 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:202 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_63: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_827, [1, 0])
        sum_dim_int_list_110: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_827, [0], True);  view_827 = None
        reshape_default_64: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_110, _shape_param_64);  sum_dim_int_list_110 = _shape_param_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:160 in forward, code: value_layer = self.value(current_states)
        permute_default_64: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_838, [1, 0])
        sum_dim_int_list_111: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_838, [0], True);  view_838 = None
        reshape_default_65: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_111, _shape_param_65);  sum_dim_int_list_111 = _shape_param_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:158 in forward, code: key_layer = self.key(current_states)
        permute_default_65: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_842, [1, 0])
        sum_dim_int_list_112: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_842, [0], True);  view_842 = None
        reshape_default_66: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_112, _shape_param_66);  sum_dim_int_list_112 = _shape_param_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:136 in forward, code: query_layer = self.query(hidden_states)
        permute_default_66: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_846, [1, 0])
        sum_dim_int_list_113: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_846, [0], True);  view_846 = None
        reshape_default_67: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_113, _shape_param_67);  sum_dim_int_list_113 = _shape_param_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        mul_tensor_23: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(add_266, mul_172);  mul_172 = None
        sum_dim_int_list_114: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_23, [0, 1]);  mul_tensor_23 = None
        sum_dim_int_list_115: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_266, [0, 1]);  add_266 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:261 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_67: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_849, [1, 0])
        sum_dim_int_list_116: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_849, [0], True);  view_849 = None
        reshape_default_68: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_116, _shape_param_68);  sum_dim_int_list_116 = _shape_param_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_68: "f32[4096, 8192]" = torch.ops.aten.permute.default(view_852, [1, 0])
        sum_dim_int_list_117: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_852, [0], True);  view_852 = None
        reshape_default_69: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_117, _shape_param_69);  sum_dim_int_list_117 = _shape_param_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        mul_tensor_24: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(view_854, mul_165);  mul_165 = None
        sum_dim_int_list_118: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_24, [0, 1]);  mul_tensor_24 = None
        sum_dim_int_list_119: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_854, [0, 1]);  view_854 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:202 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_69: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_855, [1, 0])
        sum_dim_int_list_120: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_855, [0], True);  view_855 = None
        reshape_default_70: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_120, _shape_param_70);  sum_dim_int_list_120 = _shape_param_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:160 in forward, code: value_layer = self.value(current_states)
        permute_default_70: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_866, [1, 0])
        sum_dim_int_list_121: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_866, [0], True);  view_866 = None
        reshape_default_71: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_121, _shape_param_71);  sum_dim_int_list_121 = _shape_param_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:158 in forward, code: key_layer = self.key(current_states)
        permute_default_71: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_870, [1, 0])
        sum_dim_int_list_122: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_870, [0], True);  view_870 = None
        reshape_default_72: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_122, _shape_param_72);  sum_dim_int_list_122 = _shape_param_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:136 in forward, code: query_layer = self.query(hidden_states)
        permute_default_72: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_874, [1, 0])
        sum_dim_int_list_123: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_874, [0], True);  view_874 = None
        reshape_default_73: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_123, _shape_param_73);  sum_dim_int_list_123 = _shape_param_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        mul_tensor_25: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(add_272, mul_159);  mul_159 = None
        sum_dim_int_list_124: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_25, [0, 1]);  mul_tensor_25 = None
        sum_dim_int_list_125: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_272, [0, 1]);  add_272 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:261 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_73: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_877, [1, 0])
        sum_dim_int_list_126: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_877, [0], True);  view_877 = None
        reshape_default_74: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_126, _shape_param_74);  sum_dim_int_list_126 = _shape_param_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_74: "f32[4096, 8192]" = torch.ops.aten.permute.default(view_880, [1, 0])
        sum_dim_int_list_127: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_880, [0], True);  view_880 = None
        reshape_default_75: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_127, _shape_param_75);  sum_dim_int_list_127 = _shape_param_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        mul_tensor_26: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(view_882, mul_152);  mul_152 = None
        sum_dim_int_list_128: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_26, [0, 1]);  mul_tensor_26 = None
        sum_dim_int_list_129: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_882, [0, 1]);  view_882 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:202 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_75: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_883, [1, 0])
        sum_dim_int_list_130: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_883, [0], True);  view_883 = None
        reshape_default_76: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_130, _shape_param_76);  sum_dim_int_list_130 = _shape_param_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:160 in forward, code: value_layer = self.value(current_states)
        permute_default_76: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_894, [1, 0])
        sum_dim_int_list_131: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_894, [0], True);  view_894 = None
        reshape_default_77: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_131, _shape_param_77);  sum_dim_int_list_131 = _shape_param_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:158 in forward, code: key_layer = self.key(current_states)
        permute_default_77: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_898, [1, 0])
        sum_dim_int_list_132: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_898, [0], True);  view_898 = None
        reshape_default_78: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_132, _shape_param_78);  sum_dim_int_list_132 = _shape_param_78 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:136 in forward, code: query_layer = self.query(hidden_states)
        permute_default_78: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_902, [1, 0])
        sum_dim_int_list_133: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_902, [0], True);  view_902 = None
        reshape_default_79: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_133, _shape_param_79);  sum_dim_int_list_133 = _shape_param_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        mul_tensor_27: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(add_278, mul_146);  mul_146 = None
        sum_dim_int_list_134: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_27, [0, 1]);  mul_tensor_27 = None
        sum_dim_int_list_135: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_278, [0, 1]);  add_278 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:261 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_79: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_905, [1, 0])
        sum_dim_int_list_136: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_905, [0], True);  view_905 = None
        reshape_default_80: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_136, _shape_param_80);  sum_dim_int_list_136 = _shape_param_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_80: "f32[4096, 8192]" = torch.ops.aten.permute.default(view_908, [1, 0])
        sum_dim_int_list_137: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_908, [0], True);  view_908 = None
        reshape_default_81: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_137, _shape_param_81);  sum_dim_int_list_137 = _shape_param_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        mul_tensor_28: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(view_910, mul_139);  mul_139 = None
        sum_dim_int_list_138: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_28, [0, 1]);  mul_tensor_28 = None
        sum_dim_int_list_139: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_910, [0, 1]);  view_910 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:202 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_81: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_911, [1, 0])
        sum_dim_int_list_140: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_911, [0], True);  view_911 = None
        reshape_default_82: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_140, _shape_param_82);  sum_dim_int_list_140 = _shape_param_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:160 in forward, code: value_layer = self.value(current_states)
        permute_default_82: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_922, [1, 0])
        sum_dim_int_list_141: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_922, [0], True);  view_922 = None
        reshape_default_83: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_141, _shape_param_83);  sum_dim_int_list_141 = _shape_param_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:158 in forward, code: key_layer = self.key(current_states)
        permute_default_83: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_926, [1, 0])
        sum_dim_int_list_142: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_926, [0], True);  view_926 = None
        reshape_default_84: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_142, _shape_param_84);  sum_dim_int_list_142 = _shape_param_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:136 in forward, code: query_layer = self.query(hidden_states)
        permute_default_84: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_930, [1, 0])
        sum_dim_int_list_143: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_930, [0], True);  view_930 = None
        reshape_default_85: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_143, _shape_param_85);  sum_dim_int_list_143 = _shape_param_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        mul_tensor_29: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(add_284, mul_133);  mul_133 = None
        sum_dim_int_list_144: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_29, [0, 1]);  mul_tensor_29 = None
        sum_dim_int_list_145: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_284, [0, 1]);  add_284 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:261 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_85: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_933, [1, 0])
        sum_dim_int_list_146: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_933, [0], True);  view_933 = None
        reshape_default_86: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_146, _shape_param_86);  sum_dim_int_list_146 = _shape_param_86 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_86: "f32[4096, 8192]" = torch.ops.aten.permute.default(view_936, [1, 0])
        sum_dim_int_list_147: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_936, [0], True);  view_936 = None
        reshape_default_87: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_147, _shape_param_87);  sum_dim_int_list_147 = _shape_param_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        mul_tensor_30: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(view_938, mul_126);  mul_126 = None
        sum_dim_int_list_148: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_30, [0, 1]);  mul_tensor_30 = None
        sum_dim_int_list_149: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_938, [0, 1]);  view_938 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:202 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_87: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_939, [1, 0])
        sum_dim_int_list_150: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_939, [0], True);  view_939 = None
        reshape_default_88: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_150, _shape_param_88);  sum_dim_int_list_150 = _shape_param_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:160 in forward, code: value_layer = self.value(current_states)
        permute_default_88: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_950, [1, 0])
        sum_dim_int_list_151: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_950, [0], True);  view_950 = None
        reshape_default_89: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_151, _shape_param_89);  sum_dim_int_list_151 = _shape_param_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:158 in forward, code: key_layer = self.key(current_states)
        permute_default_89: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_954, [1, 0])
        sum_dim_int_list_152: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_954, [0], True);  view_954 = None
        reshape_default_90: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_152, _shape_param_90);  sum_dim_int_list_152 = _shape_param_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:136 in forward, code: query_layer = self.query(hidden_states)
        permute_default_90: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_958, [1, 0])
        sum_dim_int_list_153: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_958, [0], True);  view_958 = None
        reshape_default_91: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_153, _shape_param_91);  sum_dim_int_list_153 = _shape_param_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        mul_tensor_31: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(add_290, mul_120);  mul_120 = None
        sum_dim_int_list_154: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_31, [0, 1]);  mul_tensor_31 = None
        sum_dim_int_list_155: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_290, [0, 1]);  add_290 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:261 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_91: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_961, [1, 0])
        sum_dim_int_list_156: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_961, [0], True);  view_961 = None
        reshape_default_92: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_156, _shape_param_92);  sum_dim_int_list_156 = _shape_param_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_92: "f32[4096, 8192]" = torch.ops.aten.permute.default(view_964, [1, 0])
        sum_dim_int_list_157: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_964, [0], True);  view_964 = None
        reshape_default_93: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_157, _shape_param_93);  sum_dim_int_list_157 = _shape_param_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        mul_tensor_32: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(view_966, mul_113);  mul_113 = None
        sum_dim_int_list_158: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_32, [0, 1]);  mul_tensor_32 = None
        sum_dim_int_list_159: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_966, [0, 1]);  view_966 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:202 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_93: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_967, [1, 0])
        sum_dim_int_list_160: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_967, [0], True);  view_967 = None
        reshape_default_94: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_160, _shape_param_94);  sum_dim_int_list_160 = _shape_param_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:160 in forward, code: value_layer = self.value(current_states)
        permute_default_94: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_978, [1, 0])
        sum_dim_int_list_161: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_978, [0], True);  view_978 = None
        reshape_default_95: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_161, _shape_param_95);  sum_dim_int_list_161 = _shape_param_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:158 in forward, code: key_layer = self.key(current_states)
        permute_default_95: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_982, [1, 0])
        sum_dim_int_list_162: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_982, [0], True);  view_982 = None
        reshape_default_96: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_162, _shape_param_96);  sum_dim_int_list_162 = _shape_param_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:136 in forward, code: query_layer = self.query(hidden_states)
        permute_default_96: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_986, [1, 0])
        sum_dim_int_list_163: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_986, [0], True);  view_986 = None
        reshape_default_97: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_163, _shape_param_97);  sum_dim_int_list_163 = _shape_param_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        mul_tensor_33: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(add_296, mul_107);  mul_107 = None
        sum_dim_int_list_164: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_33, [0, 1]);  mul_tensor_33 = None
        sum_dim_int_list_165: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_296, [0, 1]);  add_296 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:261 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_97: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_989, [1, 0])
        sum_dim_int_list_166: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_989, [0], True);  view_989 = None
        reshape_default_98: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_166, _shape_param_98);  sum_dim_int_list_166 = _shape_param_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_98: "f32[4096, 8192]" = torch.ops.aten.permute.default(view_992, [1, 0])
        sum_dim_int_list_167: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_992, [0], True);  view_992 = None
        reshape_default_99: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_167, _shape_param_99);  sum_dim_int_list_167 = _shape_param_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        mul_tensor_34: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(view_994, mul_100);  mul_100 = None
        sum_dim_int_list_168: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_34, [0, 1]);  mul_tensor_34 = None
        sum_dim_int_list_169: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_994, [0, 1]);  view_994 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:202 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_99: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_995, [1, 0])
        sum_dim_int_list_170: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_995, [0], True);  view_995 = None
        reshape_default_100: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_170, _shape_param_100);  sum_dim_int_list_170 = _shape_param_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:160 in forward, code: value_layer = self.value(current_states)
        permute_default_100: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_1006, [1, 0])
        sum_dim_int_list_171: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1006, [0], True);  view_1006 = None
        reshape_default_101: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_171, _shape_param_101);  sum_dim_int_list_171 = _shape_param_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:158 in forward, code: key_layer = self.key(current_states)
        permute_default_101: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_1010, [1, 0])
        sum_dim_int_list_172: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1010, [0], True);  view_1010 = None
        reshape_default_102: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_172, _shape_param_102);  sum_dim_int_list_172 = _shape_param_102 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:136 in forward, code: query_layer = self.query(hidden_states)
        permute_default_102: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_1014, [1, 0])
        sum_dim_int_list_173: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1014, [0], True);  view_1014 = None
        reshape_default_103: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_173, _shape_param_103);  sum_dim_int_list_173 = _shape_param_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        mul_tensor_35: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(add_302, mul_94);  mul_94 = None
        sum_dim_int_list_174: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_35, [0, 1]);  mul_tensor_35 = None
        sum_dim_int_list_175: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_302, [0, 1]);  add_302 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:261 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_103: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_1017, [1, 0])
        sum_dim_int_list_176: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1017, [0], True);  view_1017 = None
        reshape_default_104: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_176, _shape_param_104);  sum_dim_int_list_176 = _shape_param_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_104: "f32[4096, 8192]" = torch.ops.aten.permute.default(view_1020, [1, 0])
        sum_dim_int_list_177: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1020, [0], True);  view_1020 = None
        reshape_default_105: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_177, _shape_param_105);  sum_dim_int_list_177 = _shape_param_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        mul_tensor_36: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(view_1022, mul_87);  mul_87 = None
        sum_dim_int_list_178: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_36, [0, 1]);  mul_tensor_36 = None
        sum_dim_int_list_179: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_1022, [0, 1]);  view_1022 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:202 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_105: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_1023, [1, 0])
        sum_dim_int_list_180: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1023, [0], True);  view_1023 = None
        reshape_default_106: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_180, _shape_param_106);  sum_dim_int_list_180 = _shape_param_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:160 in forward, code: value_layer = self.value(current_states)
        permute_default_106: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_1034, [1, 0])
        sum_dim_int_list_181: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1034, [0], True);  view_1034 = None
        reshape_default_107: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_181, _shape_param_107);  sum_dim_int_list_181 = _shape_param_107 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:158 in forward, code: key_layer = self.key(current_states)
        permute_default_107: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_1038, [1, 0])
        sum_dim_int_list_182: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1038, [0], True);  view_1038 = None
        reshape_default_108: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_182, _shape_param_108);  sum_dim_int_list_182 = _shape_param_108 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:136 in forward, code: query_layer = self.query(hidden_states)
        permute_default_108: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_1042, [1, 0])
        sum_dim_int_list_183: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1042, [0], True);  view_1042 = None
        reshape_default_109: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_183, _shape_param_109);  sum_dim_int_list_183 = _shape_param_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        mul_tensor_37: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(add_308, mul_81);  mul_81 = None
        sum_dim_int_list_184: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_37, [0, 1]);  mul_tensor_37 = None
        sum_dim_int_list_185: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_308, [0, 1]);  add_308 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:261 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_109: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_1045, [1, 0])
        sum_dim_int_list_186: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1045, [0], True);  view_1045 = None
        reshape_default_110: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_186, _shape_param_110);  sum_dim_int_list_186 = _shape_param_110 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_110: "f32[4096, 8192]" = torch.ops.aten.permute.default(view_1048, [1, 0])
        sum_dim_int_list_187: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1048, [0], True);  view_1048 = None
        reshape_default_111: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_187, _shape_param_111);  sum_dim_int_list_187 = _shape_param_111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        mul_tensor_38: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(view_1050, mul_74);  mul_74 = None
        sum_dim_int_list_188: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_38, [0, 1]);  mul_tensor_38 = None
        sum_dim_int_list_189: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_1050, [0, 1]);  view_1050 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:202 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_111: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_1051, [1, 0])
        sum_dim_int_list_190: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1051, [0], True);  view_1051 = None
        reshape_default_112: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_190, _shape_param_112);  sum_dim_int_list_190 = _shape_param_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:160 in forward, code: value_layer = self.value(current_states)
        permute_default_112: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_1062, [1, 0])
        sum_dim_int_list_191: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1062, [0], True);  view_1062 = None
        reshape_default_113: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_191, _shape_param_113);  sum_dim_int_list_191 = _shape_param_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:158 in forward, code: key_layer = self.key(current_states)
        permute_default_113: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_1066, [1, 0])
        sum_dim_int_list_192: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1066, [0], True);  view_1066 = None
        reshape_default_114: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_192, _shape_param_114);  sum_dim_int_list_192 = _shape_param_114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:136 in forward, code: query_layer = self.query(hidden_states)
        permute_default_114: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_1070, [1, 0])
        sum_dim_int_list_193: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1070, [0], True);  view_1070 = None
        reshape_default_115: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_193, _shape_param_115);  sum_dim_int_list_193 = _shape_param_115 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        mul_tensor_39: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(add_314, mul_68);  mul_68 = None
        sum_dim_int_list_194: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_39, [0, 1]);  mul_tensor_39 = None
        sum_dim_int_list_195: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_314, [0, 1]);  add_314 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:261 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_115: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_1073, [1, 0])
        sum_dim_int_list_196: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1073, [0], True);  view_1073 = None
        reshape_default_116: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_196, _shape_param_116);  sum_dim_int_list_196 = _shape_param_116 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_116: "f32[4096, 8192]" = torch.ops.aten.permute.default(view_1076, [1, 0])
        sum_dim_int_list_197: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1076, [0], True);  view_1076 = None
        reshape_default_117: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_197, _shape_param_117);  sum_dim_int_list_197 = _shape_param_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        mul_tensor_40: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(view_1078, mul_61);  mul_61 = None
        sum_dim_int_list_198: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_40, [0, 1]);  mul_tensor_40 = None
        sum_dim_int_list_199: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_1078, [0, 1]);  view_1078 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:202 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_117: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_1079, [1, 0])
        sum_dim_int_list_200: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1079, [0], True);  view_1079 = None
        reshape_default_118: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_200, _shape_param_118);  sum_dim_int_list_200 = _shape_param_118 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:160 in forward, code: value_layer = self.value(current_states)
        permute_default_118: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_1090, [1, 0])
        sum_dim_int_list_201: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1090, [0], True);  view_1090 = None
        reshape_default_119: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_201, _shape_param_119);  sum_dim_int_list_201 = _shape_param_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:158 in forward, code: key_layer = self.key(current_states)
        permute_default_119: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_1094, [1, 0])
        sum_dim_int_list_202: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1094, [0], True);  view_1094 = None
        reshape_default_120: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_202, _shape_param_120);  sum_dim_int_list_202 = _shape_param_120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:136 in forward, code: query_layer = self.query(hidden_states)
        permute_default_120: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_1098, [1, 0])
        sum_dim_int_list_203: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1098, [0], True);  view_1098 = None
        reshape_default_121: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_203, _shape_param_121);  sum_dim_int_list_203 = _shape_param_121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        mul_tensor_41: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(add_320, mul_55);  mul_55 = None
        sum_dim_int_list_204: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_41, [0, 1]);  mul_tensor_41 = None
        sum_dim_int_list_205: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_320, [0, 1]);  add_320 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:261 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_121: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_1101, [1, 0])
        sum_dim_int_list_206: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1101, [0], True);  view_1101 = None
        reshape_default_122: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_206, _shape_param_122);  sum_dim_int_list_206 = _shape_param_122 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_122: "f32[4096, 8192]" = torch.ops.aten.permute.default(view_1104, [1, 0])
        sum_dim_int_list_207: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1104, [0], True);  view_1104 = None
        reshape_default_123: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_207, _shape_param_123);  sum_dim_int_list_207 = _shape_param_123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        mul_tensor_42: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(view_1106, mul_48);  mul_48 = None
        sum_dim_int_list_208: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_42, [0, 1]);  mul_tensor_42 = None
        sum_dim_int_list_209: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_1106, [0, 1]);  view_1106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:202 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_123: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_1107, [1, 0])
        sum_dim_int_list_210: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1107, [0], True);  view_1107 = None
        reshape_default_124: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_210, _shape_param_124);  sum_dim_int_list_210 = _shape_param_124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:160 in forward, code: value_layer = self.value(current_states)
        permute_default_124: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_1118, [1, 0])
        sum_dim_int_list_211: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1118, [0], True);  view_1118 = None
        reshape_default_125: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_211, _shape_param_125);  sum_dim_int_list_211 = _shape_param_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:158 in forward, code: key_layer = self.key(current_states)
        permute_default_125: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_1122, [1, 0])
        sum_dim_int_list_212: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1122, [0], True);  view_1122 = None
        reshape_default_126: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_212, _shape_param_126);  sum_dim_int_list_212 = _shape_param_126 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:136 in forward, code: query_layer = self.query(hidden_states)
        permute_default_126: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_1126, [1, 0])
        sum_dim_int_list_213: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1126, [0], True);  view_1126 = None
        reshape_default_127: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_213, _shape_param_127);  sum_dim_int_list_213 = _shape_param_127 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        mul_tensor_43: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(add_326, mul_42);  mul_42 = None
        sum_dim_int_list_214: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_43, [0, 1]);  mul_tensor_43 = None
        sum_dim_int_list_215: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_326, [0, 1]);  add_326 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:261 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_127: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_1129, [1, 0])
        sum_dim_int_list_216: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1129, [0], True);  view_1129 = None
        reshape_default_128: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_216, _shape_param_128);  sum_dim_int_list_216 = _shape_param_128 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_128: "f32[4096, 8192]" = torch.ops.aten.permute.default(view_1132, [1, 0])
        sum_dim_int_list_217: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1132, [0], True);  view_1132 = None
        reshape_default_129: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_217, _shape_param_129);  sum_dim_int_list_217 = _shape_param_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        mul_tensor_44: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(view_1134, mul_35);  mul_35 = None
        sum_dim_int_list_218: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_44, [0, 1]);  mul_tensor_44 = None
        sum_dim_int_list_219: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_1134, [0, 1]);  view_1134 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:202 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_129: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_1135, [1, 0])
        sum_dim_int_list_220: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1135, [0], True);  view_1135 = None
        reshape_default_130: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_220, _shape_param_130);  sum_dim_int_list_220 = _shape_param_130 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:160 in forward, code: value_layer = self.value(current_states)
        permute_default_130: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_1146, [1, 0])
        sum_dim_int_list_221: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1146, [0], True);  view_1146 = None
        reshape_default_131: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_221, _shape_param_131);  sum_dim_int_list_221 = _shape_param_131 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:158 in forward, code: key_layer = self.key(current_states)
        permute_default_131: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_1150, [1, 0])
        sum_dim_int_list_222: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1150, [0], True);  view_1150 = None
        reshape_default_132: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_222, _shape_param_132);  sum_dim_int_list_222 = _shape_param_132 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:136 in forward, code: query_layer = self.query(hidden_states)
        permute_default_132: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_1154, [1, 0])
        sum_dim_int_list_223: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1154, [0], True);  view_1154 = None
        reshape_default_133: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_223, _shape_param_133);  sum_dim_int_list_223 = _shape_param_133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        mul_tensor_45: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(add_332, mul_29);  mul_29 = None
        sum_dim_int_list_224: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_45, [0, 1]);  mul_tensor_45 = None
        sum_dim_int_list_225: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_332, [0, 1]);  add_332 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:261 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_133: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_1157, [1, 0])
        sum_dim_int_list_226: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1157, [0], True);  view_1157 = None
        reshape_default_134: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_226, _shape_param_134);  sum_dim_int_list_226 = _shape_param_134 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_134: "f32[4096, 8192]" = torch.ops.aten.permute.default(view_1160, [1, 0])
        sum_dim_int_list_227: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1160, [0], True);  view_1160 = None
        reshape_default_135: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_227, _shape_param_135);  sum_dim_int_list_227 = _shape_param_135 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        mul_tensor_46: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(view_1162, mul_22);  mul_22 = None
        sum_dim_int_list_228: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_46, [0, 1]);  mul_tensor_46 = None
        sum_dim_int_list_229: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_1162, [0, 1]);  view_1162 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:202 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_135: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_1163, [1, 0])
        sum_dim_int_list_230: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1163, [0], True);  view_1163 = None
        reshape_default_136: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_230, _shape_param_136);  sum_dim_int_list_230 = _shape_param_136 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:160 in forward, code: value_layer = self.value(current_states)
        permute_default_136: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_1174, [1, 0])
        sum_dim_int_list_231: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1174, [0], True);  view_1174 = None
        reshape_default_137: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_231, _shape_param_137);  sum_dim_int_list_231 = _shape_param_137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:158 in forward, code: key_layer = self.key(current_states)
        permute_default_137: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_1178, [1, 0])
        sum_dim_int_list_232: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1178, [0], True);  view_1178 = None
        reshape_default_138: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_232, _shape_param_138);  sum_dim_int_list_232 = _shape_param_138 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:136 in forward, code: query_layer = self.query(hidden_states)
        permute_default_138: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_1182, [1, 0])
        sum_dim_int_list_233: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1182, [0], True);  view_1182 = None
        reshape_default_139: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_233, _shape_param_139);  sum_dim_int_list_233 = _shape_param_139 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        mul_tensor_47: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(add_338, mul_16);  mul_16 = None
        sum_dim_int_list_234: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_47, [0, 1]);  mul_tensor_47 = None
        sum_dim_int_list_235: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_338, [0, 1]);  add_338 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:261 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_139: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_1185, [1, 0])
        sum_dim_int_list_236: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1185, [0], True);  view_1185 = None
        reshape_default_140: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_236, _shape_param_140);  sum_dim_int_list_236 = _shape_param_140 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_140: "f32[4096, 8192]" = torch.ops.aten.permute.default(view_1188, [1, 0])
        sum_dim_int_list_237: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1188, [0], True);  view_1188 = None
        reshape_default_141: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_237, _shape_param_141);  sum_dim_int_list_237 = _shape_param_141 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        mul_tensor_48: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(view_1190, mul_9);  mul_9 = None
        sum_dim_int_list_238: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_48, [0, 1]);  mul_tensor_48 = None
        sum_dim_int_list_239: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_1190, [0, 1]);  view_1190 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:202 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_141: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_1191, [1, 0])
        sum_dim_int_list_240: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1191, [0], True);  view_1191 = None
        reshape_default_142: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_240, _shape_param_142);  sum_dim_int_list_240 = _shape_param_142 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:160 in forward, code: value_layer = self.value(current_states)
        permute_default_142: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_1202, [1, 0])
        sum_dim_int_list_241: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1202, [0], True);  view_1202 = None
        reshape_default_143: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_241, _shape_param_143);  sum_dim_int_list_241 = _shape_param_143 = None
        reshape_default_144: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(mm_286, _shape_param_144);  mm_286 = _shape_param_144 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:158 in forward, code: key_layer = self.key(current_states)
        permute_default_143: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_1206, [1, 0])
        sum_dim_int_list_242: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1206, [0], True);  view_1206 = None
        reshape_default_145: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_242, _shape_param_145);  sum_dim_int_list_242 = _shape_param_145 = None
        reshape_default_146: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(mm_288, _shape_param_146);  mm_288 = _shape_param_146 = None
        add_tensor: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(reshape_default_144, reshape_default_146);  reshape_default_144 = reshape_default_146 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:136 in forward, code: query_layer = self.query(hidden_states)
        permute_default_144: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_1210, [1, 0])
        sum_dim_int_list_243: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1210, [0], True);  view_1210 = None
        reshape_default_147: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_243, _shape_param_147);  sum_dim_int_list_243 = _shape_param_147 = None
        reshape_default_148: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(mm_290, _shape_param_148);  mm_290 = _shape_param_148 = None
        add_tensor_1: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(add_tensor, reshape_default_148);  add_tensor = reshape_default_148 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        mul_tensor_49: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(add_tensor_1, primals_7);  primals_7 = None
        mul_tensor_50: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor_49, 1024)
        sum_dim_int_list_244: "f32[16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_49, [2], True)
        mul_tensor_51: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor_49, mul_3);  mul_tensor_49 = None
        sum_dim_int_list_245: "f32[16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_51, [2], True);  mul_tensor_51 = None
        mul_tensor_52: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_3, sum_dim_int_list_245);  sum_dim_int_list_245 = None
        sub_tensor: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_tensor_50, sum_dim_int_list_244);  mul_tensor_50 = sum_dim_int_list_244 = None
        sub_tensor_1: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_tensor, mul_tensor_52);  sub_tensor = mul_tensor_52 = None
        mul_tensor_53: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(div_123, sub_tensor_1);  div_123 = sub_tensor_1 = None
        mul_tensor_54: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(add_tensor_1, mul_3);  mul_3 = None
        sum_dim_int_list_246: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_54, [0, 1]);  mul_tensor_54 = None
        sum_dim_int_list_247: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_tensor_1, [0, 1]);  add_tensor_1 = None
        add_tensor_2: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(add_342, mul_tensor_53);  add_342 = mul_tensor_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:98 in forward, code: embeddings = self.dropout(embeddings)
        convert_element_type_default: "f32[16, 512, 1024]" = torch.ops.prims.convert_element_type.default(gt, torch.float32);  gt = None
        mul_tensor_55: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 1.1111111111111112);  convert_element_type_default = None
        mul_tensor_56: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(add_tensor_2, mul_tensor_55);  add_tensor_2 = mul_tensor_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:94 in forward, code: embeddings += position_embeddings
        sum_dim_int_list_248: "f32[1, 512, 1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_56, [0], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:93 in forward, code: position_embeddings = self.position_embeddings(position_ids)
        eq_scalar: "b8[1, 512]" = torch.ops.aten.eq.Scalar(primals_4, -1)
        unsqueeze_default: "b8[1, 512, 1]" = torch.ops.aten.unsqueeze.default(eq_scalar, -1);  eq_scalar = None
        where_self: "f32[1, 512, 1024]" = torch.ops.aten.where.self(unsqueeze_default, full_default_3, sum_dim_int_list_248);  unsqueeze_default = sum_dim_int_list_248 = None
        full_default_4: "f32[512, 1024]" = torch.ops.aten.full.default([512, 1024], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default: "f32[512, 1024]" = torch.ops.aten.index_put.default(full_default_4, [primals_4], where_self, True);  full_default_4 = primals_4 = where_self = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:90 in forward, code: token_type_embeddings = self.token_type_embeddings(token_type_ids)
        full_default_5: "b8[16, 512, 1]" = torch.ops.aten.full.default([16, 512, 1], False, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_1: "f32[16, 512, 1024]" = torch.ops.aten.where.self(full_default_5, full_default_3, mul_tensor_56);  full_default_5 = None
        full_default_6: "f32[2, 1024]" = torch.ops.aten.full.default([2, 1024], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default_1: "f32[2, 1024]" = torch.ops.aten.index_put.default(full_default_6, [full_default], where_self_1, True);  full_default_6 = full_default = where_self_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:89 in forward, code: inputs_embeds = self.word_embeddings(input_ids)
        eq_scalar_1: "b8[16, 512]" = torch.ops.aten.eq.Scalar(primals_2, 0)
        unsqueeze_default_1: "b8[16, 512, 1]" = torch.ops.aten.unsqueeze.default(eq_scalar_1, -1);  eq_scalar_1 = None
        where_self_2: "f32[16, 512, 1024]" = torch.ops.aten.where.self(unsqueeze_default_1, full_default_3, mul_tensor_56);  unsqueeze_default_1 = full_default_3 = mul_tensor_56 = None
        full_default_7: "f32[29056, 1024]" = torch.ops.aten.full.default([29056, 1024], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default_2: "f32[29056, 1024]" = torch.ops.aten.index_put.default(full_default_7, [primals_2], where_self_2, True);  full_default_7 = primals_2 = where_self_2 = None
        add_tensor_3: "f32[29056, 1024]" = torch.ops.aten.add.Tensor(mm_1, index_put_default_2);  mm_1 = index_put_default_2 = None
        return (reshape_default, sum_dim_int_list_1, sum_dim_int_list_2, permute_default, reshape_default_1, sum_dim_int_list_4, sum_dim_int_list_5, permute_default_1, reshape_default_2, permute_default_2, reshape_default_3, sum_dim_int_list_8, sum_dim_int_list_9, permute_default_3, reshape_default_4, permute_default_4, reshape_default_5, permute_default_5, reshape_default_6, permute_default_6, reshape_default_7, sum_dim_int_list_14, sum_dim_int_list_15, permute_default_7, reshape_default_8, permute_default_8, reshape_default_9, sum_dim_int_list_18, sum_dim_int_list_19, permute_default_9, reshape_default_10, permute_default_10, reshape_default_11, permute_default_11, reshape_default_12, permute_default_12, reshape_default_13, sum_dim_int_list_24, sum_dim_int_list_25, permute_default_13, reshape_default_14, permute_default_14, reshape_default_15, sum_dim_int_list_28, sum_dim_int_list_29, permute_default_15, reshape_default_16, permute_default_16, reshape_default_17, permute_default_17, reshape_default_18, permute_default_18, reshape_default_19, sum_dim_int_list_34, sum_dim_int_list_35, permute_default_19, reshape_default_20, permute_default_20, reshape_default_21, sum_dim_int_list_38, sum_dim_int_list_39, permute_default_21, reshape_default_22, permute_default_22, reshape_default_23, permute_default_23, reshape_default_24, permute_default_24, reshape_default_25, sum_dim_int_list_44, sum_dim_int_list_45, permute_default_25, reshape_default_26, permute_default_26, reshape_default_27, sum_dim_int_list_48, sum_dim_int_list_49, permute_default_27, reshape_default_28, permute_default_28, reshape_default_29, permute_default_29, reshape_default_30, permute_default_30, reshape_default_31, sum_dim_int_list_54, sum_dim_int_list_55, permute_default_31, reshape_default_32, permute_default_32, reshape_default_33, sum_dim_int_list_58, sum_dim_int_list_59, permute_default_33, reshape_default_34, permute_default_34, reshape_default_35, permute_default_35, reshape_default_36, permute_default_36, reshape_default_37, sum_dim_int_list_64, sum_dim_int_list_65, permute_default_37, reshape_default_38, permute_default_38, reshape_default_39, sum_dim_int_list_68, sum_dim_int_list_69, permute_default_39, reshape_default_40, permute_default_40, reshape_default_41, permute_default_41, reshape_default_42, permute_default_42, reshape_default_43, sum_dim_int_list_74, sum_dim_int_list_75, permute_default_43, reshape_default_44, permute_default_44, reshape_default_45, sum_dim_int_list_78, sum_dim_int_list_79, permute_default_45, reshape_default_46, permute_default_46, reshape_default_47, permute_default_47, reshape_default_48, permute_default_48, reshape_default_49, sum_dim_int_list_84, sum_dim_int_list_85, permute_default_49, reshape_default_50, permute_default_50, reshape_default_51, sum_dim_int_list_88, sum_dim_int_list_89, permute_default_51, reshape_default_52, permute_default_52, reshape_default_53, permute_default_53, reshape_default_54, permute_default_54, reshape_default_55, sum_dim_int_list_94, sum_dim_int_list_95, permute_default_55, reshape_default_56, permute_default_56, reshape_default_57, sum_dim_int_list_98, sum_dim_int_list_99, permute_default_57, reshape_default_58, permute_default_58, reshape_default_59, permute_default_59, reshape_default_60, permute_default_60, reshape_default_61, sum_dim_int_list_104, sum_dim_int_list_105, permute_default_61, reshape_default_62, permute_default_62, reshape_default_63, sum_dim_int_list_108, sum_dim_int_list_109, permute_default_63, reshape_default_64, permute_default_64, reshape_default_65, permute_default_65, reshape_default_66, permute_default_66, reshape_default_67, sum_dim_int_list_114, sum_dim_int_list_115, permute_default_67, reshape_default_68, permute_default_68, reshape_default_69, sum_dim_int_list_118, sum_dim_int_list_119, permute_default_69, reshape_default_70, permute_default_70, reshape_default_71, permute_default_71, reshape_default_72, permute_default_72, reshape_default_73, sum_dim_int_list_124, sum_dim_int_list_125, permute_default_73, reshape_default_74, permute_default_74, reshape_default_75, sum_dim_int_list_128, sum_dim_int_list_129, permute_default_75, reshape_default_76, permute_default_76, reshape_default_77, permute_default_77, reshape_default_78, permute_default_78, reshape_default_79, sum_dim_int_list_134, sum_dim_int_list_135, permute_default_79, reshape_default_80, permute_default_80, reshape_default_81, sum_dim_int_list_138, sum_dim_int_list_139, permute_default_81, reshape_default_82, permute_default_82, reshape_default_83, permute_default_83, reshape_default_84, permute_default_84, reshape_default_85, sum_dim_int_list_144, sum_dim_int_list_145, permute_default_85, reshape_default_86, permute_default_86, reshape_default_87, sum_dim_int_list_148, sum_dim_int_list_149, permute_default_87, reshape_default_88, permute_default_88, reshape_default_89, permute_default_89, reshape_default_90, permute_default_90, reshape_default_91, sum_dim_int_list_154, sum_dim_int_list_155, permute_default_91, reshape_default_92, permute_default_92, reshape_default_93, sum_dim_int_list_158, sum_dim_int_list_159, permute_default_93, reshape_default_94, permute_default_94, reshape_default_95, permute_default_95, reshape_default_96, permute_default_96, reshape_default_97, sum_dim_int_list_164, sum_dim_int_list_165, permute_default_97, reshape_default_98, permute_default_98, reshape_default_99, sum_dim_int_list_168, sum_dim_int_list_169, permute_default_99, reshape_default_100, permute_default_100, reshape_default_101, permute_default_101, reshape_default_102, permute_default_102, reshape_default_103, sum_dim_int_list_174, sum_dim_int_list_175, permute_default_103, reshape_default_104, permute_default_104, reshape_default_105, sum_dim_int_list_178, sum_dim_int_list_179, permute_default_105, reshape_default_106, permute_default_106, reshape_default_107, permute_default_107, reshape_default_108, permute_default_108, reshape_default_109, sum_dim_int_list_184, sum_dim_int_list_185, permute_default_109, reshape_default_110, permute_default_110, reshape_default_111, sum_dim_int_list_188, sum_dim_int_list_189, permute_default_111, reshape_default_112, permute_default_112, reshape_default_113, permute_default_113, reshape_default_114, permute_default_114, reshape_default_115, sum_dim_int_list_194, sum_dim_int_list_195, permute_default_115, reshape_default_116, permute_default_116, reshape_default_117, sum_dim_int_list_198, sum_dim_int_list_199, permute_default_117, reshape_default_118, permute_default_118, reshape_default_119, permute_default_119, reshape_default_120, permute_default_120, reshape_default_121, sum_dim_int_list_204, sum_dim_int_list_205, permute_default_121, reshape_default_122, permute_default_122, reshape_default_123, sum_dim_int_list_208, sum_dim_int_list_209, permute_default_123, reshape_default_124, permute_default_124, reshape_default_125, permute_default_125, reshape_default_126, permute_default_126, reshape_default_127, sum_dim_int_list_214, sum_dim_int_list_215, permute_default_127, reshape_default_128, permute_default_128, reshape_default_129, sum_dim_int_list_218, sum_dim_int_list_219, permute_default_129, reshape_default_130, permute_default_130, reshape_default_131, permute_default_131, reshape_default_132, permute_default_132, reshape_default_133, sum_dim_int_list_224, sum_dim_int_list_225, permute_default_133, reshape_default_134, permute_default_134, reshape_default_135, sum_dim_int_list_228, sum_dim_int_list_229, permute_default_135, reshape_default_136, permute_default_136, reshape_default_137, permute_default_137, reshape_default_138, permute_default_138, reshape_default_139, sum_dim_int_list_234, sum_dim_int_list_235, permute_default_139, reshape_default_140, permute_default_140, reshape_default_141, sum_dim_int_list_238, sum_dim_int_list_239, permute_default_141, reshape_default_142, permute_default_142, reshape_default_143, permute_default_143, reshape_default_145, permute_default_144, reshape_default_147, sum_dim_int_list_246, sum_dim_int_list_247, index_put_default, index_put_default_1, add_tensor_3)


def _default_make_inputs():
    return [
    torch.randn([8192, 29056], dtype=torch.float32, device='cuda'),
    torch.randn([16, 512, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([16, 512, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([16, 512, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([16, 512, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([16, 512, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([16, 512, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([16, 512, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([16, 512, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([16, 512, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([16, 512, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([16, 512, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([16, 512, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([16, 512, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([16, 512, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([16, 512, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([16, 512, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([16, 512, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([16, 512, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([16, 512, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([16, 512, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([16, 512, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([16, 512, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([16, 512, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([16, 512, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([16, 512, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([16, 512, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([16, 512, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([16, 512, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([16, 512, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([16, 512, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([16, 512, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([16, 512, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([16, 512, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([16, 512, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([16, 512, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([16, 512, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([16, 512, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([16, 512, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([16, 512, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([16, 512, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([16, 512, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([16, 512, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([16, 512, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([16, 512, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([16, 512, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([16, 512, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([16, 512, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([16, 512, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([16, 512, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([16, 512, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([16, 512, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([16, 512, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([16, 512, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([16, 512, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([16, 512, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([16, 512, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([16, 512, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([16, 512, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([16, 512, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([16, 512, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([16, 512, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([16, 512, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([16, 512, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([16, 512, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([16, 512, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([16, 512, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([16, 512, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([16, 512, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([16, 512, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([16, 512, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([16, 512, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([16, 512, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([16, 512, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([16, 512, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([16, 512, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([16, 512, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([16, 512, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([16, 512, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([16, 512, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([16, 512, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([16, 512, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([16, 512, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([16, 512, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([16, 512, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([16, 512, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([16, 512, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([16, 512, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([16, 512, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([16, 512, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([16, 512, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([16, 512, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([16, 512, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([16, 512, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([16, 512, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([16, 512, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([16, 512, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([16, 512, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([16, 512, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([1024], dtype=torch.float32, device='cuda'),
    torch.randn([16, 512, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([16, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([16, 512, 1024], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [16, 512, 1024], dtype=torch.bool, device='cuda'),
    torch.randint(0, 512, [1, 512], dtype=torch.int64, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [16, 512], dtype=torch.int64, device='cuda'),
    torch.randint(0, 29056, [16, 512], dtype=torch.int64, device='cuda'),
    torch.randn([29056, 1024], dtype=torch.float32, device='cuda'),
    [29056],  # _shape_param_0
    [1024],  # _shape_param_1
    [1024],  # _shape_param_2
    [4096],  # _shape_param_3
    [1024],  # _shape_param_4
    [1024],  # _shape_param_5
    [1024],  # _shape_param_6
    [1024],  # _shape_param_7
    [1024],  # _shape_param_8
    [4096],  # _shape_param_9
    [1024],  # _shape_param_10
    [1024],  # _shape_param_11
    [1024],  # _shape_param_12
    [1024],  # _shape_param_13
    [1024],  # _shape_param_14
    [4096],  # _shape_param_15
    [1024],  # _shape_param_16
    [1024],  # _shape_param_17
    [1024],  # _shape_param_18
    [1024],  # _shape_param_19
    [1024],  # _shape_param_20
    [4096],  # _shape_param_21
    [1024],  # _shape_param_22
    [1024],  # _shape_param_23
    [1024],  # _shape_param_24
    [1024],  # _shape_param_25
    [1024],  # _shape_param_26
    [4096],  # _shape_param_27
    [1024],  # _shape_param_28
    [1024],  # _shape_param_29
    [1024],  # _shape_param_30
    [1024],  # _shape_param_31
    [1024],  # _shape_param_32
    [4096],  # _shape_param_33
    [1024],  # _shape_param_34
    [1024],  # _shape_param_35
    [1024],  # _shape_param_36
    [1024],  # _shape_param_37
    [1024],  # _shape_param_38
    [4096],  # _shape_param_39
    [1024],  # _shape_param_40
    [1024],  # _shape_param_41
    [1024],  # _shape_param_42
    [1024],  # _shape_param_43
    [1024],  # _shape_param_44
    [4096],  # _shape_param_45
    [1024],  # _shape_param_46
    [1024],  # _shape_param_47
    [1024],  # _shape_param_48
    [1024],  # _shape_param_49
    [1024],  # _shape_param_50
    [4096],  # _shape_param_51
    [1024],  # _shape_param_52
    [1024],  # _shape_param_53
    [1024],  # _shape_param_54
    [1024],  # _shape_param_55
    [1024],  # _shape_param_56
    [4096],  # _shape_param_57
    [1024],  # _shape_param_58
    [1024],  # _shape_param_59
    [1024],  # _shape_param_60
    [1024],  # _shape_param_61
    [1024],  # _shape_param_62
    [4096],  # _shape_param_63
    [1024],  # _shape_param_64
    [1024],  # _shape_param_65
    [1024],  # _shape_param_66
    [1024],  # _shape_param_67
    [1024],  # _shape_param_68
    [4096],  # _shape_param_69
    [1024],  # _shape_param_70
    [1024],  # _shape_param_71
    [1024],  # _shape_param_72
    [1024],  # _shape_param_73
    [1024],  # _shape_param_74
    [4096],  # _shape_param_75
    [1024],  # _shape_param_76
    [1024],  # _shape_param_77
    [1024],  # _shape_param_78
    [1024],  # _shape_param_79
    [1024],  # _shape_param_80
    [4096],  # _shape_param_81
    [1024],  # _shape_param_82
    [1024],  # _shape_param_83
    [1024],  # _shape_param_84
    [1024],  # _shape_param_85
    [1024],  # _shape_param_86
    [4096],  # _shape_param_87
    [1024],  # _shape_param_88
    [1024],  # _shape_param_89
    [1024],  # _shape_param_90
    [1024],  # _shape_param_91
    [1024],  # _shape_param_92
    [4096],  # _shape_param_93
    [1024],  # _shape_param_94
    [1024],  # _shape_param_95
    [1024],  # _shape_param_96
    [1024],  # _shape_param_97
    [1024],  # _shape_param_98
    [4096],  # _shape_param_99
    [1024],  # _shape_param_100
    [1024],  # _shape_param_101
    [1024],  # _shape_param_102
    [1024],  # _shape_param_103
    [1024],  # _shape_param_104
    [4096],  # _shape_param_105
    [1024],  # _shape_param_106
    [1024],  # _shape_param_107
    [1024],  # _shape_param_108
    [1024],  # _shape_param_109
    [1024],  # _shape_param_110
    [4096],  # _shape_param_111
    [1024],  # _shape_param_112
    [1024],  # _shape_param_113
    [1024],  # _shape_param_114
    [1024],  # _shape_param_115
    [1024],  # _shape_param_116
    [4096],  # _shape_param_117
    [1024],  # _shape_param_118
    [1024],  # _shape_param_119
    [1024],  # _shape_param_120
    [1024],  # _shape_param_121
    [1024],  # _shape_param_122
    [4096],  # _shape_param_123
    [1024],  # _shape_param_124
    [1024],  # _shape_param_125
    [1024],  # _shape_param_126
    [1024],  # _shape_param_127
    [1024],  # _shape_param_128
    [4096],  # _shape_param_129
    [1024],  # _shape_param_130
    [1024],  # _shape_param_131
    [1024],  # _shape_param_132
    [1024],  # _shape_param_133
    [1024],  # _shape_param_134
    [4096],  # _shape_param_135
    [1024],  # _shape_param_136
    [1024],  # _shape_param_137
    [1024],  # _shape_param_138
    [1024],  # _shape_param_139
    [1024],  # _shape_param_140
    [4096],  # _shape_param_141
    [1024],  # _shape_param_142
    [1024],  # _shape_param_143
    [16, 512, 1024],  # _shape_param_144
    [1024],  # _shape_param_145
    [16, 512, 1024],  # _shape_param_146
    [1024],  # _shape_param_147
    [16, 512, 1024],  # _shape_param_148
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
