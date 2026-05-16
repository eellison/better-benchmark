"""
Standalone repro captured via capture_hook.
Label: hf_MT5ForConditionalGeneration_training
Pattern hash: 12f67c7a6336
Shape hash: bfb712ae
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, add_152: "f32[8, 128, 512]", rsqrt_41: "f32[8, 128, 1]", mul_338: "f32[8, 128, 512]", _shape_param_0, view_610: "f32[1024, 512]", view_612: "f32[1024, 1024]", view_614: "f32[1024, 1024]", add_148: "f32[8, 128, 512]", rsqrt_40: "f32[8, 128, 1]", add_158: "f32[8, 128, 512]", _shape_param_1, view_617: "f32[1024, 512]", view_631: "f32[1024, 384]", mul_143: "f32[8, 128, 512]", _shape_param_2, view_634: "f32[1024, 384]", _shape_param_3, view_637: "f32[1024, 384]", add_145: "f32[8, 128, 512]", rsqrt_39: "f32[8, 128, 1]", view_638: "f32[8, 128, 512]", _shape_param_4, view_640: "f32[1024, 512]", view_654: "f32[1024, 384]", view_657: "f32[1024, 384]", view_660: "f32[1024, 384]", add_142: "f32[8, 128, 512]", rsqrt_38: "f32[8, 128, 1]", add_166: "f32[8, 128, 512]", _shape_param_5, view_663: "f32[1024, 512]", view_665: "f32[1024, 1024]", view_667: "f32[1024, 1024]", add_138: "f32[8, 128, 512]", rsqrt_37: "f32[8, 128, 1]", add_171: "f32[8, 128, 512]", _shape_param_6, view_670: "f32[1024, 512]", view_684: "f32[1024, 384]", _shape_param_7, view_687: "f32[1024, 384]", _shape_param_8, view_690: "f32[1024, 384]", add_135: "f32[8, 128, 512]", rsqrt_36: "f32[8, 128, 1]", view_691: "f32[8, 128, 512]", _shape_param_9, view_693: "f32[1024, 512]", view_648: "f32[8, 6, 128, 128]", view_701: "f32[8, 6, 128, 128]", view_707: "f32[1024, 384]", view_710: "f32[1024, 384]", view_713: "f32[1024, 384]", add_132: "f32[8, 128, 512]", rsqrt_35: "f32[8, 128, 1]", add_180: "f32[8, 128, 512]", _shape_param_10, view_716: "f32[1024, 512]", view_718: "f32[1024, 1024]", view_720: "f32[1024, 1024]", add_128: "f32[8, 128, 512]", rsqrt_34: "f32[8, 128, 1]", add_185: "f32[8, 128, 512]", _shape_param_11, view_723: "f32[1024, 512]", view_737: "f32[1024, 384]", _shape_param_12, view_740: "f32[1024, 384]", _shape_param_13, view_743: "f32[1024, 384]", add_125: "f32[8, 128, 512]", rsqrt_33: "f32[8, 128, 1]", view_744: "f32[8, 128, 512]", _shape_param_14, view_746: "f32[1024, 512]", view_754: "f32[8, 6, 128, 128]", view_760: "f32[1024, 384]", view_763: "f32[1024, 384]", view_766: "f32[1024, 384]", add_122: "f32[8, 128, 512]", rsqrt_32: "f32[8, 128, 1]", add_194: "f32[8, 128, 512]", _shape_param_15, view_769: "f32[1024, 512]", view_771: "f32[1024, 1024]", view_773: "f32[1024, 1024]", add_118: "f32[8, 128, 512]", rsqrt_31: "f32[8, 128, 1]", add_199: "f32[8, 128, 512]", _shape_param_16, view_776: "f32[1024, 512]", view_790: "f32[1024, 384]", _shape_param_17, view_793: "f32[1024, 384]", _shape_param_18, view_796: "f32[1024, 384]", add_115: "f32[8, 128, 512]", rsqrt_30: "f32[8, 128, 1]", view_797: "f32[8, 128, 512]", _shape_param_19, view_799: "f32[1024, 512]", view_807: "f32[8, 6, 128, 128]", view_813: "f32[1024, 384]", view_816: "f32[1024, 384]", view_819: "f32[1024, 384]", add_112: "f32[8, 128, 512]", rsqrt_29: "f32[8, 128, 1]", add_208: "f32[8, 128, 512]", _shape_param_20, view_822: "f32[1024, 512]", view_824: "f32[1024, 1024]", view_826: "f32[1024, 1024]", add_108: "f32[8, 128, 512]", rsqrt_28: "f32[8, 128, 1]", add_213: "f32[8, 128, 512]", _shape_param_21, view_829: "f32[1024, 512]", view_843: "f32[1024, 384]", _shape_param_22, view_846: "f32[1024, 384]", _shape_param_23, view_849: "f32[1024, 384]", add_105: "f32[8, 128, 512]", rsqrt_27: "f32[8, 128, 1]", view_850: "f32[8, 128, 512]", _shape_param_24, view_852: "f32[1024, 512]", view_860: "f32[8, 6, 128, 128]", view_866: "f32[1024, 384]", view_869: "f32[1024, 384]", view_872: "f32[1024, 384]", add_102: "f32[8, 128, 512]", rsqrt_26: "f32[8, 128, 1]", add_222: "f32[8, 128, 512]", _shape_param_25, view_875: "f32[1024, 512]", view_877: "f32[1024, 1024]", view_879: "f32[1024, 1024]", add_98: "f32[8, 128, 512]", rsqrt_25: "f32[8, 128, 1]", add_227: "f32[8, 128, 512]", _shape_param_26, view_882: "f32[1024, 512]", view_896: "f32[1024, 384]", _shape_param_27, view_899: "f32[1024, 384]", _shape_param_28, view_902: "f32[1024, 384]", add_95: "f32[8, 128, 512]", rsqrt_24: "f32[8, 128, 1]", view_903: "f32[8, 128, 512]", _shape_param_29, view_905: "f32[1024, 512]", view_913: "f32[8, 6, 128, 128]", view_919: "f32[1024, 384]", view_922: "f32[1024, 384]", view_925: "f32[1024, 384]", add_92: "f32[8, 128, 512]", rsqrt_23: "f32[8, 128, 1]", add_236: "f32[8, 128, 512]", _shape_param_30, view_928: "f32[1024, 512]", view_930: "f32[1024, 1024]", view_932: "f32[1024, 1024]", add_88: "f32[8, 128, 512]", rsqrt_22: "f32[8, 128, 1]", add_241: "f32[8, 128, 512]", _shape_param_31, view_935: "f32[1024, 512]", view_949: "f32[1024, 384]", _shape_param_32, view_952: "f32[1024, 384]", _shape_param_33, view_955: "f32[1024, 384]", add_85: "f32[8, 128, 512]", rsqrt_21: "f32[8, 128, 1]", view_956: "f32[8, 128, 512]", _shape_param_34, view_958: "f32[1024, 512]", view_966: "f32[8, 6, 128, 128]", view_972: "f32[1024, 384]", view_975: "f32[1024, 384]", view_978: "f32[1024, 384]", add_82: "f32[8, 128, 512]", rsqrt_20: "f32[8, 128, 1]", add_250: "f32[8, 128, 512]", _shape_param_35, view_981: "f32[1024, 512]", view_983: "f32[1024, 1024]", view_985: "f32[1024, 1024]", add_78: "f32[8, 128, 512]", rsqrt_19: "f32[8, 128, 1]", add_255: "f32[8, 128, 512]", _shape_param_36, view_988: "f32[1024, 512]", view_1002: "f32[1024, 384]", _shape_param_37, view_1005: "f32[1024, 384]", _shape_param_38, view_1008: "f32[1024, 384]", add_74: "f32[8, 128, 512]", rsqrt_18: "f32[8, 128, 1]", view_1009: "f32[8, 128, 512]", _shape_param_39, view_1011: "f32[1024, 512]", view_1019: "f32[8, 6, 128, 128]", add_71: "i64[128, 128]", full_default: "f32[]", view_1025: "f32[1024, 384]", mm_318: "f32[1024, 512]", _shape_param_40, view_1028: "f32[1024, 384]", mm_320: "f32[1024, 512]", _shape_param_41, view_1031: "f32[1024, 384]", mm_322: "f32[1024, 512]", _shape_param_42, primals_78: "f32[512]", gt_35: "b8[8, 128, 512]", embedding: "f32[8, 128, 512]", rsqrt_17: "f32[8, 128, 1]", _shape_param_43, add_261: "f32[8, 128, 512]", _shape_param_44, primals_1: "i64[8, 128]", mm_145: "f32[250112, 512]", add_62: "f32[8, 128, 512]", rsqrt_16: "f32[8, 128, 1]", mul_742: "f32[8, 128, 512]", _shape_param_45, view_1035: "f32[1024, 512]", view_1037: "f32[1024, 1024]", view_1039: "f32[1024, 1024]", add_58: "f32[8, 128, 512]", rsqrt_15: "f32[8, 128, 1]", add_271: "f32[8, 128, 512]", _shape_param_46, view_1042: "f32[1024, 512]", view_1056: "f32[1024, 384]", view_1059: "f32[1024, 384]", view_1062: "f32[1024, 384]", add_55: "f32[8, 128, 512]", rsqrt_14: "f32[8, 128, 1]", add_275: "f32[8, 128, 512]", _shape_param_47, view_1065: "f32[1024, 512]", view_1067: "f32[1024, 1024]", view_1069: "f32[1024, 1024]", add_51: "f32[8, 128, 512]", rsqrt_13: "f32[8, 128, 1]", add_280: "f32[8, 128, 512]", _shape_param_48, view_1072: "f32[1024, 512]", view_1050: "f32[8, 6, 128, 128]", view_1080: "f32[8, 6, 128, 128]", view_1086: "f32[1024, 384]", view_1089: "f32[1024, 384]", view_1092: "f32[1024, 384]", add_48: "f32[8, 128, 512]", rsqrt_12: "f32[8, 128, 1]", add_285: "f32[8, 128, 512]", _shape_param_49, view_1095: "f32[1024, 512]", view_1097: "f32[1024, 1024]", view_1099: "f32[1024, 1024]", add_44: "f32[8, 128, 512]", rsqrt_11: "f32[8, 128, 1]", add_290: "f32[8, 128, 512]", _shape_param_50, view_1102: "f32[1024, 512]", view_1110: "f32[8, 6, 128, 128]", view_1116: "f32[1024, 384]", view_1119: "f32[1024, 384]", view_1122: "f32[1024, 384]", add_41: "f32[8, 128, 512]", rsqrt_10: "f32[8, 128, 1]", add_295: "f32[8, 128, 512]", _shape_param_51, view_1125: "f32[1024, 512]", view_1127: "f32[1024, 1024]", view_1129: "f32[1024, 1024]", add_37: "f32[8, 128, 512]", rsqrt_9: "f32[8, 128, 1]", add_300: "f32[8, 128, 512]", _shape_param_52, view_1132: "f32[1024, 512]", view_1140: "f32[8, 6, 128, 128]", view_1146: "f32[1024, 384]", view_1149: "f32[1024, 384]", view_1152: "f32[1024, 384]", add_34: "f32[8, 128, 512]", rsqrt_8: "f32[8, 128, 1]", add_305: "f32[8, 128, 512]", _shape_param_53, view_1155: "f32[1024, 512]", view_1157: "f32[1024, 1024]", view_1159: "f32[1024, 1024]", add_30: "f32[8, 128, 512]", rsqrt_7: "f32[8, 128, 1]", add_310: "f32[8, 128, 512]", _shape_param_54, view_1162: "f32[1024, 512]", view_1170: "f32[8, 6, 128, 128]", view_1176: "f32[1024, 384]", view_1179: "f32[1024, 384]", view_1182: "f32[1024, 384]", add_27: "f32[8, 128, 512]", rsqrt_6: "f32[8, 128, 1]", add_315: "f32[8, 128, 512]", _shape_param_55, view_1185: "f32[1024, 512]", view_1187: "f32[1024, 1024]", view_1189: "f32[1024, 1024]", add_23: "f32[8, 128, 512]", rsqrt_5: "f32[8, 128, 1]", add_320: "f32[8, 128, 512]", _shape_param_56, view_1192: "f32[1024, 512]", view_1200: "f32[8, 6, 128, 128]", view_1206: "f32[1024, 384]", view_1209: "f32[1024, 384]", view_1212: "f32[1024, 384]", add_20: "f32[8, 128, 512]", rsqrt_4: "f32[8, 128, 1]", add_325: "f32[8, 128, 512]", _shape_param_57, view_1215: "f32[1024, 512]", view_1217: "f32[1024, 1024]", view_1219: "f32[1024, 1024]", add_16: "f32[8, 128, 512]", rsqrt_3: "f32[8, 128, 1]", add_330: "f32[8, 128, 512]", _shape_param_58, view_1222: "f32[1024, 512]", view_1230: "f32[8, 6, 128, 128]", view_1236: "f32[1024, 384]", view_1239: "f32[1024, 384]", view_1242: "f32[1024, 384]", add_13: "f32[8, 128, 512]", rsqrt_2: "f32[8, 128, 1]", add_335: "f32[8, 128, 512]", _shape_param_59, view_1245: "f32[1024, 512]", view_1247: "f32[1024, 1024]", view_1249: "f32[1024, 1024]", add_9: "f32[8, 128, 512]", rsqrt_1: "f32[8, 128, 1]", add_340: "f32[8, 128, 512]", _shape_param_60, view_1252: "f32[1024, 512]", view_1260: "f32[8, 6, 128, 128]", add_6: "i64[128, 128]", view_1266: "f32[1024, 384]", mm_430: "f32[1024, 512]", _shape_param_61, view_1269: "f32[1024, 384]", mm_432: "f32[1024, 512]", _shape_param_62, view_1272: "f32[1024, 384]", mm_434: "f32[1024, 512]", _shape_param_63, primals_3: "f32[512]", gt: "b8[8, 128, 512]", rsqrt: "f32[8, 128, 1]", _shape_param_64, add_342: "f32[8, 128, 512]", _shape_param_65):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor: "f32[8, 128, 512]" = torch.ops.aten.mul.Tensor(add_152, rsqrt_41);  add_152 = rsqrt_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_tensor_1: "f32[8, 128, 512]" = torch.ops.aten.mul.Tensor(mul_338, mul_tensor);  mul_338 = mul_tensor = None
        sum_dim_int_list: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_1, [0, 1], True);  mul_tensor_1 = None
        reshape_default: "f32[512]" = torch.ops.aten.reshape.default(sum_dim_int_list, _shape_param_0);  sum_dim_int_list = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:121 in forward, code: hidden_states = self.wo(hidden_states)
        permute_default: "f32[512, 1024]" = torch.ops.aten.permute.default(view_610, [1, 0]);  view_610 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:107 in forward, code: hidden_linear = self.wi_1(hidden_states)
        permute_default_1: "f32[1024, 1024]" = torch.ops.aten.permute.default(view_612, [1, 0]);  view_612 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:106 in forward, code: hidden_gelu = self.act(self.wi_0(hidden_states))
        permute_default_2: "f32[1024, 1024]" = torch.ops.aten.permute.default(view_614, [1, 0]);  view_614 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_2: "f32[8, 128, 512]" = torch.ops.aten.mul.Tensor(add_148, rsqrt_40);  add_148 = rsqrt_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_tensor_3: "f32[8, 128, 512]" = torch.ops.aten.mul.Tensor(add_158, mul_tensor_2);  add_158 = mul_tensor_2 = None
        sum_dim_int_list_1: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_3, [0, 1], True);  mul_tensor_3 = None
        reshape_default_1: "f32[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_1, _shape_param_1);  sum_dim_int_list_1 = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:330 in forward, code: attn_output = self.o(attn_output)
        permute_default_3: "f32[512, 1024]" = torch.ops.aten.permute.default(view_617, [1, 0]);  view_617 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:291 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_default_4: "f32[384, 1024]" = torch.ops.aten.permute.default(view_631, [1, 0]);  view_631 = None
        reshape_default_2: "f32[1024, 512]" = torch.ops.aten.reshape.default(mul_143, _shape_param_2);  _shape_param_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:290 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_default_5: "f32[384, 1024]" = torch.ops.aten.permute.default(view_634, [1, 0]);  view_634 = None
        reshape_default_3: "f32[1024, 512]" = torch.ops.aten.reshape.default(mul_143, _shape_param_3);  _shape_param_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:269 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_6: "f32[384, 1024]" = torch.ops.aten.permute.default(view_637, [1, 0]);  view_637 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_4: "f32[8, 128, 512]" = torch.ops.aten.mul.Tensor(add_145, rsqrt_39);  add_145 = rsqrt_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_tensor_5: "f32[8, 128, 512]" = torch.ops.aten.mul.Tensor(view_638, mul_tensor_4);  view_638 = mul_tensor_4 = None
        sum_dim_int_list_2: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_5, [0, 1], True);  mul_tensor_5 = None
        reshape_default_4: "f32[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_2, _shape_param_4);  sum_dim_int_list_2 = _shape_param_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:330 in forward, code: attn_output = self.o(attn_output)
        permute_default_7: "f32[512, 1024]" = torch.ops.aten.permute.default(view_640, [1, 0]);  view_640 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:291 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_default_8: "f32[384, 1024]" = torch.ops.aten.permute.default(view_654, [1, 0]);  view_654 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:290 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_default_9: "f32[384, 1024]" = torch.ops.aten.permute.default(view_657, [1, 0]);  view_657 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:269 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_10: "f32[384, 1024]" = torch.ops.aten.permute.default(view_660, [1, 0]);  view_660 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_6: "f32[8, 128, 512]" = torch.ops.aten.mul.Tensor(add_142, rsqrt_38);  add_142 = rsqrt_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_tensor_7: "f32[8, 128, 512]" = torch.ops.aten.mul.Tensor(add_166, mul_tensor_6);  add_166 = mul_tensor_6 = None
        sum_dim_int_list_3: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_7, [0, 1], True);  mul_tensor_7 = None
        reshape_default_5: "f32[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_3, _shape_param_5);  sum_dim_int_list_3 = _shape_param_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:121 in forward, code: hidden_states = self.wo(hidden_states)
        permute_default_11: "f32[512, 1024]" = torch.ops.aten.permute.default(view_663, [1, 0]);  view_663 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:107 in forward, code: hidden_linear = self.wi_1(hidden_states)
        permute_default_12: "f32[1024, 1024]" = torch.ops.aten.permute.default(view_665, [1, 0]);  view_665 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:106 in forward, code: hidden_gelu = self.act(self.wi_0(hidden_states))
        permute_default_13: "f32[1024, 1024]" = torch.ops.aten.permute.default(view_667, [1, 0]);  view_667 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_8: "f32[8, 128, 512]" = torch.ops.aten.mul.Tensor(add_138, rsqrt_37);  add_138 = rsqrt_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_tensor_9: "f32[8, 128, 512]" = torch.ops.aten.mul.Tensor(add_171, mul_tensor_8);  add_171 = mul_tensor_8 = None
        sum_dim_int_list_4: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_9, [0, 1], True);  mul_tensor_9 = None
        reshape_default_6: "f32[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_4, _shape_param_6);  sum_dim_int_list_4 = _shape_param_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:330 in forward, code: attn_output = self.o(attn_output)
        permute_default_14: "f32[512, 1024]" = torch.ops.aten.permute.default(view_670, [1, 0]);  view_670 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:291 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_default_15: "f32[384, 1024]" = torch.ops.aten.permute.default(view_684, [1, 0]);  view_684 = None
        reshape_default_7: "f32[1024, 512]" = torch.ops.aten.reshape.default(mul_143, _shape_param_7);  _shape_param_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:290 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_default_16: "f32[384, 1024]" = torch.ops.aten.permute.default(view_687, [1, 0]);  view_687 = None
        reshape_default_8: "f32[1024, 512]" = torch.ops.aten.reshape.default(mul_143, _shape_param_8);  _shape_param_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:269 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_17: "f32[384, 1024]" = torch.ops.aten.permute.default(view_690, [1, 0]);  view_690 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_10: "f32[8, 128, 512]" = torch.ops.aten.mul.Tensor(add_135, rsqrt_36);  add_135 = rsqrt_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_tensor_11: "f32[8, 128, 512]" = torch.ops.aten.mul.Tensor(view_691, mul_tensor_10);  view_691 = mul_tensor_10 = None
        sum_dim_int_list_5: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_11, [0, 1], True);  mul_tensor_11 = None
        reshape_default_9: "f32[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_5, _shape_param_9);  sum_dim_int_list_5 = _shape_param_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:330 in forward, code: attn_output = self.o(attn_output)
        permute_default_18: "f32[512, 1024]" = torch.ops.aten.permute.default(view_693, [1, 0]);  view_693 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:320 in forward, code: scores += position_bias_masked
        add_tensor: "f32[8, 6, 128, 128]" = torch.ops.aten.add.Tensor(view_648, view_701);  view_648 = view_701 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:291 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_default_19: "f32[384, 1024]" = torch.ops.aten.permute.default(view_707, [1, 0]);  view_707 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:290 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_default_20: "f32[384, 1024]" = torch.ops.aten.permute.default(view_710, [1, 0]);  view_710 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:269 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_21: "f32[384, 1024]" = torch.ops.aten.permute.default(view_713, [1, 0]);  view_713 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_12: "f32[8, 128, 512]" = torch.ops.aten.mul.Tensor(add_132, rsqrt_35);  add_132 = rsqrt_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_tensor_13: "f32[8, 128, 512]" = torch.ops.aten.mul.Tensor(add_180, mul_tensor_12);  add_180 = mul_tensor_12 = None
        sum_dim_int_list_6: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_13, [0, 1], True);  mul_tensor_13 = None
        reshape_default_10: "f32[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_6, _shape_param_10);  sum_dim_int_list_6 = _shape_param_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:121 in forward, code: hidden_states = self.wo(hidden_states)
        permute_default_22: "f32[512, 1024]" = torch.ops.aten.permute.default(view_716, [1, 0]);  view_716 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:107 in forward, code: hidden_linear = self.wi_1(hidden_states)
        permute_default_23: "f32[1024, 1024]" = torch.ops.aten.permute.default(view_718, [1, 0]);  view_718 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:106 in forward, code: hidden_gelu = self.act(self.wi_0(hidden_states))
        permute_default_24: "f32[1024, 1024]" = torch.ops.aten.permute.default(view_720, [1, 0]);  view_720 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_14: "f32[8, 128, 512]" = torch.ops.aten.mul.Tensor(add_128, rsqrt_34);  add_128 = rsqrt_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_tensor_15: "f32[8, 128, 512]" = torch.ops.aten.mul.Tensor(add_185, mul_tensor_14);  add_185 = mul_tensor_14 = None
        sum_dim_int_list_7: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_15, [0, 1], True);  mul_tensor_15 = None
        reshape_default_11: "f32[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_7, _shape_param_11);  sum_dim_int_list_7 = _shape_param_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:330 in forward, code: attn_output = self.o(attn_output)
        permute_default_25: "f32[512, 1024]" = torch.ops.aten.permute.default(view_723, [1, 0]);  view_723 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:291 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_default_26: "f32[384, 1024]" = torch.ops.aten.permute.default(view_737, [1, 0]);  view_737 = None
        reshape_default_12: "f32[1024, 512]" = torch.ops.aten.reshape.default(mul_143, _shape_param_12);  _shape_param_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:290 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_default_27: "f32[384, 1024]" = torch.ops.aten.permute.default(view_740, [1, 0]);  view_740 = None
        reshape_default_13: "f32[1024, 512]" = torch.ops.aten.reshape.default(mul_143, _shape_param_13);  _shape_param_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:269 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_28: "f32[384, 1024]" = torch.ops.aten.permute.default(view_743, [1, 0]);  view_743 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_16: "f32[8, 128, 512]" = torch.ops.aten.mul.Tensor(add_125, rsqrt_33);  add_125 = rsqrt_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_tensor_17: "f32[8, 128, 512]" = torch.ops.aten.mul.Tensor(view_744, mul_tensor_16);  view_744 = mul_tensor_16 = None
        sum_dim_int_list_8: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_17, [0, 1], True);  mul_tensor_17 = None
        reshape_default_14: "f32[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_8, _shape_param_14);  sum_dim_int_list_8 = _shape_param_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:330 in forward, code: attn_output = self.o(attn_output)
        permute_default_29: "f32[512, 1024]" = torch.ops.aten.permute.default(view_746, [1, 0]);  view_746 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:320 in forward, code: scores += position_bias_masked
        add_tensor_1: "f32[8, 6, 128, 128]" = torch.ops.aten.add.Tensor(add_tensor, view_754);  add_tensor = view_754 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:291 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_default_30: "f32[384, 1024]" = torch.ops.aten.permute.default(view_760, [1, 0]);  view_760 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:290 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_default_31: "f32[384, 1024]" = torch.ops.aten.permute.default(view_763, [1, 0]);  view_763 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:269 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_32: "f32[384, 1024]" = torch.ops.aten.permute.default(view_766, [1, 0]);  view_766 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_18: "f32[8, 128, 512]" = torch.ops.aten.mul.Tensor(add_122, rsqrt_32);  add_122 = rsqrt_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_tensor_19: "f32[8, 128, 512]" = torch.ops.aten.mul.Tensor(add_194, mul_tensor_18);  add_194 = mul_tensor_18 = None
        sum_dim_int_list_9: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_19, [0, 1], True);  mul_tensor_19 = None
        reshape_default_15: "f32[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_9, _shape_param_15);  sum_dim_int_list_9 = _shape_param_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:121 in forward, code: hidden_states = self.wo(hidden_states)
        permute_default_33: "f32[512, 1024]" = torch.ops.aten.permute.default(view_769, [1, 0]);  view_769 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:107 in forward, code: hidden_linear = self.wi_1(hidden_states)
        permute_default_34: "f32[1024, 1024]" = torch.ops.aten.permute.default(view_771, [1, 0]);  view_771 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:106 in forward, code: hidden_gelu = self.act(self.wi_0(hidden_states))
        permute_default_35: "f32[1024, 1024]" = torch.ops.aten.permute.default(view_773, [1, 0]);  view_773 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_20: "f32[8, 128, 512]" = torch.ops.aten.mul.Tensor(add_118, rsqrt_31);  add_118 = rsqrt_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_tensor_21: "f32[8, 128, 512]" = torch.ops.aten.mul.Tensor(add_199, mul_tensor_20);  add_199 = mul_tensor_20 = None
        sum_dim_int_list_10: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_21, [0, 1], True);  mul_tensor_21 = None
        reshape_default_16: "f32[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_10, _shape_param_16);  sum_dim_int_list_10 = _shape_param_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:330 in forward, code: attn_output = self.o(attn_output)
        permute_default_36: "f32[512, 1024]" = torch.ops.aten.permute.default(view_776, [1, 0]);  view_776 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:291 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_default_37: "f32[384, 1024]" = torch.ops.aten.permute.default(view_790, [1, 0]);  view_790 = None
        reshape_default_17: "f32[1024, 512]" = torch.ops.aten.reshape.default(mul_143, _shape_param_17);  _shape_param_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:290 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_default_38: "f32[384, 1024]" = torch.ops.aten.permute.default(view_793, [1, 0]);  view_793 = None
        reshape_default_18: "f32[1024, 512]" = torch.ops.aten.reshape.default(mul_143, _shape_param_18);  _shape_param_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:269 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_39: "f32[384, 1024]" = torch.ops.aten.permute.default(view_796, [1, 0]);  view_796 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_22: "f32[8, 128, 512]" = torch.ops.aten.mul.Tensor(add_115, rsqrt_30);  add_115 = rsqrt_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_tensor_23: "f32[8, 128, 512]" = torch.ops.aten.mul.Tensor(view_797, mul_tensor_22);  view_797 = mul_tensor_22 = None
        sum_dim_int_list_11: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_23, [0, 1], True);  mul_tensor_23 = None
        reshape_default_19: "f32[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_11, _shape_param_19);  sum_dim_int_list_11 = _shape_param_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:330 in forward, code: attn_output = self.o(attn_output)
        permute_default_40: "f32[512, 1024]" = torch.ops.aten.permute.default(view_799, [1, 0]);  view_799 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:320 in forward, code: scores += position_bias_masked
        add_tensor_2: "f32[8, 6, 128, 128]" = torch.ops.aten.add.Tensor(add_tensor_1, view_807);  add_tensor_1 = view_807 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:291 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_default_41: "f32[384, 1024]" = torch.ops.aten.permute.default(view_813, [1, 0]);  view_813 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:290 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_default_42: "f32[384, 1024]" = torch.ops.aten.permute.default(view_816, [1, 0]);  view_816 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:269 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_43: "f32[384, 1024]" = torch.ops.aten.permute.default(view_819, [1, 0]);  view_819 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_24: "f32[8, 128, 512]" = torch.ops.aten.mul.Tensor(add_112, rsqrt_29);  add_112 = rsqrt_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_tensor_25: "f32[8, 128, 512]" = torch.ops.aten.mul.Tensor(add_208, mul_tensor_24);  add_208 = mul_tensor_24 = None
        sum_dim_int_list_12: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_25, [0, 1], True);  mul_tensor_25 = None
        reshape_default_20: "f32[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_12, _shape_param_20);  sum_dim_int_list_12 = _shape_param_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:121 in forward, code: hidden_states = self.wo(hidden_states)
        permute_default_44: "f32[512, 1024]" = torch.ops.aten.permute.default(view_822, [1, 0]);  view_822 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:107 in forward, code: hidden_linear = self.wi_1(hidden_states)
        permute_default_45: "f32[1024, 1024]" = torch.ops.aten.permute.default(view_824, [1, 0]);  view_824 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:106 in forward, code: hidden_gelu = self.act(self.wi_0(hidden_states))
        permute_default_46: "f32[1024, 1024]" = torch.ops.aten.permute.default(view_826, [1, 0]);  view_826 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_26: "f32[8, 128, 512]" = torch.ops.aten.mul.Tensor(add_108, rsqrt_28);  add_108 = rsqrt_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_tensor_27: "f32[8, 128, 512]" = torch.ops.aten.mul.Tensor(add_213, mul_tensor_26);  add_213 = mul_tensor_26 = None
        sum_dim_int_list_13: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_27, [0, 1], True);  mul_tensor_27 = None
        reshape_default_21: "f32[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_13, _shape_param_21);  sum_dim_int_list_13 = _shape_param_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:330 in forward, code: attn_output = self.o(attn_output)
        permute_default_47: "f32[512, 1024]" = torch.ops.aten.permute.default(view_829, [1, 0]);  view_829 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:291 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_default_48: "f32[384, 1024]" = torch.ops.aten.permute.default(view_843, [1, 0]);  view_843 = None
        reshape_default_22: "f32[1024, 512]" = torch.ops.aten.reshape.default(mul_143, _shape_param_22);  _shape_param_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:290 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_default_49: "f32[384, 1024]" = torch.ops.aten.permute.default(view_846, [1, 0]);  view_846 = None
        reshape_default_23: "f32[1024, 512]" = torch.ops.aten.reshape.default(mul_143, _shape_param_23);  _shape_param_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:269 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_50: "f32[384, 1024]" = torch.ops.aten.permute.default(view_849, [1, 0]);  view_849 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_28: "f32[8, 128, 512]" = torch.ops.aten.mul.Tensor(add_105, rsqrt_27);  add_105 = rsqrt_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_tensor_29: "f32[8, 128, 512]" = torch.ops.aten.mul.Tensor(view_850, mul_tensor_28);  view_850 = mul_tensor_28 = None
        sum_dim_int_list_14: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_29, [0, 1], True);  mul_tensor_29 = None
        reshape_default_24: "f32[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_14, _shape_param_24);  sum_dim_int_list_14 = _shape_param_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:330 in forward, code: attn_output = self.o(attn_output)
        permute_default_51: "f32[512, 1024]" = torch.ops.aten.permute.default(view_852, [1, 0]);  view_852 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:320 in forward, code: scores += position_bias_masked
        add_tensor_3: "f32[8, 6, 128, 128]" = torch.ops.aten.add.Tensor(add_tensor_2, view_860);  add_tensor_2 = view_860 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:291 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_default_52: "f32[384, 1024]" = torch.ops.aten.permute.default(view_866, [1, 0]);  view_866 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:290 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_default_53: "f32[384, 1024]" = torch.ops.aten.permute.default(view_869, [1, 0]);  view_869 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:269 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_54: "f32[384, 1024]" = torch.ops.aten.permute.default(view_872, [1, 0]);  view_872 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_30: "f32[8, 128, 512]" = torch.ops.aten.mul.Tensor(add_102, rsqrt_26);  add_102 = rsqrt_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_tensor_31: "f32[8, 128, 512]" = torch.ops.aten.mul.Tensor(add_222, mul_tensor_30);  add_222 = mul_tensor_30 = None
        sum_dim_int_list_15: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_31, [0, 1], True);  mul_tensor_31 = None
        reshape_default_25: "f32[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_15, _shape_param_25);  sum_dim_int_list_15 = _shape_param_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:121 in forward, code: hidden_states = self.wo(hidden_states)
        permute_default_55: "f32[512, 1024]" = torch.ops.aten.permute.default(view_875, [1, 0]);  view_875 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:107 in forward, code: hidden_linear = self.wi_1(hidden_states)
        permute_default_56: "f32[1024, 1024]" = torch.ops.aten.permute.default(view_877, [1, 0]);  view_877 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:106 in forward, code: hidden_gelu = self.act(self.wi_0(hidden_states))
        permute_default_57: "f32[1024, 1024]" = torch.ops.aten.permute.default(view_879, [1, 0]);  view_879 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_32: "f32[8, 128, 512]" = torch.ops.aten.mul.Tensor(add_98, rsqrt_25);  add_98 = rsqrt_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_tensor_33: "f32[8, 128, 512]" = torch.ops.aten.mul.Tensor(add_227, mul_tensor_32);  add_227 = mul_tensor_32 = None
        sum_dim_int_list_16: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_33, [0, 1], True);  mul_tensor_33 = None
        reshape_default_26: "f32[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_16, _shape_param_26);  sum_dim_int_list_16 = _shape_param_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:330 in forward, code: attn_output = self.o(attn_output)
        permute_default_58: "f32[512, 1024]" = torch.ops.aten.permute.default(view_882, [1, 0]);  view_882 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:291 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_default_59: "f32[384, 1024]" = torch.ops.aten.permute.default(view_896, [1, 0]);  view_896 = None
        reshape_default_27: "f32[1024, 512]" = torch.ops.aten.reshape.default(mul_143, _shape_param_27);  _shape_param_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:290 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_default_60: "f32[384, 1024]" = torch.ops.aten.permute.default(view_899, [1, 0]);  view_899 = None
        reshape_default_28: "f32[1024, 512]" = torch.ops.aten.reshape.default(mul_143, _shape_param_28);  _shape_param_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:269 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_61: "f32[384, 1024]" = torch.ops.aten.permute.default(view_902, [1, 0]);  view_902 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_34: "f32[8, 128, 512]" = torch.ops.aten.mul.Tensor(add_95, rsqrt_24);  add_95 = rsqrt_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_tensor_35: "f32[8, 128, 512]" = torch.ops.aten.mul.Tensor(view_903, mul_tensor_34);  view_903 = mul_tensor_34 = None
        sum_dim_int_list_17: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_35, [0, 1], True);  mul_tensor_35 = None
        reshape_default_29: "f32[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_17, _shape_param_29);  sum_dim_int_list_17 = _shape_param_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:330 in forward, code: attn_output = self.o(attn_output)
        permute_default_62: "f32[512, 1024]" = torch.ops.aten.permute.default(view_905, [1, 0]);  view_905 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:320 in forward, code: scores += position_bias_masked
        add_tensor_4: "f32[8, 6, 128, 128]" = torch.ops.aten.add.Tensor(add_tensor_3, view_913);  add_tensor_3 = view_913 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:291 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_default_63: "f32[384, 1024]" = torch.ops.aten.permute.default(view_919, [1, 0]);  view_919 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:290 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_default_64: "f32[384, 1024]" = torch.ops.aten.permute.default(view_922, [1, 0]);  view_922 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:269 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_65: "f32[384, 1024]" = torch.ops.aten.permute.default(view_925, [1, 0]);  view_925 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_36: "f32[8, 128, 512]" = torch.ops.aten.mul.Tensor(add_92, rsqrt_23);  add_92 = rsqrt_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_tensor_37: "f32[8, 128, 512]" = torch.ops.aten.mul.Tensor(add_236, mul_tensor_36);  add_236 = mul_tensor_36 = None
        sum_dim_int_list_18: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_37, [0, 1], True);  mul_tensor_37 = None
        reshape_default_30: "f32[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_18, _shape_param_30);  sum_dim_int_list_18 = _shape_param_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:121 in forward, code: hidden_states = self.wo(hidden_states)
        permute_default_66: "f32[512, 1024]" = torch.ops.aten.permute.default(view_928, [1, 0]);  view_928 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:107 in forward, code: hidden_linear = self.wi_1(hidden_states)
        permute_default_67: "f32[1024, 1024]" = torch.ops.aten.permute.default(view_930, [1, 0]);  view_930 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:106 in forward, code: hidden_gelu = self.act(self.wi_0(hidden_states))
        permute_default_68: "f32[1024, 1024]" = torch.ops.aten.permute.default(view_932, [1, 0]);  view_932 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_38: "f32[8, 128, 512]" = torch.ops.aten.mul.Tensor(add_88, rsqrt_22);  add_88 = rsqrt_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_tensor_39: "f32[8, 128, 512]" = torch.ops.aten.mul.Tensor(add_241, mul_tensor_38);  add_241 = mul_tensor_38 = None
        sum_dim_int_list_19: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_39, [0, 1], True);  mul_tensor_39 = None
        reshape_default_31: "f32[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_19, _shape_param_31);  sum_dim_int_list_19 = _shape_param_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:330 in forward, code: attn_output = self.o(attn_output)
        permute_default_69: "f32[512, 1024]" = torch.ops.aten.permute.default(view_935, [1, 0]);  view_935 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:291 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_default_70: "f32[384, 1024]" = torch.ops.aten.permute.default(view_949, [1, 0]);  view_949 = None
        reshape_default_32: "f32[1024, 512]" = torch.ops.aten.reshape.default(mul_143, _shape_param_32);  _shape_param_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:290 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_default_71: "f32[384, 1024]" = torch.ops.aten.permute.default(view_952, [1, 0]);  view_952 = None
        reshape_default_33: "f32[1024, 512]" = torch.ops.aten.reshape.default(mul_143, _shape_param_33);  _shape_param_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:269 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_72: "f32[384, 1024]" = torch.ops.aten.permute.default(view_955, [1, 0]);  view_955 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_40: "f32[8, 128, 512]" = torch.ops.aten.mul.Tensor(add_85, rsqrt_21);  add_85 = rsqrt_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_tensor_41: "f32[8, 128, 512]" = torch.ops.aten.mul.Tensor(view_956, mul_tensor_40);  view_956 = mul_tensor_40 = None
        sum_dim_int_list_20: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_41, [0, 1], True);  mul_tensor_41 = None
        reshape_default_34: "f32[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_20, _shape_param_34);  sum_dim_int_list_20 = _shape_param_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:330 in forward, code: attn_output = self.o(attn_output)
        permute_default_73: "f32[512, 1024]" = torch.ops.aten.permute.default(view_958, [1, 0]);  view_958 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:320 in forward, code: scores += position_bias_masked
        add_tensor_5: "f32[8, 6, 128, 128]" = torch.ops.aten.add.Tensor(add_tensor_4, view_966);  add_tensor_4 = view_966 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:291 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_default_74: "f32[384, 1024]" = torch.ops.aten.permute.default(view_972, [1, 0]);  view_972 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:290 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_default_75: "f32[384, 1024]" = torch.ops.aten.permute.default(view_975, [1, 0]);  view_975 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:269 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_76: "f32[384, 1024]" = torch.ops.aten.permute.default(view_978, [1, 0]);  view_978 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_42: "f32[8, 128, 512]" = torch.ops.aten.mul.Tensor(add_82, rsqrt_20);  add_82 = rsqrt_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_tensor_43: "f32[8, 128, 512]" = torch.ops.aten.mul.Tensor(add_250, mul_tensor_42);  add_250 = mul_tensor_42 = None
        sum_dim_int_list_21: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_43, [0, 1], True);  mul_tensor_43 = None
        reshape_default_35: "f32[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_21, _shape_param_35);  sum_dim_int_list_21 = _shape_param_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:121 in forward, code: hidden_states = self.wo(hidden_states)
        permute_default_77: "f32[512, 1024]" = torch.ops.aten.permute.default(view_981, [1, 0]);  view_981 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:107 in forward, code: hidden_linear = self.wi_1(hidden_states)
        permute_default_78: "f32[1024, 1024]" = torch.ops.aten.permute.default(view_983, [1, 0]);  view_983 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:106 in forward, code: hidden_gelu = self.act(self.wi_0(hidden_states))
        permute_default_79: "f32[1024, 1024]" = torch.ops.aten.permute.default(view_985, [1, 0]);  view_985 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_44: "f32[8, 128, 512]" = torch.ops.aten.mul.Tensor(add_78, rsqrt_19);  add_78 = rsqrt_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_tensor_45: "f32[8, 128, 512]" = torch.ops.aten.mul.Tensor(add_255, mul_tensor_44);  add_255 = mul_tensor_44 = None
        sum_dim_int_list_22: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_45, [0, 1], True);  mul_tensor_45 = None
        reshape_default_36: "f32[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_22, _shape_param_36);  sum_dim_int_list_22 = _shape_param_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:330 in forward, code: attn_output = self.o(attn_output)
        permute_default_80: "f32[512, 1024]" = torch.ops.aten.permute.default(view_988, [1, 0]);  view_988 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:291 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_default_81: "f32[384, 1024]" = torch.ops.aten.permute.default(view_1002, [1, 0]);  view_1002 = None
        reshape_default_37: "f32[1024, 512]" = torch.ops.aten.reshape.default(mul_143, _shape_param_37);  _shape_param_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:290 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_default_82: "f32[384, 1024]" = torch.ops.aten.permute.default(view_1005, [1, 0]);  view_1005 = None
        reshape_default_38: "f32[1024, 512]" = torch.ops.aten.reshape.default(mul_143, _shape_param_38);  mul_143 = _shape_param_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:269 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_83: "f32[384, 1024]" = torch.ops.aten.permute.default(view_1008, [1, 0]);  view_1008 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_46: "f32[8, 128, 512]" = torch.ops.aten.mul.Tensor(add_74, rsqrt_18);  add_74 = rsqrt_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_tensor_47: "f32[8, 128, 512]" = torch.ops.aten.mul.Tensor(view_1009, mul_tensor_46);  view_1009 = mul_tensor_46 = None
        sum_dim_int_list_23: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_47, [0, 1], True);  mul_tensor_47 = None
        reshape_default_39: "f32[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_23, _shape_param_39);  sum_dim_int_list_23 = _shape_param_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:330 in forward, code: attn_output = self.o(attn_output)
        permute_default_84: "f32[512, 1024]" = torch.ops.aten.permute.default(view_1011, [1, 0]);  view_1011 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:320 in forward, code: scores += position_bias_masked
        add_tensor_6: "f32[8, 6, 128, 128]" = torch.ops.aten.add.Tensor(add_tensor_5, view_1019);  add_tensor_5 = view_1019 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:317 in forward, code: position_bias = position_bias + causal_mask
        sum_dim_int_list_24: "f32[1, 6, 128, 128]" = torch.ops.aten.sum.dim_IntList(add_tensor_6, [0], True);  add_tensor_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:242 in compute_bias, code: values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
        squeeze_dim: "f32[6, 128, 128]" = torch.ops.aten.squeeze.dim(sum_dim_int_list_24, 0);  sum_dim_int_list_24 = None
        permute_default_85: "f32[128, 128, 6]" = torch.ops.aten.permute.default(squeeze_dim, [1, 2, 0]);  squeeze_dim = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:241 in compute_bias, code: values = self.relative_attention_bias(relative_position_bucket)  # shape (query_length, key_length, num_heads)
        eq_scalar: "b8[128, 128]" = torch.ops.aten.eq.Scalar(add_71, -1)
        unsqueeze_default: "b8[128, 128, 1]" = torch.ops.aten.unsqueeze.default(eq_scalar, -1);  eq_scalar = None
        where_self: "f32[128, 128, 6]" = torch.ops.aten.where.self(unsqueeze_default, full_default, permute_default_85);  unsqueeze_default = permute_default_85 = None
        clone_default: "f32[128, 128, 6]" = torch.ops.aten.clone.default(where_self, memory_format = torch.contiguous_format);  where_self = None
        full_default_1: "f32[32, 6]" = torch.ops.aten.full.default([32, 6], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default: "f32[32, 6]" = torch.ops.aten.index_put.default(full_default_1, [add_71], clone_default, True);  add_71 = clone_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:291 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_default_86: "f32[384, 1024]" = torch.ops.aten.permute.default(view_1025, [1, 0]);  view_1025 = None
        reshape_default_40: "f32[8, 128, 512]" = torch.ops.aten.reshape.default(mm_318, _shape_param_40);  mm_318 = _shape_param_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:290 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_default_87: "f32[384, 1024]" = torch.ops.aten.permute.default(view_1028, [1, 0]);  view_1028 = None
        reshape_default_41: "f32[8, 128, 512]" = torch.ops.aten.reshape.default(mm_320, _shape_param_41);  mm_320 = _shape_param_41 = None
        add_tensor_7: "f32[8, 128, 512]" = torch.ops.aten.add.Tensor(reshape_default_40, reshape_default_41);  reshape_default_40 = reshape_default_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:269 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_88: "f32[384, 1024]" = torch.ops.aten.permute.default(view_1031, [1, 0]);  view_1031 = None
        reshape_default_42: "f32[8, 128, 512]" = torch.ops.aten.reshape.default(mm_322, _shape_param_42);  mm_322 = _shape_param_42 = None
        add_tensor_8: "f32[8, 128, 512]" = torch.ops.aten.add.Tensor(add_tensor_7, reshape_default_42);  add_tensor_7 = reshape_default_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_tensor_48: "f32[8, 128, 512]" = torch.ops.aten.mul.Tensor(add_tensor_8, primals_78);  primals_78 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:732 in forward, code: hidden_states = self.dropout(inputs_embeds)
        mul_tensor_49: "f32[8, 128, 512]" = torch.ops.aten.mul.Tensor(gt_35, embedding)
        mul_tensor_50: "f32[8, 128, 512]" = torch.ops.aten.mul.Tensor(mul_tensor_49, 1.1111111111111112);  mul_tensor_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_51: "f32[8, 128, 512]" = torch.ops.aten.mul.Tensor(mul_tensor_50, rsqrt_17)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_tensor_52: "f32[8, 128, 512]" = torch.ops.aten.mul.Tensor(add_tensor_8, mul_tensor_51);  add_tensor_8 = mul_tensor_51 = None
        sum_dim_int_list_25: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_52, [0, 1], True);  mul_tensor_52 = None
        reshape_default_43: "f32[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_25, _shape_param_43);  sum_dim_int_list_25 = _shape_param_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_53: "f32[8, 128, 512]" = torch.ops.aten.mul.Tensor(mul_tensor_48, mul_tensor_50)
        mul_tensor_54: "f32[8, 128, 512]" = torch.ops.aten.mul.Tensor(mul_tensor_48, rsqrt_17);  mul_tensor_48 = None
        sum_dim_int_list_26: "f32[8, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_53, [2], True);  mul_tensor_53 = None
        add_tensor_9: "f32[8, 128, 512]" = torch.ops.aten.add.Tensor(add_261, mul_tensor_54);  add_261 = mul_tensor_54 = None
        pow_tensor_scalar: "f32[8, 128, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_17, 3);  rsqrt_17 = None
        mul_scalar: "f32[8, 128, 1]" = torch.ops.aten.mul.Scalar(sum_dim_int_list_26, -0.5);  sum_dim_int_list_26 = None
        mul_tensor_55: "f32[8, 128, 1]" = torch.ops.aten.mul.Tensor(mul_scalar, pow_tensor_scalar);  mul_scalar = pow_tensor_scalar = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_default: "f32[8, 128, 512]" = torch.ops.aten.expand.default(mul_tensor_55, _shape_param_44);  mul_tensor_55 = _shape_param_44 = None
        div_scalar: "f32[8, 128, 512]" = torch.ops.aten.div.Scalar(expand_default, 512);  expand_default = None
        pow_tensor_scalar_1: "f32[8, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(mul_tensor_50, 1.0);  mul_tensor_50 = None
        mul_scalar_1: "f32[8, 128, 512]" = torch.ops.aten.mul.Scalar(pow_tensor_scalar_1, 2.0);  pow_tensor_scalar_1 = None
        mul_tensor_56: "f32[8, 128, 512]" = torch.ops.aten.mul.Tensor(div_scalar, mul_scalar_1);  div_scalar = mul_scalar_1 = None
        add_tensor_10: "f32[8, 128, 512]" = torch.ops.aten.add.Tensor(add_tensor_9, mul_tensor_56);  add_tensor_9 = mul_tensor_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:732 in forward, code: hidden_states = self.dropout(inputs_embeds)
        convert_element_type_default: "f32[8, 128, 512]" = torch.ops.prims.convert_element_type.default(gt_35, torch.float32);  gt_35 = None
        mul_tensor_57: "f32[8, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 1.1111111111111112);  convert_element_type_default = None
        mul_tensor_58: "f32[8, 128, 512]" = torch.ops.aten.mul.Tensor(add_tensor_10, mul_tensor_57);  add_tensor_10 = mul_tensor_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:680 in forward, code: inputs_embeds = self.embed_tokens(input_ids)
        eq_scalar_1: "b8[8, 128]" = torch.ops.aten.eq.Scalar(primals_1, -1)
        unsqueeze_default_1: "b8[8, 128, 1]" = torch.ops.aten.unsqueeze.default(eq_scalar_1, -1);  eq_scalar_1 = None
        where_self_1: "f32[8, 128, 512]" = torch.ops.aten.where.self(unsqueeze_default_1, full_default, mul_tensor_58);  mul_tensor_58 = None
        full_default_2: "f32[250112, 512]" = torch.ops.aten.full.default([250112, 512], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default_1: "f32[250112, 512]" = torch.ops.aten.index_put.default(full_default_2, [primals_1], where_self_1, True);  where_self_1 = None
        add_tensor_11: "f32[250112, 512]" = torch.ops.aten.add.Tensor(mm_145, index_put_default_1);  mm_145 = index_put_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_59: "f32[8, 128, 512]" = torch.ops.aten.mul.Tensor(add_62, rsqrt_16);  add_62 = rsqrt_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_tensor_60: "f32[8, 128, 512]" = torch.ops.aten.mul.Tensor(mul_742, mul_tensor_59);  mul_742 = mul_tensor_59 = None
        sum_dim_int_list_27: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_60, [0, 1], True);  mul_tensor_60 = None
        reshape_default_44: "f32[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_27, _shape_param_45);  sum_dim_int_list_27 = _shape_param_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:121 in forward, code: hidden_states = self.wo(hidden_states)
        permute_default_89: "f32[512, 1024]" = torch.ops.aten.permute.default(view_1035, [1, 0]);  view_1035 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:107 in forward, code: hidden_linear = self.wi_1(hidden_states)
        permute_default_90: "f32[1024, 1024]" = torch.ops.aten.permute.default(view_1037, [1, 0]);  view_1037 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:106 in forward, code: hidden_gelu = self.act(self.wi_0(hidden_states))
        permute_default_91: "f32[1024, 1024]" = torch.ops.aten.permute.default(view_1039, [1, 0]);  view_1039 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_61: "f32[8, 128, 512]" = torch.ops.aten.mul.Tensor(add_58, rsqrt_15);  add_58 = rsqrt_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_tensor_62: "f32[8, 128, 512]" = torch.ops.aten.mul.Tensor(add_271, mul_tensor_61);  add_271 = mul_tensor_61 = None
        sum_dim_int_list_28: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_62, [0, 1], True);  mul_tensor_62 = None
        reshape_default_45: "f32[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_28, _shape_param_46);  sum_dim_int_list_28 = _shape_param_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:330 in forward, code: attn_output = self.o(attn_output)
        permute_default_92: "f32[512, 1024]" = torch.ops.aten.permute.default(view_1042, [1, 0]);  view_1042 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:291 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_default_93: "f32[384, 1024]" = torch.ops.aten.permute.default(view_1056, [1, 0]);  view_1056 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:290 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_default_94: "f32[384, 1024]" = torch.ops.aten.permute.default(view_1059, [1, 0]);  view_1059 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:269 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_95: "f32[384, 1024]" = torch.ops.aten.permute.default(view_1062, [1, 0]);  view_1062 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_63: "f32[8, 128, 512]" = torch.ops.aten.mul.Tensor(add_55, rsqrt_14);  add_55 = rsqrt_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_tensor_64: "f32[8, 128, 512]" = torch.ops.aten.mul.Tensor(add_275, mul_tensor_63);  add_275 = mul_tensor_63 = None
        sum_dim_int_list_29: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_64, [0, 1], True);  mul_tensor_64 = None
        reshape_default_46: "f32[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_29, _shape_param_47);  sum_dim_int_list_29 = _shape_param_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:121 in forward, code: hidden_states = self.wo(hidden_states)
        permute_default_96: "f32[512, 1024]" = torch.ops.aten.permute.default(view_1065, [1, 0]);  view_1065 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:107 in forward, code: hidden_linear = self.wi_1(hidden_states)
        permute_default_97: "f32[1024, 1024]" = torch.ops.aten.permute.default(view_1067, [1, 0]);  view_1067 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:106 in forward, code: hidden_gelu = self.act(self.wi_0(hidden_states))
        permute_default_98: "f32[1024, 1024]" = torch.ops.aten.permute.default(view_1069, [1, 0]);  view_1069 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_65: "f32[8, 128, 512]" = torch.ops.aten.mul.Tensor(add_51, rsqrt_13);  add_51 = rsqrt_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_tensor_66: "f32[8, 128, 512]" = torch.ops.aten.mul.Tensor(add_280, mul_tensor_65);  add_280 = mul_tensor_65 = None
        sum_dim_int_list_30: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_66, [0, 1], True);  mul_tensor_66 = None
        reshape_default_47: "f32[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_30, _shape_param_48);  sum_dim_int_list_30 = _shape_param_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:330 in forward, code: attn_output = self.o(attn_output)
        permute_default_99: "f32[512, 1024]" = torch.ops.aten.permute.default(view_1072, [1, 0]);  view_1072 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:320 in forward, code: scores += position_bias_masked
        add_tensor_12: "f32[8, 6, 128, 128]" = torch.ops.aten.add.Tensor(view_1050, view_1080);  view_1050 = view_1080 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:291 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_default_100: "f32[384, 1024]" = torch.ops.aten.permute.default(view_1086, [1, 0]);  view_1086 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:290 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_default_101: "f32[384, 1024]" = torch.ops.aten.permute.default(view_1089, [1, 0]);  view_1089 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:269 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_102: "f32[384, 1024]" = torch.ops.aten.permute.default(view_1092, [1, 0]);  view_1092 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_67: "f32[8, 128, 512]" = torch.ops.aten.mul.Tensor(add_48, rsqrt_12);  add_48 = rsqrt_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_tensor_68: "f32[8, 128, 512]" = torch.ops.aten.mul.Tensor(add_285, mul_tensor_67);  add_285 = mul_tensor_67 = None
        sum_dim_int_list_31: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_68, [0, 1], True);  mul_tensor_68 = None
        reshape_default_48: "f32[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_31, _shape_param_49);  sum_dim_int_list_31 = _shape_param_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:121 in forward, code: hidden_states = self.wo(hidden_states)
        permute_default_103: "f32[512, 1024]" = torch.ops.aten.permute.default(view_1095, [1, 0]);  view_1095 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:107 in forward, code: hidden_linear = self.wi_1(hidden_states)
        permute_default_104: "f32[1024, 1024]" = torch.ops.aten.permute.default(view_1097, [1, 0]);  view_1097 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:106 in forward, code: hidden_gelu = self.act(self.wi_0(hidden_states))
        permute_default_105: "f32[1024, 1024]" = torch.ops.aten.permute.default(view_1099, [1, 0]);  view_1099 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_69: "f32[8, 128, 512]" = torch.ops.aten.mul.Tensor(add_44, rsqrt_11);  add_44 = rsqrt_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_tensor_70: "f32[8, 128, 512]" = torch.ops.aten.mul.Tensor(add_290, mul_tensor_69);  add_290 = mul_tensor_69 = None
        sum_dim_int_list_32: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_70, [0, 1], True);  mul_tensor_70 = None
        reshape_default_49: "f32[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_32, _shape_param_50);  sum_dim_int_list_32 = _shape_param_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:330 in forward, code: attn_output = self.o(attn_output)
        permute_default_106: "f32[512, 1024]" = torch.ops.aten.permute.default(view_1102, [1, 0]);  view_1102 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:320 in forward, code: scores += position_bias_masked
        add_tensor_13: "f32[8, 6, 128, 128]" = torch.ops.aten.add.Tensor(add_tensor_12, view_1110);  add_tensor_12 = view_1110 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:291 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_default_107: "f32[384, 1024]" = torch.ops.aten.permute.default(view_1116, [1, 0]);  view_1116 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:290 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_default_108: "f32[384, 1024]" = torch.ops.aten.permute.default(view_1119, [1, 0]);  view_1119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:269 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_109: "f32[384, 1024]" = torch.ops.aten.permute.default(view_1122, [1, 0]);  view_1122 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_71: "f32[8, 128, 512]" = torch.ops.aten.mul.Tensor(add_41, rsqrt_10);  add_41 = rsqrt_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_tensor_72: "f32[8, 128, 512]" = torch.ops.aten.mul.Tensor(add_295, mul_tensor_71);  add_295 = mul_tensor_71 = None
        sum_dim_int_list_33: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_72, [0, 1], True);  mul_tensor_72 = None
        reshape_default_50: "f32[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_33, _shape_param_51);  sum_dim_int_list_33 = _shape_param_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:121 in forward, code: hidden_states = self.wo(hidden_states)
        permute_default_110: "f32[512, 1024]" = torch.ops.aten.permute.default(view_1125, [1, 0]);  view_1125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:107 in forward, code: hidden_linear = self.wi_1(hidden_states)
        permute_default_111: "f32[1024, 1024]" = torch.ops.aten.permute.default(view_1127, [1, 0]);  view_1127 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:106 in forward, code: hidden_gelu = self.act(self.wi_0(hidden_states))
        permute_default_112: "f32[1024, 1024]" = torch.ops.aten.permute.default(view_1129, [1, 0]);  view_1129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_73: "f32[8, 128, 512]" = torch.ops.aten.mul.Tensor(add_37, rsqrt_9);  add_37 = rsqrt_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_tensor_74: "f32[8, 128, 512]" = torch.ops.aten.mul.Tensor(add_300, mul_tensor_73);  add_300 = mul_tensor_73 = None
        sum_dim_int_list_34: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_74, [0, 1], True);  mul_tensor_74 = None
        reshape_default_51: "f32[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_34, _shape_param_52);  sum_dim_int_list_34 = _shape_param_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:330 in forward, code: attn_output = self.o(attn_output)
        permute_default_113: "f32[512, 1024]" = torch.ops.aten.permute.default(view_1132, [1, 0]);  view_1132 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:320 in forward, code: scores += position_bias_masked
        add_tensor_14: "f32[8, 6, 128, 128]" = torch.ops.aten.add.Tensor(add_tensor_13, view_1140);  add_tensor_13 = view_1140 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:291 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_default_114: "f32[384, 1024]" = torch.ops.aten.permute.default(view_1146, [1, 0]);  view_1146 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:290 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_default_115: "f32[384, 1024]" = torch.ops.aten.permute.default(view_1149, [1, 0]);  view_1149 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:269 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_116: "f32[384, 1024]" = torch.ops.aten.permute.default(view_1152, [1, 0]);  view_1152 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_75: "f32[8, 128, 512]" = torch.ops.aten.mul.Tensor(add_34, rsqrt_8);  add_34 = rsqrt_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_tensor_76: "f32[8, 128, 512]" = torch.ops.aten.mul.Tensor(add_305, mul_tensor_75);  add_305 = mul_tensor_75 = None
        sum_dim_int_list_35: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_76, [0, 1], True);  mul_tensor_76 = None
        reshape_default_52: "f32[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_35, _shape_param_53);  sum_dim_int_list_35 = _shape_param_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:121 in forward, code: hidden_states = self.wo(hidden_states)
        permute_default_117: "f32[512, 1024]" = torch.ops.aten.permute.default(view_1155, [1, 0]);  view_1155 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:107 in forward, code: hidden_linear = self.wi_1(hidden_states)
        permute_default_118: "f32[1024, 1024]" = torch.ops.aten.permute.default(view_1157, [1, 0]);  view_1157 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:106 in forward, code: hidden_gelu = self.act(self.wi_0(hidden_states))
        permute_default_119: "f32[1024, 1024]" = torch.ops.aten.permute.default(view_1159, [1, 0]);  view_1159 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_77: "f32[8, 128, 512]" = torch.ops.aten.mul.Tensor(add_30, rsqrt_7);  add_30 = rsqrt_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_tensor_78: "f32[8, 128, 512]" = torch.ops.aten.mul.Tensor(add_310, mul_tensor_77);  add_310 = mul_tensor_77 = None
        sum_dim_int_list_36: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_78, [0, 1], True);  mul_tensor_78 = None
        reshape_default_53: "f32[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_36, _shape_param_54);  sum_dim_int_list_36 = _shape_param_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:330 in forward, code: attn_output = self.o(attn_output)
        permute_default_120: "f32[512, 1024]" = torch.ops.aten.permute.default(view_1162, [1, 0]);  view_1162 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:320 in forward, code: scores += position_bias_masked
        add_tensor_15: "f32[8, 6, 128, 128]" = torch.ops.aten.add.Tensor(add_tensor_14, view_1170);  add_tensor_14 = view_1170 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:291 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_default_121: "f32[384, 1024]" = torch.ops.aten.permute.default(view_1176, [1, 0]);  view_1176 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:290 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_default_122: "f32[384, 1024]" = torch.ops.aten.permute.default(view_1179, [1, 0]);  view_1179 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:269 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_123: "f32[384, 1024]" = torch.ops.aten.permute.default(view_1182, [1, 0]);  view_1182 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_79: "f32[8, 128, 512]" = torch.ops.aten.mul.Tensor(add_27, rsqrt_6);  add_27 = rsqrt_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_tensor_80: "f32[8, 128, 512]" = torch.ops.aten.mul.Tensor(add_315, mul_tensor_79);  add_315 = mul_tensor_79 = None
        sum_dim_int_list_37: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_80, [0, 1], True);  mul_tensor_80 = None
        reshape_default_54: "f32[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_37, _shape_param_55);  sum_dim_int_list_37 = _shape_param_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:121 in forward, code: hidden_states = self.wo(hidden_states)
        permute_default_124: "f32[512, 1024]" = torch.ops.aten.permute.default(view_1185, [1, 0]);  view_1185 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:107 in forward, code: hidden_linear = self.wi_1(hidden_states)
        permute_default_125: "f32[1024, 1024]" = torch.ops.aten.permute.default(view_1187, [1, 0]);  view_1187 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:106 in forward, code: hidden_gelu = self.act(self.wi_0(hidden_states))
        permute_default_126: "f32[1024, 1024]" = torch.ops.aten.permute.default(view_1189, [1, 0]);  view_1189 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_81: "f32[8, 128, 512]" = torch.ops.aten.mul.Tensor(add_23, rsqrt_5);  add_23 = rsqrt_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_tensor_82: "f32[8, 128, 512]" = torch.ops.aten.mul.Tensor(add_320, mul_tensor_81);  add_320 = mul_tensor_81 = None
        sum_dim_int_list_38: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_82, [0, 1], True);  mul_tensor_82 = None
        reshape_default_55: "f32[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_38, _shape_param_56);  sum_dim_int_list_38 = _shape_param_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:330 in forward, code: attn_output = self.o(attn_output)
        permute_default_127: "f32[512, 1024]" = torch.ops.aten.permute.default(view_1192, [1, 0]);  view_1192 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:320 in forward, code: scores += position_bias_masked
        add_tensor_16: "f32[8, 6, 128, 128]" = torch.ops.aten.add.Tensor(add_tensor_15, view_1200);  add_tensor_15 = view_1200 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:291 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_default_128: "f32[384, 1024]" = torch.ops.aten.permute.default(view_1206, [1, 0]);  view_1206 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:290 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_default_129: "f32[384, 1024]" = torch.ops.aten.permute.default(view_1209, [1, 0]);  view_1209 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:269 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_130: "f32[384, 1024]" = torch.ops.aten.permute.default(view_1212, [1, 0]);  view_1212 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_83: "f32[8, 128, 512]" = torch.ops.aten.mul.Tensor(add_20, rsqrt_4);  add_20 = rsqrt_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_tensor_84: "f32[8, 128, 512]" = torch.ops.aten.mul.Tensor(add_325, mul_tensor_83);  add_325 = mul_tensor_83 = None
        sum_dim_int_list_39: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_84, [0, 1], True);  mul_tensor_84 = None
        reshape_default_56: "f32[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_39, _shape_param_57);  sum_dim_int_list_39 = _shape_param_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:121 in forward, code: hidden_states = self.wo(hidden_states)
        permute_default_131: "f32[512, 1024]" = torch.ops.aten.permute.default(view_1215, [1, 0]);  view_1215 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:107 in forward, code: hidden_linear = self.wi_1(hidden_states)
        permute_default_132: "f32[1024, 1024]" = torch.ops.aten.permute.default(view_1217, [1, 0]);  view_1217 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:106 in forward, code: hidden_gelu = self.act(self.wi_0(hidden_states))
        permute_default_133: "f32[1024, 1024]" = torch.ops.aten.permute.default(view_1219, [1, 0]);  view_1219 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_85: "f32[8, 128, 512]" = torch.ops.aten.mul.Tensor(add_16, rsqrt_3);  add_16 = rsqrt_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_tensor_86: "f32[8, 128, 512]" = torch.ops.aten.mul.Tensor(add_330, mul_tensor_85);  add_330 = mul_tensor_85 = None
        sum_dim_int_list_40: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_86, [0, 1], True);  mul_tensor_86 = None
        reshape_default_57: "f32[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_40, _shape_param_58);  sum_dim_int_list_40 = _shape_param_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:330 in forward, code: attn_output = self.o(attn_output)
        permute_default_134: "f32[512, 1024]" = torch.ops.aten.permute.default(view_1222, [1, 0]);  view_1222 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:320 in forward, code: scores += position_bias_masked
        add_tensor_17: "f32[8, 6, 128, 128]" = torch.ops.aten.add.Tensor(add_tensor_16, view_1230);  add_tensor_16 = view_1230 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:291 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_default_135: "f32[384, 1024]" = torch.ops.aten.permute.default(view_1236, [1, 0]);  view_1236 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:290 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_default_136: "f32[384, 1024]" = torch.ops.aten.permute.default(view_1239, [1, 0]);  view_1239 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:269 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_137: "f32[384, 1024]" = torch.ops.aten.permute.default(view_1242, [1, 0]);  view_1242 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_87: "f32[8, 128, 512]" = torch.ops.aten.mul.Tensor(add_13, rsqrt_2);  add_13 = rsqrt_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_tensor_88: "f32[8, 128, 512]" = torch.ops.aten.mul.Tensor(add_335, mul_tensor_87);  add_335 = mul_tensor_87 = None
        sum_dim_int_list_41: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_88, [0, 1], True);  mul_tensor_88 = None
        reshape_default_58: "f32[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_41, _shape_param_59);  sum_dim_int_list_41 = _shape_param_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:121 in forward, code: hidden_states = self.wo(hidden_states)
        permute_default_138: "f32[512, 1024]" = torch.ops.aten.permute.default(view_1245, [1, 0]);  view_1245 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:107 in forward, code: hidden_linear = self.wi_1(hidden_states)
        permute_default_139: "f32[1024, 1024]" = torch.ops.aten.permute.default(view_1247, [1, 0]);  view_1247 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:106 in forward, code: hidden_gelu = self.act(self.wi_0(hidden_states))
        permute_default_140: "f32[1024, 1024]" = torch.ops.aten.permute.default(view_1249, [1, 0]);  view_1249 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_89: "f32[8, 128, 512]" = torch.ops.aten.mul.Tensor(add_9, rsqrt_1);  add_9 = rsqrt_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_tensor_90: "f32[8, 128, 512]" = torch.ops.aten.mul.Tensor(add_340, mul_tensor_89);  add_340 = mul_tensor_89 = None
        sum_dim_int_list_42: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_90, [0, 1], True);  mul_tensor_90 = None
        reshape_default_59: "f32[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_42, _shape_param_60);  sum_dim_int_list_42 = _shape_param_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:330 in forward, code: attn_output = self.o(attn_output)
        permute_default_141: "f32[512, 1024]" = torch.ops.aten.permute.default(view_1252, [1, 0]);  view_1252 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:320 in forward, code: scores += position_bias_masked
        add_tensor_18: "f32[8, 6, 128, 128]" = torch.ops.aten.add.Tensor(add_tensor_17, view_1260);  add_tensor_17 = view_1260 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:317 in forward, code: position_bias = position_bias + causal_mask
        sum_dim_int_list_43: "f32[1, 6, 128, 128]" = torch.ops.aten.sum.dim_IntList(add_tensor_18, [0], True);  add_tensor_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:242 in compute_bias, code: values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
        squeeze_dim_1: "f32[6, 128, 128]" = torch.ops.aten.squeeze.dim(sum_dim_int_list_43, 0);  sum_dim_int_list_43 = None
        permute_default_142: "f32[128, 128, 6]" = torch.ops.aten.permute.default(squeeze_dim_1, [1, 2, 0]);  squeeze_dim_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:241 in compute_bias, code: values = self.relative_attention_bias(relative_position_bucket)  # shape (query_length, key_length, num_heads)
        eq_scalar_2: "b8[128, 128]" = torch.ops.aten.eq.Scalar(add_6, -1)
        unsqueeze_default_2: "b8[128, 128, 1]" = torch.ops.aten.unsqueeze.default(eq_scalar_2, -1);  eq_scalar_2 = None
        where_self_2: "f32[128, 128, 6]" = torch.ops.aten.where.self(unsqueeze_default_2, full_default, permute_default_142);  unsqueeze_default_2 = permute_default_142 = None
        clone_default_1: "f32[128, 128, 6]" = torch.ops.aten.clone.default(where_self_2, memory_format = torch.contiguous_format);  where_self_2 = None
        index_put_default_2: "f32[32, 6]" = torch.ops.aten.index_put.default(full_default_1, [add_6], clone_default_1, True);  full_default_1 = add_6 = clone_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:291 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_default_143: "f32[384, 1024]" = torch.ops.aten.permute.default(view_1266, [1, 0]);  view_1266 = None
        reshape_default_60: "f32[8, 128, 512]" = torch.ops.aten.reshape.default(mm_430, _shape_param_61);  mm_430 = _shape_param_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:290 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_default_144: "f32[384, 1024]" = torch.ops.aten.permute.default(view_1269, [1, 0]);  view_1269 = None
        reshape_default_61: "f32[8, 128, 512]" = torch.ops.aten.reshape.default(mm_432, _shape_param_62);  mm_432 = _shape_param_62 = None
        add_tensor_19: "f32[8, 128, 512]" = torch.ops.aten.add.Tensor(reshape_default_60, reshape_default_61);  reshape_default_60 = reshape_default_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:269 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_145: "f32[384, 1024]" = torch.ops.aten.permute.default(view_1272, [1, 0]);  view_1272 = None
        reshape_default_62: "f32[8, 128, 512]" = torch.ops.aten.reshape.default(mm_434, _shape_param_63);  mm_434 = _shape_param_63 = None
        add_tensor_20: "f32[8, 128, 512]" = torch.ops.aten.add.Tensor(add_tensor_19, reshape_default_62);  add_tensor_19 = reshape_default_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_tensor_91: "f32[8, 128, 512]" = torch.ops.aten.mul.Tensor(add_tensor_20, primals_3);  primals_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:732 in forward, code: hidden_states = self.dropout(inputs_embeds)
        mul_tensor_92: "f32[8, 128, 512]" = torch.ops.aten.mul.Tensor(gt, embedding);  embedding = None
        mul_tensor_93: "f32[8, 128, 512]" = torch.ops.aten.mul.Tensor(mul_tensor_92, 1.1111111111111112);  mul_tensor_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_94: "f32[8, 128, 512]" = torch.ops.aten.mul.Tensor(mul_tensor_93, rsqrt)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_tensor_95: "f32[8, 128, 512]" = torch.ops.aten.mul.Tensor(add_tensor_20, mul_tensor_94);  add_tensor_20 = mul_tensor_94 = None
        sum_dim_int_list_44: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_95, [0, 1], True);  mul_tensor_95 = None
        reshape_default_63: "f32[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_44, _shape_param_64);  sum_dim_int_list_44 = _shape_param_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_96: "f32[8, 128, 512]" = torch.ops.aten.mul.Tensor(mul_tensor_91, mul_tensor_93)
        mul_tensor_97: "f32[8, 128, 512]" = torch.ops.aten.mul.Tensor(mul_tensor_91, rsqrt);  mul_tensor_91 = None
        sum_dim_int_list_45: "f32[8, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_96, [2], True);  mul_tensor_96 = None
        add_tensor_21: "f32[8, 128, 512]" = torch.ops.aten.add.Tensor(add_342, mul_tensor_97);  add_342 = mul_tensor_97 = None
        pow_tensor_scalar_2: "f32[8, 128, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt, 3);  rsqrt = None
        mul_scalar_2: "f32[8, 128, 1]" = torch.ops.aten.mul.Scalar(sum_dim_int_list_45, -0.5);  sum_dim_int_list_45 = None
        mul_tensor_98: "f32[8, 128, 1]" = torch.ops.aten.mul.Tensor(mul_scalar_2, pow_tensor_scalar_2);  mul_scalar_2 = pow_tensor_scalar_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_default_1: "f32[8, 128, 512]" = torch.ops.aten.expand.default(mul_tensor_98, _shape_param_65);  mul_tensor_98 = _shape_param_65 = None
        div_scalar_1: "f32[8, 128, 512]" = torch.ops.aten.div.Scalar(expand_default_1, 512);  expand_default_1 = None
        pow_tensor_scalar_3: "f32[8, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(mul_tensor_93, 1.0);  mul_tensor_93 = None
        mul_scalar_3: "f32[8, 128, 512]" = torch.ops.aten.mul.Scalar(pow_tensor_scalar_3, 2.0);  pow_tensor_scalar_3 = None
        mul_tensor_99: "f32[8, 128, 512]" = torch.ops.aten.mul.Tensor(div_scalar_1, mul_scalar_3);  div_scalar_1 = mul_scalar_3 = None
        add_tensor_22: "f32[8, 128, 512]" = torch.ops.aten.add.Tensor(add_tensor_21, mul_tensor_99);  add_tensor_21 = mul_tensor_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:732 in forward, code: hidden_states = self.dropout(inputs_embeds)
        convert_element_type_default_1: "f32[8, 128, 512]" = torch.ops.prims.convert_element_type.default(gt, torch.float32);  gt = None
        mul_tensor_100: "f32[8, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_default_1, 1.1111111111111112);  convert_element_type_default_1 = None
        mul_tensor_101: "f32[8, 128, 512]" = torch.ops.aten.mul.Tensor(add_tensor_22, mul_tensor_100);  add_tensor_22 = mul_tensor_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:680 in forward, code: inputs_embeds = self.embed_tokens(input_ids)
        where_self_3: "f32[8, 128, 512]" = torch.ops.aten.where.self(unsqueeze_default_1, full_default, mul_tensor_101);  unsqueeze_default_1 = full_default = mul_tensor_101 = None
        index_put_default_3: "f32[250112, 512]" = torch.ops.aten.index_put.default(full_default_2, [primals_1], where_self_3, True);  full_default_2 = primals_1 = where_self_3 = None
        add_tensor_23: "f32[250112, 512]" = torch.ops.aten.add.Tensor(add_tensor_11, index_put_default_3);  add_tensor_11 = index_put_default_3 = None
        return (reshape_default, permute_default, permute_default_1, permute_default_2, reshape_default_1, permute_default_3, permute_default_4, reshape_default_2, permute_default_5, reshape_default_3, permute_default_6, reshape_default_4, permute_default_7, permute_default_8, permute_default_9, permute_default_10, reshape_default_5, permute_default_11, permute_default_12, permute_default_13, reshape_default_6, permute_default_14, permute_default_15, reshape_default_7, permute_default_16, reshape_default_8, permute_default_17, reshape_default_9, permute_default_18, permute_default_19, permute_default_20, permute_default_21, reshape_default_10, permute_default_22, permute_default_23, permute_default_24, reshape_default_11, permute_default_25, permute_default_26, reshape_default_12, permute_default_27, reshape_default_13, permute_default_28, reshape_default_14, permute_default_29, permute_default_30, permute_default_31, permute_default_32, reshape_default_15, permute_default_33, permute_default_34, permute_default_35, reshape_default_16, permute_default_36, permute_default_37, reshape_default_17, permute_default_38, reshape_default_18, permute_default_39, reshape_default_19, permute_default_40, permute_default_41, permute_default_42, permute_default_43, reshape_default_20, permute_default_44, permute_default_45, permute_default_46, reshape_default_21, permute_default_47, permute_default_48, reshape_default_22, permute_default_49, reshape_default_23, permute_default_50, reshape_default_24, permute_default_51, permute_default_52, permute_default_53, permute_default_54, reshape_default_25, permute_default_55, permute_default_56, permute_default_57, reshape_default_26, permute_default_58, permute_default_59, reshape_default_27, permute_default_60, reshape_default_28, permute_default_61, reshape_default_29, permute_default_62, permute_default_63, permute_default_64, permute_default_65, reshape_default_30, permute_default_66, permute_default_67, permute_default_68, reshape_default_31, permute_default_69, permute_default_70, reshape_default_32, permute_default_71, reshape_default_33, permute_default_72, reshape_default_34, permute_default_73, permute_default_74, permute_default_75, permute_default_76, reshape_default_35, permute_default_77, permute_default_78, permute_default_79, reshape_default_36, permute_default_80, permute_default_81, reshape_default_37, permute_default_82, reshape_default_38, permute_default_83, reshape_default_39, permute_default_84, index_put_default, permute_default_86, permute_default_87, permute_default_88, reshape_default_43, reshape_default_44, permute_default_89, permute_default_90, permute_default_91, reshape_default_45, permute_default_92, permute_default_93, permute_default_94, permute_default_95, reshape_default_46, permute_default_96, permute_default_97, permute_default_98, reshape_default_47, permute_default_99, permute_default_100, permute_default_101, permute_default_102, reshape_default_48, permute_default_103, permute_default_104, permute_default_105, reshape_default_49, permute_default_106, permute_default_107, permute_default_108, permute_default_109, reshape_default_50, permute_default_110, permute_default_111, permute_default_112, reshape_default_51, permute_default_113, permute_default_114, permute_default_115, permute_default_116, reshape_default_52, permute_default_117, permute_default_118, permute_default_119, reshape_default_53, permute_default_120, permute_default_121, permute_default_122, permute_default_123, reshape_default_54, permute_default_124, permute_default_125, permute_default_126, reshape_default_55, permute_default_127, permute_default_128, permute_default_129, permute_default_130, reshape_default_56, permute_default_131, permute_default_132, permute_default_133, reshape_default_57, permute_default_134, permute_default_135, permute_default_136, permute_default_137, reshape_default_58, permute_default_138, permute_default_139, permute_default_140, reshape_default_59, permute_default_141, index_put_default_2, permute_default_143, permute_default_144, permute_default_145, reshape_default_63, add_tensor_23)


def _default_make_inputs():
    return [
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    [512],  # _shape_param_0
    torch.randn([1024, 512], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    [512],  # _shape_param_1
    torch.randn([1024, 512], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 384], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    [1024, 512],  # _shape_param_2
    torch.randn([1024, 384], dtype=torch.float32, device='cuda'),
    [1024, 512],  # _shape_param_3
    torch.randn([1024, 384], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    [512],  # _shape_param_4
    torch.randn([1024, 512], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 384], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 384], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 384], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    [512],  # _shape_param_5
    torch.randn([1024, 512], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    [512],  # _shape_param_6
    torch.randn([1024, 512], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 384], dtype=torch.float32, device='cuda'),
    [1024, 512],  # _shape_param_7
    torch.randn([1024, 384], dtype=torch.float32, device='cuda'),
    [1024, 512],  # _shape_param_8
    torch.randn([1024, 384], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    [512],  # _shape_param_9
    torch.randn([1024, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 6, 128, 128], dtype=torch.float32, device='cuda'),
    torch.randn([8, 6, 128, 128], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 384], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 384], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 384], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    [512],  # _shape_param_10
    torch.randn([1024, 512], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    [512],  # _shape_param_11
    torch.randn([1024, 512], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 384], dtype=torch.float32, device='cuda'),
    [1024, 512],  # _shape_param_12
    torch.randn([1024, 384], dtype=torch.float32, device='cuda'),
    [1024, 512],  # _shape_param_13
    torch.randn([1024, 384], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    [512],  # _shape_param_14
    torch.randn([1024, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 6, 128, 128], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 384], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 384], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 384], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    [512],  # _shape_param_15
    torch.randn([1024, 512], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    [512],  # _shape_param_16
    torch.randn([1024, 512], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 384], dtype=torch.float32, device='cuda'),
    [1024, 512],  # _shape_param_17
    torch.randn([1024, 384], dtype=torch.float32, device='cuda'),
    [1024, 512],  # _shape_param_18
    torch.randn([1024, 384], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    [512],  # _shape_param_19
    torch.randn([1024, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 6, 128, 128], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 384], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 384], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 384], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    [512],  # _shape_param_20
    torch.randn([1024, 512], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    [512],  # _shape_param_21
    torch.randn([1024, 512], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 384], dtype=torch.float32, device='cuda'),
    [1024, 512],  # _shape_param_22
    torch.randn([1024, 384], dtype=torch.float32, device='cuda'),
    [1024, 512],  # _shape_param_23
    torch.randn([1024, 384], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    [512],  # _shape_param_24
    torch.randn([1024, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 6, 128, 128], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 384], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 384], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 384], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    [512],  # _shape_param_25
    torch.randn([1024, 512], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    [512],  # _shape_param_26
    torch.randn([1024, 512], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 384], dtype=torch.float32, device='cuda'),
    [1024, 512],  # _shape_param_27
    torch.randn([1024, 384], dtype=torch.float32, device='cuda'),
    [1024, 512],  # _shape_param_28
    torch.randn([1024, 384], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    [512],  # _shape_param_29
    torch.randn([1024, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 6, 128, 128], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 384], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 384], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 384], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    [512],  # _shape_param_30
    torch.randn([1024, 512], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    [512],  # _shape_param_31
    torch.randn([1024, 512], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 384], dtype=torch.float32, device='cuda'),
    [1024, 512],  # _shape_param_32
    torch.randn([1024, 384], dtype=torch.float32, device='cuda'),
    [1024, 512],  # _shape_param_33
    torch.randn([1024, 384], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    [512],  # _shape_param_34
    torch.randn([1024, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 6, 128, 128], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 384], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 384], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 384], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    [512],  # _shape_param_35
    torch.randn([1024, 512], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    [512],  # _shape_param_36
    torch.randn([1024, 512], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 384], dtype=torch.float32, device='cuda'),
    [1024, 512],  # _shape_param_37
    torch.randn([1024, 384], dtype=torch.float32, device='cuda'),
    [1024, 512],  # _shape_param_38
    torch.randn([1024, 384], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    [512],  # _shape_param_39
    torch.randn([1024, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 6, 128, 128], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [128, 128], dtype=torch.int64, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 384], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 512], dtype=torch.float32, device='cuda'),
    [8, 128, 512],  # _shape_param_40
    torch.randn([1024, 384], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 512], dtype=torch.float32, device='cuda'),
    [8, 128, 512],  # _shape_param_41
    torch.randn([1024, 384], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 512], dtype=torch.float32, device='cuda'),
    [8, 128, 512],  # _shape_param_42
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [8, 128, 512], dtype=torch.bool, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 1], dtype=torch.float32, device='cuda'),
    [512],  # _shape_param_43
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    [8, 128, 512],  # _shape_param_44
    torch.randint(0, 2, [8, 128], dtype=torch.int64, device='cuda'),
    torch.randn([250112, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    [512],  # _shape_param_45
    torch.randn([1024, 512], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    [512],  # _shape_param_46
    torch.randn([1024, 512], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 384], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 384], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 384], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    [512],  # _shape_param_47
    torch.randn([1024, 512], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    [512],  # _shape_param_48
    torch.randn([1024, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 6, 128, 128], dtype=torch.float32, device='cuda'),
    torch.randn([8, 6, 128, 128], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 384], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 384], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 384], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    [512],  # _shape_param_49
    torch.randn([1024, 512], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    [512],  # _shape_param_50
    torch.randn([1024, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 6, 128, 128], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 384], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 384], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 384], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    [512],  # _shape_param_51
    torch.randn([1024, 512], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    [512],  # _shape_param_52
    torch.randn([1024, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 6, 128, 128], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 384], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 384], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 384], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    [512],  # _shape_param_53
    torch.randn([1024, 512], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    [512],  # _shape_param_54
    torch.randn([1024, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 6, 128, 128], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 384], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 384], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 384], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    [512],  # _shape_param_55
    torch.randn([1024, 512], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    [512],  # _shape_param_56
    torch.randn([1024, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 6, 128, 128], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 384], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 384], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 384], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    [512],  # _shape_param_57
    torch.randn([1024, 512], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    [512],  # _shape_param_58
    torch.randn([1024, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 6, 128, 128], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 384], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 384], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 384], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    [512],  # _shape_param_59
    torch.randn([1024, 512], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    [512],  # _shape_param_60
    torch.randn([1024, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 6, 128, 128], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [128, 128], dtype=torch.int64, device='cuda'),
    torch.randn([1024, 384], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 512], dtype=torch.float32, device='cuda'),
    [8, 128, 512],  # _shape_param_61
    torch.randn([1024, 384], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 512], dtype=torch.float32, device='cuda'),
    [8, 128, 512],  # _shape_param_62
    torch.randn([1024, 384], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 512], dtype=torch.float32, device='cuda'),
    [8, 128, 512],  # _shape_param_63
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [8, 128, 512], dtype=torch.bool, device='cuda'),
    torch.randn([8, 128, 1], dtype=torch.float32, device='cuda'),
    [512],  # _shape_param_64
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    [8, 128, 512],  # _shape_param_65
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
