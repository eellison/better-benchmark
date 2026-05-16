"""
Standalone repro captured via capture_hook.
Label: hf_t5_base_train
Pattern hash: 3be5624028ba
Shape hash: e5d7ed1c
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, add_104: "f32[4, 128, 768]", rsqrt_36: "f32[4, 128, 1]", mul_224: "f32[4, 128, 768]", _shape_param_0, view_553: "f32[512, 768]", view_555: "f32[512, 3072]", add_102: "f32[4, 128, 768]", rsqrt_35: "f32[4, 128, 1]", view_556: "f32[4, 128, 768]", _shape_param_1, view_558: "f32[512, 768]", view_572: "f32[2048, 768]", mm_127: "f32[2048, 768]", _shape_param_2, view_575: "f32[2048, 768]", mm_129: "f32[2048, 768]", _shape_param_3, view_578: "f32[512, 768]", add_99: "f32[4, 128, 768]", rsqrt_34: "f32[4, 128, 1]", view_579: "f32[4, 128, 768]", _shape_param_4, view_581: "f32[512, 768]", view_595: "f32[512, 768]", view_598: "f32[512, 768]", view_601: "f32[512, 768]", add_96: "f32[4, 128, 768]", rsqrt_33: "f32[4, 128, 1]", add_117: "f32[4, 128, 768]", _shape_param_5, view_604: "f32[512, 768]", view_606: "f32[512, 3072]", add_94: "f32[4, 128, 768]", rsqrt_32: "f32[4, 128, 1]", view_607: "f32[4, 128, 768]", _shape_param_6, view_609: "f32[512, 768]", view_623: "f32[2048, 768]", mm_147: "f32[2048, 768]", _shape_param_7, view_626: "f32[2048, 768]", mm_149: "f32[2048, 768]", _shape_param_8, view_629: "f32[512, 768]", add_91: "f32[4, 128, 768]", rsqrt_31: "f32[4, 128, 1]", view_630: "f32[4, 128, 768]", _shape_param_9, view_632: "f32[512, 768]", view_589: "f32[4, 12, 128, 128]", view_640: "f32[4, 12, 128, 128]", view_646: "f32[512, 768]", view_649: "f32[512, 768]", view_652: "f32[512, 768]", add_88: "f32[4, 128, 768]", rsqrt_30: "f32[4, 128, 1]", add_132: "f32[4, 128, 768]", _shape_param_10, view_655: "f32[512, 768]", view_657: "f32[512, 3072]", add_86: "f32[4, 128, 768]", rsqrt_29: "f32[4, 128, 1]", view_658: "f32[4, 128, 768]", _shape_param_11, view_660: "f32[512, 768]", view_674: "f32[2048, 768]", mm_167: "f32[2048, 768]", _shape_param_12, view_677: "f32[2048, 768]", mm_169: "f32[2048, 768]", _shape_param_13, view_680: "f32[512, 768]", add_83: "f32[4, 128, 768]", rsqrt_28: "f32[4, 128, 1]", view_681: "f32[4, 128, 768]", _shape_param_14, view_683: "f32[512, 768]", view_691: "f32[4, 12, 128, 128]", view_697: "f32[512, 768]", view_700: "f32[512, 768]", view_703: "f32[512, 768]", add_80: "f32[4, 128, 768]", rsqrt_27: "f32[4, 128, 1]", add_147: "f32[4, 128, 768]", _shape_param_15, view_706: "f32[512, 768]", view_708: "f32[512, 3072]", add_78: "f32[4, 128, 768]", rsqrt_26: "f32[4, 128, 1]", view_709: "f32[4, 128, 768]", _shape_param_16, view_711: "f32[512, 768]", view_725: "f32[2048, 768]", mm_187: "f32[2048, 768]", _shape_param_17, view_728: "f32[2048, 768]", mm_189: "f32[2048, 768]", _shape_param_18, view_731: "f32[512, 768]", add_75: "f32[4, 128, 768]", rsqrt_25: "f32[4, 128, 1]", view_732: "f32[4, 128, 768]", _shape_param_19, view_734: "f32[512, 768]", view_742: "f32[4, 12, 128, 128]", view_748: "f32[512, 768]", view_751: "f32[512, 768]", view_754: "f32[512, 768]", add_72: "f32[4, 128, 768]", rsqrt_24: "f32[4, 128, 1]", add_162: "f32[4, 128, 768]", _shape_param_20, view_757: "f32[512, 768]", view_759: "f32[512, 3072]", add_70: "f32[4, 128, 768]", rsqrt_23: "f32[4, 128, 1]", view_760: "f32[4, 128, 768]", _shape_param_21, view_762: "f32[512, 768]", view_776: "f32[2048, 768]", mm_207: "f32[2048, 768]", _shape_param_22, view_779: "f32[2048, 768]", mm_209: "f32[2048, 768]", _shape_param_23, view_782: "f32[512, 768]", add_67: "f32[4, 128, 768]", rsqrt_22: "f32[4, 128, 1]", view_783: "f32[4, 128, 768]", _shape_param_24, view_785: "f32[512, 768]", view_793: "f32[4, 12, 128, 128]", view_799: "f32[512, 768]", view_802: "f32[512, 768]", view_805: "f32[512, 768]", add_64: "f32[4, 128, 768]", rsqrt_21: "f32[4, 128, 1]", add_177: "f32[4, 128, 768]", _shape_param_25, view_808: "f32[512, 768]", view_810: "f32[512, 3072]", add_62: "f32[4, 128, 768]", rsqrt_20: "f32[4, 128, 1]", view_811: "f32[4, 128, 768]", _shape_param_26, view_813: "f32[512, 768]", view_827: "f32[2048, 768]", mm_227: "f32[2048, 768]", _shape_param_27, view_830: "f32[2048, 768]", mm_229: "f32[2048, 768]", _shape_param_28, view_833: "f32[512, 768]", add_59: "f32[4, 128, 768]", rsqrt_19: "f32[4, 128, 1]", view_834: "f32[4, 128, 768]", _shape_param_29, view_836: "f32[512, 768]", view_844: "f32[4, 12, 128, 128]", view_850: "f32[512, 768]", view_853: "f32[512, 768]", view_856: "f32[512, 768]", add_56: "f32[4, 128, 768]", rsqrt_18: "f32[4, 128, 1]", add_192: "f32[4, 128, 768]", _shape_param_30, view_859: "f32[512, 768]", view_861: "f32[512, 3072]", add_54: "f32[4, 128, 768]", rsqrt_17: "f32[4, 128, 1]", view_862: "f32[4, 128, 768]", _shape_param_31, view_864: "f32[512, 768]", view_878: "f32[2048, 768]", mm_247: "f32[2048, 768]", _shape_param_32, view_881: "f32[2048, 768]", mm_249: "f32[2048, 768]", _shape_param_33, view_884: "f32[512, 768]", add_51: "f32[4, 128, 768]", rsqrt_16: "f32[4, 128, 1]", view_885: "f32[4, 128, 768]", _shape_param_34, view_887: "f32[512, 768]", view_895: "f32[4, 12, 128, 128]", view_901: "f32[512, 768]", view_904: "f32[512, 768]", view_907: "f32[512, 768]", add_48: "f32[4, 128, 768]", rsqrt_15: "f32[4, 128, 1]", add_207: "f32[4, 128, 768]", _shape_param_35, view_910: "f32[512, 768]", view_912: "f32[512, 3072]", add_46: "f32[4, 128, 768]", rsqrt_14: "f32[4, 128, 1]", view_913: "f32[4, 128, 768]", _shape_param_36, view_915: "f32[512, 768]", view_929: "f32[2048, 768]", mm_267: "f32[2048, 768]", _shape_param_37, view_932: "f32[2048, 768]", mm_269: "f32[2048, 768]", _shape_param_38, view_935: "f32[512, 768]", add_43: "f32[4, 128, 768]", rsqrt_13: "f32[4, 128, 1]", view_936: "f32[4, 128, 768]", _shape_param_39, view_938: "f32[512, 768]", view_946: "f32[4, 12, 128, 128]", view_952: "f32[512, 768]", view_955: "f32[512, 768]", view_958: "f32[512, 768]", add_40: "f32[4, 128, 768]", rsqrt_12: "f32[4, 128, 1]", add_222: "f32[4, 128, 768]", _shape_param_40, view_961: "f32[512, 768]", view_963: "f32[512, 3072]", add_38: "f32[4, 128, 768]", rsqrt_11: "f32[4, 128, 1]", view_964: "f32[4, 128, 768]", _shape_param_41, view_966: "f32[512, 768]", view_980: "f32[2048, 768]", mm_287: "f32[2048, 768]", _shape_param_42, view_983: "f32[2048, 768]", mm_289: "f32[2048, 768]", _shape_param_43, view_986: "f32[512, 768]", add_35: "f32[4, 128, 768]", rsqrt_10: "f32[4, 128, 1]", view_987: "f32[4, 128, 768]", _shape_param_44, view_989: "f32[512, 768]", view_997: "f32[4, 12, 128, 128]", view_1003: "f32[512, 768]", view_1006: "f32[512, 768]", view_1009: "f32[512, 768]", add_32: "f32[4, 128, 768]", rsqrt_9: "f32[4, 128, 1]", add_237: "f32[4, 128, 768]", _shape_param_45, view_1012: "f32[512, 768]", view_1014: "f32[512, 3072]", add_30: "f32[4, 128, 768]", rsqrt_8: "f32[4, 128, 1]", view_1015: "f32[4, 128, 768]", _shape_param_46, view_1017: "f32[512, 768]", view_1031: "f32[2048, 768]", mm_307: "f32[2048, 768]", _shape_param_47, view_1034: "f32[2048, 768]", mm_309: "f32[2048, 768]", _shape_param_48, view_1037: "f32[512, 768]", add_27: "f32[4, 128, 768]", rsqrt_7: "f32[4, 128, 1]", view_1038: "f32[4, 128, 768]", _shape_param_49, view_1040: "f32[512, 768]", view_1048: "f32[4, 12, 128, 128]", view_1054: "f32[512, 768]", view_1057: "f32[512, 768]", view_1060: "f32[512, 768]", add_24: "f32[4, 128, 768]", rsqrt_6: "f32[4, 128, 1]", add_252: "f32[4, 128, 768]", _shape_param_50, view_1063: "f32[512, 768]", view_1065: "f32[512, 3072]", add_22: "f32[4, 128, 768]", rsqrt_5: "f32[4, 128, 1]", view_1066: "f32[4, 128, 768]", _shape_param_51, view_1068: "f32[512, 768]", view_1082: "f32[2048, 768]", mm_327: "f32[2048, 768]", _shape_param_52, view_1085: "f32[2048, 768]", mm_329: "f32[2048, 768]", _shape_param_53, view_1088: "f32[512, 768]", add_19: "f32[4, 128, 768]", rsqrt_4: "f32[4, 128, 1]", view_1089: "f32[4, 128, 768]", _shape_param_54, view_1091: "f32[512, 768]", view_1099: "f32[4, 12, 128, 128]", view_1105: "f32[512, 768]", view_1108: "f32[512, 768]", view_1111: "f32[512, 768]", add_16: "f32[4, 128, 768]", rsqrt_3: "f32[4, 128, 1]", add_267: "f32[4, 128, 768]", _shape_param_55, view_1114: "f32[512, 768]", view_1116: "f32[512, 3072]", add_14: "f32[4, 128, 768]", rsqrt_2: "f32[4, 128, 1]", view_1117: "f32[4, 128, 768]", _shape_param_56, view_1119: "f32[512, 768]", view_1133: "f32[2048, 768]", mm_347: "f32[2048, 768]", _shape_param_57, view_1136: "f32[2048, 768]", mm_349: "f32[2048, 768]", _shape_param_58, view_1139: "f32[512, 768]", add_10: "f32[4, 128, 768]", rsqrt_1: "f32[4, 128, 1]", view_1140: "f32[4, 128, 768]", _shape_param_59, view_1142: "f32[512, 768]", view_1150: "f32[4, 12, 128, 128]", add_7: "i64[128, 128]", full_default: "f32[]", view_1156: "f32[512, 768]", mm_355: "f32[512, 768]", _shape_param_60, view_1159: "f32[512, 768]", mm_357: "f32[512, 768]", _shape_param_61, view_1162: "f32[512, 768]", mm_359: "f32[512, 768]", _shape_param_62, primals_3: "f32[768]", gt: "b8[4, 128, 768]", primals_1: "f32[4, 128, 768]", rsqrt: "f32[4, 128, 1]", _shape_param_63, add_277: "f32[4, 128, 768]", _shape_param_64):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(add_104, rsqrt_36);  add_104 = rsqrt_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_tensor_1: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(mul_224, mul_tensor);  mul_224 = mul_tensor = None
        sum_dim_int_list: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_1, [0, 1], True);  mul_tensor_1 = None
        reshape_default: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list, _shape_param_0);  sum_dim_int_list = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        permute_default: "f32[768, 512]" = torch.ops.aten.permute.default(view_553, [1, 0]);  view_553 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        permute_default_1: "f32[3072, 512]" = torch.ops.aten.permute.default(view_555, [1, 0]);  view_555 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_2: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(add_102, rsqrt_35);  add_102 = rsqrt_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_tensor_3: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(view_556, mul_tensor_2);  view_556 = mul_tensor_2 = None
        sum_dim_int_list_1: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_3, [0, 1], True);  mul_tensor_3 = None
        reshape_default_1: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_1, _shape_param_1);  sum_dim_int_list_1 = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        permute_default_2: "f32[768, 512]" = torch.ops.aten.permute.default(view_558, [1, 0]);  view_558 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_default_3: "f32[768, 2048]" = torch.ops.aten.permute.default(view_572, [1, 0]);  view_572 = None
        reshape_default_2: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_127, _shape_param_2);  mm_127 = _shape_param_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_default_4: "f32[768, 2048]" = torch.ops.aten.permute.default(view_575, [1, 0]);  view_575 = None
        reshape_default_3: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_129, _shape_param_3);  mm_129 = _shape_param_3 = None
        add_tensor: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(reshape_default_2, reshape_default_3);  reshape_default_2 = reshape_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_5: "f32[768, 512]" = torch.ops.aten.permute.default(view_578, [1, 0]);  view_578 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_4: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(add_99, rsqrt_34);  add_99 = rsqrt_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_tensor_5: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(view_579, mul_tensor_4);  view_579 = mul_tensor_4 = None
        sum_dim_int_list_2: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_5, [0, 1], True);  mul_tensor_5 = None
        reshape_default_4: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_2, _shape_param_4);  sum_dim_int_list_2 = _shape_param_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        permute_default_6: "f32[768, 512]" = torch.ops.aten.permute.default(view_581, [1, 0]);  view_581 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_default_7: "f32[768, 512]" = torch.ops.aten.permute.default(view_595, [1, 0]);  view_595 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_default_8: "f32[768, 512]" = torch.ops.aten.permute.default(view_598, [1, 0]);  view_598 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_9: "f32[768, 512]" = torch.ops.aten.permute.default(view_601, [1, 0]);  view_601 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_6: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(add_96, rsqrt_33);  add_96 = rsqrt_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_tensor_7: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(add_117, mul_tensor_6);  add_117 = mul_tensor_6 = None
        sum_dim_int_list_3: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_7, [0, 1], True);  mul_tensor_7 = None
        reshape_default_5: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_3, _shape_param_5);  sum_dim_int_list_3 = _shape_param_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        permute_default_10: "f32[768, 512]" = torch.ops.aten.permute.default(view_604, [1, 0]);  view_604 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        permute_default_11: "f32[3072, 512]" = torch.ops.aten.permute.default(view_606, [1, 0]);  view_606 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_8: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(add_94, rsqrt_32);  add_94 = rsqrt_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_tensor_9: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(view_607, mul_tensor_8);  view_607 = mul_tensor_8 = None
        sum_dim_int_list_4: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_9, [0, 1], True);  mul_tensor_9 = None
        reshape_default_6: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_4, _shape_param_6);  sum_dim_int_list_4 = _shape_param_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        permute_default_12: "f32[768, 512]" = torch.ops.aten.permute.default(view_609, [1, 0]);  view_609 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_default_13: "f32[768, 2048]" = torch.ops.aten.permute.default(view_623, [1, 0]);  view_623 = None
        reshape_default_7: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_147, _shape_param_7);  mm_147 = _shape_param_7 = None
        add_tensor_1: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_tensor, reshape_default_7);  add_tensor = reshape_default_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_default_14: "f32[768, 2048]" = torch.ops.aten.permute.default(view_626, [1, 0]);  view_626 = None
        reshape_default_8: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_149, _shape_param_8);  mm_149 = _shape_param_8 = None
        add_tensor_2: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_tensor_1, reshape_default_8);  add_tensor_1 = reshape_default_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_15: "f32[768, 512]" = torch.ops.aten.permute.default(view_629, [1, 0]);  view_629 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_10: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(add_91, rsqrt_31);  add_91 = rsqrt_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_tensor_11: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(view_630, mul_tensor_10);  view_630 = mul_tensor_10 = None
        sum_dim_int_list_5: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_11, [0, 1], True);  mul_tensor_11 = None
        reshape_default_9: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_5, _shape_param_9);  sum_dim_int_list_5 = _shape_param_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        permute_default_16: "f32[768, 512]" = torch.ops.aten.permute.default(view_632, [1, 0]);  view_632 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_tensor_3: "f32[4, 12, 128, 128]" = torch.ops.aten.add.Tensor(view_589, view_640);  view_589 = view_640 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_default_17: "f32[768, 512]" = torch.ops.aten.permute.default(view_646, [1, 0]);  view_646 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_default_18: "f32[768, 512]" = torch.ops.aten.permute.default(view_649, [1, 0]);  view_649 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_19: "f32[768, 512]" = torch.ops.aten.permute.default(view_652, [1, 0]);  view_652 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_12: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(add_88, rsqrt_30);  add_88 = rsqrt_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_tensor_13: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(add_132, mul_tensor_12);  add_132 = mul_tensor_12 = None
        sum_dim_int_list_6: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_13, [0, 1], True);  mul_tensor_13 = None
        reshape_default_10: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_6, _shape_param_10);  sum_dim_int_list_6 = _shape_param_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        permute_default_20: "f32[768, 512]" = torch.ops.aten.permute.default(view_655, [1, 0]);  view_655 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        permute_default_21: "f32[3072, 512]" = torch.ops.aten.permute.default(view_657, [1, 0]);  view_657 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_14: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(add_86, rsqrt_29);  add_86 = rsqrt_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_tensor_15: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(view_658, mul_tensor_14);  view_658 = mul_tensor_14 = None
        sum_dim_int_list_7: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_15, [0, 1], True);  mul_tensor_15 = None
        reshape_default_11: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_7, _shape_param_11);  sum_dim_int_list_7 = _shape_param_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        permute_default_22: "f32[768, 512]" = torch.ops.aten.permute.default(view_660, [1, 0]);  view_660 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_default_23: "f32[768, 2048]" = torch.ops.aten.permute.default(view_674, [1, 0]);  view_674 = None
        reshape_default_12: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_167, _shape_param_12);  mm_167 = _shape_param_12 = None
        add_tensor_4: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_tensor_2, reshape_default_12);  add_tensor_2 = reshape_default_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_default_24: "f32[768, 2048]" = torch.ops.aten.permute.default(view_677, [1, 0]);  view_677 = None
        reshape_default_13: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_169, _shape_param_13);  mm_169 = _shape_param_13 = None
        add_tensor_5: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_tensor_4, reshape_default_13);  add_tensor_4 = reshape_default_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_25: "f32[768, 512]" = torch.ops.aten.permute.default(view_680, [1, 0]);  view_680 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_16: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(add_83, rsqrt_28);  add_83 = rsqrt_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_tensor_17: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(view_681, mul_tensor_16);  view_681 = mul_tensor_16 = None
        sum_dim_int_list_8: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_17, [0, 1], True);  mul_tensor_17 = None
        reshape_default_14: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_8, _shape_param_14);  sum_dim_int_list_8 = _shape_param_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        permute_default_26: "f32[768, 512]" = torch.ops.aten.permute.default(view_683, [1, 0]);  view_683 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_tensor_6: "f32[4, 12, 128, 128]" = torch.ops.aten.add.Tensor(add_tensor_3, view_691);  add_tensor_3 = view_691 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_default_27: "f32[768, 512]" = torch.ops.aten.permute.default(view_697, [1, 0]);  view_697 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_default_28: "f32[768, 512]" = torch.ops.aten.permute.default(view_700, [1, 0]);  view_700 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_29: "f32[768, 512]" = torch.ops.aten.permute.default(view_703, [1, 0]);  view_703 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_18: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(add_80, rsqrt_27);  add_80 = rsqrt_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_tensor_19: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(add_147, mul_tensor_18);  add_147 = mul_tensor_18 = None
        sum_dim_int_list_9: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_19, [0, 1], True);  mul_tensor_19 = None
        reshape_default_15: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_9, _shape_param_15);  sum_dim_int_list_9 = _shape_param_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        permute_default_30: "f32[768, 512]" = torch.ops.aten.permute.default(view_706, [1, 0]);  view_706 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        permute_default_31: "f32[3072, 512]" = torch.ops.aten.permute.default(view_708, [1, 0]);  view_708 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_20: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(add_78, rsqrt_26);  add_78 = rsqrt_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_tensor_21: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(view_709, mul_tensor_20);  view_709 = mul_tensor_20 = None
        sum_dim_int_list_10: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_21, [0, 1], True);  mul_tensor_21 = None
        reshape_default_16: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_10, _shape_param_16);  sum_dim_int_list_10 = _shape_param_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        permute_default_32: "f32[768, 512]" = torch.ops.aten.permute.default(view_711, [1, 0]);  view_711 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_default_33: "f32[768, 2048]" = torch.ops.aten.permute.default(view_725, [1, 0]);  view_725 = None
        reshape_default_17: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_187, _shape_param_17);  mm_187 = _shape_param_17 = None
        add_tensor_7: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_tensor_5, reshape_default_17);  add_tensor_5 = reshape_default_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_default_34: "f32[768, 2048]" = torch.ops.aten.permute.default(view_728, [1, 0]);  view_728 = None
        reshape_default_18: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_189, _shape_param_18);  mm_189 = _shape_param_18 = None
        add_tensor_8: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_tensor_7, reshape_default_18);  add_tensor_7 = reshape_default_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_35: "f32[768, 512]" = torch.ops.aten.permute.default(view_731, [1, 0]);  view_731 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_22: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(add_75, rsqrt_25);  add_75 = rsqrt_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_tensor_23: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(view_732, mul_tensor_22);  view_732 = mul_tensor_22 = None
        sum_dim_int_list_11: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_23, [0, 1], True);  mul_tensor_23 = None
        reshape_default_19: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_11, _shape_param_19);  sum_dim_int_list_11 = _shape_param_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        permute_default_36: "f32[768, 512]" = torch.ops.aten.permute.default(view_734, [1, 0]);  view_734 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_tensor_9: "f32[4, 12, 128, 128]" = torch.ops.aten.add.Tensor(add_tensor_6, view_742);  add_tensor_6 = view_742 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_default_37: "f32[768, 512]" = torch.ops.aten.permute.default(view_748, [1, 0]);  view_748 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_default_38: "f32[768, 512]" = torch.ops.aten.permute.default(view_751, [1, 0]);  view_751 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_39: "f32[768, 512]" = torch.ops.aten.permute.default(view_754, [1, 0]);  view_754 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_24: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(add_72, rsqrt_24);  add_72 = rsqrt_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_tensor_25: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(add_162, mul_tensor_24);  add_162 = mul_tensor_24 = None
        sum_dim_int_list_12: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_25, [0, 1], True);  mul_tensor_25 = None
        reshape_default_20: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_12, _shape_param_20);  sum_dim_int_list_12 = _shape_param_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        permute_default_40: "f32[768, 512]" = torch.ops.aten.permute.default(view_757, [1, 0]);  view_757 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        permute_default_41: "f32[3072, 512]" = torch.ops.aten.permute.default(view_759, [1, 0]);  view_759 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_26: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(add_70, rsqrt_23);  add_70 = rsqrt_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_tensor_27: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(view_760, mul_tensor_26);  view_760 = mul_tensor_26 = None
        sum_dim_int_list_13: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_27, [0, 1], True);  mul_tensor_27 = None
        reshape_default_21: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_13, _shape_param_21);  sum_dim_int_list_13 = _shape_param_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        permute_default_42: "f32[768, 512]" = torch.ops.aten.permute.default(view_762, [1, 0]);  view_762 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_default_43: "f32[768, 2048]" = torch.ops.aten.permute.default(view_776, [1, 0]);  view_776 = None
        reshape_default_22: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_207, _shape_param_22);  mm_207 = _shape_param_22 = None
        add_tensor_10: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_tensor_8, reshape_default_22);  add_tensor_8 = reshape_default_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_default_44: "f32[768, 2048]" = torch.ops.aten.permute.default(view_779, [1, 0]);  view_779 = None
        reshape_default_23: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_209, _shape_param_23);  mm_209 = _shape_param_23 = None
        add_tensor_11: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_tensor_10, reshape_default_23);  add_tensor_10 = reshape_default_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_45: "f32[768, 512]" = torch.ops.aten.permute.default(view_782, [1, 0]);  view_782 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_28: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(add_67, rsqrt_22);  add_67 = rsqrt_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_tensor_29: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(view_783, mul_tensor_28);  view_783 = mul_tensor_28 = None
        sum_dim_int_list_14: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_29, [0, 1], True);  mul_tensor_29 = None
        reshape_default_24: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_14, _shape_param_24);  sum_dim_int_list_14 = _shape_param_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        permute_default_46: "f32[768, 512]" = torch.ops.aten.permute.default(view_785, [1, 0]);  view_785 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_tensor_12: "f32[4, 12, 128, 128]" = torch.ops.aten.add.Tensor(add_tensor_9, view_793);  add_tensor_9 = view_793 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_default_47: "f32[768, 512]" = torch.ops.aten.permute.default(view_799, [1, 0]);  view_799 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_default_48: "f32[768, 512]" = torch.ops.aten.permute.default(view_802, [1, 0]);  view_802 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_49: "f32[768, 512]" = torch.ops.aten.permute.default(view_805, [1, 0]);  view_805 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_30: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(add_64, rsqrt_21);  add_64 = rsqrt_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_tensor_31: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(add_177, mul_tensor_30);  add_177 = mul_tensor_30 = None
        sum_dim_int_list_15: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_31, [0, 1], True);  mul_tensor_31 = None
        reshape_default_25: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_15, _shape_param_25);  sum_dim_int_list_15 = _shape_param_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        permute_default_50: "f32[768, 512]" = torch.ops.aten.permute.default(view_808, [1, 0]);  view_808 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        permute_default_51: "f32[3072, 512]" = torch.ops.aten.permute.default(view_810, [1, 0]);  view_810 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_32: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(add_62, rsqrt_20);  add_62 = rsqrt_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_tensor_33: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(view_811, mul_tensor_32);  view_811 = mul_tensor_32 = None
        sum_dim_int_list_16: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_33, [0, 1], True);  mul_tensor_33 = None
        reshape_default_26: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_16, _shape_param_26);  sum_dim_int_list_16 = _shape_param_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        permute_default_52: "f32[768, 512]" = torch.ops.aten.permute.default(view_813, [1, 0]);  view_813 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_default_53: "f32[768, 2048]" = torch.ops.aten.permute.default(view_827, [1, 0]);  view_827 = None
        reshape_default_27: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_227, _shape_param_27);  mm_227 = _shape_param_27 = None
        add_tensor_13: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_tensor_11, reshape_default_27);  add_tensor_11 = reshape_default_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_default_54: "f32[768, 2048]" = torch.ops.aten.permute.default(view_830, [1, 0]);  view_830 = None
        reshape_default_28: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_229, _shape_param_28);  mm_229 = _shape_param_28 = None
        add_tensor_14: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_tensor_13, reshape_default_28);  add_tensor_13 = reshape_default_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_55: "f32[768, 512]" = torch.ops.aten.permute.default(view_833, [1, 0]);  view_833 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_34: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(add_59, rsqrt_19);  add_59 = rsqrt_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_tensor_35: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(view_834, mul_tensor_34);  view_834 = mul_tensor_34 = None
        sum_dim_int_list_17: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_35, [0, 1], True);  mul_tensor_35 = None
        reshape_default_29: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_17, _shape_param_29);  sum_dim_int_list_17 = _shape_param_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        permute_default_56: "f32[768, 512]" = torch.ops.aten.permute.default(view_836, [1, 0]);  view_836 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_tensor_15: "f32[4, 12, 128, 128]" = torch.ops.aten.add.Tensor(add_tensor_12, view_844);  add_tensor_12 = view_844 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_default_57: "f32[768, 512]" = torch.ops.aten.permute.default(view_850, [1, 0]);  view_850 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_default_58: "f32[768, 512]" = torch.ops.aten.permute.default(view_853, [1, 0]);  view_853 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_59: "f32[768, 512]" = torch.ops.aten.permute.default(view_856, [1, 0]);  view_856 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_36: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(add_56, rsqrt_18);  add_56 = rsqrt_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_tensor_37: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(add_192, mul_tensor_36);  add_192 = mul_tensor_36 = None
        sum_dim_int_list_18: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_37, [0, 1], True);  mul_tensor_37 = None
        reshape_default_30: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_18, _shape_param_30);  sum_dim_int_list_18 = _shape_param_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        permute_default_60: "f32[768, 512]" = torch.ops.aten.permute.default(view_859, [1, 0]);  view_859 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        permute_default_61: "f32[3072, 512]" = torch.ops.aten.permute.default(view_861, [1, 0]);  view_861 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_38: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(add_54, rsqrt_17);  add_54 = rsqrt_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_tensor_39: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(view_862, mul_tensor_38);  view_862 = mul_tensor_38 = None
        sum_dim_int_list_19: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_39, [0, 1], True);  mul_tensor_39 = None
        reshape_default_31: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_19, _shape_param_31);  sum_dim_int_list_19 = _shape_param_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        permute_default_62: "f32[768, 512]" = torch.ops.aten.permute.default(view_864, [1, 0]);  view_864 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_default_63: "f32[768, 2048]" = torch.ops.aten.permute.default(view_878, [1, 0]);  view_878 = None
        reshape_default_32: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_247, _shape_param_32);  mm_247 = _shape_param_32 = None
        add_tensor_16: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_tensor_14, reshape_default_32);  add_tensor_14 = reshape_default_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_default_64: "f32[768, 2048]" = torch.ops.aten.permute.default(view_881, [1, 0]);  view_881 = None
        reshape_default_33: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_249, _shape_param_33);  mm_249 = _shape_param_33 = None
        add_tensor_17: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_tensor_16, reshape_default_33);  add_tensor_16 = reshape_default_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_65: "f32[768, 512]" = torch.ops.aten.permute.default(view_884, [1, 0]);  view_884 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_40: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(add_51, rsqrt_16);  add_51 = rsqrt_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_tensor_41: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(view_885, mul_tensor_40);  view_885 = mul_tensor_40 = None
        sum_dim_int_list_20: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_41, [0, 1], True);  mul_tensor_41 = None
        reshape_default_34: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_20, _shape_param_34);  sum_dim_int_list_20 = _shape_param_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        permute_default_66: "f32[768, 512]" = torch.ops.aten.permute.default(view_887, [1, 0]);  view_887 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_tensor_18: "f32[4, 12, 128, 128]" = torch.ops.aten.add.Tensor(add_tensor_15, view_895);  add_tensor_15 = view_895 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_default_67: "f32[768, 512]" = torch.ops.aten.permute.default(view_901, [1, 0]);  view_901 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_default_68: "f32[768, 512]" = torch.ops.aten.permute.default(view_904, [1, 0]);  view_904 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_69: "f32[768, 512]" = torch.ops.aten.permute.default(view_907, [1, 0]);  view_907 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_42: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(add_48, rsqrt_15);  add_48 = rsqrt_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_tensor_43: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(add_207, mul_tensor_42);  add_207 = mul_tensor_42 = None
        sum_dim_int_list_21: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_43, [0, 1], True);  mul_tensor_43 = None
        reshape_default_35: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_21, _shape_param_35);  sum_dim_int_list_21 = _shape_param_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        permute_default_70: "f32[768, 512]" = torch.ops.aten.permute.default(view_910, [1, 0]);  view_910 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        permute_default_71: "f32[3072, 512]" = torch.ops.aten.permute.default(view_912, [1, 0]);  view_912 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_44: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(add_46, rsqrt_14);  add_46 = rsqrt_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_tensor_45: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(view_913, mul_tensor_44);  view_913 = mul_tensor_44 = None
        sum_dim_int_list_22: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_45, [0, 1], True);  mul_tensor_45 = None
        reshape_default_36: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_22, _shape_param_36);  sum_dim_int_list_22 = _shape_param_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        permute_default_72: "f32[768, 512]" = torch.ops.aten.permute.default(view_915, [1, 0]);  view_915 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_default_73: "f32[768, 2048]" = torch.ops.aten.permute.default(view_929, [1, 0]);  view_929 = None
        reshape_default_37: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_267, _shape_param_37);  mm_267 = _shape_param_37 = None
        add_tensor_19: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_tensor_17, reshape_default_37);  add_tensor_17 = reshape_default_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_default_74: "f32[768, 2048]" = torch.ops.aten.permute.default(view_932, [1, 0]);  view_932 = None
        reshape_default_38: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_269, _shape_param_38);  mm_269 = _shape_param_38 = None
        add_tensor_20: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_tensor_19, reshape_default_38);  add_tensor_19 = reshape_default_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_75: "f32[768, 512]" = torch.ops.aten.permute.default(view_935, [1, 0]);  view_935 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_46: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(add_43, rsqrt_13);  add_43 = rsqrt_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_tensor_47: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(view_936, mul_tensor_46);  view_936 = mul_tensor_46 = None
        sum_dim_int_list_23: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_47, [0, 1], True);  mul_tensor_47 = None
        reshape_default_39: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_23, _shape_param_39);  sum_dim_int_list_23 = _shape_param_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        permute_default_76: "f32[768, 512]" = torch.ops.aten.permute.default(view_938, [1, 0]);  view_938 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_tensor_21: "f32[4, 12, 128, 128]" = torch.ops.aten.add.Tensor(add_tensor_18, view_946);  add_tensor_18 = view_946 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_default_77: "f32[768, 512]" = torch.ops.aten.permute.default(view_952, [1, 0]);  view_952 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_default_78: "f32[768, 512]" = torch.ops.aten.permute.default(view_955, [1, 0]);  view_955 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_79: "f32[768, 512]" = torch.ops.aten.permute.default(view_958, [1, 0]);  view_958 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_48: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(add_40, rsqrt_12);  add_40 = rsqrt_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_tensor_49: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(add_222, mul_tensor_48);  add_222 = mul_tensor_48 = None
        sum_dim_int_list_24: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_49, [0, 1], True);  mul_tensor_49 = None
        reshape_default_40: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_24, _shape_param_40);  sum_dim_int_list_24 = _shape_param_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        permute_default_80: "f32[768, 512]" = torch.ops.aten.permute.default(view_961, [1, 0]);  view_961 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        permute_default_81: "f32[3072, 512]" = torch.ops.aten.permute.default(view_963, [1, 0]);  view_963 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_50: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(add_38, rsqrt_11);  add_38 = rsqrt_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_tensor_51: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(view_964, mul_tensor_50);  view_964 = mul_tensor_50 = None
        sum_dim_int_list_25: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_51, [0, 1], True);  mul_tensor_51 = None
        reshape_default_41: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_25, _shape_param_41);  sum_dim_int_list_25 = _shape_param_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        permute_default_82: "f32[768, 512]" = torch.ops.aten.permute.default(view_966, [1, 0]);  view_966 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_default_83: "f32[768, 2048]" = torch.ops.aten.permute.default(view_980, [1, 0]);  view_980 = None
        reshape_default_42: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_287, _shape_param_42);  mm_287 = _shape_param_42 = None
        add_tensor_22: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_tensor_20, reshape_default_42);  add_tensor_20 = reshape_default_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_default_84: "f32[768, 2048]" = torch.ops.aten.permute.default(view_983, [1, 0]);  view_983 = None
        reshape_default_43: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_289, _shape_param_43);  mm_289 = _shape_param_43 = None
        add_tensor_23: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_tensor_22, reshape_default_43);  add_tensor_22 = reshape_default_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_85: "f32[768, 512]" = torch.ops.aten.permute.default(view_986, [1, 0]);  view_986 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_52: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(add_35, rsqrt_10);  add_35 = rsqrt_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_tensor_53: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(view_987, mul_tensor_52);  view_987 = mul_tensor_52 = None
        sum_dim_int_list_26: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_53, [0, 1], True);  mul_tensor_53 = None
        reshape_default_44: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_26, _shape_param_44);  sum_dim_int_list_26 = _shape_param_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        permute_default_86: "f32[768, 512]" = torch.ops.aten.permute.default(view_989, [1, 0]);  view_989 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_tensor_24: "f32[4, 12, 128, 128]" = torch.ops.aten.add.Tensor(add_tensor_21, view_997);  add_tensor_21 = view_997 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_default_87: "f32[768, 512]" = torch.ops.aten.permute.default(view_1003, [1, 0]);  view_1003 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_default_88: "f32[768, 512]" = torch.ops.aten.permute.default(view_1006, [1, 0]);  view_1006 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_89: "f32[768, 512]" = torch.ops.aten.permute.default(view_1009, [1, 0]);  view_1009 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_54: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(add_32, rsqrt_9);  add_32 = rsqrt_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_tensor_55: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(add_237, mul_tensor_54);  add_237 = mul_tensor_54 = None
        sum_dim_int_list_27: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_55, [0, 1], True);  mul_tensor_55 = None
        reshape_default_45: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_27, _shape_param_45);  sum_dim_int_list_27 = _shape_param_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        permute_default_90: "f32[768, 512]" = torch.ops.aten.permute.default(view_1012, [1, 0]);  view_1012 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        permute_default_91: "f32[3072, 512]" = torch.ops.aten.permute.default(view_1014, [1, 0]);  view_1014 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_56: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(add_30, rsqrt_8);  add_30 = rsqrt_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_tensor_57: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(view_1015, mul_tensor_56);  view_1015 = mul_tensor_56 = None
        sum_dim_int_list_28: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_57, [0, 1], True);  mul_tensor_57 = None
        reshape_default_46: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_28, _shape_param_46);  sum_dim_int_list_28 = _shape_param_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        permute_default_92: "f32[768, 512]" = torch.ops.aten.permute.default(view_1017, [1, 0]);  view_1017 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_default_93: "f32[768, 2048]" = torch.ops.aten.permute.default(view_1031, [1, 0]);  view_1031 = None
        reshape_default_47: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_307, _shape_param_47);  mm_307 = _shape_param_47 = None
        add_tensor_25: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_tensor_23, reshape_default_47);  add_tensor_23 = reshape_default_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_default_94: "f32[768, 2048]" = torch.ops.aten.permute.default(view_1034, [1, 0]);  view_1034 = None
        reshape_default_48: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_309, _shape_param_48);  mm_309 = _shape_param_48 = None
        add_tensor_26: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_tensor_25, reshape_default_48);  add_tensor_25 = reshape_default_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_95: "f32[768, 512]" = torch.ops.aten.permute.default(view_1037, [1, 0]);  view_1037 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_58: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(add_27, rsqrt_7);  add_27 = rsqrt_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_tensor_59: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(view_1038, mul_tensor_58);  view_1038 = mul_tensor_58 = None
        sum_dim_int_list_29: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_59, [0, 1], True);  mul_tensor_59 = None
        reshape_default_49: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_29, _shape_param_49);  sum_dim_int_list_29 = _shape_param_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        permute_default_96: "f32[768, 512]" = torch.ops.aten.permute.default(view_1040, [1, 0]);  view_1040 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_tensor_27: "f32[4, 12, 128, 128]" = torch.ops.aten.add.Tensor(add_tensor_24, view_1048);  add_tensor_24 = view_1048 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_default_97: "f32[768, 512]" = torch.ops.aten.permute.default(view_1054, [1, 0]);  view_1054 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_default_98: "f32[768, 512]" = torch.ops.aten.permute.default(view_1057, [1, 0]);  view_1057 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_99: "f32[768, 512]" = torch.ops.aten.permute.default(view_1060, [1, 0]);  view_1060 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_60: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(add_24, rsqrt_6);  add_24 = rsqrt_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_tensor_61: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(add_252, mul_tensor_60);  add_252 = mul_tensor_60 = None
        sum_dim_int_list_30: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_61, [0, 1], True);  mul_tensor_61 = None
        reshape_default_50: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_30, _shape_param_50);  sum_dim_int_list_30 = _shape_param_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        permute_default_100: "f32[768, 512]" = torch.ops.aten.permute.default(view_1063, [1, 0]);  view_1063 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        permute_default_101: "f32[3072, 512]" = torch.ops.aten.permute.default(view_1065, [1, 0]);  view_1065 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_62: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(add_22, rsqrt_5);  add_22 = rsqrt_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_tensor_63: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(view_1066, mul_tensor_62);  view_1066 = mul_tensor_62 = None
        sum_dim_int_list_31: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_63, [0, 1], True);  mul_tensor_63 = None
        reshape_default_51: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_31, _shape_param_51);  sum_dim_int_list_31 = _shape_param_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        permute_default_102: "f32[768, 512]" = torch.ops.aten.permute.default(view_1068, [1, 0]);  view_1068 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_default_103: "f32[768, 2048]" = torch.ops.aten.permute.default(view_1082, [1, 0]);  view_1082 = None
        reshape_default_52: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_327, _shape_param_52);  mm_327 = _shape_param_52 = None
        add_tensor_28: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_tensor_26, reshape_default_52);  add_tensor_26 = reshape_default_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_default_104: "f32[768, 2048]" = torch.ops.aten.permute.default(view_1085, [1, 0]);  view_1085 = None
        reshape_default_53: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_329, _shape_param_53);  mm_329 = _shape_param_53 = None
        add_tensor_29: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_tensor_28, reshape_default_53);  add_tensor_28 = reshape_default_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_105: "f32[768, 512]" = torch.ops.aten.permute.default(view_1088, [1, 0]);  view_1088 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_64: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(add_19, rsqrt_4);  add_19 = rsqrt_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_tensor_65: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(view_1089, mul_tensor_64);  view_1089 = mul_tensor_64 = None
        sum_dim_int_list_32: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_65, [0, 1], True);  mul_tensor_65 = None
        reshape_default_54: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_32, _shape_param_54);  sum_dim_int_list_32 = _shape_param_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        permute_default_106: "f32[768, 512]" = torch.ops.aten.permute.default(view_1091, [1, 0]);  view_1091 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_tensor_30: "f32[4, 12, 128, 128]" = torch.ops.aten.add.Tensor(add_tensor_27, view_1099);  add_tensor_27 = view_1099 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_default_107: "f32[768, 512]" = torch.ops.aten.permute.default(view_1105, [1, 0]);  view_1105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_default_108: "f32[768, 512]" = torch.ops.aten.permute.default(view_1108, [1, 0]);  view_1108 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_109: "f32[768, 512]" = torch.ops.aten.permute.default(view_1111, [1, 0]);  view_1111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_66: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(add_16, rsqrt_3);  add_16 = rsqrt_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_tensor_67: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(add_267, mul_tensor_66);  add_267 = mul_tensor_66 = None
        sum_dim_int_list_33: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_67, [0, 1], True);  mul_tensor_67 = None
        reshape_default_55: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_33, _shape_param_55);  sum_dim_int_list_33 = _shape_param_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        permute_default_110: "f32[768, 512]" = torch.ops.aten.permute.default(view_1114, [1, 0]);  view_1114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        permute_default_111: "f32[3072, 512]" = torch.ops.aten.permute.default(view_1116, [1, 0]);  view_1116 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_68: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(add_14, rsqrt_2);  add_14 = rsqrt_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_tensor_69: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(view_1117, mul_tensor_68);  view_1117 = mul_tensor_68 = None
        sum_dim_int_list_34: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_69, [0, 1], True);  mul_tensor_69 = None
        reshape_default_56: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_34, _shape_param_56);  sum_dim_int_list_34 = _shape_param_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        permute_default_112: "f32[768, 512]" = torch.ops.aten.permute.default(view_1119, [1, 0]);  view_1119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_default_113: "f32[768, 2048]" = torch.ops.aten.permute.default(view_1133, [1, 0]);  view_1133 = None
        reshape_default_57: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_347, _shape_param_57);  mm_347 = _shape_param_57 = None
        add_tensor_31: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_tensor_29, reshape_default_57);  add_tensor_29 = reshape_default_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_default_114: "f32[768, 2048]" = torch.ops.aten.permute.default(view_1136, [1, 0]);  view_1136 = None
        reshape_default_58: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_349, _shape_param_58);  mm_349 = _shape_param_58 = None
        add_tensor_32: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_tensor_31, reshape_default_58);  add_tensor_31 = reshape_default_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_115: "f32[768, 512]" = torch.ops.aten.permute.default(view_1139, [1, 0]);  view_1139 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_70: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(add_10, rsqrt_1);  add_10 = rsqrt_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_tensor_71: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(view_1140, mul_tensor_70);  view_1140 = mul_tensor_70 = None
        sum_dim_int_list_35: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_71, [0, 1], True);  mul_tensor_71 = None
        reshape_default_59: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_35, _shape_param_59);  sum_dim_int_list_35 = _shape_param_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        permute_default_116: "f32[768, 512]" = torch.ops.aten.permute.default(view_1142, [1, 0]);  view_1142 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_tensor_33: "f32[4, 12, 128, 128]" = torch.ops.aten.add.Tensor(add_tensor_30, view_1150);  add_tensor_30 = view_1150 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:312 in forward, code: position_bias = position_bias + causal_mask
        sum_dim_int_list_36: "f32[1, 12, 128, 128]" = torch.ops.aten.sum.dim_IntList(add_tensor_33, [0], True);  add_tensor_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:237 in compute_bias, code: values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
        squeeze_dim: "f32[12, 128, 128]" = torch.ops.aten.squeeze.dim(sum_dim_int_list_36, 0);  sum_dim_int_list_36 = None
        permute_default_117: "f32[128, 128, 12]" = torch.ops.aten.permute.default(squeeze_dim, [1, 2, 0]);  squeeze_dim = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:236 in compute_bias, code: values = self.relative_attention_bias(relative_position_bucket)  # shape (query_length, key_length, num_heads)
        eq_scalar: "b8[128, 128]" = torch.ops.aten.eq.Scalar(add_7, -1)
        unsqueeze_default: "b8[128, 128, 1]" = torch.ops.aten.unsqueeze.default(eq_scalar, -1);  eq_scalar = None
        where_self: "f32[128, 128, 12]" = torch.ops.aten.where.self(unsqueeze_default, full_default, permute_default_117);  unsqueeze_default = full_default = permute_default_117 = None
        clone_default: "f32[128, 128, 12]" = torch.ops.aten.clone.default(where_self, memory_format = torch.contiguous_format);  where_self = None
        full_default_1: "f32[32, 12]" = torch.ops.aten.full.default([32, 12], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default: "f32[32, 12]" = torch.ops.aten.index_put.default(full_default_1, [add_7], clone_default, True);  full_default_1 = add_7 = clone_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_default_118: "f32[768, 512]" = torch.ops.aten.permute.default(view_1156, [1, 0]);  view_1156 = None
        reshape_default_60: "f32[4, 128, 768]" = torch.ops.aten.reshape.default(mm_355, _shape_param_60);  mm_355 = _shape_param_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_default_119: "f32[768, 512]" = torch.ops.aten.permute.default(view_1159, [1, 0]);  view_1159 = None
        reshape_default_61: "f32[4, 128, 768]" = torch.ops.aten.reshape.default(mm_357, _shape_param_61);  mm_357 = _shape_param_61 = None
        add_tensor_34: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(reshape_default_60, reshape_default_61);  reshape_default_60 = reshape_default_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_120: "f32[768, 512]" = torch.ops.aten.permute.default(view_1162, [1, 0]);  view_1162 = None
        reshape_default_62: "f32[4, 128, 768]" = torch.ops.aten.reshape.default(mm_359, _shape_param_62);  mm_359 = _shape_param_62 = None
        add_tensor_35: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(add_tensor_34, reshape_default_62);  add_tensor_34 = reshape_default_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_tensor_72: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(add_tensor_35, primals_3);  primals_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:721 in torch_dynamo_resume_in_forward_at_681, code: hidden_states = self.dropout(inputs_embeds)
        mul_tensor_73: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(gt, primals_1);  primals_1 = None
        mul_tensor_74: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_73, 1.1111111111111112);  mul_tensor_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_75: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_74, rsqrt)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_tensor_76: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(add_tensor_35, mul_tensor_75);  add_tensor_35 = mul_tensor_75 = None
        sum_dim_int_list_37: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_76, [0, 1], True);  mul_tensor_76 = None
        reshape_default_63: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_37, _shape_param_63);  sum_dim_int_list_37 = _shape_param_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_77: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_72, mul_tensor_74)
        mul_tensor_78: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_72, rsqrt);  mul_tensor_72 = None
        sum_dim_int_list_38: "f32[4, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_77, [2], True);  mul_tensor_77 = None
        add_tensor_36: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(add_277, mul_tensor_78);  add_277 = mul_tensor_78 = None
        pow_tensor_scalar: "f32[4, 128, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt, 3);  rsqrt = None
        mul_scalar: "f32[4, 128, 1]" = torch.ops.aten.mul.Scalar(sum_dim_int_list_38, -0.5);  sum_dim_int_list_38 = None
        mul_tensor_79: "f32[4, 128, 1]" = torch.ops.aten.mul.Tensor(mul_scalar, pow_tensor_scalar);  mul_scalar = pow_tensor_scalar = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_default: "f32[4, 128, 768]" = torch.ops.aten.expand.default(mul_tensor_79, _shape_param_64);  mul_tensor_79 = _shape_param_64 = None
        div_scalar: "f32[4, 128, 768]" = torch.ops.aten.div.Scalar(expand_default, 768);  expand_default = None
        pow_tensor_scalar_1: "f32[4, 128, 768]" = torch.ops.aten.pow.Tensor_Scalar(mul_tensor_74, 1.0);  mul_tensor_74 = None
        mul_scalar_1: "f32[4, 128, 768]" = torch.ops.aten.mul.Scalar(pow_tensor_scalar_1, 2.0);  pow_tensor_scalar_1 = None
        mul_tensor_80: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(div_scalar, mul_scalar_1);  div_scalar = mul_scalar_1 = None
        add_tensor_37: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(add_tensor_36, mul_tensor_80);  add_tensor_36 = mul_tensor_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:721 in torch_dynamo_resume_in_forward_at_681, code: hidden_states = self.dropout(inputs_embeds)
        convert_element_type_default: "f32[4, 128, 768]" = torch.ops.prims.convert_element_type.default(gt, torch.float32);  gt = None
        mul_tensor_81: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 1.1111111111111112);  convert_element_type_default = None
        mul_tensor_82: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(add_tensor_37, mul_tensor_81);  add_tensor_37 = mul_tensor_81 = None
        return (reshape_default, permute_default, permute_default_1, reshape_default_1, permute_default_2, permute_default_3, permute_default_4, permute_default_5, reshape_default_4, permute_default_6, permute_default_7, permute_default_8, permute_default_9, reshape_default_5, permute_default_10, permute_default_11, reshape_default_6, permute_default_12, permute_default_13, permute_default_14, permute_default_15, reshape_default_9, permute_default_16, permute_default_17, permute_default_18, permute_default_19, reshape_default_10, permute_default_20, permute_default_21, reshape_default_11, permute_default_22, permute_default_23, permute_default_24, permute_default_25, reshape_default_14, permute_default_26, permute_default_27, permute_default_28, permute_default_29, reshape_default_15, permute_default_30, permute_default_31, reshape_default_16, permute_default_32, permute_default_33, permute_default_34, permute_default_35, reshape_default_19, permute_default_36, permute_default_37, permute_default_38, permute_default_39, reshape_default_20, permute_default_40, permute_default_41, reshape_default_21, permute_default_42, permute_default_43, permute_default_44, permute_default_45, reshape_default_24, permute_default_46, permute_default_47, permute_default_48, permute_default_49, reshape_default_25, permute_default_50, permute_default_51, reshape_default_26, permute_default_52, permute_default_53, permute_default_54, permute_default_55, reshape_default_29, permute_default_56, permute_default_57, permute_default_58, permute_default_59, reshape_default_30, permute_default_60, permute_default_61, reshape_default_31, permute_default_62, permute_default_63, permute_default_64, permute_default_65, reshape_default_34, permute_default_66, permute_default_67, permute_default_68, permute_default_69, reshape_default_35, permute_default_70, permute_default_71, reshape_default_36, permute_default_72, permute_default_73, permute_default_74, permute_default_75, reshape_default_39, permute_default_76, permute_default_77, permute_default_78, permute_default_79, reshape_default_40, permute_default_80, permute_default_81, reshape_default_41, permute_default_82, permute_default_83, permute_default_84, permute_default_85, reshape_default_44, permute_default_86, permute_default_87, permute_default_88, permute_default_89, reshape_default_45, permute_default_90, permute_default_91, reshape_default_46, permute_default_92, permute_default_93, permute_default_94, permute_default_95, reshape_default_49, permute_default_96, permute_default_97, permute_default_98, permute_default_99, reshape_default_50, permute_default_100, permute_default_101, reshape_default_51, permute_default_102, permute_default_103, permute_default_104, permute_default_105, reshape_default_54, permute_default_106, permute_default_107, permute_default_108, permute_default_109, reshape_default_55, permute_default_110, permute_default_111, reshape_default_56, permute_default_112, permute_default_113, permute_default_114, add_tensor_32, permute_default_115, reshape_default_59, permute_default_116, index_put_default, permute_default_118, permute_default_119, permute_default_120, reshape_default_63, mul_tensor_82)


def _default_make_inputs():
    return [
    torch.randn([4, 128, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 128, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 128, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_0
    torch.randn([512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([512, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([4, 128, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 128, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 128, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_1
    torch.randn([512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 768], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 768], dtype=torch.float32, device='cuda'),
    [4, 512, 768],  # _shape_param_2
    torch.randn([2048, 768], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 768], dtype=torch.float32, device='cuda'),
    [4, 512, 768],  # _shape_param_3
    torch.randn([512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 128, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 128, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 128, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_4
    torch.randn([512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 128, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 128, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 128, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_5
    torch.randn([512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([512, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([4, 128, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 128, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 128, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_6
    torch.randn([512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 768], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 768], dtype=torch.float32, device='cuda'),
    [4, 512, 768],  # _shape_param_7
    torch.randn([2048, 768], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 768], dtype=torch.float32, device='cuda'),
    [4, 512, 768],  # _shape_param_8
    torch.randn([512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 128, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 128, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 128, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_9
    torch.randn([512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 12, 128, 128], dtype=torch.float32, device='cuda'),
    torch.randn([4, 12, 128, 128], dtype=torch.float32, device='cuda'),
    torch.randn([512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 128, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 128, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 128, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_10
    torch.randn([512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([512, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([4, 128, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 128, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 128, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_11
    torch.randn([512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 768], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 768], dtype=torch.float32, device='cuda'),
    [4, 512, 768],  # _shape_param_12
    torch.randn([2048, 768], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 768], dtype=torch.float32, device='cuda'),
    [4, 512, 768],  # _shape_param_13
    torch.randn([512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 128, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 128, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 128, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_14
    torch.randn([512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 12, 128, 128], dtype=torch.float32, device='cuda'),
    torch.randn([512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 128, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 128, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 128, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_15
    torch.randn([512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([512, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([4, 128, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 128, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 128, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_16
    torch.randn([512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 768], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 768], dtype=torch.float32, device='cuda'),
    [4, 512, 768],  # _shape_param_17
    torch.randn([2048, 768], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 768], dtype=torch.float32, device='cuda'),
    [4, 512, 768],  # _shape_param_18
    torch.randn([512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 128, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 128, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 128, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_19
    torch.randn([512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 12, 128, 128], dtype=torch.float32, device='cuda'),
    torch.randn([512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 128, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 128, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 128, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_20
    torch.randn([512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([512, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([4, 128, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 128, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 128, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_21
    torch.randn([512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 768], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 768], dtype=torch.float32, device='cuda'),
    [4, 512, 768],  # _shape_param_22
    torch.randn([2048, 768], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 768], dtype=torch.float32, device='cuda'),
    [4, 512, 768],  # _shape_param_23
    torch.randn([512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 128, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 128, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 128, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_24
    torch.randn([512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 12, 128, 128], dtype=torch.float32, device='cuda'),
    torch.randn([512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 128, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 128, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 128, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_25
    torch.randn([512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([512, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([4, 128, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 128, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 128, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_26
    torch.randn([512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 768], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 768], dtype=torch.float32, device='cuda'),
    [4, 512, 768],  # _shape_param_27
    torch.randn([2048, 768], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 768], dtype=torch.float32, device='cuda'),
    [4, 512, 768],  # _shape_param_28
    torch.randn([512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 128, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 128, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 128, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_29
    torch.randn([512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 12, 128, 128], dtype=torch.float32, device='cuda'),
    torch.randn([512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 128, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 128, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 128, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_30
    torch.randn([512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([512, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([4, 128, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 128, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 128, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_31
    torch.randn([512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 768], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 768], dtype=torch.float32, device='cuda'),
    [4, 512, 768],  # _shape_param_32
    torch.randn([2048, 768], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 768], dtype=torch.float32, device='cuda'),
    [4, 512, 768],  # _shape_param_33
    torch.randn([512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 128, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 128, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 128, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_34
    torch.randn([512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 12, 128, 128], dtype=torch.float32, device='cuda'),
    torch.randn([512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 128, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 128, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 128, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_35
    torch.randn([512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([512, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([4, 128, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 128, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 128, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_36
    torch.randn([512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 768], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 768], dtype=torch.float32, device='cuda'),
    [4, 512, 768],  # _shape_param_37
    torch.randn([2048, 768], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 768], dtype=torch.float32, device='cuda'),
    [4, 512, 768],  # _shape_param_38
    torch.randn([512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 128, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 128, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 128, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_39
    torch.randn([512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 12, 128, 128], dtype=torch.float32, device='cuda'),
    torch.randn([512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 128, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 128, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 128, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_40
    torch.randn([512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([512, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([4, 128, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 128, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 128, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_41
    torch.randn([512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 768], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 768], dtype=torch.float32, device='cuda'),
    [4, 512, 768],  # _shape_param_42
    torch.randn([2048, 768], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 768], dtype=torch.float32, device='cuda'),
    [4, 512, 768],  # _shape_param_43
    torch.randn([512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 128, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 128, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 128, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_44
    torch.randn([512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 12, 128, 128], dtype=torch.float32, device='cuda'),
    torch.randn([512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 128, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 128, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 128, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_45
    torch.randn([512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([512, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([4, 128, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 128, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 128, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_46
    torch.randn([512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 768], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 768], dtype=torch.float32, device='cuda'),
    [4, 512, 768],  # _shape_param_47
    torch.randn([2048, 768], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 768], dtype=torch.float32, device='cuda'),
    [4, 512, 768],  # _shape_param_48
    torch.randn([512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 128, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 128, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 128, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_49
    torch.randn([512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 12, 128, 128], dtype=torch.float32, device='cuda'),
    torch.randn([512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 128, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 128, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 128, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_50
    torch.randn([512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([512, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([4, 128, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 128, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 128, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_51
    torch.randn([512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 768], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 768], dtype=torch.float32, device='cuda'),
    [4, 512, 768],  # _shape_param_52
    torch.randn([2048, 768], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 768], dtype=torch.float32, device='cuda'),
    [4, 512, 768],  # _shape_param_53
    torch.randn([512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 128, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 128, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 128, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_54
    torch.randn([512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 12, 128, 128], dtype=torch.float32, device='cuda'),
    torch.randn([512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 128, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 128, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 128, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_55
    torch.randn([512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([512, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([4, 128, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 128, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 128, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_56
    torch.randn([512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 768], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 768], dtype=torch.float32, device='cuda'),
    [4, 512, 768],  # _shape_param_57
    torch.randn([2048, 768], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 768], dtype=torch.float32, device='cuda'),
    [4, 512, 768],  # _shape_param_58
    torch.randn([512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 128, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 128, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 128, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_59
    torch.randn([512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 12, 128, 128], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [128, 128], dtype=torch.int64, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([512, 768], dtype=torch.float32, device='cuda'),
    [4, 128, 768],  # _shape_param_60
    torch.randn([512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([512, 768], dtype=torch.float32, device='cuda'),
    [4, 128, 768],  # _shape_param_61
    torch.randn([512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([512, 768], dtype=torch.float32, device='cuda'),
    [4, 128, 768],  # _shape_param_62
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [4, 128, 768], dtype=torch.bool, device='cuda'),
    torch.randn([4, 128, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 128, 1], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_63
    torch.randn([4, 128, 768], dtype=torch.float32, device='cuda'),
    [4, 128, 768],  # _shape_param_64
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
