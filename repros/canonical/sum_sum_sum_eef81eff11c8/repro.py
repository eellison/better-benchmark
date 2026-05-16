"""
Standalone repro captured via capture_hook.
Label: hf_T5ForConditionalGeneration_training
Pattern hash: eef81eff11c8
Shape hash: 0e025f9c
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, add_94: "f32[8, 1024, 512]", rsqrt_31: "f32[8, 1024, 1]", mul_200: "f32[8, 1024, 512]", _shape_param_0, view_436: "f32[8192, 512]", view_438: "f32[8192, 2048]", add_92: "f32[8, 1024, 512]", rsqrt_30: "f32[8, 1024, 1]", view_439: "f32[8, 1024, 512]", _shape_param_1, view_441: "f32[8192, 512]", view_455: "f32[8192, 512]", mul_79: "f32[8, 1024, 512]", _shape_param_2, view_458: "f32[8192, 512]", _shape_param_3, view_461: "f32[8192, 512]", add_89: "f32[8, 1024, 512]", rsqrt_29: "f32[8, 1024, 1]", view_462: "f32[8, 1024, 512]", _shape_param_4, view_464: "f32[8192, 512]", view_478: "f32[8192, 512]", view_481: "f32[8192, 512]", view_484: "f32[8192, 512]", add_86: "f32[8, 1024, 512]", rsqrt_28: "f32[8, 1024, 1]", add_105: "f32[8, 1024, 512]", _shape_param_5, view_487: "f32[8192, 512]", view_489: "f32[8192, 2048]", add_84: "f32[8, 1024, 512]", rsqrt_27: "f32[8, 1024, 1]", view_490: "f32[8, 1024, 512]", _shape_param_6, view_492: "f32[8192, 512]", view_506: "f32[8192, 512]", _shape_param_7, view_509: "f32[8192, 512]", _shape_param_8, view_512: "f32[8192, 512]", add_81: "f32[8, 1024, 512]", rsqrt_26: "f32[8, 1024, 1]", view_513: "f32[8, 1024, 512]", _shape_param_9, view_515: "f32[8192, 512]", view_472: "f32[8, 8, 1024, 1024]", view_523: "f32[8, 8, 1024, 1024]", view_529: "f32[8192, 512]", view_532: "f32[8192, 512]", view_535: "f32[8192, 512]", add_78: "f32[8, 1024, 512]", rsqrt_25: "f32[8, 1024, 1]", add_116: "f32[8, 1024, 512]", _shape_param_10, view_538: "f32[8192, 512]", view_540: "f32[8192, 2048]", add_76: "f32[8, 1024, 512]", rsqrt_24: "f32[8, 1024, 1]", view_541: "f32[8, 1024, 512]", _shape_param_11, view_543: "f32[8192, 512]", view_557: "f32[8192, 512]", _shape_param_12, view_560: "f32[8192, 512]", _shape_param_13, view_563: "f32[8192, 512]", add_73: "f32[8, 1024, 512]", rsqrt_23: "f32[8, 1024, 1]", view_564: "f32[8, 1024, 512]", _shape_param_14, view_566: "f32[8192, 512]", view_574: "f32[8, 8, 1024, 1024]", view_580: "f32[8192, 512]", view_583: "f32[8192, 512]", view_586: "f32[8192, 512]", add_70: "f32[8, 1024, 512]", rsqrt_22: "f32[8, 1024, 1]", add_127: "f32[8, 1024, 512]", _shape_param_15, view_589: "f32[8192, 512]", view_591: "f32[8192, 2048]", add_68: "f32[8, 1024, 512]", rsqrt_21: "f32[8, 1024, 1]", view_592: "f32[8, 1024, 512]", _shape_param_16, view_594: "f32[8192, 512]", view_608: "f32[8192, 512]", _shape_param_17, view_611: "f32[8192, 512]", _shape_param_18, view_614: "f32[8192, 512]", add_65: "f32[8, 1024, 512]", rsqrt_20: "f32[8, 1024, 1]", view_615: "f32[8, 1024, 512]", _shape_param_19, view_617: "f32[8192, 512]", view_625: "f32[8, 8, 1024, 1024]", view_631: "f32[8192, 512]", view_634: "f32[8192, 512]", view_637: "f32[8192, 512]", add_62: "f32[8, 1024, 512]", rsqrt_19: "f32[8, 1024, 1]", add_138: "f32[8, 1024, 512]", _shape_param_20, view_640: "f32[8192, 512]", view_642: "f32[8192, 2048]", add_60: "f32[8, 1024, 512]", rsqrt_18: "f32[8, 1024, 1]", view_643: "f32[8, 1024, 512]", _shape_param_21, view_645: "f32[8192, 512]", view_659: "f32[8192, 512]", _shape_param_22, view_662: "f32[8192, 512]", _shape_param_23, view_665: "f32[8192, 512]", add_57: "f32[8, 1024, 512]", rsqrt_17: "f32[8, 1024, 1]", view_666: "f32[8, 1024, 512]", _shape_param_24, view_668: "f32[8192, 512]", view_676: "f32[8, 8, 1024, 1024]", view_682: "f32[8192, 512]", view_685: "f32[8192, 512]", view_688: "f32[8192, 512]", add_54: "f32[8, 1024, 512]", rsqrt_16: "f32[8, 1024, 1]", add_149: "f32[8, 1024, 512]", _shape_param_25, view_691: "f32[8192, 512]", view_693: "f32[8192, 2048]", add_52: "f32[8, 1024, 512]", rsqrt_15: "f32[8, 1024, 1]", view_694: "f32[8, 1024, 512]", _shape_param_26, view_696: "f32[8192, 512]", view_710: "f32[8192, 512]", _shape_param_27, view_713: "f32[8192, 512]", _shape_param_28, view_716: "f32[8192, 512]", add_48: "f32[8, 1024, 512]", rsqrt_14: "f32[8, 1024, 1]", view_717: "f32[8, 1024, 512]", _shape_param_29, view_719: "f32[8192, 512]", view_727: "f32[8, 8, 1024, 1024]", add_45: "i64[1024, 1024]", full_default: "f32[]", view_733: "f32[8192, 512]", mm_214: "f32[8192, 512]", _shape_param_30, view_736: "f32[8192, 512]", mm_216: "f32[8192, 512]", _shape_param_31, view_739: "f32[8192, 512]", mm_218: "f32[8192, 512]", _shape_param_32, primals_54: "f32[512]", gt_27: "b8[8, 1024, 512]", embedding: "f32[8, 1024, 512]", rsqrt_13: "f32[8, 1024, 1]", _shape_param_33, add_157: "f32[8, 1024, 512]", _shape_param_34, primals_1: "i64[8, 1024]", mm_97: "f32[32128, 512]", add_36: "f32[8, 1024, 512]", rsqrt_12: "f32[8, 1024, 1]", mul_440: "f32[8, 1024, 512]", _shape_param_35, view_743: "f32[8192, 512]", view_745: "f32[8192, 2048]", add_34: "f32[8, 1024, 512]", rsqrt_11: "f32[8, 1024, 1]", view_746: "f32[8, 1024, 512]", _shape_param_36, view_748: "f32[8192, 512]", view_762: "f32[8192, 512]", view_765: "f32[8192, 512]", view_768: "f32[8192, 512]", add_31: "f32[8, 1024, 512]", rsqrt_10: "f32[8, 1024, 1]", add_168: "f32[8, 1024, 512]", _shape_param_37, view_771: "f32[8192, 512]", view_773: "f32[8192, 2048]", add_29: "f32[8, 1024, 512]", rsqrt_9: "f32[8, 1024, 1]", view_774: "f32[8, 1024, 512]", _shape_param_38, view_776: "f32[8192, 512]", view_756: "f32[8, 8, 1024, 1024]", view_784: "f32[8, 8, 1024, 1024]", view_790: "f32[8192, 512]", view_793: "f32[8192, 512]", view_796: "f32[8192, 512]", add_26: "f32[8, 1024, 512]", rsqrt_8: "f32[8, 1024, 1]", add_175: "f32[8, 1024, 512]", _shape_param_39, view_799: "f32[8192, 512]", view_801: "f32[8192, 2048]", add_24: "f32[8, 1024, 512]", rsqrt_7: "f32[8, 1024, 1]", view_802: "f32[8, 1024, 512]", _shape_param_40, view_804: "f32[8192, 512]", view_812: "f32[8, 8, 1024, 1024]", view_818: "f32[8192, 512]", view_821: "f32[8192, 512]", view_824: "f32[8192, 512]", add_21: "f32[8, 1024, 512]", rsqrt_6: "f32[8, 1024, 1]", add_182: "f32[8, 1024, 512]", _shape_param_41, view_827: "f32[8192, 512]", view_829: "f32[8192, 2048]", add_19: "f32[8, 1024, 512]", rsqrt_5: "f32[8, 1024, 1]", view_830: "f32[8, 1024, 512]", _shape_param_42, view_832: "f32[8192, 512]", view_840: "f32[8, 8, 1024, 1024]", view_846: "f32[8192, 512]", view_849: "f32[8192, 512]", view_852: "f32[8192, 512]", add_16: "f32[8, 1024, 512]", rsqrt_4: "f32[8, 1024, 1]", add_189: "f32[8, 1024, 512]", _shape_param_43, view_855: "f32[8192, 512]", view_857: "f32[8192, 2048]", add_14: "f32[8, 1024, 512]", rsqrt_3: "f32[8, 1024, 1]", view_858: "f32[8, 1024, 512]", _shape_param_44, view_860: "f32[8192, 512]", view_868: "f32[8, 8, 1024, 1024]", view_874: "f32[8192, 512]", view_877: "f32[8192, 512]", view_880: "f32[8192, 512]", add_11: "f32[8, 1024, 512]", rsqrt_2: "f32[8, 1024, 1]", add_196: "f32[8, 1024, 512]", _shape_param_45, view_883: "f32[8192, 512]", view_885: "f32[8192, 2048]", add_9: "f32[8, 1024, 512]", rsqrt_1: "f32[8, 1024, 1]", view_886: "f32[8, 1024, 512]", _shape_param_46, view_888: "f32[8192, 512]", view_896: "f32[8, 8, 1024, 1024]", add_6: "i64[1024, 1024]", view_902: "f32[8192, 512]", mm_286: "f32[8192, 512]", _shape_param_47, view_905: "f32[8192, 512]", mm_288: "f32[8192, 512]", _shape_param_48, view_908: "f32[8192, 512]", mm_290: "f32[8192, 512]", _shape_param_49, primals_3: "f32[512]", gt: "b8[8, 1024, 512]", rsqrt: "f32[8, 1024, 1]", _shape_param_50, add_200: "f32[8, 1024, 512]", _shape_param_51):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_94, rsqrt_31);  add_94 = rsqrt_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_tensor_1: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_200, mul_tensor);  mul_200 = mul_tensor = None
        sum_dim_int_list: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_1, [0, 1], True);  mul_tensor_1 = None
        reshape_default: "f32[512]" = torch.ops.aten.reshape.default(sum_dim_int_list, _shape_param_0);  sum_dim_int_list = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        permute_default: "f32[512, 8192]" = torch.ops.aten.permute.default(view_436, [1, 0]);  view_436 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        permute_default_1: "f32[2048, 8192]" = torch.ops.aten.permute.default(view_438, [1, 0]);  view_438 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_2: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_92, rsqrt_30);  add_92 = rsqrt_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_tensor_3: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(view_439, mul_tensor_2);  view_439 = mul_tensor_2 = None
        sum_dim_int_list_1: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_3, [0, 1], True);  mul_tensor_3 = None
        reshape_default_1: "f32[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_1, _shape_param_1);  sum_dim_int_list_1 = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        permute_default_2: "f32[512, 8192]" = torch.ops.aten.permute.default(view_441, [1, 0]);  view_441 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_default_3: "f32[512, 8192]" = torch.ops.aten.permute.default(view_455, [1, 0]);  view_455 = None
        reshape_default_2: "f32[8192, 512]" = torch.ops.aten.reshape.default(mul_79, _shape_param_2);  _shape_param_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_default_4: "f32[512, 8192]" = torch.ops.aten.permute.default(view_458, [1, 0]);  view_458 = None
        reshape_default_3: "f32[8192, 512]" = torch.ops.aten.reshape.default(mul_79, _shape_param_3);  _shape_param_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_5: "f32[512, 8192]" = torch.ops.aten.permute.default(view_461, [1, 0]);  view_461 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_4: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_89, rsqrt_29);  add_89 = rsqrt_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_tensor_5: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(view_462, mul_tensor_4);  view_462 = mul_tensor_4 = None
        sum_dim_int_list_2: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_5, [0, 1], True);  mul_tensor_5 = None
        reshape_default_4: "f32[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_2, _shape_param_4);  sum_dim_int_list_2 = _shape_param_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        permute_default_6: "f32[512, 8192]" = torch.ops.aten.permute.default(view_464, [1, 0]);  view_464 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_default_7: "f32[512, 8192]" = torch.ops.aten.permute.default(view_478, [1, 0]);  view_478 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_default_8: "f32[512, 8192]" = torch.ops.aten.permute.default(view_481, [1, 0]);  view_481 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_9: "f32[512, 8192]" = torch.ops.aten.permute.default(view_484, [1, 0]);  view_484 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_6: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_86, rsqrt_28);  add_86 = rsqrt_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_tensor_7: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_105, mul_tensor_6);  add_105 = mul_tensor_6 = None
        sum_dim_int_list_3: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_7, [0, 1], True);  mul_tensor_7 = None
        reshape_default_5: "f32[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_3, _shape_param_5);  sum_dim_int_list_3 = _shape_param_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        permute_default_10: "f32[512, 8192]" = torch.ops.aten.permute.default(view_487, [1, 0]);  view_487 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        permute_default_11: "f32[2048, 8192]" = torch.ops.aten.permute.default(view_489, [1, 0]);  view_489 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_8: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_84, rsqrt_27);  add_84 = rsqrt_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_tensor_9: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(view_490, mul_tensor_8);  view_490 = mul_tensor_8 = None
        sum_dim_int_list_4: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_9, [0, 1], True);  mul_tensor_9 = None
        reshape_default_6: "f32[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_4, _shape_param_6);  sum_dim_int_list_4 = _shape_param_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        permute_default_12: "f32[512, 8192]" = torch.ops.aten.permute.default(view_492, [1, 0]);  view_492 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_default_13: "f32[512, 8192]" = torch.ops.aten.permute.default(view_506, [1, 0]);  view_506 = None
        reshape_default_7: "f32[8192, 512]" = torch.ops.aten.reshape.default(mul_79, _shape_param_7);  _shape_param_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_default_14: "f32[512, 8192]" = torch.ops.aten.permute.default(view_509, [1, 0]);  view_509 = None
        reshape_default_8: "f32[8192, 512]" = torch.ops.aten.reshape.default(mul_79, _shape_param_8);  _shape_param_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_15: "f32[512, 8192]" = torch.ops.aten.permute.default(view_512, [1, 0]);  view_512 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_10: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_81, rsqrt_26);  add_81 = rsqrt_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_tensor_11: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(view_513, mul_tensor_10);  view_513 = mul_tensor_10 = None
        sum_dim_int_list_5: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_11, [0, 1], True);  mul_tensor_11 = None
        reshape_default_9: "f32[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_5, _shape_param_9);  sum_dim_int_list_5 = _shape_param_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        permute_default_16: "f32[512, 8192]" = torch.ops.aten.permute.default(view_515, [1, 0]);  view_515 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_tensor: "f32[8, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(view_472, view_523);  view_472 = view_523 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_default_17: "f32[512, 8192]" = torch.ops.aten.permute.default(view_529, [1, 0]);  view_529 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_default_18: "f32[512, 8192]" = torch.ops.aten.permute.default(view_532, [1, 0]);  view_532 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_19: "f32[512, 8192]" = torch.ops.aten.permute.default(view_535, [1, 0]);  view_535 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_12: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_78, rsqrt_25);  add_78 = rsqrt_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_tensor_13: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_116, mul_tensor_12);  add_116 = mul_tensor_12 = None
        sum_dim_int_list_6: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_13, [0, 1], True);  mul_tensor_13 = None
        reshape_default_10: "f32[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_6, _shape_param_10);  sum_dim_int_list_6 = _shape_param_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        permute_default_20: "f32[512, 8192]" = torch.ops.aten.permute.default(view_538, [1, 0]);  view_538 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        permute_default_21: "f32[2048, 8192]" = torch.ops.aten.permute.default(view_540, [1, 0]);  view_540 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_14: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_76, rsqrt_24);  add_76 = rsqrt_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_tensor_15: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(view_541, mul_tensor_14);  view_541 = mul_tensor_14 = None
        sum_dim_int_list_7: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_15, [0, 1], True);  mul_tensor_15 = None
        reshape_default_11: "f32[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_7, _shape_param_11);  sum_dim_int_list_7 = _shape_param_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        permute_default_22: "f32[512, 8192]" = torch.ops.aten.permute.default(view_543, [1, 0]);  view_543 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_default_23: "f32[512, 8192]" = torch.ops.aten.permute.default(view_557, [1, 0]);  view_557 = None
        reshape_default_12: "f32[8192, 512]" = torch.ops.aten.reshape.default(mul_79, _shape_param_12);  _shape_param_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_default_24: "f32[512, 8192]" = torch.ops.aten.permute.default(view_560, [1, 0]);  view_560 = None
        reshape_default_13: "f32[8192, 512]" = torch.ops.aten.reshape.default(mul_79, _shape_param_13);  _shape_param_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_25: "f32[512, 8192]" = torch.ops.aten.permute.default(view_563, [1, 0]);  view_563 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_16: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_73, rsqrt_23);  add_73 = rsqrt_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_tensor_17: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(view_564, mul_tensor_16);  view_564 = mul_tensor_16 = None
        sum_dim_int_list_8: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_17, [0, 1], True);  mul_tensor_17 = None
        reshape_default_14: "f32[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_8, _shape_param_14);  sum_dim_int_list_8 = _shape_param_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        permute_default_26: "f32[512, 8192]" = torch.ops.aten.permute.default(view_566, [1, 0]);  view_566 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_tensor_1: "f32[8, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(add_tensor, view_574);  add_tensor = view_574 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_default_27: "f32[512, 8192]" = torch.ops.aten.permute.default(view_580, [1, 0]);  view_580 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_default_28: "f32[512, 8192]" = torch.ops.aten.permute.default(view_583, [1, 0]);  view_583 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_29: "f32[512, 8192]" = torch.ops.aten.permute.default(view_586, [1, 0]);  view_586 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_18: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_70, rsqrt_22);  add_70 = rsqrt_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_tensor_19: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_127, mul_tensor_18);  add_127 = mul_tensor_18 = None
        sum_dim_int_list_9: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_19, [0, 1], True);  mul_tensor_19 = None
        reshape_default_15: "f32[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_9, _shape_param_15);  sum_dim_int_list_9 = _shape_param_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        permute_default_30: "f32[512, 8192]" = torch.ops.aten.permute.default(view_589, [1, 0]);  view_589 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        permute_default_31: "f32[2048, 8192]" = torch.ops.aten.permute.default(view_591, [1, 0]);  view_591 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_20: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_68, rsqrt_21);  add_68 = rsqrt_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_tensor_21: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(view_592, mul_tensor_20);  view_592 = mul_tensor_20 = None
        sum_dim_int_list_10: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_21, [0, 1], True);  mul_tensor_21 = None
        reshape_default_16: "f32[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_10, _shape_param_16);  sum_dim_int_list_10 = _shape_param_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        permute_default_32: "f32[512, 8192]" = torch.ops.aten.permute.default(view_594, [1, 0]);  view_594 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_default_33: "f32[512, 8192]" = torch.ops.aten.permute.default(view_608, [1, 0]);  view_608 = None
        reshape_default_17: "f32[8192, 512]" = torch.ops.aten.reshape.default(mul_79, _shape_param_17);  _shape_param_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_default_34: "f32[512, 8192]" = torch.ops.aten.permute.default(view_611, [1, 0]);  view_611 = None
        reshape_default_18: "f32[8192, 512]" = torch.ops.aten.reshape.default(mul_79, _shape_param_18);  _shape_param_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_35: "f32[512, 8192]" = torch.ops.aten.permute.default(view_614, [1, 0]);  view_614 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_22: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_65, rsqrt_20);  add_65 = rsqrt_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_tensor_23: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(view_615, mul_tensor_22);  view_615 = mul_tensor_22 = None
        sum_dim_int_list_11: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_23, [0, 1], True);  mul_tensor_23 = None
        reshape_default_19: "f32[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_11, _shape_param_19);  sum_dim_int_list_11 = _shape_param_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        permute_default_36: "f32[512, 8192]" = torch.ops.aten.permute.default(view_617, [1, 0]);  view_617 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_tensor_2: "f32[8, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(add_tensor_1, view_625);  add_tensor_1 = view_625 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_default_37: "f32[512, 8192]" = torch.ops.aten.permute.default(view_631, [1, 0]);  view_631 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_default_38: "f32[512, 8192]" = torch.ops.aten.permute.default(view_634, [1, 0]);  view_634 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_39: "f32[512, 8192]" = torch.ops.aten.permute.default(view_637, [1, 0]);  view_637 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_24: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_62, rsqrt_19);  add_62 = rsqrt_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_tensor_25: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_138, mul_tensor_24);  add_138 = mul_tensor_24 = None
        sum_dim_int_list_12: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_25, [0, 1], True);  mul_tensor_25 = None
        reshape_default_20: "f32[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_12, _shape_param_20);  sum_dim_int_list_12 = _shape_param_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        permute_default_40: "f32[512, 8192]" = torch.ops.aten.permute.default(view_640, [1, 0]);  view_640 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        permute_default_41: "f32[2048, 8192]" = torch.ops.aten.permute.default(view_642, [1, 0]);  view_642 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_26: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_60, rsqrt_18);  add_60 = rsqrt_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_tensor_27: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(view_643, mul_tensor_26);  view_643 = mul_tensor_26 = None
        sum_dim_int_list_13: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_27, [0, 1], True);  mul_tensor_27 = None
        reshape_default_21: "f32[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_13, _shape_param_21);  sum_dim_int_list_13 = _shape_param_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        permute_default_42: "f32[512, 8192]" = torch.ops.aten.permute.default(view_645, [1, 0]);  view_645 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_default_43: "f32[512, 8192]" = torch.ops.aten.permute.default(view_659, [1, 0]);  view_659 = None
        reshape_default_22: "f32[8192, 512]" = torch.ops.aten.reshape.default(mul_79, _shape_param_22);  _shape_param_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_default_44: "f32[512, 8192]" = torch.ops.aten.permute.default(view_662, [1, 0]);  view_662 = None
        reshape_default_23: "f32[8192, 512]" = torch.ops.aten.reshape.default(mul_79, _shape_param_23);  _shape_param_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_45: "f32[512, 8192]" = torch.ops.aten.permute.default(view_665, [1, 0]);  view_665 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_28: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_57, rsqrt_17);  add_57 = rsqrt_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_tensor_29: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(view_666, mul_tensor_28);  view_666 = mul_tensor_28 = None
        sum_dim_int_list_14: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_29, [0, 1], True);  mul_tensor_29 = None
        reshape_default_24: "f32[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_14, _shape_param_24);  sum_dim_int_list_14 = _shape_param_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        permute_default_46: "f32[512, 8192]" = torch.ops.aten.permute.default(view_668, [1, 0]);  view_668 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_tensor_3: "f32[8, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(add_tensor_2, view_676);  add_tensor_2 = view_676 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_default_47: "f32[512, 8192]" = torch.ops.aten.permute.default(view_682, [1, 0]);  view_682 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_default_48: "f32[512, 8192]" = torch.ops.aten.permute.default(view_685, [1, 0]);  view_685 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_49: "f32[512, 8192]" = torch.ops.aten.permute.default(view_688, [1, 0]);  view_688 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_30: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_54, rsqrt_16);  add_54 = rsqrt_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_tensor_31: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_149, mul_tensor_30);  add_149 = mul_tensor_30 = None
        sum_dim_int_list_15: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_31, [0, 1], True);  mul_tensor_31 = None
        reshape_default_25: "f32[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_15, _shape_param_25);  sum_dim_int_list_15 = _shape_param_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        permute_default_50: "f32[512, 8192]" = torch.ops.aten.permute.default(view_691, [1, 0]);  view_691 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        permute_default_51: "f32[2048, 8192]" = torch.ops.aten.permute.default(view_693, [1, 0]);  view_693 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_32: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_52, rsqrt_15);  add_52 = rsqrt_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_tensor_33: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(view_694, mul_tensor_32);  view_694 = mul_tensor_32 = None
        sum_dim_int_list_16: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_33, [0, 1], True);  mul_tensor_33 = None
        reshape_default_26: "f32[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_16, _shape_param_26);  sum_dim_int_list_16 = _shape_param_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        permute_default_52: "f32[512, 8192]" = torch.ops.aten.permute.default(view_696, [1, 0]);  view_696 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_default_53: "f32[512, 8192]" = torch.ops.aten.permute.default(view_710, [1, 0]);  view_710 = None
        reshape_default_27: "f32[8192, 512]" = torch.ops.aten.reshape.default(mul_79, _shape_param_27);  _shape_param_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_default_54: "f32[512, 8192]" = torch.ops.aten.permute.default(view_713, [1, 0]);  view_713 = None
        reshape_default_28: "f32[8192, 512]" = torch.ops.aten.reshape.default(mul_79, _shape_param_28);  mul_79 = _shape_param_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_55: "f32[512, 8192]" = torch.ops.aten.permute.default(view_716, [1, 0]);  view_716 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_34: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_48, rsqrt_14);  add_48 = rsqrt_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_tensor_35: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(view_717, mul_tensor_34);  view_717 = mul_tensor_34 = None
        sum_dim_int_list_17: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_35, [0, 1], True);  mul_tensor_35 = None
        reshape_default_29: "f32[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_17, _shape_param_29);  sum_dim_int_list_17 = _shape_param_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        permute_default_56: "f32[512, 8192]" = torch.ops.aten.permute.default(view_719, [1, 0]);  view_719 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_tensor_4: "f32[8, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(add_tensor_3, view_727);  add_tensor_3 = view_727 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:312 in forward, code: position_bias = position_bias + causal_mask
        sum_dim_int_list_18: "f32[1, 8, 1024, 1024]" = torch.ops.aten.sum.dim_IntList(add_tensor_4, [0], True);  add_tensor_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:237 in compute_bias, code: values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
        squeeze_dim: "f32[8, 1024, 1024]" = torch.ops.aten.squeeze.dim(sum_dim_int_list_18, 0);  sum_dim_int_list_18 = None
        permute_default_57: "f32[1024, 1024, 8]" = torch.ops.aten.permute.default(squeeze_dim, [1, 2, 0]);  squeeze_dim = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:236 in compute_bias, code: values = self.relative_attention_bias(relative_position_bucket)  # shape (query_length, key_length, num_heads)
        eq_scalar: "b8[1024, 1024]" = torch.ops.aten.eq.Scalar(add_45, -1)
        unsqueeze_default: "b8[1024, 1024, 1]" = torch.ops.aten.unsqueeze.default(eq_scalar, -1);  eq_scalar = None
        where_self: "f32[1024, 1024, 8]" = torch.ops.aten.where.self(unsqueeze_default, full_default, permute_default_57);  unsqueeze_default = permute_default_57 = None
        clone_default: "f32[1024, 1024, 8]" = torch.ops.aten.clone.default(where_self, memory_format = torch.contiguous_format);  where_self = None
        full_default_1: "f32[32, 8]" = torch.ops.aten.full.default([32, 8], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default: "f32[32, 8]" = torch.ops.aten.index_put.default(full_default_1, [add_45], clone_default, True);  add_45 = clone_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_default_58: "f32[512, 8192]" = torch.ops.aten.permute.default(view_733, [1, 0]);  view_733 = None
        reshape_default_30: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_214, _shape_param_30);  mm_214 = _shape_param_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_default_59: "f32[512, 8192]" = torch.ops.aten.permute.default(view_736, [1, 0]);  view_736 = None
        reshape_default_31: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_216, _shape_param_31);  mm_216 = _shape_param_31 = None
        add_tensor_5: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(reshape_default_30, reshape_default_31);  reshape_default_30 = reshape_default_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_60: "f32[512, 8192]" = torch.ops.aten.permute.default(view_739, [1, 0]);  view_739 = None
        reshape_default_32: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_218, _shape_param_32);  mm_218 = _shape_param_32 = None
        add_tensor_6: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_tensor_5, reshape_default_32);  add_tensor_5 = reshape_default_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_tensor_36: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_tensor_6, primals_54);  primals_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:721 in forward, code: hidden_states = self.dropout(inputs_embeds)
        mul_tensor_37: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(gt_27, embedding)
        mul_tensor_38: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_tensor_37, 1.1111111111111112);  mul_tensor_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_39: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_tensor_38, rsqrt_13)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_tensor_40: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_tensor_6, mul_tensor_39);  add_tensor_6 = mul_tensor_39 = None
        sum_dim_int_list_19: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_40, [0, 1], True);  mul_tensor_40 = None
        reshape_default_33: "f32[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_19, _shape_param_33);  sum_dim_int_list_19 = _shape_param_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_41: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_tensor_36, mul_tensor_38)
        mul_tensor_42: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_tensor_36, rsqrt_13);  mul_tensor_36 = None
        sum_dim_int_list_20: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_41, [2], True);  mul_tensor_41 = None
        add_tensor_7: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_157, mul_tensor_42);  add_157 = mul_tensor_42 = None
        pow_tensor_scalar: "f32[8, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_13, 3);  rsqrt_13 = None
        mul_scalar: "f32[8, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_dim_int_list_20, -0.5);  sum_dim_int_list_20 = None
        mul_tensor_43: "f32[8, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_scalar, pow_tensor_scalar);  mul_scalar = pow_tensor_scalar = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_default: "f32[8, 1024, 512]" = torch.ops.aten.expand.default(mul_tensor_43, _shape_param_34);  mul_tensor_43 = _shape_param_34 = None
        div_scalar: "f32[8, 1024, 512]" = torch.ops.aten.div.Scalar(expand_default, 512);  expand_default = None
        pow_tensor_scalar_1: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(mul_tensor_38, 1.0);  mul_tensor_38 = None
        mul_scalar_1: "f32[8, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_tensor_scalar_1, 2.0);  pow_tensor_scalar_1 = None
        mul_tensor_44: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(div_scalar, mul_scalar_1);  div_scalar = mul_scalar_1 = None
        add_tensor_8: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_tensor_7, mul_tensor_44);  add_tensor_7 = mul_tensor_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:721 in forward, code: hidden_states = self.dropout(inputs_embeds)
        convert_element_type_default: "f32[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(gt_27, torch.float32);  gt_27 = None
        mul_tensor_45: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 1.1111111111111112);  convert_element_type_default = None
        mul_tensor_46: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_tensor_8, mul_tensor_45);  add_tensor_8 = mul_tensor_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:669 in forward, code: inputs_embeds = self.embed_tokens(input_ids)
        eq_scalar_1: "b8[8, 1024]" = torch.ops.aten.eq.Scalar(primals_1, -1)
        unsqueeze_default_1: "b8[8, 1024, 1]" = torch.ops.aten.unsqueeze.default(eq_scalar_1, -1);  eq_scalar_1 = None
        where_self_1: "f32[8, 1024, 512]" = torch.ops.aten.where.self(unsqueeze_default_1, full_default, mul_tensor_46);  mul_tensor_46 = None
        full_default_2: "f32[32128, 512]" = torch.ops.aten.full.default([32128, 512], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default_1: "f32[32128, 512]" = torch.ops.aten.index_put.default(full_default_2, [primals_1], where_self_1, True);  where_self_1 = None
        add_tensor_9: "f32[32128, 512]" = torch.ops.aten.add.Tensor(mm_97, index_put_default_1);  mm_97 = index_put_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_47: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_36, rsqrt_12);  add_36 = rsqrt_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_tensor_48: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_440, mul_tensor_47);  mul_440 = mul_tensor_47 = None
        sum_dim_int_list_21: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_48, [0, 1], True);  mul_tensor_48 = None
        reshape_default_34: "f32[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_21, _shape_param_35);  sum_dim_int_list_21 = _shape_param_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        permute_default_61: "f32[512, 8192]" = torch.ops.aten.permute.default(view_743, [1, 0]);  view_743 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        permute_default_62: "f32[2048, 8192]" = torch.ops.aten.permute.default(view_745, [1, 0]);  view_745 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_49: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_34, rsqrt_11);  add_34 = rsqrt_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_tensor_50: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(view_746, mul_tensor_49);  view_746 = mul_tensor_49 = None
        sum_dim_int_list_22: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_50, [0, 1], True);  mul_tensor_50 = None
        reshape_default_35: "f32[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_22, _shape_param_36);  sum_dim_int_list_22 = _shape_param_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        permute_default_63: "f32[512, 8192]" = torch.ops.aten.permute.default(view_748, [1, 0]);  view_748 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_default_64: "f32[512, 8192]" = torch.ops.aten.permute.default(view_762, [1, 0]);  view_762 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_default_65: "f32[512, 8192]" = torch.ops.aten.permute.default(view_765, [1, 0]);  view_765 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_66: "f32[512, 8192]" = torch.ops.aten.permute.default(view_768, [1, 0]);  view_768 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_51: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_31, rsqrt_10);  add_31 = rsqrt_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_tensor_52: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_168, mul_tensor_51);  add_168 = mul_tensor_51 = None
        sum_dim_int_list_23: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_52, [0, 1], True);  mul_tensor_52 = None
        reshape_default_36: "f32[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_23, _shape_param_37);  sum_dim_int_list_23 = _shape_param_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        permute_default_67: "f32[512, 8192]" = torch.ops.aten.permute.default(view_771, [1, 0]);  view_771 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        permute_default_68: "f32[2048, 8192]" = torch.ops.aten.permute.default(view_773, [1, 0]);  view_773 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_53: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_29, rsqrt_9);  add_29 = rsqrt_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_tensor_54: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(view_774, mul_tensor_53);  view_774 = mul_tensor_53 = None
        sum_dim_int_list_24: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_54, [0, 1], True);  mul_tensor_54 = None
        reshape_default_37: "f32[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_24, _shape_param_38);  sum_dim_int_list_24 = _shape_param_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        permute_default_69: "f32[512, 8192]" = torch.ops.aten.permute.default(view_776, [1, 0]);  view_776 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_tensor_10: "f32[8, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(view_756, view_784);  view_756 = view_784 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_default_70: "f32[512, 8192]" = torch.ops.aten.permute.default(view_790, [1, 0]);  view_790 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_default_71: "f32[512, 8192]" = torch.ops.aten.permute.default(view_793, [1, 0]);  view_793 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_72: "f32[512, 8192]" = torch.ops.aten.permute.default(view_796, [1, 0]);  view_796 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_55: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_26, rsqrt_8);  add_26 = rsqrt_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_tensor_56: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_175, mul_tensor_55);  add_175 = mul_tensor_55 = None
        sum_dim_int_list_25: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_56, [0, 1], True);  mul_tensor_56 = None
        reshape_default_38: "f32[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_25, _shape_param_39);  sum_dim_int_list_25 = _shape_param_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        permute_default_73: "f32[512, 8192]" = torch.ops.aten.permute.default(view_799, [1, 0]);  view_799 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        permute_default_74: "f32[2048, 8192]" = torch.ops.aten.permute.default(view_801, [1, 0]);  view_801 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_57: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_24, rsqrt_7);  add_24 = rsqrt_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_tensor_58: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(view_802, mul_tensor_57);  view_802 = mul_tensor_57 = None
        sum_dim_int_list_26: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_58, [0, 1], True);  mul_tensor_58 = None
        reshape_default_39: "f32[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_26, _shape_param_40);  sum_dim_int_list_26 = _shape_param_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        permute_default_75: "f32[512, 8192]" = torch.ops.aten.permute.default(view_804, [1, 0]);  view_804 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_tensor_11: "f32[8, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(add_tensor_10, view_812);  add_tensor_10 = view_812 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_default_76: "f32[512, 8192]" = torch.ops.aten.permute.default(view_818, [1, 0]);  view_818 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_default_77: "f32[512, 8192]" = torch.ops.aten.permute.default(view_821, [1, 0]);  view_821 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_78: "f32[512, 8192]" = torch.ops.aten.permute.default(view_824, [1, 0]);  view_824 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_59: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_21, rsqrt_6);  add_21 = rsqrt_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_tensor_60: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_182, mul_tensor_59);  add_182 = mul_tensor_59 = None
        sum_dim_int_list_27: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_60, [0, 1], True);  mul_tensor_60 = None
        reshape_default_40: "f32[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_27, _shape_param_41);  sum_dim_int_list_27 = _shape_param_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        permute_default_79: "f32[512, 8192]" = torch.ops.aten.permute.default(view_827, [1, 0]);  view_827 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        permute_default_80: "f32[2048, 8192]" = torch.ops.aten.permute.default(view_829, [1, 0]);  view_829 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_61: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_19, rsqrt_5);  add_19 = rsqrt_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_tensor_62: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(view_830, mul_tensor_61);  view_830 = mul_tensor_61 = None
        sum_dim_int_list_28: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_62, [0, 1], True);  mul_tensor_62 = None
        reshape_default_41: "f32[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_28, _shape_param_42);  sum_dim_int_list_28 = _shape_param_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        permute_default_81: "f32[512, 8192]" = torch.ops.aten.permute.default(view_832, [1, 0]);  view_832 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_tensor_12: "f32[8, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(add_tensor_11, view_840);  add_tensor_11 = view_840 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_default_82: "f32[512, 8192]" = torch.ops.aten.permute.default(view_846, [1, 0]);  view_846 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_default_83: "f32[512, 8192]" = torch.ops.aten.permute.default(view_849, [1, 0]);  view_849 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_84: "f32[512, 8192]" = torch.ops.aten.permute.default(view_852, [1, 0]);  view_852 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_63: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_16, rsqrt_4);  add_16 = rsqrt_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_tensor_64: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_189, mul_tensor_63);  add_189 = mul_tensor_63 = None
        sum_dim_int_list_29: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_64, [0, 1], True);  mul_tensor_64 = None
        reshape_default_42: "f32[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_29, _shape_param_43);  sum_dim_int_list_29 = _shape_param_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        permute_default_85: "f32[512, 8192]" = torch.ops.aten.permute.default(view_855, [1, 0]);  view_855 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        permute_default_86: "f32[2048, 8192]" = torch.ops.aten.permute.default(view_857, [1, 0]);  view_857 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_65: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_14, rsqrt_3);  add_14 = rsqrt_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_tensor_66: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(view_858, mul_tensor_65);  view_858 = mul_tensor_65 = None
        sum_dim_int_list_30: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_66, [0, 1], True);  mul_tensor_66 = None
        reshape_default_43: "f32[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_30, _shape_param_44);  sum_dim_int_list_30 = _shape_param_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        permute_default_87: "f32[512, 8192]" = torch.ops.aten.permute.default(view_860, [1, 0]);  view_860 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_tensor_13: "f32[8, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(add_tensor_12, view_868);  add_tensor_12 = view_868 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_default_88: "f32[512, 8192]" = torch.ops.aten.permute.default(view_874, [1, 0]);  view_874 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_default_89: "f32[512, 8192]" = torch.ops.aten.permute.default(view_877, [1, 0]);  view_877 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_90: "f32[512, 8192]" = torch.ops.aten.permute.default(view_880, [1, 0]);  view_880 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_67: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_11, rsqrt_2);  add_11 = rsqrt_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_tensor_68: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_196, mul_tensor_67);  add_196 = mul_tensor_67 = None
        sum_dim_int_list_31: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_68, [0, 1], True);  mul_tensor_68 = None
        reshape_default_44: "f32[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_31, _shape_param_45);  sum_dim_int_list_31 = _shape_param_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        permute_default_91: "f32[512, 8192]" = torch.ops.aten.permute.default(view_883, [1, 0]);  view_883 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        permute_default_92: "f32[2048, 8192]" = torch.ops.aten.permute.default(view_885, [1, 0]);  view_885 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_69: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_9, rsqrt_1);  add_9 = rsqrt_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_tensor_70: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(view_886, mul_tensor_69);  view_886 = mul_tensor_69 = None
        sum_dim_int_list_32: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_70, [0, 1], True);  mul_tensor_70 = None
        reshape_default_45: "f32[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_32, _shape_param_46);  sum_dim_int_list_32 = _shape_param_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        permute_default_93: "f32[512, 8192]" = torch.ops.aten.permute.default(view_888, [1, 0]);  view_888 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_tensor_14: "f32[8, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(add_tensor_13, view_896);  add_tensor_13 = view_896 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:312 in forward, code: position_bias = position_bias + causal_mask
        sum_dim_int_list_33: "f32[1, 8, 1024, 1024]" = torch.ops.aten.sum.dim_IntList(add_tensor_14, [0], True);  add_tensor_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:237 in compute_bias, code: values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
        squeeze_dim_1: "f32[8, 1024, 1024]" = torch.ops.aten.squeeze.dim(sum_dim_int_list_33, 0);  sum_dim_int_list_33 = None
        permute_default_94: "f32[1024, 1024, 8]" = torch.ops.aten.permute.default(squeeze_dim_1, [1, 2, 0]);  squeeze_dim_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:236 in compute_bias, code: values = self.relative_attention_bias(relative_position_bucket)  # shape (query_length, key_length, num_heads)
        eq_scalar_2: "b8[1024, 1024]" = torch.ops.aten.eq.Scalar(add_6, -1)
        unsqueeze_default_2: "b8[1024, 1024, 1]" = torch.ops.aten.unsqueeze.default(eq_scalar_2, -1);  eq_scalar_2 = None
        where_self_2: "f32[1024, 1024, 8]" = torch.ops.aten.where.self(unsqueeze_default_2, full_default, permute_default_94);  unsqueeze_default_2 = permute_default_94 = None
        clone_default_1: "f32[1024, 1024, 8]" = torch.ops.aten.clone.default(where_self_2, memory_format = torch.contiguous_format);  where_self_2 = None
        index_put_default_2: "f32[32, 8]" = torch.ops.aten.index_put.default(full_default_1, [add_6], clone_default_1, True);  full_default_1 = add_6 = clone_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_default_95: "f32[512, 8192]" = torch.ops.aten.permute.default(view_902, [1, 0]);  view_902 = None
        reshape_default_46: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_286, _shape_param_47);  mm_286 = _shape_param_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_default_96: "f32[512, 8192]" = torch.ops.aten.permute.default(view_905, [1, 0]);  view_905 = None
        reshape_default_47: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_288, _shape_param_48);  mm_288 = _shape_param_48 = None
        add_tensor_15: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(reshape_default_46, reshape_default_47);  reshape_default_46 = reshape_default_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_97: "f32[512, 8192]" = torch.ops.aten.permute.default(view_908, [1, 0]);  view_908 = None
        reshape_default_48: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_290, _shape_param_49);  mm_290 = _shape_param_49 = None
        add_tensor_16: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_tensor_15, reshape_default_48);  add_tensor_15 = reshape_default_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_tensor_71: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_tensor_16, primals_3);  primals_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:721 in forward, code: hidden_states = self.dropout(inputs_embeds)
        mul_tensor_72: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(gt, embedding);  embedding = None
        mul_tensor_73: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_tensor_72, 1.1111111111111112);  mul_tensor_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_74: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_tensor_73, rsqrt)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_tensor_75: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_tensor_16, mul_tensor_74);  add_tensor_16 = mul_tensor_74 = None
        sum_dim_int_list_34: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_75, [0, 1], True);  mul_tensor_75 = None
        reshape_default_49: "f32[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_34, _shape_param_50);  sum_dim_int_list_34 = _shape_param_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_76: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_tensor_71, mul_tensor_73)
        mul_tensor_77: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_tensor_71, rsqrt);  mul_tensor_71 = None
        sum_dim_int_list_35: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_76, [2], True);  mul_tensor_76 = None
        add_tensor_17: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_200, mul_tensor_77);  add_200 = mul_tensor_77 = None
        pow_tensor_scalar_2: "f32[8, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt, 3);  rsqrt = None
        mul_scalar_2: "f32[8, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_dim_int_list_35, -0.5);  sum_dim_int_list_35 = None
        mul_tensor_78: "f32[8, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_scalar_2, pow_tensor_scalar_2);  mul_scalar_2 = pow_tensor_scalar_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_default_1: "f32[8, 1024, 512]" = torch.ops.aten.expand.default(mul_tensor_78, _shape_param_51);  mul_tensor_78 = _shape_param_51 = None
        div_scalar_1: "f32[8, 1024, 512]" = torch.ops.aten.div.Scalar(expand_default_1, 512);  expand_default_1 = None
        pow_tensor_scalar_3: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(mul_tensor_73, 1.0);  mul_tensor_73 = None
        mul_scalar_3: "f32[8, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_tensor_scalar_3, 2.0);  pow_tensor_scalar_3 = None
        mul_tensor_79: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(div_scalar_1, mul_scalar_3);  div_scalar_1 = mul_scalar_3 = None
        add_tensor_18: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_tensor_17, mul_tensor_79);  add_tensor_17 = mul_tensor_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:721 in forward, code: hidden_states = self.dropout(inputs_embeds)
        convert_element_type_default_1: "f32[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(gt, torch.float32);  gt = None
        mul_tensor_80: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_default_1, 1.1111111111111112);  convert_element_type_default_1 = None
        mul_tensor_81: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_tensor_18, mul_tensor_80);  add_tensor_18 = mul_tensor_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:669 in forward, code: inputs_embeds = self.embed_tokens(input_ids)
        where_self_3: "f32[8, 1024, 512]" = torch.ops.aten.where.self(unsqueeze_default_1, full_default, mul_tensor_81);  unsqueeze_default_1 = full_default = mul_tensor_81 = None
        index_put_default_3: "f32[32128, 512]" = torch.ops.aten.index_put.default(full_default_2, [primals_1], where_self_3, True);  full_default_2 = primals_1 = where_self_3 = None
        add_tensor_19: "f32[32128, 512]" = torch.ops.aten.add.Tensor(add_tensor_9, index_put_default_3);  add_tensor_9 = index_put_default_3 = None
        return (reshape_default, permute_default, permute_default_1, reshape_default_1, permute_default_2, permute_default_3, reshape_default_2, permute_default_4, reshape_default_3, permute_default_5, reshape_default_4, permute_default_6, permute_default_7, permute_default_8, permute_default_9, reshape_default_5, permute_default_10, permute_default_11, reshape_default_6, permute_default_12, permute_default_13, reshape_default_7, permute_default_14, reshape_default_8, permute_default_15, reshape_default_9, permute_default_16, permute_default_17, permute_default_18, permute_default_19, reshape_default_10, permute_default_20, permute_default_21, reshape_default_11, permute_default_22, permute_default_23, reshape_default_12, permute_default_24, reshape_default_13, permute_default_25, reshape_default_14, permute_default_26, permute_default_27, permute_default_28, permute_default_29, reshape_default_15, permute_default_30, permute_default_31, reshape_default_16, permute_default_32, permute_default_33, reshape_default_17, permute_default_34, reshape_default_18, permute_default_35, reshape_default_19, permute_default_36, permute_default_37, permute_default_38, permute_default_39, reshape_default_20, permute_default_40, permute_default_41, reshape_default_21, permute_default_42, permute_default_43, reshape_default_22, permute_default_44, reshape_default_23, permute_default_45, reshape_default_24, permute_default_46, permute_default_47, permute_default_48, permute_default_49, reshape_default_25, permute_default_50, permute_default_51, reshape_default_26, permute_default_52, permute_default_53, reshape_default_27, permute_default_54, reshape_default_28, permute_default_55, reshape_default_29, permute_default_56, index_put_default, permute_default_58, permute_default_59, permute_default_60, reshape_default_33, reshape_default_34, permute_default_61, permute_default_62, reshape_default_35, permute_default_63, permute_default_64, permute_default_65, permute_default_66, reshape_default_36, permute_default_67, permute_default_68, reshape_default_37, permute_default_69, permute_default_70, permute_default_71, permute_default_72, reshape_default_38, permute_default_73, permute_default_74, reshape_default_39, permute_default_75, permute_default_76, permute_default_77, permute_default_78, reshape_default_40, permute_default_79, permute_default_80, reshape_default_41, permute_default_81, permute_default_82, permute_default_83, permute_default_84, reshape_default_42, permute_default_85, permute_default_86, reshape_default_43, permute_default_87, permute_default_88, permute_default_89, permute_default_90, reshape_default_44, permute_default_91, permute_default_92, reshape_default_45, permute_default_93, index_put_default_2, permute_default_95, permute_default_96, permute_default_97, reshape_default_49, add_tensor_19)


def _default_make_inputs():
    return [
    torch.randn([8, 1024, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 512], dtype=torch.float32, device='cuda'),
    [512],  # _shape_param_0
    torch.randn([8192, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 2048], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 512], dtype=torch.float32, device='cuda'),
    [512],  # _shape_param_1
    torch.randn([8192, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 512], dtype=torch.float32, device='cuda'),
    [8192, 512],  # _shape_param_2
    torch.randn([8192, 512], dtype=torch.float32, device='cuda'),
    [8192, 512],  # _shape_param_3
    torch.randn([8192, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 512], dtype=torch.float32, device='cuda'),
    [512],  # _shape_param_4
    torch.randn([8192, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 512], dtype=torch.float32, device='cuda'),
    [512],  # _shape_param_5
    torch.randn([8192, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 2048], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 512], dtype=torch.float32, device='cuda'),
    [512],  # _shape_param_6
    torch.randn([8192, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 512], dtype=torch.float32, device='cuda'),
    [8192, 512],  # _shape_param_7
    torch.randn([8192, 512], dtype=torch.float32, device='cuda'),
    [8192, 512],  # _shape_param_8
    torch.randn([8192, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 512], dtype=torch.float32, device='cuda'),
    [512],  # _shape_param_9
    torch.randn([8192, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 8, 1024, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8, 8, 1024, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 512], dtype=torch.float32, device='cuda'),
    [512],  # _shape_param_10
    torch.randn([8192, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 2048], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 512], dtype=torch.float32, device='cuda'),
    [512],  # _shape_param_11
    torch.randn([8192, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 512], dtype=torch.float32, device='cuda'),
    [8192, 512],  # _shape_param_12
    torch.randn([8192, 512], dtype=torch.float32, device='cuda'),
    [8192, 512],  # _shape_param_13
    torch.randn([8192, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 512], dtype=torch.float32, device='cuda'),
    [512],  # _shape_param_14
    torch.randn([8192, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 8, 1024, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 512], dtype=torch.float32, device='cuda'),
    [512],  # _shape_param_15
    torch.randn([8192, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 2048], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 512], dtype=torch.float32, device='cuda'),
    [512],  # _shape_param_16
    torch.randn([8192, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 512], dtype=torch.float32, device='cuda'),
    [8192, 512],  # _shape_param_17
    torch.randn([8192, 512], dtype=torch.float32, device='cuda'),
    [8192, 512],  # _shape_param_18
    torch.randn([8192, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 512], dtype=torch.float32, device='cuda'),
    [512],  # _shape_param_19
    torch.randn([8192, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 8, 1024, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 512], dtype=torch.float32, device='cuda'),
    [512],  # _shape_param_20
    torch.randn([8192, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 2048], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 512], dtype=torch.float32, device='cuda'),
    [512],  # _shape_param_21
    torch.randn([8192, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 512], dtype=torch.float32, device='cuda'),
    [8192, 512],  # _shape_param_22
    torch.randn([8192, 512], dtype=torch.float32, device='cuda'),
    [8192, 512],  # _shape_param_23
    torch.randn([8192, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 512], dtype=torch.float32, device='cuda'),
    [512],  # _shape_param_24
    torch.randn([8192, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 8, 1024, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 512], dtype=torch.float32, device='cuda'),
    [512],  # _shape_param_25
    torch.randn([8192, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 2048], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 512], dtype=torch.float32, device='cuda'),
    [512],  # _shape_param_26
    torch.randn([8192, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 512], dtype=torch.float32, device='cuda'),
    [8192, 512],  # _shape_param_27
    torch.randn([8192, 512], dtype=torch.float32, device='cuda'),
    [8192, 512],  # _shape_param_28
    torch.randn([8192, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 512], dtype=torch.float32, device='cuda'),
    [512],  # _shape_param_29
    torch.randn([8192, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 8, 1024, 1024], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [1024, 1024], dtype=torch.int64, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 512], dtype=torch.float32, device='cuda'),
    [8, 1024, 512],  # _shape_param_30
    torch.randn([8192, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 512], dtype=torch.float32, device='cuda'),
    [8, 1024, 512],  # _shape_param_31
    torch.randn([8192, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 512], dtype=torch.float32, device='cuda'),
    [8, 1024, 512],  # _shape_param_32
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [8, 1024, 512], dtype=torch.bool, device='cuda'),
    torch.randn([8, 1024, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 1], dtype=torch.float32, device='cuda'),
    [512],  # _shape_param_33
    torch.randn([8, 1024, 512], dtype=torch.float32, device='cuda'),
    [8, 1024, 512],  # _shape_param_34
    torch.randint(0, 2, [8, 1024], dtype=torch.int64, device='cuda'),
    torch.randn([32128, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 512], dtype=torch.float32, device='cuda'),
    [512],  # _shape_param_35
    torch.randn([8192, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 2048], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 512], dtype=torch.float32, device='cuda'),
    [512],  # _shape_param_36
    torch.randn([8192, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 512], dtype=torch.float32, device='cuda'),
    [512],  # _shape_param_37
    torch.randn([8192, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 2048], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 512], dtype=torch.float32, device='cuda'),
    [512],  # _shape_param_38
    torch.randn([8192, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 8, 1024, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8, 8, 1024, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 512], dtype=torch.float32, device='cuda'),
    [512],  # _shape_param_39
    torch.randn([8192, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 2048], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 512], dtype=torch.float32, device='cuda'),
    [512],  # _shape_param_40
    torch.randn([8192, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 8, 1024, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 512], dtype=torch.float32, device='cuda'),
    [512],  # _shape_param_41
    torch.randn([8192, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 2048], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 512], dtype=torch.float32, device='cuda'),
    [512],  # _shape_param_42
    torch.randn([8192, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 8, 1024, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 512], dtype=torch.float32, device='cuda'),
    [512],  # _shape_param_43
    torch.randn([8192, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 2048], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 512], dtype=torch.float32, device='cuda'),
    [512],  # _shape_param_44
    torch.randn([8192, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 8, 1024, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 512], dtype=torch.float32, device='cuda'),
    [512],  # _shape_param_45
    torch.randn([8192, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 2048], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 512], dtype=torch.float32, device='cuda'),
    [512],  # _shape_param_46
    torch.randn([8192, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 8, 1024, 1024], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [1024, 1024], dtype=torch.int64, device='cuda'),
    torch.randn([8192, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 512], dtype=torch.float32, device='cuda'),
    [8, 1024, 512],  # _shape_param_47
    torch.randn([8192, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 512], dtype=torch.float32, device='cuda'),
    [8, 1024, 512],  # _shape_param_48
    torch.randn([8192, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 512], dtype=torch.float32, device='cuda'),
    [8, 1024, 512],  # _shape_param_49
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [8, 1024, 512], dtype=torch.bool, device='cuda'),
    torch.randn([8, 1024, 1], dtype=torch.float32, device='cuda'),
    [512],  # _shape_param_50
    torch.randn([8, 1024, 512], dtype=torch.float32, device='cuda'),
    [8, 1024, 512],  # _shape_param_51
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
