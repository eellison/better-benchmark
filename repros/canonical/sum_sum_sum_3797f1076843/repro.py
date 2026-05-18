"""
Standalone repro captured via capture_hook.
Label: hf_t5_base_train
Pattern hash: 3797f1076843
Shape hash: 5200f7bf
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_prelude import *  # noqa: F401,F403
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, add_66: "f32[4, 512, 768]", rsqrt_24: "f32[4, 512, 1]", mul_153: "f32[4, 512, 768]", _shape_param_0, view_302: "f32[2048, 768]", view_304: "f32[2048, 3072]", add_64: "f32[4, 512, 768]", rsqrt_23: "f32[4, 512, 1]", view_305: "f32[4, 512, 768]", _shape_param_1, view_307: "f32[2048, 768]", view_321: "f32[2048, 768]", view_324: "f32[2048, 768]", view_327: "f32[2048, 768]", add_61: "f32[4, 512, 768]", rsqrt_22: "f32[4, 512, 1]", add_72: "f32[4, 512, 768]", _shape_param_2, view_330: "f32[2048, 768]", view_332: "f32[2048, 3072]", add_59: "f32[4, 512, 768]", rsqrt_21: "f32[4, 512, 1]", view_333: "f32[4, 512, 768]", _shape_param_3, view_335: "f32[2048, 768]", view_315: "f32[4, 12, 512, 512]", view_343: "f32[4, 12, 512, 512]", view_349: "f32[2048, 768]", view_352: "f32[2048, 768]", view_355: "f32[2048, 768]", add_56: "f32[4, 512, 768]", rsqrt_20: "f32[4, 512, 1]", add_79: "f32[4, 512, 768]", _shape_param_4, view_358: "f32[2048, 768]", view_360: "f32[2048, 3072]", add_54: "f32[4, 512, 768]", rsqrt_19: "f32[4, 512, 1]", view_361: "f32[4, 512, 768]", _shape_param_5, view_363: "f32[2048, 768]", view_371: "f32[4, 12, 512, 512]", view_377: "f32[2048, 768]", view_380: "f32[2048, 768]", view_383: "f32[2048, 768]", add_51: "f32[4, 512, 768]", rsqrt_18: "f32[4, 512, 1]", add_86: "f32[4, 512, 768]", _shape_param_6, view_386: "f32[2048, 768]", view_388: "f32[2048, 3072]", add_49: "f32[4, 512, 768]", rsqrt_17: "f32[4, 512, 1]", view_389: "f32[4, 512, 768]", _shape_param_7, view_391: "f32[2048, 768]", view_399: "f32[4, 12, 512, 512]", view_405: "f32[2048, 768]", view_408: "f32[2048, 768]", view_411: "f32[2048, 768]", add_46: "f32[4, 512, 768]", rsqrt_16: "f32[4, 512, 1]", add_93: "f32[4, 512, 768]", _shape_param_8, view_414: "f32[2048, 768]", view_416: "f32[2048, 3072]", add_44: "f32[4, 512, 768]", rsqrt_15: "f32[4, 512, 1]", view_417: "f32[4, 512, 768]", _shape_param_9, view_419: "f32[2048, 768]", view_427: "f32[4, 12, 512, 512]", view_433: "f32[2048, 768]", view_436: "f32[2048, 768]", view_439: "f32[2048, 768]", add_41: "f32[4, 512, 768]", rsqrt_14: "f32[4, 512, 1]", add_100: "f32[4, 512, 768]", _shape_param_10, view_442: "f32[2048, 768]", view_444: "f32[2048, 3072]", add_39: "f32[4, 512, 768]", rsqrt_13: "f32[4, 512, 1]", view_445: "f32[4, 512, 768]", _shape_param_11, view_447: "f32[2048, 768]", view_455: "f32[4, 12, 512, 512]", view_461: "f32[2048, 768]", view_464: "f32[2048, 768]", view_467: "f32[2048, 768]", add_36: "f32[4, 512, 768]", rsqrt_12: "f32[4, 512, 1]", add_107: "f32[4, 512, 768]", _shape_param_12, view_470: "f32[2048, 768]", view_472: "f32[2048, 3072]", add_34: "f32[4, 512, 768]", rsqrt_11: "f32[4, 512, 1]", view_473: "f32[4, 512, 768]", _shape_param_13, view_475: "f32[2048, 768]", view_483: "f32[4, 12, 512, 512]", view_489: "f32[2048, 768]", view_492: "f32[2048, 768]", view_495: "f32[2048, 768]", add_31: "f32[4, 512, 768]", rsqrt_10: "f32[4, 512, 1]", add_114: "f32[4, 512, 768]", _shape_param_14, view_498: "f32[2048, 768]", view_500: "f32[2048, 3072]", add_29: "f32[4, 512, 768]", rsqrt_9: "f32[4, 512, 1]", view_501: "f32[4, 512, 768]", _shape_param_15, view_503: "f32[2048, 768]", view_511: "f32[4, 12, 512, 512]", view_517: "f32[2048, 768]", view_520: "f32[2048, 768]", view_523: "f32[2048, 768]", add_26: "f32[4, 512, 768]", rsqrt_8: "f32[4, 512, 1]", add_121: "f32[4, 512, 768]", _shape_param_16, view_526: "f32[2048, 768]", view_528: "f32[2048, 3072]", add_24: "f32[4, 512, 768]", rsqrt_7: "f32[4, 512, 1]", view_529: "f32[4, 512, 768]", _shape_param_17, view_531: "f32[2048, 768]", view_539: "f32[4, 12, 512, 512]", view_545: "f32[2048, 768]", view_548: "f32[2048, 768]", view_551: "f32[2048, 768]", add_21: "f32[4, 512, 768]", rsqrt_6: "f32[4, 512, 1]", add_128: "f32[4, 512, 768]", _shape_param_18, view_554: "f32[2048, 768]", view_556: "f32[2048, 3072]", add_19: "f32[4, 512, 768]", rsqrt_5: "f32[4, 512, 1]", view_557: "f32[4, 512, 768]", _shape_param_19, view_559: "f32[2048, 768]", view_567: "f32[4, 12, 512, 512]", view_573: "f32[2048, 768]", view_576: "f32[2048, 768]", view_579: "f32[2048, 768]", add_16: "f32[4, 512, 768]", rsqrt_4: "f32[4, 512, 1]", add_135: "f32[4, 512, 768]", _shape_param_20, view_582: "f32[2048, 768]", view_584: "f32[2048, 3072]", add_14: "f32[4, 512, 768]", rsqrt_3: "f32[4, 512, 1]", view_585: "f32[4, 512, 768]", _shape_param_21, view_587: "f32[2048, 768]", view_595: "f32[4, 12, 512, 512]", view_601: "f32[2048, 768]", view_604: "f32[2048, 768]", view_607: "f32[2048, 768]", add_11: "f32[4, 512, 768]", rsqrt_2: "f32[4, 512, 1]", add_142: "f32[4, 512, 768]", _shape_param_22, view_610: "f32[2048, 768]", view_612: "f32[2048, 3072]", add_9: "f32[4, 512, 768]", rsqrt_1: "f32[4, 512, 1]", view_613: "f32[4, 512, 768]", _shape_param_23, view_615: "f32[2048, 768]", view_623: "f32[4, 12, 512, 512]", add_6: "i64[512, 512]", full_default: "f32[]", view_629: "f32[2048, 768]", mm_211: "f32[2048, 768]", _shape_param_24, view_632: "f32[2048, 768]", mm_213: "f32[2048, 768]", _shape_param_25, view_635: "f32[2048, 768]", mm_215: "f32[2048, 768]", _shape_param_26, primals_3: "f32[768]", gt: "b8[4, 512, 768]", embedding: "f32[4, 512, 768]", rsqrt: "f32[4, 512, 1]", _shape_param_27, add_146: "f32[4, 512, 768]", _shape_param_28, primals_1: "i64[4, 512]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_66, rsqrt_24);  add_66 = rsqrt_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_tensor_1: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_153, mul_tensor);  mul_153 = mul_tensor = None
        sum_dim_int_list: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_1, [0, 1], True);  mul_tensor_1 = None
        reshape_default: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list, _shape_param_0);  sum_dim_int_list = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        permute_default: "f32[768, 2048]" = torch.ops.aten.permute.default(view_302, [1, 0]);  view_302 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        permute_default_1: "f32[3072, 2048]" = torch.ops.aten.permute.default(view_304, [1, 0]);  view_304 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_2: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_64, rsqrt_23);  add_64 = rsqrt_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_tensor_3: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(view_305, mul_tensor_2);  view_305 = mul_tensor_2 = None
        sum_dim_int_list_1: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_3, [0, 1], True);  mul_tensor_3 = None
        reshape_default_1: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_1, _shape_param_1);  sum_dim_int_list_1 = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        permute_default_2: "f32[768, 2048]" = torch.ops.aten.permute.default(view_307, [1, 0]);  view_307 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_default_3: "f32[768, 2048]" = torch.ops.aten.permute.default(view_321, [1, 0]);  view_321 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_default_4: "f32[768, 2048]" = torch.ops.aten.permute.default(view_324, [1, 0]);  view_324 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_5: "f32[768, 2048]" = torch.ops.aten.permute.default(view_327, [1, 0]);  view_327 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_4: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_61, rsqrt_22);  add_61 = rsqrt_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_tensor_5: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_72, mul_tensor_4);  add_72 = mul_tensor_4 = None
        sum_dim_int_list_2: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_5, [0, 1], True);  mul_tensor_5 = None
        reshape_default_2: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_2, _shape_param_2);  sum_dim_int_list_2 = _shape_param_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        permute_default_6: "f32[768, 2048]" = torch.ops.aten.permute.default(view_330, [1, 0]);  view_330 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        permute_default_7: "f32[3072, 2048]" = torch.ops.aten.permute.default(view_332, [1, 0]);  view_332 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_6: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_59, rsqrt_21);  add_59 = rsqrt_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_tensor_7: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(view_333, mul_tensor_6);  view_333 = mul_tensor_6 = None
        sum_dim_int_list_3: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_7, [0, 1], True);  mul_tensor_7 = None
        reshape_default_3: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_3, _shape_param_3);  sum_dim_int_list_3 = _shape_param_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        permute_default_8: "f32[768, 2048]" = torch.ops.aten.permute.default(view_335, [1, 0]);  view_335 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_tensor: "f32[4, 12, 512, 512]" = torch.ops.aten.add.Tensor(view_315, view_343);  view_315 = view_343 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_default_9: "f32[768, 2048]" = torch.ops.aten.permute.default(view_349, [1, 0]);  view_349 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_default_10: "f32[768, 2048]" = torch.ops.aten.permute.default(view_352, [1, 0]);  view_352 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_11: "f32[768, 2048]" = torch.ops.aten.permute.default(view_355, [1, 0]);  view_355 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_8: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_56, rsqrt_20);  add_56 = rsqrt_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_tensor_9: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_79, mul_tensor_8);  add_79 = mul_tensor_8 = None
        sum_dim_int_list_4: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_9, [0, 1], True);  mul_tensor_9 = None
        reshape_default_4: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_4, _shape_param_4);  sum_dim_int_list_4 = _shape_param_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        permute_default_12: "f32[768, 2048]" = torch.ops.aten.permute.default(view_358, [1, 0]);  view_358 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        permute_default_13: "f32[3072, 2048]" = torch.ops.aten.permute.default(view_360, [1, 0]);  view_360 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_10: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_54, rsqrt_19);  add_54 = rsqrt_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_tensor_11: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(view_361, mul_tensor_10);  view_361 = mul_tensor_10 = None
        sum_dim_int_list_5: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_11, [0, 1], True);  mul_tensor_11 = None
        reshape_default_5: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_5, _shape_param_5);  sum_dim_int_list_5 = _shape_param_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        permute_default_14: "f32[768, 2048]" = torch.ops.aten.permute.default(view_363, [1, 0]);  view_363 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_tensor_1: "f32[4, 12, 512, 512]" = torch.ops.aten.add.Tensor(add_tensor, view_371);  add_tensor = view_371 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_default_15: "f32[768, 2048]" = torch.ops.aten.permute.default(view_377, [1, 0]);  view_377 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_default_16: "f32[768, 2048]" = torch.ops.aten.permute.default(view_380, [1, 0]);  view_380 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_17: "f32[768, 2048]" = torch.ops.aten.permute.default(view_383, [1, 0]);  view_383 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_12: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_51, rsqrt_18);  add_51 = rsqrt_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_tensor_13: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_86, mul_tensor_12);  add_86 = mul_tensor_12 = None
        sum_dim_int_list_6: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_13, [0, 1], True);  mul_tensor_13 = None
        reshape_default_6: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_6, _shape_param_6);  sum_dim_int_list_6 = _shape_param_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        permute_default_18: "f32[768, 2048]" = torch.ops.aten.permute.default(view_386, [1, 0]);  view_386 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        permute_default_19: "f32[3072, 2048]" = torch.ops.aten.permute.default(view_388, [1, 0]);  view_388 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_14: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_49, rsqrt_17);  add_49 = rsqrt_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_tensor_15: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(view_389, mul_tensor_14);  view_389 = mul_tensor_14 = None
        sum_dim_int_list_7: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_15, [0, 1], True);  mul_tensor_15 = None
        reshape_default_7: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_7, _shape_param_7);  sum_dim_int_list_7 = _shape_param_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        permute_default_20: "f32[768, 2048]" = torch.ops.aten.permute.default(view_391, [1, 0]);  view_391 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_tensor_2: "f32[4, 12, 512, 512]" = torch.ops.aten.add.Tensor(add_tensor_1, view_399);  add_tensor_1 = view_399 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_default_21: "f32[768, 2048]" = torch.ops.aten.permute.default(view_405, [1, 0]);  view_405 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_default_22: "f32[768, 2048]" = torch.ops.aten.permute.default(view_408, [1, 0]);  view_408 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_23: "f32[768, 2048]" = torch.ops.aten.permute.default(view_411, [1, 0]);  view_411 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_16: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_46, rsqrt_16);  add_46 = rsqrt_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_tensor_17: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_93, mul_tensor_16);  add_93 = mul_tensor_16 = None
        sum_dim_int_list_8: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_17, [0, 1], True);  mul_tensor_17 = None
        reshape_default_8: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_8, _shape_param_8);  sum_dim_int_list_8 = _shape_param_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        permute_default_24: "f32[768, 2048]" = torch.ops.aten.permute.default(view_414, [1, 0]);  view_414 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        permute_default_25: "f32[3072, 2048]" = torch.ops.aten.permute.default(view_416, [1, 0]);  view_416 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_18: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_44, rsqrt_15);  add_44 = rsqrt_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_tensor_19: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(view_417, mul_tensor_18);  view_417 = mul_tensor_18 = None
        sum_dim_int_list_9: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_19, [0, 1], True);  mul_tensor_19 = None
        reshape_default_9: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_9, _shape_param_9);  sum_dim_int_list_9 = _shape_param_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        permute_default_26: "f32[768, 2048]" = torch.ops.aten.permute.default(view_419, [1, 0]);  view_419 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_tensor_3: "f32[4, 12, 512, 512]" = torch.ops.aten.add.Tensor(add_tensor_2, view_427);  add_tensor_2 = view_427 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_default_27: "f32[768, 2048]" = torch.ops.aten.permute.default(view_433, [1, 0]);  view_433 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_default_28: "f32[768, 2048]" = torch.ops.aten.permute.default(view_436, [1, 0]);  view_436 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_29: "f32[768, 2048]" = torch.ops.aten.permute.default(view_439, [1, 0]);  view_439 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_20: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_41, rsqrt_14);  add_41 = rsqrt_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_tensor_21: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_100, mul_tensor_20);  add_100 = mul_tensor_20 = None
        sum_dim_int_list_10: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_21, [0, 1], True);  mul_tensor_21 = None
        reshape_default_10: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_10, _shape_param_10);  sum_dim_int_list_10 = _shape_param_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        permute_default_30: "f32[768, 2048]" = torch.ops.aten.permute.default(view_442, [1, 0]);  view_442 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        permute_default_31: "f32[3072, 2048]" = torch.ops.aten.permute.default(view_444, [1, 0]);  view_444 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_22: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_39, rsqrt_13);  add_39 = rsqrt_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_tensor_23: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(view_445, mul_tensor_22);  view_445 = mul_tensor_22 = None
        sum_dim_int_list_11: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_23, [0, 1], True);  mul_tensor_23 = None
        reshape_default_11: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_11, _shape_param_11);  sum_dim_int_list_11 = _shape_param_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        permute_default_32: "f32[768, 2048]" = torch.ops.aten.permute.default(view_447, [1, 0]);  view_447 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_tensor_4: "f32[4, 12, 512, 512]" = torch.ops.aten.add.Tensor(add_tensor_3, view_455);  add_tensor_3 = view_455 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_default_33: "f32[768, 2048]" = torch.ops.aten.permute.default(view_461, [1, 0]);  view_461 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_default_34: "f32[768, 2048]" = torch.ops.aten.permute.default(view_464, [1, 0]);  view_464 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_35: "f32[768, 2048]" = torch.ops.aten.permute.default(view_467, [1, 0]);  view_467 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_24: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_36, rsqrt_12);  add_36 = rsqrt_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_tensor_25: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_107, mul_tensor_24);  add_107 = mul_tensor_24 = None
        sum_dim_int_list_12: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_25, [0, 1], True);  mul_tensor_25 = None
        reshape_default_12: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_12, _shape_param_12);  sum_dim_int_list_12 = _shape_param_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        permute_default_36: "f32[768, 2048]" = torch.ops.aten.permute.default(view_470, [1, 0]);  view_470 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        permute_default_37: "f32[3072, 2048]" = torch.ops.aten.permute.default(view_472, [1, 0]);  view_472 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_26: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_34, rsqrt_11);  add_34 = rsqrt_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_tensor_27: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(view_473, mul_tensor_26);  view_473 = mul_tensor_26 = None
        sum_dim_int_list_13: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_27, [0, 1], True);  mul_tensor_27 = None
        reshape_default_13: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_13, _shape_param_13);  sum_dim_int_list_13 = _shape_param_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        permute_default_38: "f32[768, 2048]" = torch.ops.aten.permute.default(view_475, [1, 0]);  view_475 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_tensor_5: "f32[4, 12, 512, 512]" = torch.ops.aten.add.Tensor(add_tensor_4, view_483);  add_tensor_4 = view_483 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_default_39: "f32[768, 2048]" = torch.ops.aten.permute.default(view_489, [1, 0]);  view_489 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_default_40: "f32[768, 2048]" = torch.ops.aten.permute.default(view_492, [1, 0]);  view_492 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_41: "f32[768, 2048]" = torch.ops.aten.permute.default(view_495, [1, 0]);  view_495 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_28: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_31, rsqrt_10);  add_31 = rsqrt_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_tensor_29: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_114, mul_tensor_28);  add_114 = mul_tensor_28 = None
        sum_dim_int_list_14: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_29, [0, 1], True);  mul_tensor_29 = None
        reshape_default_14: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_14, _shape_param_14);  sum_dim_int_list_14 = _shape_param_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        permute_default_42: "f32[768, 2048]" = torch.ops.aten.permute.default(view_498, [1, 0]);  view_498 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        permute_default_43: "f32[3072, 2048]" = torch.ops.aten.permute.default(view_500, [1, 0]);  view_500 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_30: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_29, rsqrt_9);  add_29 = rsqrt_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_tensor_31: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(view_501, mul_tensor_30);  view_501 = mul_tensor_30 = None
        sum_dim_int_list_15: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_31, [0, 1], True);  mul_tensor_31 = None
        reshape_default_15: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_15, _shape_param_15);  sum_dim_int_list_15 = _shape_param_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        permute_default_44: "f32[768, 2048]" = torch.ops.aten.permute.default(view_503, [1, 0]);  view_503 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_tensor_6: "f32[4, 12, 512, 512]" = torch.ops.aten.add.Tensor(add_tensor_5, view_511);  add_tensor_5 = view_511 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_default_45: "f32[768, 2048]" = torch.ops.aten.permute.default(view_517, [1, 0]);  view_517 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_default_46: "f32[768, 2048]" = torch.ops.aten.permute.default(view_520, [1, 0]);  view_520 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_47: "f32[768, 2048]" = torch.ops.aten.permute.default(view_523, [1, 0]);  view_523 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_32: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_26, rsqrt_8);  add_26 = rsqrt_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_tensor_33: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_121, mul_tensor_32);  add_121 = mul_tensor_32 = None
        sum_dim_int_list_16: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_33, [0, 1], True);  mul_tensor_33 = None
        reshape_default_16: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_16, _shape_param_16);  sum_dim_int_list_16 = _shape_param_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        permute_default_48: "f32[768, 2048]" = torch.ops.aten.permute.default(view_526, [1, 0]);  view_526 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        permute_default_49: "f32[3072, 2048]" = torch.ops.aten.permute.default(view_528, [1, 0]);  view_528 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_34: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_24, rsqrt_7);  add_24 = rsqrt_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_tensor_35: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(view_529, mul_tensor_34);  view_529 = mul_tensor_34 = None
        sum_dim_int_list_17: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_35, [0, 1], True);  mul_tensor_35 = None
        reshape_default_17: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_17, _shape_param_17);  sum_dim_int_list_17 = _shape_param_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        permute_default_50: "f32[768, 2048]" = torch.ops.aten.permute.default(view_531, [1, 0]);  view_531 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_tensor_7: "f32[4, 12, 512, 512]" = torch.ops.aten.add.Tensor(add_tensor_6, view_539);  add_tensor_6 = view_539 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_default_51: "f32[768, 2048]" = torch.ops.aten.permute.default(view_545, [1, 0]);  view_545 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_default_52: "f32[768, 2048]" = torch.ops.aten.permute.default(view_548, [1, 0]);  view_548 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_53: "f32[768, 2048]" = torch.ops.aten.permute.default(view_551, [1, 0]);  view_551 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_36: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_21, rsqrt_6);  add_21 = rsqrt_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_tensor_37: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_128, mul_tensor_36);  add_128 = mul_tensor_36 = None
        sum_dim_int_list_18: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_37, [0, 1], True);  mul_tensor_37 = None
        reshape_default_18: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_18, _shape_param_18);  sum_dim_int_list_18 = _shape_param_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        permute_default_54: "f32[768, 2048]" = torch.ops.aten.permute.default(view_554, [1, 0]);  view_554 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        permute_default_55: "f32[3072, 2048]" = torch.ops.aten.permute.default(view_556, [1, 0]);  view_556 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_38: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_19, rsqrt_5);  add_19 = rsqrt_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_tensor_39: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(view_557, mul_tensor_38);  view_557 = mul_tensor_38 = None
        sum_dim_int_list_19: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_39, [0, 1], True);  mul_tensor_39 = None
        reshape_default_19: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_19, _shape_param_19);  sum_dim_int_list_19 = _shape_param_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        permute_default_56: "f32[768, 2048]" = torch.ops.aten.permute.default(view_559, [1, 0]);  view_559 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_tensor_8: "f32[4, 12, 512, 512]" = torch.ops.aten.add.Tensor(add_tensor_7, view_567);  add_tensor_7 = view_567 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_default_57: "f32[768, 2048]" = torch.ops.aten.permute.default(view_573, [1, 0]);  view_573 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_default_58: "f32[768, 2048]" = torch.ops.aten.permute.default(view_576, [1, 0]);  view_576 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_59: "f32[768, 2048]" = torch.ops.aten.permute.default(view_579, [1, 0]);  view_579 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_40: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_16, rsqrt_4);  add_16 = rsqrt_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_tensor_41: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_135, mul_tensor_40);  add_135 = mul_tensor_40 = None
        sum_dim_int_list_20: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_41, [0, 1], True);  mul_tensor_41 = None
        reshape_default_20: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_20, _shape_param_20);  sum_dim_int_list_20 = _shape_param_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        permute_default_60: "f32[768, 2048]" = torch.ops.aten.permute.default(view_582, [1, 0]);  view_582 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        permute_default_61: "f32[3072, 2048]" = torch.ops.aten.permute.default(view_584, [1, 0]);  view_584 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_42: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_14, rsqrt_3);  add_14 = rsqrt_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_tensor_43: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(view_585, mul_tensor_42);  view_585 = mul_tensor_42 = None
        sum_dim_int_list_21: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_43, [0, 1], True);  mul_tensor_43 = None
        reshape_default_21: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_21, _shape_param_21);  sum_dim_int_list_21 = _shape_param_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        permute_default_62: "f32[768, 2048]" = torch.ops.aten.permute.default(view_587, [1, 0]);  view_587 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_tensor_9: "f32[4, 12, 512, 512]" = torch.ops.aten.add.Tensor(add_tensor_8, view_595);  add_tensor_8 = view_595 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_default_63: "f32[768, 2048]" = torch.ops.aten.permute.default(view_601, [1, 0]);  view_601 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_default_64: "f32[768, 2048]" = torch.ops.aten.permute.default(view_604, [1, 0]);  view_604 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_65: "f32[768, 2048]" = torch.ops.aten.permute.default(view_607, [1, 0]);  view_607 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_44: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_11, rsqrt_2);  add_11 = rsqrt_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_tensor_45: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_142, mul_tensor_44);  add_142 = mul_tensor_44 = None
        sum_dim_int_list_22: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_45, [0, 1], True);  mul_tensor_45 = None
        reshape_default_22: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_22, _shape_param_22);  sum_dim_int_list_22 = _shape_param_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        permute_default_66: "f32[768, 2048]" = torch.ops.aten.permute.default(view_610, [1, 0]);  view_610 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        permute_default_67: "f32[3072, 2048]" = torch.ops.aten.permute.default(view_612, [1, 0]);  view_612 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_46: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_9, rsqrt_1);  add_9 = rsqrt_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_tensor_47: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(view_613, mul_tensor_46);  view_613 = mul_tensor_46 = None
        sum_dim_int_list_23: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_47, [0, 1], True);  mul_tensor_47 = None
        reshape_default_23: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_23, _shape_param_23);  sum_dim_int_list_23 = _shape_param_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        permute_default_68: "f32[768, 2048]" = torch.ops.aten.permute.default(view_615, [1, 0]);  view_615 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_tensor_10: "f32[4, 12, 512, 512]" = torch.ops.aten.add.Tensor(add_tensor_9, view_623);  add_tensor_9 = view_623 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:312 in forward, code: position_bias = position_bias + causal_mask
        sum_dim_int_list_24: "f32[1, 12, 512, 512]" = torch.ops.aten.sum.dim_IntList(add_tensor_10, [0], True);  add_tensor_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:237 in compute_bias, code: values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
        squeeze_dim: "f32[12, 512, 512]" = torch.ops.aten.squeeze.dim(sum_dim_int_list_24, 0);  sum_dim_int_list_24 = None
        permute_default_69: "f32[512, 512, 12]" = torch.ops.aten.permute.default(squeeze_dim, [1, 2, 0]);  squeeze_dim = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:236 in compute_bias, code: values = self.relative_attention_bias(relative_position_bucket)  # shape (query_length, key_length, num_heads)
        eq_scalar: "b8[512, 512]" = torch.ops.aten.eq.Scalar(add_6, -1)
        unsqueeze_default: "b8[512, 512, 1]" = torch.ops.aten.unsqueeze.default(eq_scalar, -1);  eq_scalar = None
        where_self: "f32[512, 512, 12]" = torch.ops.aten.where.self(unsqueeze_default, full_default, permute_default_69);  unsqueeze_default = permute_default_69 = None
        clone_default: "f32[512, 512, 12]" = torch.ops.aten.clone.default(where_self, memory_format = torch.contiguous_format);  where_self = None
        full_default_1: "f32[32, 12]" = torch.ops.aten.full.default([32, 12], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default: "f32[32, 12]" = torch.ops.aten.index_put.default(full_default_1, [add_6], clone_default, True);  full_default_1 = add_6 = clone_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_default_70: "f32[768, 2048]" = torch.ops.aten.permute.default(view_629, [1, 0]);  view_629 = None
        reshape_default_24: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_211, _shape_param_24);  mm_211 = _shape_param_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_default_71: "f32[768, 2048]" = torch.ops.aten.permute.default(view_632, [1, 0]);  view_632 = None
        reshape_default_25: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_213, _shape_param_25);  mm_213 = _shape_param_25 = None
        add_tensor_11: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(reshape_default_24, reshape_default_25);  reshape_default_24 = reshape_default_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_72: "f32[768, 2048]" = torch.ops.aten.permute.default(view_635, [1, 0]);  view_635 = None
        reshape_default_26: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_215, _shape_param_26);  mm_215 = _shape_param_26 = None
        add_tensor_12: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_tensor_11, reshape_default_26);  add_tensor_11 = reshape_default_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_tensor_48: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_tensor_12, primals_3);  primals_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:721 in forward, code: hidden_states = self.dropout(inputs_embeds)
        mul_tensor_49: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(gt, embedding);  embedding = None
        mul_tensor_50: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_49, 1.1111111111111112);  mul_tensor_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_51: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_50, rsqrt)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_tensor_52: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_tensor_12, mul_tensor_51);  add_tensor_12 = mul_tensor_51 = None
        sum_dim_int_list_25: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_52, [0, 1], True);  mul_tensor_52 = None
        reshape_default_27: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_25, _shape_param_27);  sum_dim_int_list_25 = _shape_param_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_53: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_48, mul_tensor_50)
        mul_tensor_54: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_48, rsqrt);  mul_tensor_48 = None
        sum_dim_int_list_26: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_53, [2], True);  mul_tensor_53 = None
        add_tensor_13: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_146, mul_tensor_54);  add_146 = mul_tensor_54 = None
        pow_tensor_scalar: "f32[4, 512, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt, 3);  rsqrt = None
        mul_scalar: "f32[4, 512, 1]" = torch.ops.aten.mul.Scalar(sum_dim_int_list_26, -0.5);  sum_dim_int_list_26 = None
        mul_tensor_55: "f32[4, 512, 1]" = torch.ops.aten.mul.Tensor(mul_scalar, pow_tensor_scalar);  mul_scalar = pow_tensor_scalar = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_default: "f32[4, 512, 768]" = torch.ops.aten.expand.default(mul_tensor_55, _shape_param_28);  mul_tensor_55 = _shape_param_28 = None
        div_scalar: "f32[4, 512, 768]" = torch.ops.aten.div.Scalar(expand_default, 768);  expand_default = None
        pow_tensor_scalar_1: "f32[4, 512, 768]" = torch.ops.aten.pow.Tensor_Scalar(mul_tensor_50, 1.0);  mul_tensor_50 = None
        mul_scalar_1: "f32[4, 512, 768]" = torch.ops.aten.mul.Scalar(pow_tensor_scalar_1, 2.0);  pow_tensor_scalar_1 = None
        mul_tensor_56: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(div_scalar, mul_scalar_1);  div_scalar = mul_scalar_1 = None
        add_tensor_14: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_tensor_13, mul_tensor_56);  add_tensor_13 = mul_tensor_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:721 in forward, code: hidden_states = self.dropout(inputs_embeds)
        convert_element_type_default: "f32[4, 512, 768]" = torch.ops.prims.convert_element_type.default(gt, torch.float32);  gt = None
        mul_tensor_57: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 1.1111111111111112);  convert_element_type_default = None
        mul_tensor_58: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_tensor_14, mul_tensor_57);  add_tensor_14 = mul_tensor_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:669 in forward, code: inputs_embeds = self.embed_tokens(input_ids)
        eq_scalar_1: "b8[4, 512]" = torch.ops.aten.eq.Scalar(primals_1, -1)
        unsqueeze_default_1: "b8[4, 512, 1]" = torch.ops.aten.unsqueeze.default(eq_scalar_1, -1);  eq_scalar_1 = None
        where_self_1: "f32[4, 512, 768]" = torch.ops.aten.where.self(unsqueeze_default_1, full_default, mul_tensor_58);  unsqueeze_default_1 = full_default = mul_tensor_58 = None
        full_default_2: "f32[32128, 768]" = torch.ops.aten.full.default([32128, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default_1: "f32[32128, 768]" = torch.ops.aten.index_put.default(full_default_2, [primals_1], where_self_1, True);  full_default_2 = primals_1 = where_self_1 = None
        return (reshape_default, permute_default, permute_default_1, reshape_default_1, permute_default_2, permute_default_3, permute_default_4, permute_default_5, reshape_default_2, permute_default_6, permute_default_7, reshape_default_3, permute_default_8, permute_default_9, permute_default_10, permute_default_11, reshape_default_4, permute_default_12, permute_default_13, reshape_default_5, permute_default_14, permute_default_15, permute_default_16, permute_default_17, reshape_default_6, permute_default_18, permute_default_19, reshape_default_7, permute_default_20, permute_default_21, permute_default_22, permute_default_23, reshape_default_8, permute_default_24, permute_default_25, reshape_default_9, permute_default_26, permute_default_27, permute_default_28, permute_default_29, reshape_default_10, permute_default_30, permute_default_31, reshape_default_11, permute_default_32, permute_default_33, permute_default_34, permute_default_35, reshape_default_12, permute_default_36, permute_default_37, reshape_default_13, permute_default_38, permute_default_39, permute_default_40, permute_default_41, reshape_default_14, permute_default_42, permute_default_43, reshape_default_15, permute_default_44, permute_default_45, permute_default_46, permute_default_47, reshape_default_16, permute_default_48, permute_default_49, reshape_default_17, permute_default_50, permute_default_51, permute_default_52, permute_default_53, reshape_default_18, permute_default_54, permute_default_55, reshape_default_19, permute_default_56, permute_default_57, permute_default_58, permute_default_59, reshape_default_20, permute_default_60, permute_default_61, reshape_default_21, permute_default_62, permute_default_63, permute_default_64, permute_default_65, reshape_default_22, permute_default_66, permute_default_67, reshape_default_23, permute_default_68, index_put_default, permute_default_70, permute_default_71, permute_default_72, reshape_default_27, index_put_default_1)


def _default_make_inputs():
    return [
    torch.randn([4, 512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_0
    torch.randn([2048, 768], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_1
    torch.randn([2048, 768], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 768], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 768], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_2
    torch.randn([2048, 768], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_3
    torch.randn([2048, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 12, 512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([4, 12, 512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 768], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 768], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_4
    torch.randn([2048, 768], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_5
    torch.randn([2048, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 12, 512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 768], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 768], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_6
    torch.randn([2048, 768], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_7
    torch.randn([2048, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 12, 512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 768], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 768], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_8
    torch.randn([2048, 768], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_9
    torch.randn([2048, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 12, 512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 768], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 768], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_10
    torch.randn([2048, 768], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_11
    torch.randn([2048, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 12, 512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 768], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 768], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_12
    torch.randn([2048, 768], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_13
    torch.randn([2048, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 12, 512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 768], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 768], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_14
    torch.randn([2048, 768], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_15
    torch.randn([2048, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 12, 512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 768], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 768], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_16
    torch.randn([2048, 768], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_17
    torch.randn([2048, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 12, 512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 768], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 768], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_18
    torch.randn([2048, 768], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_19
    torch.randn([2048, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 12, 512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 768], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 768], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_20
    torch.randn([2048, 768], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_21
    torch.randn([2048, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 12, 512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 768], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 768], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_22
    torch.randn([2048, 768], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_23
    torch.randn([2048, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 12, 512, 512], dtype=torch.float32, device='cuda'),
    torch.randint(0, 32, [512, 512], dtype=torch.int64, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 768], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 768], dtype=torch.float32, device='cuda'),
    [4, 512, 768],  # _shape_param_24
    torch.randn([2048, 768], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 768], dtype=torch.float32, device='cuda'),
    [4, 512, 768],  # _shape_param_25
    torch.randn([2048, 768], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 768], dtype=torch.float32, device='cuda'),
    [4, 512, 768],  # _shape_param_26
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [4, 512, 768], dtype=torch.bool, device='cuda'),
    torch.randn([4, 512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 1], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_27
    torch.randn([4, 512, 768], dtype=torch.float32, device='cuda'),
    [4, 512, 768],  # _shape_param_28
    torch.randint(0, 32128, [4, 512], dtype=torch.int64, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
