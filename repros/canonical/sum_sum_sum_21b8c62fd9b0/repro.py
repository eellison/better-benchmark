"""
Standalone repro captured via capture_hook.
Label: hf_RobertaForCausalLM_train
Pattern hash: 21b8c62fd9b0
Shape hash: 947d7429
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([50268, 768], f32), T([16384, 50265], f32), T([32, 512, 768], f32), T([32, 512, 768], f32), T([16384, 768], f32), T([32, 512, 768], f32), T([32, 512, 768], f32), T([16384, 768], f32), T([16384, 3072], f32), T([32, 512, 768], f32), T([32, 512, 768], f32), T([16384, 768], f32), T([16384, 768], f32), T([16384, 768], f32), T([16384, 768], f32), T([32, 512, 768], f32), T([32, 512, 768], f32), T([16384, 768], f32), T([16384, 3072], f32), T([32, 512, 768], f32), T([32, 512, 768], f32), T([16384, 768], f32), T([16384, 768], f32), T([16384, 768], f32), T([16384, 768], f32), T([32, 512, 768], f32), T([32, 512, 768], f32), T([16384, 768], f32), T([16384, 3072], f32), T([32, 512, 768], f32), T([32, 512, 768], f32), T([16384, 768], f32), T([16384, 768], f32), T([16384, 768], f32), T([16384, 768], f32), T([32, 512, 768], f32), T([32, 512, 768], f32), T([16384, 768], f32), T([16384, 3072], f32), T([32, 512, 768], f32), T([32, 512, 768], f32), T([16384, 768], f32), T([16384, 768], f32), T([16384, 768], f32), T([16384, 768], f32), T([32, 512, 768], f32), T([32, 512, 768], f32), T([16384, 768], f32), T([16384, 3072], f32), T([32, 512, 768], f32), T([32, 512, 768], f32), T([16384, 768], f32), T([16384, 768], f32), T([16384, 768], f32), T([16384, 768], f32), T([32, 512, 768], f32), T([32, 512, 768], f32), T([16384, 768], f32), T([16384, 3072], f32), T([32, 512, 768], f32), T([32, 512, 768], f32), T([16384, 768], f32), T([16384, 768], f32), T([16384, 768], f32), T([16384, 768], f32), T([32, 512, 768], f32), T([32, 512, 768], f32), T([16384, 768], f32), T([16384, 3072], f32), T([32, 512, 768], f32), T([32, 512, 768], f32), T([16384, 768], f32), T([16384, 768], f32), T([16384, 768], f32), T([16384, 768], f32), T([32, 512, 768], f32), T([32, 512, 768], f32), T([16384, 768], f32), T([16384, 3072], f32), T([32, 512, 768], f32), T([32, 512, 768], f32), T([16384, 768], f32), T([16384, 768], f32), T([16384, 768], f32), T([16384, 768], f32), T([32, 512, 768], f32), T([32, 512, 768], f32), T([16384, 768], f32), T([16384, 3072], f32), T([32, 512, 768], f32), T([32, 512, 768], f32), T([16384, 768], f32), T([16384, 768], f32), T([16384, 768], f32), T([16384, 768], f32), T([32, 512, 768], f32), T([32, 512, 768], f32), T([16384, 768], f32), T([16384, 3072], f32), T([32, 512, 768], f32), T([32, 512, 768], f32), T([16384, 768], f32), T([16384, 768], f32), T([16384, 768], f32), T([16384, 768], f32), T([32, 512, 768], f32), T([32, 512, 768], f32), T([16384, 768], f32), T([16384, 3072], f32), T([32, 512, 768], f32), T([32, 512, 768], f32), T([16384, 768], f32), T([16384, 768], f32), T([16384, 768], f32), T([16384, 768], f32), T([32, 512, 768], f32), T([32, 512, 768], f32), T([16384, 768], f32), T([16384, 3072], f32), T([32, 512, 768], f32), T([32, 512, 768], f32), T([16384, 768], f32), T([16384, 768], f32), T([16384, 768], f32), T([32, 512, 768], f32), T([16384, 768], f32), T([16384, 768], f32), T([16384, 768], f32), T([16384, 768], f32), T([32, 512, 768], b8), T([768], f32), T([32, 512, 768], f32), T([32, 512, 1], f32), T([32, 512], i64, gen=Index(512)), T([], f32), T([32, 512], i64, gen=Index(2)), T([32, 512], i64, gen=Index(50265)), S([50265]), S([768]), S([768]), S([3072]), S([768]), S([768]), S([768]), S([768]), S([768]), S([3072]), S([768]), S([768]), S([768]), S([768]), S([768]), S([3072]), S([768]), S([768]), S([768]), S([768]), S([768]), S([3072]), S([768]), S([768]), S([768]), S([768]), S([768]), S([3072]), S([768]), S([768]), S([768]), S([768]), S([768]), S([3072]), S([768]), S([768]), S([768]), S([768]), S([768]), S([3072]), S([768]), S([768]), S([768]), S([768]), S([768]), S([3072]), S([768]), S([768]), S([768]), S([768]), S([768]), S([3072]), S([768]), S([768]), S([768]), S([768]), S([768]), S([3072]), S([768]), S([768]), S([768]), S([768]), S([768]), S([3072]), S([768]), S([768]), S([768]), S([768]), S([768]), S([3072]), S([768]), S([768]), S([32, 512, 768]), S([768]), S([32, 512, 768]), S([768]), S([32, 512, 768]))"

class Repro(torch.nn.Module):
    def forward(self, mm_default: "f32[50268, 768]", view_271: "f32[16384, 50265]", view_273: "f32[32, 512, 768]", mul_188: "f32[32, 512, 768]", view_274: "f32[16384, 768]", view_276: "f32[32, 512, 768]", mul_183: "f32[32, 512, 768]", view_277: "f32[16384, 768]", view_280: "f32[16384, 3072]", add_112: "f32[32, 512, 768]", mul_176: "f32[32, 512, 768]", view_283: "f32[16384, 768]", view_294: "f32[16384, 768]", view_298: "f32[16384, 768]", view_302: "f32[16384, 768]", add_115: "f32[32, 512, 768]", mul_168: "f32[32, 512, 768]", view_305: "f32[16384, 768]", view_308: "f32[16384, 3072]", add_118: "f32[32, 512, 768]", mul_161: "f32[32, 512, 768]", view_311: "f32[16384, 768]", view_322: "f32[16384, 768]", view_326: "f32[16384, 768]", view_330: "f32[16384, 768]", add_121: "f32[32, 512, 768]", mul_153: "f32[32, 512, 768]", view_333: "f32[16384, 768]", view_336: "f32[16384, 3072]", add_124: "f32[32, 512, 768]", mul_146: "f32[32, 512, 768]", view_339: "f32[16384, 768]", view_350: "f32[16384, 768]", view_354: "f32[16384, 768]", view_358: "f32[16384, 768]", add_127: "f32[32, 512, 768]", mul_138: "f32[32, 512, 768]", view_361: "f32[16384, 768]", view_364: "f32[16384, 3072]", add_130: "f32[32, 512, 768]", mul_131: "f32[32, 512, 768]", view_367: "f32[16384, 768]", view_378: "f32[16384, 768]", view_382: "f32[16384, 768]", view_386: "f32[16384, 768]", add_133: "f32[32, 512, 768]", mul_123: "f32[32, 512, 768]", view_389: "f32[16384, 768]", view_392: "f32[16384, 3072]", add_136: "f32[32, 512, 768]", mul_116: "f32[32, 512, 768]", view_395: "f32[16384, 768]", view_406: "f32[16384, 768]", view_410: "f32[16384, 768]", view_414: "f32[16384, 768]", add_139: "f32[32, 512, 768]", mul_108: "f32[32, 512, 768]", view_417: "f32[16384, 768]", view_420: "f32[16384, 3072]", add_142: "f32[32, 512, 768]", mul_101: "f32[32, 512, 768]", view_423: "f32[16384, 768]", view_434: "f32[16384, 768]", view_438: "f32[16384, 768]", view_442: "f32[16384, 768]", add_145: "f32[32, 512, 768]", mul_93: "f32[32, 512, 768]", view_445: "f32[16384, 768]", view_448: "f32[16384, 3072]", add_148: "f32[32, 512, 768]", mul_86: "f32[32, 512, 768]", view_451: "f32[16384, 768]", view_462: "f32[16384, 768]", view_466: "f32[16384, 768]", view_470: "f32[16384, 768]", add_151: "f32[32, 512, 768]", mul_78: "f32[32, 512, 768]", view_473: "f32[16384, 768]", view_476: "f32[16384, 3072]", add_154: "f32[32, 512, 768]", mul_71: "f32[32, 512, 768]", view_479: "f32[16384, 768]", view_490: "f32[16384, 768]", view_494: "f32[16384, 768]", view_498: "f32[16384, 768]", add_157: "f32[32, 512, 768]", mul_63: "f32[32, 512, 768]", view_501: "f32[16384, 768]", view_504: "f32[16384, 3072]", add_160: "f32[32, 512, 768]", mul_56: "f32[32, 512, 768]", view_507: "f32[16384, 768]", view_518: "f32[16384, 768]", view_522: "f32[16384, 768]", view_526: "f32[16384, 768]", add_163: "f32[32, 512, 768]", mul_48: "f32[32, 512, 768]", view_529: "f32[16384, 768]", view_532: "f32[16384, 3072]", add_166: "f32[32, 512, 768]", mul_41: "f32[32, 512, 768]", view_535: "f32[16384, 768]", view_546: "f32[16384, 768]", view_550: "f32[16384, 768]", view_554: "f32[16384, 768]", add_169: "f32[32, 512, 768]", mul_33: "f32[32, 512, 768]", view_557: "f32[16384, 768]", view_560: "f32[16384, 3072]", add_172: "f32[32, 512, 768]", mul_26: "f32[32, 512, 768]", view_563: "f32[16384, 768]", view_574: "f32[16384, 768]", view_578: "f32[16384, 768]", view_582: "f32[16384, 768]", add_175: "f32[32, 512, 768]", mul_18: "f32[32, 512, 768]", view_585: "f32[16384, 768]", view_588: "f32[16384, 3072]", add_178: "f32[32, 512, 768]", mul_11: "f32[32, 512, 768]", view_591: "f32[16384, 768]", view_602: "f32[16384, 768]", mm_142: "f32[16384, 768]", mul_557: "f32[32, 512, 768]", view_606: "f32[16384, 768]", mm_144: "f32[16384, 768]", view_610: "f32[16384, 768]", mm_146: "f32[16384, 768]", gt: "b8[32, 512, 768]", primals_7: "f32[768]", mul_1: "f32[32, 512, 768]", div_39: "f32[32, 512, 1]", add_1: "i64[32, 512]", full_default_1: "f32[]", expand_1: "i64[32, 512]", primals_2: "i64[32, 512]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12, _shape_param_13, _shape_param_14, _shape_param_15, _shape_param_16, _shape_param_17, _shape_param_18, _shape_param_19, _shape_param_20, _shape_param_21, _shape_param_22, _shape_param_23, _shape_param_24, _shape_param_25, _shape_param_26, _shape_param_27, _shape_param_28, _shape_param_29, _shape_param_30, _shape_param_31, _shape_param_32, _shape_param_33, _shape_param_34, _shape_param_35, _shape_param_36, _shape_param_37, _shape_param_38, _shape_param_39, _shape_param_40, _shape_param_41, _shape_param_42, _shape_param_43, _shape_param_44, _shape_param_45, _shape_param_46, _shape_param_47, _shape_param_48, _shape_param_49, _shape_param_50, _shape_param_51, _shape_param_52, _shape_param_53, _shape_param_54, _shape_param_55, _shape_param_56, _shape_param_57, _shape_param_58, _shape_param_59, _shape_param_60, _shape_param_61, _shape_param_62, _shape_param_63, _shape_param_64, _shape_param_65, _shape_param_66, _shape_param_67, _shape_param_68, _shape_param_69, _shape_param_70, _shape_param_71, _shape_param_72, _shape_param_73, _shape_param_74, _shape_param_75, _shape_param_76):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:898 in forward, code: x = self.decoder(x)
        slice_tensor: "f32[50265, 768]" = torch.ops.aten.slice.Tensor(mm_default, 0, 0, -3);  mm_default = None
        sum_dim_int_list: "f32[1, 50265]" = torch.ops.aten.sum.dim_IntList(view_271, [0], True);  view_271 = None
        reshape_default: "f32[50265]" = torch.ops.aten.reshape.default(sum_dim_int_list, _shape_param_0);  sum_dim_int_list = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:895 in forward, code: x = self.layer_norm(x)
        mul_tensor: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(view_273, mul_188);  mul_188 = None
        sum_dim_int_list_1: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [0, 1]);  mul_tensor = None
        sum_dim_int_list_2: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_273, [0, 1]);  view_273 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:893 in forward, code: x = self.dense(features)
        permute_default: "f32[768, 16384]" = torch.ops.aten.permute.default(view_274, [1, 0])
        sum_dim_int_list_3: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_274, [0], True);  view_274 = None
        reshape_default_1: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_3, _shape_param_1);  sum_dim_int_list_3 = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:401 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_tensor_1: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(view_276, mul_183);  mul_183 = None
        sum_dim_int_list_4: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_1, [0, 1]);  mul_tensor_1 = None
        sum_dim_int_list_5: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_276, [0, 1]);  view_276 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:399 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_1: "f32[768, 16384]" = torch.ops.aten.permute.default(view_277, [1, 0])
        sum_dim_int_list_6: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_277, [0], True);  view_277 = None
        reshape_default_2: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_6, _shape_param_2);  sum_dim_int_list_6 = _shape_param_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:386 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_2: "f32[3072, 16384]" = torch.ops.aten.permute.default(view_280, [1, 0])
        sum_dim_int_list_7: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_280, [0], True);  view_280 = None
        reshape_default_3: "f32[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_7, _shape_param_3);  sum_dim_int_list_7 = _shape_param_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:343 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_tensor_2: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_112, mul_176);  mul_176 = None
        sum_dim_int_list_8: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [0, 1]);  mul_tensor_2 = None
        sum_dim_int_list_9: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_112, [0, 1]);  add_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:341 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_3: "f32[768, 16384]" = torch.ops.aten.permute.default(view_283, [1, 0])
        sum_dim_int_list_10: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_283, [0], True);  view_283 = None
        reshape_default_4: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_10, _shape_param_4);  sum_dim_int_list_10 = _shape_param_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:228 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_default_4: "f32[768, 16384]" = torch.ops.aten.permute.default(view_294, [1, 0])
        sum_dim_int_list_11: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_294, [0], True);  view_294 = None
        reshape_default_5: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_11, _shape_param_5);  sum_dim_int_list_11 = _shape_param_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:227 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_default_5: "f32[768, 16384]" = torch.ops.aten.permute.default(view_298, [1, 0])
        sum_dim_int_list_12: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_298, [0], True);  view_298 = None
        reshape_default_6: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_12, _shape_param_6);  sum_dim_int_list_12 = _shape_param_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:226 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_default_6: "f32[768, 16384]" = torch.ops.aten.permute.default(view_302, [1, 0])
        sum_dim_int_list_13: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_302, [0], True);  view_302 = None
        reshape_default_7: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_13, _shape_param_7);  sum_dim_int_list_13 = _shape_param_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:401 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_tensor_3: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_115, mul_168);  mul_168 = None
        sum_dim_int_list_14: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_3, [0, 1]);  mul_tensor_3 = None
        sum_dim_int_list_15: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_115, [0, 1]);  add_115 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:399 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_7: "f32[768, 16384]" = torch.ops.aten.permute.default(view_305, [1, 0])
        sum_dim_int_list_16: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_305, [0], True);  view_305 = None
        reshape_default_8: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_16, _shape_param_8);  sum_dim_int_list_16 = _shape_param_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:386 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_8: "f32[3072, 16384]" = torch.ops.aten.permute.default(view_308, [1, 0])
        sum_dim_int_list_17: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_308, [0], True);  view_308 = None
        reshape_default_9: "f32[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_17, _shape_param_9);  sum_dim_int_list_17 = _shape_param_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:343 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_tensor_4: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_118, mul_161);  mul_161 = None
        sum_dim_int_list_18: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_4, [0, 1]);  mul_tensor_4 = None
        sum_dim_int_list_19: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_118, [0, 1]);  add_118 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:341 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_9: "f32[768, 16384]" = torch.ops.aten.permute.default(view_311, [1, 0])
        sum_dim_int_list_20: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_311, [0], True);  view_311 = None
        reshape_default_10: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_20, _shape_param_10);  sum_dim_int_list_20 = _shape_param_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:228 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_default_10: "f32[768, 16384]" = torch.ops.aten.permute.default(view_322, [1, 0])
        sum_dim_int_list_21: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_322, [0], True);  view_322 = None
        reshape_default_11: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_21, _shape_param_11);  sum_dim_int_list_21 = _shape_param_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:227 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_default_11: "f32[768, 16384]" = torch.ops.aten.permute.default(view_326, [1, 0])
        sum_dim_int_list_22: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_326, [0], True);  view_326 = None
        reshape_default_12: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_22, _shape_param_12);  sum_dim_int_list_22 = _shape_param_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:226 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_default_12: "f32[768, 16384]" = torch.ops.aten.permute.default(view_330, [1, 0])
        sum_dim_int_list_23: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_330, [0], True);  view_330 = None
        reshape_default_13: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_23, _shape_param_13);  sum_dim_int_list_23 = _shape_param_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:401 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_tensor_5: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_121, mul_153);  mul_153 = None
        sum_dim_int_list_24: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_5, [0, 1]);  mul_tensor_5 = None
        sum_dim_int_list_25: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_121, [0, 1]);  add_121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:399 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_13: "f32[768, 16384]" = torch.ops.aten.permute.default(view_333, [1, 0])
        sum_dim_int_list_26: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_333, [0], True);  view_333 = None
        reshape_default_14: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_26, _shape_param_14);  sum_dim_int_list_26 = _shape_param_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:386 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_14: "f32[3072, 16384]" = torch.ops.aten.permute.default(view_336, [1, 0])
        sum_dim_int_list_27: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_336, [0], True);  view_336 = None
        reshape_default_15: "f32[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_27, _shape_param_15);  sum_dim_int_list_27 = _shape_param_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:343 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_tensor_6: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_124, mul_146);  mul_146 = None
        sum_dim_int_list_28: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_6, [0, 1]);  mul_tensor_6 = None
        sum_dim_int_list_29: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_124, [0, 1]);  add_124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:341 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_15: "f32[768, 16384]" = torch.ops.aten.permute.default(view_339, [1, 0])
        sum_dim_int_list_30: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_339, [0], True);  view_339 = None
        reshape_default_16: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_30, _shape_param_16);  sum_dim_int_list_30 = _shape_param_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:228 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_default_16: "f32[768, 16384]" = torch.ops.aten.permute.default(view_350, [1, 0])
        sum_dim_int_list_31: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_350, [0], True);  view_350 = None
        reshape_default_17: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_31, _shape_param_17);  sum_dim_int_list_31 = _shape_param_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:227 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_default_17: "f32[768, 16384]" = torch.ops.aten.permute.default(view_354, [1, 0])
        sum_dim_int_list_32: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_354, [0], True);  view_354 = None
        reshape_default_18: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_32, _shape_param_18);  sum_dim_int_list_32 = _shape_param_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:226 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_default_18: "f32[768, 16384]" = torch.ops.aten.permute.default(view_358, [1, 0])
        sum_dim_int_list_33: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_358, [0], True);  view_358 = None
        reshape_default_19: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_33, _shape_param_19);  sum_dim_int_list_33 = _shape_param_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:401 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_tensor_7: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_127, mul_138);  mul_138 = None
        sum_dim_int_list_34: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_7, [0, 1]);  mul_tensor_7 = None
        sum_dim_int_list_35: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_127, [0, 1]);  add_127 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:399 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_19: "f32[768, 16384]" = torch.ops.aten.permute.default(view_361, [1, 0])
        sum_dim_int_list_36: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_361, [0], True);  view_361 = None
        reshape_default_20: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_36, _shape_param_20);  sum_dim_int_list_36 = _shape_param_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:386 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_20: "f32[3072, 16384]" = torch.ops.aten.permute.default(view_364, [1, 0])
        sum_dim_int_list_37: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_364, [0], True);  view_364 = None
        reshape_default_21: "f32[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_37, _shape_param_21);  sum_dim_int_list_37 = _shape_param_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:343 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_tensor_8: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_130, mul_131);  mul_131 = None
        sum_dim_int_list_38: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_8, [0, 1]);  mul_tensor_8 = None
        sum_dim_int_list_39: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_130, [0, 1]);  add_130 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:341 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_21: "f32[768, 16384]" = torch.ops.aten.permute.default(view_367, [1, 0])
        sum_dim_int_list_40: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_367, [0], True);  view_367 = None
        reshape_default_22: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_40, _shape_param_22);  sum_dim_int_list_40 = _shape_param_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:228 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_default_22: "f32[768, 16384]" = torch.ops.aten.permute.default(view_378, [1, 0])
        sum_dim_int_list_41: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_378, [0], True);  view_378 = None
        reshape_default_23: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_41, _shape_param_23);  sum_dim_int_list_41 = _shape_param_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:227 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_default_23: "f32[768, 16384]" = torch.ops.aten.permute.default(view_382, [1, 0])
        sum_dim_int_list_42: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_382, [0], True);  view_382 = None
        reshape_default_24: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_42, _shape_param_24);  sum_dim_int_list_42 = _shape_param_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:226 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_default_24: "f32[768, 16384]" = torch.ops.aten.permute.default(view_386, [1, 0])
        sum_dim_int_list_43: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_386, [0], True);  view_386 = None
        reshape_default_25: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_43, _shape_param_25);  sum_dim_int_list_43 = _shape_param_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:401 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_tensor_9: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_133, mul_123);  mul_123 = None
        sum_dim_int_list_44: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_9, [0, 1]);  mul_tensor_9 = None
        sum_dim_int_list_45: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_133, [0, 1]);  add_133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:399 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_25: "f32[768, 16384]" = torch.ops.aten.permute.default(view_389, [1, 0])
        sum_dim_int_list_46: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_389, [0], True);  view_389 = None
        reshape_default_26: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_46, _shape_param_26);  sum_dim_int_list_46 = _shape_param_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:386 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_26: "f32[3072, 16384]" = torch.ops.aten.permute.default(view_392, [1, 0])
        sum_dim_int_list_47: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_392, [0], True);  view_392 = None
        reshape_default_27: "f32[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_47, _shape_param_27);  sum_dim_int_list_47 = _shape_param_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:343 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_tensor_10: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_136, mul_116);  mul_116 = None
        sum_dim_int_list_48: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_10, [0, 1]);  mul_tensor_10 = None
        sum_dim_int_list_49: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_136, [0, 1]);  add_136 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:341 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_27: "f32[768, 16384]" = torch.ops.aten.permute.default(view_395, [1, 0])
        sum_dim_int_list_50: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_395, [0], True);  view_395 = None
        reshape_default_28: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_50, _shape_param_28);  sum_dim_int_list_50 = _shape_param_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:228 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_default_28: "f32[768, 16384]" = torch.ops.aten.permute.default(view_406, [1, 0])
        sum_dim_int_list_51: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_406, [0], True);  view_406 = None
        reshape_default_29: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_51, _shape_param_29);  sum_dim_int_list_51 = _shape_param_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:227 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_default_29: "f32[768, 16384]" = torch.ops.aten.permute.default(view_410, [1, 0])
        sum_dim_int_list_52: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_410, [0], True);  view_410 = None
        reshape_default_30: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_52, _shape_param_30);  sum_dim_int_list_52 = _shape_param_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:226 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_default_30: "f32[768, 16384]" = torch.ops.aten.permute.default(view_414, [1, 0])
        sum_dim_int_list_53: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_414, [0], True);  view_414 = None
        reshape_default_31: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_53, _shape_param_31);  sum_dim_int_list_53 = _shape_param_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:401 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_tensor_11: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_139, mul_108);  mul_108 = None
        sum_dim_int_list_54: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_11, [0, 1]);  mul_tensor_11 = None
        sum_dim_int_list_55: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_139, [0, 1]);  add_139 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:399 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_31: "f32[768, 16384]" = torch.ops.aten.permute.default(view_417, [1, 0])
        sum_dim_int_list_56: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_417, [0], True);  view_417 = None
        reshape_default_32: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_56, _shape_param_32);  sum_dim_int_list_56 = _shape_param_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:386 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_32: "f32[3072, 16384]" = torch.ops.aten.permute.default(view_420, [1, 0])
        sum_dim_int_list_57: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_420, [0], True);  view_420 = None
        reshape_default_33: "f32[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_57, _shape_param_33);  sum_dim_int_list_57 = _shape_param_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:343 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_tensor_12: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_142, mul_101);  mul_101 = None
        sum_dim_int_list_58: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_12, [0, 1]);  mul_tensor_12 = None
        sum_dim_int_list_59: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_142, [0, 1]);  add_142 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:341 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_33: "f32[768, 16384]" = torch.ops.aten.permute.default(view_423, [1, 0])
        sum_dim_int_list_60: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_423, [0], True);  view_423 = None
        reshape_default_34: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_60, _shape_param_34);  sum_dim_int_list_60 = _shape_param_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:228 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_default_34: "f32[768, 16384]" = torch.ops.aten.permute.default(view_434, [1, 0])
        sum_dim_int_list_61: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_434, [0], True);  view_434 = None
        reshape_default_35: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_61, _shape_param_35);  sum_dim_int_list_61 = _shape_param_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:227 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_default_35: "f32[768, 16384]" = torch.ops.aten.permute.default(view_438, [1, 0])
        sum_dim_int_list_62: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_438, [0], True);  view_438 = None
        reshape_default_36: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_62, _shape_param_36);  sum_dim_int_list_62 = _shape_param_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:226 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_default_36: "f32[768, 16384]" = torch.ops.aten.permute.default(view_442, [1, 0])
        sum_dim_int_list_63: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_442, [0], True);  view_442 = None
        reshape_default_37: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_63, _shape_param_37);  sum_dim_int_list_63 = _shape_param_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:401 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_tensor_13: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_145, mul_93);  mul_93 = None
        sum_dim_int_list_64: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_13, [0, 1]);  mul_tensor_13 = None
        sum_dim_int_list_65: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_145, [0, 1]);  add_145 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:399 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_37: "f32[768, 16384]" = torch.ops.aten.permute.default(view_445, [1, 0])
        sum_dim_int_list_66: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_445, [0], True);  view_445 = None
        reshape_default_38: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_66, _shape_param_38);  sum_dim_int_list_66 = _shape_param_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:386 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_38: "f32[3072, 16384]" = torch.ops.aten.permute.default(view_448, [1, 0])
        sum_dim_int_list_67: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_448, [0], True);  view_448 = None
        reshape_default_39: "f32[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_67, _shape_param_39);  sum_dim_int_list_67 = _shape_param_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:343 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_tensor_14: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_148, mul_86);  mul_86 = None
        sum_dim_int_list_68: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_14, [0, 1]);  mul_tensor_14 = None
        sum_dim_int_list_69: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_148, [0, 1]);  add_148 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:341 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_39: "f32[768, 16384]" = torch.ops.aten.permute.default(view_451, [1, 0])
        sum_dim_int_list_70: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_451, [0], True);  view_451 = None
        reshape_default_40: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_70, _shape_param_40);  sum_dim_int_list_70 = _shape_param_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:228 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_default_40: "f32[768, 16384]" = torch.ops.aten.permute.default(view_462, [1, 0])
        sum_dim_int_list_71: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_462, [0], True);  view_462 = None
        reshape_default_41: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_71, _shape_param_41);  sum_dim_int_list_71 = _shape_param_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:227 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_default_41: "f32[768, 16384]" = torch.ops.aten.permute.default(view_466, [1, 0])
        sum_dim_int_list_72: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_466, [0], True);  view_466 = None
        reshape_default_42: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_72, _shape_param_42);  sum_dim_int_list_72 = _shape_param_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:226 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_default_42: "f32[768, 16384]" = torch.ops.aten.permute.default(view_470, [1, 0])
        sum_dim_int_list_73: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_470, [0], True);  view_470 = None
        reshape_default_43: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_73, _shape_param_43);  sum_dim_int_list_73 = _shape_param_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:401 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_tensor_15: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_151, mul_78);  mul_78 = None
        sum_dim_int_list_74: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_15, [0, 1]);  mul_tensor_15 = None
        sum_dim_int_list_75: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_151, [0, 1]);  add_151 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:399 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_43: "f32[768, 16384]" = torch.ops.aten.permute.default(view_473, [1, 0])
        sum_dim_int_list_76: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_473, [0], True);  view_473 = None
        reshape_default_44: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_76, _shape_param_44);  sum_dim_int_list_76 = _shape_param_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:386 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_44: "f32[3072, 16384]" = torch.ops.aten.permute.default(view_476, [1, 0])
        sum_dim_int_list_77: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_476, [0], True);  view_476 = None
        reshape_default_45: "f32[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_77, _shape_param_45);  sum_dim_int_list_77 = _shape_param_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:343 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_tensor_16: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_154, mul_71);  mul_71 = None
        sum_dim_int_list_78: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_16, [0, 1]);  mul_tensor_16 = None
        sum_dim_int_list_79: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_154, [0, 1]);  add_154 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:341 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_45: "f32[768, 16384]" = torch.ops.aten.permute.default(view_479, [1, 0])
        sum_dim_int_list_80: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_479, [0], True);  view_479 = None
        reshape_default_46: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_80, _shape_param_46);  sum_dim_int_list_80 = _shape_param_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:228 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_default_46: "f32[768, 16384]" = torch.ops.aten.permute.default(view_490, [1, 0])
        sum_dim_int_list_81: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_490, [0], True);  view_490 = None
        reshape_default_47: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_81, _shape_param_47);  sum_dim_int_list_81 = _shape_param_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:227 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_default_47: "f32[768, 16384]" = torch.ops.aten.permute.default(view_494, [1, 0])
        sum_dim_int_list_82: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_494, [0], True);  view_494 = None
        reshape_default_48: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_82, _shape_param_48);  sum_dim_int_list_82 = _shape_param_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:226 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_default_48: "f32[768, 16384]" = torch.ops.aten.permute.default(view_498, [1, 0])
        sum_dim_int_list_83: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_498, [0], True);  view_498 = None
        reshape_default_49: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_83, _shape_param_49);  sum_dim_int_list_83 = _shape_param_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:401 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_tensor_17: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_157, mul_63);  mul_63 = None
        sum_dim_int_list_84: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_17, [0, 1]);  mul_tensor_17 = None
        sum_dim_int_list_85: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_157, [0, 1]);  add_157 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:399 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_49: "f32[768, 16384]" = torch.ops.aten.permute.default(view_501, [1, 0])
        sum_dim_int_list_86: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_501, [0], True);  view_501 = None
        reshape_default_50: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_86, _shape_param_50);  sum_dim_int_list_86 = _shape_param_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:386 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_50: "f32[3072, 16384]" = torch.ops.aten.permute.default(view_504, [1, 0])
        sum_dim_int_list_87: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_504, [0], True);  view_504 = None
        reshape_default_51: "f32[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_87, _shape_param_51);  sum_dim_int_list_87 = _shape_param_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:343 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_tensor_18: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_160, mul_56);  mul_56 = None
        sum_dim_int_list_88: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_18, [0, 1]);  mul_tensor_18 = None
        sum_dim_int_list_89: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_160, [0, 1]);  add_160 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:341 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_51: "f32[768, 16384]" = torch.ops.aten.permute.default(view_507, [1, 0])
        sum_dim_int_list_90: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_507, [0], True);  view_507 = None
        reshape_default_52: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_90, _shape_param_52);  sum_dim_int_list_90 = _shape_param_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:228 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_default_52: "f32[768, 16384]" = torch.ops.aten.permute.default(view_518, [1, 0])
        sum_dim_int_list_91: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_518, [0], True);  view_518 = None
        reshape_default_53: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_91, _shape_param_53);  sum_dim_int_list_91 = _shape_param_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:227 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_default_53: "f32[768, 16384]" = torch.ops.aten.permute.default(view_522, [1, 0])
        sum_dim_int_list_92: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_522, [0], True);  view_522 = None
        reshape_default_54: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_92, _shape_param_54);  sum_dim_int_list_92 = _shape_param_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:226 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_default_54: "f32[768, 16384]" = torch.ops.aten.permute.default(view_526, [1, 0])
        sum_dim_int_list_93: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_526, [0], True);  view_526 = None
        reshape_default_55: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_93, _shape_param_55);  sum_dim_int_list_93 = _shape_param_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:401 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_tensor_19: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_163, mul_48);  mul_48 = None
        sum_dim_int_list_94: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_19, [0, 1]);  mul_tensor_19 = None
        sum_dim_int_list_95: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_163, [0, 1]);  add_163 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:399 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_55: "f32[768, 16384]" = torch.ops.aten.permute.default(view_529, [1, 0])
        sum_dim_int_list_96: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_529, [0], True);  view_529 = None
        reshape_default_56: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_96, _shape_param_56);  sum_dim_int_list_96 = _shape_param_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:386 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_56: "f32[3072, 16384]" = torch.ops.aten.permute.default(view_532, [1, 0])
        sum_dim_int_list_97: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_532, [0], True);  view_532 = None
        reshape_default_57: "f32[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_97, _shape_param_57);  sum_dim_int_list_97 = _shape_param_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:343 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_tensor_20: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_166, mul_41);  mul_41 = None
        sum_dim_int_list_98: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_20, [0, 1]);  mul_tensor_20 = None
        sum_dim_int_list_99: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_166, [0, 1]);  add_166 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:341 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_57: "f32[768, 16384]" = torch.ops.aten.permute.default(view_535, [1, 0])
        sum_dim_int_list_100: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_535, [0], True);  view_535 = None
        reshape_default_58: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_100, _shape_param_58);  sum_dim_int_list_100 = _shape_param_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:228 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_default_58: "f32[768, 16384]" = torch.ops.aten.permute.default(view_546, [1, 0])
        sum_dim_int_list_101: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_546, [0], True);  view_546 = None
        reshape_default_59: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_101, _shape_param_59);  sum_dim_int_list_101 = _shape_param_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:227 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_default_59: "f32[768, 16384]" = torch.ops.aten.permute.default(view_550, [1, 0])
        sum_dim_int_list_102: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_550, [0], True);  view_550 = None
        reshape_default_60: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_102, _shape_param_60);  sum_dim_int_list_102 = _shape_param_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:226 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_default_60: "f32[768, 16384]" = torch.ops.aten.permute.default(view_554, [1, 0])
        sum_dim_int_list_103: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_554, [0], True);  view_554 = None
        reshape_default_61: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_103, _shape_param_61);  sum_dim_int_list_103 = _shape_param_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:401 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_tensor_21: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_169, mul_33);  mul_33 = None
        sum_dim_int_list_104: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_21, [0, 1]);  mul_tensor_21 = None
        sum_dim_int_list_105: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_169, [0, 1]);  add_169 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:399 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_61: "f32[768, 16384]" = torch.ops.aten.permute.default(view_557, [1, 0])
        sum_dim_int_list_106: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_557, [0], True);  view_557 = None
        reshape_default_62: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_106, _shape_param_62);  sum_dim_int_list_106 = _shape_param_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:386 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_62: "f32[3072, 16384]" = torch.ops.aten.permute.default(view_560, [1, 0])
        sum_dim_int_list_107: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_560, [0], True);  view_560 = None
        reshape_default_63: "f32[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_107, _shape_param_63);  sum_dim_int_list_107 = _shape_param_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:343 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_tensor_22: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_172, mul_26);  mul_26 = None
        sum_dim_int_list_108: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_22, [0, 1]);  mul_tensor_22 = None
        sum_dim_int_list_109: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_172, [0, 1]);  add_172 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:341 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_63: "f32[768, 16384]" = torch.ops.aten.permute.default(view_563, [1, 0])
        sum_dim_int_list_110: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_563, [0], True);  view_563 = None
        reshape_default_64: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_110, _shape_param_64);  sum_dim_int_list_110 = _shape_param_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:228 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_default_64: "f32[768, 16384]" = torch.ops.aten.permute.default(view_574, [1, 0])
        sum_dim_int_list_111: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_574, [0], True);  view_574 = None
        reshape_default_65: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_111, _shape_param_65);  sum_dim_int_list_111 = _shape_param_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:227 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_default_65: "f32[768, 16384]" = torch.ops.aten.permute.default(view_578, [1, 0])
        sum_dim_int_list_112: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_578, [0], True);  view_578 = None
        reshape_default_66: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_112, _shape_param_66);  sum_dim_int_list_112 = _shape_param_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:226 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_default_66: "f32[768, 16384]" = torch.ops.aten.permute.default(view_582, [1, 0])
        sum_dim_int_list_113: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_582, [0], True);  view_582 = None
        reshape_default_67: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_113, _shape_param_67);  sum_dim_int_list_113 = _shape_param_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:401 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_tensor_23: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_175, mul_18);  mul_18 = None
        sum_dim_int_list_114: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_23, [0, 1]);  mul_tensor_23 = None
        sum_dim_int_list_115: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_175, [0, 1]);  add_175 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:399 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_67: "f32[768, 16384]" = torch.ops.aten.permute.default(view_585, [1, 0])
        sum_dim_int_list_116: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_585, [0], True);  view_585 = None
        reshape_default_68: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_116, _shape_param_68);  sum_dim_int_list_116 = _shape_param_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:386 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_68: "f32[3072, 16384]" = torch.ops.aten.permute.default(view_588, [1, 0])
        sum_dim_int_list_117: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_588, [0], True);  view_588 = None
        reshape_default_69: "f32[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_117, _shape_param_69);  sum_dim_int_list_117 = _shape_param_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:343 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_tensor_24: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_178, mul_11);  mul_11 = None
        sum_dim_int_list_118: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_24, [0, 1]);  mul_tensor_24 = None
        sum_dim_int_list_119: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_178, [0, 1]);  add_178 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:341 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_69: "f32[768, 16384]" = torch.ops.aten.permute.default(view_591, [1, 0])
        sum_dim_int_list_120: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_591, [0], True);  view_591 = None
        reshape_default_70: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_120, _shape_param_70);  sum_dim_int_list_120 = _shape_param_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:228 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_default_70: "f32[768, 16384]" = torch.ops.aten.permute.default(view_602, [1, 0])
        sum_dim_int_list_121: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_602, [0], True);  view_602 = None
        reshape_default_71: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_121, _shape_param_71);  sum_dim_int_list_121 = _shape_param_71 = None
        reshape_default_72: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_142, _shape_param_72);  mm_142 = _shape_param_72 = None
        add_tensor: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_557, reshape_default_72);  mul_557 = reshape_default_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:227 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_default_71: "f32[768, 16384]" = torch.ops.aten.permute.default(view_606, [1, 0])
        sum_dim_int_list_122: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_606, [0], True);  view_606 = None
        reshape_default_73: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_122, _shape_param_73);  sum_dim_int_list_122 = _shape_param_73 = None
        reshape_default_74: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_144, _shape_param_74);  mm_144 = _shape_param_74 = None
        add_tensor_1: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_tensor, reshape_default_74);  add_tensor = reshape_default_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:226 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_default_72: "f32[768, 16384]" = torch.ops.aten.permute.default(view_610, [1, 0])
        sum_dim_int_list_123: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_610, [0], True);  view_610 = None
        reshape_default_75: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_123, _shape_param_75);  sum_dim_int_list_123 = _shape_param_75 = None
        reshape_default_76: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_146, _shape_param_76);  mm_146 = _shape_param_76 = None
        add_tensor_2: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_tensor_1, reshape_default_76);  add_tensor_1 = reshape_default_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:124 in forward, code: embeddings = self.dropout(embeddings)
        convert_element_type_default: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(gt, torch.float32);  gt = None
        mul_tensor_25: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 1.1111111111111112);  convert_element_type_default = None
        mul_tensor_26: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_tensor_2, mul_tensor_25);  add_tensor_2 = mul_tensor_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:123 in forward, code: embeddings = self.LayerNorm(embeddings)
        mul_tensor_27: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_26, primals_7);  primals_7 = None
        mul_tensor_28: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_27, 768)
        sum_dim_int_list_124: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_27, [2], True)
        mul_tensor_29: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_27, mul_1);  mul_tensor_27 = None
        sum_dim_int_list_125: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_29, [2], True);  mul_tensor_29 = None
        mul_tensor_30: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_1, sum_dim_int_list_125);  sum_dim_int_list_125 = None
        sub_tensor: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_tensor_28, sum_dim_int_list_124);  mul_tensor_28 = sum_dim_int_list_124 = None
        sub_tensor_1: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub_tensor, mul_tensor_30);  sub_tensor = mul_tensor_30 = None
        mul_tensor_31: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(div_39, sub_tensor_1);  div_39 = sub_tensor_1 = None
        mul_tensor_32: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_26, mul_1);  mul_1 = None
        sum_dim_int_list_126: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_32, [0, 1]);  mul_tensor_32 = None
        sum_dim_int_list_127: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_26, [0, 1]);  mul_tensor_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:120 in forward, code: position_embeddings = self.position_embeddings(position_ids)
        eq_scalar: "b8[32, 512]" = torch.ops.aten.eq.Scalar(add_1, 0)
        unsqueeze_default: "b8[32, 512, 1]" = torch.ops.aten.unsqueeze.default(eq_scalar, -1);  eq_scalar = None
        where_self: "f32[32, 512, 768]" = torch.ops.aten.where.self(unsqueeze_default, full_default_1, mul_tensor_31);  unsqueeze_default = None
        full_default: "f32[512, 768]" = torch.ops.aten.full.default([512, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default: "f32[512, 768]" = torch.ops.aten.index_put.default(full_default, [add_1], where_self, True);  full_default = add_1 = where_self = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:117 in forward, code: token_type_embeddings = self.token_type_embeddings(token_type_ids)
        eq_scalar_1: "b8[32, 512]" = torch.ops.aten.eq.Scalar(expand_1, -1)
        unsqueeze_default_1: "b8[32, 512, 1]" = torch.ops.aten.unsqueeze.default(eq_scalar_1, -1);  eq_scalar_1 = None
        where_self_1: "f32[32, 512, 768]" = torch.ops.aten.where.self(unsqueeze_default_1, full_default_1, mul_tensor_31);  unsqueeze_default_1 = None
        full_default_2: "f32[2, 768]" = torch.ops.aten.full.default([2, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default_1: "f32[2, 768]" = torch.ops.aten.index_put.default(full_default_2, [expand_1], where_self_1, True);  full_default_2 = expand_1 = where_self_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:116 in forward, code: inputs_embeds = self.word_embeddings(input_ids)
        eq_scalar_2: "b8[32, 512]" = torch.ops.aten.eq.Scalar(primals_2, 0)
        unsqueeze_default_2: "b8[32, 512, 1]" = torch.ops.aten.unsqueeze.default(eq_scalar_2, -1);  eq_scalar_2 = None
        where_self_2: "f32[32, 512, 768]" = torch.ops.aten.where.self(unsqueeze_default_2, full_default_1, mul_tensor_31);  unsqueeze_default_2 = full_default_1 = mul_tensor_31 = None
        full_default_3: "f32[50265, 768]" = torch.ops.aten.full.default([50265, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default_2: "f32[50265, 768]" = torch.ops.aten.index_put.default(full_default_3, [primals_2], where_self_2, True);  full_default_3 = primals_2 = where_self_2 = None
        add_tensor_3: "f32[50265, 768]" = torch.ops.aten.add.Tensor(slice_tensor, index_put_default_2);  slice_tensor = index_put_default_2 = None
        return (reshape_default, sum_dim_int_list_1, sum_dim_int_list_2, permute_default, reshape_default_1, sum_dim_int_list_4, sum_dim_int_list_5, permute_default_1, reshape_default_2, permute_default_2, reshape_default_3, sum_dim_int_list_8, sum_dim_int_list_9, permute_default_3, reshape_default_4, permute_default_4, reshape_default_5, permute_default_5, reshape_default_6, permute_default_6, reshape_default_7, sum_dim_int_list_14, sum_dim_int_list_15, permute_default_7, reshape_default_8, permute_default_8, reshape_default_9, sum_dim_int_list_18, sum_dim_int_list_19, permute_default_9, reshape_default_10, permute_default_10, reshape_default_11, permute_default_11, reshape_default_12, permute_default_12, reshape_default_13, sum_dim_int_list_24, sum_dim_int_list_25, permute_default_13, reshape_default_14, permute_default_14, reshape_default_15, sum_dim_int_list_28, sum_dim_int_list_29, permute_default_15, reshape_default_16, permute_default_16, reshape_default_17, permute_default_17, reshape_default_18, permute_default_18, reshape_default_19, sum_dim_int_list_34, sum_dim_int_list_35, permute_default_19, reshape_default_20, permute_default_20, reshape_default_21, sum_dim_int_list_38, sum_dim_int_list_39, permute_default_21, reshape_default_22, permute_default_22, reshape_default_23, permute_default_23, reshape_default_24, permute_default_24, reshape_default_25, sum_dim_int_list_44, sum_dim_int_list_45, permute_default_25, reshape_default_26, permute_default_26, reshape_default_27, sum_dim_int_list_48, sum_dim_int_list_49, permute_default_27, reshape_default_28, permute_default_28, reshape_default_29, permute_default_29, reshape_default_30, permute_default_30, reshape_default_31, sum_dim_int_list_54, sum_dim_int_list_55, permute_default_31, reshape_default_32, permute_default_32, reshape_default_33, sum_dim_int_list_58, sum_dim_int_list_59, permute_default_33, reshape_default_34, permute_default_34, reshape_default_35, permute_default_35, reshape_default_36, permute_default_36, reshape_default_37, sum_dim_int_list_64, sum_dim_int_list_65, permute_default_37, reshape_default_38, permute_default_38, reshape_default_39, sum_dim_int_list_68, sum_dim_int_list_69, permute_default_39, reshape_default_40, permute_default_40, reshape_default_41, permute_default_41, reshape_default_42, permute_default_42, reshape_default_43, sum_dim_int_list_74, sum_dim_int_list_75, permute_default_43, reshape_default_44, permute_default_44, reshape_default_45, sum_dim_int_list_78, sum_dim_int_list_79, permute_default_45, reshape_default_46, permute_default_46, reshape_default_47, permute_default_47, reshape_default_48, permute_default_48, reshape_default_49, sum_dim_int_list_84, sum_dim_int_list_85, permute_default_49, reshape_default_50, permute_default_50, reshape_default_51, sum_dim_int_list_88, sum_dim_int_list_89, permute_default_51, reshape_default_52, permute_default_52, reshape_default_53, permute_default_53, reshape_default_54, permute_default_54, reshape_default_55, sum_dim_int_list_94, sum_dim_int_list_95, permute_default_55, reshape_default_56, permute_default_56, reshape_default_57, sum_dim_int_list_98, sum_dim_int_list_99, permute_default_57, reshape_default_58, permute_default_58, reshape_default_59, permute_default_59, reshape_default_60, permute_default_60, reshape_default_61, sum_dim_int_list_104, sum_dim_int_list_105, permute_default_61, reshape_default_62, permute_default_62, reshape_default_63, sum_dim_int_list_108, sum_dim_int_list_109, permute_default_63, reshape_default_64, permute_default_64, reshape_default_65, permute_default_65, reshape_default_66, permute_default_66, reshape_default_67, sum_dim_int_list_114, sum_dim_int_list_115, permute_default_67, reshape_default_68, permute_default_68, reshape_default_69, sum_dim_int_list_118, sum_dim_int_list_119, permute_default_69, reshape_default_70, permute_default_70, reshape_default_71, permute_default_71, reshape_default_73, permute_default_72, reshape_default_75, sum_dim_int_list_126, sum_dim_int_list_127, index_put_default, index_put_default_1, add_tensor_3)


def _default_make_inputs():
    from repro_harness import parse_shapes_config
    return parse_shapes_config(_shapes_config)


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
