"""
Standalone repro captured via capture_hook.
Label: hf_pythia_410m_train
Pattern hash: 628c60474fa2
Shape hash: 29fa3742
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, view_249: "f16[2048, 50304]", convert_element_type_446: "f32[4, 512, 1024]", mul_267: "f32[4, 512, 1024]", view_251: "f16[2048, 1024]", _shape_param_0, view_254: "f16[2048, 4096]", _shape_param_1, convert_element_type_464: "f32[4, 512, 1024]", mul_256: "f32[4, 512, 1024]", _shape_param_2, view_262: "f16[2048, 3072]", _shape_param_3, convert_element_type_479: "f32[4, 512, 1024]", view_265: "f16[2048, 1024]", _shape_param_4, view_268: "f16[2048, 4096]", _shape_param_5, convert_element_type_497: "f32[4, 512, 1024]", mul_245: "f32[4, 512, 1024]", _shape_param_6, view_276: "f16[2048, 3072]", _shape_param_7, convert_element_type_512: "f32[4, 512, 1024]", view_279: "f16[2048, 1024]", _shape_param_8, view_282: "f16[2048, 4096]", _shape_param_9, convert_element_type_530: "f32[4, 512, 1024]", mul_234: "f32[4, 512, 1024]", _shape_param_10, view_290: "f16[2048, 3072]", _shape_param_11, convert_element_type_545: "f32[4, 512, 1024]", view_293: "f16[2048, 1024]", _shape_param_12, view_296: "f16[2048, 4096]", _shape_param_13, convert_element_type_563: "f32[4, 512, 1024]", mul_223: "f32[4, 512, 1024]", _shape_param_14, view_304: "f16[2048, 3072]", _shape_param_15, convert_element_type_578: "f32[4, 512, 1024]", view_307: "f16[2048, 1024]", _shape_param_16, view_310: "f16[2048, 4096]", _shape_param_17, convert_element_type_596: "f32[4, 512, 1024]", mul_212: "f32[4, 512, 1024]", _shape_param_18, view_318: "f16[2048, 3072]", _shape_param_19, convert_element_type_611: "f32[4, 512, 1024]", view_321: "f16[2048, 1024]", _shape_param_20, view_324: "f16[2048, 4096]", _shape_param_21, convert_element_type_629: "f32[4, 512, 1024]", mul_201: "f32[4, 512, 1024]", _shape_param_22, view_332: "f16[2048, 3072]", _shape_param_23, convert_element_type_644: "f32[4, 512, 1024]", view_335: "f16[2048, 1024]", _shape_param_24, view_338: "f16[2048, 4096]", _shape_param_25, convert_element_type_662: "f32[4, 512, 1024]", mul_190: "f32[4, 512, 1024]", _shape_param_26, view_346: "f16[2048, 3072]", _shape_param_27, convert_element_type_677: "f32[4, 512, 1024]", view_349: "f16[2048, 1024]", _shape_param_28, view_352: "f16[2048, 4096]", _shape_param_29, convert_element_type_695: "f32[4, 512, 1024]", mul_179: "f32[4, 512, 1024]", _shape_param_30, view_360: "f16[2048, 3072]", _shape_param_31, convert_element_type_710: "f32[4, 512, 1024]", view_363: "f16[2048, 1024]", _shape_param_32, view_366: "f16[2048, 4096]", _shape_param_33, convert_element_type_728: "f32[4, 512, 1024]", mul_168: "f32[4, 512, 1024]", _shape_param_34, view_374: "f16[2048, 3072]", _shape_param_35, convert_element_type_743: "f32[4, 512, 1024]", view_377: "f16[2048, 1024]", _shape_param_36, view_380: "f16[2048, 4096]", _shape_param_37, convert_element_type_761: "f32[4, 512, 1024]", mul_157: "f32[4, 512, 1024]", _shape_param_38, view_388: "f16[2048, 3072]", _shape_param_39, convert_element_type_776: "f32[4, 512, 1024]", view_391: "f16[2048, 1024]", _shape_param_40, view_394: "f16[2048, 4096]", _shape_param_41, convert_element_type_794: "f32[4, 512, 1024]", mul_146: "f32[4, 512, 1024]", _shape_param_42, view_402: "f16[2048, 3072]", _shape_param_43, convert_element_type_809: "f32[4, 512, 1024]", view_405: "f16[2048, 1024]", _shape_param_44, view_408: "f16[2048, 4096]", _shape_param_45, convert_element_type_827: "f32[4, 512, 1024]", mul_135: "f32[4, 512, 1024]", _shape_param_46, view_416: "f16[2048, 3072]", _shape_param_47, convert_element_type_842: "f32[4, 512, 1024]", view_419: "f16[2048, 1024]", _shape_param_48, view_422: "f16[2048, 4096]", _shape_param_49, convert_element_type_860: "f32[4, 512, 1024]", mul_124: "f32[4, 512, 1024]", _shape_param_50, view_430: "f16[2048, 3072]", _shape_param_51, convert_element_type_875: "f32[4, 512, 1024]", view_433: "f16[2048, 1024]", _shape_param_52, view_436: "f16[2048, 4096]", _shape_param_53, convert_element_type_893: "f32[4, 512, 1024]", mul_113: "f32[4, 512, 1024]", _shape_param_54, view_444: "f16[2048, 3072]", _shape_param_55, convert_element_type_908: "f32[4, 512, 1024]", view_447: "f16[2048, 1024]", _shape_param_56, view_450: "f16[2048, 4096]", _shape_param_57, convert_element_type_926: "f32[4, 512, 1024]", mul_102: "f32[4, 512, 1024]", _shape_param_58, view_458: "f16[2048, 3072]", _shape_param_59, convert_element_type_941: "f32[4, 512, 1024]", view_461: "f16[2048, 1024]", _shape_param_60, view_464: "f16[2048, 4096]", _shape_param_61, convert_element_type_959: "f32[4, 512, 1024]", mul_91: "f32[4, 512, 1024]", _shape_param_62, view_472: "f16[2048, 3072]", _shape_param_63, convert_element_type_974: "f32[4, 512, 1024]", view_475: "f16[2048, 1024]", _shape_param_64, view_478: "f16[2048, 4096]", _shape_param_65, convert_element_type_992: "f32[4, 512, 1024]", mul_80: "f32[4, 512, 1024]", _shape_param_66, view_486: "f16[2048, 3072]", _shape_param_67, convert_element_type_1007: "f32[4, 512, 1024]", view_489: "f16[2048, 1024]", _shape_param_68, view_492: "f16[2048, 4096]", _shape_param_69, convert_element_type_1025: "f32[4, 512, 1024]", mul_69: "f32[4, 512, 1024]", _shape_param_70, view_500: "f16[2048, 3072]", _shape_param_71, convert_element_type_1040: "f32[4, 512, 1024]", view_503: "f16[2048, 1024]", _shape_param_72, view_506: "f16[2048, 4096]", _shape_param_73, convert_element_type_1058: "f32[4, 512, 1024]", mul_58: "f32[4, 512, 1024]", _shape_param_74, view_514: "f16[2048, 3072]", _shape_param_75, convert_element_type_1073: "f32[4, 512, 1024]", view_517: "f16[2048, 1024]", _shape_param_76, view_520: "f16[2048, 4096]", _shape_param_77, convert_element_type_1091: "f32[4, 512, 1024]", mul_47: "f32[4, 512, 1024]", _shape_param_78, view_528: "f16[2048, 3072]", _shape_param_79, convert_element_type_1106: "f32[4, 512, 1024]", view_531: "f16[2048, 1024]", _shape_param_80, view_534: "f16[2048, 4096]", _shape_param_81, convert_element_type_1124: "f32[4, 512, 1024]", mul_36: "f32[4, 512, 1024]", _shape_param_82, view_542: "f16[2048, 3072]", _shape_param_83, convert_element_type_1139: "f32[4, 512, 1024]", view_545: "f16[2048, 1024]", _shape_param_84, view_548: "f16[2048, 4096]", _shape_param_85, convert_element_type_1157: "f32[4, 512, 1024]", mul_25: "f32[4, 512, 1024]", _shape_param_86, view_556: "f16[2048, 3072]", _shape_param_87, convert_element_type_1172: "f32[4, 512, 1024]", view_559: "f16[2048, 1024]", _shape_param_88, view_562: "f16[2048, 4096]", _shape_param_89, convert_element_type_1190: "f32[4, 512, 1024]", mul_14: "f32[4, 512, 1024]", _shape_param_90, view_570: "f16[2048, 3072]", _shape_param_91, convert_element_type_1205: "f32[4, 512, 1024]", view_573: "f16[2048, 1024]", _shape_param_92, view_576: "f16[2048, 4096]", _shape_param_93, mm_189: "f16[2048, 1024]", _shape_param_94, primals_10: "f16[1024]", embedding: "f16[4, 512, 1024]", getitem_1: "f32[4, 512, 1]", rsqrt: "f32[4, 512, 1]", add_497: "f16[4, 512, 1024]", _shape_param_95, view_584: "f16[2048, 3072]", _shape_param_96, mm_193: "f16[2048, 1024]", _shape_param_97, primals_4: "f16[1024]", primals_1: "i64[4, 512]", full_default_49: "f32[]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:464 in forward, code: logits = self.embed_out(hidden_states[:, slice_indices, :])
        permute_default: "f16[50304, 2048]" = torch.ops.aten.permute.default(view_249, [1, 0]);  view_249 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:376 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        mul_tensor: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_446, mul_267);  mul_267 = None
        sum_dim_int_list: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [0, 1]);  mul_tensor = None
        sum_dim_int_list_1: "f32[1024]" = torch.ops.aten.sum.dim_IntList(convert_element_type_446, [0, 1]);  convert_element_type_446 = None
        convert_element_type_default: "f16[1024]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list, torch.float16);  sum_dim_int_list = None
        convert_element_type_default_1: "f16[1024]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_1, torch.float16);  sum_dim_int_list_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:48 in forward, code: hidden_states = self.dense_4h_to_h(hidden_states)
        permute_default_1: "f16[1024, 2048]" = torch.ops.aten.permute.default(view_251, [1, 0])
        sum_dim_int_list_2: "f16[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_251, [0], True)
        reshape_default: "f16[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_2, _shape_param_0);  sum_dim_int_list_2 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:46 in forward, code: hidden_states = self.dense_h_to_4h(hidden_states)
        permute_default_2: "f16[4096, 2048]" = torch.ops.aten.permute.default(view_254, [1, 0])
        sum_dim_int_list_3: "f16[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_254, [0], True);  view_254 = None
        reshape_default_1: "f16[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_3, _shape_param_1);  sum_dim_int_list_3 = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:280 in forward, code: mlp_output = self.mlp(self.post_attention_layernorm(hidden_states))
        mul_tensor_1: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_464, mul_256)
        sum_dim_int_list_4: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_1, [0, 1]);  mul_tensor_1 = None
        sum_dim_int_list_5: "f32[1024]" = torch.ops.aten.sum.dim_IntList(convert_element_type_464, [0, 1]);  convert_element_type_464 = None
        convert_element_type_default_2: "f16[1024]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_4, torch.float16);  sum_dim_int_list_4 = None
        convert_element_type_default_3: "f16[1024]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_5, torch.float16);  sum_dim_int_list_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:242 in forward, code: attn_output = self.dense(attn_output)
        sum_dim_int_list_6: "f16[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_251, [0], True);  view_251 = None
        reshape_default_2: "f16[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_6, _shape_param_2);  sum_dim_int_list_6 = _shape_param_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:214 in forward, code: qkv = self.query_key_value(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_3: "f16[3072, 2048]" = torch.ops.aten.permute.default(view_262, [1, 0])
        sum_dim_int_list_7: "f16[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_262, [0], True);  view_262 = None
        reshape_default_3: "f16[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_7, _shape_param_3);  sum_dim_int_list_7 = _shape_param_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:269 in forward, code: self.input_layernorm(hidden_states),
        mul_tensor_2: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_479, mul_256);  mul_256 = None
        sum_dim_int_list_8: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [0, 1]);  mul_tensor_2 = None
        sum_dim_int_list_9: "f32[1024]" = torch.ops.aten.sum.dim_IntList(convert_element_type_479, [0, 1]);  convert_element_type_479 = None
        convert_element_type_default_4: "f16[1024]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_8, torch.float16);  sum_dim_int_list_8 = None
        convert_element_type_default_5: "f16[1024]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_9, torch.float16);  sum_dim_int_list_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:48 in forward, code: hidden_states = self.dense_4h_to_h(hidden_states)
        permute_default_4: "f16[1024, 2048]" = torch.ops.aten.permute.default(view_265, [1, 0])
        sum_dim_int_list_10: "f16[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_265, [0], True)
        reshape_default_4: "f16[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_10, _shape_param_4);  sum_dim_int_list_10 = _shape_param_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:46 in forward, code: hidden_states = self.dense_h_to_4h(hidden_states)
        permute_default_5: "f16[4096, 2048]" = torch.ops.aten.permute.default(view_268, [1, 0])
        sum_dim_int_list_11: "f16[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_268, [0], True);  view_268 = None
        reshape_default_5: "f16[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_11, _shape_param_5);  sum_dim_int_list_11 = _shape_param_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:280 in forward, code: mlp_output = self.mlp(self.post_attention_layernorm(hidden_states))
        mul_tensor_3: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_497, mul_245)
        sum_dim_int_list_12: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_3, [0, 1]);  mul_tensor_3 = None
        sum_dim_int_list_13: "f32[1024]" = torch.ops.aten.sum.dim_IntList(convert_element_type_497, [0, 1]);  convert_element_type_497 = None
        convert_element_type_default_6: "f16[1024]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_12, torch.float16);  sum_dim_int_list_12 = None
        convert_element_type_default_7: "f16[1024]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_13, torch.float16);  sum_dim_int_list_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:242 in forward, code: attn_output = self.dense(attn_output)
        sum_dim_int_list_14: "f16[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_265, [0], True);  view_265 = None
        reshape_default_6: "f16[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_14, _shape_param_6);  sum_dim_int_list_14 = _shape_param_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:214 in forward, code: qkv = self.query_key_value(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_6: "f16[3072, 2048]" = torch.ops.aten.permute.default(view_276, [1, 0])
        sum_dim_int_list_15: "f16[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_276, [0], True);  view_276 = None
        reshape_default_7: "f16[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_15, _shape_param_7);  sum_dim_int_list_15 = _shape_param_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:269 in forward, code: self.input_layernorm(hidden_states),
        mul_tensor_4: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_512, mul_245);  mul_245 = None
        sum_dim_int_list_16: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_4, [0, 1]);  mul_tensor_4 = None
        sum_dim_int_list_17: "f32[1024]" = torch.ops.aten.sum.dim_IntList(convert_element_type_512, [0, 1]);  convert_element_type_512 = None
        convert_element_type_default_8: "f16[1024]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_16, torch.float16);  sum_dim_int_list_16 = None
        convert_element_type_default_9: "f16[1024]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_17, torch.float16);  sum_dim_int_list_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:48 in forward, code: hidden_states = self.dense_4h_to_h(hidden_states)
        permute_default_7: "f16[1024, 2048]" = torch.ops.aten.permute.default(view_279, [1, 0])
        sum_dim_int_list_18: "f16[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_279, [0], True)
        reshape_default_8: "f16[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_18, _shape_param_8);  sum_dim_int_list_18 = _shape_param_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:46 in forward, code: hidden_states = self.dense_h_to_4h(hidden_states)
        permute_default_8: "f16[4096, 2048]" = torch.ops.aten.permute.default(view_282, [1, 0])
        sum_dim_int_list_19: "f16[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_282, [0], True);  view_282 = None
        reshape_default_9: "f16[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_19, _shape_param_9);  sum_dim_int_list_19 = _shape_param_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:280 in forward, code: mlp_output = self.mlp(self.post_attention_layernorm(hidden_states))
        mul_tensor_5: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_530, mul_234)
        sum_dim_int_list_20: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_5, [0, 1]);  mul_tensor_5 = None
        sum_dim_int_list_21: "f32[1024]" = torch.ops.aten.sum.dim_IntList(convert_element_type_530, [0, 1]);  convert_element_type_530 = None
        convert_element_type_default_10: "f16[1024]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_20, torch.float16);  sum_dim_int_list_20 = None
        convert_element_type_default_11: "f16[1024]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_21, torch.float16);  sum_dim_int_list_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:242 in forward, code: attn_output = self.dense(attn_output)
        sum_dim_int_list_22: "f16[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_279, [0], True);  view_279 = None
        reshape_default_10: "f16[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_22, _shape_param_10);  sum_dim_int_list_22 = _shape_param_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:214 in forward, code: qkv = self.query_key_value(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_9: "f16[3072, 2048]" = torch.ops.aten.permute.default(view_290, [1, 0])
        sum_dim_int_list_23: "f16[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_290, [0], True);  view_290 = None
        reshape_default_11: "f16[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_23, _shape_param_11);  sum_dim_int_list_23 = _shape_param_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:269 in forward, code: self.input_layernorm(hidden_states),
        mul_tensor_6: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_545, mul_234);  mul_234 = None
        sum_dim_int_list_24: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_6, [0, 1]);  mul_tensor_6 = None
        sum_dim_int_list_25: "f32[1024]" = torch.ops.aten.sum.dim_IntList(convert_element_type_545, [0, 1]);  convert_element_type_545 = None
        convert_element_type_default_12: "f16[1024]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_24, torch.float16);  sum_dim_int_list_24 = None
        convert_element_type_default_13: "f16[1024]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_25, torch.float16);  sum_dim_int_list_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:48 in forward, code: hidden_states = self.dense_4h_to_h(hidden_states)
        permute_default_10: "f16[1024, 2048]" = torch.ops.aten.permute.default(view_293, [1, 0])
        sum_dim_int_list_26: "f16[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_293, [0], True)
        reshape_default_12: "f16[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_26, _shape_param_12);  sum_dim_int_list_26 = _shape_param_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:46 in forward, code: hidden_states = self.dense_h_to_4h(hidden_states)
        permute_default_11: "f16[4096, 2048]" = torch.ops.aten.permute.default(view_296, [1, 0])
        sum_dim_int_list_27: "f16[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_296, [0], True);  view_296 = None
        reshape_default_13: "f16[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_27, _shape_param_13);  sum_dim_int_list_27 = _shape_param_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:280 in forward, code: mlp_output = self.mlp(self.post_attention_layernorm(hidden_states))
        mul_tensor_7: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_563, mul_223)
        sum_dim_int_list_28: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_7, [0, 1]);  mul_tensor_7 = None
        sum_dim_int_list_29: "f32[1024]" = torch.ops.aten.sum.dim_IntList(convert_element_type_563, [0, 1]);  convert_element_type_563 = None
        convert_element_type_default_14: "f16[1024]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_28, torch.float16);  sum_dim_int_list_28 = None
        convert_element_type_default_15: "f16[1024]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_29, torch.float16);  sum_dim_int_list_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:242 in forward, code: attn_output = self.dense(attn_output)
        sum_dim_int_list_30: "f16[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_293, [0], True);  view_293 = None
        reshape_default_14: "f16[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_30, _shape_param_14);  sum_dim_int_list_30 = _shape_param_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:214 in forward, code: qkv = self.query_key_value(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_12: "f16[3072, 2048]" = torch.ops.aten.permute.default(view_304, [1, 0])
        sum_dim_int_list_31: "f16[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_304, [0], True);  view_304 = None
        reshape_default_15: "f16[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_31, _shape_param_15);  sum_dim_int_list_31 = _shape_param_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:269 in forward, code: self.input_layernorm(hidden_states),
        mul_tensor_8: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_578, mul_223);  mul_223 = None
        sum_dim_int_list_32: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_8, [0, 1]);  mul_tensor_8 = None
        sum_dim_int_list_33: "f32[1024]" = torch.ops.aten.sum.dim_IntList(convert_element_type_578, [0, 1]);  convert_element_type_578 = None
        convert_element_type_default_16: "f16[1024]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_32, torch.float16);  sum_dim_int_list_32 = None
        convert_element_type_default_17: "f16[1024]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_33, torch.float16);  sum_dim_int_list_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:48 in forward, code: hidden_states = self.dense_4h_to_h(hidden_states)
        permute_default_13: "f16[1024, 2048]" = torch.ops.aten.permute.default(view_307, [1, 0])
        sum_dim_int_list_34: "f16[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_307, [0], True)
        reshape_default_16: "f16[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_34, _shape_param_16);  sum_dim_int_list_34 = _shape_param_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:46 in forward, code: hidden_states = self.dense_h_to_4h(hidden_states)
        permute_default_14: "f16[4096, 2048]" = torch.ops.aten.permute.default(view_310, [1, 0])
        sum_dim_int_list_35: "f16[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_310, [0], True);  view_310 = None
        reshape_default_17: "f16[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_35, _shape_param_17);  sum_dim_int_list_35 = _shape_param_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:280 in forward, code: mlp_output = self.mlp(self.post_attention_layernorm(hidden_states))
        mul_tensor_9: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_596, mul_212)
        sum_dim_int_list_36: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_9, [0, 1]);  mul_tensor_9 = None
        sum_dim_int_list_37: "f32[1024]" = torch.ops.aten.sum.dim_IntList(convert_element_type_596, [0, 1]);  convert_element_type_596 = None
        convert_element_type_default_18: "f16[1024]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_36, torch.float16);  sum_dim_int_list_36 = None
        convert_element_type_default_19: "f16[1024]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_37, torch.float16);  sum_dim_int_list_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:242 in forward, code: attn_output = self.dense(attn_output)
        sum_dim_int_list_38: "f16[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_307, [0], True);  view_307 = None
        reshape_default_18: "f16[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_38, _shape_param_18);  sum_dim_int_list_38 = _shape_param_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:214 in forward, code: qkv = self.query_key_value(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_15: "f16[3072, 2048]" = torch.ops.aten.permute.default(view_318, [1, 0])
        sum_dim_int_list_39: "f16[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_318, [0], True);  view_318 = None
        reshape_default_19: "f16[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_39, _shape_param_19);  sum_dim_int_list_39 = _shape_param_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:269 in forward, code: self.input_layernorm(hidden_states),
        mul_tensor_10: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_611, mul_212);  mul_212 = None
        sum_dim_int_list_40: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_10, [0, 1]);  mul_tensor_10 = None
        sum_dim_int_list_41: "f32[1024]" = torch.ops.aten.sum.dim_IntList(convert_element_type_611, [0, 1]);  convert_element_type_611 = None
        convert_element_type_default_20: "f16[1024]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_40, torch.float16);  sum_dim_int_list_40 = None
        convert_element_type_default_21: "f16[1024]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_41, torch.float16);  sum_dim_int_list_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:48 in forward, code: hidden_states = self.dense_4h_to_h(hidden_states)
        permute_default_16: "f16[1024, 2048]" = torch.ops.aten.permute.default(view_321, [1, 0])
        sum_dim_int_list_42: "f16[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_321, [0], True)
        reshape_default_20: "f16[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_42, _shape_param_20);  sum_dim_int_list_42 = _shape_param_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:46 in forward, code: hidden_states = self.dense_h_to_4h(hidden_states)
        permute_default_17: "f16[4096, 2048]" = torch.ops.aten.permute.default(view_324, [1, 0])
        sum_dim_int_list_43: "f16[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_324, [0], True);  view_324 = None
        reshape_default_21: "f16[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_43, _shape_param_21);  sum_dim_int_list_43 = _shape_param_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:280 in forward, code: mlp_output = self.mlp(self.post_attention_layernorm(hidden_states))
        mul_tensor_11: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_629, mul_201)
        sum_dim_int_list_44: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_11, [0, 1]);  mul_tensor_11 = None
        sum_dim_int_list_45: "f32[1024]" = torch.ops.aten.sum.dim_IntList(convert_element_type_629, [0, 1]);  convert_element_type_629 = None
        convert_element_type_default_22: "f16[1024]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_44, torch.float16);  sum_dim_int_list_44 = None
        convert_element_type_default_23: "f16[1024]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_45, torch.float16);  sum_dim_int_list_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:242 in forward, code: attn_output = self.dense(attn_output)
        sum_dim_int_list_46: "f16[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_321, [0], True);  view_321 = None
        reshape_default_22: "f16[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_46, _shape_param_22);  sum_dim_int_list_46 = _shape_param_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:214 in forward, code: qkv = self.query_key_value(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_18: "f16[3072, 2048]" = torch.ops.aten.permute.default(view_332, [1, 0])
        sum_dim_int_list_47: "f16[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_332, [0], True);  view_332 = None
        reshape_default_23: "f16[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_47, _shape_param_23);  sum_dim_int_list_47 = _shape_param_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:269 in forward, code: self.input_layernorm(hidden_states),
        mul_tensor_12: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_644, mul_201);  mul_201 = None
        sum_dim_int_list_48: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_12, [0, 1]);  mul_tensor_12 = None
        sum_dim_int_list_49: "f32[1024]" = torch.ops.aten.sum.dim_IntList(convert_element_type_644, [0, 1]);  convert_element_type_644 = None
        convert_element_type_default_24: "f16[1024]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_48, torch.float16);  sum_dim_int_list_48 = None
        convert_element_type_default_25: "f16[1024]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_49, torch.float16);  sum_dim_int_list_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:48 in forward, code: hidden_states = self.dense_4h_to_h(hidden_states)
        permute_default_19: "f16[1024, 2048]" = torch.ops.aten.permute.default(view_335, [1, 0])
        sum_dim_int_list_50: "f16[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_335, [0], True)
        reshape_default_24: "f16[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_50, _shape_param_24);  sum_dim_int_list_50 = _shape_param_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:46 in forward, code: hidden_states = self.dense_h_to_4h(hidden_states)
        permute_default_20: "f16[4096, 2048]" = torch.ops.aten.permute.default(view_338, [1, 0])
        sum_dim_int_list_51: "f16[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_338, [0], True);  view_338 = None
        reshape_default_25: "f16[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_51, _shape_param_25);  sum_dim_int_list_51 = _shape_param_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:280 in forward, code: mlp_output = self.mlp(self.post_attention_layernorm(hidden_states))
        mul_tensor_13: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_662, mul_190)
        sum_dim_int_list_52: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_13, [0, 1]);  mul_tensor_13 = None
        sum_dim_int_list_53: "f32[1024]" = torch.ops.aten.sum.dim_IntList(convert_element_type_662, [0, 1]);  convert_element_type_662 = None
        convert_element_type_default_26: "f16[1024]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_52, torch.float16);  sum_dim_int_list_52 = None
        convert_element_type_default_27: "f16[1024]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_53, torch.float16);  sum_dim_int_list_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:242 in forward, code: attn_output = self.dense(attn_output)
        sum_dim_int_list_54: "f16[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_335, [0], True);  view_335 = None
        reshape_default_26: "f16[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_54, _shape_param_26);  sum_dim_int_list_54 = _shape_param_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:214 in forward, code: qkv = self.query_key_value(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_21: "f16[3072, 2048]" = torch.ops.aten.permute.default(view_346, [1, 0])
        sum_dim_int_list_55: "f16[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_346, [0], True);  view_346 = None
        reshape_default_27: "f16[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_55, _shape_param_27);  sum_dim_int_list_55 = _shape_param_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:269 in forward, code: self.input_layernorm(hidden_states),
        mul_tensor_14: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_677, mul_190);  mul_190 = None
        sum_dim_int_list_56: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_14, [0, 1]);  mul_tensor_14 = None
        sum_dim_int_list_57: "f32[1024]" = torch.ops.aten.sum.dim_IntList(convert_element_type_677, [0, 1]);  convert_element_type_677 = None
        convert_element_type_default_28: "f16[1024]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_56, torch.float16);  sum_dim_int_list_56 = None
        convert_element_type_default_29: "f16[1024]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_57, torch.float16);  sum_dim_int_list_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:48 in forward, code: hidden_states = self.dense_4h_to_h(hidden_states)
        permute_default_22: "f16[1024, 2048]" = torch.ops.aten.permute.default(view_349, [1, 0])
        sum_dim_int_list_58: "f16[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_349, [0], True)
        reshape_default_28: "f16[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_58, _shape_param_28);  sum_dim_int_list_58 = _shape_param_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:46 in forward, code: hidden_states = self.dense_h_to_4h(hidden_states)
        permute_default_23: "f16[4096, 2048]" = torch.ops.aten.permute.default(view_352, [1, 0])
        sum_dim_int_list_59: "f16[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_352, [0], True);  view_352 = None
        reshape_default_29: "f16[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_59, _shape_param_29);  sum_dim_int_list_59 = _shape_param_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:280 in forward, code: mlp_output = self.mlp(self.post_attention_layernorm(hidden_states))
        mul_tensor_15: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_695, mul_179)
        sum_dim_int_list_60: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_15, [0, 1]);  mul_tensor_15 = None
        sum_dim_int_list_61: "f32[1024]" = torch.ops.aten.sum.dim_IntList(convert_element_type_695, [0, 1]);  convert_element_type_695 = None
        convert_element_type_default_30: "f16[1024]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_60, torch.float16);  sum_dim_int_list_60 = None
        convert_element_type_default_31: "f16[1024]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_61, torch.float16);  sum_dim_int_list_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:242 in forward, code: attn_output = self.dense(attn_output)
        sum_dim_int_list_62: "f16[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_349, [0], True);  view_349 = None
        reshape_default_30: "f16[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_62, _shape_param_30);  sum_dim_int_list_62 = _shape_param_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:214 in forward, code: qkv = self.query_key_value(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_24: "f16[3072, 2048]" = torch.ops.aten.permute.default(view_360, [1, 0])
        sum_dim_int_list_63: "f16[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_360, [0], True);  view_360 = None
        reshape_default_31: "f16[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_63, _shape_param_31);  sum_dim_int_list_63 = _shape_param_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:269 in forward, code: self.input_layernorm(hidden_states),
        mul_tensor_16: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_710, mul_179);  mul_179 = None
        sum_dim_int_list_64: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_16, [0, 1]);  mul_tensor_16 = None
        sum_dim_int_list_65: "f32[1024]" = torch.ops.aten.sum.dim_IntList(convert_element_type_710, [0, 1]);  convert_element_type_710 = None
        convert_element_type_default_32: "f16[1024]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_64, torch.float16);  sum_dim_int_list_64 = None
        convert_element_type_default_33: "f16[1024]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_65, torch.float16);  sum_dim_int_list_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:48 in forward, code: hidden_states = self.dense_4h_to_h(hidden_states)
        permute_default_25: "f16[1024, 2048]" = torch.ops.aten.permute.default(view_363, [1, 0])
        sum_dim_int_list_66: "f16[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_363, [0], True)
        reshape_default_32: "f16[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_66, _shape_param_32);  sum_dim_int_list_66 = _shape_param_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:46 in forward, code: hidden_states = self.dense_h_to_4h(hidden_states)
        permute_default_26: "f16[4096, 2048]" = torch.ops.aten.permute.default(view_366, [1, 0])
        sum_dim_int_list_67: "f16[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_366, [0], True);  view_366 = None
        reshape_default_33: "f16[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_67, _shape_param_33);  sum_dim_int_list_67 = _shape_param_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:280 in forward, code: mlp_output = self.mlp(self.post_attention_layernorm(hidden_states))
        mul_tensor_17: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_728, mul_168)
        sum_dim_int_list_68: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_17, [0, 1]);  mul_tensor_17 = None
        sum_dim_int_list_69: "f32[1024]" = torch.ops.aten.sum.dim_IntList(convert_element_type_728, [0, 1]);  convert_element_type_728 = None
        convert_element_type_default_34: "f16[1024]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_68, torch.float16);  sum_dim_int_list_68 = None
        convert_element_type_default_35: "f16[1024]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_69, torch.float16);  sum_dim_int_list_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:242 in forward, code: attn_output = self.dense(attn_output)
        sum_dim_int_list_70: "f16[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_363, [0], True);  view_363 = None
        reshape_default_34: "f16[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_70, _shape_param_34);  sum_dim_int_list_70 = _shape_param_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:214 in forward, code: qkv = self.query_key_value(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_27: "f16[3072, 2048]" = torch.ops.aten.permute.default(view_374, [1, 0])
        sum_dim_int_list_71: "f16[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_374, [0], True);  view_374 = None
        reshape_default_35: "f16[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_71, _shape_param_35);  sum_dim_int_list_71 = _shape_param_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:269 in forward, code: self.input_layernorm(hidden_states),
        mul_tensor_18: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_743, mul_168);  mul_168 = None
        sum_dim_int_list_72: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_18, [0, 1]);  mul_tensor_18 = None
        sum_dim_int_list_73: "f32[1024]" = torch.ops.aten.sum.dim_IntList(convert_element_type_743, [0, 1]);  convert_element_type_743 = None
        convert_element_type_default_36: "f16[1024]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_72, torch.float16);  sum_dim_int_list_72 = None
        convert_element_type_default_37: "f16[1024]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_73, torch.float16);  sum_dim_int_list_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:48 in forward, code: hidden_states = self.dense_4h_to_h(hidden_states)
        permute_default_28: "f16[1024, 2048]" = torch.ops.aten.permute.default(view_377, [1, 0])
        sum_dim_int_list_74: "f16[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_377, [0], True)
        reshape_default_36: "f16[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_74, _shape_param_36);  sum_dim_int_list_74 = _shape_param_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:46 in forward, code: hidden_states = self.dense_h_to_4h(hidden_states)
        permute_default_29: "f16[4096, 2048]" = torch.ops.aten.permute.default(view_380, [1, 0])
        sum_dim_int_list_75: "f16[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_380, [0], True);  view_380 = None
        reshape_default_37: "f16[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_75, _shape_param_37);  sum_dim_int_list_75 = _shape_param_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:280 in forward, code: mlp_output = self.mlp(self.post_attention_layernorm(hidden_states))
        mul_tensor_19: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_761, mul_157)
        sum_dim_int_list_76: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_19, [0, 1]);  mul_tensor_19 = None
        sum_dim_int_list_77: "f32[1024]" = torch.ops.aten.sum.dim_IntList(convert_element_type_761, [0, 1]);  convert_element_type_761 = None
        convert_element_type_default_38: "f16[1024]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_76, torch.float16);  sum_dim_int_list_76 = None
        convert_element_type_default_39: "f16[1024]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_77, torch.float16);  sum_dim_int_list_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:242 in forward, code: attn_output = self.dense(attn_output)
        sum_dim_int_list_78: "f16[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_377, [0], True);  view_377 = None
        reshape_default_38: "f16[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_78, _shape_param_38);  sum_dim_int_list_78 = _shape_param_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:214 in forward, code: qkv = self.query_key_value(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_30: "f16[3072, 2048]" = torch.ops.aten.permute.default(view_388, [1, 0])
        sum_dim_int_list_79: "f16[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_388, [0], True);  view_388 = None
        reshape_default_39: "f16[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_79, _shape_param_39);  sum_dim_int_list_79 = _shape_param_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:269 in forward, code: self.input_layernorm(hidden_states),
        mul_tensor_20: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_776, mul_157);  mul_157 = None
        sum_dim_int_list_80: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_20, [0, 1]);  mul_tensor_20 = None
        sum_dim_int_list_81: "f32[1024]" = torch.ops.aten.sum.dim_IntList(convert_element_type_776, [0, 1]);  convert_element_type_776 = None
        convert_element_type_default_40: "f16[1024]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_80, torch.float16);  sum_dim_int_list_80 = None
        convert_element_type_default_41: "f16[1024]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_81, torch.float16);  sum_dim_int_list_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:48 in forward, code: hidden_states = self.dense_4h_to_h(hidden_states)
        permute_default_31: "f16[1024, 2048]" = torch.ops.aten.permute.default(view_391, [1, 0])
        sum_dim_int_list_82: "f16[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_391, [0], True)
        reshape_default_40: "f16[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_82, _shape_param_40);  sum_dim_int_list_82 = _shape_param_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:46 in forward, code: hidden_states = self.dense_h_to_4h(hidden_states)
        permute_default_32: "f16[4096, 2048]" = torch.ops.aten.permute.default(view_394, [1, 0])
        sum_dim_int_list_83: "f16[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_394, [0], True);  view_394 = None
        reshape_default_41: "f16[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_83, _shape_param_41);  sum_dim_int_list_83 = _shape_param_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:280 in forward, code: mlp_output = self.mlp(self.post_attention_layernorm(hidden_states))
        mul_tensor_21: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_794, mul_146)
        sum_dim_int_list_84: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_21, [0, 1]);  mul_tensor_21 = None
        sum_dim_int_list_85: "f32[1024]" = torch.ops.aten.sum.dim_IntList(convert_element_type_794, [0, 1]);  convert_element_type_794 = None
        convert_element_type_default_42: "f16[1024]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_84, torch.float16);  sum_dim_int_list_84 = None
        convert_element_type_default_43: "f16[1024]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_85, torch.float16);  sum_dim_int_list_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:242 in forward, code: attn_output = self.dense(attn_output)
        sum_dim_int_list_86: "f16[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_391, [0], True);  view_391 = None
        reshape_default_42: "f16[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_86, _shape_param_42);  sum_dim_int_list_86 = _shape_param_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:214 in forward, code: qkv = self.query_key_value(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_33: "f16[3072, 2048]" = torch.ops.aten.permute.default(view_402, [1, 0])
        sum_dim_int_list_87: "f16[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_402, [0], True);  view_402 = None
        reshape_default_43: "f16[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_87, _shape_param_43);  sum_dim_int_list_87 = _shape_param_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:269 in forward, code: self.input_layernorm(hidden_states),
        mul_tensor_22: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_809, mul_146);  mul_146 = None
        sum_dim_int_list_88: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_22, [0, 1]);  mul_tensor_22 = None
        sum_dim_int_list_89: "f32[1024]" = torch.ops.aten.sum.dim_IntList(convert_element_type_809, [0, 1]);  convert_element_type_809 = None
        convert_element_type_default_44: "f16[1024]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_88, torch.float16);  sum_dim_int_list_88 = None
        convert_element_type_default_45: "f16[1024]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_89, torch.float16);  sum_dim_int_list_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:48 in forward, code: hidden_states = self.dense_4h_to_h(hidden_states)
        permute_default_34: "f16[1024, 2048]" = torch.ops.aten.permute.default(view_405, [1, 0])
        sum_dim_int_list_90: "f16[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_405, [0], True)
        reshape_default_44: "f16[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_90, _shape_param_44);  sum_dim_int_list_90 = _shape_param_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:46 in forward, code: hidden_states = self.dense_h_to_4h(hidden_states)
        permute_default_35: "f16[4096, 2048]" = torch.ops.aten.permute.default(view_408, [1, 0])
        sum_dim_int_list_91: "f16[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_408, [0], True);  view_408 = None
        reshape_default_45: "f16[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_91, _shape_param_45);  sum_dim_int_list_91 = _shape_param_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:280 in forward, code: mlp_output = self.mlp(self.post_attention_layernorm(hidden_states))
        mul_tensor_23: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_827, mul_135)
        sum_dim_int_list_92: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_23, [0, 1]);  mul_tensor_23 = None
        sum_dim_int_list_93: "f32[1024]" = torch.ops.aten.sum.dim_IntList(convert_element_type_827, [0, 1]);  convert_element_type_827 = None
        convert_element_type_default_46: "f16[1024]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_92, torch.float16);  sum_dim_int_list_92 = None
        convert_element_type_default_47: "f16[1024]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_93, torch.float16);  sum_dim_int_list_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:242 in forward, code: attn_output = self.dense(attn_output)
        sum_dim_int_list_94: "f16[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_405, [0], True);  view_405 = None
        reshape_default_46: "f16[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_94, _shape_param_46);  sum_dim_int_list_94 = _shape_param_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:214 in forward, code: qkv = self.query_key_value(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_36: "f16[3072, 2048]" = torch.ops.aten.permute.default(view_416, [1, 0])
        sum_dim_int_list_95: "f16[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_416, [0], True);  view_416 = None
        reshape_default_47: "f16[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_95, _shape_param_47);  sum_dim_int_list_95 = _shape_param_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:269 in forward, code: self.input_layernorm(hidden_states),
        mul_tensor_24: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_842, mul_135);  mul_135 = None
        sum_dim_int_list_96: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_24, [0, 1]);  mul_tensor_24 = None
        sum_dim_int_list_97: "f32[1024]" = torch.ops.aten.sum.dim_IntList(convert_element_type_842, [0, 1]);  convert_element_type_842 = None
        convert_element_type_default_48: "f16[1024]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_96, torch.float16);  sum_dim_int_list_96 = None
        convert_element_type_default_49: "f16[1024]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_97, torch.float16);  sum_dim_int_list_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:48 in forward, code: hidden_states = self.dense_4h_to_h(hidden_states)
        permute_default_37: "f16[1024, 2048]" = torch.ops.aten.permute.default(view_419, [1, 0])
        sum_dim_int_list_98: "f16[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_419, [0], True)
        reshape_default_48: "f16[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_98, _shape_param_48);  sum_dim_int_list_98 = _shape_param_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:46 in forward, code: hidden_states = self.dense_h_to_4h(hidden_states)
        permute_default_38: "f16[4096, 2048]" = torch.ops.aten.permute.default(view_422, [1, 0])
        sum_dim_int_list_99: "f16[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_422, [0], True);  view_422 = None
        reshape_default_49: "f16[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_99, _shape_param_49);  sum_dim_int_list_99 = _shape_param_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:280 in forward, code: mlp_output = self.mlp(self.post_attention_layernorm(hidden_states))
        mul_tensor_25: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_860, mul_124)
        sum_dim_int_list_100: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_25, [0, 1]);  mul_tensor_25 = None
        sum_dim_int_list_101: "f32[1024]" = torch.ops.aten.sum.dim_IntList(convert_element_type_860, [0, 1]);  convert_element_type_860 = None
        convert_element_type_default_50: "f16[1024]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_100, torch.float16);  sum_dim_int_list_100 = None
        convert_element_type_default_51: "f16[1024]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_101, torch.float16);  sum_dim_int_list_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:242 in forward, code: attn_output = self.dense(attn_output)
        sum_dim_int_list_102: "f16[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_419, [0], True);  view_419 = None
        reshape_default_50: "f16[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_102, _shape_param_50);  sum_dim_int_list_102 = _shape_param_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:214 in forward, code: qkv = self.query_key_value(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_39: "f16[3072, 2048]" = torch.ops.aten.permute.default(view_430, [1, 0])
        sum_dim_int_list_103: "f16[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_430, [0], True);  view_430 = None
        reshape_default_51: "f16[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_103, _shape_param_51);  sum_dim_int_list_103 = _shape_param_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:269 in forward, code: self.input_layernorm(hidden_states),
        mul_tensor_26: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_875, mul_124);  mul_124 = None
        sum_dim_int_list_104: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_26, [0, 1]);  mul_tensor_26 = None
        sum_dim_int_list_105: "f32[1024]" = torch.ops.aten.sum.dim_IntList(convert_element_type_875, [0, 1]);  convert_element_type_875 = None
        convert_element_type_default_52: "f16[1024]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_104, torch.float16);  sum_dim_int_list_104 = None
        convert_element_type_default_53: "f16[1024]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_105, torch.float16);  sum_dim_int_list_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:48 in forward, code: hidden_states = self.dense_4h_to_h(hidden_states)
        permute_default_40: "f16[1024, 2048]" = torch.ops.aten.permute.default(view_433, [1, 0])
        sum_dim_int_list_106: "f16[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_433, [0], True)
        reshape_default_52: "f16[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_106, _shape_param_52);  sum_dim_int_list_106 = _shape_param_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:46 in forward, code: hidden_states = self.dense_h_to_4h(hidden_states)
        permute_default_41: "f16[4096, 2048]" = torch.ops.aten.permute.default(view_436, [1, 0])
        sum_dim_int_list_107: "f16[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_436, [0], True);  view_436 = None
        reshape_default_53: "f16[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_107, _shape_param_53);  sum_dim_int_list_107 = _shape_param_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:280 in forward, code: mlp_output = self.mlp(self.post_attention_layernorm(hidden_states))
        mul_tensor_27: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_893, mul_113)
        sum_dim_int_list_108: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_27, [0, 1]);  mul_tensor_27 = None
        sum_dim_int_list_109: "f32[1024]" = torch.ops.aten.sum.dim_IntList(convert_element_type_893, [0, 1]);  convert_element_type_893 = None
        convert_element_type_default_54: "f16[1024]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_108, torch.float16);  sum_dim_int_list_108 = None
        convert_element_type_default_55: "f16[1024]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_109, torch.float16);  sum_dim_int_list_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:242 in forward, code: attn_output = self.dense(attn_output)
        sum_dim_int_list_110: "f16[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_433, [0], True);  view_433 = None
        reshape_default_54: "f16[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_110, _shape_param_54);  sum_dim_int_list_110 = _shape_param_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:214 in forward, code: qkv = self.query_key_value(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_42: "f16[3072, 2048]" = torch.ops.aten.permute.default(view_444, [1, 0])
        sum_dim_int_list_111: "f16[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_444, [0], True);  view_444 = None
        reshape_default_55: "f16[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_111, _shape_param_55);  sum_dim_int_list_111 = _shape_param_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:269 in forward, code: self.input_layernorm(hidden_states),
        mul_tensor_28: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_908, mul_113);  mul_113 = None
        sum_dim_int_list_112: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_28, [0, 1]);  mul_tensor_28 = None
        sum_dim_int_list_113: "f32[1024]" = torch.ops.aten.sum.dim_IntList(convert_element_type_908, [0, 1]);  convert_element_type_908 = None
        convert_element_type_default_56: "f16[1024]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_112, torch.float16);  sum_dim_int_list_112 = None
        convert_element_type_default_57: "f16[1024]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_113, torch.float16);  sum_dim_int_list_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:48 in forward, code: hidden_states = self.dense_4h_to_h(hidden_states)
        permute_default_43: "f16[1024, 2048]" = torch.ops.aten.permute.default(view_447, [1, 0])
        sum_dim_int_list_114: "f16[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_447, [0], True)
        reshape_default_56: "f16[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_114, _shape_param_56);  sum_dim_int_list_114 = _shape_param_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:46 in forward, code: hidden_states = self.dense_h_to_4h(hidden_states)
        permute_default_44: "f16[4096, 2048]" = torch.ops.aten.permute.default(view_450, [1, 0])
        sum_dim_int_list_115: "f16[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_450, [0], True);  view_450 = None
        reshape_default_57: "f16[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_115, _shape_param_57);  sum_dim_int_list_115 = _shape_param_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:280 in forward, code: mlp_output = self.mlp(self.post_attention_layernorm(hidden_states))
        mul_tensor_29: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_926, mul_102)
        sum_dim_int_list_116: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_29, [0, 1]);  mul_tensor_29 = None
        sum_dim_int_list_117: "f32[1024]" = torch.ops.aten.sum.dim_IntList(convert_element_type_926, [0, 1]);  convert_element_type_926 = None
        convert_element_type_default_58: "f16[1024]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_116, torch.float16);  sum_dim_int_list_116 = None
        convert_element_type_default_59: "f16[1024]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_117, torch.float16);  sum_dim_int_list_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:242 in forward, code: attn_output = self.dense(attn_output)
        sum_dim_int_list_118: "f16[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_447, [0], True);  view_447 = None
        reshape_default_58: "f16[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_118, _shape_param_58);  sum_dim_int_list_118 = _shape_param_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:214 in forward, code: qkv = self.query_key_value(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_45: "f16[3072, 2048]" = torch.ops.aten.permute.default(view_458, [1, 0])
        sum_dim_int_list_119: "f16[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_458, [0], True);  view_458 = None
        reshape_default_59: "f16[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_119, _shape_param_59);  sum_dim_int_list_119 = _shape_param_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:269 in forward, code: self.input_layernorm(hidden_states),
        mul_tensor_30: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_941, mul_102);  mul_102 = None
        sum_dim_int_list_120: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_30, [0, 1]);  mul_tensor_30 = None
        sum_dim_int_list_121: "f32[1024]" = torch.ops.aten.sum.dim_IntList(convert_element_type_941, [0, 1]);  convert_element_type_941 = None
        convert_element_type_default_60: "f16[1024]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_120, torch.float16);  sum_dim_int_list_120 = None
        convert_element_type_default_61: "f16[1024]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_121, torch.float16);  sum_dim_int_list_121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:48 in forward, code: hidden_states = self.dense_4h_to_h(hidden_states)
        permute_default_46: "f16[1024, 2048]" = torch.ops.aten.permute.default(view_461, [1, 0])
        sum_dim_int_list_122: "f16[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_461, [0], True)
        reshape_default_60: "f16[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_122, _shape_param_60);  sum_dim_int_list_122 = _shape_param_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:46 in forward, code: hidden_states = self.dense_h_to_4h(hidden_states)
        permute_default_47: "f16[4096, 2048]" = torch.ops.aten.permute.default(view_464, [1, 0])
        sum_dim_int_list_123: "f16[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_464, [0], True);  view_464 = None
        reshape_default_61: "f16[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_123, _shape_param_61);  sum_dim_int_list_123 = _shape_param_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:280 in forward, code: mlp_output = self.mlp(self.post_attention_layernorm(hidden_states))
        mul_tensor_31: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_959, mul_91)
        sum_dim_int_list_124: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_31, [0, 1]);  mul_tensor_31 = None
        sum_dim_int_list_125: "f32[1024]" = torch.ops.aten.sum.dim_IntList(convert_element_type_959, [0, 1]);  convert_element_type_959 = None
        convert_element_type_default_62: "f16[1024]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_124, torch.float16);  sum_dim_int_list_124 = None
        convert_element_type_default_63: "f16[1024]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_125, torch.float16);  sum_dim_int_list_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:242 in forward, code: attn_output = self.dense(attn_output)
        sum_dim_int_list_126: "f16[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_461, [0], True);  view_461 = None
        reshape_default_62: "f16[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_126, _shape_param_62);  sum_dim_int_list_126 = _shape_param_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:214 in forward, code: qkv = self.query_key_value(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_48: "f16[3072, 2048]" = torch.ops.aten.permute.default(view_472, [1, 0])
        sum_dim_int_list_127: "f16[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_472, [0], True);  view_472 = None
        reshape_default_63: "f16[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_127, _shape_param_63);  sum_dim_int_list_127 = _shape_param_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:269 in forward, code: self.input_layernorm(hidden_states),
        mul_tensor_32: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_974, mul_91);  mul_91 = None
        sum_dim_int_list_128: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_32, [0, 1]);  mul_tensor_32 = None
        sum_dim_int_list_129: "f32[1024]" = torch.ops.aten.sum.dim_IntList(convert_element_type_974, [0, 1]);  convert_element_type_974 = None
        convert_element_type_default_64: "f16[1024]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_128, torch.float16);  sum_dim_int_list_128 = None
        convert_element_type_default_65: "f16[1024]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_129, torch.float16);  sum_dim_int_list_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:48 in forward, code: hidden_states = self.dense_4h_to_h(hidden_states)
        permute_default_49: "f16[1024, 2048]" = torch.ops.aten.permute.default(view_475, [1, 0])
        sum_dim_int_list_130: "f16[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_475, [0], True)
        reshape_default_64: "f16[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_130, _shape_param_64);  sum_dim_int_list_130 = _shape_param_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:46 in forward, code: hidden_states = self.dense_h_to_4h(hidden_states)
        permute_default_50: "f16[4096, 2048]" = torch.ops.aten.permute.default(view_478, [1, 0])
        sum_dim_int_list_131: "f16[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_478, [0], True);  view_478 = None
        reshape_default_65: "f16[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_131, _shape_param_65);  sum_dim_int_list_131 = _shape_param_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:280 in forward, code: mlp_output = self.mlp(self.post_attention_layernorm(hidden_states))
        mul_tensor_33: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_992, mul_80)
        sum_dim_int_list_132: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_33, [0, 1]);  mul_tensor_33 = None
        sum_dim_int_list_133: "f32[1024]" = torch.ops.aten.sum.dim_IntList(convert_element_type_992, [0, 1]);  convert_element_type_992 = None
        convert_element_type_default_66: "f16[1024]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_132, torch.float16);  sum_dim_int_list_132 = None
        convert_element_type_default_67: "f16[1024]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_133, torch.float16);  sum_dim_int_list_133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:242 in forward, code: attn_output = self.dense(attn_output)
        sum_dim_int_list_134: "f16[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_475, [0], True);  view_475 = None
        reshape_default_66: "f16[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_134, _shape_param_66);  sum_dim_int_list_134 = _shape_param_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:214 in forward, code: qkv = self.query_key_value(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_51: "f16[3072, 2048]" = torch.ops.aten.permute.default(view_486, [1, 0])
        sum_dim_int_list_135: "f16[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_486, [0], True);  view_486 = None
        reshape_default_67: "f16[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_135, _shape_param_67);  sum_dim_int_list_135 = _shape_param_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:269 in forward, code: self.input_layernorm(hidden_states),
        mul_tensor_34: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_1007, mul_80);  mul_80 = None
        sum_dim_int_list_136: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_34, [0, 1]);  mul_tensor_34 = None
        sum_dim_int_list_137: "f32[1024]" = torch.ops.aten.sum.dim_IntList(convert_element_type_1007, [0, 1]);  convert_element_type_1007 = None
        convert_element_type_default_68: "f16[1024]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_136, torch.float16);  sum_dim_int_list_136 = None
        convert_element_type_default_69: "f16[1024]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_137, torch.float16);  sum_dim_int_list_137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:48 in forward, code: hidden_states = self.dense_4h_to_h(hidden_states)
        permute_default_52: "f16[1024, 2048]" = torch.ops.aten.permute.default(view_489, [1, 0])
        sum_dim_int_list_138: "f16[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_489, [0], True)
        reshape_default_68: "f16[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_138, _shape_param_68);  sum_dim_int_list_138 = _shape_param_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:46 in forward, code: hidden_states = self.dense_h_to_4h(hidden_states)
        permute_default_53: "f16[4096, 2048]" = torch.ops.aten.permute.default(view_492, [1, 0])
        sum_dim_int_list_139: "f16[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_492, [0], True);  view_492 = None
        reshape_default_69: "f16[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_139, _shape_param_69);  sum_dim_int_list_139 = _shape_param_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:280 in forward, code: mlp_output = self.mlp(self.post_attention_layernorm(hidden_states))
        mul_tensor_35: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_1025, mul_69)
        sum_dim_int_list_140: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_35, [0, 1]);  mul_tensor_35 = None
        sum_dim_int_list_141: "f32[1024]" = torch.ops.aten.sum.dim_IntList(convert_element_type_1025, [0, 1]);  convert_element_type_1025 = None
        convert_element_type_default_70: "f16[1024]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_140, torch.float16);  sum_dim_int_list_140 = None
        convert_element_type_default_71: "f16[1024]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_141, torch.float16);  sum_dim_int_list_141 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:242 in forward, code: attn_output = self.dense(attn_output)
        sum_dim_int_list_142: "f16[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_489, [0], True);  view_489 = None
        reshape_default_70: "f16[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_142, _shape_param_70);  sum_dim_int_list_142 = _shape_param_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:214 in forward, code: qkv = self.query_key_value(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_54: "f16[3072, 2048]" = torch.ops.aten.permute.default(view_500, [1, 0])
        sum_dim_int_list_143: "f16[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_500, [0], True);  view_500 = None
        reshape_default_71: "f16[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_143, _shape_param_71);  sum_dim_int_list_143 = _shape_param_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:269 in forward, code: self.input_layernorm(hidden_states),
        mul_tensor_36: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_1040, mul_69);  mul_69 = None
        sum_dim_int_list_144: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_36, [0, 1]);  mul_tensor_36 = None
        sum_dim_int_list_145: "f32[1024]" = torch.ops.aten.sum.dim_IntList(convert_element_type_1040, [0, 1]);  convert_element_type_1040 = None
        convert_element_type_default_72: "f16[1024]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_144, torch.float16);  sum_dim_int_list_144 = None
        convert_element_type_default_73: "f16[1024]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_145, torch.float16);  sum_dim_int_list_145 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:48 in forward, code: hidden_states = self.dense_4h_to_h(hidden_states)
        permute_default_55: "f16[1024, 2048]" = torch.ops.aten.permute.default(view_503, [1, 0])
        sum_dim_int_list_146: "f16[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_503, [0], True)
        reshape_default_72: "f16[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_146, _shape_param_72);  sum_dim_int_list_146 = _shape_param_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:46 in forward, code: hidden_states = self.dense_h_to_4h(hidden_states)
        permute_default_56: "f16[4096, 2048]" = torch.ops.aten.permute.default(view_506, [1, 0])
        sum_dim_int_list_147: "f16[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_506, [0], True);  view_506 = None
        reshape_default_73: "f16[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_147, _shape_param_73);  sum_dim_int_list_147 = _shape_param_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:280 in forward, code: mlp_output = self.mlp(self.post_attention_layernorm(hidden_states))
        mul_tensor_37: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_1058, mul_58)
        sum_dim_int_list_148: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_37, [0, 1]);  mul_tensor_37 = None
        sum_dim_int_list_149: "f32[1024]" = torch.ops.aten.sum.dim_IntList(convert_element_type_1058, [0, 1]);  convert_element_type_1058 = None
        convert_element_type_default_74: "f16[1024]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_148, torch.float16);  sum_dim_int_list_148 = None
        convert_element_type_default_75: "f16[1024]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_149, torch.float16);  sum_dim_int_list_149 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:242 in forward, code: attn_output = self.dense(attn_output)
        sum_dim_int_list_150: "f16[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_503, [0], True);  view_503 = None
        reshape_default_74: "f16[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_150, _shape_param_74);  sum_dim_int_list_150 = _shape_param_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:214 in forward, code: qkv = self.query_key_value(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_57: "f16[3072, 2048]" = torch.ops.aten.permute.default(view_514, [1, 0])
        sum_dim_int_list_151: "f16[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_514, [0], True);  view_514 = None
        reshape_default_75: "f16[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_151, _shape_param_75);  sum_dim_int_list_151 = _shape_param_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:269 in forward, code: self.input_layernorm(hidden_states),
        mul_tensor_38: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_1073, mul_58);  mul_58 = None
        sum_dim_int_list_152: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_38, [0, 1]);  mul_tensor_38 = None
        sum_dim_int_list_153: "f32[1024]" = torch.ops.aten.sum.dim_IntList(convert_element_type_1073, [0, 1]);  convert_element_type_1073 = None
        convert_element_type_default_76: "f16[1024]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_152, torch.float16);  sum_dim_int_list_152 = None
        convert_element_type_default_77: "f16[1024]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_153, torch.float16);  sum_dim_int_list_153 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:48 in forward, code: hidden_states = self.dense_4h_to_h(hidden_states)
        permute_default_58: "f16[1024, 2048]" = torch.ops.aten.permute.default(view_517, [1, 0])
        sum_dim_int_list_154: "f16[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_517, [0], True)
        reshape_default_76: "f16[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_154, _shape_param_76);  sum_dim_int_list_154 = _shape_param_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:46 in forward, code: hidden_states = self.dense_h_to_4h(hidden_states)
        permute_default_59: "f16[4096, 2048]" = torch.ops.aten.permute.default(view_520, [1, 0])
        sum_dim_int_list_155: "f16[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_520, [0], True);  view_520 = None
        reshape_default_77: "f16[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_155, _shape_param_77);  sum_dim_int_list_155 = _shape_param_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:280 in forward, code: mlp_output = self.mlp(self.post_attention_layernorm(hidden_states))
        mul_tensor_39: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_1091, mul_47)
        sum_dim_int_list_156: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_39, [0, 1]);  mul_tensor_39 = None
        sum_dim_int_list_157: "f32[1024]" = torch.ops.aten.sum.dim_IntList(convert_element_type_1091, [0, 1]);  convert_element_type_1091 = None
        convert_element_type_default_78: "f16[1024]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_156, torch.float16);  sum_dim_int_list_156 = None
        convert_element_type_default_79: "f16[1024]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_157, torch.float16);  sum_dim_int_list_157 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:242 in forward, code: attn_output = self.dense(attn_output)
        sum_dim_int_list_158: "f16[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_517, [0], True);  view_517 = None
        reshape_default_78: "f16[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_158, _shape_param_78);  sum_dim_int_list_158 = _shape_param_78 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:214 in forward, code: qkv = self.query_key_value(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_60: "f16[3072, 2048]" = torch.ops.aten.permute.default(view_528, [1, 0])
        sum_dim_int_list_159: "f16[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_528, [0], True);  view_528 = None
        reshape_default_79: "f16[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_159, _shape_param_79);  sum_dim_int_list_159 = _shape_param_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:269 in forward, code: self.input_layernorm(hidden_states),
        mul_tensor_40: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_1106, mul_47);  mul_47 = None
        sum_dim_int_list_160: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_40, [0, 1]);  mul_tensor_40 = None
        sum_dim_int_list_161: "f32[1024]" = torch.ops.aten.sum.dim_IntList(convert_element_type_1106, [0, 1]);  convert_element_type_1106 = None
        convert_element_type_default_80: "f16[1024]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_160, torch.float16);  sum_dim_int_list_160 = None
        convert_element_type_default_81: "f16[1024]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_161, torch.float16);  sum_dim_int_list_161 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:48 in forward, code: hidden_states = self.dense_4h_to_h(hidden_states)
        permute_default_61: "f16[1024, 2048]" = torch.ops.aten.permute.default(view_531, [1, 0])
        sum_dim_int_list_162: "f16[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_531, [0], True)
        reshape_default_80: "f16[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_162, _shape_param_80);  sum_dim_int_list_162 = _shape_param_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:46 in forward, code: hidden_states = self.dense_h_to_4h(hidden_states)
        permute_default_62: "f16[4096, 2048]" = torch.ops.aten.permute.default(view_534, [1, 0])
        sum_dim_int_list_163: "f16[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_534, [0], True);  view_534 = None
        reshape_default_81: "f16[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_163, _shape_param_81);  sum_dim_int_list_163 = _shape_param_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:280 in forward, code: mlp_output = self.mlp(self.post_attention_layernorm(hidden_states))
        mul_tensor_41: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_1124, mul_36)
        sum_dim_int_list_164: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_41, [0, 1]);  mul_tensor_41 = None
        sum_dim_int_list_165: "f32[1024]" = torch.ops.aten.sum.dim_IntList(convert_element_type_1124, [0, 1]);  convert_element_type_1124 = None
        convert_element_type_default_82: "f16[1024]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_164, torch.float16);  sum_dim_int_list_164 = None
        convert_element_type_default_83: "f16[1024]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_165, torch.float16);  sum_dim_int_list_165 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:242 in forward, code: attn_output = self.dense(attn_output)
        sum_dim_int_list_166: "f16[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_531, [0], True);  view_531 = None
        reshape_default_82: "f16[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_166, _shape_param_82);  sum_dim_int_list_166 = _shape_param_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:214 in forward, code: qkv = self.query_key_value(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_63: "f16[3072, 2048]" = torch.ops.aten.permute.default(view_542, [1, 0])
        sum_dim_int_list_167: "f16[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_542, [0], True);  view_542 = None
        reshape_default_83: "f16[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_167, _shape_param_83);  sum_dim_int_list_167 = _shape_param_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:269 in forward, code: self.input_layernorm(hidden_states),
        mul_tensor_42: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_1139, mul_36);  mul_36 = None
        sum_dim_int_list_168: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_42, [0, 1]);  mul_tensor_42 = None
        sum_dim_int_list_169: "f32[1024]" = torch.ops.aten.sum.dim_IntList(convert_element_type_1139, [0, 1]);  convert_element_type_1139 = None
        convert_element_type_default_84: "f16[1024]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_168, torch.float16);  sum_dim_int_list_168 = None
        convert_element_type_default_85: "f16[1024]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_169, torch.float16);  sum_dim_int_list_169 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:48 in forward, code: hidden_states = self.dense_4h_to_h(hidden_states)
        permute_default_64: "f16[1024, 2048]" = torch.ops.aten.permute.default(view_545, [1, 0])
        sum_dim_int_list_170: "f16[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_545, [0], True)
        reshape_default_84: "f16[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_170, _shape_param_84);  sum_dim_int_list_170 = _shape_param_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:46 in forward, code: hidden_states = self.dense_h_to_4h(hidden_states)
        permute_default_65: "f16[4096, 2048]" = torch.ops.aten.permute.default(view_548, [1, 0])
        sum_dim_int_list_171: "f16[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_548, [0], True);  view_548 = None
        reshape_default_85: "f16[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_171, _shape_param_85);  sum_dim_int_list_171 = _shape_param_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:280 in forward, code: mlp_output = self.mlp(self.post_attention_layernorm(hidden_states))
        mul_tensor_43: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_1157, mul_25)
        sum_dim_int_list_172: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_43, [0, 1]);  mul_tensor_43 = None
        sum_dim_int_list_173: "f32[1024]" = torch.ops.aten.sum.dim_IntList(convert_element_type_1157, [0, 1]);  convert_element_type_1157 = None
        convert_element_type_default_86: "f16[1024]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_172, torch.float16);  sum_dim_int_list_172 = None
        convert_element_type_default_87: "f16[1024]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_173, torch.float16);  sum_dim_int_list_173 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:242 in forward, code: attn_output = self.dense(attn_output)
        sum_dim_int_list_174: "f16[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_545, [0], True);  view_545 = None
        reshape_default_86: "f16[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_174, _shape_param_86);  sum_dim_int_list_174 = _shape_param_86 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:214 in forward, code: qkv = self.query_key_value(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_66: "f16[3072, 2048]" = torch.ops.aten.permute.default(view_556, [1, 0])
        sum_dim_int_list_175: "f16[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_556, [0], True);  view_556 = None
        reshape_default_87: "f16[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_175, _shape_param_87);  sum_dim_int_list_175 = _shape_param_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:269 in forward, code: self.input_layernorm(hidden_states),
        mul_tensor_44: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_1172, mul_25);  mul_25 = None
        sum_dim_int_list_176: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_44, [0, 1]);  mul_tensor_44 = None
        sum_dim_int_list_177: "f32[1024]" = torch.ops.aten.sum.dim_IntList(convert_element_type_1172, [0, 1]);  convert_element_type_1172 = None
        convert_element_type_default_88: "f16[1024]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_176, torch.float16);  sum_dim_int_list_176 = None
        convert_element_type_default_89: "f16[1024]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_177, torch.float16);  sum_dim_int_list_177 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:48 in forward, code: hidden_states = self.dense_4h_to_h(hidden_states)
        permute_default_67: "f16[1024, 2048]" = torch.ops.aten.permute.default(view_559, [1, 0])
        sum_dim_int_list_178: "f16[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_559, [0], True)
        reshape_default_88: "f16[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_178, _shape_param_88);  sum_dim_int_list_178 = _shape_param_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:46 in forward, code: hidden_states = self.dense_h_to_4h(hidden_states)
        permute_default_68: "f16[4096, 2048]" = torch.ops.aten.permute.default(view_562, [1, 0])
        sum_dim_int_list_179: "f16[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_562, [0], True);  view_562 = None
        reshape_default_89: "f16[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_179, _shape_param_89);  sum_dim_int_list_179 = _shape_param_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:280 in forward, code: mlp_output = self.mlp(self.post_attention_layernorm(hidden_states))
        mul_tensor_45: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_1190, mul_14)
        sum_dim_int_list_180: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_45, [0, 1]);  mul_tensor_45 = None
        sum_dim_int_list_181: "f32[1024]" = torch.ops.aten.sum.dim_IntList(convert_element_type_1190, [0, 1]);  convert_element_type_1190 = None
        convert_element_type_default_90: "f16[1024]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_180, torch.float16);  sum_dim_int_list_180 = None
        convert_element_type_default_91: "f16[1024]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_181, torch.float16);  sum_dim_int_list_181 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:242 in forward, code: attn_output = self.dense(attn_output)
        sum_dim_int_list_182: "f16[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_559, [0], True);  view_559 = None
        reshape_default_90: "f16[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_182, _shape_param_90);  sum_dim_int_list_182 = _shape_param_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:214 in forward, code: qkv = self.query_key_value(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_69: "f16[3072, 2048]" = torch.ops.aten.permute.default(view_570, [1, 0])
        sum_dim_int_list_183: "f16[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_570, [0], True);  view_570 = None
        reshape_default_91: "f16[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_183, _shape_param_91);  sum_dim_int_list_183 = _shape_param_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:269 in forward, code: self.input_layernorm(hidden_states),
        mul_tensor_46: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_1205, mul_14);  mul_14 = None
        sum_dim_int_list_184: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_46, [0, 1]);  mul_tensor_46 = None
        sum_dim_int_list_185: "f32[1024]" = torch.ops.aten.sum.dim_IntList(convert_element_type_1205, [0, 1]);  convert_element_type_1205 = None
        convert_element_type_default_92: "f16[1024]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_184, torch.float16);  sum_dim_int_list_184 = None
        convert_element_type_default_93: "f16[1024]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_185, torch.float16);  sum_dim_int_list_185 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:48 in forward, code: hidden_states = self.dense_4h_to_h(hidden_states)
        permute_default_70: "f16[1024, 2048]" = torch.ops.aten.permute.default(view_573, [1, 0])
        sum_dim_int_list_186: "f16[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_573, [0], True)
        reshape_default_92: "f16[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_186, _shape_param_92);  sum_dim_int_list_186 = _shape_param_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:46 in forward, code: hidden_states = self.dense_h_to_4h(hidden_states)
        permute_default_71: "f16[4096, 2048]" = torch.ops.aten.permute.default(view_576, [1, 0])
        sum_dim_int_list_187: "f16[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_576, [0], True);  view_576 = None
        reshape_default_93: "f16[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_187, _shape_param_93);  sum_dim_int_list_187 = _shape_param_93 = None
        reshape_default_94: "f16[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_189, _shape_param_94);  mm_189 = _shape_param_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:280 in forward, code: mlp_output = self.mlp(self.post_attention_layernorm(hidden_states))
        convert_element_type_default_94: "f32[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(reshape_default_94, torch.float32);  reshape_default_94 = None
        convert_element_type_default_95: "f32[1024]" = torch.ops.prims.convert_element_type.default(primals_10, torch.float32);  primals_10 = None
        mul_tensor_47: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_default_94, convert_element_type_default_95);  convert_element_type_default_95 = None
        mul_tensor_48: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor_47, 1024)
        sum_dim_int_list_188: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_47, [2], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:269 in forward, code: self.input_layernorm(hidden_states),
        convert_element_type_default_96: "f32[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(embedding, torch.float32);  embedding = None
        sub_tensor: "f32[4, 512, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_default_96, getitem_1);  convert_element_type_default_96 = getitem_1 = None
        mul_tensor_49: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt);  sub_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:280 in forward, code: mlp_output = self.mlp(self.post_attention_layernorm(hidden_states))
        mul_tensor_50: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor_47, mul_tensor_49);  mul_tensor_47 = None
        sum_dim_int_list_189: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_50, [2], True);  mul_tensor_50 = None
        mul_tensor_51: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor_49, sum_dim_int_list_189);  sum_dim_int_list_189 = None
        sub_tensor_1: "f32[4, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_tensor_48, sum_dim_int_list_188);  mul_tensor_48 = sum_dim_int_list_188 = None
        sub_tensor_2: "f32[4, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_tensor_1, mul_tensor_51);  sub_tensor_1 = mul_tensor_51 = None
        div_tensor: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt, 1024);  rsqrt = None
        mul_tensor_52: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(div_tensor, sub_tensor_2);  sub_tensor_2 = None
        mul_tensor_53: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_default_94, mul_tensor_49)
        sum_dim_int_list_190: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_53, [0, 1]);  mul_tensor_53 = None
        sum_dim_int_list_191: "f32[1024]" = torch.ops.aten.sum.dim_IntList(convert_element_type_default_94, [0, 1]);  convert_element_type_default_94 = None
        convert_element_type_default_97: "f16[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(mul_tensor_52, torch.float16);  mul_tensor_52 = None
        convert_element_type_default_98: "f16[1024]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_190, torch.float16);  sum_dim_int_list_190 = None
        convert_element_type_default_99: "f16[1024]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_191, torch.float16);  sum_dim_int_list_191 = None
        add_tensor: "f16[4, 512, 1024]" = torch.ops.aten.add.Tensor(add_497, convert_element_type_default_97);  add_497 = convert_element_type_default_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:242 in forward, code: attn_output = self.dense(attn_output)
        sum_dim_int_list_192: "f16[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_573, [0], True);  view_573 = None
        reshape_default_95: "f16[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_192, _shape_param_95);  sum_dim_int_list_192 = _shape_param_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:214 in forward, code: qkv = self.query_key_value(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_72: "f16[3072, 2048]" = torch.ops.aten.permute.default(view_584, [1, 0])
        sum_dim_int_list_193: "f16[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_584, [0], True);  view_584 = None
        reshape_default_96: "f16[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_193, _shape_param_96);  sum_dim_int_list_193 = _shape_param_96 = None
        reshape_default_97: "f16[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_193, _shape_param_97);  mm_193 = _shape_param_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:269 in forward, code: self.input_layernorm(hidden_states),
        convert_element_type_default_100: "f32[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(reshape_default_97, torch.float32);  reshape_default_97 = None
        convert_element_type_default_101: "f32[1024]" = torch.ops.prims.convert_element_type.default(primals_4, torch.float32);  primals_4 = None
        mul_tensor_54: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_default_100, convert_element_type_default_101);  convert_element_type_default_101 = None
        mul_tensor_55: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor_54, 1024)
        sum_dim_int_list_194: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_54, [2], True)
        mul_tensor_56: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor_54, mul_tensor_49);  mul_tensor_54 = None
        sum_dim_int_list_195: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_56, [2], True);  mul_tensor_56 = None
        mul_tensor_57: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor_49, sum_dim_int_list_195);  sum_dim_int_list_195 = None
        sub_tensor_3: "f32[4, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_tensor_55, sum_dim_int_list_194);  mul_tensor_55 = sum_dim_int_list_194 = None
        sub_tensor_4: "f32[4, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_tensor_3, mul_tensor_57);  sub_tensor_3 = mul_tensor_57 = None
        mul_tensor_58: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(div_tensor, sub_tensor_4);  div_tensor = sub_tensor_4 = None
        mul_tensor_59: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_default_100, mul_tensor_49);  mul_tensor_49 = None
        sum_dim_int_list_196: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_59, [0, 1]);  mul_tensor_59 = None
        sum_dim_int_list_197: "f32[1024]" = torch.ops.aten.sum.dim_IntList(convert_element_type_default_100, [0, 1]);  convert_element_type_default_100 = None
        convert_element_type_default_102: "f16[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(mul_tensor_58, torch.float16);  mul_tensor_58 = None
        convert_element_type_default_103: "f16[1024]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_196, torch.float16);  sum_dim_int_list_196 = None
        convert_element_type_default_104: "f16[1024]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_197, torch.float16);  sum_dim_int_list_197 = None
        add_tensor_1: "f16[4, 512, 1024]" = torch.ops.aten.add.Tensor(add_tensor, convert_element_type_default_102);  add_tensor = convert_element_type_default_102 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:345 in forward, code: inputs_embeds = self.embed_in(input_ids)
        convert_element_type_default_105: "f32[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_tensor_1, torch.float32);  add_tensor_1 = None
        eq_scalar: "b8[4, 512]" = torch.ops.aten.eq.Scalar(primals_1, -1)
        unsqueeze_default: "b8[4, 512, 1]" = torch.ops.aten.unsqueeze.default(eq_scalar, -1);  eq_scalar = None
        where_self: "f32[4, 512, 1024]" = torch.ops.aten.where.self(unsqueeze_default, full_default_49, convert_element_type_default_105);  unsqueeze_default = full_default_49 = convert_element_type_default_105 = None
        full_default: "f32[50304, 1024]" = torch.ops.aten.full.default([50304, 1024], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default: "f32[50304, 1024]" = torch.ops.aten.index_put.default(full_default, [primals_1], where_self, True);  full_default = primals_1 = where_self = None
        convert_element_type_default_106: "f16[50304, 1024]" = torch.ops.prims.convert_element_type.default(index_put_default, torch.float16);  index_put_default = None
        return (permute_default, convert_element_type_default, convert_element_type_default_1, permute_default_1, reshape_default, permute_default_2, reshape_default_1, convert_element_type_default_2, convert_element_type_default_3, reshape_default_2, permute_default_3, reshape_default_3, convert_element_type_default_4, convert_element_type_default_5, permute_default_4, reshape_default_4, permute_default_5, reshape_default_5, convert_element_type_default_6, convert_element_type_default_7, reshape_default_6, permute_default_6, reshape_default_7, convert_element_type_default_8, convert_element_type_default_9, permute_default_7, reshape_default_8, permute_default_8, reshape_default_9, convert_element_type_default_10, convert_element_type_default_11, reshape_default_10, permute_default_9, reshape_default_11, convert_element_type_default_12, convert_element_type_default_13, permute_default_10, reshape_default_12, permute_default_11, reshape_default_13, convert_element_type_default_14, convert_element_type_default_15, reshape_default_14, permute_default_12, reshape_default_15, convert_element_type_default_16, convert_element_type_default_17, permute_default_13, reshape_default_16, permute_default_14, reshape_default_17, convert_element_type_default_18, convert_element_type_default_19, reshape_default_18, permute_default_15, reshape_default_19, convert_element_type_default_20, convert_element_type_default_21, permute_default_16, reshape_default_20, permute_default_17, reshape_default_21, convert_element_type_default_22, convert_element_type_default_23, reshape_default_22, permute_default_18, reshape_default_23, convert_element_type_default_24, convert_element_type_default_25, permute_default_19, reshape_default_24, permute_default_20, reshape_default_25, convert_element_type_default_26, convert_element_type_default_27, reshape_default_26, permute_default_21, reshape_default_27, convert_element_type_default_28, convert_element_type_default_29, permute_default_22, reshape_default_28, permute_default_23, reshape_default_29, convert_element_type_default_30, convert_element_type_default_31, reshape_default_30, permute_default_24, reshape_default_31, convert_element_type_default_32, convert_element_type_default_33, permute_default_25, reshape_default_32, permute_default_26, reshape_default_33, convert_element_type_default_34, convert_element_type_default_35, reshape_default_34, permute_default_27, reshape_default_35, convert_element_type_default_36, convert_element_type_default_37, permute_default_28, reshape_default_36, permute_default_29, reshape_default_37, convert_element_type_default_38, convert_element_type_default_39, reshape_default_38, permute_default_30, reshape_default_39, convert_element_type_default_40, convert_element_type_default_41, permute_default_31, reshape_default_40, permute_default_32, reshape_default_41, convert_element_type_default_42, convert_element_type_default_43, reshape_default_42, permute_default_33, reshape_default_43, convert_element_type_default_44, convert_element_type_default_45, permute_default_34, reshape_default_44, permute_default_35, reshape_default_45, convert_element_type_default_46, convert_element_type_default_47, reshape_default_46, permute_default_36, reshape_default_47, convert_element_type_default_48, convert_element_type_default_49, permute_default_37, reshape_default_48, permute_default_38, reshape_default_49, convert_element_type_default_50, convert_element_type_default_51, reshape_default_50, permute_default_39, reshape_default_51, convert_element_type_default_52, convert_element_type_default_53, permute_default_40, reshape_default_52, permute_default_41, reshape_default_53, convert_element_type_default_54, convert_element_type_default_55, reshape_default_54, permute_default_42, reshape_default_55, convert_element_type_default_56, convert_element_type_default_57, permute_default_43, reshape_default_56, permute_default_44, reshape_default_57, convert_element_type_default_58, convert_element_type_default_59, reshape_default_58, permute_default_45, reshape_default_59, convert_element_type_default_60, convert_element_type_default_61, permute_default_46, reshape_default_60, permute_default_47, reshape_default_61, convert_element_type_default_62, convert_element_type_default_63, reshape_default_62, permute_default_48, reshape_default_63, convert_element_type_default_64, convert_element_type_default_65, permute_default_49, reshape_default_64, permute_default_50, reshape_default_65, convert_element_type_default_66, convert_element_type_default_67, reshape_default_66, permute_default_51, reshape_default_67, convert_element_type_default_68, convert_element_type_default_69, permute_default_52, reshape_default_68, permute_default_53, reshape_default_69, convert_element_type_default_70, convert_element_type_default_71, reshape_default_70, permute_default_54, reshape_default_71, convert_element_type_default_72, convert_element_type_default_73, permute_default_55, reshape_default_72, permute_default_56, reshape_default_73, convert_element_type_default_74, convert_element_type_default_75, reshape_default_74, permute_default_57, reshape_default_75, convert_element_type_default_76, convert_element_type_default_77, permute_default_58, reshape_default_76, permute_default_59, reshape_default_77, convert_element_type_default_78, convert_element_type_default_79, reshape_default_78, permute_default_60, reshape_default_79, convert_element_type_default_80, convert_element_type_default_81, permute_default_61, reshape_default_80, permute_default_62, reshape_default_81, convert_element_type_default_82, convert_element_type_default_83, reshape_default_82, permute_default_63, reshape_default_83, convert_element_type_default_84, convert_element_type_default_85, permute_default_64, reshape_default_84, permute_default_65, reshape_default_85, convert_element_type_default_86, convert_element_type_default_87, reshape_default_86, permute_default_66, reshape_default_87, convert_element_type_default_88, convert_element_type_default_89, permute_default_67, reshape_default_88, permute_default_68, reshape_default_89, convert_element_type_default_90, convert_element_type_default_91, reshape_default_90, permute_default_69, reshape_default_91, convert_element_type_default_92, convert_element_type_default_93, permute_default_70, reshape_default_92, permute_default_71, reshape_default_93, convert_element_type_default_98, convert_element_type_default_99, reshape_default_95, permute_default_72, reshape_default_96, convert_element_type_default_103, convert_element_type_default_104, convert_element_type_default_106)


def _default_make_inputs():
    return [
    torch.randn([2048, 50304], dtype=torch.float16, device='cuda'),
    torch.randn([4, 512, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 1024], dtype=torch.float16, device='cuda'),
    [1024],  # _shape_param_0
    torch.randn([2048, 4096], dtype=torch.float16, device='cuda'),
    [4096],  # _shape_param_1
    torch.randn([4, 512, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 1024], dtype=torch.float32, device='cuda'),
    [1024],  # _shape_param_2
    torch.randn([2048, 3072], dtype=torch.float16, device='cuda'),
    [3072],  # _shape_param_3
    torch.randn([4, 512, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 1024], dtype=torch.float16, device='cuda'),
    [1024],  # _shape_param_4
    torch.randn([2048, 4096], dtype=torch.float16, device='cuda'),
    [4096],  # _shape_param_5
    torch.randn([4, 512, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 1024], dtype=torch.float32, device='cuda'),
    [1024],  # _shape_param_6
    torch.randn([2048, 3072], dtype=torch.float16, device='cuda'),
    [3072],  # _shape_param_7
    torch.randn([4, 512, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 1024], dtype=torch.float16, device='cuda'),
    [1024],  # _shape_param_8
    torch.randn([2048, 4096], dtype=torch.float16, device='cuda'),
    [4096],  # _shape_param_9
    torch.randn([4, 512, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 1024], dtype=torch.float32, device='cuda'),
    [1024],  # _shape_param_10
    torch.randn([2048, 3072], dtype=torch.float16, device='cuda'),
    [3072],  # _shape_param_11
    torch.randn([4, 512, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 1024], dtype=torch.float16, device='cuda'),
    [1024],  # _shape_param_12
    torch.randn([2048, 4096], dtype=torch.float16, device='cuda'),
    [4096],  # _shape_param_13
    torch.randn([4, 512, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 1024], dtype=torch.float32, device='cuda'),
    [1024],  # _shape_param_14
    torch.randn([2048, 3072], dtype=torch.float16, device='cuda'),
    [3072],  # _shape_param_15
    torch.randn([4, 512, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 1024], dtype=torch.float16, device='cuda'),
    [1024],  # _shape_param_16
    torch.randn([2048, 4096], dtype=torch.float16, device='cuda'),
    [4096],  # _shape_param_17
    torch.randn([4, 512, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 1024], dtype=torch.float32, device='cuda'),
    [1024],  # _shape_param_18
    torch.randn([2048, 3072], dtype=torch.float16, device='cuda'),
    [3072],  # _shape_param_19
    torch.randn([4, 512, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 1024], dtype=torch.float16, device='cuda'),
    [1024],  # _shape_param_20
    torch.randn([2048, 4096], dtype=torch.float16, device='cuda'),
    [4096],  # _shape_param_21
    torch.randn([4, 512, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 1024], dtype=torch.float32, device='cuda'),
    [1024],  # _shape_param_22
    torch.randn([2048, 3072], dtype=torch.float16, device='cuda'),
    [3072],  # _shape_param_23
    torch.randn([4, 512, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 1024], dtype=torch.float16, device='cuda'),
    [1024],  # _shape_param_24
    torch.randn([2048, 4096], dtype=torch.float16, device='cuda'),
    [4096],  # _shape_param_25
    torch.randn([4, 512, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 1024], dtype=torch.float32, device='cuda'),
    [1024],  # _shape_param_26
    torch.randn([2048, 3072], dtype=torch.float16, device='cuda'),
    [3072],  # _shape_param_27
    torch.randn([4, 512, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 1024], dtype=torch.float16, device='cuda'),
    [1024],  # _shape_param_28
    torch.randn([2048, 4096], dtype=torch.float16, device='cuda'),
    [4096],  # _shape_param_29
    torch.randn([4, 512, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 1024], dtype=torch.float32, device='cuda'),
    [1024],  # _shape_param_30
    torch.randn([2048, 3072], dtype=torch.float16, device='cuda'),
    [3072],  # _shape_param_31
    torch.randn([4, 512, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 1024], dtype=torch.float16, device='cuda'),
    [1024],  # _shape_param_32
    torch.randn([2048, 4096], dtype=torch.float16, device='cuda'),
    [4096],  # _shape_param_33
    torch.randn([4, 512, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 1024], dtype=torch.float32, device='cuda'),
    [1024],  # _shape_param_34
    torch.randn([2048, 3072], dtype=torch.float16, device='cuda'),
    [3072],  # _shape_param_35
    torch.randn([4, 512, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 1024], dtype=torch.float16, device='cuda'),
    [1024],  # _shape_param_36
    torch.randn([2048, 4096], dtype=torch.float16, device='cuda'),
    [4096],  # _shape_param_37
    torch.randn([4, 512, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 1024], dtype=torch.float32, device='cuda'),
    [1024],  # _shape_param_38
    torch.randn([2048, 3072], dtype=torch.float16, device='cuda'),
    [3072],  # _shape_param_39
    torch.randn([4, 512, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 1024], dtype=torch.float16, device='cuda'),
    [1024],  # _shape_param_40
    torch.randn([2048, 4096], dtype=torch.float16, device='cuda'),
    [4096],  # _shape_param_41
    torch.randn([4, 512, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 1024], dtype=torch.float32, device='cuda'),
    [1024],  # _shape_param_42
    torch.randn([2048, 3072], dtype=torch.float16, device='cuda'),
    [3072],  # _shape_param_43
    torch.randn([4, 512, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 1024], dtype=torch.float16, device='cuda'),
    [1024],  # _shape_param_44
    torch.randn([2048, 4096], dtype=torch.float16, device='cuda'),
    [4096],  # _shape_param_45
    torch.randn([4, 512, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 1024], dtype=torch.float32, device='cuda'),
    [1024],  # _shape_param_46
    torch.randn([2048, 3072], dtype=torch.float16, device='cuda'),
    [3072],  # _shape_param_47
    torch.randn([4, 512, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 1024], dtype=torch.float16, device='cuda'),
    [1024],  # _shape_param_48
    torch.randn([2048, 4096], dtype=torch.float16, device='cuda'),
    [4096],  # _shape_param_49
    torch.randn([4, 512, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 1024], dtype=torch.float32, device='cuda'),
    [1024],  # _shape_param_50
    torch.randn([2048, 3072], dtype=torch.float16, device='cuda'),
    [3072],  # _shape_param_51
    torch.randn([4, 512, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 1024], dtype=torch.float16, device='cuda'),
    [1024],  # _shape_param_52
    torch.randn([2048, 4096], dtype=torch.float16, device='cuda'),
    [4096],  # _shape_param_53
    torch.randn([4, 512, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 1024], dtype=torch.float32, device='cuda'),
    [1024],  # _shape_param_54
    torch.randn([2048, 3072], dtype=torch.float16, device='cuda'),
    [3072],  # _shape_param_55
    torch.randn([4, 512, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 1024], dtype=torch.float16, device='cuda'),
    [1024],  # _shape_param_56
    torch.randn([2048, 4096], dtype=torch.float16, device='cuda'),
    [4096],  # _shape_param_57
    torch.randn([4, 512, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 1024], dtype=torch.float32, device='cuda'),
    [1024],  # _shape_param_58
    torch.randn([2048, 3072], dtype=torch.float16, device='cuda'),
    [3072],  # _shape_param_59
    torch.randn([4, 512, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 1024], dtype=torch.float16, device='cuda'),
    [1024],  # _shape_param_60
    torch.randn([2048, 4096], dtype=torch.float16, device='cuda'),
    [4096],  # _shape_param_61
    torch.randn([4, 512, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 1024], dtype=torch.float32, device='cuda'),
    [1024],  # _shape_param_62
    torch.randn([2048, 3072], dtype=torch.float16, device='cuda'),
    [3072],  # _shape_param_63
    torch.randn([4, 512, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 1024], dtype=torch.float16, device='cuda'),
    [1024],  # _shape_param_64
    torch.randn([2048, 4096], dtype=torch.float16, device='cuda'),
    [4096],  # _shape_param_65
    torch.randn([4, 512, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 1024], dtype=torch.float32, device='cuda'),
    [1024],  # _shape_param_66
    torch.randn([2048, 3072], dtype=torch.float16, device='cuda'),
    [3072],  # _shape_param_67
    torch.randn([4, 512, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 1024], dtype=torch.float16, device='cuda'),
    [1024],  # _shape_param_68
    torch.randn([2048, 4096], dtype=torch.float16, device='cuda'),
    [4096],  # _shape_param_69
    torch.randn([4, 512, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 1024], dtype=torch.float32, device='cuda'),
    [1024],  # _shape_param_70
    torch.randn([2048, 3072], dtype=torch.float16, device='cuda'),
    [3072],  # _shape_param_71
    torch.randn([4, 512, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 1024], dtype=torch.float16, device='cuda'),
    [1024],  # _shape_param_72
    torch.randn([2048, 4096], dtype=torch.float16, device='cuda'),
    [4096],  # _shape_param_73
    torch.randn([4, 512, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 1024], dtype=torch.float32, device='cuda'),
    [1024],  # _shape_param_74
    torch.randn([2048, 3072], dtype=torch.float16, device='cuda'),
    [3072],  # _shape_param_75
    torch.randn([4, 512, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 1024], dtype=torch.float16, device='cuda'),
    [1024],  # _shape_param_76
    torch.randn([2048, 4096], dtype=torch.float16, device='cuda'),
    [4096],  # _shape_param_77
    torch.randn([4, 512, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 1024], dtype=torch.float32, device='cuda'),
    [1024],  # _shape_param_78
    torch.randn([2048, 3072], dtype=torch.float16, device='cuda'),
    [3072],  # _shape_param_79
    torch.randn([4, 512, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 1024], dtype=torch.float16, device='cuda'),
    [1024],  # _shape_param_80
    torch.randn([2048, 4096], dtype=torch.float16, device='cuda'),
    [4096],  # _shape_param_81
    torch.randn([4, 512, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 1024], dtype=torch.float32, device='cuda'),
    [1024],  # _shape_param_82
    torch.randn([2048, 3072], dtype=torch.float16, device='cuda'),
    [3072],  # _shape_param_83
    torch.randn([4, 512, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 1024], dtype=torch.float16, device='cuda'),
    [1024],  # _shape_param_84
    torch.randn([2048, 4096], dtype=torch.float16, device='cuda'),
    [4096],  # _shape_param_85
    torch.randn([4, 512, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 1024], dtype=torch.float32, device='cuda'),
    [1024],  # _shape_param_86
    torch.randn([2048, 3072], dtype=torch.float16, device='cuda'),
    [3072],  # _shape_param_87
    torch.randn([4, 512, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 1024], dtype=torch.float16, device='cuda'),
    [1024],  # _shape_param_88
    torch.randn([2048, 4096], dtype=torch.float16, device='cuda'),
    [4096],  # _shape_param_89
    torch.randn([4, 512, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 1024], dtype=torch.float32, device='cuda'),
    [1024],  # _shape_param_90
    torch.randn([2048, 3072], dtype=torch.float16, device='cuda'),
    [3072],  # _shape_param_91
    torch.randn([4, 512, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 1024], dtype=torch.float16, device='cuda'),
    [1024],  # _shape_param_92
    torch.randn([2048, 4096], dtype=torch.float16, device='cuda'),
    [4096],  # _shape_param_93
    torch.randn([2048, 1024], dtype=torch.float16, device='cuda'),
    [4, 512, 1024],  # _shape_param_94
    torch.randn([1024], dtype=torch.float16, device='cuda'),
    torch.randn([4, 512, 1024], dtype=torch.float16, device='cuda'),
    torch.randn([4, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 1024], dtype=torch.float16, device='cuda'),
    [1024],  # _shape_param_95
    torch.randn([2048, 3072], dtype=torch.float16, device='cuda'),
    [3072],  # _shape_param_96
    torch.randn([2048, 1024], dtype=torch.float16, device='cuda'),
    [4, 512, 1024],  # _shape_param_97
    torch.randn([1024], dtype=torch.float16, device='cuda'),
    torch.randint(0, 2, [4, 512], dtype=torch.int64, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
