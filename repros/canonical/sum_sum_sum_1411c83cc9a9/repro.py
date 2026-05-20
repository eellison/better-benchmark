"""
Standalone repro captured via capture_hook.
Label: hf_AlbertForMaskedLM_train
Pattern hash: 1411c83cc9a9
Shape hash: 8269d65b
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([4096, 30000], f32), T([8, 512, 128], f32), T([8, 512, 128], f32), T([4096, 128], f32), T([8, 512, 4096], f32), T([8, 512, 4096], f32), T([4096, 4096], f32), T([4096, 16384], f32), T([8, 512, 4096], f32), T([8, 512, 4096], f32), T([4096, 4096], f32), T([4096, 4096], f32), T([4096, 4096], f32), T([4096, 4096], f32), T([8, 512, 4096], f32), T([8, 512, 4096], f32), T([4096, 4096], f32), T([4096, 16384], f32), T([4096, 16384], f32), T([4096, 16384], f32), T([16384, 4096], f32), T([16384, 4096], f32), T([8, 512, 4096], f32), T([8, 512, 4096], f32), T([4096, 4096], f32), T([4096, 4096], f32), T([4096, 4096], f32), T([4096, 4096], f32), T([4096, 4096], f32), T([4096, 4096], f32), T([4096, 4096], f32), T([4096, 4096], f32), T([4096, 4096], f32), T([4096, 4096], f32), T([4096, 4096], f32), T([4096, 4096], f32), T([8, 512, 4096], f32), T([8, 512, 4096], f32), T([4096, 4096], f32), T([4096, 16384], f32), T([4096, 16384], f32), T([16384, 4096], f32), T([8, 512, 4096], f32), T([8, 512, 4096], f32), T([4096, 4096], f32), T([4096, 4096], f32), T([4096, 4096], f32), T([4096, 4096], f32), T([4096, 4096], f32), T([4096, 4096], f32), T([4096, 4096], f32), T([4096, 4096], f32), T([8, 512, 4096], f32), T([8, 512, 4096], f32), T([4096, 4096], f32), T([4096, 16384], f32), T([4096, 16384], f32), T([16384, 4096], f32), T([8, 512, 4096], f32), T([8, 512, 4096], f32), T([4096, 4096], f32), T([4096, 4096], f32), T([4096, 4096], f32), T([4096, 4096], f32), T([4096, 4096], f32), T([4096, 4096], f32), T([4096, 4096], f32), T([4096, 4096], f32), T([8, 512, 4096], f32), T([8, 512, 4096], f32), T([4096, 4096], f32), T([4096, 16384], f32), T([4096, 16384], f32), T([16384, 4096], f32), T([8, 512, 4096], f32), T([8, 512, 4096], f32), T([4096, 4096], f32), T([4096, 4096], f32), T([4096, 4096], f32), T([4096, 4096], f32), T([4096, 4096], f32), T([4096, 4096], f32), T([4096, 4096], f32), T([4096, 4096], f32), T([8, 512, 4096], f32), T([8, 512, 4096], f32), T([4096, 4096], f32), T([4096, 16384], f32), T([4096, 16384], f32), T([16384, 4096], f32), T([8, 512, 4096], f32), T([8, 512, 4096], f32), T([4096, 4096], f32), T([4096, 4096], f32), T([4096, 4096], f32), T([4096, 4096], f32), T([4096, 4096], f32), T([4096, 4096], f32), T([4096, 4096], f32), T([4096, 4096], f32), T([8, 512, 4096], f32), T([8, 512, 4096], f32), T([4096, 4096], f32), T([4096, 16384], f32), T([4096, 16384], f32), T([16384, 4096], f32), T([8, 512, 4096], f32), T([8, 512, 4096], f32), T([4096, 4096], f32), T([4096, 4096], f32), T([4096, 4096], f32), T([4096, 4096], f32), T([4096, 4096], f32), T([4096, 4096], f32), T([4096, 4096], f32), T([4096, 4096], f32), T([8, 512, 4096], f32), T([8, 512, 4096], f32), T([4096, 4096], f32), T([4096, 16384], f32), T([4096, 16384], f32), T([16384, 4096], f32), T([8, 512, 4096], f32), T([8, 512, 4096], f32), T([4096, 4096], f32), T([4096, 4096], f32), T([4096, 4096], f32), T([4096, 4096], f32), T([4096, 4096], f32), T([4096, 4096], f32), T([4096, 4096], f32), T([4096, 4096], f32), T([8, 512, 4096], f32), T([8, 512, 4096], f32), T([4096, 4096], f32), T([4096, 16384], f32), T([4096, 16384], f32), T([16384, 4096], f32), T([8, 512, 4096], f32), T([8, 512, 4096], f32), T([4096, 4096], f32), T([4096, 4096], f32), T([4096, 4096], f32), T([4096, 4096], f32), T([4096, 4096], f32), T([4096, 4096], f32), T([4096, 4096], f32), T([4096, 4096], f32), T([8, 512, 4096], f32), T([8, 512, 4096], f32), T([4096, 4096], f32), T([4096, 16384], f32), T([4096, 16384], f32), T([16384, 4096], f32), T([8, 512, 4096], f32), T([8, 512, 4096], f32), T([4096, 4096], f32), T([4096, 4096], f32), T([4096, 4096], f32), T([4096, 4096], f32), T([4096, 4096], f32), T([4096, 4096], f32), T([4096, 4096], f32), T([4096, 4096], f32), T([8, 512, 4096], f32), T([8, 512, 4096], f32), T([4096, 4096], f32), T([4096, 16384], f32), T([4096, 16384], f32), T([16384, 4096], f32), T([8, 512, 4096], f32), T([8, 512, 4096], f32), T([4096, 4096], f32), T([4096, 4096], f32), T([4096, 4096], f32), T([4096, 4096], f32), T([4096, 4096], f32), T([4096, 4096], f32), T([4096, 4096], f32), T([4096, 4096], f32), T([8, 512, 4096], f32), T([8, 512, 4096], f32), T([4096, 4096], f32), T([4096, 16384], f32), T([4096, 16384], f32), T([16384, 4096], f32), T([8, 512, 4096], f32), T([8, 512, 4096], f32), T([4096, 4096], f32), T([4096, 4096], f32), T([4096, 4096], f32), T([4096, 4096], f32), T([4096, 4096], f32), T([4096, 4096], f32), T([4096, 4096], f32), T([4096, 4096], f32), T([4096, 4096], f32), T([4096, 128], f32), T([128], f32), T([8, 512, 128], f32), T([8, 512, 1], f32), T([1, 512], i64, gen=Index(512)), T([], f32), T([1, 512], i64, gen=Index(2)), T([8, 512], i64, gen=Index(30000)), T([30000, 128], f32), S([30000]), S([128]), S([4096]), S([16384]), S([4096]), S([4096]), S([4096]), S([4096]), S([4096]), S([16384]), S([4096]), S([4096]), S([4096]), S([4096]), S([4096]), S([16384]), S([4096]), S([4096]), S([4096]), S([4096]), S([4096]), S([16384]), S([4096]), S([4096]), S([4096]), S([4096]), S([4096]), S([16384]), S([4096]), S([4096]), S([4096]), S([4096]), S([4096]), S([16384]), S([4096]), S([4096]), S([4096]), S([4096]), S([4096]), S([16384]), S([4096]), S([4096]), S([4096]), S([4096]), S([4096]), S([16384]), S([4096]), S([4096]), S([4096]), S([4096]), S([4096]), S([16384]), S([4096]), S([4096]), S([4096]), S([4096]), S([4096]), S([16384]), S([4096]), S([4096]), S([4096]), S([4096]), S([4096]), S([16384]), S([4096]), S([4096]), S([4096]), S([4096]), S([4096]), S([16384]), S([4096]), S([4096]), S([4096]), S([4096]), S([4096]), S([8, 512, 128]), S([8, 512]))"

class Repro(torch.nn.Module):
    def forward(self, view_273: "f32[4096, 30000]", view_275: "f32[8, 512, 128]", mul_126: "f32[8, 512, 128]", view_276: "f32[4096, 128]", view_278: "f32[8, 512, 4096]", mul_120: "f32[8, 512, 4096]", view_279: "f32[4096, 4096]", view_282: "f32[4096, 16384]", add_123: "f32[8, 512, 4096]", mul_114: "f32[8, 512, 4096]", view_285: "f32[4096, 4096]", view_296: "f32[4096, 4096]", view_300: "f32[4096, 4096]", view_304: "f32[4096, 4096]", add_126: "f32[8, 512, 4096]", mul_110: "f32[8, 512, 4096]", view_307: "f32[4096, 4096]", mm_5: "f32[4096, 16384]", mm_17: "f32[4096, 16384]", view_310: "f32[4096, 16384]", mm_7: "f32[16384, 4096]", mm_19: "f32[16384, 4096]", add_135: "f32[8, 512, 4096]", mul_104: "f32[8, 512, 4096]", view_313: "f32[4096, 4096]", mm_9: "f32[4096, 4096]", mm_21: "f32[4096, 4096]", view_324: "f32[4096, 4096]", mm_11: "f32[4096, 4096]", mm_23: "f32[4096, 4096]", view_328: "f32[4096, 4096]", mm_13: "f32[4096, 4096]", mm_25: "f32[4096, 4096]", view_332: "f32[4096, 4096]", mm_15: "f32[4096, 4096]", mm_27: "f32[4096, 4096]", add_148: "f32[8, 512, 4096]", mul_100: "f32[8, 512, 4096]", view_335: "f32[4096, 4096]", mm_29: "f32[4096, 16384]", view_338: "f32[4096, 16384]", mm_31: "f32[16384, 4096]", add_157: "f32[8, 512, 4096]", mul_94: "f32[8, 512, 4096]", view_341: "f32[4096, 4096]", mm_33: "f32[4096, 4096]", view_352: "f32[4096, 4096]", mm_35: "f32[4096, 4096]", view_356: "f32[4096, 4096]", mm_37: "f32[4096, 4096]", view_360: "f32[4096, 4096]", mm_39: "f32[4096, 4096]", add_170: "f32[8, 512, 4096]", mul_90: "f32[8, 512, 4096]", view_363: "f32[4096, 4096]", mm_41: "f32[4096, 16384]", view_366: "f32[4096, 16384]", mm_43: "f32[16384, 4096]", add_179: "f32[8, 512, 4096]", mul_84: "f32[8, 512, 4096]", view_369: "f32[4096, 4096]", mm_45: "f32[4096, 4096]", view_380: "f32[4096, 4096]", mm_47: "f32[4096, 4096]", view_384: "f32[4096, 4096]", mm_49: "f32[4096, 4096]", view_388: "f32[4096, 4096]", mm_51: "f32[4096, 4096]", add_192: "f32[8, 512, 4096]", mul_80: "f32[8, 512, 4096]", view_391: "f32[4096, 4096]", mm_53: "f32[4096, 16384]", view_394: "f32[4096, 16384]", mm_55: "f32[16384, 4096]", add_201: "f32[8, 512, 4096]", mul_74: "f32[8, 512, 4096]", view_397: "f32[4096, 4096]", mm_57: "f32[4096, 4096]", view_408: "f32[4096, 4096]", mm_59: "f32[4096, 4096]", view_412: "f32[4096, 4096]", mm_61: "f32[4096, 4096]", view_416: "f32[4096, 4096]", mm_63: "f32[4096, 4096]", add_214: "f32[8, 512, 4096]", mul_70: "f32[8, 512, 4096]", view_419: "f32[4096, 4096]", mm_65: "f32[4096, 16384]", view_422: "f32[4096, 16384]", mm_67: "f32[16384, 4096]", add_223: "f32[8, 512, 4096]", mul_64: "f32[8, 512, 4096]", view_425: "f32[4096, 4096]", mm_69: "f32[4096, 4096]", view_436: "f32[4096, 4096]", mm_71: "f32[4096, 4096]", view_440: "f32[4096, 4096]", mm_73: "f32[4096, 4096]", view_444: "f32[4096, 4096]", mm_75: "f32[4096, 4096]", add_236: "f32[8, 512, 4096]", mul_60: "f32[8, 512, 4096]", view_447: "f32[4096, 4096]", mm_77: "f32[4096, 16384]", view_450: "f32[4096, 16384]", mm_79: "f32[16384, 4096]", add_245: "f32[8, 512, 4096]", mul_54: "f32[8, 512, 4096]", view_453: "f32[4096, 4096]", mm_81: "f32[4096, 4096]", view_464: "f32[4096, 4096]", mm_83: "f32[4096, 4096]", view_468: "f32[4096, 4096]", mm_85: "f32[4096, 4096]", view_472: "f32[4096, 4096]", mm_87: "f32[4096, 4096]", add_258: "f32[8, 512, 4096]", mul_50: "f32[8, 512, 4096]", view_475: "f32[4096, 4096]", mm_89: "f32[4096, 16384]", view_478: "f32[4096, 16384]", mm_91: "f32[16384, 4096]", add_267: "f32[8, 512, 4096]", mul_44: "f32[8, 512, 4096]", view_481: "f32[4096, 4096]", mm_93: "f32[4096, 4096]", view_492: "f32[4096, 4096]", mm_95: "f32[4096, 4096]", view_496: "f32[4096, 4096]", mm_97: "f32[4096, 4096]", view_500: "f32[4096, 4096]", mm_99: "f32[4096, 4096]", add_280: "f32[8, 512, 4096]", mul_40: "f32[8, 512, 4096]", view_503: "f32[4096, 4096]", mm_101: "f32[4096, 16384]", view_506: "f32[4096, 16384]", mm_103: "f32[16384, 4096]", add_289: "f32[8, 512, 4096]", mul_34: "f32[8, 512, 4096]", view_509: "f32[4096, 4096]", mm_105: "f32[4096, 4096]", view_520: "f32[4096, 4096]", mm_107: "f32[4096, 4096]", view_524: "f32[4096, 4096]", mm_109: "f32[4096, 4096]", view_528: "f32[4096, 4096]", mm_111: "f32[4096, 4096]", add_302: "f32[8, 512, 4096]", mul_30: "f32[8, 512, 4096]", view_531: "f32[4096, 4096]", mm_113: "f32[4096, 16384]", view_534: "f32[4096, 16384]", mm_115: "f32[16384, 4096]", add_311: "f32[8, 512, 4096]", mul_24: "f32[8, 512, 4096]", view_537: "f32[4096, 4096]", mm_117: "f32[4096, 4096]", view_548: "f32[4096, 4096]", mm_119: "f32[4096, 4096]", view_552: "f32[4096, 4096]", mm_121: "f32[4096, 4096]", view_556: "f32[4096, 4096]", mm_123: "f32[4096, 4096]", add_324: "f32[8, 512, 4096]", mul_20: "f32[8, 512, 4096]", view_559: "f32[4096, 4096]", mm_125: "f32[4096, 16384]", view_562: "f32[4096, 16384]", mm_127: "f32[16384, 4096]", add_333: "f32[8, 512, 4096]", mul_14: "f32[8, 512, 4096]", view_565: "f32[4096, 4096]", mm_129: "f32[4096, 4096]", view_576: "f32[4096, 4096]", mm_131: "f32[4096, 4096]", view_580: "f32[4096, 4096]", mm_133: "f32[4096, 4096]", view_584: "f32[4096, 4096]", mm_135: "f32[4096, 4096]", add_346: "f32[8, 512, 4096]", mul_10: "f32[8, 512, 4096]", view_587: "f32[4096, 4096]", mm_137: "f32[4096, 16384]", view_590: "f32[4096, 16384]", mm_139: "f32[16384, 4096]", add_355: "f32[8, 512, 4096]", mul_4: "f32[8, 512, 4096]", view_593: "f32[4096, 4096]", mm_141: "f32[4096, 4096]", view_604: "f32[4096, 4096]", mm_143: "f32[4096, 4096]", view_608: "f32[4096, 4096]", mm_145: "f32[4096, 4096]", view_612: "f32[4096, 4096]", mm_147: "f32[4096, 4096]", view_615: "f32[4096, 4096]", mm_148: "f32[4096, 128]", primals_7: "f32[128]", mul: "f32[8, 512, 128]", div_39: "f32[8, 512, 1]", primals_2: "i64[1, 512]", full_default_1: "f32[]", gather: "i64[1, 512]", primals_1: "i64[8, 512]", mm_1: "f32[30000, 128]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12, _shape_param_13, _shape_param_14, _shape_param_15, _shape_param_16, _shape_param_17, _shape_param_18, _shape_param_19, _shape_param_20, _shape_param_21, _shape_param_22, _shape_param_23, _shape_param_24, _shape_param_25, _shape_param_26, _shape_param_27, _shape_param_28, _shape_param_29, _shape_param_30, _shape_param_31, _shape_param_32, _shape_param_33, _shape_param_34, _shape_param_35, _shape_param_36, _shape_param_37, _shape_param_38, _shape_param_39, _shape_param_40, _shape_param_41, _shape_param_42, _shape_param_43, _shape_param_44, _shape_param_45, _shape_param_46, _shape_param_47, _shape_param_48, _shape_param_49, _shape_param_50, _shape_param_51, _shape_param_52, _shape_param_53, _shape_param_54, _shape_param_55, _shape_param_56, _shape_param_57, _shape_param_58, _shape_param_59, _shape_param_60, _shape_param_61, _shape_param_62, _shape_param_63, _shape_param_64, _shape_param_65, _shape_param_66, _shape_param_67, _shape_param_68, _shape_param_69, _shape_param_70, _shape_param_71, _shape_param_72, _shape_param_73, _shape_param_74, _shape_param_75, _shape_param_76):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:541 in forward, code: hidden_states = self.decoder(hidden_states)
        sum_dim_int_list: "f32[1, 30000]" = torch.ops.aten.sum.dim_IntList(view_273, [0], True);  view_273 = None
        reshape_default: "f32[30000]" = torch.ops.aten.reshape.default(sum_dim_int_list, _shape_param_0);  sum_dim_int_list = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:540 in forward, code: hidden_states = self.LayerNorm(hidden_states)
        mul_tensor: "f32[8, 512, 128]" = torch.ops.aten.mul.Tensor(view_275, mul_126);  mul_126 = None
        sum_dim_int_list_1: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [0, 1]);  mul_tensor = None
        sum_dim_int_list_2: "f32[128]" = torch.ops.aten.sum.dim_IntList(view_275, [0, 1]);  view_275 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:538 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default: "f32[128, 4096]" = torch.ops.aten.permute.default(view_276, [1, 0])
        sum_dim_int_list_3: "f32[1, 128]" = torch.ops.aten.sum.dim_IntList(view_276, [0], True);  view_276 = None
        reshape_default_1: "f32[128]" = torch.ops.aten.reshape.default(sum_dim_int_list_3, _shape_param_1);  sum_dim_int_list_3 = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:235 in forward, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output)
        mul_tensor_1: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(view_278, mul_120);  mul_120 = None
        sum_dim_int_list_4: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_tensor_1, [0, 1]);  mul_tensor_1 = None
        sum_dim_int_list_5: "f32[4096]" = torch.ops.aten.sum.dim_IntList(view_278, [0, 1]);  view_278 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:241 in ff_chunk, code: ffn_output = self.ffn_output(ffn_output)
        sum_dim_int_list_6: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_279, [0], True);  view_279 = None
        reshape_default_2: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_6, _shape_param_2);  sum_dim_int_list_6 = _shape_param_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:239 in ff_chunk, code: ffn_output = self.ffn(attention_output)
        sum_dim_int_list_7: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_282, [0], True);  view_282 = None
        reshape_default_3: "f32[16384]" = torch.ops.aten.reshape.default(sum_dim_int_list_7, _shape_param_3);  sum_dim_int_list_7 = _shape_param_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:202 in forward, code: attn_output = self.LayerNorm(hidden_states + attn_output)
        mul_tensor_2: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_123, mul_114);  mul_114 = None
        sum_dim_int_list_8: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [0, 1]);  mul_tensor_2 = None
        sum_dim_int_list_9: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_123, [0, 1]);  add_123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:200 in forward, code: attn_output = self.dense(attn_output)
        sum_dim_int_list_10: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_285, [0], True);  view_285 = None
        reshape_default_4: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_10, _shape_param_4);  sum_dim_int_list_10 = _shape_param_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:182 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        sum_dim_int_list_11: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_296, [0], True);  view_296 = None
        reshape_default_5: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_11, _shape_param_5);  sum_dim_int_list_11 = _shape_param_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:181 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        sum_dim_int_list_12: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_300, [0], True);  view_300 = None
        reshape_default_6: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_12, _shape_param_6);  sum_dim_int_list_12 = _shape_param_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:180 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        sum_dim_int_list_13: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_304, [0], True);  view_304 = None
        reshape_default_7: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_13, _shape_param_7);  sum_dim_int_list_13 = _shape_param_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:235 in forward, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output)
        mul_tensor_3: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_126, mul_110);  mul_110 = None
        sum_dim_int_list_14: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_tensor_3, [0, 1]);  mul_tensor_3 = None
        sum_dim_int_list_15: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_126, [0, 1]);  add_126 = None
        add_tensor: "f32[4096]" = torch.ops.aten.add.Tensor(sum_dim_int_list_4, sum_dim_int_list_14);  sum_dim_int_list_4 = sum_dim_int_list_14 = None
        add_tensor_1: "f32[4096]" = torch.ops.aten.add.Tensor(sum_dim_int_list_5, sum_dim_int_list_15);  sum_dim_int_list_5 = sum_dim_int_list_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:241 in ff_chunk, code: ffn_output = self.ffn_output(ffn_output)
        sum_dim_int_list_16: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_307, [0], True);  view_307 = None
        reshape_default_8: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_16, _shape_param_8);  sum_dim_int_list_16 = _shape_param_8 = None
        add_tensor_2: "f32[4096]" = torch.ops.aten.add.Tensor(reshape_default_2, reshape_default_8);  reshape_default_2 = reshape_default_8 = None
        add_tensor_3: "f32[4096, 16384]" = torch.ops.aten.add.Tensor(mm_5, mm_17);  mm_5 = mm_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:239 in ff_chunk, code: ffn_output = self.ffn(attention_output)
        sum_dim_int_list_17: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_310, [0], True);  view_310 = None
        reshape_default_9: "f32[16384]" = torch.ops.aten.reshape.default(sum_dim_int_list_17, _shape_param_9);  sum_dim_int_list_17 = _shape_param_9 = None
        add_tensor_4: "f32[16384]" = torch.ops.aten.add.Tensor(reshape_default_3, reshape_default_9);  reshape_default_3 = reshape_default_9 = None
        add_tensor_5: "f32[16384, 4096]" = torch.ops.aten.add.Tensor(mm_7, mm_19);  mm_7 = mm_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:202 in forward, code: attn_output = self.LayerNorm(hidden_states + attn_output)
        mul_tensor_4: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_135, mul_104);  mul_104 = None
        sum_dim_int_list_18: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_tensor_4, [0, 1]);  mul_tensor_4 = None
        sum_dim_int_list_19: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_135, [0, 1]);  add_135 = None
        add_tensor_6: "f32[4096]" = torch.ops.aten.add.Tensor(sum_dim_int_list_8, sum_dim_int_list_18);  sum_dim_int_list_8 = sum_dim_int_list_18 = None
        add_tensor_7: "f32[4096]" = torch.ops.aten.add.Tensor(sum_dim_int_list_9, sum_dim_int_list_19);  sum_dim_int_list_9 = sum_dim_int_list_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:200 in forward, code: attn_output = self.dense(attn_output)
        sum_dim_int_list_20: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_313, [0], True);  view_313 = None
        reshape_default_10: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_20, _shape_param_10);  sum_dim_int_list_20 = _shape_param_10 = None
        add_tensor_8: "f32[4096]" = torch.ops.aten.add.Tensor(reshape_default_4, reshape_default_10);  reshape_default_4 = reshape_default_10 = None
        add_tensor_9: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(mm_9, mm_21);  mm_9 = mm_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:182 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        sum_dim_int_list_21: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_324, [0], True);  view_324 = None
        reshape_default_11: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_21, _shape_param_11);  sum_dim_int_list_21 = _shape_param_11 = None
        add_tensor_10: "f32[4096]" = torch.ops.aten.add.Tensor(reshape_default_5, reshape_default_11);  reshape_default_5 = reshape_default_11 = None
        add_tensor_11: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(mm_11, mm_23);  mm_11 = mm_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:181 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        sum_dim_int_list_22: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_328, [0], True);  view_328 = None
        reshape_default_12: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_22, _shape_param_12);  sum_dim_int_list_22 = _shape_param_12 = None
        add_tensor_12: "f32[4096]" = torch.ops.aten.add.Tensor(reshape_default_6, reshape_default_12);  reshape_default_6 = reshape_default_12 = None
        add_tensor_13: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(mm_13, mm_25);  mm_13 = mm_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:180 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        sum_dim_int_list_23: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_332, [0], True);  view_332 = None
        reshape_default_13: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_23, _shape_param_13);  sum_dim_int_list_23 = _shape_param_13 = None
        add_tensor_14: "f32[4096]" = torch.ops.aten.add.Tensor(reshape_default_7, reshape_default_13);  reshape_default_7 = reshape_default_13 = None
        add_tensor_15: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(mm_15, mm_27);  mm_15 = mm_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:235 in forward, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output)
        mul_tensor_5: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_148, mul_100);  mul_100 = None
        sum_dim_int_list_24: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_tensor_5, [0, 1]);  mul_tensor_5 = None
        sum_dim_int_list_25: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_148, [0, 1]);  add_148 = None
        add_tensor_16: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor, sum_dim_int_list_24);  add_tensor = sum_dim_int_list_24 = None
        add_tensor_17: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_1, sum_dim_int_list_25);  add_tensor_1 = sum_dim_int_list_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:241 in ff_chunk, code: ffn_output = self.ffn_output(ffn_output)
        sum_dim_int_list_26: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_335, [0], True);  view_335 = None
        reshape_default_14: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_26, _shape_param_14);  sum_dim_int_list_26 = _shape_param_14 = None
        add_tensor_18: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_2, reshape_default_14);  add_tensor_2 = reshape_default_14 = None
        add_tensor_19: "f32[4096, 16384]" = torch.ops.aten.add.Tensor(add_tensor_3, mm_29);  add_tensor_3 = mm_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:239 in ff_chunk, code: ffn_output = self.ffn(attention_output)
        sum_dim_int_list_27: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_338, [0], True);  view_338 = None
        reshape_default_15: "f32[16384]" = torch.ops.aten.reshape.default(sum_dim_int_list_27, _shape_param_15);  sum_dim_int_list_27 = _shape_param_15 = None
        add_tensor_20: "f32[16384]" = torch.ops.aten.add.Tensor(add_tensor_4, reshape_default_15);  add_tensor_4 = reshape_default_15 = None
        add_tensor_21: "f32[16384, 4096]" = torch.ops.aten.add.Tensor(add_tensor_5, mm_31);  add_tensor_5 = mm_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:202 in forward, code: attn_output = self.LayerNorm(hidden_states + attn_output)
        mul_tensor_6: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_157, mul_94);  mul_94 = None
        sum_dim_int_list_28: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_tensor_6, [0, 1]);  mul_tensor_6 = None
        sum_dim_int_list_29: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_157, [0, 1]);  add_157 = None
        add_tensor_22: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_6, sum_dim_int_list_28);  add_tensor_6 = sum_dim_int_list_28 = None
        add_tensor_23: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_7, sum_dim_int_list_29);  add_tensor_7 = sum_dim_int_list_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:200 in forward, code: attn_output = self.dense(attn_output)
        sum_dim_int_list_30: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_341, [0], True);  view_341 = None
        reshape_default_16: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_30, _shape_param_16);  sum_dim_int_list_30 = _shape_param_16 = None
        add_tensor_24: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_8, reshape_default_16);  add_tensor_8 = reshape_default_16 = None
        add_tensor_25: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_tensor_9, mm_33);  add_tensor_9 = mm_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:182 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        sum_dim_int_list_31: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_352, [0], True);  view_352 = None
        reshape_default_17: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_31, _shape_param_17);  sum_dim_int_list_31 = _shape_param_17 = None
        add_tensor_26: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_10, reshape_default_17);  add_tensor_10 = reshape_default_17 = None
        add_tensor_27: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_tensor_11, mm_35);  add_tensor_11 = mm_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:181 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        sum_dim_int_list_32: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_356, [0], True);  view_356 = None
        reshape_default_18: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_32, _shape_param_18);  sum_dim_int_list_32 = _shape_param_18 = None
        add_tensor_28: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_12, reshape_default_18);  add_tensor_12 = reshape_default_18 = None
        add_tensor_29: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_tensor_13, mm_37);  add_tensor_13 = mm_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:180 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        sum_dim_int_list_33: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_360, [0], True);  view_360 = None
        reshape_default_19: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_33, _shape_param_19);  sum_dim_int_list_33 = _shape_param_19 = None
        add_tensor_30: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_14, reshape_default_19);  add_tensor_14 = reshape_default_19 = None
        add_tensor_31: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_tensor_15, mm_39);  add_tensor_15 = mm_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:235 in forward, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output)
        mul_tensor_7: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_170, mul_90);  mul_90 = None
        sum_dim_int_list_34: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_tensor_7, [0, 1]);  mul_tensor_7 = None
        sum_dim_int_list_35: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_170, [0, 1]);  add_170 = None
        add_tensor_32: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_16, sum_dim_int_list_34);  add_tensor_16 = sum_dim_int_list_34 = None
        add_tensor_33: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_17, sum_dim_int_list_35);  add_tensor_17 = sum_dim_int_list_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:241 in ff_chunk, code: ffn_output = self.ffn_output(ffn_output)
        sum_dim_int_list_36: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_363, [0], True);  view_363 = None
        reshape_default_20: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_36, _shape_param_20);  sum_dim_int_list_36 = _shape_param_20 = None
        add_tensor_34: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_18, reshape_default_20);  add_tensor_18 = reshape_default_20 = None
        add_tensor_35: "f32[4096, 16384]" = torch.ops.aten.add.Tensor(add_tensor_19, mm_41);  add_tensor_19 = mm_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:239 in ff_chunk, code: ffn_output = self.ffn(attention_output)
        sum_dim_int_list_37: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_366, [0], True);  view_366 = None
        reshape_default_21: "f32[16384]" = torch.ops.aten.reshape.default(sum_dim_int_list_37, _shape_param_21);  sum_dim_int_list_37 = _shape_param_21 = None
        add_tensor_36: "f32[16384]" = torch.ops.aten.add.Tensor(add_tensor_20, reshape_default_21);  add_tensor_20 = reshape_default_21 = None
        add_tensor_37: "f32[16384, 4096]" = torch.ops.aten.add.Tensor(add_tensor_21, mm_43);  add_tensor_21 = mm_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:202 in forward, code: attn_output = self.LayerNorm(hidden_states + attn_output)
        mul_tensor_8: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_179, mul_84);  mul_84 = None
        sum_dim_int_list_38: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_tensor_8, [0, 1]);  mul_tensor_8 = None
        sum_dim_int_list_39: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_179, [0, 1]);  add_179 = None
        add_tensor_38: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_22, sum_dim_int_list_38);  add_tensor_22 = sum_dim_int_list_38 = None
        add_tensor_39: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_23, sum_dim_int_list_39);  add_tensor_23 = sum_dim_int_list_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:200 in forward, code: attn_output = self.dense(attn_output)
        sum_dim_int_list_40: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_369, [0], True);  view_369 = None
        reshape_default_22: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_40, _shape_param_22);  sum_dim_int_list_40 = _shape_param_22 = None
        add_tensor_40: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_24, reshape_default_22);  add_tensor_24 = reshape_default_22 = None
        add_tensor_41: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_tensor_25, mm_45);  add_tensor_25 = mm_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:182 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        sum_dim_int_list_41: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_380, [0], True);  view_380 = None
        reshape_default_23: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_41, _shape_param_23);  sum_dim_int_list_41 = _shape_param_23 = None
        add_tensor_42: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_26, reshape_default_23);  add_tensor_26 = reshape_default_23 = None
        add_tensor_43: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_tensor_27, mm_47);  add_tensor_27 = mm_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:181 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        sum_dim_int_list_42: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_384, [0], True);  view_384 = None
        reshape_default_24: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_42, _shape_param_24);  sum_dim_int_list_42 = _shape_param_24 = None
        add_tensor_44: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_28, reshape_default_24);  add_tensor_28 = reshape_default_24 = None
        add_tensor_45: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_tensor_29, mm_49);  add_tensor_29 = mm_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:180 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        sum_dim_int_list_43: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_388, [0], True);  view_388 = None
        reshape_default_25: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_43, _shape_param_25);  sum_dim_int_list_43 = _shape_param_25 = None
        add_tensor_46: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_30, reshape_default_25);  add_tensor_30 = reshape_default_25 = None
        add_tensor_47: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_tensor_31, mm_51);  add_tensor_31 = mm_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:235 in forward, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output)
        mul_tensor_9: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_192, mul_80);  mul_80 = None
        sum_dim_int_list_44: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_tensor_9, [0, 1]);  mul_tensor_9 = None
        sum_dim_int_list_45: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_192, [0, 1]);  add_192 = None
        add_tensor_48: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_32, sum_dim_int_list_44);  add_tensor_32 = sum_dim_int_list_44 = None
        add_tensor_49: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_33, sum_dim_int_list_45);  add_tensor_33 = sum_dim_int_list_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:241 in ff_chunk, code: ffn_output = self.ffn_output(ffn_output)
        sum_dim_int_list_46: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_391, [0], True);  view_391 = None
        reshape_default_26: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_46, _shape_param_26);  sum_dim_int_list_46 = _shape_param_26 = None
        add_tensor_50: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_34, reshape_default_26);  add_tensor_34 = reshape_default_26 = None
        add_tensor_51: "f32[4096, 16384]" = torch.ops.aten.add.Tensor(add_tensor_35, mm_53);  add_tensor_35 = mm_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:239 in ff_chunk, code: ffn_output = self.ffn(attention_output)
        sum_dim_int_list_47: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_394, [0], True);  view_394 = None
        reshape_default_27: "f32[16384]" = torch.ops.aten.reshape.default(sum_dim_int_list_47, _shape_param_27);  sum_dim_int_list_47 = _shape_param_27 = None
        add_tensor_52: "f32[16384]" = torch.ops.aten.add.Tensor(add_tensor_36, reshape_default_27);  add_tensor_36 = reshape_default_27 = None
        add_tensor_53: "f32[16384, 4096]" = torch.ops.aten.add.Tensor(add_tensor_37, mm_55);  add_tensor_37 = mm_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:202 in forward, code: attn_output = self.LayerNorm(hidden_states + attn_output)
        mul_tensor_10: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_201, mul_74);  mul_74 = None
        sum_dim_int_list_48: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_tensor_10, [0, 1]);  mul_tensor_10 = None
        sum_dim_int_list_49: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_201, [0, 1]);  add_201 = None
        add_tensor_54: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_38, sum_dim_int_list_48);  add_tensor_38 = sum_dim_int_list_48 = None
        add_tensor_55: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_39, sum_dim_int_list_49);  add_tensor_39 = sum_dim_int_list_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:200 in forward, code: attn_output = self.dense(attn_output)
        sum_dim_int_list_50: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_397, [0], True);  view_397 = None
        reshape_default_28: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_50, _shape_param_28);  sum_dim_int_list_50 = _shape_param_28 = None
        add_tensor_56: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_40, reshape_default_28);  add_tensor_40 = reshape_default_28 = None
        add_tensor_57: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_tensor_41, mm_57);  add_tensor_41 = mm_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:182 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        sum_dim_int_list_51: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_408, [0], True);  view_408 = None
        reshape_default_29: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_51, _shape_param_29);  sum_dim_int_list_51 = _shape_param_29 = None
        add_tensor_58: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_42, reshape_default_29);  add_tensor_42 = reshape_default_29 = None
        add_tensor_59: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_tensor_43, mm_59);  add_tensor_43 = mm_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:181 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        sum_dim_int_list_52: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_412, [0], True);  view_412 = None
        reshape_default_30: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_52, _shape_param_30);  sum_dim_int_list_52 = _shape_param_30 = None
        add_tensor_60: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_44, reshape_default_30);  add_tensor_44 = reshape_default_30 = None
        add_tensor_61: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_tensor_45, mm_61);  add_tensor_45 = mm_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:180 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        sum_dim_int_list_53: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_416, [0], True);  view_416 = None
        reshape_default_31: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_53, _shape_param_31);  sum_dim_int_list_53 = _shape_param_31 = None
        add_tensor_62: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_46, reshape_default_31);  add_tensor_46 = reshape_default_31 = None
        add_tensor_63: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_tensor_47, mm_63);  add_tensor_47 = mm_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:235 in forward, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output)
        mul_tensor_11: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_214, mul_70);  mul_70 = None
        sum_dim_int_list_54: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_tensor_11, [0, 1]);  mul_tensor_11 = None
        sum_dim_int_list_55: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_214, [0, 1]);  add_214 = None
        add_tensor_64: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_48, sum_dim_int_list_54);  add_tensor_48 = sum_dim_int_list_54 = None
        add_tensor_65: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_49, sum_dim_int_list_55);  add_tensor_49 = sum_dim_int_list_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:241 in ff_chunk, code: ffn_output = self.ffn_output(ffn_output)
        sum_dim_int_list_56: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_419, [0], True);  view_419 = None
        reshape_default_32: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_56, _shape_param_32);  sum_dim_int_list_56 = _shape_param_32 = None
        add_tensor_66: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_50, reshape_default_32);  add_tensor_50 = reshape_default_32 = None
        add_tensor_67: "f32[4096, 16384]" = torch.ops.aten.add.Tensor(add_tensor_51, mm_65);  add_tensor_51 = mm_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:239 in ff_chunk, code: ffn_output = self.ffn(attention_output)
        sum_dim_int_list_57: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_422, [0], True);  view_422 = None
        reshape_default_33: "f32[16384]" = torch.ops.aten.reshape.default(sum_dim_int_list_57, _shape_param_33);  sum_dim_int_list_57 = _shape_param_33 = None
        add_tensor_68: "f32[16384]" = torch.ops.aten.add.Tensor(add_tensor_52, reshape_default_33);  add_tensor_52 = reshape_default_33 = None
        add_tensor_69: "f32[16384, 4096]" = torch.ops.aten.add.Tensor(add_tensor_53, mm_67);  add_tensor_53 = mm_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:202 in forward, code: attn_output = self.LayerNorm(hidden_states + attn_output)
        mul_tensor_12: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_223, mul_64);  mul_64 = None
        sum_dim_int_list_58: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_tensor_12, [0, 1]);  mul_tensor_12 = None
        sum_dim_int_list_59: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_223, [0, 1]);  add_223 = None
        add_tensor_70: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_54, sum_dim_int_list_58);  add_tensor_54 = sum_dim_int_list_58 = None
        add_tensor_71: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_55, sum_dim_int_list_59);  add_tensor_55 = sum_dim_int_list_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:200 in forward, code: attn_output = self.dense(attn_output)
        sum_dim_int_list_60: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_425, [0], True);  view_425 = None
        reshape_default_34: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_60, _shape_param_34);  sum_dim_int_list_60 = _shape_param_34 = None
        add_tensor_72: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_56, reshape_default_34);  add_tensor_56 = reshape_default_34 = None
        add_tensor_73: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_tensor_57, mm_69);  add_tensor_57 = mm_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:182 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        sum_dim_int_list_61: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_436, [0], True);  view_436 = None
        reshape_default_35: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_61, _shape_param_35);  sum_dim_int_list_61 = _shape_param_35 = None
        add_tensor_74: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_58, reshape_default_35);  add_tensor_58 = reshape_default_35 = None
        add_tensor_75: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_tensor_59, mm_71);  add_tensor_59 = mm_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:181 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        sum_dim_int_list_62: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_440, [0], True);  view_440 = None
        reshape_default_36: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_62, _shape_param_36);  sum_dim_int_list_62 = _shape_param_36 = None
        add_tensor_76: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_60, reshape_default_36);  add_tensor_60 = reshape_default_36 = None
        add_tensor_77: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_tensor_61, mm_73);  add_tensor_61 = mm_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:180 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        sum_dim_int_list_63: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_444, [0], True);  view_444 = None
        reshape_default_37: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_63, _shape_param_37);  sum_dim_int_list_63 = _shape_param_37 = None
        add_tensor_78: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_62, reshape_default_37);  add_tensor_62 = reshape_default_37 = None
        add_tensor_79: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_tensor_63, mm_75);  add_tensor_63 = mm_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:235 in forward, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output)
        mul_tensor_13: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_236, mul_60);  mul_60 = None
        sum_dim_int_list_64: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_tensor_13, [0, 1]);  mul_tensor_13 = None
        sum_dim_int_list_65: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_236, [0, 1]);  add_236 = None
        add_tensor_80: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_64, sum_dim_int_list_64);  add_tensor_64 = sum_dim_int_list_64 = None
        add_tensor_81: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_65, sum_dim_int_list_65);  add_tensor_65 = sum_dim_int_list_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:241 in ff_chunk, code: ffn_output = self.ffn_output(ffn_output)
        sum_dim_int_list_66: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_447, [0], True);  view_447 = None
        reshape_default_38: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_66, _shape_param_38);  sum_dim_int_list_66 = _shape_param_38 = None
        add_tensor_82: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_66, reshape_default_38);  add_tensor_66 = reshape_default_38 = None
        add_tensor_83: "f32[4096, 16384]" = torch.ops.aten.add.Tensor(add_tensor_67, mm_77);  add_tensor_67 = mm_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:239 in ff_chunk, code: ffn_output = self.ffn(attention_output)
        sum_dim_int_list_67: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_450, [0], True);  view_450 = None
        reshape_default_39: "f32[16384]" = torch.ops.aten.reshape.default(sum_dim_int_list_67, _shape_param_39);  sum_dim_int_list_67 = _shape_param_39 = None
        add_tensor_84: "f32[16384]" = torch.ops.aten.add.Tensor(add_tensor_68, reshape_default_39);  add_tensor_68 = reshape_default_39 = None
        add_tensor_85: "f32[16384, 4096]" = torch.ops.aten.add.Tensor(add_tensor_69, mm_79);  add_tensor_69 = mm_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:202 in forward, code: attn_output = self.LayerNorm(hidden_states + attn_output)
        mul_tensor_14: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_245, mul_54);  mul_54 = None
        sum_dim_int_list_68: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_tensor_14, [0, 1]);  mul_tensor_14 = None
        sum_dim_int_list_69: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_245, [0, 1]);  add_245 = None
        add_tensor_86: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_70, sum_dim_int_list_68);  add_tensor_70 = sum_dim_int_list_68 = None
        add_tensor_87: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_71, sum_dim_int_list_69);  add_tensor_71 = sum_dim_int_list_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:200 in forward, code: attn_output = self.dense(attn_output)
        sum_dim_int_list_70: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_453, [0], True);  view_453 = None
        reshape_default_40: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_70, _shape_param_40);  sum_dim_int_list_70 = _shape_param_40 = None
        add_tensor_88: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_72, reshape_default_40);  add_tensor_72 = reshape_default_40 = None
        add_tensor_89: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_tensor_73, mm_81);  add_tensor_73 = mm_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:182 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        sum_dim_int_list_71: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_464, [0], True);  view_464 = None
        reshape_default_41: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_71, _shape_param_41);  sum_dim_int_list_71 = _shape_param_41 = None
        add_tensor_90: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_74, reshape_default_41);  add_tensor_74 = reshape_default_41 = None
        add_tensor_91: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_tensor_75, mm_83);  add_tensor_75 = mm_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:181 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        sum_dim_int_list_72: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_468, [0], True);  view_468 = None
        reshape_default_42: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_72, _shape_param_42);  sum_dim_int_list_72 = _shape_param_42 = None
        add_tensor_92: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_76, reshape_default_42);  add_tensor_76 = reshape_default_42 = None
        add_tensor_93: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_tensor_77, mm_85);  add_tensor_77 = mm_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:180 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        sum_dim_int_list_73: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_472, [0], True);  view_472 = None
        reshape_default_43: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_73, _shape_param_43);  sum_dim_int_list_73 = _shape_param_43 = None
        add_tensor_94: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_78, reshape_default_43);  add_tensor_78 = reshape_default_43 = None
        add_tensor_95: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_tensor_79, mm_87);  add_tensor_79 = mm_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:235 in forward, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output)
        mul_tensor_15: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_258, mul_50);  mul_50 = None
        sum_dim_int_list_74: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_tensor_15, [0, 1]);  mul_tensor_15 = None
        sum_dim_int_list_75: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_258, [0, 1]);  add_258 = None
        add_tensor_96: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_80, sum_dim_int_list_74);  add_tensor_80 = sum_dim_int_list_74 = None
        add_tensor_97: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_81, sum_dim_int_list_75);  add_tensor_81 = sum_dim_int_list_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:241 in ff_chunk, code: ffn_output = self.ffn_output(ffn_output)
        sum_dim_int_list_76: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_475, [0], True);  view_475 = None
        reshape_default_44: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_76, _shape_param_44);  sum_dim_int_list_76 = _shape_param_44 = None
        add_tensor_98: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_82, reshape_default_44);  add_tensor_82 = reshape_default_44 = None
        add_tensor_99: "f32[4096, 16384]" = torch.ops.aten.add.Tensor(add_tensor_83, mm_89);  add_tensor_83 = mm_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:239 in ff_chunk, code: ffn_output = self.ffn(attention_output)
        sum_dim_int_list_77: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_478, [0], True);  view_478 = None
        reshape_default_45: "f32[16384]" = torch.ops.aten.reshape.default(sum_dim_int_list_77, _shape_param_45);  sum_dim_int_list_77 = _shape_param_45 = None
        add_tensor_100: "f32[16384]" = torch.ops.aten.add.Tensor(add_tensor_84, reshape_default_45);  add_tensor_84 = reshape_default_45 = None
        add_tensor_101: "f32[16384, 4096]" = torch.ops.aten.add.Tensor(add_tensor_85, mm_91);  add_tensor_85 = mm_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:202 in forward, code: attn_output = self.LayerNorm(hidden_states + attn_output)
        mul_tensor_16: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_267, mul_44);  mul_44 = None
        sum_dim_int_list_78: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_tensor_16, [0, 1]);  mul_tensor_16 = None
        sum_dim_int_list_79: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_267, [0, 1]);  add_267 = None
        add_tensor_102: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_86, sum_dim_int_list_78);  add_tensor_86 = sum_dim_int_list_78 = None
        add_tensor_103: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_87, sum_dim_int_list_79);  add_tensor_87 = sum_dim_int_list_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:200 in forward, code: attn_output = self.dense(attn_output)
        sum_dim_int_list_80: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_481, [0], True);  view_481 = None
        reshape_default_46: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_80, _shape_param_46);  sum_dim_int_list_80 = _shape_param_46 = None
        add_tensor_104: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_88, reshape_default_46);  add_tensor_88 = reshape_default_46 = None
        add_tensor_105: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_tensor_89, mm_93);  add_tensor_89 = mm_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:182 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        sum_dim_int_list_81: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_492, [0], True);  view_492 = None
        reshape_default_47: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_81, _shape_param_47);  sum_dim_int_list_81 = _shape_param_47 = None
        add_tensor_106: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_90, reshape_default_47);  add_tensor_90 = reshape_default_47 = None
        add_tensor_107: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_tensor_91, mm_95);  add_tensor_91 = mm_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:181 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        sum_dim_int_list_82: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_496, [0], True);  view_496 = None
        reshape_default_48: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_82, _shape_param_48);  sum_dim_int_list_82 = _shape_param_48 = None
        add_tensor_108: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_92, reshape_default_48);  add_tensor_92 = reshape_default_48 = None
        add_tensor_109: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_tensor_93, mm_97);  add_tensor_93 = mm_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:180 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        sum_dim_int_list_83: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_500, [0], True);  view_500 = None
        reshape_default_49: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_83, _shape_param_49);  sum_dim_int_list_83 = _shape_param_49 = None
        add_tensor_110: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_94, reshape_default_49);  add_tensor_94 = reshape_default_49 = None
        add_tensor_111: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_tensor_95, mm_99);  add_tensor_95 = mm_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:235 in forward, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output)
        mul_tensor_17: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_280, mul_40);  mul_40 = None
        sum_dim_int_list_84: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_tensor_17, [0, 1]);  mul_tensor_17 = None
        sum_dim_int_list_85: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_280, [0, 1]);  add_280 = None
        add_tensor_112: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_96, sum_dim_int_list_84);  add_tensor_96 = sum_dim_int_list_84 = None
        add_tensor_113: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_97, sum_dim_int_list_85);  add_tensor_97 = sum_dim_int_list_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:241 in ff_chunk, code: ffn_output = self.ffn_output(ffn_output)
        sum_dim_int_list_86: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_503, [0], True);  view_503 = None
        reshape_default_50: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_86, _shape_param_50);  sum_dim_int_list_86 = _shape_param_50 = None
        add_tensor_114: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_98, reshape_default_50);  add_tensor_98 = reshape_default_50 = None
        add_tensor_115: "f32[4096, 16384]" = torch.ops.aten.add.Tensor(add_tensor_99, mm_101);  add_tensor_99 = mm_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:239 in ff_chunk, code: ffn_output = self.ffn(attention_output)
        sum_dim_int_list_87: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_506, [0], True);  view_506 = None
        reshape_default_51: "f32[16384]" = torch.ops.aten.reshape.default(sum_dim_int_list_87, _shape_param_51);  sum_dim_int_list_87 = _shape_param_51 = None
        add_tensor_116: "f32[16384]" = torch.ops.aten.add.Tensor(add_tensor_100, reshape_default_51);  add_tensor_100 = reshape_default_51 = None
        add_tensor_117: "f32[16384, 4096]" = torch.ops.aten.add.Tensor(add_tensor_101, mm_103);  add_tensor_101 = mm_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:202 in forward, code: attn_output = self.LayerNorm(hidden_states + attn_output)
        mul_tensor_18: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_289, mul_34);  mul_34 = None
        sum_dim_int_list_88: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_tensor_18, [0, 1]);  mul_tensor_18 = None
        sum_dim_int_list_89: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_289, [0, 1]);  add_289 = None
        add_tensor_118: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_102, sum_dim_int_list_88);  add_tensor_102 = sum_dim_int_list_88 = None
        add_tensor_119: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_103, sum_dim_int_list_89);  add_tensor_103 = sum_dim_int_list_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:200 in forward, code: attn_output = self.dense(attn_output)
        sum_dim_int_list_90: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_509, [0], True);  view_509 = None
        reshape_default_52: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_90, _shape_param_52);  sum_dim_int_list_90 = _shape_param_52 = None
        add_tensor_120: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_104, reshape_default_52);  add_tensor_104 = reshape_default_52 = None
        add_tensor_121: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_tensor_105, mm_105);  add_tensor_105 = mm_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:182 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        sum_dim_int_list_91: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_520, [0], True);  view_520 = None
        reshape_default_53: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_91, _shape_param_53);  sum_dim_int_list_91 = _shape_param_53 = None
        add_tensor_122: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_106, reshape_default_53);  add_tensor_106 = reshape_default_53 = None
        add_tensor_123: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_tensor_107, mm_107);  add_tensor_107 = mm_107 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:181 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        sum_dim_int_list_92: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_524, [0], True);  view_524 = None
        reshape_default_54: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_92, _shape_param_54);  sum_dim_int_list_92 = _shape_param_54 = None
        add_tensor_124: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_108, reshape_default_54);  add_tensor_108 = reshape_default_54 = None
        add_tensor_125: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_tensor_109, mm_109);  add_tensor_109 = mm_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:180 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        sum_dim_int_list_93: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_528, [0], True);  view_528 = None
        reshape_default_55: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_93, _shape_param_55);  sum_dim_int_list_93 = _shape_param_55 = None
        add_tensor_126: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_110, reshape_default_55);  add_tensor_110 = reshape_default_55 = None
        add_tensor_127: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_tensor_111, mm_111);  add_tensor_111 = mm_111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:235 in forward, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output)
        mul_tensor_19: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_302, mul_30);  mul_30 = None
        sum_dim_int_list_94: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_tensor_19, [0, 1]);  mul_tensor_19 = None
        sum_dim_int_list_95: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_302, [0, 1]);  add_302 = None
        add_tensor_128: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_112, sum_dim_int_list_94);  add_tensor_112 = sum_dim_int_list_94 = None
        add_tensor_129: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_113, sum_dim_int_list_95);  add_tensor_113 = sum_dim_int_list_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:241 in ff_chunk, code: ffn_output = self.ffn_output(ffn_output)
        sum_dim_int_list_96: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_531, [0], True);  view_531 = None
        reshape_default_56: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_96, _shape_param_56);  sum_dim_int_list_96 = _shape_param_56 = None
        add_tensor_130: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_114, reshape_default_56);  add_tensor_114 = reshape_default_56 = None
        add_tensor_131: "f32[4096, 16384]" = torch.ops.aten.add.Tensor(add_tensor_115, mm_113);  add_tensor_115 = mm_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:239 in ff_chunk, code: ffn_output = self.ffn(attention_output)
        sum_dim_int_list_97: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_534, [0], True);  view_534 = None
        reshape_default_57: "f32[16384]" = torch.ops.aten.reshape.default(sum_dim_int_list_97, _shape_param_57);  sum_dim_int_list_97 = _shape_param_57 = None
        add_tensor_132: "f32[16384]" = torch.ops.aten.add.Tensor(add_tensor_116, reshape_default_57);  add_tensor_116 = reshape_default_57 = None
        add_tensor_133: "f32[16384, 4096]" = torch.ops.aten.add.Tensor(add_tensor_117, mm_115);  add_tensor_117 = mm_115 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:202 in forward, code: attn_output = self.LayerNorm(hidden_states + attn_output)
        mul_tensor_20: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_311, mul_24);  mul_24 = None
        sum_dim_int_list_98: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_tensor_20, [0, 1]);  mul_tensor_20 = None
        sum_dim_int_list_99: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_311, [0, 1]);  add_311 = None
        add_tensor_134: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_118, sum_dim_int_list_98);  add_tensor_118 = sum_dim_int_list_98 = None
        add_tensor_135: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_119, sum_dim_int_list_99);  add_tensor_119 = sum_dim_int_list_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:200 in forward, code: attn_output = self.dense(attn_output)
        sum_dim_int_list_100: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_537, [0], True);  view_537 = None
        reshape_default_58: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_100, _shape_param_58);  sum_dim_int_list_100 = _shape_param_58 = None
        add_tensor_136: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_120, reshape_default_58);  add_tensor_120 = reshape_default_58 = None
        add_tensor_137: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_tensor_121, mm_117);  add_tensor_121 = mm_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:182 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        sum_dim_int_list_101: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_548, [0], True);  view_548 = None
        reshape_default_59: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_101, _shape_param_59);  sum_dim_int_list_101 = _shape_param_59 = None
        add_tensor_138: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_122, reshape_default_59);  add_tensor_122 = reshape_default_59 = None
        add_tensor_139: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_tensor_123, mm_119);  add_tensor_123 = mm_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:181 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        sum_dim_int_list_102: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_552, [0], True);  view_552 = None
        reshape_default_60: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_102, _shape_param_60);  sum_dim_int_list_102 = _shape_param_60 = None
        add_tensor_140: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_124, reshape_default_60);  add_tensor_124 = reshape_default_60 = None
        add_tensor_141: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_tensor_125, mm_121);  add_tensor_125 = mm_121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:180 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        sum_dim_int_list_103: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_556, [0], True);  view_556 = None
        reshape_default_61: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_103, _shape_param_61);  sum_dim_int_list_103 = _shape_param_61 = None
        add_tensor_142: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_126, reshape_default_61);  add_tensor_126 = reshape_default_61 = None
        add_tensor_143: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_tensor_127, mm_123);  add_tensor_127 = mm_123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:235 in forward, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output)
        mul_tensor_21: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_324, mul_20);  mul_20 = None
        sum_dim_int_list_104: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_tensor_21, [0, 1]);  mul_tensor_21 = None
        sum_dim_int_list_105: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_324, [0, 1]);  add_324 = None
        add_tensor_144: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_128, sum_dim_int_list_104);  add_tensor_128 = sum_dim_int_list_104 = None
        add_tensor_145: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_129, sum_dim_int_list_105);  add_tensor_129 = sum_dim_int_list_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:241 in ff_chunk, code: ffn_output = self.ffn_output(ffn_output)
        sum_dim_int_list_106: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_559, [0], True);  view_559 = None
        reshape_default_62: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_106, _shape_param_62);  sum_dim_int_list_106 = _shape_param_62 = None
        add_tensor_146: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_130, reshape_default_62);  add_tensor_130 = reshape_default_62 = None
        add_tensor_147: "f32[4096, 16384]" = torch.ops.aten.add.Tensor(add_tensor_131, mm_125);  add_tensor_131 = mm_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:239 in ff_chunk, code: ffn_output = self.ffn(attention_output)
        sum_dim_int_list_107: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_562, [0], True);  view_562 = None
        reshape_default_63: "f32[16384]" = torch.ops.aten.reshape.default(sum_dim_int_list_107, _shape_param_63);  sum_dim_int_list_107 = _shape_param_63 = None
        add_tensor_148: "f32[16384]" = torch.ops.aten.add.Tensor(add_tensor_132, reshape_default_63);  add_tensor_132 = reshape_default_63 = None
        add_tensor_149: "f32[16384, 4096]" = torch.ops.aten.add.Tensor(add_tensor_133, mm_127);  add_tensor_133 = mm_127 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:202 in forward, code: attn_output = self.LayerNorm(hidden_states + attn_output)
        mul_tensor_22: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_333, mul_14);  mul_14 = None
        sum_dim_int_list_108: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_tensor_22, [0, 1]);  mul_tensor_22 = None
        sum_dim_int_list_109: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_333, [0, 1]);  add_333 = None
        add_tensor_150: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_134, sum_dim_int_list_108);  add_tensor_134 = sum_dim_int_list_108 = None
        add_tensor_151: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_135, sum_dim_int_list_109);  add_tensor_135 = sum_dim_int_list_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:200 in forward, code: attn_output = self.dense(attn_output)
        sum_dim_int_list_110: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_565, [0], True);  view_565 = None
        reshape_default_64: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_110, _shape_param_64);  sum_dim_int_list_110 = _shape_param_64 = None
        add_tensor_152: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_136, reshape_default_64);  add_tensor_136 = reshape_default_64 = None
        add_tensor_153: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_tensor_137, mm_129);  add_tensor_137 = mm_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:182 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        sum_dim_int_list_111: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_576, [0], True);  view_576 = None
        reshape_default_65: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_111, _shape_param_65);  sum_dim_int_list_111 = _shape_param_65 = None
        add_tensor_154: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_138, reshape_default_65);  add_tensor_138 = reshape_default_65 = None
        add_tensor_155: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_tensor_139, mm_131);  add_tensor_139 = mm_131 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:181 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        sum_dim_int_list_112: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_580, [0], True);  view_580 = None
        reshape_default_66: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_112, _shape_param_66);  sum_dim_int_list_112 = _shape_param_66 = None
        add_tensor_156: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_140, reshape_default_66);  add_tensor_140 = reshape_default_66 = None
        add_tensor_157: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_tensor_141, mm_133);  add_tensor_141 = mm_133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:180 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        sum_dim_int_list_113: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_584, [0], True);  view_584 = None
        reshape_default_67: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_113, _shape_param_67);  sum_dim_int_list_113 = _shape_param_67 = None
        add_tensor_158: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_142, reshape_default_67);  add_tensor_142 = reshape_default_67 = None
        add_tensor_159: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_tensor_143, mm_135);  add_tensor_143 = mm_135 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:235 in forward, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output)
        mul_tensor_23: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_346, mul_10);  mul_10 = None
        sum_dim_int_list_114: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_tensor_23, [0, 1]);  mul_tensor_23 = None
        sum_dim_int_list_115: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_346, [0, 1]);  add_346 = None
        add_tensor_160: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_144, sum_dim_int_list_114);  add_tensor_144 = sum_dim_int_list_114 = None
        add_tensor_161: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_145, sum_dim_int_list_115);  add_tensor_145 = sum_dim_int_list_115 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:241 in ff_chunk, code: ffn_output = self.ffn_output(ffn_output)
        sum_dim_int_list_116: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_587, [0], True);  view_587 = None
        reshape_default_68: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_116, _shape_param_68);  sum_dim_int_list_116 = _shape_param_68 = None
        add_tensor_162: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_146, reshape_default_68);  add_tensor_146 = reshape_default_68 = None
        add_tensor_163: "f32[4096, 16384]" = torch.ops.aten.add.Tensor(add_tensor_147, mm_137);  add_tensor_147 = mm_137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:239 in ff_chunk, code: ffn_output = self.ffn(attention_output)
        sum_dim_int_list_117: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_590, [0], True);  view_590 = None
        reshape_default_69: "f32[16384]" = torch.ops.aten.reshape.default(sum_dim_int_list_117, _shape_param_69);  sum_dim_int_list_117 = _shape_param_69 = None
        add_tensor_164: "f32[16384]" = torch.ops.aten.add.Tensor(add_tensor_148, reshape_default_69);  add_tensor_148 = reshape_default_69 = None
        add_tensor_165: "f32[16384, 4096]" = torch.ops.aten.add.Tensor(add_tensor_149, mm_139);  add_tensor_149 = mm_139 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:202 in forward, code: attn_output = self.LayerNorm(hidden_states + attn_output)
        mul_tensor_24: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_355, mul_4);  mul_4 = None
        sum_dim_int_list_118: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_tensor_24, [0, 1]);  mul_tensor_24 = None
        sum_dim_int_list_119: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_355, [0, 1]);  add_355 = None
        add_tensor_166: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_150, sum_dim_int_list_118);  add_tensor_150 = sum_dim_int_list_118 = None
        add_tensor_167: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_151, sum_dim_int_list_119);  add_tensor_151 = sum_dim_int_list_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:200 in forward, code: attn_output = self.dense(attn_output)
        sum_dim_int_list_120: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_593, [0], True);  view_593 = None
        reshape_default_70: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_120, _shape_param_70);  sum_dim_int_list_120 = _shape_param_70 = None
        add_tensor_168: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_152, reshape_default_70);  add_tensor_152 = reshape_default_70 = None
        add_tensor_169: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_tensor_153, mm_141);  add_tensor_153 = mm_141 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:182 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        sum_dim_int_list_121: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_604, [0], True);  view_604 = None
        reshape_default_71: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_121, _shape_param_71);  sum_dim_int_list_121 = _shape_param_71 = None
        add_tensor_170: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_154, reshape_default_71);  add_tensor_154 = reshape_default_71 = None
        add_tensor_171: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_tensor_155, mm_143);  add_tensor_155 = mm_143 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:181 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        sum_dim_int_list_122: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_608, [0], True);  view_608 = None
        reshape_default_72: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_122, _shape_param_72);  sum_dim_int_list_122 = _shape_param_72 = None
        add_tensor_172: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_156, reshape_default_72);  add_tensor_156 = reshape_default_72 = None
        add_tensor_173: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_tensor_157, mm_145);  add_tensor_157 = mm_145 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:180 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        sum_dim_int_list_123: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_612, [0], True);  view_612 = None
        reshape_default_73: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_123, _shape_param_73);  sum_dim_int_list_123 = _shape_param_73 = None
        add_tensor_174: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_158, reshape_default_73);  add_tensor_158 = reshape_default_73 = None
        add_tensor_175: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_tensor_159, mm_147);  add_tensor_159 = mm_147 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:276 in forward, code: hidden_states = self.embedding_hidden_mapping_in(hidden_states)
        permute_default_1: "f32[4096, 4096]" = torch.ops.aten.permute.default(view_615, [1, 0])
        sum_dim_int_list_124: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_615, [0], True);  view_615 = None
        reshape_default_74: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_124, _shape_param_74);  sum_dim_int_list_124 = _shape_param_74 = None
        reshape_default_75: "f32[8, 512, 128]" = torch.ops.aten.reshape.default(mm_148, _shape_param_75);  mm_148 = _shape_param_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:108 in forward, code: embeddings = self.LayerNorm(embeddings)
        mul_tensor_25: "f32[8, 512, 128]" = torch.ops.aten.mul.Tensor(reshape_default_75, primals_7);  primals_7 = None
        mul_tensor_26: "f32[8, 512, 128]" = torch.ops.aten.mul.Tensor(mul_tensor_25, 128)
        sum_dim_int_list_125: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_25, [2], True)
        mul_tensor_27: "f32[8, 512, 128]" = torch.ops.aten.mul.Tensor(mul_tensor_25, mul);  mul_tensor_25 = None
        sum_dim_int_list_126: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_27, [2], True);  mul_tensor_27 = None
        mul_tensor_28: "f32[8, 512, 128]" = torch.ops.aten.mul.Tensor(mul, sum_dim_int_list_126);  sum_dim_int_list_126 = None
        sub_tensor: "f32[8, 512, 128]" = torch.ops.aten.sub.Tensor(mul_tensor_26, sum_dim_int_list_125);  mul_tensor_26 = sum_dim_int_list_125 = None
        sub_tensor_1: "f32[8, 512, 128]" = torch.ops.aten.sub.Tensor(sub_tensor, mul_tensor_28);  sub_tensor = mul_tensor_28 = None
        mul_tensor_29: "f32[8, 512, 128]" = torch.ops.aten.mul.Tensor(div_39, sub_tensor_1);  div_39 = sub_tensor_1 = None
        mul_tensor_30: "f32[8, 512, 128]" = torch.ops.aten.mul.Tensor(reshape_default_75, mul);  mul = None
        sum_dim_int_list_127: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_tensor_30, [0, 1]);  mul_tensor_30 = None
        sum_dim_int_list_128: "f32[128]" = torch.ops.aten.sum.dim_IntList(reshape_default_75, [0, 1]);  reshape_default_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:106 in forward, code: embeddings = embeddings + position_embeddings
        sum_dim_int_list_129: "f32[1, 512, 128]" = torch.ops.aten.sum.dim_IntList(mul_tensor_29, [0], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:105 in forward, code: position_embeddings = self.position_embeddings(position_ids)
        eq_scalar: "b8[1, 512]" = torch.ops.aten.eq.Scalar(primals_2, -1)
        unsqueeze_default: "b8[1, 512, 1]" = torch.ops.aten.unsqueeze.default(eq_scalar, -1);  eq_scalar = None
        where_self: "f32[1, 512, 128]" = torch.ops.aten.where.self(unsqueeze_default, full_default_1, sum_dim_int_list_129);  unsqueeze_default = sum_dim_int_list_129 = None
        full_default: "f32[512, 128]" = torch.ops.aten.full.default([512, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default: "f32[512, 128]" = torch.ops.aten.index_put.default(full_default, [primals_2], where_self, True);  full_default = primals_2 = where_self = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:96 in forward, code: token_type_ids = buffered_token_type_ids.expand(batch_size, seq_length)
        expand_default: "i64[8, 512]" = torch.ops.aten.expand.default(gather, _shape_param_76);  gather = _shape_param_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:102 in forward, code: token_type_embeddings = self.token_type_embeddings(token_type_ids)
        eq_scalar_1: "b8[8, 512]" = torch.ops.aten.eq.Scalar(expand_default, -1)
        unsqueeze_default_1: "b8[8, 512, 1]" = torch.ops.aten.unsqueeze.default(eq_scalar_1, -1);  eq_scalar_1 = None
        where_self_1: "f32[8, 512, 128]" = torch.ops.aten.where.self(unsqueeze_default_1, full_default_1, mul_tensor_29);  unsqueeze_default_1 = None
        full_default_2: "f32[2, 128]" = torch.ops.aten.full.default([2, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default_1: "f32[2, 128]" = torch.ops.aten.index_put.default(full_default_2, [expand_default], where_self_1, True);  full_default_2 = expand_default = where_self_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:101 in forward, code: inputs_embeds = self.word_embeddings(input_ids)
        eq_scalar_2: "b8[8, 512]" = torch.ops.aten.eq.Scalar(primals_1, 0)
        unsqueeze_default_2: "b8[8, 512, 1]" = torch.ops.aten.unsqueeze.default(eq_scalar_2, -1);  eq_scalar_2 = None
        where_self_2: "f32[8, 512, 128]" = torch.ops.aten.where.self(unsqueeze_default_2, full_default_1, mul_tensor_29);  unsqueeze_default_2 = full_default_1 = mul_tensor_29 = None
        full_default_3: "f32[30000, 128]" = torch.ops.aten.full.default([30000, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default_2: "f32[30000, 128]" = torch.ops.aten.index_put.default(full_default_3, [primals_1], where_self_2, True);  full_default_3 = primals_1 = where_self_2 = None
        add_tensor_176: "f32[30000, 128]" = torch.ops.aten.add.Tensor(mm_1, index_put_default_2);  mm_1 = index_put_default_2 = None
        return (reshape_default, sum_dim_int_list_1, sum_dim_int_list_2, permute_default, reshape_default_1, add_tensor_160, add_tensor_161, add_tensor_162, add_tensor_163, add_tensor_164, add_tensor_165, add_tensor_166, add_tensor_167, add_tensor_168, add_tensor_169, add_tensor_170, add_tensor_171, add_tensor_172, add_tensor_173, add_tensor_174, add_tensor_175, permute_default_1, reshape_default_74, sum_dim_int_list_127, sum_dim_int_list_128, index_put_default, index_put_default_1, add_tensor_176)


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
