"""
Standalone repro captured via capture_hook.
Label: tlparse_huggingface_s4_g35
Pattern hash: 5da1b775f6b5
Shape hash: 355084fe
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, convert_element_type_4: "f16[4096, 32128]", view_4: "f16[4096, 512]", view_6: "f16[4096, 2048]", view_9: "f16[4096, 512]", view_19: "f16[4096, 512]", view_22: "f16[4096, 512]", view_25: "f16[4096, 512]", view_28: "f16[4096, 512]", view_40: "f16[4096, 512]", view_43: "f16[4096, 512]", view_46: "f16[4096, 512]", view_49: "f16[4096, 512]", view_51: "f16[4096, 2048]", view_54: "f16[4096, 512]", view_64: "f16[4096, 512]", view_67: "f16[4096, 512]", view_70: "f16[4096, 512]", view_73: "f16[4096, 512]", view_85: "f16[4096, 512]", view_88: "f16[4096, 512]", view_91: "f16[4096, 512]", view_94: "f16[4096, 512]", view_96: "f16[4096, 2048]", view_99: "f16[4096, 512]", view_109: "f16[4096, 512]", view_112: "f16[4096, 512]", view_115: "f16[4096, 512]", view_118: "f16[4096, 512]", view_130: "f16[4096, 512]", view_133: "f16[4096, 512]", view_136: "f16[4096, 512]", view_139: "f16[4096, 512]", view_141: "f16[4096, 2048]", view_144: "f16[4096, 512]", view_154: "f16[4096, 512]", view_157: "f16[4096, 512]", view_160: "f16[4096, 512]", view_163: "f16[4096, 512]", view_175: "f16[4096, 512]", view_178: "f16[4096, 512]", view_181: "f16[4096, 512]", view_184: "f16[4096, 512]", view_186: "f16[4096, 2048]", view_189: "f16[4096, 512]", view_199: "f16[4096, 512]", view_202: "f16[4096, 512]", view_205: "f16[4096, 512]", view_208: "f16[4096, 512]", view_220: "f16[4096, 512]", view_223: "f16[4096, 512]", view_226: "f16[4096, 512]", view_229: "f16[4096, 512]", view_231: "f16[4096, 2048]", view_234: "f16[4096, 512]", view_244: "f16[4096, 512]", view_247: "f16[4096, 512]", view_250: "f16[4096, 512]", view_253: "f16[4096, 512]", bmm_44: "f16[32, 1024, 64]", bmm_46: "f16[32, 64, 1024]", bmm_47: "f16[32, 1024, 64]", view_276: "f16[4096, 512]", view_278: "f16[4096, 2048]", view_281: "f16[4096, 512]", view_293: "f16[4096, 512]", view_296: "f16[4096, 512]", view_299: "f16[4096, 512]", view_302: "f16[4096, 512]", view_304: "f16[4096, 2048]", view_307: "f16[4096, 512]", view_319: "f16[4096, 512]", view_322: "f16[4096, 512]", view_325: "f16[4096, 512]", view_328: "f16[4096, 512]", view_330: "f16[4096, 2048]", view_333: "f16[4096, 512]", view_345: "f16[4096, 512]", view_348: "f16[4096, 512]", view_351: "f16[4096, 512]", view_354: "f16[4096, 512]", view_356: "f16[4096, 2048]", view_359: "f16[4096, 512]", view_371: "f16[4096, 512]", view_374: "f16[4096, 512]", view_377: "f16[4096, 512]", view_380: "f16[4096, 512]", view_382: "f16[4096, 2048]", view_385: "f16[4096, 512]", view_397: "f16[4096, 512]", view_400: "f16[4096, 512]", view_403: "f16[4096, 512]", view_406: "f16[4096, 512]", view_408: "f16[4096, 2048]", view_411: "f16[4096, 512]", bmm_68: "f16[32, 1024, 64]", bmm_70: "f16[32, 64, 1024]", bmm_71: "f16[32, 1024, 64]"):
        # No stacktrace found for following nodes
        permute_default: "f16[32128, 4096]" = torch.ops.aten.permute.default(convert_element_type_4, [1, 0]);  convert_element_type_4 = None
        permute_default_1: "f16[512, 4096]" = torch.ops.aten.permute.default(view_4, [1, 0]);  view_4 = None
        permute_default_2: "f16[2048, 4096]" = torch.ops.aten.permute.default(view_6, [1, 0]);  view_6 = None
        permute_default_3: "f16[512, 4096]" = torch.ops.aten.permute.default(view_9, [1, 0]);  view_9 = None
        permute_default_4: "f16[512, 4096]" = torch.ops.aten.permute.default(view_19, [1, 0]);  view_19 = None
        permute_default_5: "f16[512, 4096]" = torch.ops.aten.permute.default(view_22, [1, 0]);  view_22 = None
        permute_default_6: "f16[512, 4096]" = torch.ops.aten.permute.default(view_25, [1, 0]);  view_25 = None
        permute_default_7: "f16[512, 4096]" = torch.ops.aten.permute.default(view_28, [1, 0]);  view_28 = None
        permute_default_8: "f16[512, 4096]" = torch.ops.aten.permute.default(view_40, [1, 0]);  view_40 = None
        permute_default_9: "f16[512, 4096]" = torch.ops.aten.permute.default(view_43, [1, 0]);  view_43 = None
        permute_default_10: "f16[512, 4096]" = torch.ops.aten.permute.default(view_46, [1, 0]);  view_46 = None
        permute_default_11: "f16[512, 4096]" = torch.ops.aten.permute.default(view_49, [1, 0]);  view_49 = None
        permute_default_12: "f16[2048, 4096]" = torch.ops.aten.permute.default(view_51, [1, 0]);  view_51 = None
        permute_default_13: "f16[512, 4096]" = torch.ops.aten.permute.default(view_54, [1, 0]);  view_54 = None
        permute_default_14: "f16[512, 4096]" = torch.ops.aten.permute.default(view_64, [1, 0]);  view_64 = None
        permute_default_15: "f16[512, 4096]" = torch.ops.aten.permute.default(view_67, [1, 0]);  view_67 = None
        permute_default_16: "f16[512, 4096]" = torch.ops.aten.permute.default(view_70, [1, 0]);  view_70 = None
        permute_default_17: "f16[512, 4096]" = torch.ops.aten.permute.default(view_73, [1, 0]);  view_73 = None
        permute_default_18: "f16[512, 4096]" = torch.ops.aten.permute.default(view_85, [1, 0]);  view_85 = None
        permute_default_19: "f16[512, 4096]" = torch.ops.aten.permute.default(view_88, [1, 0]);  view_88 = None
        permute_default_20: "f16[512, 4096]" = torch.ops.aten.permute.default(view_91, [1, 0]);  view_91 = None
        permute_default_21: "f16[512, 4096]" = torch.ops.aten.permute.default(view_94, [1, 0]);  view_94 = None
        permute_default_22: "f16[2048, 4096]" = torch.ops.aten.permute.default(view_96, [1, 0]);  view_96 = None
        permute_default_23: "f16[512, 4096]" = torch.ops.aten.permute.default(view_99, [1, 0]);  view_99 = None
        permute_default_24: "f16[512, 4096]" = torch.ops.aten.permute.default(view_109, [1, 0]);  view_109 = None
        permute_default_25: "f16[512, 4096]" = torch.ops.aten.permute.default(view_112, [1, 0]);  view_112 = None
        permute_default_26: "f16[512, 4096]" = torch.ops.aten.permute.default(view_115, [1, 0]);  view_115 = None
        permute_default_27: "f16[512, 4096]" = torch.ops.aten.permute.default(view_118, [1, 0]);  view_118 = None
        permute_default_28: "f16[512, 4096]" = torch.ops.aten.permute.default(view_130, [1, 0]);  view_130 = None
        permute_default_29: "f16[512, 4096]" = torch.ops.aten.permute.default(view_133, [1, 0]);  view_133 = None
        permute_default_30: "f16[512, 4096]" = torch.ops.aten.permute.default(view_136, [1, 0]);  view_136 = None
        permute_default_31: "f16[512, 4096]" = torch.ops.aten.permute.default(view_139, [1, 0]);  view_139 = None
        permute_default_32: "f16[2048, 4096]" = torch.ops.aten.permute.default(view_141, [1, 0]);  view_141 = None
        permute_default_33: "f16[512, 4096]" = torch.ops.aten.permute.default(view_144, [1, 0]);  view_144 = None
        permute_default_34: "f16[512, 4096]" = torch.ops.aten.permute.default(view_154, [1, 0]);  view_154 = None
        permute_default_35: "f16[512, 4096]" = torch.ops.aten.permute.default(view_157, [1, 0]);  view_157 = None
        permute_default_36: "f16[512, 4096]" = torch.ops.aten.permute.default(view_160, [1, 0]);  view_160 = None
        permute_default_37: "f16[512, 4096]" = torch.ops.aten.permute.default(view_163, [1, 0]);  view_163 = None
        permute_default_38: "f16[512, 4096]" = torch.ops.aten.permute.default(view_175, [1, 0]);  view_175 = None
        permute_default_39: "f16[512, 4096]" = torch.ops.aten.permute.default(view_178, [1, 0]);  view_178 = None
        permute_default_40: "f16[512, 4096]" = torch.ops.aten.permute.default(view_181, [1, 0]);  view_181 = None
        permute_default_41: "f16[512, 4096]" = torch.ops.aten.permute.default(view_184, [1, 0]);  view_184 = None
        permute_default_42: "f16[2048, 4096]" = torch.ops.aten.permute.default(view_186, [1, 0]);  view_186 = None
        permute_default_43: "f16[512, 4096]" = torch.ops.aten.permute.default(view_189, [1, 0]);  view_189 = None
        permute_default_44: "f16[512, 4096]" = torch.ops.aten.permute.default(view_199, [1, 0]);  view_199 = None
        permute_default_45: "f16[512, 4096]" = torch.ops.aten.permute.default(view_202, [1, 0]);  view_202 = None
        permute_default_46: "f16[512, 4096]" = torch.ops.aten.permute.default(view_205, [1, 0]);  view_205 = None
        permute_default_47: "f16[512, 4096]" = torch.ops.aten.permute.default(view_208, [1, 0]);  view_208 = None
        permute_default_48: "f16[512, 4096]" = torch.ops.aten.permute.default(view_220, [1, 0]);  view_220 = None
        permute_default_49: "f16[512, 4096]" = torch.ops.aten.permute.default(view_223, [1, 0]);  view_223 = None
        permute_default_50: "f16[512, 4096]" = torch.ops.aten.permute.default(view_226, [1, 0]);  view_226 = None
        permute_default_51: "f16[512, 4096]" = torch.ops.aten.permute.default(view_229, [1, 0]);  view_229 = None
        permute_default_52: "f16[2048, 4096]" = torch.ops.aten.permute.default(view_231, [1, 0]);  view_231 = None
        permute_default_53: "f16[512, 4096]" = torch.ops.aten.permute.default(view_234, [1, 0]);  view_234 = None
        permute_default_54: "f16[512, 4096]" = torch.ops.aten.permute.default(view_244, [1, 0]);  view_244 = None
        permute_default_55: "f16[512, 4096]" = torch.ops.aten.permute.default(view_247, [1, 0]);  view_247 = None
        permute_default_56: "f16[512, 4096]" = torch.ops.aten.permute.default(view_250, [1, 0]);  view_250 = None
        permute_default_57: "f16[512, 4096]" = torch.ops.aten.permute.default(view_253, [1, 0]);  view_253 = None
        reshape_default: "f16[4, 8, 1024, 64]" = torch.ops.aten.reshape.default(bmm_44, [4, 8, 1024, 64]);  bmm_44 = None
        reshape_default_1: "f16[4, 8, 64, 1024]" = torch.ops.aten.reshape.default(bmm_46, [4, 8, 64, 1024]);  bmm_46 = None
        reshape_default_2: "f16[4, 8, 1024, 64]" = torch.ops.aten.reshape.default(bmm_47, [4, 8, 1024, 64]);  bmm_47 = None
        permute_default_58: "f16[4, 8, 1024, 64]" = torch.ops.aten.permute.default(reshape_default_1, [0, 1, 3, 2]);  reshape_default_1 = None
        permute_default_59: "f16[4, 1024, 8, 64]" = torch.ops.aten.permute.default(reshape_default, [0, 2, 1, 3]);  reshape_default = None
        clone_default: "f16[4, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_default_59, memory_format = torch.contiguous_format);  permute_default_59 = None
        reshape_default_3: "f16[4, 1024, 512]" = torch.ops.aten.reshape.default(clone_default, [4, 1024, 512]);  clone_default = None
        reshape_default_4: "f16[4096, 512]" = torch.ops.aten.reshape.default(reshape_default_3, [4096, 512]);  reshape_default_3 = None
        permute_default_60: "f16[512, 4096]" = torch.ops.aten.permute.default(reshape_default_4, [1, 0]);  reshape_default_4 = None
        permute_default_61: "f16[4, 1024, 8, 64]" = torch.ops.aten.permute.default(permute_default_58, [0, 2, 1, 3]);  permute_default_58 = None
        reshape_default_5: "f16[4, 1024, 512]" = torch.ops.aten.reshape.default(permute_default_61, [4, 1024, 512]);  permute_default_61 = None
        clone_default_1: "f16[4, 1024, 512]" = torch.ops.aten.clone.default(reshape_default_5, memory_format = torch.contiguous_format);  reshape_default_5 = None
        reshape_default_6: "f16[4096, 512]" = torch.ops.aten.reshape.default(clone_default_1, [4096, 512]);  clone_default_1 = None
        permute_default_62: "f16[512, 4096]" = torch.ops.aten.permute.default(reshape_default_6, [1, 0]);  reshape_default_6 = None
        permute_default_63: "f16[4, 1024, 8, 64]" = torch.ops.aten.permute.default(reshape_default_2, [0, 2, 1, 3]);  reshape_default_2 = None
        clone_default_2: "f16[4, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_default_63, memory_format = torch.contiguous_format);  permute_default_63 = None
        reshape_default_7: "f16[4, 1024, 512]" = torch.ops.aten.reshape.default(clone_default_2, [4, 1024, 512]);  clone_default_2 = None
        reshape_default_8: "f16[4096, 512]" = torch.ops.aten.reshape.default(reshape_default_7, [4096, 512]);  reshape_default_7 = None
        permute_default_64: "f16[512, 4096]" = torch.ops.aten.permute.default(reshape_default_8, [1, 0]);  reshape_default_8 = None
        permute_default_65: "f16[512, 4096]" = torch.ops.aten.permute.default(view_276, [1, 0]);  view_276 = None
        permute_default_66: "f16[2048, 4096]" = torch.ops.aten.permute.default(view_278, [1, 0]);  view_278 = None
        permute_default_67: "f16[512, 4096]" = torch.ops.aten.permute.default(view_281, [1, 0]);  view_281 = None
        permute_default_68: "f16[512, 4096]" = torch.ops.aten.permute.default(view_293, [1, 0]);  view_293 = None
        permute_default_69: "f16[512, 4096]" = torch.ops.aten.permute.default(view_296, [1, 0]);  view_296 = None
        permute_default_70: "f16[512, 4096]" = torch.ops.aten.permute.default(view_299, [1, 0]);  view_299 = None
        permute_default_71: "f16[512, 4096]" = torch.ops.aten.permute.default(view_302, [1, 0]);  view_302 = None
        permute_default_72: "f16[2048, 4096]" = torch.ops.aten.permute.default(view_304, [1, 0]);  view_304 = None
        permute_default_73: "f16[512, 4096]" = torch.ops.aten.permute.default(view_307, [1, 0]);  view_307 = None
        permute_default_74: "f16[512, 4096]" = torch.ops.aten.permute.default(view_319, [1, 0]);  view_319 = None
        permute_default_75: "f16[512, 4096]" = torch.ops.aten.permute.default(view_322, [1, 0]);  view_322 = None
        permute_default_76: "f16[512, 4096]" = torch.ops.aten.permute.default(view_325, [1, 0]);  view_325 = None
        permute_default_77: "f16[512, 4096]" = torch.ops.aten.permute.default(view_328, [1, 0]);  view_328 = None
        permute_default_78: "f16[2048, 4096]" = torch.ops.aten.permute.default(view_330, [1, 0]);  view_330 = None
        permute_default_79: "f16[512, 4096]" = torch.ops.aten.permute.default(view_333, [1, 0]);  view_333 = None
        permute_default_80: "f16[512, 4096]" = torch.ops.aten.permute.default(view_345, [1, 0]);  view_345 = None
        permute_default_81: "f16[512, 4096]" = torch.ops.aten.permute.default(view_348, [1, 0]);  view_348 = None
        permute_default_82: "f16[512, 4096]" = torch.ops.aten.permute.default(view_351, [1, 0]);  view_351 = None
        permute_default_83: "f16[512, 4096]" = torch.ops.aten.permute.default(view_354, [1, 0]);  view_354 = None
        permute_default_84: "f16[2048, 4096]" = torch.ops.aten.permute.default(view_356, [1, 0]);  view_356 = None
        permute_default_85: "f16[512, 4096]" = torch.ops.aten.permute.default(view_359, [1, 0]);  view_359 = None
        permute_default_86: "f16[512, 4096]" = torch.ops.aten.permute.default(view_371, [1, 0]);  view_371 = None
        permute_default_87: "f16[512, 4096]" = torch.ops.aten.permute.default(view_374, [1, 0]);  view_374 = None
        permute_default_88: "f16[512, 4096]" = torch.ops.aten.permute.default(view_377, [1, 0]);  view_377 = None
        permute_default_89: "f16[512, 4096]" = torch.ops.aten.permute.default(view_380, [1, 0]);  view_380 = None
        permute_default_90: "f16[2048, 4096]" = torch.ops.aten.permute.default(view_382, [1, 0]);  view_382 = None
        permute_default_91: "f16[512, 4096]" = torch.ops.aten.permute.default(view_385, [1, 0]);  view_385 = None
        permute_default_92: "f16[512, 4096]" = torch.ops.aten.permute.default(view_397, [1, 0]);  view_397 = None
        permute_default_93: "f16[512, 4096]" = torch.ops.aten.permute.default(view_400, [1, 0]);  view_400 = None
        permute_default_94: "f16[512, 4096]" = torch.ops.aten.permute.default(view_403, [1, 0]);  view_403 = None
        permute_default_95: "f16[512, 4096]" = torch.ops.aten.permute.default(view_406, [1, 0]);  view_406 = None
        permute_default_96: "f16[2048, 4096]" = torch.ops.aten.permute.default(view_408, [1, 0]);  view_408 = None
        permute_default_97: "f16[512, 4096]" = torch.ops.aten.permute.default(view_411, [1, 0]);  view_411 = None
        reshape_default_9: "f16[4, 8, 1024, 64]" = torch.ops.aten.reshape.default(bmm_68, [4, 8, 1024, 64]);  bmm_68 = None
        reshape_default_10: "f16[4, 8, 64, 1024]" = torch.ops.aten.reshape.default(bmm_70, [4, 8, 64, 1024]);  bmm_70 = None
        reshape_default_11: "f16[4, 8, 1024, 64]" = torch.ops.aten.reshape.default(bmm_71, [4, 8, 1024, 64]);  bmm_71 = None
        permute_default_98: "f16[4, 8, 1024, 64]" = torch.ops.aten.permute.default(reshape_default_10, [0, 1, 3, 2]);  reshape_default_10 = None
        permute_default_99: "f16[4, 1024, 8, 64]" = torch.ops.aten.permute.default(reshape_default_9, [0, 2, 1, 3]);  reshape_default_9 = None
        clone_default_3: "f16[4, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_default_99, memory_format = torch.contiguous_format);  permute_default_99 = None
        reshape_default_12: "f16[4, 1024, 512]" = torch.ops.aten.reshape.default(clone_default_3, [4, 1024, 512]);  clone_default_3 = None
        reshape_default_13: "f16[4096, 512]" = torch.ops.aten.reshape.default(reshape_default_12, [4096, 512]);  reshape_default_12 = None
        permute_default_100: "f16[512, 4096]" = torch.ops.aten.permute.default(reshape_default_13, [1, 0]);  reshape_default_13 = None
        permute_default_101: "f16[4, 1024, 8, 64]" = torch.ops.aten.permute.default(permute_default_98, [0, 2, 1, 3]);  permute_default_98 = None
        reshape_default_14: "f16[4, 1024, 512]" = torch.ops.aten.reshape.default(permute_default_101, [4, 1024, 512]);  permute_default_101 = None
        clone_default_4: "f16[4, 1024, 512]" = torch.ops.aten.clone.default(reshape_default_14, memory_format = torch.contiguous_format);  reshape_default_14 = None
        reshape_default_15: "f16[4096, 512]" = torch.ops.aten.reshape.default(clone_default_4, [4096, 512]);  clone_default_4 = None
        permute_default_102: "f16[512, 4096]" = torch.ops.aten.permute.default(reshape_default_15, [1, 0]);  reshape_default_15 = None
        permute_default_103: "f16[4, 1024, 8, 64]" = torch.ops.aten.permute.default(reshape_default_11, [0, 2, 1, 3]);  reshape_default_11 = None
        clone_default_5: "f16[4, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_default_103, memory_format = torch.contiguous_format);  permute_default_103 = None
        reshape_default_16: "f16[4, 1024, 512]" = torch.ops.aten.reshape.default(clone_default_5, [4, 1024, 512]);  clone_default_5 = None
        reshape_default_17: "f16[4096, 512]" = torch.ops.aten.reshape.default(reshape_default_16, [4096, 512]);  reshape_default_16 = None
        permute_default_104: "f16[512, 4096]" = torch.ops.aten.permute.default(reshape_default_17, [1, 0]);  reshape_default_17 = None
        return (permute_default, permute_default_1, permute_default_2, permute_default_3, permute_default_4, permute_default_5, permute_default_6, permute_default_7, permute_default_8, permute_default_9, permute_default_10, permute_default_11, permute_default_12, permute_default_13, permute_default_14, permute_default_15, permute_default_16, permute_default_17, permute_default_18, permute_default_19, permute_default_20, permute_default_21, permute_default_22, permute_default_23, permute_default_24, permute_default_25, permute_default_26, permute_default_27, permute_default_28, permute_default_29, permute_default_30, permute_default_31, permute_default_32, permute_default_33, permute_default_34, permute_default_35, permute_default_36, permute_default_37, permute_default_38, permute_default_39, permute_default_40, permute_default_41, permute_default_42, permute_default_43, permute_default_44, permute_default_45, permute_default_46, permute_default_47, permute_default_48, permute_default_49, permute_default_50, permute_default_51, permute_default_52, permute_default_53, permute_default_54, permute_default_55, permute_default_56, permute_default_57, permute_default_60, permute_default_62, permute_default_64, permute_default_65, permute_default_66, permute_default_67, permute_default_68, permute_default_69, permute_default_70, permute_default_71, permute_default_72, permute_default_73, permute_default_74, permute_default_75, permute_default_76, permute_default_77, permute_default_78, permute_default_79, permute_default_80, permute_default_81, permute_default_82, permute_default_83, permute_default_84, permute_default_85, permute_default_86, permute_default_87, permute_default_88, permute_default_89, permute_default_90, permute_default_91, permute_default_92, permute_default_93, permute_default_94, permute_default_95, permute_default_96, permute_default_97, permute_default_100, permute_default_102, permute_default_104)


def _default_make_inputs():
    return [
    torch.randn([4096, 32128], dtype=torch.float16, device='cuda'),
    torch.randn([4096, 512], dtype=torch.float16, device='cuda'),
    torch.randn([4096, 2048], dtype=torch.float16, device='cuda'),
    torch.randn([4096, 512], dtype=torch.float16, device='cuda'),
    torch.randn([4096, 512], dtype=torch.float16, device='cuda'),
    torch.randn([4096, 512], dtype=torch.float16, device='cuda'),
    torch.randn([4096, 512], dtype=torch.float16, device='cuda'),
    torch.randn([4096, 512], dtype=torch.float16, device='cuda'),
    torch.randn([4096, 512], dtype=torch.float16, device='cuda'),
    torch.randn([4096, 512], dtype=torch.float16, device='cuda'),
    torch.randn([4096, 512], dtype=torch.float16, device='cuda'),
    torch.randn([4096, 512], dtype=torch.float16, device='cuda'),
    torch.randn([4096, 2048], dtype=torch.float16, device='cuda'),
    torch.randn([4096, 512], dtype=torch.float16, device='cuda'),
    torch.randn([4096, 512], dtype=torch.float16, device='cuda'),
    torch.randn([4096, 512], dtype=torch.float16, device='cuda'),
    torch.randn([4096, 512], dtype=torch.float16, device='cuda'),
    torch.randn([4096, 512], dtype=torch.float16, device='cuda'),
    torch.randn([4096, 512], dtype=torch.float16, device='cuda'),
    torch.randn([4096, 512], dtype=torch.float16, device='cuda'),
    torch.randn([4096, 512], dtype=torch.float16, device='cuda'),
    torch.randn([4096, 512], dtype=torch.float16, device='cuda'),
    torch.randn([4096, 2048], dtype=torch.float16, device='cuda'),
    torch.randn([4096, 512], dtype=torch.float16, device='cuda'),
    torch.randn([4096, 512], dtype=torch.float16, device='cuda'),
    torch.randn([4096, 512], dtype=torch.float16, device='cuda'),
    torch.randn([4096, 512], dtype=torch.float16, device='cuda'),
    torch.randn([4096, 512], dtype=torch.float16, device='cuda'),
    torch.randn([4096, 512], dtype=torch.float16, device='cuda'),
    torch.randn([4096, 512], dtype=torch.float16, device='cuda'),
    torch.randn([4096, 512], dtype=torch.float16, device='cuda'),
    torch.randn([4096, 512], dtype=torch.float16, device='cuda'),
    torch.randn([4096, 2048], dtype=torch.float16, device='cuda'),
    torch.randn([4096, 512], dtype=torch.float16, device='cuda'),
    torch.randn([4096, 512], dtype=torch.float16, device='cuda'),
    torch.randn([4096, 512], dtype=torch.float16, device='cuda'),
    torch.randn([4096, 512], dtype=torch.float16, device='cuda'),
    torch.randn([4096, 512], dtype=torch.float16, device='cuda'),
    torch.randn([4096, 512], dtype=torch.float16, device='cuda'),
    torch.randn([4096, 512], dtype=torch.float16, device='cuda'),
    torch.randn([4096, 512], dtype=torch.float16, device='cuda'),
    torch.randn([4096, 512], dtype=torch.float16, device='cuda'),
    torch.randn([4096, 2048], dtype=torch.float16, device='cuda'),
    torch.randn([4096, 512], dtype=torch.float16, device='cuda'),
    torch.randn([4096, 512], dtype=torch.float16, device='cuda'),
    torch.randn([4096, 512], dtype=torch.float16, device='cuda'),
    torch.randn([4096, 512], dtype=torch.float16, device='cuda'),
    torch.randn([4096, 512], dtype=torch.float16, device='cuda'),
    torch.randn([4096, 512], dtype=torch.float16, device='cuda'),
    torch.randn([4096, 512], dtype=torch.float16, device='cuda'),
    torch.randn([4096, 512], dtype=torch.float16, device='cuda'),
    torch.randn([4096, 512], dtype=torch.float16, device='cuda'),
    torch.randn([4096, 2048], dtype=torch.float16, device='cuda'),
    torch.randn([4096, 512], dtype=torch.float16, device='cuda'),
    torch.randn([4096, 512], dtype=torch.float16, device='cuda'),
    torch.randn([4096, 512], dtype=torch.float16, device='cuda'),
    torch.randn([4096, 512], dtype=torch.float16, device='cuda'),
    torch.randn([4096, 512], dtype=torch.float16, device='cuda'),
    torch.randn([32, 1024, 64], dtype=torch.float16, device='cuda'),
    torch.randn([32, 64, 1024], dtype=torch.float16, device='cuda'),
    torch.randn([32, 1024, 64], dtype=torch.float16, device='cuda'),
    torch.randn([4096, 512], dtype=torch.float16, device='cuda'),
    torch.randn([4096, 2048], dtype=torch.float16, device='cuda'),
    torch.randn([4096, 512], dtype=torch.float16, device='cuda'),
    torch.randn([4096, 512], dtype=torch.float16, device='cuda'),
    torch.randn([4096, 512], dtype=torch.float16, device='cuda'),
    torch.randn([4096, 512], dtype=torch.float16, device='cuda'),
    torch.randn([4096, 512], dtype=torch.float16, device='cuda'),
    torch.randn([4096, 2048], dtype=torch.float16, device='cuda'),
    torch.randn([4096, 512], dtype=torch.float16, device='cuda'),
    torch.randn([4096, 512], dtype=torch.float16, device='cuda'),
    torch.randn([4096, 512], dtype=torch.float16, device='cuda'),
    torch.randn([4096, 512], dtype=torch.float16, device='cuda'),
    torch.randn([4096, 512], dtype=torch.float16, device='cuda'),
    torch.randn([4096, 2048], dtype=torch.float16, device='cuda'),
    torch.randn([4096, 512], dtype=torch.float16, device='cuda'),
    torch.randn([4096, 512], dtype=torch.float16, device='cuda'),
    torch.randn([4096, 512], dtype=torch.float16, device='cuda'),
    torch.randn([4096, 512], dtype=torch.float16, device='cuda'),
    torch.randn([4096, 512], dtype=torch.float16, device='cuda'),
    torch.randn([4096, 2048], dtype=torch.float16, device='cuda'),
    torch.randn([4096, 512], dtype=torch.float16, device='cuda'),
    torch.randn([4096, 512], dtype=torch.float16, device='cuda'),
    torch.randn([4096, 512], dtype=torch.float16, device='cuda'),
    torch.randn([4096, 512], dtype=torch.float16, device='cuda'),
    torch.randn([4096, 512], dtype=torch.float16, device='cuda'),
    torch.randn([4096, 2048], dtype=torch.float16, device='cuda'),
    torch.randn([4096, 512], dtype=torch.float16, device='cuda'),
    torch.randn([4096, 512], dtype=torch.float16, device='cuda'),
    torch.randn([4096, 512], dtype=torch.float16, device='cuda'),
    torch.randn([4096, 512], dtype=torch.float16, device='cuda'),
    torch.randn([4096, 512], dtype=torch.float16, device='cuda'),
    torch.randn([4096, 2048], dtype=torch.float16, device='cuda'),
    torch.randn([4096, 512], dtype=torch.float16, device='cuda'),
    torch.randn([32, 1024, 64], dtype=torch.float16, device='cuda'),
    torch.randn([32, 64, 1024], dtype=torch.float16, device='cuda'),
    torch.randn([32, 1024, 64], dtype=torch.float16, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
