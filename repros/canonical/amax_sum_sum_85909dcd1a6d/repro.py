"""
Standalone repro captured via capture_hook.
Label: tlparse_huggingface_s4_g20
Pattern hash: 85909dcd1a6d
Shape hash: 8e91b307
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
    def forward(self, arg52_1: "i64[4, 1024]", mm_96: "f16[4096, 32128]", full: "f32[]", permute_188: "f16[512, 32128]", permute_187: "f16[2048, 512]", relu_11: "f16[4, 1024, 2048]", permute_186: "f16[512, 2048]", permute_185: "f16[512, 512]", view_327: "f16[32, 1024, 1024]", view_328: "f16[32, 1024, 64]", view_324: "f16[32, 1024, 64]", view_325: "f16[32, 64, 1024]", permute_181: "f16[512, 512]", permute_179: "f16[512, 512]", permute_177: "f16[512, 512]", permute_176: "f16[512, 512]", view_311: "f16[32, 1024, 1024]", view_312: "f16[32, 1024, 64]", view_308: "f16[32, 1024, 64]", view_309: "f16[32, 64, 1024]", permute_172: "f16[512, 512]", permute_170: "f16[512, 512]", permute_168: "f16[512, 512]", permute_167: "f16[2048, 512]", relu_10: "f16[4, 1024, 2048]", permute_166: "f16[512, 2048]", permute_165: "f16[512, 512]", view_291: "f16[32, 1024, 1024]", view_292: "f16[32, 1024, 64]", view_288: "f16[32, 1024, 64]", view_289: "f16[32, 64, 1024]", permute_161: "f16[512, 512]", permute_159: "f16[512, 512]", permute_157: "f16[512, 512]", permute_156: "f16[512, 512]", view_275: "f16[32, 1024, 1024]", view_276: "f16[32, 1024, 64]", view_272: "f16[32, 1024, 64]", view_273: "f16[32, 64, 1024]", permute_152: "f16[512, 512]", permute_150: "f16[512, 512]", permute_148: "f16[512, 512]", permute_147: "f16[2048, 512]", relu_9: "f16[4, 1024, 2048]", permute_146: "f16[512, 2048]", permute_145: "f16[512, 512]", view_255: "f16[32, 1024, 1024]", view_256: "f16[32, 1024, 64]", view_252: "f16[32, 1024, 64]", view_253: "f16[32, 64, 1024]", permute_141: "f16[512, 512]", permute_139: "f16[512, 512]", permute_137: "f16[512, 512]", permute_136: "f16[512, 512]", view_239: "f16[32, 1024, 1024]", view_240: "f16[32, 1024, 64]", view_236: "f16[32, 1024, 64]", view_237: "f16[32, 64, 1024]", permute_132: "f16[512, 512]", permute_130: "f16[512, 512]", permute_128: "f16[512, 512]", permute_127: "f16[2048, 512]", relu_8: "f16[4, 1024, 2048]", permute_126: "f16[512, 2048]", permute_125: "f16[512, 512]", view_219: "f16[32, 1024, 1024]", view_220: "f16[32, 1024, 64]", view_216: "f16[32, 1024, 64]", view_217: "f16[32, 64, 1024]", permute_121: "f16[512, 512]", permute_119: "f16[512, 512]", permute_117: "f16[512, 512]", permute_116: "f16[512, 512]", view_203: "f16[32, 1024, 1024]", view_204: "f16[32, 1024, 64]", view_200: "f16[32, 1024, 64]", view_201: "f16[32, 64, 1024]", permute_112: "f16[512, 512]", permute_110: "f16[512, 512]", permute_108: "f16[512, 512]", permute_107: "f16[2048, 512]", relu_7: "f16[4, 1024, 2048]", permute_106: "f16[512, 2048]", permute_105: "f16[512, 512]", view_183: "f16[32, 1024, 1024]", view_184: "f16[32, 1024, 64]", view_180: "f16[32, 1024, 64]", view_181: "f16[32, 64, 1024]", permute_101: "f16[512, 512]", permute_99: "f16[512, 512]", permute_97: "f16[512, 512]", permute_96: "f16[512, 512]", view_167: "f16[32, 1024, 1024]", view_168: "f16[32, 1024, 64]", view_164: "f16[32, 1024, 64]", view_165: "f16[32, 64, 1024]", permute_92: "f16[512, 512]", permute_90: "f16[512, 512]", permute_88: "f16[512, 512]", permute_87: "f16[2048, 512]", relu_6: "f16[4, 1024, 2048]", permute_86: "f16[512, 2048]", permute_85: "f16[512, 512]", view_147: "f16[32, 1024, 1024]", view_148: "f16[32, 1024, 64]", view_144: "f16[32, 1024, 64]", view_145: "f16[32, 64, 1024]", permute_81: "f16[512, 512]", permute_79: "f16[512, 512]", permute_77: "f16[512, 512]", permute_76: "f16[512, 512]", view_130: "f16[32, 1024, 1024]", view_131: "f16[32, 1024, 64]", view_127: "f16[32, 1024, 64]", view_128: "f16[32, 64, 1024]", permute_71: "f16[512, 512]", permute_69: "f16[512, 512]", permute_67: "f16[512, 512]", permute_66: "f16[2048, 512]", relu_5: "f16[4, 1024, 2048]", permute_65: "f16[512, 2048]", permute_64: "f16[512, 512]", view_110: "f16[32, 1024, 1024]", view_111: "f16[32, 1024, 64]", view_107: "f16[32, 1024, 64]", view_108: "f16[32, 64, 1024]", permute_60: "f16[512, 512]", permute_58: "f16[512, 512]", permute_56: "f16[512, 512]", permute_55: "f16[2048, 512]", relu_4: "f16[4, 1024, 2048]", permute_54: "f16[512, 2048]", permute_53: "f16[512, 512]", view_90: "f16[32, 1024, 1024]", view_91: "f16[32, 1024, 64]", view_87: "f16[32, 1024, 64]", view_88: "f16[32, 64, 1024]", permute_49: "f16[512, 512]", permute_47: "f16[512, 512]", permute_45: "f16[512, 512]", permute_44: "f16[2048, 512]", relu_3: "f16[4, 1024, 2048]", permute_43: "f16[512, 2048]", permute_42: "f16[512, 512]", view_70: "f16[32, 1024, 1024]", view_71: "f16[32, 1024, 64]", view_67: "f16[32, 1024, 64]", view_68: "f16[32, 64, 1024]", permute_38: "f16[512, 512]", permute_36: "f16[512, 512]", permute_34: "f16[512, 512]", permute_33: "f16[2048, 512]", relu_2: "f16[4, 1024, 2048]", permute_32: "f16[512, 2048]", permute_31: "f16[512, 512]", view_50: "f16[32, 1024, 1024]", view_51: "f16[32, 1024, 64]", view_47: "f16[32, 1024, 64]", view_48: "f16[32, 64, 1024]", permute_27: "f16[512, 512]", permute_25: "f16[512, 512]", permute_23: "f16[512, 512]", permute_22: "f16[2048, 512]", relu_1: "f16[4, 1024, 2048]", permute_21: "f16[512, 2048]", permute_20: "f16[512, 512]", view_30: "f16[32, 1024, 1024]", view_31: "f16[32, 1024, 64]", view_27: "f16[32, 1024, 64]", view_28: "f16[32, 64, 1024]", permute_16: "f16[512, 512]", permute_14: "f16[512, 512]", permute_12: "f16[512, 512]", permute_11: "f16[2048, 512]", relu: "f16[4, 1024, 2048]", permute_10: "f16[512, 2048]", permute_9: "f16[512, 512]", view_10: "f16[32, 1024, 1024]", view_11: "f16[32, 1024, 64]", view_7: "f16[32, 1024, 64]", view_8: "f16[32, 64, 1024]", permute_4: "f16[512, 512]", permute_2: "f16[512, 512]", permute: "f16[512, 512]"):
        # No stacktrace found for following nodes
        reshape_default: "i64[4096]" = torch.ops.aten.reshape.default(arg52_1, [-1]);  arg52_1 = None
        convert_element_type_default: "f32[4096, 32128]" = torch.ops.prims.convert_element_type.default(mm_96, torch.float32);  mm_96 = None
        amax_default: "f32[4096, 1]" = torch.ops.aten.amax.default(convert_element_type_default, [1], True)
        sub_tensor: "f32[4096, 32128]" = torch.ops.aten.sub.Tensor(convert_element_type_default, amax_default);  convert_element_type_default = amax_default = None
        exp_default: "f32[4096, 32128]" = torch.ops.aten.exp.default(sub_tensor)
        sum_dim_int_list: "f32[4096, 1]" = torch.ops.aten.sum.dim_IntList(exp_default, [1], True);  exp_default = None
        log_default: "f32[4096, 1]" = torch.ops.aten.log.default(sum_dim_int_list);  sum_dim_int_list = None
        sub_tensor_1: "f32[4096, 32128]" = torch.ops.aten.sub.Tensor(sub_tensor, log_default);  sub_tensor = log_default = None
        convert_element_type_default_1: "f16[4096, 32128]" = torch.ops.prims.convert_element_type.default(sub_tensor_1, torch.float16);  sub_tensor_1 = None
        convert_element_type_default_2: "f32[4096, 32128]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_1, torch.float32);  convert_element_type_default_1 = None
        ne_scalar: "b8[4096]" = torch.ops.aten.ne.Scalar(reshape_default, -100)
        full_default: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "i64[4096]" = torch.ops.aten.where.self(ne_scalar, reshape_default, full_default);  reshape_default = full_default = None
        unsqueeze_default: "i64[4096, 1]" = torch.ops.aten.unsqueeze.default(where_self, 1);  where_self = None
        gather_default: "f32[4096, 1]" = torch.ops.aten.gather.default(convert_element_type_default_2, 1, unsqueeze_default);  convert_element_type_default_2 = unsqueeze_default = None
        squeeze_dim: "f32[4096]" = torch.ops.aten.squeeze.dim(gather_default, 1);  gather_default = None
        neg_default: "f32[4096]" = torch.ops.aten.neg.default(squeeze_dim);  squeeze_dim = None
        where_self_1: "f32[4096]" = torch.ops.aten.where.self(ne_scalar, neg_default, full);  neg_default = full = None
        sum_default: "i64[]" = torch.ops.aten.sum.default(ne_scalar);  ne_scalar = None
        convert_element_type_default_3: "f32[]" = torch.ops.prims.convert_element_type.default(sum_default, torch.float32);  sum_default = None
        sum_default_1: "f32[]" = torch.ops.aten.sum.default(where_self_1);  where_self_1 = None
        div_tensor: "f32[]" = torch.ops.aten.div.Tensor(sum_default_1, convert_element_type_default_3);  sum_default_1 = convert_element_type_default_3 = None
        permute_default: "f16[32128, 512]" = torch.ops.aten.permute.default(permute_188, [1, 0]);  permute_188 = None
        permute_default_1: "f16[512, 2048]" = torch.ops.aten.permute.default(permute_187, [1, 0]);  permute_187 = None
        le_scalar: "b8[4, 1024, 2048]" = torch.ops.aten.le.Scalar(relu_11, 0);  relu_11 = None
        permute_default_2: "f16[2048, 512]" = torch.ops.aten.permute.default(permute_186, [1, 0]);  permute_186 = None
        permute_default_3: "f16[512, 512]" = torch.ops.aten.permute.default(permute_185, [1, 0]);  permute_185 = None
        permute_default_4: "f16[32, 1024, 1024]" = torch.ops.aten.permute.default(view_327, [0, 2, 1]);  view_327 = None
        permute_default_5: "f16[32, 64, 1024]" = torch.ops.aten.permute.default(view_328, [0, 2, 1]);  view_328 = None
        permute_default_6: "f16[32, 64, 1024]" = torch.ops.aten.permute.default(view_324, [0, 2, 1]);  view_324 = None
        permute_default_7: "f16[32, 1024, 64]" = torch.ops.aten.permute.default(view_325, [0, 2, 1]);  view_325 = None
        permute_default_8: "f16[512, 512]" = torch.ops.aten.permute.default(permute_181, [1, 0]);  permute_181 = None
        permute_default_9: "f16[512, 512]" = torch.ops.aten.permute.default(permute_179, [1, 0]);  permute_179 = None
        permute_default_10: "f16[512, 512]" = torch.ops.aten.permute.default(permute_177, [1, 0]);  permute_177 = None
        permute_default_11: "f16[512, 512]" = torch.ops.aten.permute.default(permute_176, [1, 0]);  permute_176 = None
        permute_default_12: "f16[32, 1024, 1024]" = torch.ops.aten.permute.default(view_311, [0, 2, 1]);  view_311 = None
        permute_default_13: "f16[32, 64, 1024]" = torch.ops.aten.permute.default(view_312, [0, 2, 1]);  view_312 = None
        permute_default_14: "f16[32, 64, 1024]" = torch.ops.aten.permute.default(view_308, [0, 2, 1]);  view_308 = None
        permute_default_15: "f16[32, 1024, 64]" = torch.ops.aten.permute.default(view_309, [0, 2, 1]);  view_309 = None
        permute_default_16: "f16[512, 512]" = torch.ops.aten.permute.default(permute_172, [1, 0]);  permute_172 = None
        permute_default_17: "f16[512, 512]" = torch.ops.aten.permute.default(permute_170, [1, 0]);  permute_170 = None
        permute_default_18: "f16[512, 512]" = torch.ops.aten.permute.default(permute_168, [1, 0]);  permute_168 = None
        permute_default_19: "f16[512, 2048]" = torch.ops.aten.permute.default(permute_167, [1, 0]);  permute_167 = None
        le_scalar_1: "b8[4, 1024, 2048]" = torch.ops.aten.le.Scalar(relu_10, 0);  relu_10 = None
        permute_default_20: "f16[2048, 512]" = torch.ops.aten.permute.default(permute_166, [1, 0]);  permute_166 = None
        permute_default_21: "f16[512, 512]" = torch.ops.aten.permute.default(permute_165, [1, 0]);  permute_165 = None
        permute_default_22: "f16[32, 1024, 1024]" = torch.ops.aten.permute.default(view_291, [0, 2, 1]);  view_291 = None
        permute_default_23: "f16[32, 64, 1024]" = torch.ops.aten.permute.default(view_292, [0, 2, 1]);  view_292 = None
        permute_default_24: "f16[32, 64, 1024]" = torch.ops.aten.permute.default(view_288, [0, 2, 1]);  view_288 = None
        permute_default_25: "f16[32, 1024, 64]" = torch.ops.aten.permute.default(view_289, [0, 2, 1]);  view_289 = None
        permute_default_26: "f16[512, 512]" = torch.ops.aten.permute.default(permute_161, [1, 0]);  permute_161 = None
        permute_default_27: "f16[512, 512]" = torch.ops.aten.permute.default(permute_159, [1, 0]);  permute_159 = None
        permute_default_28: "f16[512, 512]" = torch.ops.aten.permute.default(permute_157, [1, 0]);  permute_157 = None
        permute_default_29: "f16[512, 512]" = torch.ops.aten.permute.default(permute_156, [1, 0]);  permute_156 = None
        permute_default_30: "f16[32, 1024, 1024]" = torch.ops.aten.permute.default(view_275, [0, 2, 1]);  view_275 = None
        permute_default_31: "f16[32, 64, 1024]" = torch.ops.aten.permute.default(view_276, [0, 2, 1]);  view_276 = None
        permute_default_32: "f16[32, 64, 1024]" = torch.ops.aten.permute.default(view_272, [0, 2, 1]);  view_272 = None
        permute_default_33: "f16[32, 1024, 64]" = torch.ops.aten.permute.default(view_273, [0, 2, 1]);  view_273 = None
        permute_default_34: "f16[512, 512]" = torch.ops.aten.permute.default(permute_152, [1, 0]);  permute_152 = None
        permute_default_35: "f16[512, 512]" = torch.ops.aten.permute.default(permute_150, [1, 0]);  permute_150 = None
        permute_default_36: "f16[512, 512]" = torch.ops.aten.permute.default(permute_148, [1, 0]);  permute_148 = None
        permute_default_37: "f16[512, 2048]" = torch.ops.aten.permute.default(permute_147, [1, 0]);  permute_147 = None
        le_scalar_2: "b8[4, 1024, 2048]" = torch.ops.aten.le.Scalar(relu_9, 0);  relu_9 = None
        permute_default_38: "f16[2048, 512]" = torch.ops.aten.permute.default(permute_146, [1, 0]);  permute_146 = None
        permute_default_39: "f16[512, 512]" = torch.ops.aten.permute.default(permute_145, [1, 0]);  permute_145 = None
        permute_default_40: "f16[32, 1024, 1024]" = torch.ops.aten.permute.default(view_255, [0, 2, 1]);  view_255 = None
        permute_default_41: "f16[32, 64, 1024]" = torch.ops.aten.permute.default(view_256, [0, 2, 1]);  view_256 = None
        permute_default_42: "f16[32, 64, 1024]" = torch.ops.aten.permute.default(view_252, [0, 2, 1]);  view_252 = None
        permute_default_43: "f16[32, 1024, 64]" = torch.ops.aten.permute.default(view_253, [0, 2, 1]);  view_253 = None
        permute_default_44: "f16[512, 512]" = torch.ops.aten.permute.default(permute_141, [1, 0]);  permute_141 = None
        permute_default_45: "f16[512, 512]" = torch.ops.aten.permute.default(permute_139, [1, 0]);  permute_139 = None
        permute_default_46: "f16[512, 512]" = torch.ops.aten.permute.default(permute_137, [1, 0]);  permute_137 = None
        permute_default_47: "f16[512, 512]" = torch.ops.aten.permute.default(permute_136, [1, 0]);  permute_136 = None
        permute_default_48: "f16[32, 1024, 1024]" = torch.ops.aten.permute.default(view_239, [0, 2, 1]);  view_239 = None
        permute_default_49: "f16[32, 64, 1024]" = torch.ops.aten.permute.default(view_240, [0, 2, 1]);  view_240 = None
        permute_default_50: "f16[32, 64, 1024]" = torch.ops.aten.permute.default(view_236, [0, 2, 1]);  view_236 = None
        permute_default_51: "f16[32, 1024, 64]" = torch.ops.aten.permute.default(view_237, [0, 2, 1]);  view_237 = None
        permute_default_52: "f16[512, 512]" = torch.ops.aten.permute.default(permute_132, [1, 0]);  permute_132 = None
        permute_default_53: "f16[512, 512]" = torch.ops.aten.permute.default(permute_130, [1, 0]);  permute_130 = None
        permute_default_54: "f16[512, 512]" = torch.ops.aten.permute.default(permute_128, [1, 0]);  permute_128 = None
        permute_default_55: "f16[512, 2048]" = torch.ops.aten.permute.default(permute_127, [1, 0]);  permute_127 = None
        le_scalar_3: "b8[4, 1024, 2048]" = torch.ops.aten.le.Scalar(relu_8, 0);  relu_8 = None
        permute_default_56: "f16[2048, 512]" = torch.ops.aten.permute.default(permute_126, [1, 0]);  permute_126 = None
        permute_default_57: "f16[512, 512]" = torch.ops.aten.permute.default(permute_125, [1, 0]);  permute_125 = None
        permute_default_58: "f16[32, 1024, 1024]" = torch.ops.aten.permute.default(view_219, [0, 2, 1]);  view_219 = None
        permute_default_59: "f16[32, 64, 1024]" = torch.ops.aten.permute.default(view_220, [0, 2, 1]);  view_220 = None
        permute_default_60: "f16[32, 64, 1024]" = torch.ops.aten.permute.default(view_216, [0, 2, 1]);  view_216 = None
        permute_default_61: "f16[32, 1024, 64]" = torch.ops.aten.permute.default(view_217, [0, 2, 1]);  view_217 = None
        permute_default_62: "f16[512, 512]" = torch.ops.aten.permute.default(permute_121, [1, 0]);  permute_121 = None
        permute_default_63: "f16[512, 512]" = torch.ops.aten.permute.default(permute_119, [1, 0]);  permute_119 = None
        permute_default_64: "f16[512, 512]" = torch.ops.aten.permute.default(permute_117, [1, 0]);  permute_117 = None
        permute_default_65: "f16[512, 512]" = torch.ops.aten.permute.default(permute_116, [1, 0]);  permute_116 = None
        permute_default_66: "f16[32, 1024, 1024]" = torch.ops.aten.permute.default(view_203, [0, 2, 1]);  view_203 = None
        permute_default_67: "f16[32, 64, 1024]" = torch.ops.aten.permute.default(view_204, [0, 2, 1]);  view_204 = None
        permute_default_68: "f16[32, 64, 1024]" = torch.ops.aten.permute.default(view_200, [0, 2, 1]);  view_200 = None
        permute_default_69: "f16[32, 1024, 64]" = torch.ops.aten.permute.default(view_201, [0, 2, 1]);  view_201 = None
        permute_default_70: "f16[512, 512]" = torch.ops.aten.permute.default(permute_112, [1, 0]);  permute_112 = None
        permute_default_71: "f16[512, 512]" = torch.ops.aten.permute.default(permute_110, [1, 0]);  permute_110 = None
        permute_default_72: "f16[512, 512]" = torch.ops.aten.permute.default(permute_108, [1, 0]);  permute_108 = None
        permute_default_73: "f16[512, 2048]" = torch.ops.aten.permute.default(permute_107, [1, 0]);  permute_107 = None
        le_scalar_4: "b8[4, 1024, 2048]" = torch.ops.aten.le.Scalar(relu_7, 0);  relu_7 = None
        permute_default_74: "f16[2048, 512]" = torch.ops.aten.permute.default(permute_106, [1, 0]);  permute_106 = None
        permute_default_75: "f16[512, 512]" = torch.ops.aten.permute.default(permute_105, [1, 0]);  permute_105 = None
        permute_default_76: "f16[32, 1024, 1024]" = torch.ops.aten.permute.default(view_183, [0, 2, 1]);  view_183 = None
        permute_default_77: "f16[32, 64, 1024]" = torch.ops.aten.permute.default(view_184, [0, 2, 1]);  view_184 = None
        permute_default_78: "f16[32, 64, 1024]" = torch.ops.aten.permute.default(view_180, [0, 2, 1]);  view_180 = None
        permute_default_79: "f16[32, 1024, 64]" = torch.ops.aten.permute.default(view_181, [0, 2, 1]);  view_181 = None
        permute_default_80: "f16[512, 512]" = torch.ops.aten.permute.default(permute_101, [1, 0]);  permute_101 = None
        permute_default_81: "f16[512, 512]" = torch.ops.aten.permute.default(permute_99, [1, 0]);  permute_99 = None
        permute_default_82: "f16[512, 512]" = torch.ops.aten.permute.default(permute_97, [1, 0]);  permute_97 = None
        permute_default_83: "f16[512, 512]" = torch.ops.aten.permute.default(permute_96, [1, 0]);  permute_96 = None
        permute_default_84: "f16[32, 1024, 1024]" = torch.ops.aten.permute.default(view_167, [0, 2, 1]);  view_167 = None
        permute_default_85: "f16[32, 64, 1024]" = torch.ops.aten.permute.default(view_168, [0, 2, 1]);  view_168 = None
        permute_default_86: "f16[32, 64, 1024]" = torch.ops.aten.permute.default(view_164, [0, 2, 1]);  view_164 = None
        permute_default_87: "f16[32, 1024, 64]" = torch.ops.aten.permute.default(view_165, [0, 2, 1]);  view_165 = None
        permute_default_88: "f16[512, 512]" = torch.ops.aten.permute.default(permute_92, [1, 0]);  permute_92 = None
        permute_default_89: "f16[512, 512]" = torch.ops.aten.permute.default(permute_90, [1, 0]);  permute_90 = None
        permute_default_90: "f16[512, 512]" = torch.ops.aten.permute.default(permute_88, [1, 0]);  permute_88 = None
        permute_default_91: "f16[512, 2048]" = torch.ops.aten.permute.default(permute_87, [1, 0]);  permute_87 = None
        le_scalar_5: "b8[4, 1024, 2048]" = torch.ops.aten.le.Scalar(relu_6, 0);  relu_6 = None
        permute_default_92: "f16[2048, 512]" = torch.ops.aten.permute.default(permute_86, [1, 0]);  permute_86 = None
        permute_default_93: "f16[512, 512]" = torch.ops.aten.permute.default(permute_85, [1, 0]);  permute_85 = None
        permute_default_94: "f16[32, 1024, 1024]" = torch.ops.aten.permute.default(view_147, [0, 2, 1]);  view_147 = None
        permute_default_95: "f16[32, 64, 1024]" = torch.ops.aten.permute.default(view_148, [0, 2, 1]);  view_148 = None
        permute_default_96: "f16[32, 64, 1024]" = torch.ops.aten.permute.default(view_144, [0, 2, 1]);  view_144 = None
        permute_default_97: "f16[32, 1024, 64]" = torch.ops.aten.permute.default(view_145, [0, 2, 1]);  view_145 = None
        permute_default_98: "f16[512, 512]" = torch.ops.aten.permute.default(permute_81, [1, 0]);  permute_81 = None
        permute_default_99: "f16[512, 512]" = torch.ops.aten.permute.default(permute_79, [1, 0]);  permute_79 = None
        permute_default_100: "f16[512, 512]" = torch.ops.aten.permute.default(permute_77, [1, 0]);  permute_77 = None
        permute_default_101: "f16[512, 512]" = torch.ops.aten.permute.default(permute_76, [1, 0]);  permute_76 = None
        permute_default_102: "f16[32, 1024, 1024]" = torch.ops.aten.permute.default(view_130, [0, 2, 1]);  view_130 = None
        permute_default_103: "f16[32, 64, 1024]" = torch.ops.aten.permute.default(view_131, [0, 2, 1]);  view_131 = None
        permute_default_104: "f16[32, 64, 1024]" = torch.ops.aten.permute.default(view_127, [0, 2, 1]);  view_127 = None
        permute_default_105: "f16[32, 1024, 64]" = torch.ops.aten.permute.default(view_128, [0, 2, 1]);  view_128 = None
        permute_default_106: "f16[512, 512]" = torch.ops.aten.permute.default(permute_71, [1, 0]);  permute_71 = None
        permute_default_107: "f16[512, 512]" = torch.ops.aten.permute.default(permute_69, [1, 0]);  permute_69 = None
        permute_default_108: "f16[512, 512]" = torch.ops.aten.permute.default(permute_67, [1, 0]);  permute_67 = None
        permute_default_109: "f16[512, 2048]" = torch.ops.aten.permute.default(permute_66, [1, 0]);  permute_66 = None
        le_scalar_6: "b8[4, 1024, 2048]" = torch.ops.aten.le.Scalar(relu_5, 0);  relu_5 = None
        permute_default_110: "f16[2048, 512]" = torch.ops.aten.permute.default(permute_65, [1, 0]);  permute_65 = None
        permute_default_111: "f16[512, 512]" = torch.ops.aten.permute.default(permute_64, [1, 0]);  permute_64 = None
        permute_default_112: "f16[32, 1024, 1024]" = torch.ops.aten.permute.default(view_110, [0, 2, 1]);  view_110 = None
        permute_default_113: "f16[32, 64, 1024]" = torch.ops.aten.permute.default(view_111, [0, 2, 1]);  view_111 = None
        permute_default_114: "f16[32, 64, 1024]" = torch.ops.aten.permute.default(view_107, [0, 2, 1]);  view_107 = None
        permute_default_115: "f16[32, 1024, 64]" = torch.ops.aten.permute.default(view_108, [0, 2, 1]);  view_108 = None
        permute_default_116: "f16[512, 512]" = torch.ops.aten.permute.default(permute_60, [1, 0]);  permute_60 = None
        permute_default_117: "f16[512, 512]" = torch.ops.aten.permute.default(permute_58, [1, 0]);  permute_58 = None
        permute_default_118: "f16[512, 512]" = torch.ops.aten.permute.default(permute_56, [1, 0]);  permute_56 = None
        permute_default_119: "f16[512, 2048]" = torch.ops.aten.permute.default(permute_55, [1, 0]);  permute_55 = None
        le_scalar_7: "b8[4, 1024, 2048]" = torch.ops.aten.le.Scalar(relu_4, 0);  relu_4 = None
        permute_default_120: "f16[2048, 512]" = torch.ops.aten.permute.default(permute_54, [1, 0]);  permute_54 = None
        permute_default_121: "f16[512, 512]" = torch.ops.aten.permute.default(permute_53, [1, 0]);  permute_53 = None
        permute_default_122: "f16[32, 1024, 1024]" = torch.ops.aten.permute.default(view_90, [0, 2, 1]);  view_90 = None
        permute_default_123: "f16[32, 64, 1024]" = torch.ops.aten.permute.default(view_91, [0, 2, 1]);  view_91 = None
        permute_default_124: "f16[32, 64, 1024]" = torch.ops.aten.permute.default(view_87, [0, 2, 1]);  view_87 = None
        permute_default_125: "f16[32, 1024, 64]" = torch.ops.aten.permute.default(view_88, [0, 2, 1]);  view_88 = None
        permute_default_126: "f16[512, 512]" = torch.ops.aten.permute.default(permute_49, [1, 0]);  permute_49 = None
        permute_default_127: "f16[512, 512]" = torch.ops.aten.permute.default(permute_47, [1, 0]);  permute_47 = None
        permute_default_128: "f16[512, 512]" = torch.ops.aten.permute.default(permute_45, [1, 0]);  permute_45 = None
        permute_default_129: "f16[512, 2048]" = torch.ops.aten.permute.default(permute_44, [1, 0]);  permute_44 = None
        le_scalar_8: "b8[4, 1024, 2048]" = torch.ops.aten.le.Scalar(relu_3, 0);  relu_3 = None
        permute_default_130: "f16[2048, 512]" = torch.ops.aten.permute.default(permute_43, [1, 0]);  permute_43 = None
        permute_default_131: "f16[512, 512]" = torch.ops.aten.permute.default(permute_42, [1, 0]);  permute_42 = None
        permute_default_132: "f16[32, 1024, 1024]" = torch.ops.aten.permute.default(view_70, [0, 2, 1]);  view_70 = None
        permute_default_133: "f16[32, 64, 1024]" = torch.ops.aten.permute.default(view_71, [0, 2, 1]);  view_71 = None
        permute_default_134: "f16[32, 64, 1024]" = torch.ops.aten.permute.default(view_67, [0, 2, 1]);  view_67 = None
        permute_default_135: "f16[32, 1024, 64]" = torch.ops.aten.permute.default(view_68, [0, 2, 1]);  view_68 = None
        permute_default_136: "f16[512, 512]" = torch.ops.aten.permute.default(permute_38, [1, 0]);  permute_38 = None
        permute_default_137: "f16[512, 512]" = torch.ops.aten.permute.default(permute_36, [1, 0]);  permute_36 = None
        permute_default_138: "f16[512, 512]" = torch.ops.aten.permute.default(permute_34, [1, 0]);  permute_34 = None
        permute_default_139: "f16[512, 2048]" = torch.ops.aten.permute.default(permute_33, [1, 0]);  permute_33 = None
        le_scalar_9: "b8[4, 1024, 2048]" = torch.ops.aten.le.Scalar(relu_2, 0);  relu_2 = None
        permute_default_140: "f16[2048, 512]" = torch.ops.aten.permute.default(permute_32, [1, 0]);  permute_32 = None
        permute_default_141: "f16[512, 512]" = torch.ops.aten.permute.default(permute_31, [1, 0]);  permute_31 = None
        permute_default_142: "f16[32, 1024, 1024]" = torch.ops.aten.permute.default(view_50, [0, 2, 1]);  view_50 = None
        permute_default_143: "f16[32, 64, 1024]" = torch.ops.aten.permute.default(view_51, [0, 2, 1]);  view_51 = None
        permute_default_144: "f16[32, 64, 1024]" = torch.ops.aten.permute.default(view_47, [0, 2, 1]);  view_47 = None
        permute_default_145: "f16[32, 1024, 64]" = torch.ops.aten.permute.default(view_48, [0, 2, 1]);  view_48 = None
        permute_default_146: "f16[512, 512]" = torch.ops.aten.permute.default(permute_27, [1, 0]);  permute_27 = None
        permute_default_147: "f16[512, 512]" = torch.ops.aten.permute.default(permute_25, [1, 0]);  permute_25 = None
        permute_default_148: "f16[512, 512]" = torch.ops.aten.permute.default(permute_23, [1, 0]);  permute_23 = None
        permute_default_149: "f16[512, 2048]" = torch.ops.aten.permute.default(permute_22, [1, 0]);  permute_22 = None
        le_scalar_10: "b8[4, 1024, 2048]" = torch.ops.aten.le.Scalar(relu_1, 0);  relu_1 = None
        permute_default_150: "f16[2048, 512]" = torch.ops.aten.permute.default(permute_21, [1, 0]);  permute_21 = None
        permute_default_151: "f16[512, 512]" = torch.ops.aten.permute.default(permute_20, [1, 0]);  permute_20 = None
        permute_default_152: "f16[32, 1024, 1024]" = torch.ops.aten.permute.default(view_30, [0, 2, 1]);  view_30 = None
        permute_default_153: "f16[32, 64, 1024]" = torch.ops.aten.permute.default(view_31, [0, 2, 1]);  view_31 = None
        permute_default_154: "f16[32, 64, 1024]" = torch.ops.aten.permute.default(view_27, [0, 2, 1]);  view_27 = None
        permute_default_155: "f16[32, 1024, 64]" = torch.ops.aten.permute.default(view_28, [0, 2, 1]);  view_28 = None
        permute_default_156: "f16[512, 512]" = torch.ops.aten.permute.default(permute_16, [1, 0]);  permute_16 = None
        permute_default_157: "f16[512, 512]" = torch.ops.aten.permute.default(permute_14, [1, 0]);  permute_14 = None
        permute_default_158: "f16[512, 512]" = torch.ops.aten.permute.default(permute_12, [1, 0]);  permute_12 = None
        permute_default_159: "f16[512, 2048]" = torch.ops.aten.permute.default(permute_11, [1, 0]);  permute_11 = None
        le_scalar_11: "b8[4, 1024, 2048]" = torch.ops.aten.le.Scalar(relu, 0);  relu = None
        permute_default_160: "f16[2048, 512]" = torch.ops.aten.permute.default(permute_10, [1, 0]);  permute_10 = None
        permute_default_161: "f16[512, 512]" = torch.ops.aten.permute.default(permute_9, [1, 0]);  permute_9 = None
        permute_default_162: "f16[32, 1024, 1024]" = torch.ops.aten.permute.default(view_10, [0, 2, 1]);  view_10 = None
        permute_default_163: "f16[32, 64, 1024]" = torch.ops.aten.permute.default(view_11, [0, 2, 1]);  view_11 = None
        permute_default_164: "f16[32, 64, 1024]" = torch.ops.aten.permute.default(view_7, [0, 2, 1]);  view_7 = None
        permute_default_165: "f16[32, 1024, 64]" = torch.ops.aten.permute.default(view_8, [0, 2, 1]);  view_8 = None
        permute_default_166: "f16[512, 512]" = torch.ops.aten.permute.default(permute_4, [1, 0]);  permute_4 = None
        permute_default_167: "f16[512, 512]" = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None
        permute_default_168: "f16[512, 512]" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
        return (div_tensor, permute_default, permute_default_1, le_scalar, permute_default_2, permute_default_3, permute_default_4, permute_default_5, permute_default_6, permute_default_7, permute_default_8, permute_default_9, permute_default_10, permute_default_11, permute_default_12, permute_default_13, permute_default_14, permute_default_15, permute_default_16, permute_default_17, permute_default_18, permute_default_19, le_scalar_1, permute_default_20, permute_default_21, permute_default_22, permute_default_23, permute_default_24, permute_default_25, permute_default_26, permute_default_27, permute_default_28, permute_default_29, permute_default_30, permute_default_31, permute_default_32, permute_default_33, permute_default_34, permute_default_35, permute_default_36, permute_default_37, le_scalar_2, permute_default_38, permute_default_39, permute_default_40, permute_default_41, permute_default_42, permute_default_43, permute_default_44, permute_default_45, permute_default_46, permute_default_47, permute_default_48, permute_default_49, permute_default_50, permute_default_51, permute_default_52, permute_default_53, permute_default_54, permute_default_55, le_scalar_3, permute_default_56, permute_default_57, permute_default_58, permute_default_59, permute_default_60, permute_default_61, permute_default_62, permute_default_63, permute_default_64, permute_default_65, permute_default_66, permute_default_67, permute_default_68, permute_default_69, permute_default_70, permute_default_71, permute_default_72, permute_default_73, le_scalar_4, permute_default_74, permute_default_75, permute_default_76, permute_default_77, permute_default_78, permute_default_79, permute_default_80, permute_default_81, permute_default_82, permute_default_83, permute_default_84, permute_default_85, permute_default_86, permute_default_87, permute_default_88, permute_default_89, permute_default_90, permute_default_91, le_scalar_5, permute_default_92, permute_default_93, permute_default_94, permute_default_95, permute_default_96, permute_default_97, permute_default_98, permute_default_99, permute_default_100, permute_default_101, permute_default_102, permute_default_103, permute_default_104, permute_default_105, permute_default_106, permute_default_107, permute_default_108, permute_default_109, le_scalar_6, permute_default_110, permute_default_111, permute_default_112, permute_default_113, permute_default_114, permute_default_115, permute_default_116, permute_default_117, permute_default_118, permute_default_119, le_scalar_7, permute_default_120, permute_default_121, permute_default_122, permute_default_123, permute_default_124, permute_default_125, permute_default_126, permute_default_127, permute_default_128, permute_default_129, le_scalar_8, permute_default_130, permute_default_131, permute_default_132, permute_default_133, permute_default_134, permute_default_135, permute_default_136, permute_default_137, permute_default_138, permute_default_139, le_scalar_9, permute_default_140, permute_default_141, permute_default_142, permute_default_143, permute_default_144, permute_default_145, permute_default_146, permute_default_147, permute_default_148, permute_default_149, le_scalar_10, permute_default_150, permute_default_151, permute_default_152, permute_default_153, permute_default_154, permute_default_155, permute_default_156, permute_default_157, permute_default_158, permute_default_159, le_scalar_11, permute_default_160, permute_default_161, permute_default_162, permute_default_163, permute_default_164, permute_default_165, permute_default_166, permute_default_167, permute_default_168)


def _default_make_inputs():
    return [
    torch.randint(0, 32128, [4, 1024], dtype=torch.int64, device='cuda'),
    torch.randn([4096, 32128], dtype=torch.float16, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn(16449536, dtype=torch.float16, device='cuda').as_strided([512, 32128], [1, 512]),  # permute_188
    torch.randn(1048576, dtype=torch.float16, device='cuda').as_strided([2048, 512], [1, 2048]),  # permute_187
    torch.randn([4, 1024, 2048], dtype=torch.float16, device='cuda'),
    torch.randn(1048576, dtype=torch.float16, device='cuda').as_strided([512, 2048], [1, 512]),  # permute_186
    torch.randn(262144, dtype=torch.float16, device='cuda').as_strided([512, 512], [1, 512]),  # permute_185
    torch.randn([32, 1024, 1024], dtype=torch.float16, device='cuda'),
    torch.randn([32, 1024, 64], dtype=torch.float16, device='cuda'),
    torch.randn([32, 1024, 64], dtype=torch.float16, device='cuda'),
    torch.randn([32, 64, 1024], dtype=torch.float16, device='cuda'),
    torch.randn(262144, dtype=torch.float16, device='cuda').as_strided([512, 512], [1, 512]),  # permute_181
    torch.randn(262144, dtype=torch.float16, device='cuda').as_strided([512, 512], [1, 512]),  # permute_179
    torch.randn(262144, dtype=torch.float16, device='cuda').as_strided([512, 512], [1, 512]),  # permute_177
    torch.randn(262144, dtype=torch.float16, device='cuda').as_strided([512, 512], [1, 512]),  # permute_176
    torch.randn([32, 1024, 1024], dtype=torch.float16, device='cuda'),
    torch.randn([32, 1024, 64], dtype=torch.float16, device='cuda'),
    torch.randn([32, 1024, 64], dtype=torch.float16, device='cuda'),
    torch.randn([32, 64, 1024], dtype=torch.float16, device='cuda'),
    torch.randn(262144, dtype=torch.float16, device='cuda').as_strided([512, 512], [1, 512]),  # permute_172
    torch.randn(262144, dtype=torch.float16, device='cuda').as_strided([512, 512], [1, 512]),  # permute_170
    torch.randn(262144, dtype=torch.float16, device='cuda').as_strided([512, 512], [1, 512]),  # permute_168
    torch.randn(1048576, dtype=torch.float16, device='cuda').as_strided([2048, 512], [1, 2048]),  # permute_167
    torch.randn([4, 1024, 2048], dtype=torch.float16, device='cuda'),
    torch.randn(1048576, dtype=torch.float16, device='cuda').as_strided([512, 2048], [1, 512]),  # permute_166
    torch.randn(262144, dtype=torch.float16, device='cuda').as_strided([512, 512], [1, 512]),  # permute_165
    torch.randn([32, 1024, 1024], dtype=torch.float16, device='cuda'),
    torch.randn([32, 1024, 64], dtype=torch.float16, device='cuda'),
    torch.randn([32, 1024, 64], dtype=torch.float16, device='cuda'),
    torch.randn([32, 64, 1024], dtype=torch.float16, device='cuda'),
    torch.randn(262144, dtype=torch.float16, device='cuda').as_strided([512, 512], [1, 512]),  # permute_161
    torch.randn(262144, dtype=torch.float16, device='cuda').as_strided([512, 512], [1, 512]),  # permute_159
    torch.randn(262144, dtype=torch.float16, device='cuda').as_strided([512, 512], [1, 512]),  # permute_157
    torch.randn(262144, dtype=torch.float16, device='cuda').as_strided([512, 512], [1, 512]),  # permute_156
    torch.randn([32, 1024, 1024], dtype=torch.float16, device='cuda'),
    torch.randn([32, 1024, 64], dtype=torch.float16, device='cuda'),
    torch.randn([32, 1024, 64], dtype=torch.float16, device='cuda'),
    torch.randn([32, 64, 1024], dtype=torch.float16, device='cuda'),
    torch.randn(262144, dtype=torch.float16, device='cuda').as_strided([512, 512], [1, 512]),  # permute_152
    torch.randn(262144, dtype=torch.float16, device='cuda').as_strided([512, 512], [1, 512]),  # permute_150
    torch.randn(262144, dtype=torch.float16, device='cuda').as_strided([512, 512], [1, 512]),  # permute_148
    torch.randn(1048576, dtype=torch.float16, device='cuda').as_strided([2048, 512], [1, 2048]),  # permute_147
    torch.randn([4, 1024, 2048], dtype=torch.float16, device='cuda'),
    torch.randn(1048576, dtype=torch.float16, device='cuda').as_strided([512, 2048], [1, 512]),  # permute_146
    torch.randn(262144, dtype=torch.float16, device='cuda').as_strided([512, 512], [1, 512]),  # permute_145
    torch.randn([32, 1024, 1024], dtype=torch.float16, device='cuda'),
    torch.randn([32, 1024, 64], dtype=torch.float16, device='cuda'),
    torch.randn([32, 1024, 64], dtype=torch.float16, device='cuda'),
    torch.randn([32, 64, 1024], dtype=torch.float16, device='cuda'),
    torch.randn(262144, dtype=torch.float16, device='cuda').as_strided([512, 512], [1, 512]),  # permute_141
    torch.randn(262144, dtype=torch.float16, device='cuda').as_strided([512, 512], [1, 512]),  # permute_139
    torch.randn(262144, dtype=torch.float16, device='cuda').as_strided([512, 512], [1, 512]),  # permute_137
    torch.randn(262144, dtype=torch.float16, device='cuda').as_strided([512, 512], [1, 512]),  # permute_136
    torch.randn([32, 1024, 1024], dtype=torch.float16, device='cuda'),
    torch.randn([32, 1024, 64], dtype=torch.float16, device='cuda'),
    torch.randn([32, 1024, 64], dtype=torch.float16, device='cuda'),
    torch.randn([32, 64, 1024], dtype=torch.float16, device='cuda'),
    torch.randn(262144, dtype=torch.float16, device='cuda').as_strided([512, 512], [1, 512]),  # permute_132
    torch.randn(262144, dtype=torch.float16, device='cuda').as_strided([512, 512], [1, 512]),  # permute_130
    torch.randn(262144, dtype=torch.float16, device='cuda').as_strided([512, 512], [1, 512]),  # permute_128
    torch.randn(1048576, dtype=torch.float16, device='cuda').as_strided([2048, 512], [1, 2048]),  # permute_127
    torch.randn([4, 1024, 2048], dtype=torch.float16, device='cuda'),
    torch.randn(1048576, dtype=torch.float16, device='cuda').as_strided([512, 2048], [1, 512]),  # permute_126
    torch.randn(262144, dtype=torch.float16, device='cuda').as_strided([512, 512], [1, 512]),  # permute_125
    torch.randn([32, 1024, 1024], dtype=torch.float16, device='cuda'),
    torch.randn([32, 1024, 64], dtype=torch.float16, device='cuda'),
    torch.randn([32, 1024, 64], dtype=torch.float16, device='cuda'),
    torch.randn([32, 64, 1024], dtype=torch.float16, device='cuda'),
    torch.randn(262144, dtype=torch.float16, device='cuda').as_strided([512, 512], [1, 512]),  # permute_121
    torch.randn(262144, dtype=torch.float16, device='cuda').as_strided([512, 512], [1, 512]),  # permute_119
    torch.randn(262144, dtype=torch.float16, device='cuda').as_strided([512, 512], [1, 512]),  # permute_117
    torch.randn(262144, dtype=torch.float16, device='cuda').as_strided([512, 512], [1, 512]),  # permute_116
    torch.randn([32, 1024, 1024], dtype=torch.float16, device='cuda'),
    torch.randn([32, 1024, 64], dtype=torch.float16, device='cuda'),
    torch.randn([32, 1024, 64], dtype=torch.float16, device='cuda'),
    torch.randn([32, 64, 1024], dtype=torch.float16, device='cuda'),
    torch.randn(262144, dtype=torch.float16, device='cuda').as_strided([512, 512], [1, 512]),  # permute_112
    torch.randn(262144, dtype=torch.float16, device='cuda').as_strided([512, 512], [1, 512]),  # permute_110
    torch.randn(262144, dtype=torch.float16, device='cuda').as_strided([512, 512], [1, 512]),  # permute_108
    torch.randn(1048576, dtype=torch.float16, device='cuda').as_strided([2048, 512], [1, 2048]),  # permute_107
    torch.randn([4, 1024, 2048], dtype=torch.float16, device='cuda'),
    torch.randn(1048576, dtype=torch.float16, device='cuda').as_strided([512, 2048], [1, 512]),  # permute_106
    torch.randn(262144, dtype=torch.float16, device='cuda').as_strided([512, 512], [1, 512]),  # permute_105
    torch.randn([32, 1024, 1024], dtype=torch.float16, device='cuda'),
    torch.randn([32, 1024, 64], dtype=torch.float16, device='cuda'),
    torch.randn([32, 1024, 64], dtype=torch.float16, device='cuda'),
    torch.randn([32, 64, 1024], dtype=torch.float16, device='cuda'),
    torch.randn(262144, dtype=torch.float16, device='cuda').as_strided([512, 512], [1, 512]),  # permute_101
    torch.randn(262144, dtype=torch.float16, device='cuda').as_strided([512, 512], [1, 512]),  # permute_99
    torch.randn(262144, dtype=torch.float16, device='cuda').as_strided([512, 512], [1, 512]),  # permute_97
    torch.randn(262144, dtype=torch.float16, device='cuda').as_strided([512, 512], [1, 512]),  # permute_96
    torch.randn([32, 1024, 1024], dtype=torch.float16, device='cuda'),
    torch.randn([32, 1024, 64], dtype=torch.float16, device='cuda'),
    torch.randn([32, 1024, 64], dtype=torch.float16, device='cuda'),
    torch.randn([32, 64, 1024], dtype=torch.float16, device='cuda'),
    torch.randn(262144, dtype=torch.float16, device='cuda').as_strided([512, 512], [1, 512]),  # permute_92
    torch.randn(262144, dtype=torch.float16, device='cuda').as_strided([512, 512], [1, 512]),  # permute_90
    torch.randn(262144, dtype=torch.float16, device='cuda').as_strided([512, 512], [1, 512]),  # permute_88
    torch.randn(1048576, dtype=torch.float16, device='cuda').as_strided([2048, 512], [1, 2048]),  # permute_87
    torch.randn([4, 1024, 2048], dtype=torch.float16, device='cuda'),
    torch.randn(1048576, dtype=torch.float16, device='cuda').as_strided([512, 2048], [1, 512]),  # permute_86
    torch.randn(262144, dtype=torch.float16, device='cuda').as_strided([512, 512], [1, 512]),  # permute_85
    torch.randn([32, 1024, 1024], dtype=torch.float16, device='cuda'),
    torch.randn([32, 1024, 64], dtype=torch.float16, device='cuda'),
    torch.randn([32, 1024, 64], dtype=torch.float16, device='cuda'),
    torch.randn([32, 64, 1024], dtype=torch.float16, device='cuda'),
    torch.randn(262144, dtype=torch.float16, device='cuda').as_strided([512, 512], [1, 512]),  # permute_81
    torch.randn(262144, dtype=torch.float16, device='cuda').as_strided([512, 512], [1, 512]),  # permute_79
    torch.randn(262144, dtype=torch.float16, device='cuda').as_strided([512, 512], [1, 512]),  # permute_77
    torch.randn(262144, dtype=torch.float16, device='cuda').as_strided([512, 512], [1, 512]),  # permute_76
    torch.randn([32, 1024, 1024], dtype=torch.float16, device='cuda'),
    torch.randn([32, 1024, 64], dtype=torch.float16, device='cuda'),
    torch.randn([32, 1024, 64], dtype=torch.float16, device='cuda'),
    torch.randn([32, 64, 1024], dtype=torch.float16, device='cuda'),
    torch.randn(262144, dtype=torch.float16, device='cuda').as_strided([512, 512], [1, 512]),  # permute_71
    torch.randn(262144, dtype=torch.float16, device='cuda').as_strided([512, 512], [1, 512]),  # permute_69
    torch.randn(262144, dtype=torch.float16, device='cuda').as_strided([512, 512], [1, 512]),  # permute_67
    torch.randn(1048576, dtype=torch.float16, device='cuda').as_strided([2048, 512], [1, 2048]),  # permute_66
    torch.randn([4, 1024, 2048], dtype=torch.float16, device='cuda'),
    torch.randn(1048576, dtype=torch.float16, device='cuda').as_strided([512, 2048], [1, 512]),  # permute_65
    torch.randn(262144, dtype=torch.float16, device='cuda').as_strided([512, 512], [1, 512]),  # permute_64
    torch.randn([32, 1024, 1024], dtype=torch.float16, device='cuda'),
    torch.randn([32, 1024, 64], dtype=torch.float16, device='cuda'),
    torch.randn([32, 1024, 64], dtype=torch.float16, device='cuda'),
    torch.randn([32, 64, 1024], dtype=torch.float16, device='cuda'),
    torch.randn(262144, dtype=torch.float16, device='cuda').as_strided([512, 512], [1, 512]),  # permute_60
    torch.randn(262144, dtype=torch.float16, device='cuda').as_strided([512, 512], [1, 512]),  # permute_58
    torch.randn(262144, dtype=torch.float16, device='cuda').as_strided([512, 512], [1, 512]),  # permute_56
    torch.randn(1048576, dtype=torch.float16, device='cuda').as_strided([2048, 512], [1, 2048]),  # permute_55
    torch.randn([4, 1024, 2048], dtype=torch.float16, device='cuda'),
    torch.randn(1048576, dtype=torch.float16, device='cuda').as_strided([512, 2048], [1, 512]),  # permute_54
    torch.randn(262144, dtype=torch.float16, device='cuda').as_strided([512, 512], [1, 512]),  # permute_53
    torch.randn([32, 1024, 1024], dtype=torch.float16, device='cuda'),
    torch.randn([32, 1024, 64], dtype=torch.float16, device='cuda'),
    torch.randn([32, 1024, 64], dtype=torch.float16, device='cuda'),
    torch.randn([32, 64, 1024], dtype=torch.float16, device='cuda'),
    torch.randn(262144, dtype=torch.float16, device='cuda').as_strided([512, 512], [1, 512]),  # permute_49
    torch.randn(262144, dtype=torch.float16, device='cuda').as_strided([512, 512], [1, 512]),  # permute_47
    torch.randn(262144, dtype=torch.float16, device='cuda').as_strided([512, 512], [1, 512]),  # permute_45
    torch.randn(1048576, dtype=torch.float16, device='cuda').as_strided([2048, 512], [1, 2048]),  # permute_44
    torch.randn([4, 1024, 2048], dtype=torch.float16, device='cuda'),
    torch.randn(1048576, dtype=torch.float16, device='cuda').as_strided([512, 2048], [1, 512]),  # permute_43
    torch.randn(262144, dtype=torch.float16, device='cuda').as_strided([512, 512], [1, 512]),  # permute_42
    torch.randn([32, 1024, 1024], dtype=torch.float16, device='cuda'),
    torch.randn([32, 1024, 64], dtype=torch.float16, device='cuda'),
    torch.randn([32, 1024, 64], dtype=torch.float16, device='cuda'),
    torch.randn([32, 64, 1024], dtype=torch.float16, device='cuda'),
    torch.randn(262144, dtype=torch.float16, device='cuda').as_strided([512, 512], [1, 512]),  # permute_38
    torch.randn(262144, dtype=torch.float16, device='cuda').as_strided([512, 512], [1, 512]),  # permute_36
    torch.randn(262144, dtype=torch.float16, device='cuda').as_strided([512, 512], [1, 512]),  # permute_34
    torch.randn(1048576, dtype=torch.float16, device='cuda').as_strided([2048, 512], [1, 2048]),  # permute_33
    torch.randn([4, 1024, 2048], dtype=torch.float16, device='cuda'),
    torch.randn(1048576, dtype=torch.float16, device='cuda').as_strided([512, 2048], [1, 512]),  # permute_32
    torch.randn(262144, dtype=torch.float16, device='cuda').as_strided([512, 512], [1, 512]),  # permute_31
    torch.randn([32, 1024, 1024], dtype=torch.float16, device='cuda'),
    torch.randn([32, 1024, 64], dtype=torch.float16, device='cuda'),
    torch.randn([32, 1024, 64], dtype=torch.float16, device='cuda'),
    torch.randn([32, 64, 1024], dtype=torch.float16, device='cuda'),
    torch.randn(262144, dtype=torch.float16, device='cuda').as_strided([512, 512], [1, 512]),  # permute_27
    torch.randn(262144, dtype=torch.float16, device='cuda').as_strided([512, 512], [1, 512]),  # permute_25
    torch.randn(262144, dtype=torch.float16, device='cuda').as_strided([512, 512], [1, 512]),  # permute_23
    torch.randn(1048576, dtype=torch.float16, device='cuda').as_strided([2048, 512], [1, 2048]),  # permute_22
    torch.randn([4, 1024, 2048], dtype=torch.float16, device='cuda'),
    torch.randn(1048576, dtype=torch.float16, device='cuda').as_strided([512, 2048], [1, 512]),  # permute_21
    torch.randn(262144, dtype=torch.float16, device='cuda').as_strided([512, 512], [1, 512]),  # permute_20
    torch.randn([32, 1024, 1024], dtype=torch.float16, device='cuda'),
    torch.randn([32, 1024, 64], dtype=torch.float16, device='cuda'),
    torch.randn([32, 1024, 64], dtype=torch.float16, device='cuda'),
    torch.randn([32, 64, 1024], dtype=torch.float16, device='cuda'),
    torch.randn(262144, dtype=torch.float16, device='cuda').as_strided([512, 512], [1, 512]),  # permute_16
    torch.randn(262144, dtype=torch.float16, device='cuda').as_strided([512, 512], [1, 512]),  # permute_14
    torch.randn(262144, dtype=torch.float16, device='cuda').as_strided([512, 512], [1, 512]),  # permute_12
    torch.randn(1048576, dtype=torch.float16, device='cuda').as_strided([2048, 512], [1, 2048]),  # permute_11
    torch.randn([4, 1024, 2048], dtype=torch.float16, device='cuda'),
    torch.randn(1048576, dtype=torch.float16, device='cuda').as_strided([512, 2048], [1, 512]),  # permute_10
    torch.randn(262144, dtype=torch.float16, device='cuda').as_strided([512, 512], [1, 512]),  # permute_9
    torch.randn([32, 1024, 1024], dtype=torch.float16, device='cuda'),
    torch.randn([32, 1024, 64], dtype=torch.float16, device='cuda'),
    torch.randn([32, 1024, 64], dtype=torch.float16, device='cuda'),
    torch.randn([32, 64, 1024], dtype=torch.float16, device='cuda'),
    torch.randn(262144, dtype=torch.float16, device='cuda').as_strided([512, 512], [1, 512]),  # permute_4
    torch.randn(262144, dtype=torch.float16, device='cuda').as_strided([512, 512], [1, 512]),  # permute_2
    torch.randn(262144, dtype=torch.float16, device='cuda').as_strided([512, 512], [1, 512]),  # permute
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
