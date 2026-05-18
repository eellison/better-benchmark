"""
Standalone repro captured via capture_hook.
Label: tlparse_huggingface_s3_g21
Pattern hash: ce4456d5f22d
Shape hash: a18fe07c
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
    def forward(self, arg210_1: "i64[1, 512]", addmm_73: "f16[512, 30522]", permute_133: "f16[768, 30522]", permute_132: "f16[768, 768]", rsqrt_24: "f32[1, 512, 1]", permute_131: "f16[3072, 768]", permute_130: "f16[768, 3072]", rsqrt_23: "f32[1, 512, 1]", permute_129: "f16[768, 768]", view_230: "f16[12, 512, 512]", view_231: "f16[12, 512, 64]", view_227: "f16[12, 512, 64]", view_228: "f16[12, 64, 512]", permute_125: "f16[768, 768]", permute_123: "f16[768, 768]", permute_121: "f16[768, 768]", rsqrt_22: "f32[1, 512, 1]", permute_120: "f16[3072, 768]", permute_119: "f16[768, 3072]", rsqrt_21: "f32[1, 512, 1]", permute_118: "f16[768, 768]", view_210: "f16[12, 512, 512]", view_211: "f16[12, 512, 64]", view_207: "f16[12, 512, 64]", view_208: "f16[12, 64, 512]", permute_114: "f16[768, 768]", permute_112: "f16[768, 768]", permute_110: "f16[768, 768]", rsqrt_20: "f32[1, 512, 1]", permute_109: "f16[3072, 768]", permute_108: "f16[768, 3072]", rsqrt_19: "f32[1, 512, 1]", permute_107: "f16[768, 768]", view_190: "f16[12, 512, 512]", view_191: "f16[12, 512, 64]", view_187: "f16[12, 512, 64]", view_188: "f16[12, 64, 512]", permute_103: "f16[768, 768]", permute_101: "f16[768, 768]", permute_99: "f16[768, 768]", rsqrt_18: "f32[1, 512, 1]", permute_98: "f16[3072, 768]", permute_97: "f16[768, 3072]", rsqrt_17: "f32[1, 512, 1]", permute_96: "f16[768, 768]", view_170: "f16[12, 512, 512]", view_171: "f16[12, 512, 64]", view_167: "f16[12, 512, 64]", view_168: "f16[12, 64, 512]", permute_92: "f16[768, 768]", permute_90: "f16[768, 768]", permute_88: "f16[768, 768]", rsqrt_16: "f32[1, 512, 1]", permute_87: "f16[3072, 768]", permute_86: "f16[768, 3072]", rsqrt_15: "f32[1, 512, 1]", permute_85: "f16[768, 768]", view_150: "f16[12, 512, 512]", view_151: "f16[12, 512, 64]", view_147: "f16[12, 512, 64]", view_148: "f16[12, 64, 512]", permute_81: "f16[768, 768]", permute_79: "f16[768, 768]", permute_77: "f16[768, 768]", rsqrt_14: "f32[1, 512, 1]", permute_76: "f16[3072, 768]", permute_75: "f16[768, 3072]", rsqrt_13: "f32[1, 512, 1]", permute_74: "f16[768, 768]", view_130: "f16[12, 512, 512]", view_131: "f16[12, 512, 64]", view_127: "f16[12, 512, 64]", view_128: "f16[12, 64, 512]", permute_70: "f16[768, 768]", permute_68: "f16[768, 768]", permute_66: "f16[768, 768]", rsqrt_12: "f32[1, 512, 1]", permute_65: "f16[3072, 768]", permute_64: "f16[768, 3072]", rsqrt_11: "f32[1, 512, 1]", permute_63: "f16[768, 768]", view_110: "f16[12, 512, 512]", view_111: "f16[12, 512, 64]", view_107: "f16[12, 512, 64]", view_108: "f16[12, 64, 512]", permute_59: "f16[768, 768]", permute_57: "f16[768, 768]", permute_55: "f16[768, 768]", rsqrt_10: "f32[1, 512, 1]", permute_54: "f16[3072, 768]", permute_53: "f16[768, 3072]", rsqrt_9: "f32[1, 512, 1]", permute_52: "f16[768, 768]", view_90: "f16[12, 512, 512]", view_91: "f16[12, 512, 64]", view_87: "f16[12, 512, 64]", view_88: "f16[12, 64, 512]", permute_48: "f16[768, 768]", permute_46: "f16[768, 768]", permute_44: "f16[768, 768]", rsqrt_8: "f32[1, 512, 1]", permute_43: "f16[3072, 768]", permute_42: "f16[768, 3072]", rsqrt_7: "f32[1, 512, 1]", permute_41: "f16[768, 768]", view_70: "f16[12, 512, 512]", view_71: "f16[12, 512, 64]", view_67: "f16[12, 512, 64]", view_68: "f16[12, 64, 512]", permute_37: "f16[768, 768]", permute_35: "f16[768, 768]", permute_33: "f16[768, 768]", rsqrt_6: "f32[1, 512, 1]", permute_32: "f16[3072, 768]", permute_31: "f16[768, 3072]", rsqrt_5: "f32[1, 512, 1]", permute_30: "f16[768, 768]", view_50: "f16[12, 512, 512]", view_51: "f16[12, 512, 64]", view_47: "f16[12, 512, 64]", view_48: "f16[12, 64, 512]", permute_26: "f16[768, 768]", permute_24: "f16[768, 768]", permute_22: "f16[768, 768]", rsqrt_4: "f32[1, 512, 1]", permute_21: "f16[3072, 768]", permute_20: "f16[768, 3072]", rsqrt_3: "f32[1, 512, 1]", permute_19: "f16[768, 768]", view_30: "f16[12, 512, 512]", view_31: "f16[12, 512, 64]", view_27: "f16[12, 512, 64]", view_28: "f16[12, 64, 512]", permute_15: "f16[768, 768]", permute_13: "f16[768, 768]", permute_11: "f16[768, 768]", rsqrt_2: "f32[1, 512, 1]", permute_10: "f16[3072, 768]", permute_9: "f16[768, 3072]", rsqrt_1: "f32[1, 512, 1]", permute_8: "f16[768, 768]", view_10: "f16[12, 512, 512]", view_11: "f16[12, 512, 64]", view_7: "f16[12, 512, 64]", view_8: "f16[12, 64, 512]", permute_4: "f16[768, 768]", permute_2: "f16[768, 768]", permute: "f16[768, 768]", rsqrt: "f32[1, 512, 1]"):
        # No stacktrace found for following nodes
        reshape_default: "i64[512]" = torch.ops.aten.reshape.default(arg210_1, [-1]);  arg210_1 = None
        convert_element_type_default: "f32[512, 30522]" = torch.ops.prims.convert_element_type.default(addmm_73, torch.float32);  addmm_73 = None
        amax_default: "f32[512, 1]" = torch.ops.aten.amax.default(convert_element_type_default, [1], True)
        sub_tensor: "f32[512, 30522]" = torch.ops.aten.sub.Tensor(convert_element_type_default, amax_default);  convert_element_type_default = amax_default = None
        exp_default: "f32[512, 30522]" = torch.ops.aten.exp.default(sub_tensor)
        sum_dim_int_list: "f32[512, 1]" = torch.ops.aten.sum.dim_IntList(exp_default, [1], True);  exp_default = None
        log_default: "f32[512, 1]" = torch.ops.aten.log.default(sum_dim_int_list);  sum_dim_int_list = None
        sub_tensor_1: "f32[512, 30522]" = torch.ops.aten.sub.Tensor(sub_tensor, log_default);  sub_tensor = log_default = None
        convert_element_type_default_1: "f16[512, 30522]" = torch.ops.prims.convert_element_type.default(sub_tensor_1, torch.float16);  sub_tensor_1 = None
        convert_element_type_default_2: "f32[512, 30522]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_1, torch.float32);  convert_element_type_default_1 = None
        ne_scalar: "b8[512]" = torch.ops.aten.ne.Scalar(reshape_default, -100)
        full_default: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "i64[512]" = torch.ops.aten.where.self(ne_scalar, reshape_default, full_default);  reshape_default = full_default = None
        unsqueeze_default: "i64[512, 1]" = torch.ops.aten.unsqueeze.default(where_self, 1);  where_self = None
        gather_default: "f32[512, 1]" = torch.ops.aten.gather.default(convert_element_type_default_2, 1, unsqueeze_default);  convert_element_type_default_2 = unsqueeze_default = None
        squeeze_dim: "f32[512]" = torch.ops.aten.squeeze.dim(gather_default, 1);  gather_default = None
        neg_default: "f32[512]" = torch.ops.aten.neg.default(squeeze_dim);  squeeze_dim = None
        full_default_1: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_1: "f32[512]" = torch.ops.aten.where.self(ne_scalar, neg_default, full_default_1);  neg_default = full_default_1 = None
        sum_default: "i64[]" = torch.ops.aten.sum.default(ne_scalar);  ne_scalar = None
        convert_element_type_default_3: "f32[]" = torch.ops.prims.convert_element_type.default(sum_default, torch.float32);  sum_default = None
        sum_default_1: "f32[]" = torch.ops.aten.sum.default(where_self_1);  where_self_1 = None
        div_tensor: "f32[]" = torch.ops.aten.div.Tensor(sum_default_1, convert_element_type_default_3);  sum_default_1 = convert_element_type_default_3 = None
        permute_default: "f16[30522, 768]" = torch.ops.aten.permute.default(permute_133, [1, 0]);  permute_133 = None
        permute_default_1: "f16[768, 768]" = torch.ops.aten.permute.default(permute_132, [1, 0]);  permute_132 = None
        div_tensor_1: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_24, 768);  rsqrt_24 = None
        permute_default_2: "f16[768, 3072]" = torch.ops.aten.permute.default(permute_131, [1, 0]);  permute_131 = None
        permute_default_3: "f16[3072, 768]" = torch.ops.aten.permute.default(permute_130, [1, 0]);  permute_130 = None
        div_tensor_2: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_23, 768);  rsqrt_23 = None
        permute_default_4: "f16[768, 768]" = torch.ops.aten.permute.default(permute_129, [1, 0]);  permute_129 = None
        permute_default_5: "f16[12, 512, 512]" = torch.ops.aten.permute.default(view_230, [0, 2, 1]);  view_230 = None
        permute_default_6: "f16[12, 64, 512]" = torch.ops.aten.permute.default(view_231, [0, 2, 1]);  view_231 = None
        permute_default_7: "f16[12, 64, 512]" = torch.ops.aten.permute.default(view_227, [0, 2, 1]);  view_227 = None
        permute_default_8: "f16[12, 512, 64]" = torch.ops.aten.permute.default(view_228, [0, 2, 1]);  view_228 = None
        permute_default_9: "f16[768, 768]" = torch.ops.aten.permute.default(permute_125, [1, 0]);  permute_125 = None
        permute_default_10: "f16[768, 768]" = torch.ops.aten.permute.default(permute_123, [1, 0]);  permute_123 = None
        permute_default_11: "f16[768, 768]" = torch.ops.aten.permute.default(permute_121, [1, 0]);  permute_121 = None
        div_tensor_3: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_22, 768);  rsqrt_22 = None
        permute_default_12: "f16[768, 3072]" = torch.ops.aten.permute.default(permute_120, [1, 0]);  permute_120 = None
        permute_default_13: "f16[3072, 768]" = torch.ops.aten.permute.default(permute_119, [1, 0]);  permute_119 = None
        div_tensor_4: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_21, 768);  rsqrt_21 = None
        permute_default_14: "f16[768, 768]" = torch.ops.aten.permute.default(permute_118, [1, 0]);  permute_118 = None
        permute_default_15: "f16[12, 512, 512]" = torch.ops.aten.permute.default(view_210, [0, 2, 1]);  view_210 = None
        permute_default_16: "f16[12, 64, 512]" = torch.ops.aten.permute.default(view_211, [0, 2, 1]);  view_211 = None
        permute_default_17: "f16[12, 64, 512]" = torch.ops.aten.permute.default(view_207, [0, 2, 1]);  view_207 = None
        permute_default_18: "f16[12, 512, 64]" = torch.ops.aten.permute.default(view_208, [0, 2, 1]);  view_208 = None
        permute_default_19: "f16[768, 768]" = torch.ops.aten.permute.default(permute_114, [1, 0]);  permute_114 = None
        permute_default_20: "f16[768, 768]" = torch.ops.aten.permute.default(permute_112, [1, 0]);  permute_112 = None
        permute_default_21: "f16[768, 768]" = torch.ops.aten.permute.default(permute_110, [1, 0]);  permute_110 = None
        div_tensor_5: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_20, 768);  rsqrt_20 = None
        permute_default_22: "f16[768, 3072]" = torch.ops.aten.permute.default(permute_109, [1, 0]);  permute_109 = None
        permute_default_23: "f16[3072, 768]" = torch.ops.aten.permute.default(permute_108, [1, 0]);  permute_108 = None
        div_tensor_6: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_19, 768);  rsqrt_19 = None
        permute_default_24: "f16[768, 768]" = torch.ops.aten.permute.default(permute_107, [1, 0]);  permute_107 = None
        permute_default_25: "f16[12, 512, 512]" = torch.ops.aten.permute.default(view_190, [0, 2, 1]);  view_190 = None
        permute_default_26: "f16[12, 64, 512]" = torch.ops.aten.permute.default(view_191, [0, 2, 1]);  view_191 = None
        permute_default_27: "f16[12, 64, 512]" = torch.ops.aten.permute.default(view_187, [0, 2, 1]);  view_187 = None
        permute_default_28: "f16[12, 512, 64]" = torch.ops.aten.permute.default(view_188, [0, 2, 1]);  view_188 = None
        permute_default_29: "f16[768, 768]" = torch.ops.aten.permute.default(permute_103, [1, 0]);  permute_103 = None
        permute_default_30: "f16[768, 768]" = torch.ops.aten.permute.default(permute_101, [1, 0]);  permute_101 = None
        permute_default_31: "f16[768, 768]" = torch.ops.aten.permute.default(permute_99, [1, 0]);  permute_99 = None
        div_tensor_7: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_18, 768);  rsqrt_18 = None
        permute_default_32: "f16[768, 3072]" = torch.ops.aten.permute.default(permute_98, [1, 0]);  permute_98 = None
        permute_default_33: "f16[3072, 768]" = torch.ops.aten.permute.default(permute_97, [1, 0]);  permute_97 = None
        div_tensor_8: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_17, 768);  rsqrt_17 = None
        permute_default_34: "f16[768, 768]" = torch.ops.aten.permute.default(permute_96, [1, 0]);  permute_96 = None
        permute_default_35: "f16[12, 512, 512]" = torch.ops.aten.permute.default(view_170, [0, 2, 1]);  view_170 = None
        permute_default_36: "f16[12, 64, 512]" = torch.ops.aten.permute.default(view_171, [0, 2, 1]);  view_171 = None
        permute_default_37: "f16[12, 64, 512]" = torch.ops.aten.permute.default(view_167, [0, 2, 1]);  view_167 = None
        permute_default_38: "f16[12, 512, 64]" = torch.ops.aten.permute.default(view_168, [0, 2, 1]);  view_168 = None
        permute_default_39: "f16[768, 768]" = torch.ops.aten.permute.default(permute_92, [1, 0]);  permute_92 = None
        permute_default_40: "f16[768, 768]" = torch.ops.aten.permute.default(permute_90, [1, 0]);  permute_90 = None
        permute_default_41: "f16[768, 768]" = torch.ops.aten.permute.default(permute_88, [1, 0]);  permute_88 = None
        div_tensor_9: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_16, 768);  rsqrt_16 = None
        permute_default_42: "f16[768, 3072]" = torch.ops.aten.permute.default(permute_87, [1, 0]);  permute_87 = None
        permute_default_43: "f16[3072, 768]" = torch.ops.aten.permute.default(permute_86, [1, 0]);  permute_86 = None
        div_tensor_10: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_15, 768);  rsqrt_15 = None
        permute_default_44: "f16[768, 768]" = torch.ops.aten.permute.default(permute_85, [1, 0]);  permute_85 = None
        permute_default_45: "f16[12, 512, 512]" = torch.ops.aten.permute.default(view_150, [0, 2, 1]);  view_150 = None
        permute_default_46: "f16[12, 64, 512]" = torch.ops.aten.permute.default(view_151, [0, 2, 1]);  view_151 = None
        permute_default_47: "f16[12, 64, 512]" = torch.ops.aten.permute.default(view_147, [0, 2, 1]);  view_147 = None
        permute_default_48: "f16[12, 512, 64]" = torch.ops.aten.permute.default(view_148, [0, 2, 1]);  view_148 = None
        permute_default_49: "f16[768, 768]" = torch.ops.aten.permute.default(permute_81, [1, 0]);  permute_81 = None
        permute_default_50: "f16[768, 768]" = torch.ops.aten.permute.default(permute_79, [1, 0]);  permute_79 = None
        permute_default_51: "f16[768, 768]" = torch.ops.aten.permute.default(permute_77, [1, 0]);  permute_77 = None
        div_tensor_11: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_14, 768);  rsqrt_14 = None
        permute_default_52: "f16[768, 3072]" = torch.ops.aten.permute.default(permute_76, [1, 0]);  permute_76 = None
        permute_default_53: "f16[3072, 768]" = torch.ops.aten.permute.default(permute_75, [1, 0]);  permute_75 = None
        div_tensor_12: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_13, 768);  rsqrt_13 = None
        permute_default_54: "f16[768, 768]" = torch.ops.aten.permute.default(permute_74, [1, 0]);  permute_74 = None
        permute_default_55: "f16[12, 512, 512]" = torch.ops.aten.permute.default(view_130, [0, 2, 1]);  view_130 = None
        permute_default_56: "f16[12, 64, 512]" = torch.ops.aten.permute.default(view_131, [0, 2, 1]);  view_131 = None
        permute_default_57: "f16[12, 64, 512]" = torch.ops.aten.permute.default(view_127, [0, 2, 1]);  view_127 = None
        permute_default_58: "f16[12, 512, 64]" = torch.ops.aten.permute.default(view_128, [0, 2, 1]);  view_128 = None
        permute_default_59: "f16[768, 768]" = torch.ops.aten.permute.default(permute_70, [1, 0]);  permute_70 = None
        permute_default_60: "f16[768, 768]" = torch.ops.aten.permute.default(permute_68, [1, 0]);  permute_68 = None
        permute_default_61: "f16[768, 768]" = torch.ops.aten.permute.default(permute_66, [1, 0]);  permute_66 = None
        div_tensor_13: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_12, 768);  rsqrt_12 = None
        permute_default_62: "f16[768, 3072]" = torch.ops.aten.permute.default(permute_65, [1, 0]);  permute_65 = None
        permute_default_63: "f16[3072, 768]" = torch.ops.aten.permute.default(permute_64, [1, 0]);  permute_64 = None
        div_tensor_14: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_11, 768);  rsqrt_11 = None
        permute_default_64: "f16[768, 768]" = torch.ops.aten.permute.default(permute_63, [1, 0]);  permute_63 = None
        permute_default_65: "f16[12, 512, 512]" = torch.ops.aten.permute.default(view_110, [0, 2, 1]);  view_110 = None
        permute_default_66: "f16[12, 64, 512]" = torch.ops.aten.permute.default(view_111, [0, 2, 1]);  view_111 = None
        permute_default_67: "f16[12, 64, 512]" = torch.ops.aten.permute.default(view_107, [0, 2, 1]);  view_107 = None
        permute_default_68: "f16[12, 512, 64]" = torch.ops.aten.permute.default(view_108, [0, 2, 1]);  view_108 = None
        permute_default_69: "f16[768, 768]" = torch.ops.aten.permute.default(permute_59, [1, 0]);  permute_59 = None
        permute_default_70: "f16[768, 768]" = torch.ops.aten.permute.default(permute_57, [1, 0]);  permute_57 = None
        permute_default_71: "f16[768, 768]" = torch.ops.aten.permute.default(permute_55, [1, 0]);  permute_55 = None
        div_tensor_15: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_10, 768);  rsqrt_10 = None
        permute_default_72: "f16[768, 3072]" = torch.ops.aten.permute.default(permute_54, [1, 0]);  permute_54 = None
        permute_default_73: "f16[3072, 768]" = torch.ops.aten.permute.default(permute_53, [1, 0]);  permute_53 = None
        div_tensor_16: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_9, 768);  rsqrt_9 = None
        permute_default_74: "f16[768, 768]" = torch.ops.aten.permute.default(permute_52, [1, 0]);  permute_52 = None
        permute_default_75: "f16[12, 512, 512]" = torch.ops.aten.permute.default(view_90, [0, 2, 1]);  view_90 = None
        permute_default_76: "f16[12, 64, 512]" = torch.ops.aten.permute.default(view_91, [0, 2, 1]);  view_91 = None
        permute_default_77: "f16[12, 64, 512]" = torch.ops.aten.permute.default(view_87, [0, 2, 1]);  view_87 = None
        permute_default_78: "f16[12, 512, 64]" = torch.ops.aten.permute.default(view_88, [0, 2, 1]);  view_88 = None
        permute_default_79: "f16[768, 768]" = torch.ops.aten.permute.default(permute_48, [1, 0]);  permute_48 = None
        permute_default_80: "f16[768, 768]" = torch.ops.aten.permute.default(permute_46, [1, 0]);  permute_46 = None
        permute_default_81: "f16[768, 768]" = torch.ops.aten.permute.default(permute_44, [1, 0]);  permute_44 = None
        div_tensor_17: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_8, 768);  rsqrt_8 = None
        permute_default_82: "f16[768, 3072]" = torch.ops.aten.permute.default(permute_43, [1, 0]);  permute_43 = None
        permute_default_83: "f16[3072, 768]" = torch.ops.aten.permute.default(permute_42, [1, 0]);  permute_42 = None
        div_tensor_18: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_7, 768);  rsqrt_7 = None
        permute_default_84: "f16[768, 768]" = torch.ops.aten.permute.default(permute_41, [1, 0]);  permute_41 = None
        permute_default_85: "f16[12, 512, 512]" = torch.ops.aten.permute.default(view_70, [0, 2, 1]);  view_70 = None
        permute_default_86: "f16[12, 64, 512]" = torch.ops.aten.permute.default(view_71, [0, 2, 1]);  view_71 = None
        permute_default_87: "f16[12, 64, 512]" = torch.ops.aten.permute.default(view_67, [0, 2, 1]);  view_67 = None
        permute_default_88: "f16[12, 512, 64]" = torch.ops.aten.permute.default(view_68, [0, 2, 1]);  view_68 = None
        permute_default_89: "f16[768, 768]" = torch.ops.aten.permute.default(permute_37, [1, 0]);  permute_37 = None
        permute_default_90: "f16[768, 768]" = torch.ops.aten.permute.default(permute_35, [1, 0]);  permute_35 = None
        permute_default_91: "f16[768, 768]" = torch.ops.aten.permute.default(permute_33, [1, 0]);  permute_33 = None
        div_tensor_19: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_6, 768);  rsqrt_6 = None
        permute_default_92: "f16[768, 3072]" = torch.ops.aten.permute.default(permute_32, [1, 0]);  permute_32 = None
        permute_default_93: "f16[3072, 768]" = torch.ops.aten.permute.default(permute_31, [1, 0]);  permute_31 = None
        div_tensor_20: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_5, 768);  rsqrt_5 = None
        permute_default_94: "f16[768, 768]" = torch.ops.aten.permute.default(permute_30, [1, 0]);  permute_30 = None
        permute_default_95: "f16[12, 512, 512]" = torch.ops.aten.permute.default(view_50, [0, 2, 1]);  view_50 = None
        permute_default_96: "f16[12, 64, 512]" = torch.ops.aten.permute.default(view_51, [0, 2, 1]);  view_51 = None
        permute_default_97: "f16[12, 64, 512]" = torch.ops.aten.permute.default(view_47, [0, 2, 1]);  view_47 = None
        permute_default_98: "f16[12, 512, 64]" = torch.ops.aten.permute.default(view_48, [0, 2, 1]);  view_48 = None
        permute_default_99: "f16[768, 768]" = torch.ops.aten.permute.default(permute_26, [1, 0]);  permute_26 = None
        permute_default_100: "f16[768, 768]" = torch.ops.aten.permute.default(permute_24, [1, 0]);  permute_24 = None
        permute_default_101: "f16[768, 768]" = torch.ops.aten.permute.default(permute_22, [1, 0]);  permute_22 = None
        div_tensor_21: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_4, 768);  rsqrt_4 = None
        permute_default_102: "f16[768, 3072]" = torch.ops.aten.permute.default(permute_21, [1, 0]);  permute_21 = None
        permute_default_103: "f16[3072, 768]" = torch.ops.aten.permute.default(permute_20, [1, 0]);  permute_20 = None
        div_tensor_22: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_3, 768);  rsqrt_3 = None
        permute_default_104: "f16[768, 768]" = torch.ops.aten.permute.default(permute_19, [1, 0]);  permute_19 = None
        permute_default_105: "f16[12, 512, 512]" = torch.ops.aten.permute.default(view_30, [0, 2, 1]);  view_30 = None
        permute_default_106: "f16[12, 64, 512]" = torch.ops.aten.permute.default(view_31, [0, 2, 1]);  view_31 = None
        permute_default_107: "f16[12, 64, 512]" = torch.ops.aten.permute.default(view_27, [0, 2, 1]);  view_27 = None
        permute_default_108: "f16[12, 512, 64]" = torch.ops.aten.permute.default(view_28, [0, 2, 1]);  view_28 = None
        permute_default_109: "f16[768, 768]" = torch.ops.aten.permute.default(permute_15, [1, 0]);  permute_15 = None
        permute_default_110: "f16[768, 768]" = torch.ops.aten.permute.default(permute_13, [1, 0]);  permute_13 = None
        permute_default_111: "f16[768, 768]" = torch.ops.aten.permute.default(permute_11, [1, 0]);  permute_11 = None
        div_tensor_23: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_2, 768);  rsqrt_2 = None
        permute_default_112: "f16[768, 3072]" = torch.ops.aten.permute.default(permute_10, [1, 0]);  permute_10 = None
        permute_default_113: "f16[3072, 768]" = torch.ops.aten.permute.default(permute_9, [1, 0]);  permute_9 = None
        div_tensor_24: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_1, 768);  rsqrt_1 = None
        permute_default_114: "f16[768, 768]" = torch.ops.aten.permute.default(permute_8, [1, 0]);  permute_8 = None
        permute_default_115: "f16[12, 512, 512]" = torch.ops.aten.permute.default(view_10, [0, 2, 1]);  view_10 = None
        permute_default_116: "f16[12, 64, 512]" = torch.ops.aten.permute.default(view_11, [0, 2, 1]);  view_11 = None
        permute_default_117: "f16[12, 64, 512]" = torch.ops.aten.permute.default(view_7, [0, 2, 1]);  view_7 = None
        permute_default_118: "f16[12, 512, 64]" = torch.ops.aten.permute.default(view_8, [0, 2, 1]);  view_8 = None
        permute_default_119: "f16[768, 768]" = torch.ops.aten.permute.default(permute_4, [1, 0]);  permute_4 = None
        permute_default_120: "f16[768, 768]" = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None
        permute_default_121: "f16[768, 768]" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
        div_tensor_25: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt, 768);  rsqrt = None
        return (div_tensor, permute_default, permute_default_1, div_tensor_1, permute_default_2, permute_default_3, div_tensor_2, permute_default_4, permute_default_5, permute_default_6, permute_default_7, permute_default_8, permute_default_9, permute_default_10, permute_default_11, div_tensor_3, permute_default_12, permute_default_13, div_tensor_4, permute_default_14, permute_default_15, permute_default_16, permute_default_17, permute_default_18, permute_default_19, permute_default_20, permute_default_21, div_tensor_5, permute_default_22, permute_default_23, div_tensor_6, permute_default_24, permute_default_25, permute_default_26, permute_default_27, permute_default_28, permute_default_29, permute_default_30, permute_default_31, div_tensor_7, permute_default_32, permute_default_33, div_tensor_8, permute_default_34, permute_default_35, permute_default_36, permute_default_37, permute_default_38, permute_default_39, permute_default_40, permute_default_41, div_tensor_9, permute_default_42, permute_default_43, div_tensor_10, permute_default_44, permute_default_45, permute_default_46, permute_default_47, permute_default_48, permute_default_49, permute_default_50, permute_default_51, div_tensor_11, permute_default_52, permute_default_53, div_tensor_12, permute_default_54, permute_default_55, permute_default_56, permute_default_57, permute_default_58, permute_default_59, permute_default_60, permute_default_61, div_tensor_13, permute_default_62, permute_default_63, div_tensor_14, permute_default_64, permute_default_65, permute_default_66, permute_default_67, permute_default_68, permute_default_69, permute_default_70, permute_default_71, div_tensor_15, permute_default_72, permute_default_73, div_tensor_16, permute_default_74, permute_default_75, permute_default_76, permute_default_77, permute_default_78, permute_default_79, permute_default_80, permute_default_81, div_tensor_17, permute_default_82, permute_default_83, div_tensor_18, permute_default_84, permute_default_85, permute_default_86, permute_default_87, permute_default_88, permute_default_89, permute_default_90, permute_default_91, div_tensor_19, permute_default_92, permute_default_93, div_tensor_20, permute_default_94, permute_default_95, permute_default_96, permute_default_97, permute_default_98, permute_default_99, permute_default_100, permute_default_101, div_tensor_21, permute_default_102, permute_default_103, div_tensor_22, permute_default_104, permute_default_105, permute_default_106, permute_default_107, permute_default_108, permute_default_109, permute_default_110, permute_default_111, div_tensor_23, permute_default_112, permute_default_113, div_tensor_24, permute_default_114, permute_default_115, permute_default_116, permute_default_117, permute_default_118, permute_default_119, permute_default_120, permute_default_121, div_tensor_25)


def _default_make_inputs():
    return [
    torch.randint(0, 2, [1, 512], dtype=torch.int64, device='cuda'),
    torch.randn([512, 30522], dtype=torch.float16, device='cuda'),
    torch.randn(23440896, dtype=torch.float16, device='cuda').as_strided([768, 30522], [1, 768]),  # permute_133
    torch.randn(589824, dtype=torch.float16, device='cuda').as_strided([768, 768], [1, 768]),  # permute_132
    torch.randn([1, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn(2359296, dtype=torch.float16, device='cuda').as_strided([3072, 768], [1, 3072]),  # permute_131
    torch.randn(2359296, dtype=torch.float16, device='cuda').as_strided([768, 3072], [1, 768]),  # permute_130
    torch.randn([1, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn(589824, dtype=torch.float16, device='cuda').as_strided([768, 768], [1, 768]),  # permute_129
    torch.randn([12, 512, 512], dtype=torch.float16, device='cuda'),
    torch.randn(393216, dtype=torch.float16, device='cuda').as_strided([12, 512, 64], [64, 768, 1]),  # view_231
    torch.randn(393216, dtype=torch.float16, device='cuda').as_strided([12, 512, 64], [64, 768, 1]),  # view_227
    torch.randn(393216, dtype=torch.float16, device='cuda').as_strided([12, 64, 512], [64, 1, 768]),  # view_228
    torch.randn(589824, dtype=torch.float16, device='cuda').as_strided([768, 768], [1, 768]),  # permute_125
    torch.randn(589824, dtype=torch.float16, device='cuda').as_strided([768, 768], [1, 768]),  # permute_123
    torch.randn(589824, dtype=torch.float16, device='cuda').as_strided([768, 768], [1, 768]),  # permute_121
    torch.randn([1, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn(2359296, dtype=torch.float16, device='cuda').as_strided([3072, 768], [1, 3072]),  # permute_120
    torch.randn(2359296, dtype=torch.float16, device='cuda').as_strided([768, 3072], [1, 768]),  # permute_119
    torch.randn([1, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn(589824, dtype=torch.float16, device='cuda').as_strided([768, 768], [1, 768]),  # permute_118
    torch.randn([12, 512, 512], dtype=torch.float16, device='cuda'),
    torch.randn(393216, dtype=torch.float16, device='cuda').as_strided([12, 512, 64], [64, 768, 1]),  # view_211
    torch.randn(393216, dtype=torch.float16, device='cuda').as_strided([12, 512, 64], [64, 768, 1]),  # view_207
    torch.randn(393216, dtype=torch.float16, device='cuda').as_strided([12, 64, 512], [64, 1, 768]),  # view_208
    torch.randn(589824, dtype=torch.float16, device='cuda').as_strided([768, 768], [1, 768]),  # permute_114
    torch.randn(589824, dtype=torch.float16, device='cuda').as_strided([768, 768], [1, 768]),  # permute_112
    torch.randn(589824, dtype=torch.float16, device='cuda').as_strided([768, 768], [1, 768]),  # permute_110
    torch.randn([1, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn(2359296, dtype=torch.float16, device='cuda').as_strided([3072, 768], [1, 3072]),  # permute_109
    torch.randn(2359296, dtype=torch.float16, device='cuda').as_strided([768, 3072], [1, 768]),  # permute_108
    torch.randn([1, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn(589824, dtype=torch.float16, device='cuda').as_strided([768, 768], [1, 768]),  # permute_107
    torch.randn([12, 512, 512], dtype=torch.float16, device='cuda'),
    torch.randn(393216, dtype=torch.float16, device='cuda').as_strided([12, 512, 64], [64, 768, 1]),  # view_191
    torch.randn(393216, dtype=torch.float16, device='cuda').as_strided([12, 512, 64], [64, 768, 1]),  # view_187
    torch.randn(393216, dtype=torch.float16, device='cuda').as_strided([12, 64, 512], [64, 1, 768]),  # view_188
    torch.randn(589824, dtype=torch.float16, device='cuda').as_strided([768, 768], [1, 768]),  # permute_103
    torch.randn(589824, dtype=torch.float16, device='cuda').as_strided([768, 768], [1, 768]),  # permute_101
    torch.randn(589824, dtype=torch.float16, device='cuda').as_strided([768, 768], [1, 768]),  # permute_99
    torch.randn([1, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn(2359296, dtype=torch.float16, device='cuda').as_strided([3072, 768], [1, 3072]),  # permute_98
    torch.randn(2359296, dtype=torch.float16, device='cuda').as_strided([768, 3072], [1, 768]),  # permute_97
    torch.randn([1, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn(589824, dtype=torch.float16, device='cuda').as_strided([768, 768], [1, 768]),  # permute_96
    torch.randn([12, 512, 512], dtype=torch.float16, device='cuda'),
    torch.randn(393216, dtype=torch.float16, device='cuda').as_strided([12, 512, 64], [64, 768, 1]),  # view_171
    torch.randn(393216, dtype=torch.float16, device='cuda').as_strided([12, 512, 64], [64, 768, 1]),  # view_167
    torch.randn(393216, dtype=torch.float16, device='cuda').as_strided([12, 64, 512], [64, 1, 768]),  # view_168
    torch.randn(589824, dtype=torch.float16, device='cuda').as_strided([768, 768], [1, 768]),  # permute_92
    torch.randn(589824, dtype=torch.float16, device='cuda').as_strided([768, 768], [1, 768]),  # permute_90
    torch.randn(589824, dtype=torch.float16, device='cuda').as_strided([768, 768], [1, 768]),  # permute_88
    torch.randn([1, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn(2359296, dtype=torch.float16, device='cuda').as_strided([3072, 768], [1, 3072]),  # permute_87
    torch.randn(2359296, dtype=torch.float16, device='cuda').as_strided([768, 3072], [1, 768]),  # permute_86
    torch.randn([1, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn(589824, dtype=torch.float16, device='cuda').as_strided([768, 768], [1, 768]),  # permute_85
    torch.randn([12, 512, 512], dtype=torch.float16, device='cuda'),
    torch.randn(393216, dtype=torch.float16, device='cuda').as_strided([12, 512, 64], [64, 768, 1]),  # view_151
    torch.randn(393216, dtype=torch.float16, device='cuda').as_strided([12, 512, 64], [64, 768, 1]),  # view_147
    torch.randn(393216, dtype=torch.float16, device='cuda').as_strided([12, 64, 512], [64, 1, 768]),  # view_148
    torch.randn(589824, dtype=torch.float16, device='cuda').as_strided([768, 768], [1, 768]),  # permute_81
    torch.randn(589824, dtype=torch.float16, device='cuda').as_strided([768, 768], [1, 768]),  # permute_79
    torch.randn(589824, dtype=torch.float16, device='cuda').as_strided([768, 768], [1, 768]),  # permute_77
    torch.randn([1, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn(2359296, dtype=torch.float16, device='cuda').as_strided([3072, 768], [1, 3072]),  # permute_76
    torch.randn(2359296, dtype=torch.float16, device='cuda').as_strided([768, 3072], [1, 768]),  # permute_75
    torch.randn([1, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn(589824, dtype=torch.float16, device='cuda').as_strided([768, 768], [1, 768]),  # permute_74
    torch.randn([12, 512, 512], dtype=torch.float16, device='cuda'),
    torch.randn(393216, dtype=torch.float16, device='cuda').as_strided([12, 512, 64], [64, 768, 1]),  # view_131
    torch.randn(393216, dtype=torch.float16, device='cuda').as_strided([12, 512, 64], [64, 768, 1]),  # view_127
    torch.randn(393216, dtype=torch.float16, device='cuda').as_strided([12, 64, 512], [64, 1, 768]),  # view_128
    torch.randn(589824, dtype=torch.float16, device='cuda').as_strided([768, 768], [1, 768]),  # permute_70
    torch.randn(589824, dtype=torch.float16, device='cuda').as_strided([768, 768], [1, 768]),  # permute_68
    torch.randn(589824, dtype=torch.float16, device='cuda').as_strided([768, 768], [1, 768]),  # permute_66
    torch.randn([1, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn(2359296, dtype=torch.float16, device='cuda').as_strided([3072, 768], [1, 3072]),  # permute_65
    torch.randn(2359296, dtype=torch.float16, device='cuda').as_strided([768, 3072], [1, 768]),  # permute_64
    torch.randn([1, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn(589824, dtype=torch.float16, device='cuda').as_strided([768, 768], [1, 768]),  # permute_63
    torch.randn([12, 512, 512], dtype=torch.float16, device='cuda'),
    torch.randn(393216, dtype=torch.float16, device='cuda').as_strided([12, 512, 64], [64, 768, 1]),  # view_111
    torch.randn(393216, dtype=torch.float16, device='cuda').as_strided([12, 512, 64], [64, 768, 1]),  # view_107
    torch.randn(393216, dtype=torch.float16, device='cuda').as_strided([12, 64, 512], [64, 1, 768]),  # view_108
    torch.randn(589824, dtype=torch.float16, device='cuda').as_strided([768, 768], [1, 768]),  # permute_59
    torch.randn(589824, dtype=torch.float16, device='cuda').as_strided([768, 768], [1, 768]),  # permute_57
    torch.randn(589824, dtype=torch.float16, device='cuda').as_strided([768, 768], [1, 768]),  # permute_55
    torch.randn([1, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn(2359296, dtype=torch.float16, device='cuda').as_strided([3072, 768], [1, 3072]),  # permute_54
    torch.randn(2359296, dtype=torch.float16, device='cuda').as_strided([768, 3072], [1, 768]),  # permute_53
    torch.randn([1, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn(589824, dtype=torch.float16, device='cuda').as_strided([768, 768], [1, 768]),  # permute_52
    torch.randn([12, 512, 512], dtype=torch.float16, device='cuda'),
    torch.randn(393216, dtype=torch.float16, device='cuda').as_strided([12, 512, 64], [64, 768, 1]),  # view_91
    torch.randn(393216, dtype=torch.float16, device='cuda').as_strided([12, 512, 64], [64, 768, 1]),  # view_87
    torch.randn(393216, dtype=torch.float16, device='cuda').as_strided([12, 64, 512], [64, 1, 768]),  # view_88
    torch.randn(589824, dtype=torch.float16, device='cuda').as_strided([768, 768], [1, 768]),  # permute_48
    torch.randn(589824, dtype=torch.float16, device='cuda').as_strided([768, 768], [1, 768]),  # permute_46
    torch.randn(589824, dtype=torch.float16, device='cuda').as_strided([768, 768], [1, 768]),  # permute_44
    torch.randn([1, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn(2359296, dtype=torch.float16, device='cuda').as_strided([3072, 768], [1, 3072]),  # permute_43
    torch.randn(2359296, dtype=torch.float16, device='cuda').as_strided([768, 3072], [1, 768]),  # permute_42
    torch.randn([1, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn(589824, dtype=torch.float16, device='cuda').as_strided([768, 768], [1, 768]),  # permute_41
    torch.randn([12, 512, 512], dtype=torch.float16, device='cuda'),
    torch.randn(393216, dtype=torch.float16, device='cuda').as_strided([12, 512, 64], [64, 768, 1]),  # view_71
    torch.randn(393216, dtype=torch.float16, device='cuda').as_strided([12, 512, 64], [64, 768, 1]),  # view_67
    torch.randn(393216, dtype=torch.float16, device='cuda').as_strided([12, 64, 512], [64, 1, 768]),  # view_68
    torch.randn(589824, dtype=torch.float16, device='cuda').as_strided([768, 768], [1, 768]),  # permute_37
    torch.randn(589824, dtype=torch.float16, device='cuda').as_strided([768, 768], [1, 768]),  # permute_35
    torch.randn(589824, dtype=torch.float16, device='cuda').as_strided([768, 768], [1, 768]),  # permute_33
    torch.randn([1, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn(2359296, dtype=torch.float16, device='cuda').as_strided([3072, 768], [1, 3072]),  # permute_32
    torch.randn(2359296, dtype=torch.float16, device='cuda').as_strided([768, 3072], [1, 768]),  # permute_31
    torch.randn([1, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn(589824, dtype=torch.float16, device='cuda').as_strided([768, 768], [1, 768]),  # permute_30
    torch.randn([12, 512, 512], dtype=torch.float16, device='cuda'),
    torch.randn(393216, dtype=torch.float16, device='cuda').as_strided([12, 512, 64], [64, 768, 1]),  # view_51
    torch.randn(393216, dtype=torch.float16, device='cuda').as_strided([12, 512, 64], [64, 768, 1]),  # view_47
    torch.randn(393216, dtype=torch.float16, device='cuda').as_strided([12, 64, 512], [64, 1, 768]),  # view_48
    torch.randn(589824, dtype=torch.float16, device='cuda').as_strided([768, 768], [1, 768]),  # permute_26
    torch.randn(589824, dtype=torch.float16, device='cuda').as_strided([768, 768], [1, 768]),  # permute_24
    torch.randn(589824, dtype=torch.float16, device='cuda').as_strided([768, 768], [1, 768]),  # permute_22
    torch.randn([1, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn(2359296, dtype=torch.float16, device='cuda').as_strided([3072, 768], [1, 3072]),  # permute_21
    torch.randn(2359296, dtype=torch.float16, device='cuda').as_strided([768, 3072], [1, 768]),  # permute_20
    torch.randn([1, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn(589824, dtype=torch.float16, device='cuda').as_strided([768, 768], [1, 768]),  # permute_19
    torch.randn([12, 512, 512], dtype=torch.float16, device='cuda'),
    torch.randn(393216, dtype=torch.float16, device='cuda').as_strided([12, 512, 64], [64, 768, 1]),  # view_31
    torch.randn(393216, dtype=torch.float16, device='cuda').as_strided([12, 512, 64], [64, 768, 1]),  # view_27
    torch.randn(393216, dtype=torch.float16, device='cuda').as_strided([12, 64, 512], [64, 1, 768]),  # view_28
    torch.randn(589824, dtype=torch.float16, device='cuda').as_strided([768, 768], [1, 768]),  # permute_15
    torch.randn(589824, dtype=torch.float16, device='cuda').as_strided([768, 768], [1, 768]),  # permute_13
    torch.randn(589824, dtype=torch.float16, device='cuda').as_strided([768, 768], [1, 768]),  # permute_11
    torch.randn([1, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn(2359296, dtype=torch.float16, device='cuda').as_strided([3072, 768], [1, 3072]),  # permute_10
    torch.randn(2359296, dtype=torch.float16, device='cuda').as_strided([768, 3072], [1, 768]),  # permute_9
    torch.randn([1, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn(589824, dtype=torch.float16, device='cuda').as_strided([768, 768], [1, 768]),  # permute_8
    torch.randn([12, 512, 512], dtype=torch.float16, device='cuda'),
    torch.randn(393216, dtype=torch.float16, device='cuda').as_strided([12, 512, 64], [64, 768, 1]),  # view_11
    torch.randn(393216, dtype=torch.float16, device='cuda').as_strided([12, 512, 64], [64, 768, 1]),  # view_7
    torch.randn(393216, dtype=torch.float16, device='cuda').as_strided([12, 64, 512], [64, 1, 768]),  # view_8
    torch.randn(589824, dtype=torch.float16, device='cuda').as_strided([768, 768], [1, 768]),  # permute_4
    torch.randn(589824, dtype=torch.float16, device='cuda').as_strided([768, 768], [1, 768]),  # permute_2
    torch.randn(589824, dtype=torch.float16, device='cuda').as_strided([768, 768], [1, 768]),  # permute
    torch.randn([1, 512, 1], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
