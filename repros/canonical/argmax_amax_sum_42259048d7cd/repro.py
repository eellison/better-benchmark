"""
Standalone repro captured via capture_hook.
Label: tlparse_huggingface_s2_g21
Pattern hash: 42259048d7cd
Shape hash: 3bfa6d9c
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
    def forward(self, mm: "f16[4096, 2]", arg1_1: "i64[4, 1024]", iota_1: "i64[4]", arg152_1: "i64[4]", permute_48: "f16[768, 2]", rsqrt_24: "f32[4, 1024, 1]", convert_element_type_164: "f16[3072, 768]", convert_element_type_165: "f16[4096, 3072]", convert_element_type_160: "f16[768, 3072]", convert_element_type_161: "f16[4096, 768]", rsqrt_23: "f32[4, 1024, 1]", convert_element_type_157: "f16[768, 768]", convert_element_type_154: "f16[768, 2304]", convert_element_type_155: "f16[4096, 768]", rsqrt_22: "f32[4, 1024, 1]", convert_element_type_150: "f16[3072, 768]", convert_element_type_151: "f16[4096, 3072]", convert_element_type_146: "f16[768, 3072]", convert_element_type_147: "f16[4096, 768]", rsqrt_21: "f32[4, 1024, 1]", convert_element_type_143: "f16[768, 768]", convert_element_type_140: "f16[768, 2304]", convert_element_type_141: "f16[4096, 768]", rsqrt_20: "f32[4, 1024, 1]", convert_element_type_136: "f16[3072, 768]", convert_element_type_137: "f16[4096, 3072]", convert_element_type_132: "f16[768, 3072]", convert_element_type_133: "f16[4096, 768]", rsqrt_19: "f32[4, 1024, 1]", convert_element_type_129: "f16[768, 768]", convert_element_type_126: "f16[768, 2304]", convert_element_type_127: "f16[4096, 768]", rsqrt_18: "f32[4, 1024, 1]", convert_element_type_122: "f16[3072, 768]", convert_element_type_123: "f16[4096, 3072]", convert_element_type_118: "f16[768, 3072]", convert_element_type_119: "f16[4096, 768]", rsqrt_17: "f32[4, 1024, 1]", convert_element_type_115: "f16[768, 768]", convert_element_type_112: "f16[768, 2304]", convert_element_type_113: "f16[4096, 768]", rsqrt_16: "f32[4, 1024, 1]", convert_element_type_108: "f16[3072, 768]", convert_element_type_109: "f16[4096, 3072]", convert_element_type_104: "f16[768, 3072]", convert_element_type_105: "f16[4096, 768]", rsqrt_15: "f32[4, 1024, 1]", convert_element_type_101: "f16[768, 768]", convert_element_type_98: "f16[768, 2304]", convert_element_type_99: "f16[4096, 768]", rsqrt_14: "f32[4, 1024, 1]", convert_element_type_94: "f16[3072, 768]", convert_element_type_95: "f16[4096, 3072]", convert_element_type_90: "f16[768, 3072]", convert_element_type_91: "f16[4096, 768]", rsqrt_13: "f32[4, 1024, 1]", convert_element_type_87: "f16[768, 768]", convert_element_type_84: "f16[768, 2304]", convert_element_type_85: "f16[4096, 768]", rsqrt_12: "f32[4, 1024, 1]", convert_element_type_80: "f16[3072, 768]", convert_element_type_81: "f16[4096, 3072]", convert_element_type_76: "f16[768, 3072]", convert_element_type_77: "f16[4096, 768]", rsqrt_11: "f32[4, 1024, 1]", convert_element_type_73: "f16[768, 768]", convert_element_type_70: "f16[768, 2304]", convert_element_type_71: "f16[4096, 768]", rsqrt_10: "f32[4, 1024, 1]", convert_element_type_66: "f16[3072, 768]", convert_element_type_67: "f16[4096, 3072]", convert_element_type_62: "f16[768, 3072]", convert_element_type_63: "f16[4096, 768]", rsqrt_9: "f32[4, 1024, 1]", convert_element_type_59: "f16[768, 768]", convert_element_type_56: "f16[768, 2304]", convert_element_type_57: "f16[4096, 768]", rsqrt_8: "f32[4, 1024, 1]", convert_element_type_52: "f16[3072, 768]", convert_element_type_53: "f16[4096, 3072]", convert_element_type_48: "f16[768, 3072]", convert_element_type_49: "f16[4096, 768]", rsqrt_7: "f32[4, 1024, 1]", convert_element_type_45: "f16[768, 768]", convert_element_type_42: "f16[768, 2304]", convert_element_type_43: "f16[4096, 768]", rsqrt_6: "f32[4, 1024, 1]", convert_element_type_38: "f16[3072, 768]", convert_element_type_39: "f16[4096, 3072]", convert_element_type_34: "f16[768, 3072]", convert_element_type_35: "f16[4096, 768]", rsqrt_5: "f32[4, 1024, 1]", convert_element_type_31: "f16[768, 768]", convert_element_type_28: "f16[768, 2304]", convert_element_type_29: "f16[4096, 768]", rsqrt_4: "f32[4, 1024, 1]", convert_element_type_24: "f16[3072, 768]", convert_element_type_25: "f16[4096, 3072]", convert_element_type_20: "f16[768, 3072]", convert_element_type_21: "f16[4096, 768]", rsqrt_3: "f32[4, 1024, 1]", convert_element_type_17: "f16[768, 768]", convert_element_type_14: "f16[768, 2304]", convert_element_type_15: "f16[4096, 768]", rsqrt_2: "f32[4, 1024, 1]", convert_element_type_10: "f16[3072, 768]", convert_element_type_11: "f16[4096, 3072]", convert_element_type_6: "f16[768, 3072]", convert_element_type_7: "f16[4096, 768]", rsqrt_1: "f32[4, 1024, 1]", convert_element_type_3: "f16[768, 768]", convert_element_type: "f16[768, 2304]", convert_element_type_1: "f16[4096, 768]", rsqrt: "f32[4, 1024, 1]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        reshape_default: "f16[4, 1024, 2]" = torch.ops.aten.reshape.default(mm, _shape_param_0);  mm = _shape_param_0 = None
        ne_scalar: "b8[4, 1024]" = torch.ops.aten.ne.Scalar(arg1_1, 0);  arg1_1 = None
        convert_element_type_default: "i32[4, 1024]" = torch.ops.prims.convert_element_type.default(ne_scalar, torch.int32);  ne_scalar = None
        iota_default: "i32[1024]" = torch.ops.prims.iota.default(1024, start = 0, step = 1, dtype = torch.int32, device = device(type='cuda', index=0), requires_grad = False)
        mul_tensor: "i32[4, 1024]" = torch.ops.aten.mul.Tensor(iota_default, convert_element_type_default);  iota_default = convert_element_type_default = None
        argmax_default: "i64[4]" = torch.ops.aten.argmax.default(mul_tensor, -1);  mul_tensor = None
        index_tensor: "f16[4, 2]" = torch.ops.aten.index.Tensor(reshape_default, [iota_1, argmax_default]);  reshape_default = iota_1 = argmax_default = None
        convert_element_type_default_1: "f32[4, 2]" = torch.ops.prims.convert_element_type.default(index_tensor, torch.float32);  index_tensor = None
        amax_default: "f32[4, 1]" = torch.ops.aten.amax.default(convert_element_type_default_1, [1], True)
        sub_tensor: "f32[4, 2]" = torch.ops.aten.sub.Tensor(convert_element_type_default_1, amax_default);  convert_element_type_default_1 = amax_default = None
        exp_default: "f32[4, 2]" = torch.ops.aten.exp.default(sub_tensor)
        sum_dim_int_list: "f32[4, 1]" = torch.ops.aten.sum.dim_IntList(exp_default, [1], True);  exp_default = None
        log_default: "f32[4, 1]" = torch.ops.aten.log.default(sum_dim_int_list);  sum_dim_int_list = None
        sub_tensor_1: "f32[4, 2]" = torch.ops.aten.sub.Tensor(sub_tensor, log_default);  sub_tensor = log_default = None
        convert_element_type_default_2: "f16[4, 2]" = torch.ops.prims.convert_element_type.default(sub_tensor_1, torch.float16);  sub_tensor_1 = None
        convert_element_type_default_3: "f32[4, 2]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_2, torch.float32);  convert_element_type_default_2 = None
        ne_scalar_1: "b8[4]" = torch.ops.aten.ne.Scalar(arg152_1, -100)
        full_default: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "i64[4]" = torch.ops.aten.where.self(ne_scalar_1, arg152_1, full_default)
        unsqueeze_default: "i64[4, 1]" = torch.ops.aten.unsqueeze.default(where_self, 1);  where_self = None
        gather_default: "f32[4, 1]" = torch.ops.aten.gather.default(convert_element_type_default_3, 1, unsqueeze_default);  convert_element_type_default_3 = unsqueeze_default = None
        squeeze_dim: "f32[4]" = torch.ops.aten.squeeze.dim(gather_default, 1);  gather_default = None
        neg_default: "f32[4]" = torch.ops.aten.neg.default(squeeze_dim);  squeeze_dim = None
        full_default_1: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_1: "f32[4]" = torch.ops.aten.where.self(ne_scalar_1, neg_default, full_default_1);  neg_default = full_default_1 = None
        sum_default: "i64[]" = torch.ops.aten.sum.default(ne_scalar_1);  ne_scalar_1 = None
        convert_element_type_default_4: "f32[]" = torch.ops.prims.convert_element_type.default(sum_default, torch.float32);  sum_default = None
        sum_default_1: "f32[]" = torch.ops.aten.sum.default(where_self_1);  where_self_1 = None
        div_tensor: "f32[]" = torch.ops.aten.div.Tensor(sum_default_1, convert_element_type_default_4);  sum_default_1 = convert_element_type_default_4 = None
        unsqueeze_default_1: "i64[4, 1]" = torch.ops.aten.unsqueeze.default(arg152_1, 1);  arg152_1 = None
        ne_scalar_2: "b8[4, 1]" = torch.ops.aten.ne.Scalar(unsqueeze_default_1, -100)
        where_self_2: "i64[4, 1]" = torch.ops.aten.where.self(ne_scalar_2, unsqueeze_default_1, full_default);  ne_scalar_2 = unsqueeze_default_1 = full_default = None
        iota_default_1: "i64[2]" = torch.ops.prims.iota.default(2, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        reshape_default_1: "i64[1, 2]" = torch.ops.aten.reshape.default(iota_default_1, _shape_param_1);  iota_default_1 = _shape_param_1 = None
        expand_default: "i64[4, 2]" = torch.ops.aten.expand.default(where_self_2, _shape_param_2);  where_self_2 = _shape_param_2 = None
        eq_tensor: "b8[4, 2]" = torch.ops.aten.eq.Tensor(expand_default, reshape_default_1);  expand_default = reshape_default_1 = None
        permute_default: "f16[2, 768]" = torch.ops.aten.permute.default(permute_48, [1, 0]);  permute_48 = None
        div_tensor_1: "f32[4, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_24, 768);  rsqrt_24 = None
        permute_default_1: "f16[768, 3072]" = torch.ops.aten.permute.default(convert_element_type_164, [1, 0]);  convert_element_type_164 = None
        permute_default_2: "f16[3072, 4096]" = torch.ops.aten.permute.default(convert_element_type_165, [1, 0]);  convert_element_type_165 = None
        permute_default_3: "f16[3072, 768]" = torch.ops.aten.permute.default(convert_element_type_160, [1, 0]);  convert_element_type_160 = None
        permute_default_4: "f16[768, 4096]" = torch.ops.aten.permute.default(convert_element_type_161, [1, 0]);  convert_element_type_161 = None
        div_tensor_2: "f32[4, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_23, 768);  rsqrt_23 = None
        permute_default_5: "f16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_157, [1, 0]);  convert_element_type_157 = None
        permute_default_6: "f16[2304, 768]" = torch.ops.aten.permute.default(convert_element_type_154, [1, 0]);  convert_element_type_154 = None
        permute_default_7: "f16[768, 4096]" = torch.ops.aten.permute.default(convert_element_type_155, [1, 0]);  convert_element_type_155 = None
        div_tensor_3: "f32[4, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_22, 768);  rsqrt_22 = None
        permute_default_8: "f16[768, 3072]" = torch.ops.aten.permute.default(convert_element_type_150, [1, 0]);  convert_element_type_150 = None
        permute_default_9: "f16[3072, 4096]" = torch.ops.aten.permute.default(convert_element_type_151, [1, 0]);  convert_element_type_151 = None
        permute_default_10: "f16[3072, 768]" = torch.ops.aten.permute.default(convert_element_type_146, [1, 0]);  convert_element_type_146 = None
        permute_default_11: "f16[768, 4096]" = torch.ops.aten.permute.default(convert_element_type_147, [1, 0]);  convert_element_type_147 = None
        div_tensor_4: "f32[4, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_21, 768);  rsqrt_21 = None
        permute_default_12: "f16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_143, [1, 0]);  convert_element_type_143 = None
        permute_default_13: "f16[2304, 768]" = torch.ops.aten.permute.default(convert_element_type_140, [1, 0]);  convert_element_type_140 = None
        permute_default_14: "f16[768, 4096]" = torch.ops.aten.permute.default(convert_element_type_141, [1, 0]);  convert_element_type_141 = None
        div_tensor_5: "f32[4, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_20, 768);  rsqrt_20 = None
        permute_default_15: "f16[768, 3072]" = torch.ops.aten.permute.default(convert_element_type_136, [1, 0]);  convert_element_type_136 = None
        permute_default_16: "f16[3072, 4096]" = torch.ops.aten.permute.default(convert_element_type_137, [1, 0]);  convert_element_type_137 = None
        permute_default_17: "f16[3072, 768]" = torch.ops.aten.permute.default(convert_element_type_132, [1, 0]);  convert_element_type_132 = None
        permute_default_18: "f16[768, 4096]" = torch.ops.aten.permute.default(convert_element_type_133, [1, 0]);  convert_element_type_133 = None
        div_tensor_6: "f32[4, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_19, 768);  rsqrt_19 = None
        permute_default_19: "f16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_129, [1, 0]);  convert_element_type_129 = None
        permute_default_20: "f16[2304, 768]" = torch.ops.aten.permute.default(convert_element_type_126, [1, 0]);  convert_element_type_126 = None
        permute_default_21: "f16[768, 4096]" = torch.ops.aten.permute.default(convert_element_type_127, [1, 0]);  convert_element_type_127 = None
        div_tensor_7: "f32[4, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_18, 768);  rsqrt_18 = None
        permute_default_22: "f16[768, 3072]" = torch.ops.aten.permute.default(convert_element_type_122, [1, 0]);  convert_element_type_122 = None
        permute_default_23: "f16[3072, 4096]" = torch.ops.aten.permute.default(convert_element_type_123, [1, 0]);  convert_element_type_123 = None
        permute_default_24: "f16[3072, 768]" = torch.ops.aten.permute.default(convert_element_type_118, [1, 0]);  convert_element_type_118 = None
        permute_default_25: "f16[768, 4096]" = torch.ops.aten.permute.default(convert_element_type_119, [1, 0]);  convert_element_type_119 = None
        div_tensor_8: "f32[4, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_17, 768);  rsqrt_17 = None
        permute_default_26: "f16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_115, [1, 0]);  convert_element_type_115 = None
        permute_default_27: "f16[2304, 768]" = torch.ops.aten.permute.default(convert_element_type_112, [1, 0]);  convert_element_type_112 = None
        permute_default_28: "f16[768, 4096]" = torch.ops.aten.permute.default(convert_element_type_113, [1, 0]);  convert_element_type_113 = None
        div_tensor_9: "f32[4, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_16, 768);  rsqrt_16 = None
        permute_default_29: "f16[768, 3072]" = torch.ops.aten.permute.default(convert_element_type_108, [1, 0]);  convert_element_type_108 = None
        permute_default_30: "f16[3072, 4096]" = torch.ops.aten.permute.default(convert_element_type_109, [1, 0]);  convert_element_type_109 = None
        permute_default_31: "f16[3072, 768]" = torch.ops.aten.permute.default(convert_element_type_104, [1, 0]);  convert_element_type_104 = None
        permute_default_32: "f16[768, 4096]" = torch.ops.aten.permute.default(convert_element_type_105, [1, 0]);  convert_element_type_105 = None
        div_tensor_10: "f32[4, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_15, 768);  rsqrt_15 = None
        permute_default_33: "f16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_101, [1, 0]);  convert_element_type_101 = None
        permute_default_34: "f16[2304, 768]" = torch.ops.aten.permute.default(convert_element_type_98, [1, 0]);  convert_element_type_98 = None
        permute_default_35: "f16[768, 4096]" = torch.ops.aten.permute.default(convert_element_type_99, [1, 0]);  convert_element_type_99 = None
        div_tensor_11: "f32[4, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_14, 768);  rsqrt_14 = None
        permute_default_36: "f16[768, 3072]" = torch.ops.aten.permute.default(convert_element_type_94, [1, 0]);  convert_element_type_94 = None
        permute_default_37: "f16[3072, 4096]" = torch.ops.aten.permute.default(convert_element_type_95, [1, 0]);  convert_element_type_95 = None
        permute_default_38: "f16[3072, 768]" = torch.ops.aten.permute.default(convert_element_type_90, [1, 0]);  convert_element_type_90 = None
        permute_default_39: "f16[768, 4096]" = torch.ops.aten.permute.default(convert_element_type_91, [1, 0]);  convert_element_type_91 = None
        div_tensor_12: "f32[4, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_13, 768);  rsqrt_13 = None
        permute_default_40: "f16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_87, [1, 0]);  convert_element_type_87 = None
        permute_default_41: "f16[2304, 768]" = torch.ops.aten.permute.default(convert_element_type_84, [1, 0]);  convert_element_type_84 = None
        permute_default_42: "f16[768, 4096]" = torch.ops.aten.permute.default(convert_element_type_85, [1, 0]);  convert_element_type_85 = None
        div_tensor_13: "f32[4, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_12, 768);  rsqrt_12 = None
        permute_default_43: "f16[768, 3072]" = torch.ops.aten.permute.default(convert_element_type_80, [1, 0]);  convert_element_type_80 = None
        permute_default_44: "f16[3072, 4096]" = torch.ops.aten.permute.default(convert_element_type_81, [1, 0]);  convert_element_type_81 = None
        permute_default_45: "f16[3072, 768]" = torch.ops.aten.permute.default(convert_element_type_76, [1, 0]);  convert_element_type_76 = None
        permute_default_46: "f16[768, 4096]" = torch.ops.aten.permute.default(convert_element_type_77, [1, 0]);  convert_element_type_77 = None
        div_tensor_14: "f32[4, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_11, 768);  rsqrt_11 = None
        permute_default_47: "f16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_73, [1, 0]);  convert_element_type_73 = None
        permute_default_48: "f16[2304, 768]" = torch.ops.aten.permute.default(convert_element_type_70, [1, 0]);  convert_element_type_70 = None
        permute_default_49: "f16[768, 4096]" = torch.ops.aten.permute.default(convert_element_type_71, [1, 0]);  convert_element_type_71 = None
        div_tensor_15: "f32[4, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_10, 768);  rsqrt_10 = None
        permute_default_50: "f16[768, 3072]" = torch.ops.aten.permute.default(convert_element_type_66, [1, 0]);  convert_element_type_66 = None
        permute_default_51: "f16[3072, 4096]" = torch.ops.aten.permute.default(convert_element_type_67, [1, 0]);  convert_element_type_67 = None
        permute_default_52: "f16[3072, 768]" = torch.ops.aten.permute.default(convert_element_type_62, [1, 0]);  convert_element_type_62 = None
        permute_default_53: "f16[768, 4096]" = torch.ops.aten.permute.default(convert_element_type_63, [1, 0]);  convert_element_type_63 = None
        div_tensor_16: "f32[4, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_9, 768);  rsqrt_9 = None
        permute_default_54: "f16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_59, [1, 0]);  convert_element_type_59 = None
        permute_default_55: "f16[2304, 768]" = torch.ops.aten.permute.default(convert_element_type_56, [1, 0]);  convert_element_type_56 = None
        permute_default_56: "f16[768, 4096]" = torch.ops.aten.permute.default(convert_element_type_57, [1, 0]);  convert_element_type_57 = None
        div_tensor_17: "f32[4, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_8, 768);  rsqrt_8 = None
        permute_default_57: "f16[768, 3072]" = torch.ops.aten.permute.default(convert_element_type_52, [1, 0]);  convert_element_type_52 = None
        permute_default_58: "f16[3072, 4096]" = torch.ops.aten.permute.default(convert_element_type_53, [1, 0]);  convert_element_type_53 = None
        permute_default_59: "f16[3072, 768]" = torch.ops.aten.permute.default(convert_element_type_48, [1, 0]);  convert_element_type_48 = None
        permute_default_60: "f16[768, 4096]" = torch.ops.aten.permute.default(convert_element_type_49, [1, 0]);  convert_element_type_49 = None
        div_tensor_18: "f32[4, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_7, 768);  rsqrt_7 = None
        permute_default_61: "f16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_45, [1, 0]);  convert_element_type_45 = None
        permute_default_62: "f16[2304, 768]" = torch.ops.aten.permute.default(convert_element_type_42, [1, 0]);  convert_element_type_42 = None
        permute_default_63: "f16[768, 4096]" = torch.ops.aten.permute.default(convert_element_type_43, [1, 0]);  convert_element_type_43 = None
        div_tensor_19: "f32[4, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_6, 768);  rsqrt_6 = None
        permute_default_64: "f16[768, 3072]" = torch.ops.aten.permute.default(convert_element_type_38, [1, 0]);  convert_element_type_38 = None
        permute_default_65: "f16[3072, 4096]" = torch.ops.aten.permute.default(convert_element_type_39, [1, 0]);  convert_element_type_39 = None
        permute_default_66: "f16[3072, 768]" = torch.ops.aten.permute.default(convert_element_type_34, [1, 0]);  convert_element_type_34 = None
        permute_default_67: "f16[768, 4096]" = torch.ops.aten.permute.default(convert_element_type_35, [1, 0]);  convert_element_type_35 = None
        div_tensor_20: "f32[4, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_5, 768);  rsqrt_5 = None
        permute_default_68: "f16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_31, [1, 0]);  convert_element_type_31 = None
        permute_default_69: "f16[2304, 768]" = torch.ops.aten.permute.default(convert_element_type_28, [1, 0]);  convert_element_type_28 = None
        permute_default_70: "f16[768, 4096]" = torch.ops.aten.permute.default(convert_element_type_29, [1, 0]);  convert_element_type_29 = None
        div_tensor_21: "f32[4, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_4, 768);  rsqrt_4 = None
        permute_default_71: "f16[768, 3072]" = torch.ops.aten.permute.default(convert_element_type_24, [1, 0]);  convert_element_type_24 = None
        permute_default_72: "f16[3072, 4096]" = torch.ops.aten.permute.default(convert_element_type_25, [1, 0]);  convert_element_type_25 = None
        permute_default_73: "f16[3072, 768]" = torch.ops.aten.permute.default(convert_element_type_20, [1, 0]);  convert_element_type_20 = None
        permute_default_74: "f16[768, 4096]" = torch.ops.aten.permute.default(convert_element_type_21, [1, 0]);  convert_element_type_21 = None
        div_tensor_22: "f32[4, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_3, 768);  rsqrt_3 = None
        permute_default_75: "f16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_17, [1, 0]);  convert_element_type_17 = None
        permute_default_76: "f16[2304, 768]" = torch.ops.aten.permute.default(convert_element_type_14, [1, 0]);  convert_element_type_14 = None
        permute_default_77: "f16[768, 4096]" = torch.ops.aten.permute.default(convert_element_type_15, [1, 0]);  convert_element_type_15 = None
        div_tensor_23: "f32[4, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_2, 768);  rsqrt_2 = None
        permute_default_78: "f16[768, 3072]" = torch.ops.aten.permute.default(convert_element_type_10, [1, 0]);  convert_element_type_10 = None
        permute_default_79: "f16[3072, 4096]" = torch.ops.aten.permute.default(convert_element_type_11, [1, 0]);  convert_element_type_11 = None
        permute_default_80: "f16[3072, 768]" = torch.ops.aten.permute.default(convert_element_type_6, [1, 0]);  convert_element_type_6 = None
        permute_default_81: "f16[768, 4096]" = torch.ops.aten.permute.default(convert_element_type_7, [1, 0]);  convert_element_type_7 = None
        div_tensor_24: "f32[4, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_1, 768);  rsqrt_1 = None
        permute_default_82: "f16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_3, [1, 0]);  convert_element_type_3 = None
        permute_default_83: "f16[2304, 768]" = torch.ops.aten.permute.default(convert_element_type, [1, 0]);  convert_element_type = None
        permute_default_84: "f16[768, 4096]" = torch.ops.aten.permute.default(convert_element_type_1, [1, 0]);  convert_element_type_1 = None
        div_tensor_25: "f32[4, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt, 768);  rsqrt = None
        _output_to_half_0: "f16[]" = torch.ops.prims.convert_element_type.default(div_tensor, torch.float16);  div_tensor = None
        _output_to_half_1: "f16[4, 1024, 1]" = torch.ops.prims.convert_element_type.default(div_tensor_1, torch.float16);  div_tensor_1 = None
        _output_to_half_2: "f16[4, 1024, 1]" = torch.ops.prims.convert_element_type.default(div_tensor_2, torch.float16);  div_tensor_2 = None
        _output_to_half_3: "f16[4, 1024, 1]" = torch.ops.prims.convert_element_type.default(div_tensor_3, torch.float16);  div_tensor_3 = None
        _output_to_half_4: "f16[4, 1024, 1]" = torch.ops.prims.convert_element_type.default(div_tensor_4, torch.float16);  div_tensor_4 = None
        _output_to_half_5: "f16[4, 1024, 1]" = torch.ops.prims.convert_element_type.default(div_tensor_5, torch.float16);  div_tensor_5 = None
        _output_to_half_6: "f16[4, 1024, 1]" = torch.ops.prims.convert_element_type.default(div_tensor_6, torch.float16);  div_tensor_6 = None
        _output_to_half_7: "f16[4, 1024, 1]" = torch.ops.prims.convert_element_type.default(div_tensor_7, torch.float16);  div_tensor_7 = None
        _output_to_half_8: "f16[4, 1024, 1]" = torch.ops.prims.convert_element_type.default(div_tensor_8, torch.float16);  div_tensor_8 = None
        _output_to_half_9: "f16[4, 1024, 1]" = torch.ops.prims.convert_element_type.default(div_tensor_9, torch.float16);  div_tensor_9 = None
        _output_to_half_10: "f16[4, 1024, 1]" = torch.ops.prims.convert_element_type.default(div_tensor_10, torch.float16);  div_tensor_10 = None
        _output_to_half_11: "f16[4, 1024, 1]" = torch.ops.prims.convert_element_type.default(div_tensor_11, torch.float16);  div_tensor_11 = None
        _output_to_half_12: "f16[4, 1024, 1]" = torch.ops.prims.convert_element_type.default(div_tensor_12, torch.float16);  div_tensor_12 = None
        _output_to_half_13: "f16[4, 1024, 1]" = torch.ops.prims.convert_element_type.default(div_tensor_13, torch.float16);  div_tensor_13 = None
        _output_to_half_14: "f16[4, 1024, 1]" = torch.ops.prims.convert_element_type.default(div_tensor_14, torch.float16);  div_tensor_14 = None
        _output_to_half_15: "f16[4, 1024, 1]" = torch.ops.prims.convert_element_type.default(div_tensor_15, torch.float16);  div_tensor_15 = None
        _output_to_half_16: "f16[4, 1024, 1]" = torch.ops.prims.convert_element_type.default(div_tensor_16, torch.float16);  div_tensor_16 = None
        _output_to_half_17: "f16[4, 1024, 1]" = torch.ops.prims.convert_element_type.default(div_tensor_17, torch.float16);  div_tensor_17 = None
        _output_to_half_18: "f16[4, 1024, 1]" = torch.ops.prims.convert_element_type.default(div_tensor_18, torch.float16);  div_tensor_18 = None
        _output_to_half_19: "f16[4, 1024, 1]" = torch.ops.prims.convert_element_type.default(div_tensor_19, torch.float16);  div_tensor_19 = None
        _output_to_half_20: "f16[4, 1024, 1]" = torch.ops.prims.convert_element_type.default(div_tensor_20, torch.float16);  div_tensor_20 = None
        _output_to_half_21: "f16[4, 1024, 1]" = torch.ops.prims.convert_element_type.default(div_tensor_21, torch.float16);  div_tensor_21 = None
        _output_to_half_22: "f16[4, 1024, 1]" = torch.ops.prims.convert_element_type.default(div_tensor_22, torch.float16);  div_tensor_22 = None
        _output_to_half_23: "f16[4, 1024, 1]" = torch.ops.prims.convert_element_type.default(div_tensor_23, torch.float16);  div_tensor_23 = None
        _output_to_half_24: "f16[4, 1024, 1]" = torch.ops.prims.convert_element_type.default(div_tensor_24, torch.float16);  div_tensor_24 = None
        _output_to_half_25: "f16[4, 1024, 1]" = torch.ops.prims.convert_element_type.default(div_tensor_25, torch.float16);  div_tensor_25 = None
        return (_output_to_half_0, eq_tensor, permute_default, _output_to_half_1, permute_default_1, permute_default_2, permute_default_3, permute_default_4, _output_to_half_2, permute_default_5, permute_default_6, permute_default_7, _output_to_half_3, permute_default_8, permute_default_9, permute_default_10, permute_default_11, _output_to_half_4, permute_default_12, permute_default_13, permute_default_14, _output_to_half_5, permute_default_15, permute_default_16, permute_default_17, permute_default_18, _output_to_half_6, permute_default_19, permute_default_20, permute_default_21, _output_to_half_7, permute_default_22, permute_default_23, permute_default_24, permute_default_25, _output_to_half_8, permute_default_26, permute_default_27, permute_default_28, _output_to_half_9, permute_default_29, permute_default_30, permute_default_31, permute_default_32, _output_to_half_10, permute_default_33, permute_default_34, permute_default_35, _output_to_half_11, permute_default_36, permute_default_37, permute_default_38, permute_default_39, _output_to_half_12, permute_default_40, permute_default_41, permute_default_42, _output_to_half_13, permute_default_43, permute_default_44, permute_default_45, permute_default_46, _output_to_half_14, permute_default_47, permute_default_48, permute_default_49, _output_to_half_15, permute_default_50, permute_default_51, permute_default_52, permute_default_53, _output_to_half_16, permute_default_54, permute_default_55, permute_default_56, _output_to_half_17, permute_default_57, permute_default_58, permute_default_59, permute_default_60, _output_to_half_18, permute_default_61, permute_default_62, permute_default_63, _output_to_half_19, permute_default_64, permute_default_65, permute_default_66, permute_default_67, _output_to_half_20, permute_default_68, permute_default_69, permute_default_70, _output_to_half_21, permute_default_71, permute_default_72, permute_default_73, permute_default_74, _output_to_half_22, permute_default_75, permute_default_76, permute_default_77, _output_to_half_23, permute_default_78, permute_default_79, permute_default_80, permute_default_81, _output_to_half_24, permute_default_82, permute_default_83, permute_default_84, _output_to_half_25)


def _default_make_inputs():
    return [
    torch.randn([4096, 2], dtype=torch.float16, device='cuda'),
    torch.randint(0, 2, [4, 1024], dtype=torch.int64, device='cuda'),
    torch.randint(0, 4, [4], dtype=torch.int64, device='cuda'),
    torch.randint(0, 2, [4], dtype=torch.int64, device='cuda'),
    torch.randn(1536, dtype=torch.float16, device='cuda').as_strided([768, 2], [1, 768]),  # permute_48
    torch.randn([4, 1024, 1], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float16, device='cuda'),
    torch.randn([4096, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([4096, 768], dtype=torch.float16, device='cuda'),
    torch.randn([4, 1024, 1], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 2304], dtype=torch.float16, device='cuda'),
    torch.randn([4096, 768], dtype=torch.float16, device='cuda'),
    torch.randn([4, 1024, 1], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float16, device='cuda'),
    torch.randn([4096, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([4096, 768], dtype=torch.float16, device='cuda'),
    torch.randn([4, 1024, 1], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 2304], dtype=torch.float16, device='cuda'),
    torch.randn([4096, 768], dtype=torch.float16, device='cuda'),
    torch.randn([4, 1024, 1], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float16, device='cuda'),
    torch.randn([4096, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([4096, 768], dtype=torch.float16, device='cuda'),
    torch.randn([4, 1024, 1], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 2304], dtype=torch.float16, device='cuda'),
    torch.randn([4096, 768], dtype=torch.float16, device='cuda'),
    torch.randn([4, 1024, 1], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float16, device='cuda'),
    torch.randn([4096, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([4096, 768], dtype=torch.float16, device='cuda'),
    torch.randn([4, 1024, 1], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 2304], dtype=torch.float16, device='cuda'),
    torch.randn([4096, 768], dtype=torch.float16, device='cuda'),
    torch.randn([4, 1024, 1], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float16, device='cuda'),
    torch.randn([4096, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([4096, 768], dtype=torch.float16, device='cuda'),
    torch.randn([4, 1024, 1], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 2304], dtype=torch.float16, device='cuda'),
    torch.randn([4096, 768], dtype=torch.float16, device='cuda'),
    torch.randn([4, 1024, 1], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float16, device='cuda'),
    torch.randn([4096, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([4096, 768], dtype=torch.float16, device='cuda'),
    torch.randn([4, 1024, 1], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 2304], dtype=torch.float16, device='cuda'),
    torch.randn([4096, 768], dtype=torch.float16, device='cuda'),
    torch.randn([4, 1024, 1], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float16, device='cuda'),
    torch.randn([4096, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([4096, 768], dtype=torch.float16, device='cuda'),
    torch.randn([4, 1024, 1], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 2304], dtype=torch.float16, device='cuda'),
    torch.randn([4096, 768], dtype=torch.float16, device='cuda'),
    torch.randn([4, 1024, 1], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float16, device='cuda'),
    torch.randn([4096, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([4096, 768], dtype=torch.float16, device='cuda'),
    torch.randn([4, 1024, 1], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 2304], dtype=torch.float16, device='cuda'),
    torch.randn([4096, 768], dtype=torch.float16, device='cuda'),
    torch.randn([4, 1024, 1], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float16, device='cuda'),
    torch.randn([4096, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([4096, 768], dtype=torch.float16, device='cuda'),
    torch.randn([4, 1024, 1], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 2304], dtype=torch.float16, device='cuda'),
    torch.randn([4096, 768], dtype=torch.float16, device='cuda'),
    torch.randn([4, 1024, 1], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float16, device='cuda'),
    torch.randn([4096, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([4096, 768], dtype=torch.float16, device='cuda'),
    torch.randn([4, 1024, 1], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 2304], dtype=torch.float16, device='cuda'),
    torch.randn([4096, 768], dtype=torch.float16, device='cuda'),
    torch.randn([4, 1024, 1], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float16, device='cuda'),
    torch.randn([4096, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([4096, 768], dtype=torch.float16, device='cuda'),
    torch.randn([4, 1024, 1], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 2304], dtype=torch.float16, device='cuda'),
    torch.randn([4096, 768], dtype=torch.float16, device='cuda'),
    torch.randn([4, 1024, 1], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float16, device='cuda'),
    torch.randn([4096, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([4096, 768], dtype=torch.float16, device='cuda'),
    torch.randn([4, 1024, 1], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 2304], dtype=torch.float16, device='cuda'),
    torch.randn([4096, 768], dtype=torch.float16, device='cuda'),
    torch.randn([4, 1024, 1], dtype=torch.float32, device='cuda'),
    [4, 1024, 2],  # _shape_param_0
    [1, 2],  # _shape_param_1
    [4, 2],  # _shape_param_2
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
