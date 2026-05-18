"""
Standalone repro captured via capture_hook.
Label: tlparse_torchbench_s9_g20
Pattern hash: f4f737ce1c07
Shape hash: 81c1f4f1
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
    def forward(self, mm_12: "f16[32, 512]", mm_13: "f16[32, 512]", permute_243: "f16[512, 512]", rsqrt_50: "f32[32, 77, 1]", permute_241: "f16[2048, 512]", permute_240: "f16[512, 2048]", add_158: "f32[77, 32, 512]", getitem_315: "f32[77, 32, 1]", rsqrt_49: "f32[77, 32, 1]", permute_239: "f16[512, 512]", permute_233: "f16[512, 1536]", add_155: "f32[77, 32, 512]", getitem_304: "f32[77, 32, 1]", rsqrt_48: "f32[77, 32, 1]", permute_232: "f16[2048, 512]", permute_231: "f16[512, 2048]", add_152: "f32[77, 32, 512]", getitem_302: "f32[77, 32, 1]", rsqrt_47: "f32[77, 32, 1]", permute_230: "f16[512, 512]", permute_224: "f16[512, 1536]", add_149: "f32[77, 32, 512]", getitem_291: "f32[77, 32, 1]", rsqrt_46: "f32[77, 32, 1]", permute_223: "f16[2048, 512]", permute_222: "f16[512, 2048]", add_146: "f32[77, 32, 512]", getitem_289: "f32[77, 32, 1]", rsqrt_45: "f32[77, 32, 1]", permute_221: "f16[512, 512]", permute_215: "f16[512, 1536]", add_143: "f32[77, 32, 512]", getitem_278: "f32[77, 32, 1]", rsqrt_44: "f32[77, 32, 1]", permute_214: "f16[2048, 512]", permute_213: "f16[512, 2048]", add_140: "f32[77, 32, 512]", getitem_276: "f32[77, 32, 1]", rsqrt_43: "f32[77, 32, 1]", permute_212: "f16[512, 512]", permute_206: "f16[512, 1536]", add_137: "f32[77, 32, 512]", getitem_265: "f32[77, 32, 1]", rsqrt_42: "f32[77, 32, 1]", permute_205: "f16[2048, 512]", permute_204: "f16[512, 2048]", add_134: "f32[77, 32, 512]", getitem_263: "f32[77, 32, 1]", rsqrt_41: "f32[77, 32, 1]", permute_203: "f16[512, 512]", permute_197: "f16[512, 1536]", add_131: "f32[77, 32, 512]", getitem_252: "f32[77, 32, 1]", rsqrt_40: "f32[77, 32, 1]", permute_196: "f16[2048, 512]", permute_195: "f16[512, 2048]", add_128: "f32[77, 32, 512]", getitem_250: "f32[77, 32, 1]", rsqrt_39: "f32[77, 32, 1]", permute_194: "f16[512, 512]", permute_188: "f16[512, 1536]", add_125: "f32[77, 32, 512]", getitem_239: "f32[77, 32, 1]", rsqrt_38: "f32[77, 32, 1]", permute_187: "f16[2048, 512]", permute_186: "f16[512, 2048]", add_122: "f32[77, 32, 512]", getitem_237: "f32[77, 32, 1]", rsqrt_37: "f32[77, 32, 1]", permute_185: "f16[512, 512]", permute_179: "f16[512, 1536]", add_119: "f32[77, 32, 512]", getitem_226: "f32[77, 32, 1]", rsqrt_36: "f32[77, 32, 1]", permute_178: "f16[2048, 512]", permute_177: "f16[512, 2048]", add_116: "f32[77, 32, 512]", getitem_224: "f32[77, 32, 1]", rsqrt_35: "f32[77, 32, 1]", permute_176: "f16[512, 512]", permute_170: "f16[512, 1536]", add_113: "f32[77, 32, 512]", getitem_213: "f32[77, 32, 1]", rsqrt_34: "f32[77, 32, 1]", permute_169: "f16[2048, 512]", permute_168: "f16[512, 2048]", add_110: "f32[77, 32, 512]", getitem_211: "f32[77, 32, 1]", rsqrt_33: "f32[77, 32, 1]", permute_167: "f16[512, 512]", permute_161: "f16[512, 1536]", add_107: "f32[77, 32, 512]", getitem_200: "f32[77, 32, 1]", rsqrt_32: "f32[77, 32, 1]", permute_160: "f16[2048, 512]", permute_159: "f16[512, 2048]", add_104: "f32[77, 32, 512]", getitem_198: "f32[77, 32, 1]", rsqrt_31: "f32[77, 32, 1]", permute_158: "f16[512, 512]", permute_152: "f16[512, 1536]", add_101: "f32[77, 32, 512]", getitem_187: "f32[77, 32, 1]", rsqrt_30: "f32[77, 32, 1]", permute_151: "f16[2048, 512]", permute_150: "f16[512, 2048]", add_98: "f32[77, 32, 512]", getitem_185: "f32[77, 32, 1]", rsqrt_29: "f32[77, 32, 1]", permute_149: "f16[512, 512]", permute_143: "f16[512, 1536]", add_95: "f32[77, 32, 512]", getitem_174: "f32[77, 32, 1]", rsqrt_28: "f32[77, 32, 1]", permute_142: "f16[2048, 512]", permute_141: "f16[512, 2048]", add_92: "f32[77, 32, 512]", getitem_172: "f32[77, 32, 1]", rsqrt_27: "f32[77, 32, 1]", permute_140: "f16[512, 512]", permute_134: "f16[512, 1536]", convert_element_type_123: "f16[32, 768]", convert_element_type_122: "f16[768, 512]", select_36: "f32[32, 768]", getitem_159: "f32[32, 1]", rsqrt_25: "f32[32, 1]", permute_132: "f16[3072, 768]", permute_131: "f16[768, 3072]", rsqrt_24: "f32[32, 50, 1]", permute_129: "f16[768, 768]", permute_123: "f16[768, 2304]", rsqrt_23: "f32[32, 50, 1]", permute_121: "f16[3072, 768]", permute_120: "f16[768, 3072]", rsqrt_22: "f32[32, 50, 1]", permute_118: "f16[768, 768]", permute_112: "f16[768, 2304]", rsqrt_21: "f32[32, 50, 1]", permute_110: "f16[3072, 768]", permute_109: "f16[768, 3072]", rsqrt_20: "f32[32, 50, 1]", permute_107: "f16[768, 768]", permute_101: "f16[768, 2304]", rsqrt_19: "f32[32, 50, 1]", permute_99: "f16[3072, 768]", permute_98: "f16[768, 3072]", rsqrt_18: "f32[32, 50, 1]", permute_96: "f16[768, 768]", permute_90: "f16[768, 2304]", rsqrt_17: "f32[32, 50, 1]", permute_88: "f16[3072, 768]", permute_87: "f16[768, 3072]", rsqrt_16: "f32[32, 50, 1]", permute_85: "f16[768, 768]", permute_79: "f16[768, 2304]", rsqrt_15: "f32[32, 50, 1]", permute_77: "f16[3072, 768]", permute_76: "f16[768, 3072]", rsqrt_14: "f32[32, 50, 1]", permute_74: "f16[768, 768]", permute_68: "f16[768, 2304]", rsqrt_13: "f32[32, 50, 1]", permute_66: "f16[3072, 768]", permute_65: "f16[768, 3072]", rsqrt_12: "f32[32, 50, 1]", permute_63: "f16[768, 768]", permute_57: "f16[768, 2304]", rsqrt_11: "f32[32, 50, 1]", permute_55: "f16[3072, 768]", permute_54: "f16[768, 3072]", rsqrt_10: "f32[32, 50, 1]", permute_52: "f16[768, 768]", permute_46: "f16[768, 2304]", rsqrt_9: "f32[32, 50, 1]", permute_44: "f16[3072, 768]", permute_43: "f16[768, 3072]", rsqrt_8: "f32[32, 50, 1]", permute_41: "f16[768, 768]", permute_35: "f16[768, 2304]", rsqrt_7: "f32[32, 50, 1]", permute_33: "f16[3072, 768]", permute_32: "f16[768, 3072]", rsqrt_6: "f32[32, 50, 1]", permute_30: "f16[768, 768]", permute_24: "f16[768, 2304]", rsqrt_5: "f32[32, 50, 1]", permute_22: "f16[3072, 768]", permute_21: "f16[768, 3072]", rsqrt_4: "f32[32, 50, 1]", permute_19: "f16[768, 768]", permute_13: "f16[768, 2304]", rsqrt_3: "f32[32, 50, 1]", permute_11: "f16[3072, 768]", permute_10: "f16[768, 3072]", rsqrt_2: "f32[32, 50, 1]", permute_8: "f16[768, 768]", permute_2: "f16[768, 2304]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        convert_element_type_default: "f32[32, 512]" = torch.ops.prims.convert_element_type.default(mm_12, torch.float32)
        pow_tensor_scalar: "f32[32, 512]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_default, 2.0);  convert_element_type_default = None
        sum_dim_int_list: "f32[32, 1]" = torch.ops.aten.sum.dim_IntList(pow_tensor_scalar, [1], True);  pow_tensor_scalar = None
        pow_tensor_scalar_1: "f32[32, 1]" = torch.ops.aten.pow.Tensor_Scalar(sum_dim_int_list, 0.5);  sum_dim_int_list = None
        clamp_min_default: "f32[32, 1]" = torch.ops.aten.clamp_min.default(pow_tensor_scalar_1, 1e-12);  pow_tensor_scalar_1 = None
        expand_default: "f32[32, 512]" = torch.ops.aten.expand.default(clamp_min_default, _shape_param_0);  clamp_min_default = _shape_param_0 = None
        div_tensor: "f32[32, 512]" = torch.ops.aten.div.Tensor(mm_12, expand_default);  mm_12 = expand_default = None
        convert_element_type_default_1: "f32[32, 512]" = torch.ops.prims.convert_element_type.default(mm_13, torch.float32)
        pow_tensor_scalar_2: "f32[32, 512]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_default_1, 2.0);  convert_element_type_default_1 = None
        sum_dim_int_list_1: "f32[32, 1]" = torch.ops.aten.sum.dim_IntList(pow_tensor_scalar_2, [1], True);  pow_tensor_scalar_2 = None
        pow_tensor_scalar_3: "f32[32, 1]" = torch.ops.aten.pow.Tensor_Scalar(sum_dim_int_list_1, 0.5);  sum_dim_int_list_1 = None
        clamp_min_default_1: "f32[32, 1]" = torch.ops.aten.clamp_min.default(pow_tensor_scalar_3, 1e-12);  pow_tensor_scalar_3 = None
        expand_default_1: "f32[32, 512]" = torch.ops.aten.expand.default(clamp_min_default_1, _shape_param_1);  clamp_min_default_1 = _shape_param_1 = None
        div_tensor_1: "f32[32, 512]" = torch.ops.aten.div.Tensor(mm_13, expand_default_1);  mm_13 = expand_default_1 = None
        permute_default: "f16[512, 512]" = torch.ops.aten.permute.default(permute_243, [1, 0]);  permute_243 = None
        div_tensor_2: "f32[32, 77, 1]" = torch.ops.aten.div.Tensor(rsqrt_50, 512);  rsqrt_50 = None
        permute_default_1: "f16[512, 2048]" = torch.ops.aten.permute.default(permute_241, [1, 0]);  permute_241 = None
        permute_default_2: "f16[2048, 512]" = torch.ops.aten.permute.default(permute_240, [1, 0]);  permute_240 = None
        sub_tensor: "f32[77, 32, 512]" = torch.ops.aten.sub.Tensor(add_158, getitem_315);  add_158 = getitem_315 = None
        mul_tensor: "f32[77, 32, 512]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_49);  sub_tensor = None
        div_tensor_3: "f32[77, 32, 1]" = torch.ops.aten.div.Tensor(rsqrt_49, 512);  rsqrt_49 = None
        permute_default_3: "f16[512, 512]" = torch.ops.aten.permute.default(permute_239, [1, 0]);  permute_239 = None
        permute_default_4: "f16[1536, 512]" = torch.ops.aten.permute.default(permute_233, [1, 0]);  permute_233 = None
        sub_tensor_1: "f32[77, 32, 512]" = torch.ops.aten.sub.Tensor(add_155, getitem_304);  add_155 = getitem_304 = None
        mul_tensor_1: "f32[77, 32, 512]" = torch.ops.aten.mul.Tensor(sub_tensor_1, rsqrt_48);  sub_tensor_1 = None
        div_tensor_4: "f32[77, 32, 1]" = torch.ops.aten.div.Tensor(rsqrt_48, 512);  rsqrt_48 = None
        permute_default_5: "f16[512, 2048]" = torch.ops.aten.permute.default(permute_232, [1, 0]);  permute_232 = None
        permute_default_6: "f16[2048, 512]" = torch.ops.aten.permute.default(permute_231, [1, 0]);  permute_231 = None
        sub_tensor_2: "f32[77, 32, 512]" = torch.ops.aten.sub.Tensor(add_152, getitem_302);  add_152 = getitem_302 = None
        mul_tensor_2: "f32[77, 32, 512]" = torch.ops.aten.mul.Tensor(sub_tensor_2, rsqrt_47);  sub_tensor_2 = None
        div_tensor_5: "f32[77, 32, 1]" = torch.ops.aten.div.Tensor(rsqrt_47, 512);  rsqrt_47 = None
        permute_default_7: "f16[512, 512]" = torch.ops.aten.permute.default(permute_230, [1, 0]);  permute_230 = None
        permute_default_8: "f16[1536, 512]" = torch.ops.aten.permute.default(permute_224, [1, 0]);  permute_224 = None
        sub_tensor_3: "f32[77, 32, 512]" = torch.ops.aten.sub.Tensor(add_149, getitem_291);  add_149 = getitem_291 = None
        mul_tensor_3: "f32[77, 32, 512]" = torch.ops.aten.mul.Tensor(sub_tensor_3, rsqrt_46);  sub_tensor_3 = None
        div_tensor_6: "f32[77, 32, 1]" = torch.ops.aten.div.Tensor(rsqrt_46, 512);  rsqrt_46 = None
        permute_default_9: "f16[512, 2048]" = torch.ops.aten.permute.default(permute_223, [1, 0]);  permute_223 = None
        permute_default_10: "f16[2048, 512]" = torch.ops.aten.permute.default(permute_222, [1, 0]);  permute_222 = None
        sub_tensor_4: "f32[77, 32, 512]" = torch.ops.aten.sub.Tensor(add_146, getitem_289);  add_146 = getitem_289 = None
        mul_tensor_4: "f32[77, 32, 512]" = torch.ops.aten.mul.Tensor(sub_tensor_4, rsqrt_45);  sub_tensor_4 = None
        div_tensor_7: "f32[77, 32, 1]" = torch.ops.aten.div.Tensor(rsqrt_45, 512);  rsqrt_45 = None
        permute_default_11: "f16[512, 512]" = torch.ops.aten.permute.default(permute_221, [1, 0]);  permute_221 = None
        permute_default_12: "f16[1536, 512]" = torch.ops.aten.permute.default(permute_215, [1, 0]);  permute_215 = None
        sub_tensor_5: "f32[77, 32, 512]" = torch.ops.aten.sub.Tensor(add_143, getitem_278);  add_143 = getitem_278 = None
        mul_tensor_5: "f32[77, 32, 512]" = torch.ops.aten.mul.Tensor(sub_tensor_5, rsqrt_44);  sub_tensor_5 = None
        div_tensor_8: "f32[77, 32, 1]" = torch.ops.aten.div.Tensor(rsqrt_44, 512);  rsqrt_44 = None
        permute_default_13: "f16[512, 2048]" = torch.ops.aten.permute.default(permute_214, [1, 0]);  permute_214 = None
        permute_default_14: "f16[2048, 512]" = torch.ops.aten.permute.default(permute_213, [1, 0]);  permute_213 = None
        sub_tensor_6: "f32[77, 32, 512]" = torch.ops.aten.sub.Tensor(add_140, getitem_276);  add_140 = getitem_276 = None
        mul_tensor_6: "f32[77, 32, 512]" = torch.ops.aten.mul.Tensor(sub_tensor_6, rsqrt_43);  sub_tensor_6 = None
        div_tensor_9: "f32[77, 32, 1]" = torch.ops.aten.div.Tensor(rsqrt_43, 512);  rsqrt_43 = None
        permute_default_15: "f16[512, 512]" = torch.ops.aten.permute.default(permute_212, [1, 0]);  permute_212 = None
        permute_default_16: "f16[1536, 512]" = torch.ops.aten.permute.default(permute_206, [1, 0]);  permute_206 = None
        sub_tensor_7: "f32[77, 32, 512]" = torch.ops.aten.sub.Tensor(add_137, getitem_265);  add_137 = getitem_265 = None
        mul_tensor_7: "f32[77, 32, 512]" = torch.ops.aten.mul.Tensor(sub_tensor_7, rsqrt_42);  sub_tensor_7 = None
        div_tensor_10: "f32[77, 32, 1]" = torch.ops.aten.div.Tensor(rsqrt_42, 512);  rsqrt_42 = None
        permute_default_17: "f16[512, 2048]" = torch.ops.aten.permute.default(permute_205, [1, 0]);  permute_205 = None
        permute_default_18: "f16[2048, 512]" = torch.ops.aten.permute.default(permute_204, [1, 0]);  permute_204 = None
        sub_tensor_8: "f32[77, 32, 512]" = torch.ops.aten.sub.Tensor(add_134, getitem_263);  add_134 = getitem_263 = None
        mul_tensor_8: "f32[77, 32, 512]" = torch.ops.aten.mul.Tensor(sub_tensor_8, rsqrt_41);  sub_tensor_8 = None
        div_tensor_11: "f32[77, 32, 1]" = torch.ops.aten.div.Tensor(rsqrt_41, 512);  rsqrt_41 = None
        permute_default_19: "f16[512, 512]" = torch.ops.aten.permute.default(permute_203, [1, 0]);  permute_203 = None
        permute_default_20: "f16[1536, 512]" = torch.ops.aten.permute.default(permute_197, [1, 0]);  permute_197 = None
        sub_tensor_9: "f32[77, 32, 512]" = torch.ops.aten.sub.Tensor(add_131, getitem_252);  add_131 = getitem_252 = None
        mul_tensor_9: "f32[77, 32, 512]" = torch.ops.aten.mul.Tensor(sub_tensor_9, rsqrt_40);  sub_tensor_9 = None
        div_tensor_12: "f32[77, 32, 1]" = torch.ops.aten.div.Tensor(rsqrt_40, 512);  rsqrt_40 = None
        permute_default_21: "f16[512, 2048]" = torch.ops.aten.permute.default(permute_196, [1, 0]);  permute_196 = None
        permute_default_22: "f16[2048, 512]" = torch.ops.aten.permute.default(permute_195, [1, 0]);  permute_195 = None
        sub_tensor_10: "f32[77, 32, 512]" = torch.ops.aten.sub.Tensor(add_128, getitem_250);  add_128 = getitem_250 = None
        mul_tensor_10: "f32[77, 32, 512]" = torch.ops.aten.mul.Tensor(sub_tensor_10, rsqrt_39);  sub_tensor_10 = None
        div_tensor_13: "f32[77, 32, 1]" = torch.ops.aten.div.Tensor(rsqrt_39, 512);  rsqrt_39 = None
        permute_default_23: "f16[512, 512]" = torch.ops.aten.permute.default(permute_194, [1, 0]);  permute_194 = None
        permute_default_24: "f16[1536, 512]" = torch.ops.aten.permute.default(permute_188, [1, 0]);  permute_188 = None
        sub_tensor_11: "f32[77, 32, 512]" = torch.ops.aten.sub.Tensor(add_125, getitem_239);  add_125 = getitem_239 = None
        mul_tensor_11: "f32[77, 32, 512]" = torch.ops.aten.mul.Tensor(sub_tensor_11, rsqrt_38);  sub_tensor_11 = None
        div_tensor_14: "f32[77, 32, 1]" = torch.ops.aten.div.Tensor(rsqrt_38, 512);  rsqrt_38 = None
        permute_default_25: "f16[512, 2048]" = torch.ops.aten.permute.default(permute_187, [1, 0]);  permute_187 = None
        permute_default_26: "f16[2048, 512]" = torch.ops.aten.permute.default(permute_186, [1, 0]);  permute_186 = None
        sub_tensor_12: "f32[77, 32, 512]" = torch.ops.aten.sub.Tensor(add_122, getitem_237);  add_122 = getitem_237 = None
        mul_tensor_12: "f32[77, 32, 512]" = torch.ops.aten.mul.Tensor(sub_tensor_12, rsqrt_37);  sub_tensor_12 = None
        div_tensor_15: "f32[77, 32, 1]" = torch.ops.aten.div.Tensor(rsqrt_37, 512);  rsqrt_37 = None
        permute_default_27: "f16[512, 512]" = torch.ops.aten.permute.default(permute_185, [1, 0]);  permute_185 = None
        permute_default_28: "f16[1536, 512]" = torch.ops.aten.permute.default(permute_179, [1, 0]);  permute_179 = None
        sub_tensor_13: "f32[77, 32, 512]" = torch.ops.aten.sub.Tensor(add_119, getitem_226);  add_119 = getitem_226 = None
        mul_tensor_13: "f32[77, 32, 512]" = torch.ops.aten.mul.Tensor(sub_tensor_13, rsqrt_36);  sub_tensor_13 = None
        div_tensor_16: "f32[77, 32, 1]" = torch.ops.aten.div.Tensor(rsqrt_36, 512);  rsqrt_36 = None
        permute_default_29: "f16[512, 2048]" = torch.ops.aten.permute.default(permute_178, [1, 0]);  permute_178 = None
        permute_default_30: "f16[2048, 512]" = torch.ops.aten.permute.default(permute_177, [1, 0]);  permute_177 = None
        sub_tensor_14: "f32[77, 32, 512]" = torch.ops.aten.sub.Tensor(add_116, getitem_224);  add_116 = getitem_224 = None
        mul_tensor_14: "f32[77, 32, 512]" = torch.ops.aten.mul.Tensor(sub_tensor_14, rsqrt_35);  sub_tensor_14 = None
        div_tensor_17: "f32[77, 32, 1]" = torch.ops.aten.div.Tensor(rsqrt_35, 512);  rsqrt_35 = None
        permute_default_31: "f16[512, 512]" = torch.ops.aten.permute.default(permute_176, [1, 0]);  permute_176 = None
        permute_default_32: "f16[1536, 512]" = torch.ops.aten.permute.default(permute_170, [1, 0]);  permute_170 = None
        sub_tensor_15: "f32[77, 32, 512]" = torch.ops.aten.sub.Tensor(add_113, getitem_213);  add_113 = getitem_213 = None
        mul_tensor_15: "f32[77, 32, 512]" = torch.ops.aten.mul.Tensor(sub_tensor_15, rsqrt_34);  sub_tensor_15 = None
        div_tensor_18: "f32[77, 32, 1]" = torch.ops.aten.div.Tensor(rsqrt_34, 512);  rsqrt_34 = None
        permute_default_33: "f16[512, 2048]" = torch.ops.aten.permute.default(permute_169, [1, 0]);  permute_169 = None
        permute_default_34: "f16[2048, 512]" = torch.ops.aten.permute.default(permute_168, [1, 0]);  permute_168 = None
        sub_tensor_16: "f32[77, 32, 512]" = torch.ops.aten.sub.Tensor(add_110, getitem_211);  add_110 = getitem_211 = None
        mul_tensor_16: "f32[77, 32, 512]" = torch.ops.aten.mul.Tensor(sub_tensor_16, rsqrt_33);  sub_tensor_16 = None
        div_tensor_19: "f32[77, 32, 1]" = torch.ops.aten.div.Tensor(rsqrt_33, 512);  rsqrt_33 = None
        permute_default_35: "f16[512, 512]" = torch.ops.aten.permute.default(permute_167, [1, 0]);  permute_167 = None
        permute_default_36: "f16[1536, 512]" = torch.ops.aten.permute.default(permute_161, [1, 0]);  permute_161 = None
        sub_tensor_17: "f32[77, 32, 512]" = torch.ops.aten.sub.Tensor(add_107, getitem_200);  add_107 = getitem_200 = None
        mul_tensor_17: "f32[77, 32, 512]" = torch.ops.aten.mul.Tensor(sub_tensor_17, rsqrt_32);  sub_tensor_17 = None
        div_tensor_20: "f32[77, 32, 1]" = torch.ops.aten.div.Tensor(rsqrt_32, 512);  rsqrt_32 = None
        permute_default_37: "f16[512, 2048]" = torch.ops.aten.permute.default(permute_160, [1, 0]);  permute_160 = None
        permute_default_38: "f16[2048, 512]" = torch.ops.aten.permute.default(permute_159, [1, 0]);  permute_159 = None
        sub_tensor_18: "f32[77, 32, 512]" = torch.ops.aten.sub.Tensor(add_104, getitem_198);  add_104 = getitem_198 = None
        mul_tensor_18: "f32[77, 32, 512]" = torch.ops.aten.mul.Tensor(sub_tensor_18, rsqrt_31);  sub_tensor_18 = None
        div_tensor_21: "f32[77, 32, 1]" = torch.ops.aten.div.Tensor(rsqrt_31, 512);  rsqrt_31 = None
        permute_default_39: "f16[512, 512]" = torch.ops.aten.permute.default(permute_158, [1, 0]);  permute_158 = None
        permute_default_40: "f16[1536, 512]" = torch.ops.aten.permute.default(permute_152, [1, 0]);  permute_152 = None
        sub_tensor_19: "f32[77, 32, 512]" = torch.ops.aten.sub.Tensor(add_101, getitem_187);  add_101 = getitem_187 = None
        mul_tensor_19: "f32[77, 32, 512]" = torch.ops.aten.mul.Tensor(sub_tensor_19, rsqrt_30);  sub_tensor_19 = None
        div_tensor_22: "f32[77, 32, 1]" = torch.ops.aten.div.Tensor(rsqrt_30, 512);  rsqrt_30 = None
        permute_default_41: "f16[512, 2048]" = torch.ops.aten.permute.default(permute_151, [1, 0]);  permute_151 = None
        permute_default_42: "f16[2048, 512]" = torch.ops.aten.permute.default(permute_150, [1, 0]);  permute_150 = None
        sub_tensor_20: "f32[77, 32, 512]" = torch.ops.aten.sub.Tensor(add_98, getitem_185);  add_98 = getitem_185 = None
        mul_tensor_20: "f32[77, 32, 512]" = torch.ops.aten.mul.Tensor(sub_tensor_20, rsqrt_29);  sub_tensor_20 = None
        div_tensor_23: "f32[77, 32, 1]" = torch.ops.aten.div.Tensor(rsqrt_29, 512);  rsqrt_29 = None
        permute_default_43: "f16[512, 512]" = torch.ops.aten.permute.default(permute_149, [1, 0]);  permute_149 = None
        permute_default_44: "f16[1536, 512]" = torch.ops.aten.permute.default(permute_143, [1, 0]);  permute_143 = None
        sub_tensor_21: "f32[77, 32, 512]" = torch.ops.aten.sub.Tensor(add_95, getitem_174);  add_95 = getitem_174 = None
        mul_tensor_21: "f32[77, 32, 512]" = torch.ops.aten.mul.Tensor(sub_tensor_21, rsqrt_28);  sub_tensor_21 = None
        div_tensor_24: "f32[77, 32, 1]" = torch.ops.aten.div.Tensor(rsqrt_28, 512);  rsqrt_28 = None
        permute_default_45: "f16[512, 2048]" = torch.ops.aten.permute.default(permute_142, [1, 0]);  permute_142 = None
        permute_default_46: "f16[2048, 512]" = torch.ops.aten.permute.default(permute_141, [1, 0]);  permute_141 = None
        sub_tensor_22: "f32[77, 32, 512]" = torch.ops.aten.sub.Tensor(add_92, getitem_172);  add_92 = getitem_172 = None
        mul_tensor_22: "f32[77, 32, 512]" = torch.ops.aten.mul.Tensor(sub_tensor_22, rsqrt_27);  sub_tensor_22 = None
        div_tensor_25: "f32[77, 32, 1]" = torch.ops.aten.div.Tensor(rsqrt_27, 512);  rsqrt_27 = None
        permute_default_47: "f16[512, 512]" = torch.ops.aten.permute.default(permute_140, [1, 0]);  permute_140 = None
        permute_default_48: "f16[1536, 512]" = torch.ops.aten.permute.default(permute_134, [1, 0]);  permute_134 = None
        permute_default_49: "f16[768, 32]" = torch.ops.aten.permute.default(convert_element_type_123, [1, 0]);  convert_element_type_123 = None
        permute_default_50: "f16[512, 768]" = torch.ops.aten.permute.default(convert_element_type_122, [1, 0]);  convert_element_type_122 = None
        sub_tensor_23: "f32[32, 768]" = torch.ops.aten.sub.Tensor(select_36, getitem_159);  select_36 = getitem_159 = None
        mul_tensor_23: "f32[32, 768]" = torch.ops.aten.mul.Tensor(sub_tensor_23, rsqrt_25);  sub_tensor_23 = None
        div_tensor_26: "f32[32, 1]" = torch.ops.aten.div.Tensor(rsqrt_25, 768);  rsqrt_25 = None
        permute_default_51: "f16[768, 3072]" = torch.ops.aten.permute.default(permute_132, [1, 0]);  permute_132 = None
        permute_default_52: "f16[3072, 768]" = torch.ops.aten.permute.default(permute_131, [1, 0]);  permute_131 = None
        div_tensor_27: "f32[32, 50, 1]" = torch.ops.aten.div.Tensor(rsqrt_24, 768);  rsqrt_24 = None
        permute_default_53: "f16[768, 768]" = torch.ops.aten.permute.default(permute_129, [1, 0]);  permute_129 = None
        permute_default_54: "f16[2304, 768]" = torch.ops.aten.permute.default(permute_123, [1, 0]);  permute_123 = None
        div_tensor_28: "f32[32, 50, 1]" = torch.ops.aten.div.Tensor(rsqrt_23, 768);  rsqrt_23 = None
        permute_default_55: "f16[768, 3072]" = torch.ops.aten.permute.default(permute_121, [1, 0]);  permute_121 = None
        permute_default_56: "f16[3072, 768]" = torch.ops.aten.permute.default(permute_120, [1, 0]);  permute_120 = None
        div_tensor_29: "f32[32, 50, 1]" = torch.ops.aten.div.Tensor(rsqrt_22, 768);  rsqrt_22 = None
        permute_default_57: "f16[768, 768]" = torch.ops.aten.permute.default(permute_118, [1, 0]);  permute_118 = None
        permute_default_58: "f16[2304, 768]" = torch.ops.aten.permute.default(permute_112, [1, 0]);  permute_112 = None
        div_tensor_30: "f32[32, 50, 1]" = torch.ops.aten.div.Tensor(rsqrt_21, 768);  rsqrt_21 = None
        permute_default_59: "f16[768, 3072]" = torch.ops.aten.permute.default(permute_110, [1, 0]);  permute_110 = None
        permute_default_60: "f16[3072, 768]" = torch.ops.aten.permute.default(permute_109, [1, 0]);  permute_109 = None
        div_tensor_31: "f32[32, 50, 1]" = torch.ops.aten.div.Tensor(rsqrt_20, 768);  rsqrt_20 = None
        permute_default_61: "f16[768, 768]" = torch.ops.aten.permute.default(permute_107, [1, 0]);  permute_107 = None
        permute_default_62: "f16[2304, 768]" = torch.ops.aten.permute.default(permute_101, [1, 0]);  permute_101 = None
        div_tensor_32: "f32[32, 50, 1]" = torch.ops.aten.div.Tensor(rsqrt_19, 768);  rsqrt_19 = None
        permute_default_63: "f16[768, 3072]" = torch.ops.aten.permute.default(permute_99, [1, 0]);  permute_99 = None
        permute_default_64: "f16[3072, 768]" = torch.ops.aten.permute.default(permute_98, [1, 0]);  permute_98 = None
        div_tensor_33: "f32[32, 50, 1]" = torch.ops.aten.div.Tensor(rsqrt_18, 768);  rsqrt_18 = None
        permute_default_65: "f16[768, 768]" = torch.ops.aten.permute.default(permute_96, [1, 0]);  permute_96 = None
        permute_default_66: "f16[2304, 768]" = torch.ops.aten.permute.default(permute_90, [1, 0]);  permute_90 = None
        div_tensor_34: "f32[32, 50, 1]" = torch.ops.aten.div.Tensor(rsqrt_17, 768);  rsqrt_17 = None
        permute_default_67: "f16[768, 3072]" = torch.ops.aten.permute.default(permute_88, [1, 0]);  permute_88 = None
        permute_default_68: "f16[3072, 768]" = torch.ops.aten.permute.default(permute_87, [1, 0]);  permute_87 = None
        div_tensor_35: "f32[32, 50, 1]" = torch.ops.aten.div.Tensor(rsqrt_16, 768);  rsqrt_16 = None
        permute_default_69: "f16[768, 768]" = torch.ops.aten.permute.default(permute_85, [1, 0]);  permute_85 = None
        permute_default_70: "f16[2304, 768]" = torch.ops.aten.permute.default(permute_79, [1, 0]);  permute_79 = None
        div_tensor_36: "f32[32, 50, 1]" = torch.ops.aten.div.Tensor(rsqrt_15, 768);  rsqrt_15 = None
        permute_default_71: "f16[768, 3072]" = torch.ops.aten.permute.default(permute_77, [1, 0]);  permute_77 = None
        permute_default_72: "f16[3072, 768]" = torch.ops.aten.permute.default(permute_76, [1, 0]);  permute_76 = None
        div_tensor_37: "f32[32, 50, 1]" = torch.ops.aten.div.Tensor(rsqrt_14, 768);  rsqrt_14 = None
        permute_default_73: "f16[768, 768]" = torch.ops.aten.permute.default(permute_74, [1, 0]);  permute_74 = None
        permute_default_74: "f16[2304, 768]" = torch.ops.aten.permute.default(permute_68, [1, 0]);  permute_68 = None
        div_tensor_38: "f32[32, 50, 1]" = torch.ops.aten.div.Tensor(rsqrt_13, 768);  rsqrt_13 = None
        permute_default_75: "f16[768, 3072]" = torch.ops.aten.permute.default(permute_66, [1, 0]);  permute_66 = None
        permute_default_76: "f16[3072, 768]" = torch.ops.aten.permute.default(permute_65, [1, 0]);  permute_65 = None
        div_tensor_39: "f32[32, 50, 1]" = torch.ops.aten.div.Tensor(rsqrt_12, 768);  rsqrt_12 = None
        permute_default_77: "f16[768, 768]" = torch.ops.aten.permute.default(permute_63, [1, 0]);  permute_63 = None
        permute_default_78: "f16[2304, 768]" = torch.ops.aten.permute.default(permute_57, [1, 0]);  permute_57 = None
        div_tensor_40: "f32[32, 50, 1]" = torch.ops.aten.div.Tensor(rsqrt_11, 768);  rsqrt_11 = None
        permute_default_79: "f16[768, 3072]" = torch.ops.aten.permute.default(permute_55, [1, 0]);  permute_55 = None
        permute_default_80: "f16[3072, 768]" = torch.ops.aten.permute.default(permute_54, [1, 0]);  permute_54 = None
        div_tensor_41: "f32[32, 50, 1]" = torch.ops.aten.div.Tensor(rsqrt_10, 768);  rsqrt_10 = None
        permute_default_81: "f16[768, 768]" = torch.ops.aten.permute.default(permute_52, [1, 0]);  permute_52 = None
        permute_default_82: "f16[2304, 768]" = torch.ops.aten.permute.default(permute_46, [1, 0]);  permute_46 = None
        div_tensor_42: "f32[32, 50, 1]" = torch.ops.aten.div.Tensor(rsqrt_9, 768);  rsqrt_9 = None
        permute_default_83: "f16[768, 3072]" = torch.ops.aten.permute.default(permute_44, [1, 0]);  permute_44 = None
        permute_default_84: "f16[3072, 768]" = torch.ops.aten.permute.default(permute_43, [1, 0]);  permute_43 = None
        div_tensor_43: "f32[32, 50, 1]" = torch.ops.aten.div.Tensor(rsqrt_8, 768);  rsqrt_8 = None
        permute_default_85: "f16[768, 768]" = torch.ops.aten.permute.default(permute_41, [1, 0]);  permute_41 = None
        permute_default_86: "f16[2304, 768]" = torch.ops.aten.permute.default(permute_35, [1, 0]);  permute_35 = None
        div_tensor_44: "f32[32, 50, 1]" = torch.ops.aten.div.Tensor(rsqrt_7, 768);  rsqrt_7 = None
        permute_default_87: "f16[768, 3072]" = torch.ops.aten.permute.default(permute_33, [1, 0]);  permute_33 = None
        permute_default_88: "f16[3072, 768]" = torch.ops.aten.permute.default(permute_32, [1, 0]);  permute_32 = None
        div_tensor_45: "f32[32, 50, 1]" = torch.ops.aten.div.Tensor(rsqrt_6, 768);  rsqrt_6 = None
        permute_default_89: "f16[768, 768]" = torch.ops.aten.permute.default(permute_30, [1, 0]);  permute_30 = None
        permute_default_90: "f16[2304, 768]" = torch.ops.aten.permute.default(permute_24, [1, 0]);  permute_24 = None
        div_tensor_46: "f32[32, 50, 1]" = torch.ops.aten.div.Tensor(rsqrt_5, 768);  rsqrt_5 = None
        permute_default_91: "f16[768, 3072]" = torch.ops.aten.permute.default(permute_22, [1, 0]);  permute_22 = None
        permute_default_92: "f16[3072, 768]" = torch.ops.aten.permute.default(permute_21, [1, 0]);  permute_21 = None
        div_tensor_47: "f32[32, 50, 1]" = torch.ops.aten.div.Tensor(rsqrt_4, 768);  rsqrt_4 = None
        permute_default_93: "f16[768, 768]" = torch.ops.aten.permute.default(permute_19, [1, 0]);  permute_19 = None
        permute_default_94: "f16[2304, 768]" = torch.ops.aten.permute.default(permute_13, [1, 0]);  permute_13 = None
        div_tensor_48: "f32[32, 50, 1]" = torch.ops.aten.div.Tensor(rsqrt_3, 768);  rsqrt_3 = None
        permute_default_95: "f16[768, 3072]" = torch.ops.aten.permute.default(permute_11, [1, 0]);  permute_11 = None
        permute_default_96: "f16[3072, 768]" = torch.ops.aten.permute.default(permute_10, [1, 0]);  permute_10 = None
        div_tensor_49: "f32[32, 50, 1]" = torch.ops.aten.div.Tensor(rsqrt_2, 768);  rsqrt_2 = None
        permute_default_97: "f16[768, 768]" = torch.ops.aten.permute.default(permute_8, [1, 0]);  permute_8 = None
        permute_default_98: "f16[2304, 768]" = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None
        return (div_tensor, div_tensor_1, permute_default, div_tensor_2, permute_default_1, permute_default_2, mul_tensor, div_tensor_3, permute_default_3, permute_default_4, mul_tensor_1, div_tensor_4, permute_default_5, permute_default_6, mul_tensor_2, div_tensor_5, permute_default_7, permute_default_8, mul_tensor_3, div_tensor_6, permute_default_9, permute_default_10, mul_tensor_4, div_tensor_7, permute_default_11, permute_default_12, mul_tensor_5, div_tensor_8, permute_default_13, permute_default_14, mul_tensor_6, div_tensor_9, permute_default_15, permute_default_16, mul_tensor_7, div_tensor_10, permute_default_17, permute_default_18, mul_tensor_8, div_tensor_11, permute_default_19, permute_default_20, mul_tensor_9, div_tensor_12, permute_default_21, permute_default_22, mul_tensor_10, div_tensor_13, permute_default_23, permute_default_24, mul_tensor_11, div_tensor_14, permute_default_25, permute_default_26, mul_tensor_12, div_tensor_15, permute_default_27, permute_default_28, mul_tensor_13, div_tensor_16, permute_default_29, permute_default_30, mul_tensor_14, div_tensor_17, permute_default_31, permute_default_32, mul_tensor_15, div_tensor_18, permute_default_33, permute_default_34, mul_tensor_16, div_tensor_19, permute_default_35, permute_default_36, mul_tensor_17, div_tensor_20, permute_default_37, permute_default_38, mul_tensor_18, div_tensor_21, permute_default_39, permute_default_40, mul_tensor_19, div_tensor_22, permute_default_41, permute_default_42, mul_tensor_20, div_tensor_23, permute_default_43, permute_default_44, mul_tensor_21, div_tensor_24, permute_default_45, permute_default_46, mul_tensor_22, div_tensor_25, permute_default_47, permute_default_48, permute_default_49, permute_default_50, mul_tensor_23, div_tensor_26, permute_default_51, permute_default_52, div_tensor_27, permute_default_53, permute_default_54, div_tensor_28, permute_default_55, permute_default_56, div_tensor_29, permute_default_57, permute_default_58, div_tensor_30, permute_default_59, permute_default_60, div_tensor_31, permute_default_61, permute_default_62, div_tensor_32, permute_default_63, permute_default_64, div_tensor_33, permute_default_65, permute_default_66, div_tensor_34, permute_default_67, permute_default_68, div_tensor_35, permute_default_69, permute_default_70, div_tensor_36, permute_default_71, permute_default_72, div_tensor_37, permute_default_73, permute_default_74, div_tensor_38, permute_default_75, permute_default_76, div_tensor_39, permute_default_77, permute_default_78, div_tensor_40, permute_default_79, permute_default_80, div_tensor_41, permute_default_81, permute_default_82, div_tensor_42, permute_default_83, permute_default_84, div_tensor_43, permute_default_85, permute_default_86, div_tensor_44, permute_default_87, permute_default_88, div_tensor_45, permute_default_89, permute_default_90, div_tensor_46, permute_default_91, permute_default_92, div_tensor_47, permute_default_93, permute_default_94, div_tensor_48, permute_default_95, permute_default_96, div_tensor_49, permute_default_97, permute_default_98)


def _default_make_inputs():
    return [
    torch.randn([32, 512], dtype=torch.float16, device='cuda'),
    torch.randn([32, 512], dtype=torch.float16, device='cuda'),
    torch.randn(262144, dtype=torch.float16, device='cuda').as_strided([512, 512], [1, 512]),  # permute_243
    torch.randn([32, 77, 1], dtype=torch.float32, device='cuda'),
    torch.randn(1048576, dtype=torch.float16, device='cuda').as_strided([2048, 512], [1, 2048]),  # permute_241
    torch.randn(1048576, dtype=torch.float16, device='cuda').as_strided([512, 2048], [1, 512]),  # permute_240
    torch.randn(1261568, dtype=torch.float32, device='cuda').as_strided([77, 32, 512], [512, 39424, 1]),  # add_158
    torch.randn([77, 32, 1], dtype=torch.float32, device='cuda'),
    torch.randn([77, 32, 1], dtype=torch.float32, device='cuda'),
    torch.randn(262144, dtype=torch.float16, device='cuda').as_strided([512, 512], [1, 512]),  # permute_239
    torch.randn(786432, dtype=torch.float16, device='cuda').as_strided([512, 1536], [1, 512]),  # permute_233
    torch.randn(1261568, dtype=torch.float32, device='cuda').as_strided([77, 32, 512], [512, 39424, 1]),  # add_155
    torch.randn([77, 32, 1], dtype=torch.float32, device='cuda'),
    torch.randn([77, 32, 1], dtype=torch.float32, device='cuda'),
    torch.randn(1048576, dtype=torch.float16, device='cuda').as_strided([2048, 512], [1, 2048]),  # permute_232
    torch.randn(1048576, dtype=torch.float16, device='cuda').as_strided([512, 2048], [1, 512]),  # permute_231
    torch.randn(1261568, dtype=torch.float32, device='cuda').as_strided([77, 32, 512], [512, 39424, 1]),  # add_152
    torch.randn([77, 32, 1], dtype=torch.float32, device='cuda'),
    torch.randn([77, 32, 1], dtype=torch.float32, device='cuda'),
    torch.randn(262144, dtype=torch.float16, device='cuda').as_strided([512, 512], [1, 512]),  # permute_230
    torch.randn(786432, dtype=torch.float16, device='cuda').as_strided([512, 1536], [1, 512]),  # permute_224
    torch.randn(1261568, dtype=torch.float32, device='cuda').as_strided([77, 32, 512], [512, 39424, 1]),  # add_149
    torch.randn([77, 32, 1], dtype=torch.float32, device='cuda'),
    torch.randn([77, 32, 1], dtype=torch.float32, device='cuda'),
    torch.randn(1048576, dtype=torch.float16, device='cuda').as_strided([2048, 512], [1, 2048]),  # permute_223
    torch.randn(1048576, dtype=torch.float16, device='cuda').as_strided([512, 2048], [1, 512]),  # permute_222
    torch.randn(1261568, dtype=torch.float32, device='cuda').as_strided([77, 32, 512], [512, 39424, 1]),  # add_146
    torch.randn([77, 32, 1], dtype=torch.float32, device='cuda'),
    torch.randn([77, 32, 1], dtype=torch.float32, device='cuda'),
    torch.randn(262144, dtype=torch.float16, device='cuda').as_strided([512, 512], [1, 512]),  # permute_221
    torch.randn(786432, dtype=torch.float16, device='cuda').as_strided([512, 1536], [1, 512]),  # permute_215
    torch.randn(1261568, dtype=torch.float32, device='cuda').as_strided([77, 32, 512], [512, 39424, 1]),  # add_143
    torch.randn([77, 32, 1], dtype=torch.float32, device='cuda'),
    torch.randn([77, 32, 1], dtype=torch.float32, device='cuda'),
    torch.randn(1048576, dtype=torch.float16, device='cuda').as_strided([2048, 512], [1, 2048]),  # permute_214
    torch.randn(1048576, dtype=torch.float16, device='cuda').as_strided([512, 2048], [1, 512]),  # permute_213
    torch.randn(1261568, dtype=torch.float32, device='cuda').as_strided([77, 32, 512], [512, 39424, 1]),  # add_140
    torch.randn([77, 32, 1], dtype=torch.float32, device='cuda'),
    torch.randn([77, 32, 1], dtype=torch.float32, device='cuda'),
    torch.randn(262144, dtype=torch.float16, device='cuda').as_strided([512, 512], [1, 512]),  # permute_212
    torch.randn(786432, dtype=torch.float16, device='cuda').as_strided([512, 1536], [1, 512]),  # permute_206
    torch.randn(1261568, dtype=torch.float32, device='cuda').as_strided([77, 32, 512], [512, 39424, 1]),  # add_137
    torch.randn([77, 32, 1], dtype=torch.float32, device='cuda'),
    torch.randn([77, 32, 1], dtype=torch.float32, device='cuda'),
    torch.randn(1048576, dtype=torch.float16, device='cuda').as_strided([2048, 512], [1, 2048]),  # permute_205
    torch.randn(1048576, dtype=torch.float16, device='cuda').as_strided([512, 2048], [1, 512]),  # permute_204
    torch.randn(1261568, dtype=torch.float32, device='cuda').as_strided([77, 32, 512], [512, 39424, 1]),  # add_134
    torch.randn([77, 32, 1], dtype=torch.float32, device='cuda'),
    torch.randn([77, 32, 1], dtype=torch.float32, device='cuda'),
    torch.randn(262144, dtype=torch.float16, device='cuda').as_strided([512, 512], [1, 512]),  # permute_203
    torch.randn(786432, dtype=torch.float16, device='cuda').as_strided([512, 1536], [1, 512]),  # permute_197
    torch.randn(1261568, dtype=torch.float32, device='cuda').as_strided([77, 32, 512], [512, 39424, 1]),  # add_131
    torch.randn([77, 32, 1], dtype=torch.float32, device='cuda'),
    torch.randn([77, 32, 1], dtype=torch.float32, device='cuda'),
    torch.randn(1048576, dtype=torch.float16, device='cuda').as_strided([2048, 512], [1, 2048]),  # permute_196
    torch.randn(1048576, dtype=torch.float16, device='cuda').as_strided([512, 2048], [1, 512]),  # permute_195
    torch.randn(1261568, dtype=torch.float32, device='cuda').as_strided([77, 32, 512], [512, 39424, 1]),  # add_128
    torch.randn([77, 32, 1], dtype=torch.float32, device='cuda'),
    torch.randn([77, 32, 1], dtype=torch.float32, device='cuda'),
    torch.randn(262144, dtype=torch.float16, device='cuda').as_strided([512, 512], [1, 512]),  # permute_194
    torch.randn(786432, dtype=torch.float16, device='cuda').as_strided([512, 1536], [1, 512]),  # permute_188
    torch.randn(1261568, dtype=torch.float32, device='cuda').as_strided([77, 32, 512], [512, 39424, 1]),  # add_125
    torch.randn([77, 32, 1], dtype=torch.float32, device='cuda'),
    torch.randn([77, 32, 1], dtype=torch.float32, device='cuda'),
    torch.randn(1048576, dtype=torch.float16, device='cuda').as_strided([2048, 512], [1, 2048]),  # permute_187
    torch.randn(1048576, dtype=torch.float16, device='cuda').as_strided([512, 2048], [1, 512]),  # permute_186
    torch.randn(1261568, dtype=torch.float32, device='cuda').as_strided([77, 32, 512], [512, 39424, 1]),  # add_122
    torch.randn([77, 32, 1], dtype=torch.float32, device='cuda'),
    torch.randn([77, 32, 1], dtype=torch.float32, device='cuda'),
    torch.randn(262144, dtype=torch.float16, device='cuda').as_strided([512, 512], [1, 512]),  # permute_185
    torch.randn(786432, dtype=torch.float16, device='cuda').as_strided([512, 1536], [1, 512]),  # permute_179
    torch.randn(1261568, dtype=torch.float32, device='cuda').as_strided([77, 32, 512], [512, 39424, 1]),  # add_119
    torch.randn([77, 32, 1], dtype=torch.float32, device='cuda'),
    torch.randn([77, 32, 1], dtype=torch.float32, device='cuda'),
    torch.randn(1048576, dtype=torch.float16, device='cuda').as_strided([2048, 512], [1, 2048]),  # permute_178
    torch.randn(1048576, dtype=torch.float16, device='cuda').as_strided([512, 2048], [1, 512]),  # permute_177
    torch.randn(1261568, dtype=torch.float32, device='cuda').as_strided([77, 32, 512], [512, 39424, 1]),  # add_116
    torch.randn([77, 32, 1], dtype=torch.float32, device='cuda'),
    torch.randn([77, 32, 1], dtype=torch.float32, device='cuda'),
    torch.randn(262144, dtype=torch.float16, device='cuda').as_strided([512, 512], [1, 512]),  # permute_176
    torch.randn(786432, dtype=torch.float16, device='cuda').as_strided([512, 1536], [1, 512]),  # permute_170
    torch.randn(1261568, dtype=torch.float32, device='cuda').as_strided([77, 32, 512], [512, 39424, 1]),  # add_113
    torch.randn([77, 32, 1], dtype=torch.float32, device='cuda'),
    torch.randn([77, 32, 1], dtype=torch.float32, device='cuda'),
    torch.randn(1048576, dtype=torch.float16, device='cuda').as_strided([2048, 512], [1, 2048]),  # permute_169
    torch.randn(1048576, dtype=torch.float16, device='cuda').as_strided([512, 2048], [1, 512]),  # permute_168
    torch.randn(1261568, dtype=torch.float32, device='cuda').as_strided([77, 32, 512], [512, 39424, 1]),  # add_110
    torch.randn([77, 32, 1], dtype=torch.float32, device='cuda'),
    torch.randn([77, 32, 1], dtype=torch.float32, device='cuda'),
    torch.randn(262144, dtype=torch.float16, device='cuda').as_strided([512, 512], [1, 512]),  # permute_167
    torch.randn(786432, dtype=torch.float16, device='cuda').as_strided([512, 1536], [1, 512]),  # permute_161
    torch.randn(1261568, dtype=torch.float32, device='cuda').as_strided([77, 32, 512], [512, 39424, 1]),  # add_107
    torch.randn([77, 32, 1], dtype=torch.float32, device='cuda'),
    torch.randn([77, 32, 1], dtype=torch.float32, device='cuda'),
    torch.randn(1048576, dtype=torch.float16, device='cuda').as_strided([2048, 512], [1, 2048]),  # permute_160
    torch.randn(1048576, dtype=torch.float16, device='cuda').as_strided([512, 2048], [1, 512]),  # permute_159
    torch.randn(1261568, dtype=torch.float32, device='cuda').as_strided([77, 32, 512], [512, 39424, 1]),  # add_104
    torch.randn([77, 32, 1], dtype=torch.float32, device='cuda'),
    torch.randn([77, 32, 1], dtype=torch.float32, device='cuda'),
    torch.randn(262144, dtype=torch.float16, device='cuda').as_strided([512, 512], [1, 512]),  # permute_158
    torch.randn(786432, dtype=torch.float16, device='cuda').as_strided([512, 1536], [1, 512]),  # permute_152
    torch.randn(1261568, dtype=torch.float32, device='cuda').as_strided([77, 32, 512], [512, 39424, 1]),  # add_101
    torch.randn([77, 32, 1], dtype=torch.float32, device='cuda'),
    torch.randn([77, 32, 1], dtype=torch.float32, device='cuda'),
    torch.randn(1048576, dtype=torch.float16, device='cuda').as_strided([2048, 512], [1, 2048]),  # permute_151
    torch.randn(1048576, dtype=torch.float16, device='cuda').as_strided([512, 2048], [1, 512]),  # permute_150
    torch.randn(1261568, dtype=torch.float32, device='cuda').as_strided([77, 32, 512], [512, 39424, 1]),  # add_98
    torch.randn([77, 32, 1], dtype=torch.float32, device='cuda'),
    torch.randn([77, 32, 1], dtype=torch.float32, device='cuda'),
    torch.randn(262144, dtype=torch.float16, device='cuda').as_strided([512, 512], [1, 512]),  # permute_149
    torch.randn(786432, dtype=torch.float16, device='cuda').as_strided([512, 1536], [1, 512]),  # permute_143
    torch.randn(1261568, dtype=torch.float32, device='cuda').as_strided([77, 32, 512], [512, 39424, 1]),  # add_95
    torch.randn([77, 32, 1], dtype=torch.float32, device='cuda'),
    torch.randn([77, 32, 1], dtype=torch.float32, device='cuda'),
    torch.randn(1048576, dtype=torch.float16, device='cuda').as_strided([2048, 512], [1, 2048]),  # permute_142
    torch.randn(1048576, dtype=torch.float16, device='cuda').as_strided([512, 2048], [1, 512]),  # permute_141
    torch.randn(1261568, dtype=torch.float32, device='cuda').as_strided([77, 32, 512], [512, 39424, 1]),  # add_92
    torch.randn([77, 32, 1], dtype=torch.float32, device='cuda'),
    torch.randn([77, 32, 1], dtype=torch.float32, device='cuda'),
    torch.randn(262144, dtype=torch.float16, device='cuda').as_strided([512, 512], [1, 512]),  # permute_140
    torch.randn(786432, dtype=torch.float16, device='cuda').as_strided([512, 1536], [1, 512]),  # permute_134
    torch.randn([32, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 512], dtype=torch.float16, device='cuda'),
    torch.randn(1191168, dtype=torch.float32, device='cuda').as_strided([32, 768], [38400, 1]),  # select_36
    torch.randn([32, 1], dtype=torch.float32, device='cuda'),
    torch.randn([32, 1], dtype=torch.float32, device='cuda'),
    torch.randn(2359296, dtype=torch.float16, device='cuda').as_strided([3072, 768], [1, 3072]),  # permute_132
    torch.randn(2359296, dtype=torch.float16, device='cuda').as_strided([768, 3072], [1, 768]),  # permute_131
    torch.randn([32, 50, 1], dtype=torch.float32, device='cuda'),
    torch.randn(589824, dtype=torch.float16, device='cuda').as_strided([768, 768], [1, 768]),  # permute_129
    torch.randn(1769472, dtype=torch.float16, device='cuda').as_strided([768, 2304], [1, 768]),  # permute_123
    torch.randn([32, 50, 1], dtype=torch.float32, device='cuda'),
    torch.randn(2359296, dtype=torch.float16, device='cuda').as_strided([3072, 768], [1, 3072]),  # permute_121
    torch.randn(2359296, dtype=torch.float16, device='cuda').as_strided([768, 3072], [1, 768]),  # permute_120
    torch.randn([32, 50, 1], dtype=torch.float32, device='cuda'),
    torch.randn(589824, dtype=torch.float16, device='cuda').as_strided([768, 768], [1, 768]),  # permute_118
    torch.randn(1769472, dtype=torch.float16, device='cuda').as_strided([768, 2304], [1, 768]),  # permute_112
    torch.randn([32, 50, 1], dtype=torch.float32, device='cuda'),
    torch.randn(2359296, dtype=torch.float16, device='cuda').as_strided([3072, 768], [1, 3072]),  # permute_110
    torch.randn(2359296, dtype=torch.float16, device='cuda').as_strided([768, 3072], [1, 768]),  # permute_109
    torch.randn([32, 50, 1], dtype=torch.float32, device='cuda'),
    torch.randn(589824, dtype=torch.float16, device='cuda').as_strided([768, 768], [1, 768]),  # permute_107
    torch.randn(1769472, dtype=torch.float16, device='cuda').as_strided([768, 2304], [1, 768]),  # permute_101
    torch.randn([32, 50, 1], dtype=torch.float32, device='cuda'),
    torch.randn(2359296, dtype=torch.float16, device='cuda').as_strided([3072, 768], [1, 3072]),  # permute_99
    torch.randn(2359296, dtype=torch.float16, device='cuda').as_strided([768, 3072], [1, 768]),  # permute_98
    torch.randn([32, 50, 1], dtype=torch.float32, device='cuda'),
    torch.randn(589824, dtype=torch.float16, device='cuda').as_strided([768, 768], [1, 768]),  # permute_96
    torch.randn(1769472, dtype=torch.float16, device='cuda').as_strided([768, 2304], [1, 768]),  # permute_90
    torch.randn([32, 50, 1], dtype=torch.float32, device='cuda'),
    torch.randn(2359296, dtype=torch.float16, device='cuda').as_strided([3072, 768], [1, 3072]),  # permute_88
    torch.randn(2359296, dtype=torch.float16, device='cuda').as_strided([768, 3072], [1, 768]),  # permute_87
    torch.randn([32, 50, 1], dtype=torch.float32, device='cuda'),
    torch.randn(589824, dtype=torch.float16, device='cuda').as_strided([768, 768], [1, 768]),  # permute_85
    torch.randn(1769472, dtype=torch.float16, device='cuda').as_strided([768, 2304], [1, 768]),  # permute_79
    torch.randn([32, 50, 1], dtype=torch.float32, device='cuda'),
    torch.randn(2359296, dtype=torch.float16, device='cuda').as_strided([3072, 768], [1, 3072]),  # permute_77
    torch.randn(2359296, dtype=torch.float16, device='cuda').as_strided([768, 3072], [1, 768]),  # permute_76
    torch.randn([32, 50, 1], dtype=torch.float32, device='cuda'),
    torch.randn(589824, dtype=torch.float16, device='cuda').as_strided([768, 768], [1, 768]),  # permute_74
    torch.randn(1769472, dtype=torch.float16, device='cuda').as_strided([768, 2304], [1, 768]),  # permute_68
    torch.randn([32, 50, 1], dtype=torch.float32, device='cuda'),
    torch.randn(2359296, dtype=torch.float16, device='cuda').as_strided([3072, 768], [1, 3072]),  # permute_66
    torch.randn(2359296, dtype=torch.float16, device='cuda').as_strided([768, 3072], [1, 768]),  # permute_65
    torch.randn([32, 50, 1], dtype=torch.float32, device='cuda'),
    torch.randn(589824, dtype=torch.float16, device='cuda').as_strided([768, 768], [1, 768]),  # permute_63
    torch.randn(1769472, dtype=torch.float16, device='cuda').as_strided([768, 2304], [1, 768]),  # permute_57
    torch.randn([32, 50, 1], dtype=torch.float32, device='cuda'),
    torch.randn(2359296, dtype=torch.float16, device='cuda').as_strided([3072, 768], [1, 3072]),  # permute_55
    torch.randn(2359296, dtype=torch.float16, device='cuda').as_strided([768, 3072], [1, 768]),  # permute_54
    torch.randn([32, 50, 1], dtype=torch.float32, device='cuda'),
    torch.randn(589824, dtype=torch.float16, device='cuda').as_strided([768, 768], [1, 768]),  # permute_52
    torch.randn(1769472, dtype=torch.float16, device='cuda').as_strided([768, 2304], [1, 768]),  # permute_46
    torch.randn([32, 50, 1], dtype=torch.float32, device='cuda'),
    torch.randn(2359296, dtype=torch.float16, device='cuda').as_strided([3072, 768], [1, 3072]),  # permute_44
    torch.randn(2359296, dtype=torch.float16, device='cuda').as_strided([768, 3072], [1, 768]),  # permute_43
    torch.randn([32, 50, 1], dtype=torch.float32, device='cuda'),
    torch.randn(589824, dtype=torch.float16, device='cuda').as_strided([768, 768], [1, 768]),  # permute_41
    torch.randn(1769472, dtype=torch.float16, device='cuda').as_strided([768, 2304], [1, 768]),  # permute_35
    torch.randn([32, 50, 1], dtype=torch.float32, device='cuda'),
    torch.randn(2359296, dtype=torch.float16, device='cuda').as_strided([3072, 768], [1, 3072]),  # permute_33
    torch.randn(2359296, dtype=torch.float16, device='cuda').as_strided([768, 3072], [1, 768]),  # permute_32
    torch.randn([32, 50, 1], dtype=torch.float32, device='cuda'),
    torch.randn(589824, dtype=torch.float16, device='cuda').as_strided([768, 768], [1, 768]),  # permute_30
    torch.randn(1769472, dtype=torch.float16, device='cuda').as_strided([768, 2304], [1, 768]),  # permute_24
    torch.randn([32, 50, 1], dtype=torch.float32, device='cuda'),
    torch.randn(2359296, dtype=torch.float16, device='cuda').as_strided([3072, 768], [1, 3072]),  # permute_22
    torch.randn(2359296, dtype=torch.float16, device='cuda').as_strided([768, 3072], [1, 768]),  # permute_21
    torch.randn([32, 50, 1], dtype=torch.float32, device='cuda'),
    torch.randn(589824, dtype=torch.float16, device='cuda').as_strided([768, 768], [1, 768]),  # permute_19
    torch.randn(1769472, dtype=torch.float16, device='cuda').as_strided([768, 2304], [1, 768]),  # permute_13
    torch.randn([32, 50, 1], dtype=torch.float32, device='cuda'),
    torch.randn(2359296, dtype=torch.float16, device='cuda').as_strided([3072, 768], [1, 3072]),  # permute_11
    torch.randn(2359296, dtype=torch.float16, device='cuda').as_strided([768, 3072], [1, 768]),  # permute_10
    torch.randn([32, 50, 1], dtype=torch.float32, device='cuda'),
    torch.randn(589824, dtype=torch.float16, device='cuda').as_strided([768, 768], [1, 768]),  # permute_8
    torch.randn(1769472, dtype=torch.float16, device='cuda').as_strided([768, 2304], [1, 768]),  # permute_2,
    [32, 512],  # _shape_param_0
    [32, 512],  # _shape_param_1
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
