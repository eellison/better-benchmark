"""
Standalone repro captured via capture_hook.
Label: tlparse_torchbench_s3_g174
Pattern hash: 9506d7dc941f
Shape hash: a2e10534
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
    def forward(self, convert_element_type_2: "f16[4, 768]", view_1: "f16[1904, 768]", view_5: "f16[1904, 3072]", view_8: "f16[1904, 768]", view_21: "f16[1904, 768]", view_24: "f16[1904, 768]", view_27: "f16[1904, 768]", view_30: "f16[1904, 768]", view_34: "f16[1904, 3072]", view_37: "f16[1904, 768]", view_50: "f16[1904, 768]", view_53: "f16[1904, 768]", view_56: "f16[1904, 768]", view_59: "f16[1904, 768]", view_63: "f16[1904, 3072]", view_66: "f16[1904, 768]", view_79: "f16[1904, 768]", view_82: "f16[1904, 768]", view_85: "f16[1904, 768]", view_88: "f16[1904, 768]", view_92: "f16[1904, 3072]", view_95: "f16[1904, 768]", view_108: "f16[1904, 768]", view_111: "f16[1904, 768]", view_114: "f16[1904, 768]", view_117: "f16[1904, 768]", view_121: "f16[1904, 3072]", view_124: "f16[1904, 768]", view_137: "f16[1904, 768]", view_140: "f16[1904, 768]", view_143: "f16[1904, 768]", view_146: "f16[1904, 768]", view_150: "f16[1904, 3072]", view_153: "f16[1904, 768]", view_166: "f16[1904, 768]", view_169: "f16[1904, 768]", view_172: "f16[1904, 768]", view_175: "f16[1904, 768]", view_179: "f16[1904, 3072]", view_182: "f16[1904, 768]", view_195: "f16[1904, 768]", view_198: "f16[1904, 768]", view_201: "f16[1904, 768]", view_204: "f16[1904, 768]", view_208: "f16[1904, 3072]", view_211: "f16[1904, 768]", view_224: "f16[1904, 768]", view_227: "f16[1904, 768]", view_230: "f16[1904, 768]", view_233: "f16[1904, 768]", view_237: "f16[1904, 3072]", view_240: "f16[1904, 768]", view_253: "f16[1904, 768]", view_256: "f16[1904, 768]", view_259: "f16[1904, 768]", view_262: "f16[1904, 768]", view_266: "f16[1904, 3072]", view_269: "f16[1904, 768]", view_282: "f16[1904, 768]", view_285: "f16[1904, 768]", view_288: "f16[1904, 768]", view_291: "f16[1904, 768]", view_295: "f16[1904, 3072]", view_298: "f16[1904, 768]", view_311: "f16[1904, 768]", view_314: "f16[1904, 768]", view_317: "f16[1904, 768]", view_320: "f16[1904, 768]", view_324: "f16[1904, 3072]", view_327: "f16[1904, 768]", bmm_44: "f16[48, 476, 64]", bmm_46: "f16[48, 64, 476]", bmm_47: "f16[48, 476, 64]"):
        # No stacktrace found for following nodes
        permute_default: "f16[768, 4]" = torch.ops.aten.permute.default(convert_element_type_2, [1, 0]);  convert_element_type_2 = None
        permute_default_1: "f16[768, 1904]" = torch.ops.aten.permute.default(view_1, [1, 0]);  view_1 = None
        permute_default_2: "f16[3072, 1904]" = torch.ops.aten.permute.default(view_5, [1, 0]);  view_5 = None
        permute_default_3: "f16[768, 1904]" = torch.ops.aten.permute.default(view_8, [1, 0]);  view_8 = None
        permute_default_4: "f16[768, 1904]" = torch.ops.aten.permute.default(view_21, [1, 0]);  view_21 = None
        permute_default_5: "f16[768, 1904]" = torch.ops.aten.permute.default(view_24, [1, 0]);  view_24 = None
        permute_default_6: "f16[768, 1904]" = torch.ops.aten.permute.default(view_27, [1, 0]);  view_27 = None
        permute_default_7: "f16[768, 1904]" = torch.ops.aten.permute.default(view_30, [1, 0]);  view_30 = None
        permute_default_8: "f16[3072, 1904]" = torch.ops.aten.permute.default(view_34, [1, 0]);  view_34 = None
        permute_default_9: "f16[768, 1904]" = torch.ops.aten.permute.default(view_37, [1, 0]);  view_37 = None
        permute_default_10: "f16[768, 1904]" = torch.ops.aten.permute.default(view_50, [1, 0]);  view_50 = None
        permute_default_11: "f16[768, 1904]" = torch.ops.aten.permute.default(view_53, [1, 0]);  view_53 = None
        permute_default_12: "f16[768, 1904]" = torch.ops.aten.permute.default(view_56, [1, 0]);  view_56 = None
        permute_default_13: "f16[768, 1904]" = torch.ops.aten.permute.default(view_59, [1, 0]);  view_59 = None
        permute_default_14: "f16[3072, 1904]" = torch.ops.aten.permute.default(view_63, [1, 0]);  view_63 = None
        permute_default_15: "f16[768, 1904]" = torch.ops.aten.permute.default(view_66, [1, 0]);  view_66 = None
        permute_default_16: "f16[768, 1904]" = torch.ops.aten.permute.default(view_79, [1, 0]);  view_79 = None
        permute_default_17: "f16[768, 1904]" = torch.ops.aten.permute.default(view_82, [1, 0]);  view_82 = None
        permute_default_18: "f16[768, 1904]" = torch.ops.aten.permute.default(view_85, [1, 0]);  view_85 = None
        permute_default_19: "f16[768, 1904]" = torch.ops.aten.permute.default(view_88, [1, 0]);  view_88 = None
        permute_default_20: "f16[3072, 1904]" = torch.ops.aten.permute.default(view_92, [1, 0]);  view_92 = None
        permute_default_21: "f16[768, 1904]" = torch.ops.aten.permute.default(view_95, [1, 0]);  view_95 = None
        permute_default_22: "f16[768, 1904]" = torch.ops.aten.permute.default(view_108, [1, 0]);  view_108 = None
        permute_default_23: "f16[768, 1904]" = torch.ops.aten.permute.default(view_111, [1, 0]);  view_111 = None
        permute_default_24: "f16[768, 1904]" = torch.ops.aten.permute.default(view_114, [1, 0]);  view_114 = None
        permute_default_25: "f16[768, 1904]" = torch.ops.aten.permute.default(view_117, [1, 0]);  view_117 = None
        permute_default_26: "f16[3072, 1904]" = torch.ops.aten.permute.default(view_121, [1, 0]);  view_121 = None
        permute_default_27: "f16[768, 1904]" = torch.ops.aten.permute.default(view_124, [1, 0]);  view_124 = None
        permute_default_28: "f16[768, 1904]" = torch.ops.aten.permute.default(view_137, [1, 0]);  view_137 = None
        permute_default_29: "f16[768, 1904]" = torch.ops.aten.permute.default(view_140, [1, 0]);  view_140 = None
        permute_default_30: "f16[768, 1904]" = torch.ops.aten.permute.default(view_143, [1, 0]);  view_143 = None
        permute_default_31: "f16[768, 1904]" = torch.ops.aten.permute.default(view_146, [1, 0]);  view_146 = None
        permute_default_32: "f16[3072, 1904]" = torch.ops.aten.permute.default(view_150, [1, 0]);  view_150 = None
        permute_default_33: "f16[768, 1904]" = torch.ops.aten.permute.default(view_153, [1, 0]);  view_153 = None
        permute_default_34: "f16[768, 1904]" = torch.ops.aten.permute.default(view_166, [1, 0]);  view_166 = None
        permute_default_35: "f16[768, 1904]" = torch.ops.aten.permute.default(view_169, [1, 0]);  view_169 = None
        permute_default_36: "f16[768, 1904]" = torch.ops.aten.permute.default(view_172, [1, 0]);  view_172 = None
        permute_default_37: "f16[768, 1904]" = torch.ops.aten.permute.default(view_175, [1, 0]);  view_175 = None
        permute_default_38: "f16[3072, 1904]" = torch.ops.aten.permute.default(view_179, [1, 0]);  view_179 = None
        permute_default_39: "f16[768, 1904]" = torch.ops.aten.permute.default(view_182, [1, 0]);  view_182 = None
        permute_default_40: "f16[768, 1904]" = torch.ops.aten.permute.default(view_195, [1, 0]);  view_195 = None
        permute_default_41: "f16[768, 1904]" = torch.ops.aten.permute.default(view_198, [1, 0]);  view_198 = None
        permute_default_42: "f16[768, 1904]" = torch.ops.aten.permute.default(view_201, [1, 0]);  view_201 = None
        permute_default_43: "f16[768, 1904]" = torch.ops.aten.permute.default(view_204, [1, 0]);  view_204 = None
        permute_default_44: "f16[3072, 1904]" = torch.ops.aten.permute.default(view_208, [1, 0]);  view_208 = None
        permute_default_45: "f16[768, 1904]" = torch.ops.aten.permute.default(view_211, [1, 0]);  view_211 = None
        permute_default_46: "f16[768, 1904]" = torch.ops.aten.permute.default(view_224, [1, 0]);  view_224 = None
        permute_default_47: "f16[768, 1904]" = torch.ops.aten.permute.default(view_227, [1, 0]);  view_227 = None
        permute_default_48: "f16[768, 1904]" = torch.ops.aten.permute.default(view_230, [1, 0]);  view_230 = None
        permute_default_49: "f16[768, 1904]" = torch.ops.aten.permute.default(view_233, [1, 0]);  view_233 = None
        permute_default_50: "f16[3072, 1904]" = torch.ops.aten.permute.default(view_237, [1, 0]);  view_237 = None
        permute_default_51: "f16[768, 1904]" = torch.ops.aten.permute.default(view_240, [1, 0]);  view_240 = None
        permute_default_52: "f16[768, 1904]" = torch.ops.aten.permute.default(view_253, [1, 0]);  view_253 = None
        permute_default_53: "f16[768, 1904]" = torch.ops.aten.permute.default(view_256, [1, 0]);  view_256 = None
        permute_default_54: "f16[768, 1904]" = torch.ops.aten.permute.default(view_259, [1, 0]);  view_259 = None
        permute_default_55: "f16[768, 1904]" = torch.ops.aten.permute.default(view_262, [1, 0]);  view_262 = None
        permute_default_56: "f16[3072, 1904]" = torch.ops.aten.permute.default(view_266, [1, 0]);  view_266 = None
        permute_default_57: "f16[768, 1904]" = torch.ops.aten.permute.default(view_269, [1, 0]);  view_269 = None
        permute_default_58: "f16[768, 1904]" = torch.ops.aten.permute.default(view_282, [1, 0]);  view_282 = None
        permute_default_59: "f16[768, 1904]" = torch.ops.aten.permute.default(view_285, [1, 0]);  view_285 = None
        permute_default_60: "f16[768, 1904]" = torch.ops.aten.permute.default(view_288, [1, 0]);  view_288 = None
        permute_default_61: "f16[768, 1904]" = torch.ops.aten.permute.default(view_291, [1, 0]);  view_291 = None
        permute_default_62: "f16[3072, 1904]" = torch.ops.aten.permute.default(view_295, [1, 0]);  view_295 = None
        permute_default_63: "f16[768, 1904]" = torch.ops.aten.permute.default(view_298, [1, 0]);  view_298 = None
        permute_default_64: "f16[768, 1904]" = torch.ops.aten.permute.default(view_311, [1, 0]);  view_311 = None
        permute_default_65: "f16[768, 1904]" = torch.ops.aten.permute.default(view_314, [1, 0]);  view_314 = None
        permute_default_66: "f16[768, 1904]" = torch.ops.aten.permute.default(view_317, [1, 0]);  view_317 = None
        permute_default_67: "f16[768, 1904]" = torch.ops.aten.permute.default(view_320, [1, 0]);  view_320 = None
        permute_default_68: "f16[3072, 1904]" = torch.ops.aten.permute.default(view_324, [1, 0]);  view_324 = None
        permute_default_69: "f16[768, 1904]" = torch.ops.aten.permute.default(view_327, [1, 0]);  view_327 = None
        reshape_default: "f16[4, 12, 476, 64]" = torch.ops.aten.reshape.default(bmm_44, [4, 12, 476, 64]);  bmm_44 = None
        reshape_default_1: "f16[4, 12, 64, 476]" = torch.ops.aten.reshape.default(bmm_46, [4, 12, 64, 476]);  bmm_46 = None
        reshape_default_2: "f16[4, 12, 476, 64]" = torch.ops.aten.reshape.default(bmm_47, [4, 12, 476, 64]);  bmm_47 = None
        permute_default_70: "f16[4, 12, 476, 64]" = torch.ops.aten.permute.default(reshape_default_1, [0, 1, 3, 2]);  reshape_default_1 = None
        permute_default_71: "f16[4, 476, 12, 64]" = torch.ops.aten.permute.default(reshape_default, [0, 2, 1, 3]);  reshape_default = None
        clone_default: "f16[4, 476, 12, 64]" = torch.ops.aten.clone.default(permute_default_71, memory_format = torch.contiguous_format);  permute_default_71 = None
        reshape_default_3: "f16[4, 476, 768]" = torch.ops.aten.reshape.default(clone_default, [4, 476, 768]);  clone_default = None
        permute_default_72: "f16[4, 476, 12, 64]" = torch.ops.aten.permute.default(permute_default_70, [0, 2, 1, 3]);  permute_default_70 = None
        reshape_default_4: "f16[4, 476, 768]" = torch.ops.aten.reshape.default(permute_default_72, [4, 476, 768]);  permute_default_72 = None
        permute_default_73: "f16[4, 476, 12, 64]" = torch.ops.aten.permute.default(reshape_default_2, [0, 2, 1, 3]);  reshape_default_2 = None
        clone_default_1: "f16[4, 476, 12, 64]" = torch.ops.aten.clone.default(permute_default_73, memory_format = torch.contiguous_format);  permute_default_73 = None
        reshape_default_5: "f16[4, 476, 768]" = torch.ops.aten.reshape.default(clone_default_1, [4, 476, 768]);  clone_default_1 = None
        reshape_default_6: "f16[1904, 768]" = torch.ops.aten.reshape.default(reshape_default_3, [1904, 768]);  reshape_default_3 = None
        permute_default_74: "f16[768, 1904]" = torch.ops.aten.permute.default(reshape_default_6, [1, 0]);  reshape_default_6 = None
        clone_default_2: "f16[4, 476, 768]" = torch.ops.aten.clone.default(reshape_default_4, memory_format = torch.contiguous_format);  reshape_default_4 = None
        reshape_default_7: "f16[1904, 768]" = torch.ops.aten.reshape.default(clone_default_2, [1904, 768]);  clone_default_2 = None
        permute_default_75: "f16[768, 1904]" = torch.ops.aten.permute.default(reshape_default_7, [1, 0]);  reshape_default_7 = None
        reshape_default_8: "f16[1904, 768]" = torch.ops.aten.reshape.default(reshape_default_5, [1904, 768]);  reshape_default_5 = None
        permute_default_76: "f16[768, 1904]" = torch.ops.aten.permute.default(reshape_default_8, [1, 0]);  reshape_default_8 = None
        return (permute_default, permute_default_1, permute_default_2, permute_default_3, permute_default_4, permute_default_5, permute_default_6, permute_default_7, permute_default_8, permute_default_9, permute_default_10, permute_default_11, permute_default_12, permute_default_13, permute_default_14, permute_default_15, permute_default_16, permute_default_17, permute_default_18, permute_default_19, permute_default_20, permute_default_21, permute_default_22, permute_default_23, permute_default_24, permute_default_25, permute_default_26, permute_default_27, permute_default_28, permute_default_29, permute_default_30, permute_default_31, permute_default_32, permute_default_33, permute_default_34, permute_default_35, permute_default_36, permute_default_37, permute_default_38, permute_default_39, permute_default_40, permute_default_41, permute_default_42, permute_default_43, permute_default_44, permute_default_45, permute_default_46, permute_default_47, permute_default_48, permute_default_49, permute_default_50, permute_default_51, permute_default_52, permute_default_53, permute_default_54, permute_default_55, permute_default_56, permute_default_57, permute_default_58, permute_default_59, permute_default_60, permute_default_61, permute_default_62, permute_default_63, permute_default_64, permute_default_65, permute_default_66, permute_default_67, permute_default_68, permute_default_69, permute_default_74, permute_default_75, permute_default_76)


def _default_make_inputs():
    return [
    torch.randn([4, 768], dtype=torch.float16, device='cuda'),
    torch.randn([1904, 768], dtype=torch.float16, device='cuda'),
    torch.randn([1904, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([1904, 768], dtype=torch.float16, device='cuda'),
    torch.randn([1904, 768], dtype=torch.float16, device='cuda'),
    torch.randn([1904, 768], dtype=torch.float16, device='cuda'),
    torch.randn([1904, 768], dtype=torch.float16, device='cuda'),
    torch.randn([1904, 768], dtype=torch.float16, device='cuda'),
    torch.randn([1904, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([1904, 768], dtype=torch.float16, device='cuda'),
    torch.randn([1904, 768], dtype=torch.float16, device='cuda'),
    torch.randn([1904, 768], dtype=torch.float16, device='cuda'),
    torch.randn([1904, 768], dtype=torch.float16, device='cuda'),
    torch.randn([1904, 768], dtype=torch.float16, device='cuda'),
    torch.randn([1904, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([1904, 768], dtype=torch.float16, device='cuda'),
    torch.randn([1904, 768], dtype=torch.float16, device='cuda'),
    torch.randn([1904, 768], dtype=torch.float16, device='cuda'),
    torch.randn([1904, 768], dtype=torch.float16, device='cuda'),
    torch.randn([1904, 768], dtype=torch.float16, device='cuda'),
    torch.randn([1904, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([1904, 768], dtype=torch.float16, device='cuda'),
    torch.randn([1904, 768], dtype=torch.float16, device='cuda'),
    torch.randn([1904, 768], dtype=torch.float16, device='cuda'),
    torch.randn([1904, 768], dtype=torch.float16, device='cuda'),
    torch.randn([1904, 768], dtype=torch.float16, device='cuda'),
    torch.randn([1904, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([1904, 768], dtype=torch.float16, device='cuda'),
    torch.randn([1904, 768], dtype=torch.float16, device='cuda'),
    torch.randn([1904, 768], dtype=torch.float16, device='cuda'),
    torch.randn([1904, 768], dtype=torch.float16, device='cuda'),
    torch.randn([1904, 768], dtype=torch.float16, device='cuda'),
    torch.randn([1904, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([1904, 768], dtype=torch.float16, device='cuda'),
    torch.randn([1904, 768], dtype=torch.float16, device='cuda'),
    torch.randn([1904, 768], dtype=torch.float16, device='cuda'),
    torch.randn([1904, 768], dtype=torch.float16, device='cuda'),
    torch.randn([1904, 768], dtype=torch.float16, device='cuda'),
    torch.randn([1904, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([1904, 768], dtype=torch.float16, device='cuda'),
    torch.randn([1904, 768], dtype=torch.float16, device='cuda'),
    torch.randn([1904, 768], dtype=torch.float16, device='cuda'),
    torch.randn([1904, 768], dtype=torch.float16, device='cuda'),
    torch.randn([1904, 768], dtype=torch.float16, device='cuda'),
    torch.randn([1904, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([1904, 768], dtype=torch.float16, device='cuda'),
    torch.randn([1904, 768], dtype=torch.float16, device='cuda'),
    torch.randn([1904, 768], dtype=torch.float16, device='cuda'),
    torch.randn([1904, 768], dtype=torch.float16, device='cuda'),
    torch.randn([1904, 768], dtype=torch.float16, device='cuda'),
    torch.randn([1904, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([1904, 768], dtype=torch.float16, device='cuda'),
    torch.randn([1904, 768], dtype=torch.float16, device='cuda'),
    torch.randn([1904, 768], dtype=torch.float16, device='cuda'),
    torch.randn([1904, 768], dtype=torch.float16, device='cuda'),
    torch.randn([1904, 768], dtype=torch.float16, device='cuda'),
    torch.randn([1904, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([1904, 768], dtype=torch.float16, device='cuda'),
    torch.randn([1904, 768], dtype=torch.float16, device='cuda'),
    torch.randn([1904, 768], dtype=torch.float16, device='cuda'),
    torch.randn([1904, 768], dtype=torch.float16, device='cuda'),
    torch.randn([1904, 768], dtype=torch.float16, device='cuda'),
    torch.randn([1904, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([1904, 768], dtype=torch.float16, device='cuda'),
    torch.randn([1904, 768], dtype=torch.float16, device='cuda'),
    torch.randn([1904, 768], dtype=torch.float16, device='cuda'),
    torch.randn([1904, 768], dtype=torch.float16, device='cuda'),
    torch.randn([1904, 768], dtype=torch.float16, device='cuda'),
    torch.randn([1904, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([1904, 768], dtype=torch.float16, device='cuda'),
    torch.randn([48, 476, 64], dtype=torch.float16, device='cuda'),
    torch.randn([48, 64, 476], dtype=torch.float16, device='cuda'),
    torch.randn([48, 476, 64], dtype=torch.float16, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
