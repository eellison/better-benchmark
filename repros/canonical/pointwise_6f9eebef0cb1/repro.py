"""
Standalone repro captured via capture_hook.
Label: tlparse_huggingface_s3_g36
Pattern hash: 6f9eebef0cb1
Shape hash: 2bf4669c
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, convert_element_type_4: "f16[512, 30522]", view_5: "f16[512, 768]", view_8: "f16[512, 768]", view_12: "f16[512, 3072]", view_15: "f16[512, 768]", view_27: "f16[512, 768]", view_31: "f16[512, 768]", view_35: "f16[512, 768]", view_38: "f16[512, 768]", view_42: "f16[512, 3072]", view_45: "f16[512, 768]", view_57: "f16[512, 768]", view_61: "f16[512, 768]", view_65: "f16[512, 768]", view_68: "f16[512, 768]", view_72: "f16[512, 3072]", view_75: "f16[512, 768]", view_87: "f16[512, 768]", view_91: "f16[512, 768]", view_95: "f16[512, 768]", view_98: "f16[512, 768]", view_102: "f16[512, 3072]", view_105: "f16[512, 768]", view_117: "f16[512, 768]", view_121: "f16[512, 768]", view_125: "f16[512, 768]", view_128: "f16[512, 768]", view_132: "f16[512, 3072]", view_135: "f16[512, 768]", view_147: "f16[512, 768]", view_151: "f16[512, 768]", view_155: "f16[512, 768]", view_158: "f16[512, 768]", view_162: "f16[512, 3072]", view_165: "f16[512, 768]", view_177: "f16[512, 768]", view_181: "f16[512, 768]", view_185: "f16[512, 768]", view_188: "f16[512, 768]", view_192: "f16[512, 3072]", view_195: "f16[512, 768]", view_207: "f16[512, 768]", view_211: "f16[512, 768]", view_215: "f16[512, 768]", view_218: "f16[512, 768]", view_222: "f16[512, 3072]", view_225: "f16[512, 768]", view_237: "f16[512, 768]", view_241: "f16[512, 768]", view_245: "f16[512, 768]", view_248: "f16[512, 768]", view_252: "f16[512, 3072]", view_255: "f16[512, 768]", view_267: "f16[512, 768]", view_271: "f16[512, 768]", view_275: "f16[512, 768]", view_278: "f16[512, 768]", view_282: "f16[512, 3072]", view_285: "f16[512, 768]", view_297: "f16[512, 768]", view_301: "f16[512, 768]", view_305: "f16[512, 768]", view_308: "f16[512, 768]", view_312: "f16[512, 3072]", view_315: "f16[512, 768]", view_327: "f16[512, 768]", view_331: "f16[512, 768]", view_335: "f16[512, 768]", view_338: "f16[512, 768]", view_342: "f16[512, 3072]", view_345: "f16[512, 768]", bmm_44: "f16[12, 512, 64]", bmm_46: "f16[12, 64, 512]", bmm_47: "f16[12, 512, 64]"):
        # No stacktrace found for following nodes
        permute_default: "f16[30522, 512]" = torch.ops.aten.permute.default(convert_element_type_4, [1, 0]);  convert_element_type_4 = None
        permute_default_1: "f16[768, 512]" = torch.ops.aten.permute.default(view_5, [1, 0]);  view_5 = None
        permute_default_2: "f16[768, 512]" = torch.ops.aten.permute.default(view_8, [1, 0]);  view_8 = None
        permute_default_3: "f16[3072, 512]" = torch.ops.aten.permute.default(view_12, [1, 0]);  view_12 = None
        permute_default_4: "f16[768, 512]" = torch.ops.aten.permute.default(view_15, [1, 0]);  view_15 = None
        permute_default_5: "f16[768, 512]" = torch.ops.aten.permute.default(view_27, [1, 0]);  view_27 = None
        permute_default_6: "f16[768, 512]" = torch.ops.aten.permute.default(view_31, [1, 0]);  view_31 = None
        permute_default_7: "f16[768, 512]" = torch.ops.aten.permute.default(view_35, [1, 0]);  view_35 = None
        permute_default_8: "f16[768, 512]" = torch.ops.aten.permute.default(view_38, [1, 0]);  view_38 = None
        permute_default_9: "f16[3072, 512]" = torch.ops.aten.permute.default(view_42, [1, 0]);  view_42 = None
        permute_default_10: "f16[768, 512]" = torch.ops.aten.permute.default(view_45, [1, 0]);  view_45 = None
        permute_default_11: "f16[768, 512]" = torch.ops.aten.permute.default(view_57, [1, 0]);  view_57 = None
        permute_default_12: "f16[768, 512]" = torch.ops.aten.permute.default(view_61, [1, 0]);  view_61 = None
        permute_default_13: "f16[768, 512]" = torch.ops.aten.permute.default(view_65, [1, 0]);  view_65 = None
        permute_default_14: "f16[768, 512]" = torch.ops.aten.permute.default(view_68, [1, 0]);  view_68 = None
        permute_default_15: "f16[3072, 512]" = torch.ops.aten.permute.default(view_72, [1, 0]);  view_72 = None
        permute_default_16: "f16[768, 512]" = torch.ops.aten.permute.default(view_75, [1, 0]);  view_75 = None
        permute_default_17: "f16[768, 512]" = torch.ops.aten.permute.default(view_87, [1, 0]);  view_87 = None
        permute_default_18: "f16[768, 512]" = torch.ops.aten.permute.default(view_91, [1, 0]);  view_91 = None
        permute_default_19: "f16[768, 512]" = torch.ops.aten.permute.default(view_95, [1, 0]);  view_95 = None
        permute_default_20: "f16[768, 512]" = torch.ops.aten.permute.default(view_98, [1, 0]);  view_98 = None
        permute_default_21: "f16[3072, 512]" = torch.ops.aten.permute.default(view_102, [1, 0]);  view_102 = None
        permute_default_22: "f16[768, 512]" = torch.ops.aten.permute.default(view_105, [1, 0]);  view_105 = None
        permute_default_23: "f16[768, 512]" = torch.ops.aten.permute.default(view_117, [1, 0]);  view_117 = None
        permute_default_24: "f16[768, 512]" = torch.ops.aten.permute.default(view_121, [1, 0]);  view_121 = None
        permute_default_25: "f16[768, 512]" = torch.ops.aten.permute.default(view_125, [1, 0]);  view_125 = None
        permute_default_26: "f16[768, 512]" = torch.ops.aten.permute.default(view_128, [1, 0]);  view_128 = None
        permute_default_27: "f16[3072, 512]" = torch.ops.aten.permute.default(view_132, [1, 0]);  view_132 = None
        permute_default_28: "f16[768, 512]" = torch.ops.aten.permute.default(view_135, [1, 0]);  view_135 = None
        permute_default_29: "f16[768, 512]" = torch.ops.aten.permute.default(view_147, [1, 0]);  view_147 = None
        permute_default_30: "f16[768, 512]" = torch.ops.aten.permute.default(view_151, [1, 0]);  view_151 = None
        permute_default_31: "f16[768, 512]" = torch.ops.aten.permute.default(view_155, [1, 0]);  view_155 = None
        permute_default_32: "f16[768, 512]" = torch.ops.aten.permute.default(view_158, [1, 0]);  view_158 = None
        permute_default_33: "f16[3072, 512]" = torch.ops.aten.permute.default(view_162, [1, 0]);  view_162 = None
        permute_default_34: "f16[768, 512]" = torch.ops.aten.permute.default(view_165, [1, 0]);  view_165 = None
        permute_default_35: "f16[768, 512]" = torch.ops.aten.permute.default(view_177, [1, 0]);  view_177 = None
        permute_default_36: "f16[768, 512]" = torch.ops.aten.permute.default(view_181, [1, 0]);  view_181 = None
        permute_default_37: "f16[768, 512]" = torch.ops.aten.permute.default(view_185, [1, 0]);  view_185 = None
        permute_default_38: "f16[768, 512]" = torch.ops.aten.permute.default(view_188, [1, 0]);  view_188 = None
        permute_default_39: "f16[3072, 512]" = torch.ops.aten.permute.default(view_192, [1, 0]);  view_192 = None
        permute_default_40: "f16[768, 512]" = torch.ops.aten.permute.default(view_195, [1, 0]);  view_195 = None
        permute_default_41: "f16[768, 512]" = torch.ops.aten.permute.default(view_207, [1, 0]);  view_207 = None
        permute_default_42: "f16[768, 512]" = torch.ops.aten.permute.default(view_211, [1, 0]);  view_211 = None
        permute_default_43: "f16[768, 512]" = torch.ops.aten.permute.default(view_215, [1, 0]);  view_215 = None
        permute_default_44: "f16[768, 512]" = torch.ops.aten.permute.default(view_218, [1, 0]);  view_218 = None
        permute_default_45: "f16[3072, 512]" = torch.ops.aten.permute.default(view_222, [1, 0]);  view_222 = None
        permute_default_46: "f16[768, 512]" = torch.ops.aten.permute.default(view_225, [1, 0]);  view_225 = None
        permute_default_47: "f16[768, 512]" = torch.ops.aten.permute.default(view_237, [1, 0]);  view_237 = None
        permute_default_48: "f16[768, 512]" = torch.ops.aten.permute.default(view_241, [1, 0]);  view_241 = None
        permute_default_49: "f16[768, 512]" = torch.ops.aten.permute.default(view_245, [1, 0]);  view_245 = None
        permute_default_50: "f16[768, 512]" = torch.ops.aten.permute.default(view_248, [1, 0]);  view_248 = None
        permute_default_51: "f16[3072, 512]" = torch.ops.aten.permute.default(view_252, [1, 0]);  view_252 = None
        permute_default_52: "f16[768, 512]" = torch.ops.aten.permute.default(view_255, [1, 0]);  view_255 = None
        permute_default_53: "f16[768, 512]" = torch.ops.aten.permute.default(view_267, [1, 0]);  view_267 = None
        permute_default_54: "f16[768, 512]" = torch.ops.aten.permute.default(view_271, [1, 0]);  view_271 = None
        permute_default_55: "f16[768, 512]" = torch.ops.aten.permute.default(view_275, [1, 0]);  view_275 = None
        permute_default_56: "f16[768, 512]" = torch.ops.aten.permute.default(view_278, [1, 0]);  view_278 = None
        permute_default_57: "f16[3072, 512]" = torch.ops.aten.permute.default(view_282, [1, 0]);  view_282 = None
        permute_default_58: "f16[768, 512]" = torch.ops.aten.permute.default(view_285, [1, 0]);  view_285 = None
        permute_default_59: "f16[768, 512]" = torch.ops.aten.permute.default(view_297, [1, 0]);  view_297 = None
        permute_default_60: "f16[768, 512]" = torch.ops.aten.permute.default(view_301, [1, 0]);  view_301 = None
        permute_default_61: "f16[768, 512]" = torch.ops.aten.permute.default(view_305, [1, 0]);  view_305 = None
        permute_default_62: "f16[768, 512]" = torch.ops.aten.permute.default(view_308, [1, 0]);  view_308 = None
        permute_default_63: "f16[3072, 512]" = torch.ops.aten.permute.default(view_312, [1, 0]);  view_312 = None
        permute_default_64: "f16[768, 512]" = torch.ops.aten.permute.default(view_315, [1, 0]);  view_315 = None
        permute_default_65: "f16[768, 512]" = torch.ops.aten.permute.default(view_327, [1, 0]);  view_327 = None
        permute_default_66: "f16[768, 512]" = torch.ops.aten.permute.default(view_331, [1, 0]);  view_331 = None
        permute_default_67: "f16[768, 512]" = torch.ops.aten.permute.default(view_335, [1, 0]);  view_335 = None
        permute_default_68: "f16[768, 512]" = torch.ops.aten.permute.default(view_338, [1, 0]);  view_338 = None
        permute_default_69: "f16[3072, 512]" = torch.ops.aten.permute.default(view_342, [1, 0]);  view_342 = None
        permute_default_70: "f16[768, 512]" = torch.ops.aten.permute.default(view_345, [1, 0]);  view_345 = None
        reshape_default: "f16[1, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_44, [1, 12, 512, 64]);  bmm_44 = None
        reshape_default_1: "f16[1, 12, 64, 512]" = torch.ops.aten.reshape.default(bmm_46, [1, 12, 64, 512]);  bmm_46 = None
        reshape_default_2: "f16[1, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_47, [1, 12, 512, 64]);  bmm_47 = None
        permute_default_71: "f16[1, 12, 512, 64]" = torch.ops.aten.permute.default(reshape_default_1, [0, 1, 3, 2]);  reshape_default_1 = None
        permute_default_72: "f16[1, 512, 12, 64]" = torch.ops.aten.permute.default(reshape_default, [0, 2, 1, 3]);  reshape_default = None
        clone_default: "f16[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_default_72, memory_format = torch.contiguous_format);  permute_default_72 = None
        reshape_default_3: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(clone_default, [1, 512, 768]);  clone_default = None
        reshape_default_4: "f16[512, 768]" = torch.ops.aten.reshape.default(reshape_default_3, [512, 768]);  reshape_default_3 = None
        permute_default_73: "f16[768, 512]" = torch.ops.aten.permute.default(reshape_default_4, [1, 0]);  reshape_default_4 = None
        permute_default_74: "f16[1, 512, 12, 64]" = torch.ops.aten.permute.default(permute_default_71, [0, 2, 1, 3]);  permute_default_71 = None
        reshape_default_5: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(permute_default_74, [1, 512, 768]);  permute_default_74 = None
        reshape_default_6: "f16[512, 768]" = torch.ops.aten.reshape.default(reshape_default_5, [512, 768]);  reshape_default_5 = None
        permute_default_75: "f16[768, 512]" = torch.ops.aten.permute.default(reshape_default_6, [1, 0]);  reshape_default_6 = None
        permute_default_76: "f16[1, 512, 12, 64]" = torch.ops.aten.permute.default(reshape_default_2, [0, 2, 1, 3]);  reshape_default_2 = None
        clone_default_1: "f16[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_default_76, memory_format = torch.contiguous_format);  permute_default_76 = None
        reshape_default_7: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(clone_default_1, [1, 512, 768]);  clone_default_1 = None
        reshape_default_8: "f16[512, 768]" = torch.ops.aten.reshape.default(reshape_default_7, [512, 768]);  reshape_default_7 = None
        permute_default_77: "f16[768, 512]" = torch.ops.aten.permute.default(reshape_default_8, [1, 0]);  reshape_default_8 = None
        return (permute_default, permute_default_1, permute_default_2, permute_default_3, permute_default_4, permute_default_5, permute_default_6, permute_default_7, permute_default_8, permute_default_9, permute_default_10, permute_default_11, permute_default_12, permute_default_13, permute_default_14, permute_default_15, permute_default_16, permute_default_17, permute_default_18, permute_default_19, permute_default_20, permute_default_21, permute_default_22, permute_default_23, permute_default_24, permute_default_25, permute_default_26, permute_default_27, permute_default_28, permute_default_29, permute_default_30, permute_default_31, permute_default_32, permute_default_33, permute_default_34, permute_default_35, permute_default_36, permute_default_37, permute_default_38, permute_default_39, permute_default_40, permute_default_41, permute_default_42, permute_default_43, permute_default_44, permute_default_45, permute_default_46, permute_default_47, permute_default_48, permute_default_49, permute_default_50, permute_default_51, permute_default_52, permute_default_53, permute_default_54, permute_default_55, permute_default_56, permute_default_57, permute_default_58, permute_default_59, permute_default_60, permute_default_61, permute_default_62, permute_default_63, permute_default_64, permute_default_65, permute_default_66, permute_default_67, permute_default_68, permute_default_69, permute_default_70, permute_default_73, permute_default_75, permute_default_77)


def _default_make_inputs():
    return [
    torch.randn([512, 30522], dtype=torch.float16, device='cuda'),
    torch.randn([512, 768], dtype=torch.float16, device='cuda'),
    torch.randn([512, 768], dtype=torch.float16, device='cuda'),
    torch.randn([512, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([512, 768], dtype=torch.float16, device='cuda'),
    torch.randn([512, 768], dtype=torch.float16, device='cuda'),
    torch.randn(393216, dtype=torch.float16, device='cuda').as_strided([512, 768], [1, 512]),  # view_31
    torch.randn([512, 768], dtype=torch.float16, device='cuda'),
    torch.randn([512, 768], dtype=torch.float16, device='cuda'),
    torch.randn([512, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([512, 768], dtype=torch.float16, device='cuda'),
    torch.randn([512, 768], dtype=torch.float16, device='cuda'),
    torch.randn(393216, dtype=torch.float16, device='cuda').as_strided([512, 768], [1, 512]),  # view_61
    torch.randn([512, 768], dtype=torch.float16, device='cuda'),
    torch.randn([512, 768], dtype=torch.float16, device='cuda'),
    torch.randn([512, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([512, 768], dtype=torch.float16, device='cuda'),
    torch.randn([512, 768], dtype=torch.float16, device='cuda'),
    torch.randn(393216, dtype=torch.float16, device='cuda').as_strided([512, 768], [1, 512]),  # view_91
    torch.randn([512, 768], dtype=torch.float16, device='cuda'),
    torch.randn([512, 768], dtype=torch.float16, device='cuda'),
    torch.randn([512, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([512, 768], dtype=torch.float16, device='cuda'),
    torch.randn([512, 768], dtype=torch.float16, device='cuda'),
    torch.randn(393216, dtype=torch.float16, device='cuda').as_strided([512, 768], [1, 512]),  # view_121
    torch.randn([512, 768], dtype=torch.float16, device='cuda'),
    torch.randn([512, 768], dtype=torch.float16, device='cuda'),
    torch.randn([512, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([512, 768], dtype=torch.float16, device='cuda'),
    torch.randn([512, 768], dtype=torch.float16, device='cuda'),
    torch.randn(393216, dtype=torch.float16, device='cuda').as_strided([512, 768], [1, 512]),  # view_151
    torch.randn([512, 768], dtype=torch.float16, device='cuda'),
    torch.randn([512, 768], dtype=torch.float16, device='cuda'),
    torch.randn([512, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([512, 768], dtype=torch.float16, device='cuda'),
    torch.randn([512, 768], dtype=torch.float16, device='cuda'),
    torch.randn(393216, dtype=torch.float16, device='cuda').as_strided([512, 768], [1, 512]),  # view_181
    torch.randn([512, 768], dtype=torch.float16, device='cuda'),
    torch.randn([512, 768], dtype=torch.float16, device='cuda'),
    torch.randn([512, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([512, 768], dtype=torch.float16, device='cuda'),
    torch.randn([512, 768], dtype=torch.float16, device='cuda'),
    torch.randn(393216, dtype=torch.float16, device='cuda').as_strided([512, 768], [1, 512]),  # view_211
    torch.randn([512, 768], dtype=torch.float16, device='cuda'),
    torch.randn([512, 768], dtype=torch.float16, device='cuda'),
    torch.randn([512, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([512, 768], dtype=torch.float16, device='cuda'),
    torch.randn([512, 768], dtype=torch.float16, device='cuda'),
    torch.randn(393216, dtype=torch.float16, device='cuda').as_strided([512, 768], [1, 512]),  # view_241
    torch.randn([512, 768], dtype=torch.float16, device='cuda'),
    torch.randn([512, 768], dtype=torch.float16, device='cuda'),
    torch.randn([512, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([512, 768], dtype=torch.float16, device='cuda'),
    torch.randn([512, 768], dtype=torch.float16, device='cuda'),
    torch.randn(393216, dtype=torch.float16, device='cuda').as_strided([512, 768], [1, 512]),  # view_271
    torch.randn([512, 768], dtype=torch.float16, device='cuda'),
    torch.randn([512, 768], dtype=torch.float16, device='cuda'),
    torch.randn([512, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([512, 768], dtype=torch.float16, device='cuda'),
    torch.randn([512, 768], dtype=torch.float16, device='cuda'),
    torch.randn(393216, dtype=torch.float16, device='cuda').as_strided([512, 768], [1, 512]),  # view_301
    torch.randn([512, 768], dtype=torch.float16, device='cuda'),
    torch.randn([512, 768], dtype=torch.float16, device='cuda'),
    torch.randn([512, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([512, 768], dtype=torch.float16, device='cuda'),
    torch.randn([512, 768], dtype=torch.float16, device='cuda'),
    torch.randn(393216, dtype=torch.float16, device='cuda').as_strided([512, 768], [1, 512]),  # view_331
    torch.randn([512, 768], dtype=torch.float16, device='cuda'),
    torch.randn([512, 768], dtype=torch.float16, device='cuda'),
    torch.randn([512, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([512, 768], dtype=torch.float16, device='cuda'),
    torch.randn([12, 512, 64], dtype=torch.float16, device='cuda'),
    torch.randn([12, 64, 512], dtype=torch.float16, device='cuda'),
    torch.randn([12, 512, 64], dtype=torch.float16, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
