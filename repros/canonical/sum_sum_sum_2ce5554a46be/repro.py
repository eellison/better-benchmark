"""
Standalone repro captured via capture_hook.
Label: tlparse_torchbench_s9_g53
Pattern hash: 2ce5554a46be
Shape hash: 851f8cc6
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
    def forward(self, add: "f16[32, 512]", view: "f16[2464, 512]", view_4: "f16[2464, 2048]", view_7: "f16[2464, 512]", arg346_1: "f16[32, 8, 77, 64]", view_18: "f16[2464, 1536]", view_21: "f16[2464, 512]", view_25: "f16[2464, 2048]", view_28: "f16[2464, 512]", arg335_1: "f16[32, 8, 77, 64]", view_39: "f16[2464, 1536]", view_42: "f16[2464, 512]", view_46: "f16[2464, 2048]", view_49: "f16[2464, 512]", arg324_1: "f16[32, 8, 77, 64]", view_60: "f16[2464, 1536]", view_63: "f16[2464, 512]", view_67: "f16[2464, 2048]", view_70: "f16[2464, 512]", arg313_1: "f16[32, 8, 77, 64]", view_81: "f16[2464, 1536]", view_84: "f16[2464, 512]", view_88: "f16[2464, 2048]", view_91: "f16[2464, 512]", arg302_1: "f16[32, 8, 77, 64]", view_102: "f16[2464, 1536]", view_105: "f16[2464, 512]", view_109: "f16[2464, 2048]", view_112: "f16[2464, 512]", arg291_1: "f16[32, 8, 77, 64]", view_123: "f16[2464, 1536]", view_126: "f16[2464, 512]", view_130: "f16[2464, 2048]", view_133: "f16[2464, 512]", arg280_1: "f16[32, 8, 77, 64]", view_144: "f16[2464, 1536]", view_147: "f16[2464, 512]", view_151: "f16[2464, 2048]", view_154: "f16[2464, 512]", arg269_1: "f16[32, 8, 77, 64]", view_165: "f16[2464, 1536]", view_168: "f16[2464, 512]", view_172: "f16[2464, 2048]", view_175: "f16[2464, 512]", arg258_1: "f16[32, 8, 77, 64]", view_186: "f16[2464, 1536]", view_189: "f16[2464, 512]", view_193: "f16[2464, 2048]", view_196: "f16[2464, 512]", arg247_1: "f16[32, 8, 77, 64]", view_207: "f16[2464, 1536]", view_210: "f16[2464, 512]", view_214: "f16[2464, 2048]", view_217: "f16[2464, 512]", arg236_1: "f16[32, 8, 77, 64]", view_228: "f16[2464, 1536]", view_231: "f16[2464, 512]", view_235: "f16[2464, 2048]", view_238: "f16[2464, 512]", arg225_1: "f16[32, 8, 77, 64]", getitem_35: "f16[32, 8, 77, 64]", getitem_34: "f16[32, 8, 77, 64]", getitem_33: "f16[32, 8, 77, 64]", full_2: "f16[3, 77, 32, 512]", view_253: "f16[1600, 768]", view_257: "f16[1600, 3072]", view_260: "f16[1600, 768]", arg209_1: "f16[32, 12, 50, 64]", view_272: "f16[1600, 2304]", view_274: "f16[1600, 768]", view_278: "f16[1600, 3072]", view_281: "f16[1600, 768]", arg196_1: "f16[32, 12, 50, 64]", view_293: "f16[1600, 2304]", view_295: "f16[1600, 768]", view_299: "f16[1600, 3072]", view_302: "f16[1600, 768]", arg183_1: "f16[32, 12, 50, 64]", view_314: "f16[1600, 2304]", view_316: "f16[1600, 768]", view_320: "f16[1600, 3072]", view_323: "f16[1600, 768]", arg170_1: "f16[32, 12, 50, 64]", view_335: "f16[1600, 2304]", view_337: "f16[1600, 768]", view_341: "f16[1600, 3072]", view_344: "f16[1600, 768]", arg157_1: "f16[32, 12, 50, 64]", view_356: "f16[1600, 2304]", view_358: "f16[1600, 768]", view_362: "f16[1600, 3072]", view_365: "f16[1600, 768]", arg144_1: "f16[32, 12, 50, 64]", view_377: "f16[1600, 2304]", view_379: "f16[1600, 768]", view_383: "f16[1600, 3072]", view_386: "f16[1600, 768]", arg131_1: "f16[32, 12, 50, 64]", view_398: "f16[1600, 2304]", view_400: "f16[1600, 768]", view_404: "f16[1600, 3072]", view_407: "f16[1600, 768]", arg118_1: "f16[32, 12, 50, 64]", view_419: "f16[1600, 2304]", view_421: "f16[1600, 768]", view_425: "f16[1600, 3072]", view_428: "f16[1600, 768]", arg105_1: "f16[32, 12, 50, 64]", view_440: "f16[1600, 2304]", view_442: "f16[1600, 768]", view_446: "f16[1600, 3072]", view_449: "f16[1600, 768]", arg92_1: "f16[32, 12, 50, 64]", view_461: "f16[1600, 2304]", view_463: "f16[1600, 768]", view_467: "f16[1600, 3072]", view_470: "f16[1600, 768]", arg79_1: "f16[32, 12, 50, 64]", view_482: "f16[1600, 2304]", view_484: "f16[1600, 768]", view_488: "f16[1600, 3072]", view_491: "f16[1600, 768]", arg66_1: "f16[32, 12, 50, 64]", view_503: "f16[1600, 2304]", mm_195: "f16[1600, 768]", arg3_1: "f32[768]", arg57_1: "f32[32, 50, 768]", arg0_1: "f32[50, 768]", arg58_1: "f32[32, 50, 1]", arg59_1: "f32[32, 50, 1]", arg1_1: "f32[768]", arg2_1: "f32[768]", arg60_1: "f32[32, 50, 1]", arg61_1: "f32[32, 50, 1]", add_119: "f32[32, 50, 768]"):
        # No stacktrace found for following nodes
        permute_default: "f16[512, 32]" = torch.ops.aten.permute.default(add, [1, 0]);  add = None
        permute_default_1: "f16[512, 2464]" = torch.ops.aten.permute.default(view, [1, 0]);  view = None
        permute_default_2: "f16[2048, 2464]" = torch.ops.aten.permute.default(view_4, [1, 0]);  view_4 = None
        permute_default_3: "f16[512, 2464]" = torch.ops.aten.permute.default(view_7, [1, 0]);  view_7 = None
        permute_default_4: "f16[77, 32, 8, 64]" = torch.ops.aten.permute.default(arg346_1, [2, 0, 1, 3]);  arg346_1 = None
        reshape_default: "f16[2464, 512]" = torch.ops.aten.reshape.default(permute_default_4, [2464, 512]);  permute_default_4 = None
        permute_default_5: "f16[1536, 2464]" = torch.ops.aten.permute.default(view_18, [1, 0]);  view_18 = None
        permute_default_6: "f16[512, 2464]" = torch.ops.aten.permute.default(view_21, [1, 0]);  view_21 = None
        permute_default_7: "f16[2048, 2464]" = torch.ops.aten.permute.default(view_25, [1, 0]);  view_25 = None
        permute_default_8: "f16[512, 2464]" = torch.ops.aten.permute.default(view_28, [1, 0]);  view_28 = None
        permute_default_9: "f16[77, 32, 8, 64]" = torch.ops.aten.permute.default(arg335_1, [2, 0, 1, 3]);  arg335_1 = None
        reshape_default_1: "f16[2464, 512]" = torch.ops.aten.reshape.default(permute_default_9, [2464, 512]);  permute_default_9 = None
        permute_default_10: "f16[1536, 2464]" = torch.ops.aten.permute.default(view_39, [1, 0]);  view_39 = None
        permute_default_11: "f16[512, 2464]" = torch.ops.aten.permute.default(view_42, [1, 0]);  view_42 = None
        permute_default_12: "f16[2048, 2464]" = torch.ops.aten.permute.default(view_46, [1, 0]);  view_46 = None
        permute_default_13: "f16[512, 2464]" = torch.ops.aten.permute.default(view_49, [1, 0]);  view_49 = None
        permute_default_14: "f16[77, 32, 8, 64]" = torch.ops.aten.permute.default(arg324_1, [2, 0, 1, 3]);  arg324_1 = None
        reshape_default_2: "f16[2464, 512]" = torch.ops.aten.reshape.default(permute_default_14, [2464, 512]);  permute_default_14 = None
        permute_default_15: "f16[1536, 2464]" = torch.ops.aten.permute.default(view_60, [1, 0]);  view_60 = None
        permute_default_16: "f16[512, 2464]" = torch.ops.aten.permute.default(view_63, [1, 0]);  view_63 = None
        permute_default_17: "f16[2048, 2464]" = torch.ops.aten.permute.default(view_67, [1, 0]);  view_67 = None
        permute_default_18: "f16[512, 2464]" = torch.ops.aten.permute.default(view_70, [1, 0]);  view_70 = None
        permute_default_19: "f16[77, 32, 8, 64]" = torch.ops.aten.permute.default(arg313_1, [2, 0, 1, 3]);  arg313_1 = None
        reshape_default_3: "f16[2464, 512]" = torch.ops.aten.reshape.default(permute_default_19, [2464, 512]);  permute_default_19 = None
        permute_default_20: "f16[1536, 2464]" = torch.ops.aten.permute.default(view_81, [1, 0]);  view_81 = None
        permute_default_21: "f16[512, 2464]" = torch.ops.aten.permute.default(view_84, [1, 0]);  view_84 = None
        permute_default_22: "f16[2048, 2464]" = torch.ops.aten.permute.default(view_88, [1, 0]);  view_88 = None
        permute_default_23: "f16[512, 2464]" = torch.ops.aten.permute.default(view_91, [1, 0]);  view_91 = None
        permute_default_24: "f16[77, 32, 8, 64]" = torch.ops.aten.permute.default(arg302_1, [2, 0, 1, 3]);  arg302_1 = None
        reshape_default_4: "f16[2464, 512]" = torch.ops.aten.reshape.default(permute_default_24, [2464, 512]);  permute_default_24 = None
        permute_default_25: "f16[1536, 2464]" = torch.ops.aten.permute.default(view_102, [1, 0]);  view_102 = None
        permute_default_26: "f16[512, 2464]" = torch.ops.aten.permute.default(view_105, [1, 0]);  view_105 = None
        permute_default_27: "f16[2048, 2464]" = torch.ops.aten.permute.default(view_109, [1, 0]);  view_109 = None
        permute_default_28: "f16[512, 2464]" = torch.ops.aten.permute.default(view_112, [1, 0]);  view_112 = None
        permute_default_29: "f16[77, 32, 8, 64]" = torch.ops.aten.permute.default(arg291_1, [2, 0, 1, 3]);  arg291_1 = None
        reshape_default_5: "f16[2464, 512]" = torch.ops.aten.reshape.default(permute_default_29, [2464, 512]);  permute_default_29 = None
        permute_default_30: "f16[1536, 2464]" = torch.ops.aten.permute.default(view_123, [1, 0]);  view_123 = None
        permute_default_31: "f16[512, 2464]" = torch.ops.aten.permute.default(view_126, [1, 0]);  view_126 = None
        permute_default_32: "f16[2048, 2464]" = torch.ops.aten.permute.default(view_130, [1, 0]);  view_130 = None
        permute_default_33: "f16[512, 2464]" = torch.ops.aten.permute.default(view_133, [1, 0]);  view_133 = None
        permute_default_34: "f16[77, 32, 8, 64]" = torch.ops.aten.permute.default(arg280_1, [2, 0, 1, 3]);  arg280_1 = None
        reshape_default_6: "f16[2464, 512]" = torch.ops.aten.reshape.default(permute_default_34, [2464, 512]);  permute_default_34 = None
        permute_default_35: "f16[1536, 2464]" = torch.ops.aten.permute.default(view_144, [1, 0]);  view_144 = None
        permute_default_36: "f16[512, 2464]" = torch.ops.aten.permute.default(view_147, [1, 0]);  view_147 = None
        permute_default_37: "f16[2048, 2464]" = torch.ops.aten.permute.default(view_151, [1, 0]);  view_151 = None
        permute_default_38: "f16[512, 2464]" = torch.ops.aten.permute.default(view_154, [1, 0]);  view_154 = None
        permute_default_39: "f16[77, 32, 8, 64]" = torch.ops.aten.permute.default(arg269_1, [2, 0, 1, 3]);  arg269_1 = None
        reshape_default_7: "f16[2464, 512]" = torch.ops.aten.reshape.default(permute_default_39, [2464, 512]);  permute_default_39 = None
        permute_default_40: "f16[1536, 2464]" = torch.ops.aten.permute.default(view_165, [1, 0]);  view_165 = None
        permute_default_41: "f16[512, 2464]" = torch.ops.aten.permute.default(view_168, [1, 0]);  view_168 = None
        permute_default_42: "f16[2048, 2464]" = torch.ops.aten.permute.default(view_172, [1, 0]);  view_172 = None
        permute_default_43: "f16[512, 2464]" = torch.ops.aten.permute.default(view_175, [1, 0]);  view_175 = None
        permute_default_44: "f16[77, 32, 8, 64]" = torch.ops.aten.permute.default(arg258_1, [2, 0, 1, 3]);  arg258_1 = None
        reshape_default_8: "f16[2464, 512]" = torch.ops.aten.reshape.default(permute_default_44, [2464, 512]);  permute_default_44 = None
        permute_default_45: "f16[1536, 2464]" = torch.ops.aten.permute.default(view_186, [1, 0]);  view_186 = None
        permute_default_46: "f16[512, 2464]" = torch.ops.aten.permute.default(view_189, [1, 0]);  view_189 = None
        permute_default_47: "f16[2048, 2464]" = torch.ops.aten.permute.default(view_193, [1, 0]);  view_193 = None
        permute_default_48: "f16[512, 2464]" = torch.ops.aten.permute.default(view_196, [1, 0]);  view_196 = None
        permute_default_49: "f16[77, 32, 8, 64]" = torch.ops.aten.permute.default(arg247_1, [2, 0, 1, 3]);  arg247_1 = None
        reshape_default_9: "f16[2464, 512]" = torch.ops.aten.reshape.default(permute_default_49, [2464, 512]);  permute_default_49 = None
        permute_default_50: "f16[1536, 2464]" = torch.ops.aten.permute.default(view_207, [1, 0]);  view_207 = None
        permute_default_51: "f16[512, 2464]" = torch.ops.aten.permute.default(view_210, [1, 0]);  view_210 = None
        permute_default_52: "f16[2048, 2464]" = torch.ops.aten.permute.default(view_214, [1, 0]);  view_214 = None
        permute_default_53: "f16[512, 2464]" = torch.ops.aten.permute.default(view_217, [1, 0]);  view_217 = None
        permute_default_54: "f16[77, 32, 8, 64]" = torch.ops.aten.permute.default(arg236_1, [2, 0, 1, 3]);  arg236_1 = None
        reshape_default_10: "f16[2464, 512]" = torch.ops.aten.reshape.default(permute_default_54, [2464, 512]);  permute_default_54 = None
        permute_default_55: "f16[1536, 2464]" = torch.ops.aten.permute.default(view_228, [1, 0]);  view_228 = None
        permute_default_56: "f16[512, 2464]" = torch.ops.aten.permute.default(view_231, [1, 0]);  view_231 = None
        permute_default_57: "f16[2048, 2464]" = torch.ops.aten.permute.default(view_235, [1, 0]);  view_235 = None
        permute_default_58: "f16[512, 2464]" = torch.ops.aten.permute.default(view_238, [1, 0]);  view_238 = None
        permute_default_59: "f16[77, 32, 8, 64]" = torch.ops.aten.permute.default(arg225_1, [2, 0, 1, 3]);  arg225_1 = None
        reshape_default_11: "f16[2464, 512]" = torch.ops.aten.reshape.default(permute_default_59, [2464, 512]);  permute_default_59 = None
        reshape_default_12: "f16[256, 77, 64]" = torch.ops.aten.reshape.default(getitem_35, [256, 77, 64]);  getitem_35 = None
        reshape_default_13: "f16[256, 77, 64]" = torch.ops.aten.reshape.default(getitem_34, [256, 77, 64]);  getitem_34 = None
        reshape_default_14: "f16[256, 77, 64]" = torch.ops.aten.reshape.default(getitem_33, [256, 77, 64]);  getitem_33 = None
        permute_default_60: "f16[77, 256, 64]" = torch.ops.aten.permute.default(reshape_default_12, [1, 0, 2]);  reshape_default_12 = None
        reshape_default_15: "f16[77, 32, 512]" = torch.ops.aten.reshape.default(permute_default_60, [77, 32, 512]);  permute_default_60 = None
        permute_default_61: "f16[77, 256, 64]" = torch.ops.aten.permute.default(reshape_default_13, [1, 0, 2]);  reshape_default_13 = None
        reshape_default_16: "f16[77, 32, 512]" = torch.ops.aten.reshape.default(permute_default_61, [77, 32, 512]);  permute_default_61 = None
        permute_default_62: "f16[77, 256, 64]" = torch.ops.aten.permute.default(reshape_default_14, [1, 0, 2]);  reshape_default_14 = None
        reshape_default_17: "f16[77, 32, 512]" = torch.ops.aten.reshape.default(permute_default_62, [77, 32, 512]);  permute_default_62 = None
        select_scatter_default: "f16[3, 77, 32, 512]" = torch.ops.aten.select_scatter.default(full_2, reshape_default_15, 0, 2);  reshape_default_15 = None
        select_scatter_default_1: "f16[3, 77, 32, 512]" = torch.ops.aten.select_scatter.default(full_2, reshape_default_16, 0, 1);  reshape_default_16 = None
        add_tensor: "f16[3, 77, 32, 512]" = torch.ops.aten.add.Tensor(select_scatter_default, select_scatter_default_1);  select_scatter_default = select_scatter_default_1 = None
        select_scatter_default_2: "f16[3, 77, 32, 512]" = torch.ops.aten.select_scatter.default(full_2, reshape_default_17, 0, 0);  full_2 = reshape_default_17 = None
        add_tensor_1: "f16[3, 77, 32, 512]" = torch.ops.aten.add.Tensor(add_tensor, select_scatter_default_2);  add_tensor = select_scatter_default_2 = None
        unsqueeze_default: "f16[3, 77, 32, 1, 512]" = torch.ops.aten.unsqueeze.default(add_tensor_1, 3);  add_tensor_1 = None
        permute_default_63: "f16[1, 77, 32, 3, 512]" = torch.ops.aten.permute.default(unsqueeze_default, [3, 1, 2, 0, 4]);  unsqueeze_default = None
        squeeze_dim: "f16[77, 32, 3, 512]" = torch.ops.aten.squeeze.dim(permute_default_63, 0);  permute_default_63 = None
        clone_default: "f16[77, 32, 3, 512]" = torch.ops.aten.clone.default(squeeze_dim, memory_format = torch.contiguous_format);  squeeze_dim = None
        reshape_default_18: "f16[77, 32, 1536]" = torch.ops.aten.reshape.default(clone_default, [77, 32, 1536]);  clone_default = None
        reshape_default_19: "f16[2464, 1536]" = torch.ops.aten.reshape.default(reshape_default_18, [2464, 1536]);  reshape_default_18 = None
        permute_default_64: "f16[1536, 2464]" = torch.ops.aten.permute.default(reshape_default_19, [1, 0]);  reshape_default_19 = None
        permute_default_65: "f16[768, 1600]" = torch.ops.aten.permute.default(view_253, [1, 0]);  view_253 = None
        permute_default_66: "f16[3072, 1600]" = torch.ops.aten.permute.default(view_257, [1, 0]);  view_257 = None
        permute_default_67: "f16[768, 1600]" = torch.ops.aten.permute.default(view_260, [1, 0]);  view_260 = None
        permute_default_68: "f16[50, 32, 12, 64]" = torch.ops.aten.permute.default(arg209_1, [2, 0, 1, 3]);  arg209_1 = None
        reshape_default_20: "f16[1600, 768]" = torch.ops.aten.reshape.default(permute_default_68, [1600, 768]);  permute_default_68 = None
        permute_default_69: "f16[2304, 1600]" = torch.ops.aten.permute.default(view_272, [1, 0]);  view_272 = None
        permute_default_70: "f16[768, 1600]" = torch.ops.aten.permute.default(view_274, [1, 0]);  view_274 = None
        permute_default_71: "f16[3072, 1600]" = torch.ops.aten.permute.default(view_278, [1, 0]);  view_278 = None
        permute_default_72: "f16[768, 1600]" = torch.ops.aten.permute.default(view_281, [1, 0]);  view_281 = None
        permute_default_73: "f16[50, 32, 12, 64]" = torch.ops.aten.permute.default(arg196_1, [2, 0, 1, 3]);  arg196_1 = None
        reshape_default_21: "f16[1600, 768]" = torch.ops.aten.reshape.default(permute_default_73, [1600, 768]);  permute_default_73 = None
        permute_default_74: "f16[2304, 1600]" = torch.ops.aten.permute.default(view_293, [1, 0]);  view_293 = None
        permute_default_75: "f16[768, 1600]" = torch.ops.aten.permute.default(view_295, [1, 0]);  view_295 = None
        permute_default_76: "f16[3072, 1600]" = torch.ops.aten.permute.default(view_299, [1, 0]);  view_299 = None
        permute_default_77: "f16[768, 1600]" = torch.ops.aten.permute.default(view_302, [1, 0]);  view_302 = None
        permute_default_78: "f16[50, 32, 12, 64]" = torch.ops.aten.permute.default(arg183_1, [2, 0, 1, 3]);  arg183_1 = None
        reshape_default_22: "f16[1600, 768]" = torch.ops.aten.reshape.default(permute_default_78, [1600, 768]);  permute_default_78 = None
        permute_default_79: "f16[2304, 1600]" = torch.ops.aten.permute.default(view_314, [1, 0]);  view_314 = None
        permute_default_80: "f16[768, 1600]" = torch.ops.aten.permute.default(view_316, [1, 0]);  view_316 = None
        permute_default_81: "f16[3072, 1600]" = torch.ops.aten.permute.default(view_320, [1, 0]);  view_320 = None
        permute_default_82: "f16[768, 1600]" = torch.ops.aten.permute.default(view_323, [1, 0]);  view_323 = None
        permute_default_83: "f16[50, 32, 12, 64]" = torch.ops.aten.permute.default(arg170_1, [2, 0, 1, 3]);  arg170_1 = None
        reshape_default_23: "f16[1600, 768]" = torch.ops.aten.reshape.default(permute_default_83, [1600, 768]);  permute_default_83 = None
        permute_default_84: "f16[2304, 1600]" = torch.ops.aten.permute.default(view_335, [1, 0]);  view_335 = None
        permute_default_85: "f16[768, 1600]" = torch.ops.aten.permute.default(view_337, [1, 0]);  view_337 = None
        permute_default_86: "f16[3072, 1600]" = torch.ops.aten.permute.default(view_341, [1, 0]);  view_341 = None
        permute_default_87: "f16[768, 1600]" = torch.ops.aten.permute.default(view_344, [1, 0]);  view_344 = None
        permute_default_88: "f16[50, 32, 12, 64]" = torch.ops.aten.permute.default(arg157_1, [2, 0, 1, 3]);  arg157_1 = None
        reshape_default_24: "f16[1600, 768]" = torch.ops.aten.reshape.default(permute_default_88, [1600, 768]);  permute_default_88 = None
        permute_default_89: "f16[2304, 1600]" = torch.ops.aten.permute.default(view_356, [1, 0]);  view_356 = None
        permute_default_90: "f16[768, 1600]" = torch.ops.aten.permute.default(view_358, [1, 0]);  view_358 = None
        permute_default_91: "f16[3072, 1600]" = torch.ops.aten.permute.default(view_362, [1, 0]);  view_362 = None
        permute_default_92: "f16[768, 1600]" = torch.ops.aten.permute.default(view_365, [1, 0]);  view_365 = None
        permute_default_93: "f16[50, 32, 12, 64]" = torch.ops.aten.permute.default(arg144_1, [2, 0, 1, 3]);  arg144_1 = None
        reshape_default_25: "f16[1600, 768]" = torch.ops.aten.reshape.default(permute_default_93, [1600, 768]);  permute_default_93 = None
        permute_default_94: "f16[2304, 1600]" = torch.ops.aten.permute.default(view_377, [1, 0]);  view_377 = None
        permute_default_95: "f16[768, 1600]" = torch.ops.aten.permute.default(view_379, [1, 0]);  view_379 = None
        permute_default_96: "f16[3072, 1600]" = torch.ops.aten.permute.default(view_383, [1, 0]);  view_383 = None
        permute_default_97: "f16[768, 1600]" = torch.ops.aten.permute.default(view_386, [1, 0]);  view_386 = None
        permute_default_98: "f16[50, 32, 12, 64]" = torch.ops.aten.permute.default(arg131_1, [2, 0, 1, 3]);  arg131_1 = None
        reshape_default_26: "f16[1600, 768]" = torch.ops.aten.reshape.default(permute_default_98, [1600, 768]);  permute_default_98 = None
        permute_default_99: "f16[2304, 1600]" = torch.ops.aten.permute.default(view_398, [1, 0]);  view_398 = None
        permute_default_100: "f16[768, 1600]" = torch.ops.aten.permute.default(view_400, [1, 0]);  view_400 = None
        permute_default_101: "f16[3072, 1600]" = torch.ops.aten.permute.default(view_404, [1, 0]);  view_404 = None
        permute_default_102: "f16[768, 1600]" = torch.ops.aten.permute.default(view_407, [1, 0]);  view_407 = None
        permute_default_103: "f16[50, 32, 12, 64]" = torch.ops.aten.permute.default(arg118_1, [2, 0, 1, 3]);  arg118_1 = None
        reshape_default_27: "f16[1600, 768]" = torch.ops.aten.reshape.default(permute_default_103, [1600, 768]);  permute_default_103 = None
        permute_default_104: "f16[2304, 1600]" = torch.ops.aten.permute.default(view_419, [1, 0]);  view_419 = None
        permute_default_105: "f16[768, 1600]" = torch.ops.aten.permute.default(view_421, [1, 0]);  view_421 = None
        permute_default_106: "f16[3072, 1600]" = torch.ops.aten.permute.default(view_425, [1, 0]);  view_425 = None
        permute_default_107: "f16[768, 1600]" = torch.ops.aten.permute.default(view_428, [1, 0]);  view_428 = None
        permute_default_108: "f16[50, 32, 12, 64]" = torch.ops.aten.permute.default(arg105_1, [2, 0, 1, 3]);  arg105_1 = None
        reshape_default_28: "f16[1600, 768]" = torch.ops.aten.reshape.default(permute_default_108, [1600, 768]);  permute_default_108 = None
        permute_default_109: "f16[2304, 1600]" = torch.ops.aten.permute.default(view_440, [1, 0]);  view_440 = None
        permute_default_110: "f16[768, 1600]" = torch.ops.aten.permute.default(view_442, [1, 0]);  view_442 = None
        permute_default_111: "f16[3072, 1600]" = torch.ops.aten.permute.default(view_446, [1, 0]);  view_446 = None
        permute_default_112: "f16[768, 1600]" = torch.ops.aten.permute.default(view_449, [1, 0]);  view_449 = None
        permute_default_113: "f16[50, 32, 12, 64]" = torch.ops.aten.permute.default(arg92_1, [2, 0, 1, 3]);  arg92_1 = None
        reshape_default_29: "f16[1600, 768]" = torch.ops.aten.reshape.default(permute_default_113, [1600, 768]);  permute_default_113 = None
        permute_default_114: "f16[2304, 1600]" = torch.ops.aten.permute.default(view_461, [1, 0]);  view_461 = None
        permute_default_115: "f16[768, 1600]" = torch.ops.aten.permute.default(view_463, [1, 0]);  view_463 = None
        permute_default_116: "f16[3072, 1600]" = torch.ops.aten.permute.default(view_467, [1, 0]);  view_467 = None
        permute_default_117: "f16[768, 1600]" = torch.ops.aten.permute.default(view_470, [1, 0]);  view_470 = None
        permute_default_118: "f16[50, 32, 12, 64]" = torch.ops.aten.permute.default(arg79_1, [2, 0, 1, 3]);  arg79_1 = None
        reshape_default_30: "f16[1600, 768]" = torch.ops.aten.reshape.default(permute_default_118, [1600, 768]);  permute_default_118 = None
        permute_default_119: "f16[2304, 1600]" = torch.ops.aten.permute.default(view_482, [1, 0]);  view_482 = None
        permute_default_120: "f16[768, 1600]" = torch.ops.aten.permute.default(view_484, [1, 0]);  view_484 = None
        permute_default_121: "f16[3072, 1600]" = torch.ops.aten.permute.default(view_488, [1, 0]);  view_488 = None
        permute_default_122: "f16[768, 1600]" = torch.ops.aten.permute.default(view_491, [1, 0]);  view_491 = None
        permute_default_123: "f16[50, 32, 12, 64]" = torch.ops.aten.permute.default(arg66_1, [2, 0, 1, 3]);  arg66_1 = None
        reshape_default_31: "f16[1600, 768]" = torch.ops.aten.reshape.default(permute_default_123, [1600, 768]);  permute_default_123 = None
        permute_default_124: "f16[2304, 1600]" = torch.ops.aten.permute.default(view_503, [1, 0]);  view_503 = None
        reshape_default_32: "f16[50, 32, 768]" = torch.ops.aten.reshape.default(mm_195, [50, 32, 768]);  mm_195 = None
        convert_element_type_default: "f32[50, 32, 768]" = torch.ops.prims.convert_element_type.default(reshape_default_32, torch.float32);  reshape_default_32 = None
        permute_default_125: "f32[32, 50, 768]" = torch.ops.aten.permute.default(convert_element_type_default, [1, 0, 2]);  convert_element_type_default = None
        mul_tensor: "f32[32, 50, 768]" = torch.ops.aten.mul.Tensor(permute_default_125, arg3_1);  permute_default_125 = arg3_1 = None
        mul_tensor_1: "f32[32, 50, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, 768)
        sum_dim_int_list: "f32[32, 50, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [2], True)
        add_tensor_2: "f32[32, 50, 768]" = torch.ops.aten.add.Tensor(arg57_1, arg0_1);  arg57_1 = arg0_1 = None
        sub_tensor: "f32[32, 50, 768]" = torch.ops.aten.sub.Tensor(add_tensor_2, arg58_1);  add_tensor_2 = arg58_1 = None
        mul_tensor_2: "f32[32, 50, 768]" = torch.ops.aten.mul.Tensor(sub_tensor, arg59_1);  sub_tensor = None
        mul_tensor_3: "f32[32, 50, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_2, arg1_1)
        add_tensor_3: "f32[32, 50, 768]" = torch.ops.aten.add.Tensor(mul_tensor_3, arg2_1);  mul_tensor_3 = arg2_1 = None
        sub_tensor_1: "f32[32, 50, 768]" = torch.ops.aten.sub.Tensor(add_tensor_3, arg60_1);  add_tensor_3 = arg60_1 = None
        mul_tensor_4: "f32[32, 50, 768]" = torch.ops.aten.mul.Tensor(sub_tensor_1, arg61_1);  sub_tensor_1 = None
        mul_tensor_5: "f32[32, 50, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, mul_tensor_4);  mul_tensor = None
        sum_dim_int_list_1: "f32[32, 50, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_5, [2], True);  mul_tensor_5 = None
        mul_tensor_6: "f32[32, 50, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_4, sum_dim_int_list_1);  mul_tensor_4 = sum_dim_int_list_1 = None
        sub_tensor_2: "f32[32, 50, 768]" = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list);  mul_tensor_1 = sum_dim_int_list = None
        sub_tensor_3: "f32[32, 50, 768]" = torch.ops.aten.sub.Tensor(sub_tensor_2, mul_tensor_6);  sub_tensor_2 = mul_tensor_6 = None
        div_tensor: "f32[32, 50, 1]" = torch.ops.aten.div.Tensor(arg61_1, 768);  arg61_1 = None
        mul_tensor_7: "f32[32, 50, 768]" = torch.ops.aten.mul.Tensor(div_tensor, sub_tensor_3);  div_tensor = sub_tensor_3 = None
        add_tensor_4: "f32[32, 50, 768]" = torch.ops.aten.add.Tensor(add_119, mul_tensor_7);  add_119 = mul_tensor_7 = None
        mul_tensor_8: "f32[32, 50, 768]" = torch.ops.aten.mul.Tensor(add_tensor_4, arg1_1);  add_tensor_4 = arg1_1 = None
        mul_tensor_9: "f32[32, 50, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_8, 768)
        sum_dim_int_list_2: "f32[32, 50, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_8, [2], True)
        mul_tensor_10: "f32[32, 50, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_8, mul_tensor_2);  mul_tensor_8 = None
        sum_dim_int_list_3: "f32[32, 50, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_10, [2], True);  mul_tensor_10 = None
        mul_tensor_11: "f32[32, 50, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_2, sum_dim_int_list_3);  mul_tensor_2 = sum_dim_int_list_3 = None
        sub_tensor_4: "f32[32, 50, 768]" = torch.ops.aten.sub.Tensor(mul_tensor_9, sum_dim_int_list_2);  mul_tensor_9 = sum_dim_int_list_2 = None
        sub_tensor_5: "f32[32, 50, 768]" = torch.ops.aten.sub.Tensor(sub_tensor_4, mul_tensor_11);  sub_tensor_4 = mul_tensor_11 = None
        div_tensor_1: "f32[32, 50, 1]" = torch.ops.aten.div.Tensor(arg59_1, 768);  arg59_1 = None
        mul_tensor_12: "f32[32, 50, 768]" = torch.ops.aten.mul.Tensor(div_tensor_1, sub_tensor_5);  div_tensor_1 = sub_tensor_5 = None
        slice_tensor: "f32[32, 49, 768]" = torch.ops.aten.slice.Tensor(mul_tensor_12, 1, 1, 50);  mul_tensor_12 = None
        convert_element_type_default_1: "f16[32, 49, 768]" = torch.ops.prims.convert_element_type.default(slice_tensor, torch.float16);  slice_tensor = None
        permute_default_126: "f16[32, 768, 49]" = torch.ops.aten.permute.default(convert_element_type_default_1, [0, 2, 1]);  convert_element_type_default_1 = None
        reshape_default_33: "f16[32, 768, 7, 7]" = torch.ops.aten.reshape.default(permute_default_126, [32, 768, 7, 7]);  permute_default_126 = None
        return (permute_default, permute_default_1, permute_default_2, permute_default_3, reshape_default, permute_default_5, permute_default_6, permute_default_7, permute_default_8, reshape_default_1, permute_default_10, permute_default_11, permute_default_12, permute_default_13, reshape_default_2, permute_default_15, permute_default_16, permute_default_17, permute_default_18, reshape_default_3, permute_default_20, permute_default_21, permute_default_22, permute_default_23, reshape_default_4, permute_default_25, permute_default_26, permute_default_27, permute_default_28, reshape_default_5, permute_default_30, permute_default_31, permute_default_32, permute_default_33, reshape_default_6, permute_default_35, permute_default_36, permute_default_37, permute_default_38, reshape_default_7, permute_default_40, permute_default_41, permute_default_42, permute_default_43, reshape_default_8, permute_default_45, permute_default_46, permute_default_47, permute_default_48, reshape_default_9, permute_default_50, permute_default_51, permute_default_52, permute_default_53, reshape_default_10, permute_default_55, permute_default_56, permute_default_57, permute_default_58, reshape_default_11, permute_default_64, permute_default_65, permute_default_66, permute_default_67, reshape_default_20, permute_default_69, permute_default_70, permute_default_71, permute_default_72, reshape_default_21, permute_default_74, permute_default_75, permute_default_76, permute_default_77, reshape_default_22, permute_default_79, permute_default_80, permute_default_81, permute_default_82, reshape_default_23, permute_default_84, permute_default_85, permute_default_86, permute_default_87, reshape_default_24, permute_default_89, permute_default_90, permute_default_91, permute_default_92, reshape_default_25, permute_default_94, permute_default_95, permute_default_96, permute_default_97, reshape_default_26, permute_default_99, permute_default_100, permute_default_101, permute_default_102, reshape_default_27, permute_default_104, permute_default_105, permute_default_106, permute_default_107, reshape_default_28, permute_default_109, permute_default_110, permute_default_111, permute_default_112, reshape_default_29, permute_default_114, permute_default_115, permute_default_116, permute_default_117, reshape_default_30, permute_default_119, permute_default_120, permute_default_121, permute_default_122, reshape_default_31, permute_default_124, reshape_default_33)


def _default_make_inputs():
    return [
    torch.randn([32, 512], dtype=torch.float16, device='cuda'),
    torch.randn([2464, 512], dtype=torch.float16, device='cuda'),
    torch.randn([2464, 2048], dtype=torch.float16, device='cuda'),
    torch.randn([2464, 512], dtype=torch.float16, device='cuda'),
    torch.randn(1261568, dtype=torch.float16, device='cuda').as_strided([32, 8, 77, 64], [512, 64, 16384, 1]),  # arg346_1
    torch.randn([2464, 1536], dtype=torch.float16, device='cuda'),
    torch.randn([2464, 512], dtype=torch.float16, device='cuda'),
    torch.randn([2464, 2048], dtype=torch.float16, device='cuda'),
    torch.randn([2464, 512], dtype=torch.float16, device='cuda'),
    torch.randn(1261568, dtype=torch.float16, device='cuda').as_strided([32, 8, 77, 64], [512, 64, 16384, 1]),  # arg335_1
    torch.randn([2464, 1536], dtype=torch.float16, device='cuda'),
    torch.randn([2464, 512], dtype=torch.float16, device='cuda'),
    torch.randn([2464, 2048], dtype=torch.float16, device='cuda'),
    torch.randn([2464, 512], dtype=torch.float16, device='cuda'),
    torch.randn(1261568, dtype=torch.float16, device='cuda').as_strided([32, 8, 77, 64], [512, 64, 16384, 1]),  # arg324_1
    torch.randn([2464, 1536], dtype=torch.float16, device='cuda'),
    torch.randn([2464, 512], dtype=torch.float16, device='cuda'),
    torch.randn([2464, 2048], dtype=torch.float16, device='cuda'),
    torch.randn([2464, 512], dtype=torch.float16, device='cuda'),
    torch.randn(1261568, dtype=torch.float16, device='cuda').as_strided([32, 8, 77, 64], [512, 64, 16384, 1]),  # arg313_1
    torch.randn([2464, 1536], dtype=torch.float16, device='cuda'),
    torch.randn([2464, 512], dtype=torch.float16, device='cuda'),
    torch.randn([2464, 2048], dtype=torch.float16, device='cuda'),
    torch.randn([2464, 512], dtype=torch.float16, device='cuda'),
    torch.randn(1261568, dtype=torch.float16, device='cuda').as_strided([32, 8, 77, 64], [512, 64, 16384, 1]),  # arg302_1
    torch.randn([2464, 1536], dtype=torch.float16, device='cuda'),
    torch.randn([2464, 512], dtype=torch.float16, device='cuda'),
    torch.randn([2464, 2048], dtype=torch.float16, device='cuda'),
    torch.randn([2464, 512], dtype=torch.float16, device='cuda'),
    torch.randn(1261568, dtype=torch.float16, device='cuda').as_strided([32, 8, 77, 64], [512, 64, 16384, 1]),  # arg291_1
    torch.randn([2464, 1536], dtype=torch.float16, device='cuda'),
    torch.randn([2464, 512], dtype=torch.float16, device='cuda'),
    torch.randn([2464, 2048], dtype=torch.float16, device='cuda'),
    torch.randn([2464, 512], dtype=torch.float16, device='cuda'),
    torch.randn(1261568, dtype=torch.float16, device='cuda').as_strided([32, 8, 77, 64], [512, 64, 16384, 1]),  # arg280_1
    torch.randn([2464, 1536], dtype=torch.float16, device='cuda'),
    torch.randn([2464, 512], dtype=torch.float16, device='cuda'),
    torch.randn([2464, 2048], dtype=torch.float16, device='cuda'),
    torch.randn([2464, 512], dtype=torch.float16, device='cuda'),
    torch.randn(1261568, dtype=torch.float16, device='cuda').as_strided([32, 8, 77, 64], [512, 64, 16384, 1]),  # arg269_1
    torch.randn([2464, 1536], dtype=torch.float16, device='cuda'),
    torch.randn([2464, 512], dtype=torch.float16, device='cuda'),
    torch.randn([2464, 2048], dtype=torch.float16, device='cuda'),
    torch.randn([2464, 512], dtype=torch.float16, device='cuda'),
    torch.randn(1261568, dtype=torch.float16, device='cuda').as_strided([32, 8, 77, 64], [512, 64, 16384, 1]),  # arg258_1
    torch.randn([2464, 1536], dtype=torch.float16, device='cuda'),
    torch.randn([2464, 512], dtype=torch.float16, device='cuda'),
    torch.randn([2464, 2048], dtype=torch.float16, device='cuda'),
    torch.randn([2464, 512], dtype=torch.float16, device='cuda'),
    torch.randn(1261568, dtype=torch.float16, device='cuda').as_strided([32, 8, 77, 64], [512, 64, 16384, 1]),  # arg247_1
    torch.randn([2464, 1536], dtype=torch.float16, device='cuda'),
    torch.randn([2464, 512], dtype=torch.float16, device='cuda'),
    torch.randn([2464, 2048], dtype=torch.float16, device='cuda'),
    torch.randn([2464, 512], dtype=torch.float16, device='cuda'),
    torch.randn(1261568, dtype=torch.float16, device='cuda').as_strided([32, 8, 77, 64], [512, 64, 16384, 1]),  # arg236_1
    torch.randn([2464, 1536], dtype=torch.float16, device='cuda'),
    torch.randn([2464, 512], dtype=torch.float16, device='cuda'),
    torch.randn([2464, 2048], dtype=torch.float16, device='cuda'),
    torch.randn([2464, 512], dtype=torch.float16, device='cuda'),
    torch.randn(1261568, dtype=torch.float16, device='cuda').as_strided([32, 8, 77, 64], [512, 64, 16384, 1]),  # arg225_1
    torch.randn(1261568, dtype=torch.float16, device='cuda').as_strided([32, 8, 77, 64], [512, 64, 16384, 1]),  # getitem_35
    torch.randn(1261568, dtype=torch.float16, device='cuda').as_strided([32, 8, 77, 64], [512, 64, 16384, 1]),  # getitem_34
    torch.randn(1261568, dtype=torch.float16, device='cuda').as_strided([32, 8, 77, 64], [512, 64, 16384, 1]),  # getitem_33
    torch.randn([3, 77, 32, 512], dtype=torch.float16, device='cuda'),
    torch.randn([1600, 768], dtype=torch.float16, device='cuda'),
    torch.randn([1600, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([1600, 768], dtype=torch.float16, device='cuda'),
    torch.randn(1228800, dtype=torch.float16, device='cuda').as_strided([32, 12, 50, 64], [768, 64, 24576, 1]),  # arg209_1
    torch.randn([1600, 2304], dtype=torch.float16, device='cuda'),
    torch.randn([1600, 768], dtype=torch.float16, device='cuda'),
    torch.randn([1600, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([1600, 768], dtype=torch.float16, device='cuda'),
    torch.randn(1228800, dtype=torch.float16, device='cuda').as_strided([32, 12, 50, 64], [768, 64, 24576, 1]),  # arg196_1
    torch.randn([1600, 2304], dtype=torch.float16, device='cuda'),
    torch.randn([1600, 768], dtype=torch.float16, device='cuda'),
    torch.randn([1600, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([1600, 768], dtype=torch.float16, device='cuda'),
    torch.randn(1228800, dtype=torch.float16, device='cuda').as_strided([32, 12, 50, 64], [768, 64, 24576, 1]),  # arg183_1
    torch.randn([1600, 2304], dtype=torch.float16, device='cuda'),
    torch.randn([1600, 768], dtype=torch.float16, device='cuda'),
    torch.randn([1600, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([1600, 768], dtype=torch.float16, device='cuda'),
    torch.randn(1228800, dtype=torch.float16, device='cuda').as_strided([32, 12, 50, 64], [768, 64, 24576, 1]),  # arg170_1
    torch.randn([1600, 2304], dtype=torch.float16, device='cuda'),
    torch.randn([1600, 768], dtype=torch.float16, device='cuda'),
    torch.randn([1600, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([1600, 768], dtype=torch.float16, device='cuda'),
    torch.randn(1228800, dtype=torch.float16, device='cuda').as_strided([32, 12, 50, 64], [768, 64, 24576, 1]),  # arg157_1
    torch.randn([1600, 2304], dtype=torch.float16, device='cuda'),
    torch.randn([1600, 768], dtype=torch.float16, device='cuda'),
    torch.randn([1600, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([1600, 768], dtype=torch.float16, device='cuda'),
    torch.randn(1228800, dtype=torch.float16, device='cuda').as_strided([32, 12, 50, 64], [768, 64, 24576, 1]),  # arg144_1
    torch.randn([1600, 2304], dtype=torch.float16, device='cuda'),
    torch.randn([1600, 768], dtype=torch.float16, device='cuda'),
    torch.randn([1600, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([1600, 768], dtype=torch.float16, device='cuda'),
    torch.randn(1228800, dtype=torch.float16, device='cuda').as_strided([32, 12, 50, 64], [768, 64, 24576, 1]),  # arg131_1
    torch.randn([1600, 2304], dtype=torch.float16, device='cuda'),
    torch.randn([1600, 768], dtype=torch.float16, device='cuda'),
    torch.randn([1600, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([1600, 768], dtype=torch.float16, device='cuda'),
    torch.randn(1228800, dtype=torch.float16, device='cuda').as_strided([32, 12, 50, 64], [768, 64, 24576, 1]),  # arg118_1
    torch.randn([1600, 2304], dtype=torch.float16, device='cuda'),
    torch.randn([1600, 768], dtype=torch.float16, device='cuda'),
    torch.randn([1600, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([1600, 768], dtype=torch.float16, device='cuda'),
    torch.randn(1228800, dtype=torch.float16, device='cuda').as_strided([32, 12, 50, 64], [768, 64, 24576, 1]),  # arg105_1
    torch.randn([1600, 2304], dtype=torch.float16, device='cuda'),
    torch.randn([1600, 768], dtype=torch.float16, device='cuda'),
    torch.randn([1600, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([1600, 768], dtype=torch.float16, device='cuda'),
    torch.randn(1228800, dtype=torch.float16, device='cuda').as_strided([32, 12, 50, 64], [768, 64, 24576, 1]),  # arg92_1
    torch.randn([1600, 2304], dtype=torch.float16, device='cuda'),
    torch.randn([1600, 768], dtype=torch.float16, device='cuda'),
    torch.randn([1600, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([1600, 768], dtype=torch.float16, device='cuda'),
    torch.randn(1228800, dtype=torch.float16, device='cuda').as_strided([32, 12, 50, 64], [768, 64, 24576, 1]),  # arg79_1
    torch.randn([1600, 2304], dtype=torch.float16, device='cuda'),
    torch.randn([1600, 768], dtype=torch.float16, device='cuda'),
    torch.randn([1600, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([1600, 768], dtype=torch.float16, device='cuda'),
    torch.randn(1228800, dtype=torch.float16, device='cuda').as_strided([32, 12, 50, 64], [768, 64, 24576, 1]),  # arg66_1
    torch.randn([1600, 2304], dtype=torch.float16, device='cuda'),
    torch.randn([1600, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([32, 50, 768], dtype=torch.float32, device='cuda'),
    torch.randn([50, 768], dtype=torch.float32, device='cuda'),
    torch.randn([32, 50, 1], dtype=torch.float32, device='cuda'),
    torch.randn([32, 50, 1], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([32, 50, 1], dtype=torch.float32, device='cuda'),
    torch.randn([32, 50, 1], dtype=torch.float32, device='cuda'),
    torch.randn([32, 50, 768], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
