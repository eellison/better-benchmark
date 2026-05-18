"""
Standalone repro captured via capture_hook.
Label: timm_dm_nfnet_f0_training
Pattern hash: 40a599d5668c
Shape hash: fc65fdf2
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, getitem_1: "f32[1, 16, 1]", rsqrt: "f32[1, 16, 1]", getitem_3: "f32[1, 32, 1]", rsqrt_1: "f32[1, 32, 1]", getitem_5: "f32[1, 64, 1]", rsqrt_2: "f32[1, 64, 1]", getitem_7: "f32[1, 128, 1]", rsqrt_3: "f32[1, 128, 1]", getitem_9: "f32[1, 256, 1]", rsqrt_4: "f32[1, 256, 1]", getitem_11: "f32[1, 128, 1]", rsqrt_5: "f32[1, 128, 1]", getitem_13: "f32[1, 128, 1]", rsqrt_6: "f32[1, 128, 1]", getitem_15: "f32[1, 128, 1]", rsqrt_7: "f32[1, 128, 1]", getitem_17: "f32[1, 256, 1]", rsqrt_8: "f32[1, 256, 1]", getitem_19: "f32[1, 512, 1]", rsqrt_9: "f32[1, 512, 1]", getitem_21: "f32[1, 256, 1]", rsqrt_10: "f32[1, 256, 1]", getitem_23: "f32[1, 256, 1]", rsqrt_11: "f32[1, 256, 1]", getitem_25: "f32[1, 256, 1]", rsqrt_12: "f32[1, 256, 1]", getitem_27: "f32[1, 512, 1]", rsqrt_13: "f32[1, 512, 1]", getitem_29: "f32[1, 256, 1]", rsqrt_14: "f32[1, 256, 1]", getitem_31: "f32[1, 256, 1]", rsqrt_15: "f32[1, 256, 1]", getitem_33: "f32[1, 256, 1]", rsqrt_16: "f32[1, 256, 1]", getitem_35: "f32[1, 512, 1]", rsqrt_17: "f32[1, 512, 1]", getitem_37: "f32[1, 1536, 1]", rsqrt_18: "f32[1, 1536, 1]", getitem_39: "f32[1, 768, 1]", rsqrt_19: "f32[1, 768, 1]", getitem_41: "f32[1, 768, 1]", rsqrt_20: "f32[1, 768, 1]", getitem_43: "f32[1, 768, 1]", rsqrt_21: "f32[1, 768, 1]", getitem_45: "f32[1, 1536, 1]", rsqrt_22: "f32[1, 1536, 1]", getitem_47: "f32[1, 768, 1]", rsqrt_23: "f32[1, 768, 1]", getitem_49: "f32[1, 768, 1]", rsqrt_24: "f32[1, 768, 1]", getitem_51: "f32[1, 768, 1]", rsqrt_25: "f32[1, 768, 1]", getitem_53: "f32[1, 1536, 1]", rsqrt_26: "f32[1, 1536, 1]", getitem_55: "f32[1, 768, 1]", rsqrt_27: "f32[1, 768, 1]", getitem_57: "f32[1, 768, 1]", rsqrt_28: "f32[1, 768, 1]", getitem_59: "f32[1, 768, 1]", rsqrt_29: "f32[1, 768, 1]", getitem_61: "f32[1, 1536, 1]", rsqrt_30: "f32[1, 1536, 1]", getitem_63: "f32[1, 768, 1]", rsqrt_31: "f32[1, 768, 1]", getitem_65: "f32[1, 768, 1]", rsqrt_32: "f32[1, 768, 1]", getitem_67: "f32[1, 768, 1]", rsqrt_33: "f32[1, 768, 1]", getitem_69: "f32[1, 1536, 1]", rsqrt_34: "f32[1, 1536, 1]", getitem_71: "f32[1, 768, 1]", rsqrt_35: "f32[1, 768, 1]", getitem_73: "f32[1, 768, 1]", rsqrt_36: "f32[1, 768, 1]", getitem_75: "f32[1, 768, 1]", rsqrt_37: "f32[1, 768, 1]", getitem_77: "f32[1, 1536, 1]", rsqrt_38: "f32[1, 1536, 1]", getitem_79: "f32[1, 768, 1]", rsqrt_39: "f32[1, 768, 1]", getitem_81: "f32[1, 768, 1]", rsqrt_40: "f32[1, 768, 1]", getitem_83: "f32[1, 768, 1]", rsqrt_41: "f32[1, 768, 1]", getitem_85: "f32[1, 1536, 1]", rsqrt_42: "f32[1, 1536, 1]", getitem_87: "f32[1, 1536, 1]", rsqrt_43: "f32[1, 1536, 1]", getitem_89: "f32[1, 768, 1]", rsqrt_44: "f32[1, 768, 1]", getitem_91: "f32[1, 768, 1]", rsqrt_45: "f32[1, 768, 1]", getitem_93: "f32[1, 768, 1]", rsqrt_46: "f32[1, 768, 1]", getitem_95: "f32[1, 1536, 1]", rsqrt_47: "f32[1, 1536, 1]", getitem_97: "f32[1, 768, 1]", rsqrt_48: "f32[1, 768, 1]", getitem_99: "f32[1, 768, 1]", rsqrt_49: "f32[1, 768, 1]", getitem_101: "f32[1, 768, 1]", rsqrt_50: "f32[1, 768, 1]", getitem_103: "f32[1, 1536, 1]", rsqrt_51: "f32[1, 1536, 1]", getitem_105: "f32[1, 768, 1]", rsqrt_52: "f32[1, 768, 1]", getitem_107: "f32[1, 768, 1]", rsqrt_53: "f32[1, 768, 1]", getitem_109: "f32[1, 768, 1]", rsqrt_54: "f32[1, 768, 1]", getitem_111: "f32[1, 1536, 1]", rsqrt_55: "f32[1, 1536, 1]", getitem_113: "f32[1, 3072, 1]", rsqrt_56: "f32[1, 3072, 1]", convolution_80: "f32[8, 3072, 7, 7]", _shape_param_0, primals_233: "f32[1000, 3072]", add_110: "f32[8, 1536, 7, 7]", add_109: "f32[8, 1536, 7, 7]", add_91: "f32[8, 1536, 14, 14]", add_90: "f32[8, 1536, 14, 14]", add_82: "f32[8, 1536, 14, 14]", add_81: "f32[8, 1536, 14, 14]", add_73: "f32[8, 1536, 14, 14]", add_72: "f32[8, 1536, 14, 14]", add_64: "f32[8, 1536, 14, 14]", add_63: "f32[8, 1536, 14, 14]", add_55: "f32[8, 1536, 14, 14]", add_54: "f32[8, 1536, 14, 14]", add_36: "f32[8, 512, 28, 28]", add_35: "f32[8, 512, 28, 28]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        squeeze_dims: "f32[16]" = torch.ops.aten.squeeze.dims(getitem_1, [0, 2]);  getitem_1 = None
        squeeze_dims_1: "f32[16]" = torch.ops.aten.squeeze.dims(rsqrt, [0, 2]);  rsqrt = None
        squeeze_dims_2: "f32[32]" = torch.ops.aten.squeeze.dims(getitem_3, [0, 2]);  getitem_3 = None
        squeeze_dims_3: "f32[32]" = torch.ops.aten.squeeze.dims(rsqrt_1, [0, 2]);  rsqrt_1 = None
        squeeze_dims_4: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_5, [0, 2]);  getitem_5 = None
        squeeze_dims_5: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_2, [0, 2]);  rsqrt_2 = None
        squeeze_dims_6: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_7, [0, 2]);  getitem_7 = None
        squeeze_dims_7: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_3, [0, 2]);  rsqrt_3 = None
        squeeze_dims_8: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_9, [0, 2]);  getitem_9 = None
        squeeze_dims_9: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_4, [0, 2]);  rsqrt_4 = None
        squeeze_dims_10: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_11, [0, 2]);  getitem_11 = None
        squeeze_dims_11: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_5, [0, 2]);  rsqrt_5 = None
        squeeze_dims_12: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_13, [0, 2]);  getitem_13 = None
        squeeze_dims_13: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_6, [0, 2]);  rsqrt_6 = None
        squeeze_dims_14: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_15, [0, 2]);  getitem_15 = None
        squeeze_dims_15: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_7, [0, 2]);  rsqrt_7 = None
        squeeze_dims_16: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_17, [0, 2]);  getitem_17 = None
        squeeze_dims_17: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_8, [0, 2]);  rsqrt_8 = None
        squeeze_dims_18: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_19, [0, 2]);  getitem_19 = None
        squeeze_dims_19: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_9, [0, 2]);  rsqrt_9 = None
        squeeze_dims_20: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_21, [0, 2]);  getitem_21 = None
        squeeze_dims_21: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_10, [0, 2]);  rsqrt_10 = None
        squeeze_dims_22: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_23, [0, 2]);  getitem_23 = None
        squeeze_dims_23: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_11, [0, 2]);  rsqrt_11 = None
        squeeze_dims_24: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_25, [0, 2]);  getitem_25 = None
        squeeze_dims_25: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_12, [0, 2]);  rsqrt_12 = None
        squeeze_dims_26: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_27, [0, 2]);  getitem_27 = None
        squeeze_dims_27: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_13, [0, 2]);  rsqrt_13 = None
        squeeze_dims_28: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_29, [0, 2]);  getitem_29 = None
        squeeze_dims_29: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_14, [0, 2]);  rsqrt_14 = None
        squeeze_dims_30: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_31, [0, 2]);  getitem_31 = None
        squeeze_dims_31: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_15, [0, 2]);  rsqrt_15 = None
        squeeze_dims_32: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_33, [0, 2]);  getitem_33 = None
        squeeze_dims_33: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_16, [0, 2]);  rsqrt_16 = None
        squeeze_dims_34: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_35, [0, 2]);  getitem_35 = None
        squeeze_dims_35: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_17, [0, 2]);  rsqrt_17 = None
        squeeze_dims_36: "f32[1536]" = torch.ops.aten.squeeze.dims(getitem_37, [0, 2]);  getitem_37 = None
        squeeze_dims_37: "f32[1536]" = torch.ops.aten.squeeze.dims(rsqrt_18, [0, 2]);  rsqrt_18 = None
        squeeze_dims_38: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_39, [0, 2]);  getitem_39 = None
        squeeze_dims_39: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_19, [0, 2]);  rsqrt_19 = None
        squeeze_dims_40: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_41, [0, 2]);  getitem_41 = None
        squeeze_dims_41: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_20, [0, 2]);  rsqrt_20 = None
        squeeze_dims_42: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_43, [0, 2]);  getitem_43 = None
        squeeze_dims_43: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_21, [0, 2]);  rsqrt_21 = None
        squeeze_dims_44: "f32[1536]" = torch.ops.aten.squeeze.dims(getitem_45, [0, 2]);  getitem_45 = None
        squeeze_dims_45: "f32[1536]" = torch.ops.aten.squeeze.dims(rsqrt_22, [0, 2]);  rsqrt_22 = None
        squeeze_dims_46: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_47, [0, 2]);  getitem_47 = None
        squeeze_dims_47: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_23, [0, 2]);  rsqrt_23 = None
        squeeze_dims_48: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_49, [0, 2]);  getitem_49 = None
        squeeze_dims_49: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_24, [0, 2]);  rsqrt_24 = None
        squeeze_dims_50: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_51, [0, 2]);  getitem_51 = None
        squeeze_dims_51: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_25, [0, 2]);  rsqrt_25 = None
        squeeze_dims_52: "f32[1536]" = torch.ops.aten.squeeze.dims(getitem_53, [0, 2]);  getitem_53 = None
        squeeze_dims_53: "f32[1536]" = torch.ops.aten.squeeze.dims(rsqrt_26, [0, 2]);  rsqrt_26 = None
        squeeze_dims_54: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_55, [0, 2]);  getitem_55 = None
        squeeze_dims_55: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_27, [0, 2]);  rsqrt_27 = None
        squeeze_dims_56: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_57, [0, 2]);  getitem_57 = None
        squeeze_dims_57: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_28, [0, 2]);  rsqrt_28 = None
        squeeze_dims_58: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_59, [0, 2]);  getitem_59 = None
        squeeze_dims_59: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_29, [0, 2]);  rsqrt_29 = None
        squeeze_dims_60: "f32[1536]" = torch.ops.aten.squeeze.dims(getitem_61, [0, 2]);  getitem_61 = None
        squeeze_dims_61: "f32[1536]" = torch.ops.aten.squeeze.dims(rsqrt_30, [0, 2]);  rsqrt_30 = None
        squeeze_dims_62: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_63, [0, 2]);  getitem_63 = None
        squeeze_dims_63: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_31, [0, 2]);  rsqrt_31 = None
        squeeze_dims_64: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_65, [0, 2]);  getitem_65 = None
        squeeze_dims_65: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_32, [0, 2]);  rsqrt_32 = None
        squeeze_dims_66: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_67, [0, 2]);  getitem_67 = None
        squeeze_dims_67: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_33, [0, 2]);  rsqrt_33 = None
        squeeze_dims_68: "f32[1536]" = torch.ops.aten.squeeze.dims(getitem_69, [0, 2]);  getitem_69 = None
        squeeze_dims_69: "f32[1536]" = torch.ops.aten.squeeze.dims(rsqrt_34, [0, 2]);  rsqrt_34 = None
        squeeze_dims_70: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_71, [0, 2]);  getitem_71 = None
        squeeze_dims_71: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_35, [0, 2]);  rsqrt_35 = None
        squeeze_dims_72: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_73, [0, 2]);  getitem_73 = None
        squeeze_dims_73: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_36, [0, 2]);  rsqrt_36 = None
        squeeze_dims_74: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_75, [0, 2]);  getitem_75 = None
        squeeze_dims_75: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_37, [0, 2]);  rsqrt_37 = None
        squeeze_dims_76: "f32[1536]" = torch.ops.aten.squeeze.dims(getitem_77, [0, 2]);  getitem_77 = None
        squeeze_dims_77: "f32[1536]" = torch.ops.aten.squeeze.dims(rsqrt_38, [0, 2]);  rsqrt_38 = None
        squeeze_dims_78: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_79, [0, 2]);  getitem_79 = None
        squeeze_dims_79: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_39, [0, 2]);  rsqrt_39 = None
        squeeze_dims_80: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_81, [0, 2]);  getitem_81 = None
        squeeze_dims_81: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_40, [0, 2]);  rsqrt_40 = None
        squeeze_dims_82: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_83, [0, 2]);  getitem_83 = None
        squeeze_dims_83: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_41, [0, 2]);  rsqrt_41 = None
        squeeze_dims_84: "f32[1536]" = torch.ops.aten.squeeze.dims(getitem_85, [0, 2]);  getitem_85 = None
        squeeze_dims_85: "f32[1536]" = torch.ops.aten.squeeze.dims(rsqrt_42, [0, 2]);  rsqrt_42 = None
        squeeze_dims_86: "f32[1536]" = torch.ops.aten.squeeze.dims(getitem_87, [0, 2]);  getitem_87 = None
        squeeze_dims_87: "f32[1536]" = torch.ops.aten.squeeze.dims(rsqrt_43, [0, 2]);  rsqrt_43 = None
        squeeze_dims_88: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_89, [0, 2]);  getitem_89 = None
        squeeze_dims_89: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_44, [0, 2]);  rsqrt_44 = None
        squeeze_dims_90: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_91, [0, 2]);  getitem_91 = None
        squeeze_dims_91: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_45, [0, 2]);  rsqrt_45 = None
        squeeze_dims_92: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_93, [0, 2]);  getitem_93 = None
        squeeze_dims_93: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_46, [0, 2]);  rsqrt_46 = None
        squeeze_dims_94: "f32[1536]" = torch.ops.aten.squeeze.dims(getitem_95, [0, 2]);  getitem_95 = None
        squeeze_dims_95: "f32[1536]" = torch.ops.aten.squeeze.dims(rsqrt_47, [0, 2]);  rsqrt_47 = None
        squeeze_dims_96: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_97, [0, 2]);  getitem_97 = None
        squeeze_dims_97: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_48, [0, 2]);  rsqrt_48 = None
        squeeze_dims_98: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_99, [0, 2]);  getitem_99 = None
        squeeze_dims_99: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_49, [0, 2]);  rsqrt_49 = None
        squeeze_dims_100: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_101, [0, 2]);  getitem_101 = None
        squeeze_dims_101: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_50, [0, 2]);  rsqrt_50 = None
        squeeze_dims_102: "f32[1536]" = torch.ops.aten.squeeze.dims(getitem_103, [0, 2]);  getitem_103 = None
        squeeze_dims_103: "f32[1536]" = torch.ops.aten.squeeze.dims(rsqrt_51, [0, 2]);  rsqrt_51 = None
        squeeze_dims_104: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_105, [0, 2]);  getitem_105 = None
        squeeze_dims_105: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_52, [0, 2]);  rsqrt_52 = None
        squeeze_dims_106: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_107, [0, 2]);  getitem_107 = None
        squeeze_dims_107: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_53, [0, 2]);  rsqrt_53 = None
        squeeze_dims_108: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_109, [0, 2]);  getitem_109 = None
        squeeze_dims_109: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_54, [0, 2]);  rsqrt_54 = None
        squeeze_dims_110: "f32[1536]" = torch.ops.aten.squeeze.dims(getitem_111, [0, 2]);  getitem_111 = None
        squeeze_dims_111: "f32[1536]" = torch.ops.aten.squeeze.dims(rsqrt_55, [0, 2]);  rsqrt_55 = None
        squeeze_dims_112: "f32[3072]" = torch.ops.aten.squeeze.dims(getitem_113, [0, 2]);  getitem_113 = None
        squeeze_dims_113: "f32[3072]" = torch.ops.aten.squeeze.dims(rsqrt_56, [0, 2]);  rsqrt_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:135 in gelu, code: return F.gelu(x)
        mul_tensor: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(convolution_80, 0.5)
        mul_tensor_1: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(convolution_80, 0.7071067811865476);  convolution_80 = None
        erf_default: "f32[8, 3072, 7, 7]" = torch.ops.aten.erf.default(mul_tensor_1);  mul_tensor_1 = None
        add_tensor: "f32[8, 3072, 7, 7]" = torch.ops.aten.add.Tensor(erf_default, 1);  erf_default = None
        mul_tensor_2: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(mul_tensor, add_tensor);  mul_tensor = add_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:89 in forward, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
        mul_tensor_3: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(mul_tensor_2, 1.7015043497085571);  mul_tensor_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/adaptive_avgmax_pool.py:172 in forward, code: x = self.pool(x)
        mean_dim: "f32[8, 3072, 1, 1]" = torch.ops.aten.mean.dim(mul_tensor_3, [-1, -2], True);  mul_tensor_3 = None
        as_strided_default: "f32[8, 3072, 1, 1]" = torch.ops.aten.as_strided.default(mean_dim, [8, 3072, 1, 1], [3072, 1, 3072, 3072]);  mean_dim = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/adaptive_avgmax_pool.py:173 in forward, code: x = self.flatten(x)
        reshape_default: "f32[8, 3072]" = torch.ops.aten.reshape.default(as_strided_default, _shape_param_0);  as_strided_default = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/classifier.py:141 in forward, code: x = self.fc(x)
        permute_default: "f32[3072, 1000]" = torch.ops.aten.permute.default(primals_233, [1, 0]);  primals_233 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        unsqueeze_default: "f32[1, 3072]" = torch.ops.aten.unsqueeze.default(squeeze_dims_112, 0);  squeeze_dims_112 = None
        unsqueeze_default_1: "f32[1, 3072, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, 2);  unsqueeze_default = None
        unsqueeze_default_2: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(squeeze_dims_110, 0);  squeeze_dims_110 = None
        unsqueeze_default_3: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, 2);  unsqueeze_default_2 = None
        unsqueeze_default_4: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_dims_108, 0);  squeeze_dims_108 = None
        unsqueeze_default_5: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 2);  unsqueeze_default_4 = None
        unsqueeze_default_6: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_dims_106, 0);  squeeze_dims_106 = None
        unsqueeze_default_7: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, 2);  unsqueeze_default_6 = None
        unsqueeze_default_8: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_dims_104, 0);  squeeze_dims_104 = None
        unsqueeze_default_9: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_8, 2);  unsqueeze_default_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:135 in gelu, code: return F.gelu(x)
        mul_tensor_4: "f32[8, 1536, 7, 7]" = torch.ops.aten.mul.Tensor(add_110, 0.5);  add_110 = None
        mul_tensor_5: "f32[8, 1536, 7, 7]" = torch.ops.aten.mul.Tensor(add_109, add_109)
        mul_tensor_6: "f32[8, 1536, 7, 7]" = torch.ops.aten.mul.Tensor(mul_tensor_5, -0.5);  mul_tensor_5 = None
        exp_default: "f32[8, 1536, 7, 7]" = torch.ops.aten.exp.default(mul_tensor_6);  mul_tensor_6 = None
        mul_tensor_7: "f32[8, 1536, 7, 7]" = torch.ops.aten.mul.Tensor(exp_default, 0.3989422804014327);  exp_default = None
        mul_tensor_8: "f32[8, 1536, 7, 7]" = torch.ops.aten.mul.Tensor(add_109, mul_tensor_7);  add_109 = mul_tensor_7 = None
        add_tensor_1: "f32[8, 1536, 7, 7]" = torch.ops.aten.add.Tensor(mul_tensor_4, mul_tensor_8);  mul_tensor_4 = mul_tensor_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        unsqueeze_default_10: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(squeeze_dims_102, 0);  squeeze_dims_102 = None
        unsqueeze_default_11: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_10, 2);  unsqueeze_default_10 = None
        unsqueeze_default_12: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_dims_100, 0);  squeeze_dims_100 = None
        unsqueeze_default_13: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_12, 2);  unsqueeze_default_12 = None
        unsqueeze_default_14: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_dims_98, 0);  squeeze_dims_98 = None
        unsqueeze_default_15: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_14, 2);  unsqueeze_default_14 = None
        unsqueeze_default_16: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_dims_96, 0);  squeeze_dims_96 = None
        unsqueeze_default_17: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_16, 2);  unsqueeze_default_16 = None
        unsqueeze_default_18: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(squeeze_dims_94, 0);  squeeze_dims_94 = None
        unsqueeze_default_19: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_18, 2);  unsqueeze_default_18 = None
        unsqueeze_default_20: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_dims_92, 0);  squeeze_dims_92 = None
        unsqueeze_default_21: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_20, 2);  unsqueeze_default_20 = None
        unsqueeze_default_22: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_dims_90, 0);  squeeze_dims_90 = None
        unsqueeze_default_23: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_22, 2);  unsqueeze_default_22 = None
        unsqueeze_default_24: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_dims_88, 0);  squeeze_dims_88 = None
        unsqueeze_default_25: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_24, 2);  unsqueeze_default_24 = None
        unsqueeze_default_26: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(squeeze_dims_86, 0);  squeeze_dims_86 = None
        unsqueeze_default_27: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_26, 2);  unsqueeze_default_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:135 in gelu, code: return F.gelu(x)
        mul_tensor_9: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(add_91, 0.5);  add_91 = None
        mul_tensor_10: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(add_90, add_90)
        mul_tensor_11: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_tensor_10, -0.5);  mul_tensor_10 = None
        exp_default_1: "f32[8, 1536, 14, 14]" = torch.ops.aten.exp.default(mul_tensor_11);  mul_tensor_11 = None
        mul_tensor_12: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(exp_default_1, 0.3989422804014327);  exp_default_1 = None
        mul_tensor_13: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(add_90, mul_tensor_12);  add_90 = mul_tensor_12 = None
        add_tensor_2: "f32[8, 1536, 14, 14]" = torch.ops.aten.add.Tensor(mul_tensor_9, mul_tensor_13);  mul_tensor_9 = mul_tensor_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        unsqueeze_default_28: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(squeeze_dims_84, 0);  squeeze_dims_84 = None
        unsqueeze_default_29: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_28, 2);  unsqueeze_default_28 = None
        unsqueeze_default_30: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_dims_82, 0);  squeeze_dims_82 = None
        unsqueeze_default_31: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_30, 2);  unsqueeze_default_30 = None
        unsqueeze_default_32: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_dims_80, 0);  squeeze_dims_80 = None
        unsqueeze_default_33: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_32, 2);  unsqueeze_default_32 = None
        unsqueeze_default_34: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_dims_78, 0);  squeeze_dims_78 = None
        unsqueeze_default_35: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_34, 2);  unsqueeze_default_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:135 in gelu, code: return F.gelu(x)
        mul_tensor_14: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(add_82, 0.5);  add_82 = None
        mul_tensor_15: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(add_81, add_81)
        mul_tensor_16: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_tensor_15, -0.5);  mul_tensor_15 = None
        exp_default_2: "f32[8, 1536, 14, 14]" = torch.ops.aten.exp.default(mul_tensor_16);  mul_tensor_16 = None
        mul_tensor_17: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(exp_default_2, 0.3989422804014327);  exp_default_2 = None
        mul_tensor_18: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(add_81, mul_tensor_17);  add_81 = mul_tensor_17 = None
        add_tensor_3: "f32[8, 1536, 14, 14]" = torch.ops.aten.add.Tensor(mul_tensor_14, mul_tensor_18);  mul_tensor_14 = mul_tensor_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        unsqueeze_default_36: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(squeeze_dims_76, 0);  squeeze_dims_76 = None
        unsqueeze_default_37: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_36, 2);  unsqueeze_default_36 = None
        unsqueeze_default_38: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_dims_74, 0);  squeeze_dims_74 = None
        unsqueeze_default_39: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_38, 2);  unsqueeze_default_38 = None
        unsqueeze_default_40: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_dims_72, 0);  squeeze_dims_72 = None
        unsqueeze_default_41: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_40, 2);  unsqueeze_default_40 = None
        unsqueeze_default_42: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_dims_70, 0);  squeeze_dims_70 = None
        unsqueeze_default_43: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_42, 2);  unsqueeze_default_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:135 in gelu, code: return F.gelu(x)
        mul_tensor_19: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(add_73, 0.5);  add_73 = None
        mul_tensor_20: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(add_72, add_72)
        mul_tensor_21: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_tensor_20, -0.5);  mul_tensor_20 = None
        exp_default_3: "f32[8, 1536, 14, 14]" = torch.ops.aten.exp.default(mul_tensor_21);  mul_tensor_21 = None
        mul_tensor_22: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(exp_default_3, 0.3989422804014327);  exp_default_3 = None
        mul_tensor_23: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(add_72, mul_tensor_22);  add_72 = mul_tensor_22 = None
        add_tensor_4: "f32[8, 1536, 14, 14]" = torch.ops.aten.add.Tensor(mul_tensor_19, mul_tensor_23);  mul_tensor_19 = mul_tensor_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        unsqueeze_default_44: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(squeeze_dims_68, 0);  squeeze_dims_68 = None
        unsqueeze_default_45: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_44, 2);  unsqueeze_default_44 = None
        unsqueeze_default_46: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_dims_66, 0);  squeeze_dims_66 = None
        unsqueeze_default_47: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_46, 2);  unsqueeze_default_46 = None
        unsqueeze_default_48: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_dims_64, 0);  squeeze_dims_64 = None
        unsqueeze_default_49: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_48, 2);  unsqueeze_default_48 = None
        unsqueeze_default_50: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_dims_62, 0);  squeeze_dims_62 = None
        unsqueeze_default_51: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_50, 2);  unsqueeze_default_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:135 in gelu, code: return F.gelu(x)
        mul_tensor_24: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(add_64, 0.5);  add_64 = None
        mul_tensor_25: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(add_63, add_63)
        mul_tensor_26: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_tensor_25, -0.5);  mul_tensor_25 = None
        exp_default_4: "f32[8, 1536, 14, 14]" = torch.ops.aten.exp.default(mul_tensor_26);  mul_tensor_26 = None
        mul_tensor_27: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(exp_default_4, 0.3989422804014327);  exp_default_4 = None
        mul_tensor_28: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(add_63, mul_tensor_27);  add_63 = mul_tensor_27 = None
        add_tensor_5: "f32[8, 1536, 14, 14]" = torch.ops.aten.add.Tensor(mul_tensor_24, mul_tensor_28);  mul_tensor_24 = mul_tensor_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        unsqueeze_default_52: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(squeeze_dims_60, 0);  squeeze_dims_60 = None
        unsqueeze_default_53: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_52, 2);  unsqueeze_default_52 = None
        unsqueeze_default_54: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_dims_58, 0);  squeeze_dims_58 = None
        unsqueeze_default_55: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_54, 2);  unsqueeze_default_54 = None
        unsqueeze_default_56: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_dims_56, 0);  squeeze_dims_56 = None
        unsqueeze_default_57: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_56, 2);  unsqueeze_default_56 = None
        unsqueeze_default_58: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_dims_54, 0);  squeeze_dims_54 = None
        unsqueeze_default_59: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_58, 2);  unsqueeze_default_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:135 in gelu, code: return F.gelu(x)
        mul_tensor_29: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(add_55, 0.5);  add_55 = None
        mul_tensor_30: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(add_54, add_54)
        mul_tensor_31: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_tensor_30, -0.5);  mul_tensor_30 = None
        exp_default_5: "f32[8, 1536, 14, 14]" = torch.ops.aten.exp.default(mul_tensor_31);  mul_tensor_31 = None
        mul_tensor_32: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(exp_default_5, 0.3989422804014327);  exp_default_5 = None
        mul_tensor_33: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(add_54, mul_tensor_32);  add_54 = mul_tensor_32 = None
        add_tensor_6: "f32[8, 1536, 14, 14]" = torch.ops.aten.add.Tensor(mul_tensor_29, mul_tensor_33);  mul_tensor_29 = mul_tensor_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        unsqueeze_default_60: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(squeeze_dims_52, 0);  squeeze_dims_52 = None
        unsqueeze_default_61: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_60, 2);  unsqueeze_default_60 = None
        unsqueeze_default_62: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_dims_50, 0);  squeeze_dims_50 = None
        unsqueeze_default_63: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_62, 2);  unsqueeze_default_62 = None
        unsqueeze_default_64: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_dims_48, 0);  squeeze_dims_48 = None
        unsqueeze_default_65: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_64, 2);  unsqueeze_default_64 = None
        unsqueeze_default_66: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_dims_46, 0);  squeeze_dims_46 = None
        unsqueeze_default_67: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_66, 2);  unsqueeze_default_66 = None
        unsqueeze_default_68: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(squeeze_dims_44, 0);  squeeze_dims_44 = None
        unsqueeze_default_69: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_68, 2);  unsqueeze_default_68 = None
        unsqueeze_default_70: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_dims_42, 0);  squeeze_dims_42 = None
        unsqueeze_default_71: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_70, 2);  unsqueeze_default_70 = None
        unsqueeze_default_72: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_dims_40, 0);  squeeze_dims_40 = None
        unsqueeze_default_73: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_72, 2);  unsqueeze_default_72 = None
        unsqueeze_default_74: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_dims_38, 0);  squeeze_dims_38 = None
        unsqueeze_default_75: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_74, 2);  unsqueeze_default_74 = None
        unsqueeze_default_76: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(squeeze_dims_36, 0);  squeeze_dims_36 = None
        unsqueeze_default_77: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_76, 2);  unsqueeze_default_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:135 in gelu, code: return F.gelu(x)
        mul_tensor_34: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(add_36, 0.5);  add_36 = None
        mul_tensor_35: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(add_35, add_35)
        mul_tensor_36: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(mul_tensor_35, -0.5);  mul_tensor_35 = None
        exp_default_6: "f32[8, 512, 28, 28]" = torch.ops.aten.exp.default(mul_tensor_36);  mul_tensor_36 = None
        mul_tensor_37: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(exp_default_6, 0.3989422804014327);  exp_default_6 = None
        mul_tensor_38: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(add_35, mul_tensor_37);  add_35 = mul_tensor_37 = None
        add_tensor_7: "f32[8, 512, 28, 28]" = torch.ops.aten.add.Tensor(mul_tensor_34, mul_tensor_38);  mul_tensor_34 = mul_tensor_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        unsqueeze_default_78: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_dims_34, 0);  squeeze_dims_34 = None
        unsqueeze_default_79: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_78, 2);  unsqueeze_default_78 = None
        unsqueeze_default_80: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_dims_32, 0);  squeeze_dims_32 = None
        unsqueeze_default_81: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_80, 2);  unsqueeze_default_80 = None
        unsqueeze_default_82: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_dims_30, 0);  squeeze_dims_30 = None
        unsqueeze_default_83: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_82, 2);  unsqueeze_default_82 = None
        unsqueeze_default_84: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_dims_28, 0);  squeeze_dims_28 = None
        unsqueeze_default_85: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_84, 2);  unsqueeze_default_84 = None
        unsqueeze_default_86: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_dims_26, 0);  squeeze_dims_26 = None
        unsqueeze_default_87: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_86, 2);  unsqueeze_default_86 = None
        unsqueeze_default_88: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_dims_24, 0);  squeeze_dims_24 = None
        unsqueeze_default_89: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_88, 2);  unsqueeze_default_88 = None
        unsqueeze_default_90: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_dims_22, 0);  squeeze_dims_22 = None
        unsqueeze_default_91: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_90, 2);  unsqueeze_default_90 = None
        unsqueeze_default_92: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_dims_20, 0);  squeeze_dims_20 = None
        unsqueeze_default_93: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_92, 2);  unsqueeze_default_92 = None
        unsqueeze_default_94: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_dims_18, 0);  squeeze_dims_18 = None
        unsqueeze_default_95: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_94, 2);  unsqueeze_default_94 = None
        unsqueeze_default_96: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_dims_16, 0);  squeeze_dims_16 = None
        unsqueeze_default_97: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_96, 2);  unsqueeze_default_96 = None
        unsqueeze_default_98: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_dims_14, 0);  squeeze_dims_14 = None
        unsqueeze_default_99: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_98, 2);  unsqueeze_default_98 = None
        unsqueeze_default_100: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_dims_12, 0);  squeeze_dims_12 = None
        unsqueeze_default_101: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_100, 2);  unsqueeze_default_100 = None
        unsqueeze_default_102: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_dims_10, 0);  squeeze_dims_10 = None
        unsqueeze_default_103: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_102, 2);  unsqueeze_default_102 = None
        unsqueeze_default_104: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_dims_8, 0);  squeeze_dims_8 = None
        unsqueeze_default_105: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_104, 2);  unsqueeze_default_104 = None
        unsqueeze_default_106: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_dims_6, 0);  squeeze_dims_6 = None
        unsqueeze_default_107: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_106, 2);  unsqueeze_default_106 = None
        unsqueeze_default_108: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_dims_4, 0);  squeeze_dims_4 = None
        unsqueeze_default_109: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_108, 2);  unsqueeze_default_108 = None
        unsqueeze_default_110: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(squeeze_dims_2, 0);  squeeze_dims_2 = None
        unsqueeze_default_111: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_110, 2);  unsqueeze_default_110 = None
        unsqueeze_default_112: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(squeeze_dims, 0);  squeeze_dims = None
        unsqueeze_default_113: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_112, 2);  unsqueeze_default_112 = None
        return (squeeze_dims_1, squeeze_dims_3, squeeze_dims_5, squeeze_dims_7, squeeze_dims_9, squeeze_dims_11, squeeze_dims_13, squeeze_dims_15, squeeze_dims_17, squeeze_dims_19, squeeze_dims_21, squeeze_dims_23, squeeze_dims_25, squeeze_dims_27, squeeze_dims_29, squeeze_dims_31, squeeze_dims_33, squeeze_dims_35, squeeze_dims_37, squeeze_dims_39, squeeze_dims_41, squeeze_dims_43, squeeze_dims_45, squeeze_dims_47, squeeze_dims_49, squeeze_dims_51, squeeze_dims_53, squeeze_dims_55, squeeze_dims_57, squeeze_dims_59, squeeze_dims_61, squeeze_dims_63, squeeze_dims_65, squeeze_dims_67, squeeze_dims_69, squeeze_dims_71, squeeze_dims_73, squeeze_dims_75, squeeze_dims_77, squeeze_dims_79, squeeze_dims_81, squeeze_dims_83, squeeze_dims_85, squeeze_dims_87, squeeze_dims_89, squeeze_dims_91, squeeze_dims_93, squeeze_dims_95, squeeze_dims_97, squeeze_dims_99, squeeze_dims_101, squeeze_dims_103, squeeze_dims_105, squeeze_dims_107, squeeze_dims_109, squeeze_dims_111, squeeze_dims_113, reshape_default, permute_default, unsqueeze_default_1, unsqueeze_default_3, unsqueeze_default_5, unsqueeze_default_7, unsqueeze_default_9, add_tensor_1, unsqueeze_default_11, unsqueeze_default_13, unsqueeze_default_15, unsqueeze_default_17, unsqueeze_default_19, unsqueeze_default_21, unsqueeze_default_23, unsqueeze_default_25, unsqueeze_default_27, add_tensor_2, unsqueeze_default_29, unsqueeze_default_31, unsqueeze_default_33, unsqueeze_default_35, add_tensor_3, unsqueeze_default_37, unsqueeze_default_39, unsqueeze_default_41, unsqueeze_default_43, add_tensor_4, unsqueeze_default_45, unsqueeze_default_47, unsqueeze_default_49, unsqueeze_default_51, add_tensor_5, unsqueeze_default_53, unsqueeze_default_55, unsqueeze_default_57, unsqueeze_default_59, add_tensor_6, unsqueeze_default_61, unsqueeze_default_63, unsqueeze_default_65, unsqueeze_default_67, unsqueeze_default_69, unsqueeze_default_71, unsqueeze_default_73, unsqueeze_default_75, unsqueeze_default_77, add_tensor_7, unsqueeze_default_79, unsqueeze_default_81, unsqueeze_default_83, unsqueeze_default_85, unsqueeze_default_87, unsqueeze_default_89, unsqueeze_default_91, unsqueeze_default_93, unsqueeze_default_95, unsqueeze_default_97, unsqueeze_default_99, unsqueeze_default_101, unsqueeze_default_103, unsqueeze_default_105, unsqueeze_default_107, unsqueeze_default_109, unsqueeze_default_111, unsqueeze_default_113)


def _default_make_inputs():
    return [
    torch.randn([1, 16, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 16, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 32, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 32, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 64, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 64, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 256, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 256, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 256, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 256, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 256, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 256, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 256, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 256, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 256, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 256, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 256, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 256, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 256, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 256, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 256, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 256, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 1536, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 1536, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 768, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 768, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 768, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 768, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 768, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 768, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 1536, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 1536, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 768, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 768, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 768, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 768, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 768, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 768, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 1536, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 1536, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 768, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 768, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 768, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 768, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 768, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 768, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 1536, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 1536, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 768, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 768, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 768, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 768, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 768, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 768, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 1536, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 1536, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 768, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 768, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 768, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 768, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 768, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 768, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 1536, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 1536, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 768, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 768, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 768, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 768, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 768, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 768, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 1536, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 1536, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 1536, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 1536, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 768, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 768, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 768, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 768, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 768, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 768, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 1536, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 1536, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 768, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 768, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 768, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 768, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 768, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 768, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 1536, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 1536, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 768, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 768, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 768, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 768, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 768, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 768, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 1536, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 1536, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 3072, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 3072, 1], dtype=torch.float32, device='cuda'),
    torch.randn(1204224, dtype=torch.float32, device='cuda').as_strided([8, 3072, 7, 7], [150528, 1, 21504, 3072]),  # convolution_80
    [8, 3072],  # _shape_param_0
    torch.randn([1000, 3072], dtype=torch.float32, device='cuda'),
    torch.randn(602112, dtype=torch.float32, device='cuda').as_strided([8, 1536, 7, 7], [75264, 1, 10752, 1536]),  # add_110
    torch.randn(602112, dtype=torch.float32, device='cuda').as_strided([8, 1536, 7, 7], [75264, 1, 10752, 1536]),  # add_109
    torch.randn(2408448, dtype=torch.float32, device='cuda').as_strided([8, 1536, 14, 14], [301056, 1, 21504, 1536]),  # add_91
    torch.randn(2408448, dtype=torch.float32, device='cuda').as_strided([8, 1536, 14, 14], [301056, 1, 21504, 1536]),  # add_90
    torch.randn(2408448, dtype=torch.float32, device='cuda').as_strided([8, 1536, 14, 14], [301056, 1, 21504, 1536]),  # add_82
    torch.randn(2408448, dtype=torch.float32, device='cuda').as_strided([8, 1536, 14, 14], [301056, 1, 21504, 1536]),  # add_81
    torch.randn(2408448, dtype=torch.float32, device='cuda').as_strided([8, 1536, 14, 14], [301056, 1, 21504, 1536]),  # add_73
    torch.randn(2408448, dtype=torch.float32, device='cuda').as_strided([8, 1536, 14, 14], [301056, 1, 21504, 1536]),  # add_72
    torch.randn(2408448, dtype=torch.float32, device='cuda').as_strided([8, 1536, 14, 14], [301056, 1, 21504, 1536]),  # add_64
    torch.randn(2408448, dtype=torch.float32, device='cuda').as_strided([8, 1536, 14, 14], [301056, 1, 21504, 1536]),  # add_63
    torch.randn(2408448, dtype=torch.float32, device='cuda').as_strided([8, 1536, 14, 14], [301056, 1, 21504, 1536]),  # add_55
    torch.randn(2408448, dtype=torch.float32, device='cuda').as_strided([8, 1536, 14, 14], [301056, 1, 21504, 1536]),  # add_54
    torch.randn(3211264, dtype=torch.float32, device='cuda').as_strided([8, 512, 28, 28], [401408, 1, 14336, 512]),  # add_36
    torch.randn(3211264, dtype=torch.float32, device='cuda').as_strided([8, 512, 28, 28], [401408, 1, 14336, 512]),  # add_35
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
