import torch
from torch import device
from math import inf, nan

class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[128, 3, 256, 256][196608, 1, 768, 3]cuda:0", primals_2: "f32[768, 3, 16, 16][768, 1, 48, 3]cuda:0", primals_4: "f32[1, 256, 768][196608, 768, 1]cuda:0", primals_5: "f32[768][1]cuda:0", primals_7: "f32[2304, 768][768, 1]cuda:0", primals_9: "f32[768, 768][768, 1]cuda:0", primals_11: "f32[768][1]cuda:0", primals_13: "f32[3072, 768][768, 1]cuda:0", primals_15: "f32[768, 3072][3072, 1]cuda:0", primals_17: "f32[768][1]cuda:0", primals_19: "f32[2304, 768][768, 1]cuda:0", primals_21: "f32[768, 768][768, 1]cuda:0", primals_23: "f32[768][1]cuda:0", primals_25: "f32[3072, 768][768, 1]cuda:0", primals_27: "f32[768, 3072][3072, 1]cuda:0", primals_29: "f32[768][1]cuda:0", primals_31: "f32[2304, 768][768, 1]cuda:0", primals_33: "f32[768, 768][768, 1]cuda:0", primals_35: "f32[768][1]cuda:0", primals_37: "f32[3072, 768][768, 1]cuda:0", primals_39: "f32[768, 3072][3072, 1]cuda:0", primals_41: "f32[768][1]cuda:0", primals_43: "f32[2304, 768][768, 1]cuda:0", primals_45: "f32[768, 768][768, 1]cuda:0", primals_47: "f32[768][1]cuda:0", primals_49: "f32[3072, 768][768, 1]cuda:0", primals_51: "f32[768, 3072][3072, 1]cuda:0", primals_53: "f32[768][1]cuda:0", primals_55: "f32[2304, 768][768, 1]cuda:0", primals_57: "f32[768, 768][768, 1]cuda:0", primals_59: "f32[768][1]cuda:0", primals_61: "f32[3072, 768][768, 1]cuda:0", primals_63: "f32[768, 3072][3072, 1]cuda:0", primals_65: "f32[768][1]cuda:0", primals_67: "f32[2304, 768][768, 1]cuda:0", primals_69: "f32[768, 768][768, 1]cuda:0", primals_71: "f32[768][1]cuda:0", primals_73: "f32[3072, 768][768, 1]cuda:0", primals_75: "f32[768, 3072][3072, 1]cuda:0", primals_77: "f32[768][1]cuda:0", primals_79: "f32[2304, 768][768, 1]cuda:0", primals_81: "f32[768, 768][768, 1]cuda:0", primals_83: "f32[768][1]cuda:0", primals_85: "f32[3072, 768][768, 1]cuda:0", primals_87: "f32[768, 3072][3072, 1]cuda:0", primals_89: "f32[768][1]cuda:0", primals_91: "f32[2304, 768][768, 1]cuda:0", primals_93: "f32[768, 768][768, 1]cuda:0", primals_95: "f32[768][1]cuda:0", primals_97: "f32[3072, 768][768, 1]cuda:0", primals_99: "f32[768, 3072][3072, 1]cuda:0", primals_101: "f32[768][1]cuda:0", primals_103: "f32[2304, 768][768, 1]cuda:0", primals_105: "f32[768, 768][768, 1]cuda:0", primals_107: "f32[768][1]cuda:0", primals_109: "f32[3072, 768][768, 1]cuda:0", primals_111: "f32[768, 3072][3072, 1]cuda:0", primals_113: "f32[768][1]cuda:0", primals_115: "f32[2304, 768][768, 1]cuda:0", primals_117: "f32[768, 768][768, 1]cuda:0", primals_119: "f32[768][1]cuda:0", primals_121: "f32[3072, 768][768, 1]cuda:0", primals_123: "f32[768, 3072][3072, 1]cuda:0", primals_125: "f32[768][1]cuda:0", primals_127: "f32[2304, 768][768, 1]cuda:0", primals_129: "f32[768, 768][768, 1]cuda:0", primals_131: "f32[768][1]cuda:0", primals_133: "f32[3072, 768][768, 1]cuda:0", primals_135: "f32[768, 3072][3072, 1]cuda:0", primals_137: "f32[768][1]cuda:0", primals_139: "f32[2304, 768][768, 1]cuda:0", primals_141: "f32[768, 768][768, 1]cuda:0", primals_143: "f32[768][1]cuda:0", primals_145: "f32[3072, 768][768, 1]cuda:0", primals_147: "f32[768, 3072][3072, 1]cuda:0", primals_149: "f32[768][1]cuda:0", primals_151: "f32[1, 1, 768][768, 768, 1]cuda:0", primals_152: "f32[768, 768][768, 1]cuda:0", primals_154: "f32[1536, 768][768, 1]cuda:0", primals_156: "f32[768, 768][768, 1]cuda:0", primals_158: "f32[768][1]cuda:0", primals_160: "f32[3072, 768][768, 1]cuda:0", primals_162: "f32[768, 3072][3072, 1]cuda:0", convolution: "f32[128, 768, 16, 16][196608, 1, 12288, 768]cuda:0", getitem_1: "f32[128, 256, 1][256, 1, 1]cuda:0", rsqrt: "f32[128, 256, 1][256, 1, 1]cuda:0", view_1: "f32[32768, 768][768, 1]cuda:0", getitem_2: "f32[128, 12, 256, 64][589824, 64, 2304, 1]cuda:0", getitem_3: "f32[128, 12, 256, 64][589824, 64, 2304, 1]cuda:0", getitem_4: "f32[128, 12, 256, 64][589824, 64, 2304, 1]cuda:0", getitem_5: "f32[128, 12, 256, 64][196608, 64, 768, 1]cuda:0", getitem_6: "f32[128, 12, 256][3072, 256, 1]cuda:0", getitem_7: "i64[][]cuda:0", getitem_8: "i64[][]cuda:0", mul_2: "f32[128, 256, 768][196608, 768, 1]cuda:0", view_7: "f32[32768, 768][768, 1]cuda:0", addmm_2: "f32[32768, 3072][3072, 1]cuda:0", view_9: "f32[32768, 3072][3072, 1]cuda:0", mul_7: "f32[128, 256, 768][196608, 768, 1]cuda:0", view_11: "f32[32768, 768][768, 1]cuda:0", getitem_13: "f32[128, 12, 256, 64][589824, 64, 2304, 1]cuda:0", getitem_14: "f32[128, 12, 256, 64][589824, 64, 2304, 1]cuda:0", getitem_15: "f32[128, 12, 256, 64][589824, 64, 2304, 1]cuda:0", getitem_16: "f32[128, 12, 256, 64][196608, 64, 768, 1]cuda:0", getitem_17: "f32[128, 12, 256][3072, 256, 1]cuda:0", getitem_18: "i64[][]cuda:0", getitem_19: "i64[][]cuda:0", mul_9: "f32[128, 256, 768][196608, 768, 1]cuda:0", view_17: "f32[32768, 768][768, 1]cuda:0", addmm_6: "f32[32768, 3072][3072, 1]cuda:0", view_19: "f32[32768, 3072][3072, 1]cuda:0", mul_14: "f32[128, 256, 768][196608, 768, 1]cuda:0", view_21: "f32[32768, 768][768, 1]cuda:0", getitem_24: "f32[128, 12, 256, 64][589824, 64, 2304, 1]cuda:0", getitem_25: "f32[128, 12, 256, 64][589824, 64, 2304, 1]cuda:0", getitem_26: "f32[128, 12, 256, 64][589824, 64, 2304, 1]cuda:0", getitem_27: "f32[128, 12, 256, 64][196608, 64, 768, 1]cuda:0", getitem_28: "f32[128, 12, 256][3072, 256, 1]cuda:0", getitem_29: "i64[][]cuda:0", getitem_30: "i64[][]cuda:0", mul_16: "f32[128, 256, 768][196608, 768, 1]cuda:0", view_27: "f32[32768, 768][768, 1]cuda:0", addmm_10: "f32[32768, 3072][3072, 1]cuda:0", view_29: "f32[32768, 3072][3072, 1]cuda:0", mul_21: "f32[128, 256, 768][196608, 768, 1]cuda:0", view_31: "f32[32768, 768][768, 1]cuda:0", getitem_35: "f32[128, 12, 256, 64][589824, 64, 2304, 1]cuda:0", getitem_36: "f32[128, 12, 256, 64][589824, 64, 2304, 1]cuda:0", getitem_37: "f32[128, 12, 256, 64][589824, 64, 2304, 1]cuda:0", getitem_38: "f32[128, 12, 256, 64][196608, 64, 768, 1]cuda:0", getitem_39: "f32[128, 12, 256][3072, 256, 1]cuda:0", getitem_40: "i64[][]cuda:0", getitem_41: "i64[][]cuda:0", mul_23: "f32[128, 256, 768][196608, 768, 1]cuda:0", view_37: "f32[32768, 768][768, 1]cuda:0", addmm_14: "f32[32768, 3072][3072, 1]cuda:0", view_39: "f32[32768, 3072][3072, 1]cuda:0", mul_28: "f32[128, 256, 768][196608, 768, 1]cuda:0", view_41: "f32[32768, 768][768, 1]cuda:0", getitem_46: "f32[128, 12, 256, 64][589824, 64, 2304, 1]cuda:0", getitem_47: "f32[128, 12, 256, 64][589824, 64, 2304, 1]cuda:0", getitem_48: "f32[128, 12, 256, 64][589824, 64, 2304, 1]cuda:0", getitem_49: "f32[128, 12, 256, 64][196608, 64, 768, 1]cuda:0", getitem_50: "f32[128, 12, 256][3072, 256, 1]cuda:0", getitem_51: "i64[][]cuda:0", getitem_52: "i64[][]cuda:0", mul_30: "f32[128, 256, 768][196608, 768, 1]cuda:0", view_47: "f32[32768, 768][768, 1]cuda:0", addmm_18: "f32[32768, 3072][3072, 1]cuda:0", view_49: "f32[32768, 3072][3072, 1]cuda:0", mul_35: "f32[128, 256, 768][196608, 768, 1]cuda:0", view_51: "f32[32768, 768][768, 1]cuda:0", getitem_57: "f32[128, 12, 256, 64][589824, 64, 2304, 1]cuda:0", getitem_58: "f32[128, 12, 256, 64][589824, 64, 2304, 1]cuda:0", getitem_59: "f32[128, 12, 256, 64][589824, 64, 2304, 1]cuda:0", getitem_60: "f32[128, 12, 256, 64][196608, 64, 768, 1]cuda:0", getitem_61: "f32[128, 12, 256][3072, 256, 1]cuda:0", getitem_62: "i64[][]cuda:0", getitem_63: "i64[][]cuda:0", mul_37: "f32[128, 256, 768][196608, 768, 1]cuda:0", view_57: "f32[32768, 768][768, 1]cuda:0", addmm_22: "f32[32768, 3072][3072, 1]cuda:0", view_59: "f32[32768, 3072][3072, 1]cuda:0", mul_42: "f32[128, 256, 768][196608, 768, 1]cuda:0", view_61: "f32[32768, 768][768, 1]cuda:0", getitem_68: "f32[128, 12, 256, 64][589824, 64, 2304, 1]cuda:0", getitem_69: "f32[128, 12, 256, 64][589824, 64, 2304, 1]cuda:0", getitem_70: "f32[128, 12, 256, 64][589824, 64, 2304, 1]cuda:0", getitem_71: "f32[128, 12, 256, 64][196608, 64, 768, 1]cuda:0", getitem_72: "f32[128, 12, 256][3072, 256, 1]cuda:0", getitem_73: "i64[][]cuda:0", getitem_74: "i64[][]cuda:0", mul_44: "f32[128, 256, 768][196608, 768, 1]cuda:0", view_67: "f32[32768, 768][768, 1]cuda:0", addmm_26: "f32[32768, 3072][3072, 1]cuda:0", view_69: "f32[32768, 3072][3072, 1]cuda:0", mul_49: "f32[128, 256, 768][196608, 768, 1]cuda:0", view_71: "f32[32768, 768][768, 1]cuda:0", getitem_79: "f32[128, 12, 256, 64][589824, 64, 2304, 1]cuda:0", getitem_80: "f32[128, 12, 256, 64][589824, 64, 2304, 1]cuda:0", getitem_81: "f32[128, 12, 256, 64][589824, 64, 2304, 1]cuda:0", getitem_82: "f32[128, 12, 256, 64][196608, 64, 768, 1]cuda:0", getitem_83: "f32[128, 12, 256][3072, 256, 1]cuda:0", getitem_84: "i64[][]cuda:0", getitem_85: "i64[][]cuda:0", mul_51: "f32[128, 256, 768][196608, 768, 1]cuda:0", view_77: "f32[32768, 768][768, 1]cuda:0", addmm_30: "f32[32768, 3072][3072, 1]cuda:0", view_79: "f32[32768, 3072][3072, 1]cuda:0", mul_56: "f32[128, 256, 768][196608, 768, 1]cuda:0", view_81: "f32[32768, 768][768, 1]cuda:0", getitem_90: "f32[128, 12, 256, 64][589824, 64, 2304, 1]cuda:0", getitem_91: "f32[128, 12, 256, 64][589824, 64, 2304, 1]cuda:0", getitem_92: "f32[128, 12, 256, 64][589824, 64, 2304, 1]cuda:0", getitem_93: "f32[128, 12, 256, 64][196608, 64, 768, 1]cuda:0", getitem_94: "f32[128, 12, 256][3072, 256, 1]cuda:0", getitem_95: "i64[][]cuda:0", getitem_96: "i64[][]cuda:0", mul_58: "f32[128, 256, 768][196608, 768, 1]cuda:0", view_87: "f32[32768, 768][768, 1]cuda:0", addmm_34: "f32[32768, 3072][3072, 1]cuda:0", view_89: "f32[32768, 3072][3072, 1]cuda:0", mul_63: "f32[128, 256, 768][196608, 768, 1]cuda:0", view_91: "f32[32768, 768][768, 1]cuda:0", getitem_101: "f32[128, 12, 256, 64][589824, 64, 2304, 1]cuda:0", getitem_102: "f32[128, 12, 256, 64][589824, 64, 2304, 1]cuda:0", getitem_103: "f32[128, 12, 256, 64][589824, 64, 2304, 1]cuda:0", getitem_104: "f32[128, 12, 256, 64][196608, 64, 768, 1]cuda:0", getitem_105: "f32[128, 12, 256][3072, 256, 1]cuda:0", getitem_106: "i64[][]cuda:0", getitem_107: "i64[][]cuda:0", mul_65: "f32[128, 256, 768][196608, 768, 1]cuda:0", view_97: "f32[32768, 768][768, 1]cuda:0", addmm_38: "f32[32768, 3072][3072, 1]cuda:0", view_99: "f32[32768, 3072][3072, 1]cuda:0", mul_70: "f32[128, 256, 768][196608, 768, 1]cuda:0", view_101: "f32[32768, 768][768, 1]cuda:0", getitem_112: "f32[128, 12, 256, 64][589824, 64, 2304, 1]cuda:0", getitem_113: "f32[128, 12, 256, 64][589824, 64, 2304, 1]cuda:0", getitem_114: "f32[128, 12, 256, 64][589824, 64, 2304, 1]cuda:0", getitem_115: "f32[128, 12, 256, 64][196608, 64, 768, 1]cuda:0", getitem_116: "f32[128, 12, 256][3072, 256, 1]cuda:0", getitem_117: "i64[][]cuda:0", getitem_118: "i64[][]cuda:0", mul_72: "f32[128, 256, 768][196608, 768, 1]cuda:0", view_107: "f32[32768, 768][768, 1]cuda:0", addmm_42: "f32[32768, 3072][3072, 1]cuda:0", view_109: "f32[32768, 3072][3072, 1]cuda:0", mul_77: "f32[128, 256, 768][196608, 768, 1]cuda:0", view_111: "f32[32768, 768][768, 1]cuda:0", getitem_123: "f32[128, 12, 256, 64][589824, 64, 2304, 1]cuda:0", getitem_124: "f32[128, 12, 256, 64][589824, 64, 2304, 1]cuda:0", getitem_125: "f32[128, 12, 256, 64][589824, 64, 2304, 1]cuda:0", getitem_126: "f32[128, 12, 256, 64][196608, 64, 768, 1]cuda:0", getitem_127: "f32[128, 12, 256][3072, 256, 1]cuda:0", getitem_128: "i64[][]cuda:0", getitem_129: "i64[][]cuda:0", mul_79: "f32[128, 256, 768][196608, 768, 1]cuda:0", view_117: "f32[32768, 768][768, 1]cuda:0", addmm_46: "f32[32768, 3072][3072, 1]cuda:0", view_119: "f32[32768, 3072][3072, 1]cuda:0", mul_84: "f32[128, 256, 768][196608, 768, 1]cuda:0", permute_74: "f32[128, 12, 1, 64][768, 64, 768, 1]cuda:0", view_124: "f32[32768, 768][768, 1]cuda:0", getitem_134: "f32[128, 12, 256, 64][393216, 64, 1536, 1]cuda:0", getitem_135: "f32[128, 12, 256, 64][393216, 64, 1536, 1]cuda:0", getitem_136: "f32[128, 12, 1, 64][768, 64, 768, 1]cuda:0", getitem_137: "f32[128, 12, 32][384, 32, 1]cuda:0", getitem_138: "i64[][]cuda:0", getitem_139: "i64[][]cuda:0", addmm_49: "f32[128, 768][768, 1]cuda:0", getitem_141: "f32[128, 1, 1][1, 1, 1]cuda:0", rsqrt_25: "f32[128, 1, 1][1, 1, 1]cuda:0", view_130: "f32[128, 768][768, 1]cuda:0", addmm_50: "f32[128, 3072][3072, 1]cuda:0", view_132: "f32[128, 3072][3072, 1]cuda:0", div_1: "f32[128, 256, 1][256, 1, 1]cuda:0", div_2: "f32[128, 256, 1][256, 1, 1]cuda:0", div_3: "f32[128, 256, 1][256, 1, 1]cuda:0", div_4: "f32[128, 256, 1][256, 1, 1]cuda:0", div_5: "f32[128, 256, 1][256, 1, 1]cuda:0", div_6: "f32[128, 256, 1][256, 1, 1]cuda:0", div_7: "f32[128, 256, 1][256, 1, 1]cuda:0", div_8: "f32[128, 256, 1][256, 1, 1]cuda:0", div_9: "f32[128, 256, 1][256, 1, 1]cuda:0", div_10: "f32[128, 256, 1][256, 1, 1]cuda:0", div_11: "f32[128, 256, 1][256, 1, 1]cuda:0", div_12: "f32[128, 256, 1][256, 1, 1]cuda:0", div_13: "f32[128, 256, 1][256, 1, 1]cuda:0", div_14: "f32[128, 256, 1][256, 1, 1]cuda:0", div_15: "f32[128, 256, 1][256, 1, 1]cuda:0", div_16: "f32[128, 256, 1][256, 1, 1]cuda:0", div_17: "f32[128, 256, 1][256, 1, 1]cuda:0", div_18: "f32[128, 256, 1][256, 1, 1]cuda:0", div_19: "f32[128, 256, 1][256, 1, 1]cuda:0", div_20: "f32[128, 256, 1][256, 1, 1]cuda:0", div_21: "f32[128, 256, 1][256, 1, 1]cuda:0", div_22: "f32[128, 256, 1][256, 1, 1]cuda:0", div_23: "f32[128, 256, 1][256, 1, 1]cuda:0", div_24: "f32[128, 256, 1][256, 1, 1]cuda:0", tangents_1: "f32[128, 768][768, 1]cuda:0"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention_pool.py:126 in forward, code: x = x[:, 0]
        full_default: "f32[128, 1, 768][768, 768, 1]cuda:0" = torch.ops.aten.full.default([128, 1, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        select_scatter: "f32[128, 1, 768][768, 768, 1]cuda:0" = torch.ops.aten.select_scatter.default(full_default, tangents_1, 1, 0);  full_default = tangents_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_134: "f32[128, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(select_scatter, [128, 768])
        permute_80: "f32[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(primals_162, [1, 0]);  primals_162 = None
        permute_81: "f32[768, 3072][3072, 1]cuda:0" = torch.ops.aten.permute.default(permute_80, [1, 0]);  permute_80 = None
        mm_1: "f32[128, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(view_134, permute_81);  permute_81 = None
        permute_82: "f32[768, 128][1, 768]cuda:0" = torch.ops.aten.permute.default(view_134, [1, 0])
        mm_2: "f32[768, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(permute_82, view_132);  permute_82 = view_132 = None
        sum_1: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_134, [0], True);  view_134 = None
        view_135: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_1, [768]);  sum_1 = None
        view_136: "f32[128, 1, 3072][3072, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(mm_1, [128, 1, 3072]);  mm_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_131: "f32[128, 1, 3072][3072, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_50, [128, 1, 3072]);  addmm_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        mul_89: "f32[128, 1, 3072][3072, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_131, 0.7071067811865476)
        erf_12: "f32[128, 1, 3072][3072, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_89);  mul_89 = None
        add_90: "f32[128, 1, 3072][3072, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
        mul_92: "f32[128, 1, 3072][3072, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_90, 0.5);  add_90 = None
        mul_93: "f32[128, 1, 3072][3072, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_131, view_131)
        mul_94: "f32[128, 1, 3072][3072, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_93, -0.5);  mul_93 = None
        exp: "f32[128, 1, 3072][3072, 3072, 1]cuda:0" = torch.ops.aten.exp.default(mul_94);  mul_94 = None
        mul_95: "f32[128, 1, 3072][3072, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp, 0.3989422804014327);  exp = None
        mul_96: "f32[128, 1, 3072][3072, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_131, mul_95);  view_131 = mul_95 = None
        add_93: "f32[128, 1, 3072][3072, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_92, mul_96);  mul_92 = mul_96 = None
        mul_97: "f32[128, 1, 3072][3072, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_136, add_93);  view_136 = add_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_137: "f32[128, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(mul_97, [128, 3072]);  mul_97 = None
        permute_79: "f32[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(primals_160, [1, 0]);  primals_160 = None
        permute_85: "f32[3072, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_79, [1, 0]);  permute_79 = None
        mm_3: "f32[128, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_137, permute_85);  permute_85 = None
        permute_86: "f32[3072, 128][1, 3072]cuda:0" = torch.ops.aten.permute.default(view_137, [1, 0])
        mm_4: "f32[3072, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_86, view_130);  permute_86 = view_130 = None
        sum_2: "f32[1, 3072][3072, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_137, [0], True);  view_137 = None
        view_138: "f32[3072][1]cuda:0" = torch.ops.aten.reshape.default(sum_2, [3072]);  sum_2 = None
        view_139: "f32[128, 1, 768][768, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_3, [128, 1, 768]);  mm_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_99: "f32[128, 1, 768][768, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_139, primals_158);  primals_158 = None
        mul_100: "f32[128, 1, 768][768, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_99, 768)
        sum_3: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_99, [2], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention_pool.py:118 in forward, code: x = self.proj(x)
        view_129: "f32[128, 1, 768][768, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_49, [128, 1, 768]);  addmm_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        sub_25: "f32[128, 1, 768][768, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_129, getitem_141);  view_129 = getitem_141 = None
        mul_86: "f32[128, 1, 768][768, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_25, rsqrt_25);  sub_25 = None
        mul_101: "f32[128, 1, 768][768, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_99, mul_86);  mul_99 = None
        sum_4: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_101, [2], True);  mul_101 = None
        mul_102: "f32[128, 1, 768][768, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_86, sum_4);  sum_4 = None
        sub_27: "f32[128, 1, 768][768, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_100, sum_3);  mul_100 = sum_3 = None
        sub_28: "f32[128, 1, 768][768, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_27, mul_102);  sub_27 = mul_102 = None
        div: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_25, 768);  rsqrt_25 = None
        mul_103: "f32[128, 1, 768][768, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div, sub_28);  div = sub_28 = None
        mul_104: "f32[128, 1, 768][768, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_139, mul_86);  mul_86 = None
        sum_5: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_104, [0, 1]);  mul_104 = None
        sum_6: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_139, [0, 1]);  view_139 = None
        add_94: "f32[128, 1, 768][768, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(select_scatter, mul_103);  select_scatter = mul_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention_pool.py:118 in forward, code: x = self.proj(x)
        view_140: "f32[128, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(add_94, [128, 768]);  add_94 = None
        permute_78: "f32[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(primals_156, [1, 0]);  primals_156 = None
        permute_89: "f32[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_78, [1, 0]);  permute_78 = None
        mm_5: "f32[128, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_140, permute_89);  permute_89 = None
        permute_90: "f32[768, 128][1, 768]cuda:0" = torch.ops.aten.permute.default(view_140, [1, 0])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention_pool.py:117 in forward, code: x = x.transpose(1, 2).reshape(B, self.latent_len, C)
        permute_77: "f32[128, 1, 12, 64][768, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_136, [0, 2, 1, 3])
        view_127: "f32[128, 1, 768][768, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_77, [128, 1, 768]);  permute_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention_pool.py:118 in forward, code: x = self.proj(x)
        view_128: "f32[128, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_127, [128, 768]);  view_127 = None
        mm_6: "f32[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_90, view_128);  permute_90 = view_128 = None
        sum_7: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_140, [0], True);  view_140 = None
        view_141: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_7, [768]);  sum_7 = None
        view_142: "f32[128, 1, 768][768, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_5, [128, 1, 768]);  mm_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention_pool.py:117 in forward, code: x = x.transpose(1, 2).reshape(B, self.latent_len, C)
        view_143: "f32[128, 1, 12, 64][768, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_142, [128, 1, 12, 64]);  view_142 = None
        permute_93: "f32[128, 12, 1, 64][768, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_143, [0, 2, 1, 3]);  view_143 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention_pool.py:110 in forward, code: x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        _scaled_dot_product_efficient_attention_backward = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_93, permute_74, getitem_134, getitem_135, None, getitem_136, getitem_137, getitem_138, getitem_139, 0.0, [True, True, True, False]);  permute_93 = permute_74 = getitem_134 = getitem_135 = getitem_136 = getitem_137 = getitem_138 = getitem_139 = None
        getitem_142: "f32[128, 12, 1, 64][768, 64, 768, 1]cuda:0" = _scaled_dot_product_efficient_attention_backward[0]
        getitem_143: "f32[128, 12, 256, 64][196608, 64, 768, 1]cuda:0" = _scaled_dot_product_efficient_attention_backward[1]
        getitem_144: "f32[128, 12, 256, 64][196608, 64, 768, 1]cuda:0" = _scaled_dot_product_efficient_attention_backward[2];  _scaled_dot_product_efficient_attention_backward = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention_pool.py:105 in forward, code: k, v = kv.unbind(0)
        cat: "f32[256, 12, 256, 64][196608, 16384, 64, 1]cuda:0" = torch.ops.aten.cat.default([getitem_143, getitem_144]);  getitem_143 = getitem_144 = None
        view_144: "f32[2, 128, 12, 256, 64][25165824, 196608, 16384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(cat, [2, 128, 12, 256, 64]);  cat = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention_pool.py:104 in forward, code: kv = self.kv(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        permute_94: "f32[128, 256, 2, 12, 64][196608, 64, 25165824, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_144, [1, 3, 0, 2, 4]);  view_144 = None
        clone_41: "f32[128, 256, 2, 12, 64][393216, 1536, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_94, memory_format = torch.contiguous_format);  permute_94 = None
        view_145: "f32[128, 256, 1536][393216, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(clone_41, [128, 256, 1536]);  clone_41 = None
        view_146: "f32[32768, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(view_145, [32768, 1536]);  view_145 = None
        permute_75: "f32[768, 1536][1, 768]cuda:0" = torch.ops.aten.permute.default(primals_154, [1, 0]);  primals_154 = None
        permute_95: "f32[1536, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_75, [1, 0]);  permute_75 = None
        mm_7: "f32[32768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_146, permute_95);  permute_95 = None
        permute_96: "f32[1536, 32768][1, 1536]cuda:0" = torch.ops.aten.permute.default(view_146, [1, 0])
        mm_8: "f32[1536, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_96, view_124);  permute_96 = view_124 = None
        sum_8: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_146, [0], True);  view_146 = None
        view_147: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(sum_8, [1536]);  sum_8 = None
        view_148: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_7, [128, 256, 768]);  mm_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention_pool.py:102 in forward, code: q = self.q(q_latent).reshape(B, self.latent_len, self.num_heads, self.head_dim).transpose(1, 2)
        permute_99: "f32[128, 1, 12, 64][768, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_142, [0, 2, 1, 3]);  getitem_142 = None
        view_149: "f32[128, 1, 768][768, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_99, [128, 1, 768]);  permute_99 = None
        sum_9: "f32[1, 1, 768][768, 768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_149, [0, 1], True)
        view_150: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_9, [768]);  sum_9 = None
        view_151: "f32[128, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_149, [128, 768]);  view_149 = None
        permute_100: "f32[768, 128][1, 768]cuda:0" = torch.ops.aten.permute.default(view_151, [1, 0])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention_pool.py:101 in forward, code: q_latent = self.latent.expand(B, -1, -1)
        expand: "f32[128, 1, 768][0, 768, 1]cuda:0" = torch.ops.aten.expand.default(primals_151, [128, -1, -1]);  primals_151 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention_pool.py:102 in forward, code: q = self.q(q_latent).reshape(B, self.latent_len, self.num_heads, self.head_dim).transpose(1, 2)
        view_121: "f32[128, 768][0, 1]cuda:0" = torch.ops.aten.reshape.default(expand, [128, 768]);  expand = None
        mm_9: "f32[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_100, view_121);  permute_100 = view_121 = None
        permute_73: "f32[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(primals_152, [1, 0]);  primals_152 = None
        permute_102: "f32[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_73, [1, 0]);  permute_73 = None
        mm_10: "f32[128, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_151, permute_102);  view_151 = permute_102 = None
        view_152: "f32[128, 1, 768][768, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_10, [128, 1, 768]);  mm_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention_pool.py:101 in forward, code: q_latent = self.latent.expand(B, -1, -1)
        sum_10: "f32[1, 1, 768][768, 768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_152, [0], True);  view_152 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_106: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_148, primals_149);  primals_149 = None
        mul_107: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_106, 768)
        sum_11: "f32[128, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_106, [2], True)
        mul_108: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_106, mul_84);  mul_106 = None
        sum_12: "f32[128, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_108, [2], True);  mul_108 = None
        mul_109: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_84, sum_12);  sum_12 = None
        sub_30: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_107, sum_11);  mul_107 = sum_11 = None
        sub_31: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_30, mul_109);  sub_30 = mul_109 = None
        mul_110: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_1, sub_31);  div_1 = sub_31 = None
        mul_111: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_148, mul_84);  mul_84 = None
        sum_13: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_111, [0, 1]);  mul_111 = None
        sum_14: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_148, [0, 1]);  view_148 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_153: "f32[32768, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(mul_110, [32768, 768])
        permute_72: "f32[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(primals_147, [1, 0]);  primals_147 = None
        permute_104: "f32[768, 3072][3072, 1]cuda:0" = torch.ops.aten.permute.default(permute_72, [1, 0]);  permute_72 = None
        mm_11: "f32[32768, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(view_153, permute_104);  permute_104 = None
        permute_105: "f32[768, 32768][1, 768]cuda:0" = torch.ops.aten.permute.default(view_153, [1, 0])
        mm_12: "f32[768, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(permute_105, view_119);  permute_105 = view_119 = None
        sum_15: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_153, [0], True);  view_153 = None
        view_154: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_15, [768]);  sum_15 = None
        view_155: "f32[128, 256, 3072][786432, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(mm_11, [128, 256, 3072]);  mm_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_118: "f32[128, 256, 3072][786432, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_46, [128, 256, 3072]);  addmm_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        mul_82: "f32[128, 256, 3072][786432, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_118, 0.7071067811865476)
        erf_11: "f32[128, 256, 3072][786432, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_82);  mul_82 = None
        add_83: "f32[128, 256, 3072][786432, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
        mul_113: "f32[128, 256, 3072][786432, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_83, 0.5);  add_83 = None
        mul_114: "f32[128, 256, 3072][786432, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_118, view_118)
        mul_115: "f32[128, 256, 3072][786432, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_114, -0.5);  mul_114 = None
        exp_1: "f32[128, 256, 3072][786432, 3072, 1]cuda:0" = torch.ops.aten.exp.default(mul_115);  mul_115 = None
        mul_116: "f32[128, 256, 3072][786432, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_1, 0.3989422804014327);  exp_1 = None
        mul_117: "f32[128, 256, 3072][786432, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_118, mul_116);  view_118 = mul_116 = None
        add_96: "f32[128, 256, 3072][786432, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_113, mul_117);  mul_113 = mul_117 = None
        mul_118: "f32[128, 256, 3072][786432, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_155, add_96);  view_155 = add_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_156: "f32[32768, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(mul_118, [32768, 3072]);  mul_118 = None
        permute_71: "f32[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(primals_145, [1, 0]);  primals_145 = None
        permute_108: "f32[3072, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_71, [1, 0]);  permute_71 = None
        mm_13: "f32[32768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_156, permute_108);  permute_108 = None
        permute_109: "f32[3072, 32768][1, 3072]cuda:0" = torch.ops.aten.permute.default(view_156, [1, 0])
        mm_14: "f32[3072, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_109, view_117);  permute_109 = view_117 = None
        sum_16: "f32[1, 3072][3072, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_156, [0], True);  view_156 = None
        view_157: "f32[3072][1]cuda:0" = torch.ops.aten.reshape.default(sum_16, [3072]);  sum_16 = None
        view_158: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_13, [128, 256, 768]);  mm_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_120: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_158, primals_143);  primals_143 = None
        mul_121: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_120, 768)
        sum_17: "f32[128, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_120, [2], True)
        mul_122: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_120, mul_79);  mul_120 = None
        sum_18: "f32[128, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_122, [2], True);  mul_122 = None
        mul_123: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_79, sum_18);  sum_18 = None
        sub_33: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_121, sum_17);  mul_121 = sum_17 = None
        sub_34: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_33, mul_123);  sub_33 = mul_123 = None
        mul_124: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_2, sub_34);  div_2 = sub_34 = None
        mul_125: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_158, mul_79);  mul_79 = None
        sum_19: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_125, [0, 1]);  mul_125 = None
        sum_20: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_158, [0, 1]);  view_158 = None
        add_97: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_110, mul_124);  mul_110 = mul_124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_159: "f32[32768, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(add_97, [32768, 768])
        permute_70: "f32[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(primals_141, [1, 0]);  primals_141 = None
        permute_112: "f32[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_70, [1, 0]);  permute_70 = None
        mm_15: "f32[32768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_159, permute_112);  permute_112 = None
        permute_113: "f32[768, 32768][1, 768]cuda:0" = torch.ops.aten.permute.default(view_159, [1, 0])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_69: "f32[128, 256, 12, 64][196608, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_126, [0, 2, 1, 3])
        view_114: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_69, [128, 256, 768]);  permute_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_115: "f32[32768, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_114, [32768, 768]);  view_114 = None
        mm_16: "f32[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_113, view_115);  permute_113 = view_115 = None
        sum_21: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_159, [0], True);  view_159 = None
        view_160: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_21, [768]);  sum_21 = None
        view_161: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_15, [128, 256, 768]);  mm_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        view_162: "f32[128, 256, 12, 64][196608, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_161, [128, 256, 12, 64]);  view_161 = None
        permute_116: "f32[128, 12, 256, 64][196608, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_162, [0, 2, 1, 3]);  view_162 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:120 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_backward_1 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_116, getitem_123, getitem_124, getitem_125, None, getitem_126, getitem_127, getitem_128, getitem_129, 0.0, [True, True, True, False]);  permute_116 = getitem_123 = getitem_124 = getitem_125 = getitem_126 = getitem_127 = getitem_128 = getitem_129 = None
        getitem_146: "f32[128, 12, 256, 64][196608, 64, 768, 1]cuda:0" = _scaled_dot_product_efficient_attention_backward_1[0]
        getitem_147: "f32[128, 12, 256, 64][196608, 64, 768, 1]cuda:0" = _scaled_dot_product_efficient_attention_backward_1[1]
        getitem_148: "f32[128, 12, 256, 64][196608, 64, 768, 1]cuda:0" = _scaled_dot_product_efficient_attention_backward_1[2];  _scaled_dot_product_efficient_attention_backward_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        cat_1: "f32[384, 12, 256, 64][196608, 16384, 64, 1]cuda:0" = torch.ops.aten.cat.default([getitem_146, getitem_147, getitem_148]);  getitem_146 = getitem_147 = getitem_148 = None
        view_163: "f32[3, 128, 12, 256, 64][25165824, 196608, 16384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(cat_1, [3, 128, 12, 256, 64]);  cat_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        permute_117: "f32[128, 256, 3, 12, 64][196608, 64, 25165824, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_163, [1, 3, 0, 2, 4]);  view_163 = None
        clone_42: "f32[128, 256, 3, 12, 64][589824, 2304, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_117, memory_format = torch.contiguous_format);  permute_117 = None
        view_164: "f32[128, 256, 2304][589824, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(clone_42, [128, 256, 2304]);  clone_42 = None
        view_165: "f32[32768, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(view_164, [32768, 2304]);  view_164 = None
        permute_67: "f32[768, 2304][1, 768]cuda:0" = torch.ops.aten.permute.default(primals_139, [1, 0]);  primals_139 = None
        permute_118: "f32[2304, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_67, [1, 0]);  permute_67 = None
        mm_17: "f32[32768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_165, permute_118);  permute_118 = None
        permute_119: "f32[2304, 32768][1, 2304]cuda:0" = torch.ops.aten.permute.default(view_165, [1, 0])
        mm_18: "f32[2304, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_119, view_111);  permute_119 = view_111 = None
        sum_22: "f32[1, 2304][2304, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_165, [0], True);  view_165 = None
        view_166: "f32[2304][1]cuda:0" = torch.ops.aten.reshape.default(sum_22, [2304]);  sum_22 = None
        view_167: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_17, [128, 256, 768]);  mm_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_127: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_167, primals_137);  primals_137 = None
        mul_128: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_127, 768)
        sum_23: "f32[128, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_127, [2], True)
        mul_129: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_127, mul_77);  mul_127 = None
        sum_24: "f32[128, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_129, [2], True);  mul_129 = None
        mul_130: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_77, sum_24);  sum_24 = None
        sub_36: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_128, sum_23);  mul_128 = sum_23 = None
        sub_37: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_36, mul_130);  sub_36 = mul_130 = None
        mul_131: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_3, sub_37);  div_3 = sub_37 = None
        mul_132: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_167, mul_77);  mul_77 = None
        sum_25: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_132, [0, 1]);  mul_132 = None
        sum_26: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_167, [0, 1]);  view_167 = None
        add_98: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_97, mul_131);  add_97 = mul_131 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_168: "f32[32768, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(add_98, [32768, 768])
        permute_66: "f32[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(primals_135, [1, 0]);  primals_135 = None
        permute_122: "f32[768, 3072][3072, 1]cuda:0" = torch.ops.aten.permute.default(permute_66, [1, 0]);  permute_66 = None
        mm_19: "f32[32768, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(view_168, permute_122);  permute_122 = None
        permute_123: "f32[768, 32768][1, 768]cuda:0" = torch.ops.aten.permute.default(view_168, [1, 0])
        mm_20: "f32[768, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(permute_123, view_109);  permute_123 = view_109 = None
        sum_27: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_168, [0], True);  view_168 = None
        view_169: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_27, [768]);  sum_27 = None
        view_170: "f32[128, 256, 3072][786432, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(mm_19, [128, 256, 3072]);  mm_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_108: "f32[128, 256, 3072][786432, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_42, [128, 256, 3072]);  addmm_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        mul_75: "f32[128, 256, 3072][786432, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_108, 0.7071067811865476)
        erf_10: "f32[128, 256, 3072][786432, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_75);  mul_75 = None
        add_76: "f32[128, 256, 3072][786432, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
        mul_134: "f32[128, 256, 3072][786432, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_76, 0.5);  add_76 = None
        mul_135: "f32[128, 256, 3072][786432, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_108, view_108)
        mul_136: "f32[128, 256, 3072][786432, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_135, -0.5);  mul_135 = None
        exp_2: "f32[128, 256, 3072][786432, 3072, 1]cuda:0" = torch.ops.aten.exp.default(mul_136);  mul_136 = None
        mul_137: "f32[128, 256, 3072][786432, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_2, 0.3989422804014327);  exp_2 = None
        mul_138: "f32[128, 256, 3072][786432, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_108, mul_137);  view_108 = mul_137 = None
        add_100: "f32[128, 256, 3072][786432, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_134, mul_138);  mul_134 = mul_138 = None
        mul_139: "f32[128, 256, 3072][786432, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_170, add_100);  view_170 = add_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_171: "f32[32768, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(mul_139, [32768, 3072]);  mul_139 = None
        permute_65: "f32[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(primals_133, [1, 0]);  primals_133 = None
        permute_126: "f32[3072, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_65, [1, 0]);  permute_65 = None
        mm_21: "f32[32768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_171, permute_126);  permute_126 = None
        permute_127: "f32[3072, 32768][1, 3072]cuda:0" = torch.ops.aten.permute.default(view_171, [1, 0])
        mm_22: "f32[3072, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_127, view_107);  permute_127 = view_107 = None
        sum_28: "f32[1, 3072][3072, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_171, [0], True);  view_171 = None
        view_172: "f32[3072][1]cuda:0" = torch.ops.aten.reshape.default(sum_28, [3072]);  sum_28 = None
        view_173: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_21, [128, 256, 768]);  mm_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_141: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_173, primals_131);  primals_131 = None
        mul_142: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_141, 768)
        sum_29: "f32[128, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_141, [2], True)
        mul_143: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_141, mul_72);  mul_141 = None
        sum_30: "f32[128, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_143, [2], True);  mul_143 = None
        mul_144: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_72, sum_30);  sum_30 = None
        sub_39: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_142, sum_29);  mul_142 = sum_29 = None
        sub_40: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_39, mul_144);  sub_39 = mul_144 = None
        mul_145: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_4, sub_40);  div_4 = sub_40 = None
        mul_146: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_173, mul_72);  mul_72 = None
        sum_31: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_146, [0, 1]);  mul_146 = None
        sum_32: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_173, [0, 1]);  view_173 = None
        add_101: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_98, mul_145);  add_98 = mul_145 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_174: "f32[32768, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(add_101, [32768, 768])
        permute_64: "f32[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(primals_129, [1, 0]);  primals_129 = None
        permute_130: "f32[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_64, [1, 0]);  permute_64 = None
        mm_23: "f32[32768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_174, permute_130);  permute_130 = None
        permute_131: "f32[768, 32768][1, 768]cuda:0" = torch.ops.aten.permute.default(view_174, [1, 0])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_63: "f32[128, 256, 12, 64][196608, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_115, [0, 2, 1, 3])
        view_104: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_63, [128, 256, 768]);  permute_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_105: "f32[32768, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_104, [32768, 768]);  view_104 = None
        mm_24: "f32[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_131, view_105);  permute_131 = view_105 = None
        sum_33: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_174, [0], True);  view_174 = None
        view_175: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_33, [768]);  sum_33 = None
        view_176: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_23, [128, 256, 768]);  mm_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        view_177: "f32[128, 256, 12, 64][196608, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_176, [128, 256, 12, 64]);  view_176 = None
        permute_134: "f32[128, 12, 256, 64][196608, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_177, [0, 2, 1, 3]);  view_177 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:120 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_backward_2 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_134, getitem_112, getitem_113, getitem_114, None, getitem_115, getitem_116, getitem_117, getitem_118, 0.0, [True, True, True, False]);  permute_134 = getitem_112 = getitem_113 = getitem_114 = getitem_115 = getitem_116 = getitem_117 = getitem_118 = None
        getitem_150: "f32[128, 12, 256, 64][196608, 64, 768, 1]cuda:0" = _scaled_dot_product_efficient_attention_backward_2[0]
        getitem_151: "f32[128, 12, 256, 64][196608, 64, 768, 1]cuda:0" = _scaled_dot_product_efficient_attention_backward_2[1]
        getitem_152: "f32[128, 12, 256, 64][196608, 64, 768, 1]cuda:0" = _scaled_dot_product_efficient_attention_backward_2[2];  _scaled_dot_product_efficient_attention_backward_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        cat_2: "f32[384, 12, 256, 64][196608, 16384, 64, 1]cuda:0" = torch.ops.aten.cat.default([getitem_150, getitem_151, getitem_152]);  getitem_150 = getitem_151 = getitem_152 = None
        view_178: "f32[3, 128, 12, 256, 64][25165824, 196608, 16384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(cat_2, [3, 128, 12, 256, 64]);  cat_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        permute_135: "f32[128, 256, 3, 12, 64][196608, 64, 25165824, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_178, [1, 3, 0, 2, 4]);  view_178 = None
        clone_43: "f32[128, 256, 3, 12, 64][589824, 2304, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_135, memory_format = torch.contiguous_format);  permute_135 = None
        view_179: "f32[128, 256, 2304][589824, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(clone_43, [128, 256, 2304]);  clone_43 = None
        view_180: "f32[32768, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(view_179, [32768, 2304]);  view_179 = None
        permute_61: "f32[768, 2304][1, 768]cuda:0" = torch.ops.aten.permute.default(primals_127, [1, 0]);  primals_127 = None
        permute_136: "f32[2304, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_61, [1, 0]);  permute_61 = None
        mm_25: "f32[32768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_180, permute_136);  permute_136 = None
        permute_137: "f32[2304, 32768][1, 2304]cuda:0" = torch.ops.aten.permute.default(view_180, [1, 0])
        mm_26: "f32[2304, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_137, view_101);  permute_137 = view_101 = None
        sum_34: "f32[1, 2304][2304, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_180, [0], True);  view_180 = None
        view_181: "f32[2304][1]cuda:0" = torch.ops.aten.reshape.default(sum_34, [2304]);  sum_34 = None
        view_182: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_25, [128, 256, 768]);  mm_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_148: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_182, primals_125);  primals_125 = None
        mul_149: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_148, 768)
        sum_35: "f32[128, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_148, [2], True)
        mul_150: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_148, mul_70);  mul_148 = None
        sum_36: "f32[128, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_150, [2], True);  mul_150 = None
        mul_151: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_70, sum_36);  sum_36 = None
        sub_42: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_149, sum_35);  mul_149 = sum_35 = None
        sub_43: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_42, mul_151);  sub_42 = mul_151 = None
        mul_152: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_5, sub_43);  div_5 = sub_43 = None
        mul_153: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_182, mul_70);  mul_70 = None
        sum_37: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_153, [0, 1]);  mul_153 = None
        sum_38: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_182, [0, 1]);  view_182 = None
        add_102: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_101, mul_152);  add_101 = mul_152 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_183: "f32[32768, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(add_102, [32768, 768])
        permute_60: "f32[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(primals_123, [1, 0]);  primals_123 = None
        permute_140: "f32[768, 3072][3072, 1]cuda:0" = torch.ops.aten.permute.default(permute_60, [1, 0]);  permute_60 = None
        mm_27: "f32[32768, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(view_183, permute_140);  permute_140 = None
        permute_141: "f32[768, 32768][1, 768]cuda:0" = torch.ops.aten.permute.default(view_183, [1, 0])
        mm_28: "f32[768, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(permute_141, view_99);  permute_141 = view_99 = None
        sum_39: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_183, [0], True);  view_183 = None
        view_184: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_39, [768]);  sum_39 = None
        view_185: "f32[128, 256, 3072][786432, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(mm_27, [128, 256, 3072]);  mm_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_98: "f32[128, 256, 3072][786432, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_38, [128, 256, 3072]);  addmm_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        mul_68: "f32[128, 256, 3072][786432, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_98, 0.7071067811865476)
        erf_9: "f32[128, 256, 3072][786432, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_68);  mul_68 = None
        add_69: "f32[128, 256, 3072][786432, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
        mul_155: "f32[128, 256, 3072][786432, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_69, 0.5);  add_69 = None
        mul_156: "f32[128, 256, 3072][786432, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_98, view_98)
        mul_157: "f32[128, 256, 3072][786432, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_156, -0.5);  mul_156 = None
        exp_3: "f32[128, 256, 3072][786432, 3072, 1]cuda:0" = torch.ops.aten.exp.default(mul_157);  mul_157 = None
        mul_158: "f32[128, 256, 3072][786432, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_3, 0.3989422804014327);  exp_3 = None
        mul_159: "f32[128, 256, 3072][786432, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_98, mul_158);  view_98 = mul_158 = None
        add_104: "f32[128, 256, 3072][786432, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_155, mul_159);  mul_155 = mul_159 = None
        mul_160: "f32[128, 256, 3072][786432, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_185, add_104);  view_185 = add_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_186: "f32[32768, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(mul_160, [32768, 3072]);  mul_160 = None
        permute_59: "f32[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(primals_121, [1, 0]);  primals_121 = None
        permute_144: "f32[3072, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_59, [1, 0]);  permute_59 = None
        mm_29: "f32[32768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_186, permute_144);  permute_144 = None
        permute_145: "f32[3072, 32768][1, 3072]cuda:0" = torch.ops.aten.permute.default(view_186, [1, 0])
        mm_30: "f32[3072, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_145, view_97);  permute_145 = view_97 = None
        sum_40: "f32[1, 3072][3072, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_186, [0], True);  view_186 = None
        view_187: "f32[3072][1]cuda:0" = torch.ops.aten.reshape.default(sum_40, [3072]);  sum_40 = None
        view_188: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_29, [128, 256, 768]);  mm_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_162: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_188, primals_119);  primals_119 = None
        mul_163: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_162, 768)
        sum_41: "f32[128, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_162, [2], True)
        mul_164: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_162, mul_65);  mul_162 = None
        sum_42: "f32[128, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_164, [2], True);  mul_164 = None
        mul_165: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_65, sum_42);  sum_42 = None
        sub_45: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_163, sum_41);  mul_163 = sum_41 = None
        sub_46: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_45, mul_165);  sub_45 = mul_165 = None
        mul_166: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_6, sub_46);  div_6 = sub_46 = None
        mul_167: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_188, mul_65);  mul_65 = None
        sum_43: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_167, [0, 1]);  mul_167 = None
        sum_44: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_188, [0, 1]);  view_188 = None
        add_105: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_102, mul_166);  add_102 = mul_166 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_189: "f32[32768, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(add_105, [32768, 768])
        permute_58: "f32[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(primals_117, [1, 0]);  primals_117 = None
        permute_148: "f32[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_58, [1, 0]);  permute_58 = None
        mm_31: "f32[32768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_189, permute_148);  permute_148 = None
        permute_149: "f32[768, 32768][1, 768]cuda:0" = torch.ops.aten.permute.default(view_189, [1, 0])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_57: "f32[128, 256, 12, 64][196608, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_104, [0, 2, 1, 3])
        view_94: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_57, [128, 256, 768]);  permute_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_95: "f32[32768, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_94, [32768, 768]);  view_94 = None
        mm_32: "f32[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_149, view_95);  permute_149 = view_95 = None
        sum_45: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_189, [0], True);  view_189 = None
        view_190: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_45, [768]);  sum_45 = None
        view_191: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_31, [128, 256, 768]);  mm_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        view_192: "f32[128, 256, 12, 64][196608, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_191, [128, 256, 12, 64]);  view_191 = None
        permute_152: "f32[128, 12, 256, 64][196608, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_192, [0, 2, 1, 3]);  view_192 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:120 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_backward_3 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_152, getitem_101, getitem_102, getitem_103, None, getitem_104, getitem_105, getitem_106, getitem_107, 0.0, [True, True, True, False]);  permute_152 = getitem_101 = getitem_102 = getitem_103 = getitem_104 = getitem_105 = getitem_106 = getitem_107 = None
        getitem_154: "f32[128, 12, 256, 64][196608, 64, 768, 1]cuda:0" = _scaled_dot_product_efficient_attention_backward_3[0]
        getitem_155: "f32[128, 12, 256, 64][196608, 64, 768, 1]cuda:0" = _scaled_dot_product_efficient_attention_backward_3[1]
        getitem_156: "f32[128, 12, 256, 64][196608, 64, 768, 1]cuda:0" = _scaled_dot_product_efficient_attention_backward_3[2];  _scaled_dot_product_efficient_attention_backward_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        cat_3: "f32[384, 12, 256, 64][196608, 16384, 64, 1]cuda:0" = torch.ops.aten.cat.default([getitem_154, getitem_155, getitem_156]);  getitem_154 = getitem_155 = getitem_156 = None
        view_193: "f32[3, 128, 12, 256, 64][25165824, 196608, 16384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(cat_3, [3, 128, 12, 256, 64]);  cat_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        permute_153: "f32[128, 256, 3, 12, 64][196608, 64, 25165824, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_193, [1, 3, 0, 2, 4]);  view_193 = None
        clone_44: "f32[128, 256, 3, 12, 64][589824, 2304, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_153, memory_format = torch.contiguous_format);  permute_153 = None
        view_194: "f32[128, 256, 2304][589824, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(clone_44, [128, 256, 2304]);  clone_44 = None
        view_195: "f32[32768, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(view_194, [32768, 2304]);  view_194 = None
        permute_55: "f32[768, 2304][1, 768]cuda:0" = torch.ops.aten.permute.default(primals_115, [1, 0]);  primals_115 = None
        permute_154: "f32[2304, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_55, [1, 0]);  permute_55 = None
        mm_33: "f32[32768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_195, permute_154);  permute_154 = None
        permute_155: "f32[2304, 32768][1, 2304]cuda:0" = torch.ops.aten.permute.default(view_195, [1, 0])
        mm_34: "f32[2304, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_155, view_91);  permute_155 = view_91 = None
        sum_46: "f32[1, 2304][2304, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_195, [0], True);  view_195 = None
        view_196: "f32[2304][1]cuda:0" = torch.ops.aten.reshape.default(sum_46, [2304]);  sum_46 = None
        view_197: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_33, [128, 256, 768]);  mm_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_169: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_197, primals_113);  primals_113 = None
        mul_170: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_169, 768)
        sum_47: "f32[128, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_169, [2], True)
        mul_171: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_169, mul_63);  mul_169 = None
        sum_48: "f32[128, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_171, [2], True);  mul_171 = None
        mul_172: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_63, sum_48);  sum_48 = None
        sub_48: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_170, sum_47);  mul_170 = sum_47 = None
        sub_49: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_48, mul_172);  sub_48 = mul_172 = None
        mul_173: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_7, sub_49);  div_7 = sub_49 = None
        mul_174: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_197, mul_63);  mul_63 = None
        sum_49: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_174, [0, 1]);  mul_174 = None
        sum_50: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_197, [0, 1]);  view_197 = None
        add_106: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_105, mul_173);  add_105 = mul_173 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_198: "f32[32768, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(add_106, [32768, 768])
        permute_54: "f32[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(primals_111, [1, 0]);  primals_111 = None
        permute_158: "f32[768, 3072][3072, 1]cuda:0" = torch.ops.aten.permute.default(permute_54, [1, 0]);  permute_54 = None
        mm_35: "f32[32768, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(view_198, permute_158);  permute_158 = None
        permute_159: "f32[768, 32768][1, 768]cuda:0" = torch.ops.aten.permute.default(view_198, [1, 0])
        mm_36: "f32[768, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(permute_159, view_89);  permute_159 = view_89 = None
        sum_51: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_198, [0], True);  view_198 = None
        view_199: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_51, [768]);  sum_51 = None
        view_200: "f32[128, 256, 3072][786432, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(mm_35, [128, 256, 3072]);  mm_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_88: "f32[128, 256, 3072][786432, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_34, [128, 256, 3072]);  addmm_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        mul_61: "f32[128, 256, 3072][786432, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_88, 0.7071067811865476)
        erf_8: "f32[128, 256, 3072][786432, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_61);  mul_61 = None
        add_62: "f32[128, 256, 3072][786432, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
        mul_176: "f32[128, 256, 3072][786432, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_62, 0.5);  add_62 = None
        mul_177: "f32[128, 256, 3072][786432, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_88, view_88)
        mul_178: "f32[128, 256, 3072][786432, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_177, -0.5);  mul_177 = None
        exp_4: "f32[128, 256, 3072][786432, 3072, 1]cuda:0" = torch.ops.aten.exp.default(mul_178);  mul_178 = None
        mul_179: "f32[128, 256, 3072][786432, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_4, 0.3989422804014327);  exp_4 = None
        mul_180: "f32[128, 256, 3072][786432, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_88, mul_179);  view_88 = mul_179 = None
        add_108: "f32[128, 256, 3072][786432, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_176, mul_180);  mul_176 = mul_180 = None
        mul_181: "f32[128, 256, 3072][786432, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_200, add_108);  view_200 = add_108 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_201: "f32[32768, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(mul_181, [32768, 3072]);  mul_181 = None
        permute_53: "f32[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(primals_109, [1, 0]);  primals_109 = None
        permute_162: "f32[3072, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_53, [1, 0]);  permute_53 = None
        mm_37: "f32[32768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_201, permute_162);  permute_162 = None
        permute_163: "f32[3072, 32768][1, 3072]cuda:0" = torch.ops.aten.permute.default(view_201, [1, 0])
        mm_38: "f32[3072, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_163, view_87);  permute_163 = view_87 = None
        sum_52: "f32[1, 3072][3072, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_201, [0], True);  view_201 = None
        view_202: "f32[3072][1]cuda:0" = torch.ops.aten.reshape.default(sum_52, [3072]);  sum_52 = None
        view_203: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_37, [128, 256, 768]);  mm_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_183: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_203, primals_107);  primals_107 = None
        mul_184: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_183, 768)
        sum_53: "f32[128, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_183, [2], True)
        mul_185: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_183, mul_58);  mul_183 = None
        sum_54: "f32[128, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_185, [2], True);  mul_185 = None
        mul_186: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_58, sum_54);  sum_54 = None
        sub_51: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_184, sum_53);  mul_184 = sum_53 = None
        sub_52: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_51, mul_186);  sub_51 = mul_186 = None
        mul_187: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_8, sub_52);  div_8 = sub_52 = None
        mul_188: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_203, mul_58);  mul_58 = None
        sum_55: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_188, [0, 1]);  mul_188 = None
        sum_56: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_203, [0, 1]);  view_203 = None
        add_109: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_106, mul_187);  add_106 = mul_187 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_204: "f32[32768, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(add_109, [32768, 768])
        permute_52: "f32[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(primals_105, [1, 0]);  primals_105 = None
        permute_166: "f32[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_52, [1, 0]);  permute_52 = None
        mm_39: "f32[32768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_204, permute_166);  permute_166 = None
        permute_167: "f32[768, 32768][1, 768]cuda:0" = torch.ops.aten.permute.default(view_204, [1, 0])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_51: "f32[128, 256, 12, 64][196608, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_93, [0, 2, 1, 3])
        view_84: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_51, [128, 256, 768]);  permute_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_85: "f32[32768, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_84, [32768, 768]);  view_84 = None
        mm_40: "f32[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_167, view_85);  permute_167 = view_85 = None
        sum_57: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_204, [0], True);  view_204 = None
        view_205: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_57, [768]);  sum_57 = None
        view_206: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_39, [128, 256, 768]);  mm_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        view_207: "f32[128, 256, 12, 64][196608, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_206, [128, 256, 12, 64]);  view_206 = None
        permute_170: "f32[128, 12, 256, 64][196608, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_207, [0, 2, 1, 3]);  view_207 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:120 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_backward_4 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_170, getitem_90, getitem_91, getitem_92, None, getitem_93, getitem_94, getitem_95, getitem_96, 0.0, [True, True, True, False]);  permute_170 = getitem_90 = getitem_91 = getitem_92 = getitem_93 = getitem_94 = getitem_95 = getitem_96 = None
        getitem_158: "f32[128, 12, 256, 64][196608, 64, 768, 1]cuda:0" = _scaled_dot_product_efficient_attention_backward_4[0]
        getitem_159: "f32[128, 12, 256, 64][196608, 64, 768, 1]cuda:0" = _scaled_dot_product_efficient_attention_backward_4[1]
        getitem_160: "f32[128, 12, 256, 64][196608, 64, 768, 1]cuda:0" = _scaled_dot_product_efficient_attention_backward_4[2];  _scaled_dot_product_efficient_attention_backward_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        cat_4: "f32[384, 12, 256, 64][196608, 16384, 64, 1]cuda:0" = torch.ops.aten.cat.default([getitem_158, getitem_159, getitem_160]);  getitem_158 = getitem_159 = getitem_160 = None
        view_208: "f32[3, 128, 12, 256, 64][25165824, 196608, 16384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(cat_4, [3, 128, 12, 256, 64]);  cat_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        permute_171: "f32[128, 256, 3, 12, 64][196608, 64, 25165824, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_208, [1, 3, 0, 2, 4]);  view_208 = None
        clone_45: "f32[128, 256, 3, 12, 64][589824, 2304, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_171, memory_format = torch.contiguous_format);  permute_171 = None
        view_209: "f32[128, 256, 2304][589824, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(clone_45, [128, 256, 2304]);  clone_45 = None
        view_210: "f32[32768, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(view_209, [32768, 2304]);  view_209 = None
        permute_49: "f32[768, 2304][1, 768]cuda:0" = torch.ops.aten.permute.default(primals_103, [1, 0]);  primals_103 = None
        permute_172: "f32[2304, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_49, [1, 0]);  permute_49 = None
        mm_41: "f32[32768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_210, permute_172);  permute_172 = None
        permute_173: "f32[2304, 32768][1, 2304]cuda:0" = torch.ops.aten.permute.default(view_210, [1, 0])
        mm_42: "f32[2304, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_173, view_81);  permute_173 = view_81 = None
        sum_58: "f32[1, 2304][2304, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_210, [0], True);  view_210 = None
        view_211: "f32[2304][1]cuda:0" = torch.ops.aten.reshape.default(sum_58, [2304]);  sum_58 = None
        view_212: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_41, [128, 256, 768]);  mm_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_190: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_212, primals_101);  primals_101 = None
        mul_191: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_190, 768)
        sum_59: "f32[128, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_190, [2], True)
        mul_192: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_190, mul_56);  mul_190 = None
        sum_60: "f32[128, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_192, [2], True);  mul_192 = None
        mul_193: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_56, sum_60);  sum_60 = None
        sub_54: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_191, sum_59);  mul_191 = sum_59 = None
        sub_55: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_54, mul_193);  sub_54 = mul_193 = None
        mul_194: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_9, sub_55);  div_9 = sub_55 = None
        mul_195: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_212, mul_56);  mul_56 = None
        sum_61: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_195, [0, 1]);  mul_195 = None
        sum_62: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_212, [0, 1]);  view_212 = None
        add_110: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_109, mul_194);  add_109 = mul_194 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_213: "f32[32768, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(add_110, [32768, 768])
        permute_48: "f32[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(primals_99, [1, 0]);  primals_99 = None
        permute_176: "f32[768, 3072][3072, 1]cuda:0" = torch.ops.aten.permute.default(permute_48, [1, 0]);  permute_48 = None
        mm_43: "f32[32768, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(view_213, permute_176);  permute_176 = None
        permute_177: "f32[768, 32768][1, 768]cuda:0" = torch.ops.aten.permute.default(view_213, [1, 0])
        mm_44: "f32[768, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(permute_177, view_79);  permute_177 = view_79 = None
        sum_63: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_213, [0], True);  view_213 = None
        view_214: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_63, [768]);  sum_63 = None
        view_215: "f32[128, 256, 3072][786432, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(mm_43, [128, 256, 3072]);  mm_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_78: "f32[128, 256, 3072][786432, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_30, [128, 256, 3072]);  addmm_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        mul_54: "f32[128, 256, 3072][786432, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_78, 0.7071067811865476)
        erf_7: "f32[128, 256, 3072][786432, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_54);  mul_54 = None
        add_55: "f32[128, 256, 3072][786432, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
        mul_197: "f32[128, 256, 3072][786432, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_55, 0.5);  add_55 = None
        mul_198: "f32[128, 256, 3072][786432, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_78, view_78)
        mul_199: "f32[128, 256, 3072][786432, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_198, -0.5);  mul_198 = None
        exp_5: "f32[128, 256, 3072][786432, 3072, 1]cuda:0" = torch.ops.aten.exp.default(mul_199);  mul_199 = None
        mul_200: "f32[128, 256, 3072][786432, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_5, 0.3989422804014327);  exp_5 = None
        mul_201: "f32[128, 256, 3072][786432, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_78, mul_200);  view_78 = mul_200 = None
        add_112: "f32[128, 256, 3072][786432, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_197, mul_201);  mul_197 = mul_201 = None
        mul_202: "f32[128, 256, 3072][786432, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_215, add_112);  view_215 = add_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_216: "f32[32768, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(mul_202, [32768, 3072]);  mul_202 = None
        permute_47: "f32[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(primals_97, [1, 0]);  primals_97 = None
        permute_180: "f32[3072, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_47, [1, 0]);  permute_47 = None
        mm_45: "f32[32768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_216, permute_180);  permute_180 = None
        permute_181: "f32[3072, 32768][1, 3072]cuda:0" = torch.ops.aten.permute.default(view_216, [1, 0])
        mm_46: "f32[3072, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_181, view_77);  permute_181 = view_77 = None
        sum_64: "f32[1, 3072][3072, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_216, [0], True);  view_216 = None
        view_217: "f32[3072][1]cuda:0" = torch.ops.aten.reshape.default(sum_64, [3072]);  sum_64 = None
        view_218: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_45, [128, 256, 768]);  mm_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_204: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_218, primals_95);  primals_95 = None
        mul_205: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_204, 768)
        sum_65: "f32[128, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_204, [2], True)
        mul_206: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_204, mul_51);  mul_204 = None
        sum_66: "f32[128, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_206, [2], True);  mul_206 = None
        mul_207: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_51, sum_66);  sum_66 = None
        sub_57: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_205, sum_65);  mul_205 = sum_65 = None
        sub_58: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_57, mul_207);  sub_57 = mul_207 = None
        mul_208: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_10, sub_58);  div_10 = sub_58 = None
        mul_209: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_218, mul_51);  mul_51 = None
        sum_67: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_209, [0, 1]);  mul_209 = None
        sum_68: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_218, [0, 1]);  view_218 = None
        add_113: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_110, mul_208);  add_110 = mul_208 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_219: "f32[32768, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(add_113, [32768, 768])
        permute_46: "f32[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(primals_93, [1, 0]);  primals_93 = None
        permute_184: "f32[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_46, [1, 0]);  permute_46 = None
        mm_47: "f32[32768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_219, permute_184);  permute_184 = None
        permute_185: "f32[768, 32768][1, 768]cuda:0" = torch.ops.aten.permute.default(view_219, [1, 0])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_45: "f32[128, 256, 12, 64][196608, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_82, [0, 2, 1, 3])
        view_74: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_45, [128, 256, 768]);  permute_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_75: "f32[32768, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_74, [32768, 768]);  view_74 = None
        mm_48: "f32[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_185, view_75);  permute_185 = view_75 = None
        sum_69: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_219, [0], True);  view_219 = None
        view_220: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_69, [768]);  sum_69 = None
        view_221: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_47, [128, 256, 768]);  mm_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        view_222: "f32[128, 256, 12, 64][196608, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_221, [128, 256, 12, 64]);  view_221 = None
        permute_188: "f32[128, 12, 256, 64][196608, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_222, [0, 2, 1, 3]);  view_222 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:120 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_backward_5 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_188, getitem_79, getitem_80, getitem_81, None, getitem_82, getitem_83, getitem_84, getitem_85, 0.0, [True, True, True, False]);  permute_188 = getitem_79 = getitem_80 = getitem_81 = getitem_82 = getitem_83 = getitem_84 = getitem_85 = None
        getitem_162: "f32[128, 12, 256, 64][196608, 64, 768, 1]cuda:0" = _scaled_dot_product_efficient_attention_backward_5[0]
        getitem_163: "f32[128, 12, 256, 64][196608, 64, 768, 1]cuda:0" = _scaled_dot_product_efficient_attention_backward_5[1]
        getitem_164: "f32[128, 12, 256, 64][196608, 64, 768, 1]cuda:0" = _scaled_dot_product_efficient_attention_backward_5[2];  _scaled_dot_product_efficient_attention_backward_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        cat_5: "f32[384, 12, 256, 64][196608, 16384, 64, 1]cuda:0" = torch.ops.aten.cat.default([getitem_162, getitem_163, getitem_164]);  getitem_162 = getitem_163 = getitem_164 = None
        view_223: "f32[3, 128, 12, 256, 64][25165824, 196608, 16384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(cat_5, [3, 128, 12, 256, 64]);  cat_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        permute_189: "f32[128, 256, 3, 12, 64][196608, 64, 25165824, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_223, [1, 3, 0, 2, 4]);  view_223 = None
        clone_46: "f32[128, 256, 3, 12, 64][589824, 2304, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_189, memory_format = torch.contiguous_format);  permute_189 = None
        view_224: "f32[128, 256, 2304][589824, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(clone_46, [128, 256, 2304]);  clone_46 = None
        view_225: "f32[32768, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(view_224, [32768, 2304]);  view_224 = None
        permute_43: "f32[768, 2304][1, 768]cuda:0" = torch.ops.aten.permute.default(primals_91, [1, 0]);  primals_91 = None
        permute_190: "f32[2304, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_43, [1, 0]);  permute_43 = None
        mm_49: "f32[32768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_225, permute_190);  permute_190 = None
        permute_191: "f32[2304, 32768][1, 2304]cuda:0" = torch.ops.aten.permute.default(view_225, [1, 0])
        mm_50: "f32[2304, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_191, view_71);  permute_191 = view_71 = None
        sum_70: "f32[1, 2304][2304, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_225, [0], True);  view_225 = None
        view_226: "f32[2304][1]cuda:0" = torch.ops.aten.reshape.default(sum_70, [2304]);  sum_70 = None
        view_227: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_49, [128, 256, 768]);  mm_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_211: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_227, primals_89);  primals_89 = None
        mul_212: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_211, 768)
        sum_71: "f32[128, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_211, [2], True)
        mul_213: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_211, mul_49);  mul_211 = None
        sum_72: "f32[128, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_213, [2], True);  mul_213 = None
        mul_214: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_49, sum_72);  sum_72 = None
        sub_60: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_212, sum_71);  mul_212 = sum_71 = None
        sub_61: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_60, mul_214);  sub_60 = mul_214 = None
        mul_215: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_11, sub_61);  div_11 = sub_61 = None
        mul_216: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_227, mul_49);  mul_49 = None
        sum_73: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_216, [0, 1]);  mul_216 = None
        sum_74: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_227, [0, 1]);  view_227 = None
        add_114: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_113, mul_215);  add_113 = mul_215 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_228: "f32[32768, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(add_114, [32768, 768])
        permute_42: "f32[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(primals_87, [1, 0]);  primals_87 = None
        permute_194: "f32[768, 3072][3072, 1]cuda:0" = torch.ops.aten.permute.default(permute_42, [1, 0]);  permute_42 = None
        mm_51: "f32[32768, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(view_228, permute_194);  permute_194 = None
        permute_195: "f32[768, 32768][1, 768]cuda:0" = torch.ops.aten.permute.default(view_228, [1, 0])
        mm_52: "f32[768, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(permute_195, view_69);  permute_195 = view_69 = None
        sum_75: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_228, [0], True);  view_228 = None
        view_229: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_75, [768]);  sum_75 = None
        view_230: "f32[128, 256, 3072][786432, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(mm_51, [128, 256, 3072]);  mm_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_68: "f32[128, 256, 3072][786432, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_26, [128, 256, 3072]);  addmm_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        mul_47: "f32[128, 256, 3072][786432, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_68, 0.7071067811865476)
        erf_6: "f32[128, 256, 3072][786432, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_47);  mul_47 = None
        add_48: "f32[128, 256, 3072][786432, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
        mul_218: "f32[128, 256, 3072][786432, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_48, 0.5);  add_48 = None
        mul_219: "f32[128, 256, 3072][786432, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_68, view_68)
        mul_220: "f32[128, 256, 3072][786432, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_219, -0.5);  mul_219 = None
        exp_6: "f32[128, 256, 3072][786432, 3072, 1]cuda:0" = torch.ops.aten.exp.default(mul_220);  mul_220 = None
        mul_221: "f32[128, 256, 3072][786432, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_6, 0.3989422804014327);  exp_6 = None
        mul_222: "f32[128, 256, 3072][786432, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_68, mul_221);  view_68 = mul_221 = None
        add_116: "f32[128, 256, 3072][786432, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_218, mul_222);  mul_218 = mul_222 = None
        mul_223: "f32[128, 256, 3072][786432, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_230, add_116);  view_230 = add_116 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_231: "f32[32768, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(mul_223, [32768, 3072]);  mul_223 = None
        permute_41: "f32[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(primals_85, [1, 0]);  primals_85 = None
        permute_198: "f32[3072, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_41, [1, 0]);  permute_41 = None
        mm_53: "f32[32768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_231, permute_198);  permute_198 = None
        permute_199: "f32[3072, 32768][1, 3072]cuda:0" = torch.ops.aten.permute.default(view_231, [1, 0])
        mm_54: "f32[3072, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_199, view_67);  permute_199 = view_67 = None
        sum_76: "f32[1, 3072][3072, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_231, [0], True);  view_231 = None
        view_232: "f32[3072][1]cuda:0" = torch.ops.aten.reshape.default(sum_76, [3072]);  sum_76 = None
        view_233: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_53, [128, 256, 768]);  mm_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_225: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_233, primals_83);  primals_83 = None
        mul_226: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_225, 768)
        sum_77: "f32[128, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_225, [2], True)
        mul_227: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_225, mul_44);  mul_225 = None
        sum_78: "f32[128, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_227, [2], True);  mul_227 = None
        mul_228: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_44, sum_78);  sum_78 = None
        sub_63: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_226, sum_77);  mul_226 = sum_77 = None
        sub_64: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_63, mul_228);  sub_63 = mul_228 = None
        mul_229: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_12, sub_64);  div_12 = sub_64 = None
        mul_230: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_233, mul_44);  mul_44 = None
        sum_79: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_230, [0, 1]);  mul_230 = None
        sum_80: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_233, [0, 1]);  view_233 = None
        add_117: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_114, mul_229);  add_114 = mul_229 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_234: "f32[32768, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(add_117, [32768, 768])
        permute_40: "f32[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(primals_81, [1, 0]);  primals_81 = None
        permute_202: "f32[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_40, [1, 0]);  permute_40 = None
        mm_55: "f32[32768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_234, permute_202);  permute_202 = None
        permute_203: "f32[768, 32768][1, 768]cuda:0" = torch.ops.aten.permute.default(view_234, [1, 0])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_39: "f32[128, 256, 12, 64][196608, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_71, [0, 2, 1, 3])
        view_64: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_39, [128, 256, 768]);  permute_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_65: "f32[32768, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_64, [32768, 768]);  view_64 = None
        mm_56: "f32[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_203, view_65);  permute_203 = view_65 = None
        sum_81: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_234, [0], True);  view_234 = None
        view_235: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_81, [768]);  sum_81 = None
        view_236: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_55, [128, 256, 768]);  mm_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        view_237: "f32[128, 256, 12, 64][196608, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_236, [128, 256, 12, 64]);  view_236 = None
        permute_206: "f32[128, 12, 256, 64][196608, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_237, [0, 2, 1, 3]);  view_237 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:120 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_backward_6 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_206, getitem_68, getitem_69, getitem_70, None, getitem_71, getitem_72, getitem_73, getitem_74, 0.0, [True, True, True, False]);  permute_206 = getitem_68 = getitem_69 = getitem_70 = getitem_71 = getitem_72 = getitem_73 = getitem_74 = None
        getitem_166: "f32[128, 12, 256, 64][196608, 64, 768, 1]cuda:0" = _scaled_dot_product_efficient_attention_backward_6[0]
        getitem_167: "f32[128, 12, 256, 64][196608, 64, 768, 1]cuda:0" = _scaled_dot_product_efficient_attention_backward_6[1]
        getitem_168: "f32[128, 12, 256, 64][196608, 64, 768, 1]cuda:0" = _scaled_dot_product_efficient_attention_backward_6[2];  _scaled_dot_product_efficient_attention_backward_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        cat_6: "f32[384, 12, 256, 64][196608, 16384, 64, 1]cuda:0" = torch.ops.aten.cat.default([getitem_166, getitem_167, getitem_168]);  getitem_166 = getitem_167 = getitem_168 = None
        view_238: "f32[3, 128, 12, 256, 64][25165824, 196608, 16384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(cat_6, [3, 128, 12, 256, 64]);  cat_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        permute_207: "f32[128, 256, 3, 12, 64][196608, 64, 25165824, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_238, [1, 3, 0, 2, 4]);  view_238 = None
        clone_47: "f32[128, 256, 3, 12, 64][589824, 2304, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_207, memory_format = torch.contiguous_format);  permute_207 = None
        view_239: "f32[128, 256, 2304][589824, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(clone_47, [128, 256, 2304]);  clone_47 = None
        view_240: "f32[32768, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(view_239, [32768, 2304]);  view_239 = None
        permute_37: "f32[768, 2304][1, 768]cuda:0" = torch.ops.aten.permute.default(primals_79, [1, 0]);  primals_79 = None
        permute_208: "f32[2304, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_37, [1, 0]);  permute_37 = None
        mm_57: "f32[32768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_240, permute_208);  permute_208 = None
        permute_209: "f32[2304, 32768][1, 2304]cuda:0" = torch.ops.aten.permute.default(view_240, [1, 0])
        mm_58: "f32[2304, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_209, view_61);  permute_209 = view_61 = None
        sum_82: "f32[1, 2304][2304, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_240, [0], True);  view_240 = None
        view_241: "f32[2304][1]cuda:0" = torch.ops.aten.reshape.default(sum_82, [2304]);  sum_82 = None
        view_242: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_57, [128, 256, 768]);  mm_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_232: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_242, primals_77);  primals_77 = None
        mul_233: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_232, 768)
        sum_83: "f32[128, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_232, [2], True)
        mul_234: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_232, mul_42);  mul_232 = None
        sum_84: "f32[128, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_234, [2], True);  mul_234 = None
        mul_235: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_42, sum_84);  sum_84 = None
        sub_66: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_233, sum_83);  mul_233 = sum_83 = None
        sub_67: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_66, mul_235);  sub_66 = mul_235 = None
        mul_236: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_13, sub_67);  div_13 = sub_67 = None
        mul_237: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_242, mul_42);  mul_42 = None
        sum_85: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_237, [0, 1]);  mul_237 = None
        sum_86: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_242, [0, 1]);  view_242 = None
        add_118: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_117, mul_236);  add_117 = mul_236 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_243: "f32[32768, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(add_118, [32768, 768])
        permute_36: "f32[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(primals_75, [1, 0]);  primals_75 = None
        permute_212: "f32[768, 3072][3072, 1]cuda:0" = torch.ops.aten.permute.default(permute_36, [1, 0]);  permute_36 = None
        mm_59: "f32[32768, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(view_243, permute_212);  permute_212 = None
        permute_213: "f32[768, 32768][1, 768]cuda:0" = torch.ops.aten.permute.default(view_243, [1, 0])
        mm_60: "f32[768, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(permute_213, view_59);  permute_213 = view_59 = None
        sum_87: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_243, [0], True);  view_243 = None
        view_244: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_87, [768]);  sum_87 = None
        view_245: "f32[128, 256, 3072][786432, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(mm_59, [128, 256, 3072]);  mm_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_58: "f32[128, 256, 3072][786432, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_22, [128, 256, 3072]);  addmm_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        mul_40: "f32[128, 256, 3072][786432, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_58, 0.7071067811865476)
        erf_5: "f32[128, 256, 3072][786432, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_40);  mul_40 = None
        add_41: "f32[128, 256, 3072][786432, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
        mul_239: "f32[128, 256, 3072][786432, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_41, 0.5);  add_41 = None
        mul_240: "f32[128, 256, 3072][786432, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_58, view_58)
        mul_241: "f32[128, 256, 3072][786432, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_240, -0.5);  mul_240 = None
        exp_7: "f32[128, 256, 3072][786432, 3072, 1]cuda:0" = torch.ops.aten.exp.default(mul_241);  mul_241 = None
        mul_242: "f32[128, 256, 3072][786432, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_7, 0.3989422804014327);  exp_7 = None
        mul_243: "f32[128, 256, 3072][786432, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_58, mul_242);  view_58 = mul_242 = None
        add_120: "f32[128, 256, 3072][786432, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_239, mul_243);  mul_239 = mul_243 = None
        mul_244: "f32[128, 256, 3072][786432, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_245, add_120);  view_245 = add_120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_246: "f32[32768, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(mul_244, [32768, 3072]);  mul_244 = None
        permute_35: "f32[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(primals_73, [1, 0]);  primals_73 = None
        permute_216: "f32[3072, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_35, [1, 0]);  permute_35 = None
        mm_61: "f32[32768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_246, permute_216);  permute_216 = None
        permute_217: "f32[3072, 32768][1, 3072]cuda:0" = torch.ops.aten.permute.default(view_246, [1, 0])
        mm_62: "f32[3072, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_217, view_57);  permute_217 = view_57 = None
        sum_88: "f32[1, 3072][3072, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_246, [0], True);  view_246 = None
        view_247: "f32[3072][1]cuda:0" = torch.ops.aten.reshape.default(sum_88, [3072]);  sum_88 = None
        view_248: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_61, [128, 256, 768]);  mm_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_246: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_248, primals_71);  primals_71 = None
        mul_247: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_246, 768)
        sum_89: "f32[128, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_246, [2], True)
        mul_248: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_246, mul_37);  mul_246 = None
        sum_90: "f32[128, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_248, [2], True);  mul_248 = None
        mul_249: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_37, sum_90);  sum_90 = None
        sub_69: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_247, sum_89);  mul_247 = sum_89 = None
        sub_70: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_69, mul_249);  sub_69 = mul_249 = None
        mul_250: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_14, sub_70);  div_14 = sub_70 = None
        mul_251: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_248, mul_37);  mul_37 = None
        sum_91: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_251, [0, 1]);  mul_251 = None
        sum_92: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_248, [0, 1]);  view_248 = None
        add_121: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_118, mul_250);  add_118 = mul_250 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_249: "f32[32768, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(add_121, [32768, 768])
        permute_34: "f32[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(primals_69, [1, 0]);  primals_69 = None
        permute_220: "f32[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_34, [1, 0]);  permute_34 = None
        mm_63: "f32[32768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_249, permute_220);  permute_220 = None
        permute_221: "f32[768, 32768][1, 768]cuda:0" = torch.ops.aten.permute.default(view_249, [1, 0])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_33: "f32[128, 256, 12, 64][196608, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_60, [0, 2, 1, 3])
        view_54: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_33, [128, 256, 768]);  permute_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_55: "f32[32768, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_54, [32768, 768]);  view_54 = None
        mm_64: "f32[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_221, view_55);  permute_221 = view_55 = None
        sum_93: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_249, [0], True);  view_249 = None
        view_250: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_93, [768]);  sum_93 = None
        view_251: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_63, [128, 256, 768]);  mm_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        view_252: "f32[128, 256, 12, 64][196608, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_251, [128, 256, 12, 64]);  view_251 = None
        permute_224: "f32[128, 12, 256, 64][196608, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_252, [0, 2, 1, 3]);  view_252 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:120 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_backward_7 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_224, getitem_57, getitem_58, getitem_59, None, getitem_60, getitem_61, getitem_62, getitem_63, 0.0, [True, True, True, False]);  permute_224 = getitem_57 = getitem_58 = getitem_59 = getitem_60 = getitem_61 = getitem_62 = getitem_63 = None
        getitem_170: "f32[128, 12, 256, 64][196608, 64, 768, 1]cuda:0" = _scaled_dot_product_efficient_attention_backward_7[0]
        getitem_171: "f32[128, 12, 256, 64][196608, 64, 768, 1]cuda:0" = _scaled_dot_product_efficient_attention_backward_7[1]
        getitem_172: "f32[128, 12, 256, 64][196608, 64, 768, 1]cuda:0" = _scaled_dot_product_efficient_attention_backward_7[2];  _scaled_dot_product_efficient_attention_backward_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        cat_7: "f32[384, 12, 256, 64][196608, 16384, 64, 1]cuda:0" = torch.ops.aten.cat.default([getitem_170, getitem_171, getitem_172]);  getitem_170 = getitem_171 = getitem_172 = None
        view_253: "f32[3, 128, 12, 256, 64][25165824, 196608, 16384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(cat_7, [3, 128, 12, 256, 64]);  cat_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        permute_225: "f32[128, 256, 3, 12, 64][196608, 64, 25165824, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_253, [1, 3, 0, 2, 4]);  view_253 = None
        clone_48: "f32[128, 256, 3, 12, 64][589824, 2304, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_225, memory_format = torch.contiguous_format);  permute_225 = None
        view_254: "f32[128, 256, 2304][589824, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(clone_48, [128, 256, 2304]);  clone_48 = None
        view_255: "f32[32768, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(view_254, [32768, 2304]);  view_254 = None
        permute_31: "f32[768, 2304][1, 768]cuda:0" = torch.ops.aten.permute.default(primals_67, [1, 0]);  primals_67 = None
        permute_226: "f32[2304, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_31, [1, 0]);  permute_31 = None
        mm_65: "f32[32768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_255, permute_226);  permute_226 = None
        permute_227: "f32[2304, 32768][1, 2304]cuda:0" = torch.ops.aten.permute.default(view_255, [1, 0])
        mm_66: "f32[2304, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_227, view_51);  permute_227 = view_51 = None
        sum_94: "f32[1, 2304][2304, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_255, [0], True);  view_255 = None
        view_256: "f32[2304][1]cuda:0" = torch.ops.aten.reshape.default(sum_94, [2304]);  sum_94 = None
        view_257: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_65, [128, 256, 768]);  mm_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_253: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_257, primals_65);  primals_65 = None
        mul_254: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_253, 768)
        sum_95: "f32[128, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_253, [2], True)
        mul_255: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_253, mul_35);  mul_253 = None
        sum_96: "f32[128, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_255, [2], True);  mul_255 = None
        mul_256: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_35, sum_96);  sum_96 = None
        sub_72: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_254, sum_95);  mul_254 = sum_95 = None
        sub_73: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_72, mul_256);  sub_72 = mul_256 = None
        mul_257: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_15, sub_73);  div_15 = sub_73 = None
        mul_258: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_257, mul_35);  mul_35 = None
        sum_97: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_258, [0, 1]);  mul_258 = None
        sum_98: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_257, [0, 1]);  view_257 = None
        add_122: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_121, mul_257);  add_121 = mul_257 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_258: "f32[32768, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(add_122, [32768, 768])
        permute_30: "f32[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(primals_63, [1, 0]);  primals_63 = None
        permute_230: "f32[768, 3072][3072, 1]cuda:0" = torch.ops.aten.permute.default(permute_30, [1, 0]);  permute_30 = None
        mm_67: "f32[32768, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(view_258, permute_230);  permute_230 = None
        permute_231: "f32[768, 32768][1, 768]cuda:0" = torch.ops.aten.permute.default(view_258, [1, 0])
        mm_68: "f32[768, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(permute_231, view_49);  permute_231 = view_49 = None
        sum_99: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_258, [0], True);  view_258 = None
        view_259: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_99, [768]);  sum_99 = None
        view_260: "f32[128, 256, 3072][786432, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(mm_67, [128, 256, 3072]);  mm_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_48: "f32[128, 256, 3072][786432, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_18, [128, 256, 3072]);  addmm_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        mul_33: "f32[128, 256, 3072][786432, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_48, 0.7071067811865476)
        erf_4: "f32[128, 256, 3072][786432, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_33);  mul_33 = None
        add_34: "f32[128, 256, 3072][786432, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
        mul_260: "f32[128, 256, 3072][786432, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_34, 0.5);  add_34 = None
        mul_261: "f32[128, 256, 3072][786432, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_48, view_48)
        mul_262: "f32[128, 256, 3072][786432, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_261, -0.5);  mul_261 = None
        exp_8: "f32[128, 256, 3072][786432, 3072, 1]cuda:0" = torch.ops.aten.exp.default(mul_262);  mul_262 = None
        mul_263: "f32[128, 256, 3072][786432, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_8, 0.3989422804014327);  exp_8 = None
        mul_264: "f32[128, 256, 3072][786432, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_48, mul_263);  view_48 = mul_263 = None
        add_124: "f32[128, 256, 3072][786432, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_260, mul_264);  mul_260 = mul_264 = None
        mul_265: "f32[128, 256, 3072][786432, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_260, add_124);  view_260 = add_124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_261: "f32[32768, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(mul_265, [32768, 3072]);  mul_265 = None
        permute_29: "f32[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(primals_61, [1, 0]);  primals_61 = None
        permute_234: "f32[3072, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_29, [1, 0]);  permute_29 = None
        mm_69: "f32[32768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_261, permute_234);  permute_234 = None
        permute_235: "f32[3072, 32768][1, 3072]cuda:0" = torch.ops.aten.permute.default(view_261, [1, 0])
        mm_70: "f32[3072, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_235, view_47);  permute_235 = view_47 = None
        sum_100: "f32[1, 3072][3072, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_261, [0], True);  view_261 = None
        view_262: "f32[3072][1]cuda:0" = torch.ops.aten.reshape.default(sum_100, [3072]);  sum_100 = None
        view_263: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_69, [128, 256, 768]);  mm_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_267: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_263, primals_59);  primals_59 = None
        mul_268: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_267, 768)
        sum_101: "f32[128, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_267, [2], True)
        mul_269: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_267, mul_30);  mul_267 = None
        sum_102: "f32[128, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_269, [2], True);  mul_269 = None
        mul_270: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_30, sum_102);  sum_102 = None
        sub_75: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_268, sum_101);  mul_268 = sum_101 = None
        sub_76: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_75, mul_270);  sub_75 = mul_270 = None
        mul_271: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_16, sub_76);  div_16 = sub_76 = None
        mul_272: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_263, mul_30);  mul_30 = None
        sum_103: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_272, [0, 1]);  mul_272 = None
        sum_104: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_263, [0, 1]);  view_263 = None
        add_125: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_122, mul_271);  add_122 = mul_271 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_264: "f32[32768, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(add_125, [32768, 768])
        permute_28: "f32[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(primals_57, [1, 0]);  primals_57 = None
        permute_238: "f32[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_28, [1, 0]);  permute_28 = None
        mm_71: "f32[32768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_264, permute_238);  permute_238 = None
        permute_239: "f32[768, 32768][1, 768]cuda:0" = torch.ops.aten.permute.default(view_264, [1, 0])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_27: "f32[128, 256, 12, 64][196608, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_49, [0, 2, 1, 3])
        view_44: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_27, [128, 256, 768]);  permute_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_45: "f32[32768, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_44, [32768, 768]);  view_44 = None
        mm_72: "f32[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_239, view_45);  permute_239 = view_45 = None
        sum_105: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_264, [0], True);  view_264 = None
        view_265: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_105, [768]);  sum_105 = None
        view_266: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_71, [128, 256, 768]);  mm_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        view_267: "f32[128, 256, 12, 64][196608, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_266, [128, 256, 12, 64]);  view_266 = None
        permute_242: "f32[128, 12, 256, 64][196608, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_267, [0, 2, 1, 3]);  view_267 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:120 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_backward_8 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_242, getitem_46, getitem_47, getitem_48, None, getitem_49, getitem_50, getitem_51, getitem_52, 0.0, [True, True, True, False]);  permute_242 = getitem_46 = getitem_47 = getitem_48 = getitem_49 = getitem_50 = getitem_51 = getitem_52 = None
        getitem_174: "f32[128, 12, 256, 64][196608, 64, 768, 1]cuda:0" = _scaled_dot_product_efficient_attention_backward_8[0]
        getitem_175: "f32[128, 12, 256, 64][196608, 64, 768, 1]cuda:0" = _scaled_dot_product_efficient_attention_backward_8[1]
        getitem_176: "f32[128, 12, 256, 64][196608, 64, 768, 1]cuda:0" = _scaled_dot_product_efficient_attention_backward_8[2];  _scaled_dot_product_efficient_attention_backward_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        cat_8: "f32[384, 12, 256, 64][196608, 16384, 64, 1]cuda:0" = torch.ops.aten.cat.default([getitem_174, getitem_175, getitem_176]);  getitem_174 = getitem_175 = getitem_176 = None
        view_268: "f32[3, 128, 12, 256, 64][25165824, 196608, 16384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(cat_8, [3, 128, 12, 256, 64]);  cat_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        permute_243: "f32[128, 256, 3, 12, 64][196608, 64, 25165824, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_268, [1, 3, 0, 2, 4]);  view_268 = None
        clone_49: "f32[128, 256, 3, 12, 64][589824, 2304, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_243, memory_format = torch.contiguous_format);  permute_243 = None
        view_269: "f32[128, 256, 2304][589824, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(clone_49, [128, 256, 2304]);  clone_49 = None
        view_270: "f32[32768, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(view_269, [32768, 2304]);  view_269 = None
        permute_25: "f32[768, 2304][1, 768]cuda:0" = torch.ops.aten.permute.default(primals_55, [1, 0]);  primals_55 = None
        permute_244: "f32[2304, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_25, [1, 0]);  permute_25 = None
        mm_73: "f32[32768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_270, permute_244);  permute_244 = None
        permute_245: "f32[2304, 32768][1, 2304]cuda:0" = torch.ops.aten.permute.default(view_270, [1, 0])
        mm_74: "f32[2304, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_245, view_41);  permute_245 = view_41 = None
        sum_106: "f32[1, 2304][2304, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_270, [0], True);  view_270 = None
        view_271: "f32[2304][1]cuda:0" = torch.ops.aten.reshape.default(sum_106, [2304]);  sum_106 = None
        view_272: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_73, [128, 256, 768]);  mm_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_274: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_272, primals_53);  primals_53 = None
        mul_275: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_274, 768)
        sum_107: "f32[128, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_274, [2], True)
        mul_276: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_274, mul_28);  mul_274 = None
        sum_108: "f32[128, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_276, [2], True);  mul_276 = None
        mul_277: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_28, sum_108);  sum_108 = None
        sub_78: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_275, sum_107);  mul_275 = sum_107 = None
        sub_79: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_78, mul_277);  sub_78 = mul_277 = None
        mul_278: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_17, sub_79);  div_17 = sub_79 = None
        mul_279: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_272, mul_28);  mul_28 = None
        sum_109: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_279, [0, 1]);  mul_279 = None
        sum_110: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_272, [0, 1]);  view_272 = None
        add_126: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_125, mul_278);  add_125 = mul_278 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_273: "f32[32768, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(add_126, [32768, 768])
        permute_24: "f32[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(primals_51, [1, 0]);  primals_51 = None
        permute_248: "f32[768, 3072][3072, 1]cuda:0" = torch.ops.aten.permute.default(permute_24, [1, 0]);  permute_24 = None
        mm_75: "f32[32768, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(view_273, permute_248);  permute_248 = None
        permute_249: "f32[768, 32768][1, 768]cuda:0" = torch.ops.aten.permute.default(view_273, [1, 0])
        mm_76: "f32[768, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(permute_249, view_39);  permute_249 = view_39 = None
        sum_111: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_273, [0], True);  view_273 = None
        view_274: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_111, [768]);  sum_111 = None
        view_275: "f32[128, 256, 3072][786432, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(mm_75, [128, 256, 3072]);  mm_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_38: "f32[128, 256, 3072][786432, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_14, [128, 256, 3072]);  addmm_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        mul_26: "f32[128, 256, 3072][786432, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_38, 0.7071067811865476)
        erf_3: "f32[128, 256, 3072][786432, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_26);  mul_26 = None
        add_27: "f32[128, 256, 3072][786432, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
        mul_281: "f32[128, 256, 3072][786432, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_27, 0.5);  add_27 = None
        mul_282: "f32[128, 256, 3072][786432, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_38, view_38)
        mul_283: "f32[128, 256, 3072][786432, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_282, -0.5);  mul_282 = None
        exp_9: "f32[128, 256, 3072][786432, 3072, 1]cuda:0" = torch.ops.aten.exp.default(mul_283);  mul_283 = None
        mul_284: "f32[128, 256, 3072][786432, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_9, 0.3989422804014327);  exp_9 = None
        mul_285: "f32[128, 256, 3072][786432, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_38, mul_284);  view_38 = mul_284 = None
        add_128: "f32[128, 256, 3072][786432, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_281, mul_285);  mul_281 = mul_285 = None
        mul_286: "f32[128, 256, 3072][786432, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_275, add_128);  view_275 = add_128 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_276: "f32[32768, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(mul_286, [32768, 3072]);  mul_286 = None
        permute_23: "f32[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(primals_49, [1, 0]);  primals_49 = None
        permute_252: "f32[3072, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_23, [1, 0]);  permute_23 = None
        mm_77: "f32[32768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_276, permute_252);  permute_252 = None
        permute_253: "f32[3072, 32768][1, 3072]cuda:0" = torch.ops.aten.permute.default(view_276, [1, 0])
        mm_78: "f32[3072, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_253, view_37);  permute_253 = view_37 = None
        sum_112: "f32[1, 3072][3072, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_276, [0], True);  view_276 = None
        view_277: "f32[3072][1]cuda:0" = torch.ops.aten.reshape.default(sum_112, [3072]);  sum_112 = None
        view_278: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_77, [128, 256, 768]);  mm_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_288: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_278, primals_47);  primals_47 = None
        mul_289: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_288, 768)
        sum_113: "f32[128, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_288, [2], True)
        mul_290: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_288, mul_23);  mul_288 = None
        sum_114: "f32[128, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_290, [2], True);  mul_290 = None
        mul_291: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_23, sum_114);  sum_114 = None
        sub_81: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_289, sum_113);  mul_289 = sum_113 = None
        sub_82: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_81, mul_291);  sub_81 = mul_291 = None
        mul_292: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_18, sub_82);  div_18 = sub_82 = None
        mul_293: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_278, mul_23);  mul_23 = None
        sum_115: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_293, [0, 1]);  mul_293 = None
        sum_116: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_278, [0, 1]);  view_278 = None
        add_129: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_126, mul_292);  add_126 = mul_292 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_279: "f32[32768, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(add_129, [32768, 768])
        permute_22: "f32[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(primals_45, [1, 0]);  primals_45 = None
        permute_256: "f32[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_22, [1, 0]);  permute_22 = None
        mm_79: "f32[32768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_279, permute_256);  permute_256 = None
        permute_257: "f32[768, 32768][1, 768]cuda:0" = torch.ops.aten.permute.default(view_279, [1, 0])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_21: "f32[128, 256, 12, 64][196608, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_38, [0, 2, 1, 3])
        view_34: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_21, [128, 256, 768]);  permute_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_35: "f32[32768, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_34, [32768, 768]);  view_34 = None
        mm_80: "f32[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_257, view_35);  permute_257 = view_35 = None
        sum_117: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_279, [0], True);  view_279 = None
        view_280: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_117, [768]);  sum_117 = None
        view_281: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_79, [128, 256, 768]);  mm_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        view_282: "f32[128, 256, 12, 64][196608, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_281, [128, 256, 12, 64]);  view_281 = None
        permute_260: "f32[128, 12, 256, 64][196608, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_282, [0, 2, 1, 3]);  view_282 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:120 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_backward_9 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_260, getitem_35, getitem_36, getitem_37, None, getitem_38, getitem_39, getitem_40, getitem_41, 0.0, [True, True, True, False]);  permute_260 = getitem_35 = getitem_36 = getitem_37 = getitem_38 = getitem_39 = getitem_40 = getitem_41 = None
        getitem_178: "f32[128, 12, 256, 64][196608, 64, 768, 1]cuda:0" = _scaled_dot_product_efficient_attention_backward_9[0]
        getitem_179: "f32[128, 12, 256, 64][196608, 64, 768, 1]cuda:0" = _scaled_dot_product_efficient_attention_backward_9[1]
        getitem_180: "f32[128, 12, 256, 64][196608, 64, 768, 1]cuda:0" = _scaled_dot_product_efficient_attention_backward_9[2];  _scaled_dot_product_efficient_attention_backward_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        cat_9: "f32[384, 12, 256, 64][196608, 16384, 64, 1]cuda:0" = torch.ops.aten.cat.default([getitem_178, getitem_179, getitem_180]);  getitem_178 = getitem_179 = getitem_180 = None
        view_283: "f32[3, 128, 12, 256, 64][25165824, 196608, 16384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(cat_9, [3, 128, 12, 256, 64]);  cat_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        permute_261: "f32[128, 256, 3, 12, 64][196608, 64, 25165824, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_283, [1, 3, 0, 2, 4]);  view_283 = None
        clone_50: "f32[128, 256, 3, 12, 64][589824, 2304, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_261, memory_format = torch.contiguous_format);  permute_261 = None
        view_284: "f32[128, 256, 2304][589824, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(clone_50, [128, 256, 2304]);  clone_50 = None
        view_285: "f32[32768, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(view_284, [32768, 2304]);  view_284 = None
        permute_19: "f32[768, 2304][1, 768]cuda:0" = torch.ops.aten.permute.default(primals_43, [1, 0]);  primals_43 = None
        permute_262: "f32[2304, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_19, [1, 0]);  permute_19 = None
        mm_81: "f32[32768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_285, permute_262);  permute_262 = None
        permute_263: "f32[2304, 32768][1, 2304]cuda:0" = torch.ops.aten.permute.default(view_285, [1, 0])
        mm_82: "f32[2304, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_263, view_31);  permute_263 = view_31 = None
        sum_118: "f32[1, 2304][2304, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_285, [0], True);  view_285 = None
        view_286: "f32[2304][1]cuda:0" = torch.ops.aten.reshape.default(sum_118, [2304]);  sum_118 = None
        view_287: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_81, [128, 256, 768]);  mm_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_295: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_287, primals_41);  primals_41 = None
        mul_296: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_295, 768)
        sum_119: "f32[128, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_295, [2], True)
        mul_297: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_295, mul_21);  mul_295 = None
        sum_120: "f32[128, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_297, [2], True);  mul_297 = None
        mul_298: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_21, sum_120);  sum_120 = None
        sub_84: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_296, sum_119);  mul_296 = sum_119 = None
        sub_85: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_84, mul_298);  sub_84 = mul_298 = None
        mul_299: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_19, sub_85);  div_19 = sub_85 = None
        mul_300: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_287, mul_21);  mul_21 = None
        sum_121: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_300, [0, 1]);  mul_300 = None
        sum_122: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_287, [0, 1]);  view_287 = None
        add_130: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_129, mul_299);  add_129 = mul_299 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_288: "f32[32768, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(add_130, [32768, 768])
        permute_18: "f32[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(primals_39, [1, 0]);  primals_39 = None
        permute_266: "f32[768, 3072][3072, 1]cuda:0" = torch.ops.aten.permute.default(permute_18, [1, 0]);  permute_18 = None
        mm_83: "f32[32768, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(view_288, permute_266);  permute_266 = None
        permute_267: "f32[768, 32768][1, 768]cuda:0" = torch.ops.aten.permute.default(view_288, [1, 0])
        mm_84: "f32[768, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(permute_267, view_29);  permute_267 = view_29 = None
        sum_123: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_288, [0], True);  view_288 = None
        view_289: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_123, [768]);  sum_123 = None
        view_290: "f32[128, 256, 3072][786432, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(mm_83, [128, 256, 3072]);  mm_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_28: "f32[128, 256, 3072][786432, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_10, [128, 256, 3072]);  addmm_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        mul_19: "f32[128, 256, 3072][786432, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_28, 0.7071067811865476)
        erf_2: "f32[128, 256, 3072][786432, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_19);  mul_19 = None
        add_20: "f32[128, 256, 3072][786432, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
        mul_302: "f32[128, 256, 3072][786432, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_20, 0.5);  add_20 = None
        mul_303: "f32[128, 256, 3072][786432, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_28, view_28)
        mul_304: "f32[128, 256, 3072][786432, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_303, -0.5);  mul_303 = None
        exp_10: "f32[128, 256, 3072][786432, 3072, 1]cuda:0" = torch.ops.aten.exp.default(mul_304);  mul_304 = None
        mul_305: "f32[128, 256, 3072][786432, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_10, 0.3989422804014327);  exp_10 = None
        mul_306: "f32[128, 256, 3072][786432, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_28, mul_305);  view_28 = mul_305 = None
        add_132: "f32[128, 256, 3072][786432, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_302, mul_306);  mul_302 = mul_306 = None
        mul_307: "f32[128, 256, 3072][786432, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_290, add_132);  view_290 = add_132 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_291: "f32[32768, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(mul_307, [32768, 3072]);  mul_307 = None
        permute_17: "f32[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(primals_37, [1, 0]);  primals_37 = None
        permute_270: "f32[3072, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_17, [1, 0]);  permute_17 = None
        mm_85: "f32[32768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_291, permute_270);  permute_270 = None
        permute_271: "f32[3072, 32768][1, 3072]cuda:0" = torch.ops.aten.permute.default(view_291, [1, 0])
        mm_86: "f32[3072, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_271, view_27);  permute_271 = view_27 = None
        sum_124: "f32[1, 3072][3072, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_291, [0], True);  view_291 = None
        view_292: "f32[3072][1]cuda:0" = torch.ops.aten.reshape.default(sum_124, [3072]);  sum_124 = None
        view_293: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_85, [128, 256, 768]);  mm_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_309: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_293, primals_35);  primals_35 = None
        mul_310: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_309, 768)
        sum_125: "f32[128, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_309, [2], True)
        mul_311: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_309, mul_16);  mul_309 = None
        sum_126: "f32[128, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_311, [2], True);  mul_311 = None
        mul_312: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_16, sum_126);  sum_126 = None
        sub_87: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_310, sum_125);  mul_310 = sum_125 = None
        sub_88: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_87, mul_312);  sub_87 = mul_312 = None
        mul_313: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_20, sub_88);  div_20 = sub_88 = None
        mul_314: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_293, mul_16);  mul_16 = None
        sum_127: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_314, [0, 1]);  mul_314 = None
        sum_128: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_293, [0, 1]);  view_293 = None
        add_133: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_130, mul_313);  add_130 = mul_313 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_294: "f32[32768, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(add_133, [32768, 768])
        permute_16: "f32[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(primals_33, [1, 0]);  primals_33 = None
        permute_274: "f32[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_16, [1, 0]);  permute_16 = None
        mm_87: "f32[32768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_294, permute_274);  permute_274 = None
        permute_275: "f32[768, 32768][1, 768]cuda:0" = torch.ops.aten.permute.default(view_294, [1, 0])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_15: "f32[128, 256, 12, 64][196608, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_27, [0, 2, 1, 3])
        view_24: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_15, [128, 256, 768]);  permute_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_25: "f32[32768, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_24, [32768, 768]);  view_24 = None
        mm_88: "f32[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_275, view_25);  permute_275 = view_25 = None
        sum_129: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_294, [0], True);  view_294 = None
        view_295: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_129, [768]);  sum_129 = None
        view_296: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_87, [128, 256, 768]);  mm_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        view_297: "f32[128, 256, 12, 64][196608, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_296, [128, 256, 12, 64]);  view_296 = None
        permute_278: "f32[128, 12, 256, 64][196608, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_297, [0, 2, 1, 3]);  view_297 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:120 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_backward_10 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_278, getitem_24, getitem_25, getitem_26, None, getitem_27, getitem_28, getitem_29, getitem_30, 0.0, [True, True, True, False]);  permute_278 = getitem_24 = getitem_25 = getitem_26 = getitem_27 = getitem_28 = getitem_29 = getitem_30 = None
        getitem_182: "f32[128, 12, 256, 64][196608, 64, 768, 1]cuda:0" = _scaled_dot_product_efficient_attention_backward_10[0]
        getitem_183: "f32[128, 12, 256, 64][196608, 64, 768, 1]cuda:0" = _scaled_dot_product_efficient_attention_backward_10[1]
        getitem_184: "f32[128, 12, 256, 64][196608, 64, 768, 1]cuda:0" = _scaled_dot_product_efficient_attention_backward_10[2];  _scaled_dot_product_efficient_attention_backward_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        cat_10: "f32[384, 12, 256, 64][196608, 16384, 64, 1]cuda:0" = torch.ops.aten.cat.default([getitem_182, getitem_183, getitem_184]);  getitem_182 = getitem_183 = getitem_184 = None
        view_298: "f32[3, 128, 12, 256, 64][25165824, 196608, 16384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(cat_10, [3, 128, 12, 256, 64]);  cat_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        permute_279: "f32[128, 256, 3, 12, 64][196608, 64, 25165824, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_298, [1, 3, 0, 2, 4]);  view_298 = None
        clone_51: "f32[128, 256, 3, 12, 64][589824, 2304, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_279, memory_format = torch.contiguous_format);  permute_279 = None
        view_299: "f32[128, 256, 2304][589824, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(clone_51, [128, 256, 2304]);  clone_51 = None
        view_300: "f32[32768, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(view_299, [32768, 2304]);  view_299 = None
        permute_13: "f32[768, 2304][1, 768]cuda:0" = torch.ops.aten.permute.default(primals_31, [1, 0]);  primals_31 = None
        permute_280: "f32[2304, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_13, [1, 0]);  permute_13 = None
        mm_89: "f32[32768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_300, permute_280);  permute_280 = None
        permute_281: "f32[2304, 32768][1, 2304]cuda:0" = torch.ops.aten.permute.default(view_300, [1, 0])
        mm_90: "f32[2304, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_281, view_21);  permute_281 = view_21 = None
        sum_130: "f32[1, 2304][2304, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_300, [0], True);  view_300 = None
        view_301: "f32[2304][1]cuda:0" = torch.ops.aten.reshape.default(sum_130, [2304]);  sum_130 = None
        view_302: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_89, [128, 256, 768]);  mm_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_316: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_302, primals_29);  primals_29 = None
        mul_317: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_316, 768)
        sum_131: "f32[128, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_316, [2], True)
        mul_318: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_316, mul_14);  mul_316 = None
        sum_132: "f32[128, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_318, [2], True);  mul_318 = None
        mul_319: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_14, sum_132);  sum_132 = None
        sub_90: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_317, sum_131);  mul_317 = sum_131 = None
        sub_91: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_90, mul_319);  sub_90 = mul_319 = None
        mul_320: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_21, sub_91);  div_21 = sub_91 = None
        mul_321: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_302, mul_14);  mul_14 = None
        sum_133: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_321, [0, 1]);  mul_321 = None
        sum_134: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_302, [0, 1]);  view_302 = None
        add_134: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_133, mul_320);  add_133 = mul_320 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_303: "f32[32768, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(add_134, [32768, 768])
        permute_12: "f32[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(primals_27, [1, 0]);  primals_27 = None
        permute_284: "f32[768, 3072][3072, 1]cuda:0" = torch.ops.aten.permute.default(permute_12, [1, 0]);  permute_12 = None
        mm_91: "f32[32768, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(view_303, permute_284);  permute_284 = None
        permute_285: "f32[768, 32768][1, 768]cuda:0" = torch.ops.aten.permute.default(view_303, [1, 0])
        mm_92: "f32[768, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(permute_285, view_19);  permute_285 = view_19 = None
        sum_135: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_303, [0], True);  view_303 = None
        view_304: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_135, [768]);  sum_135 = None
        view_305: "f32[128, 256, 3072][786432, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(mm_91, [128, 256, 3072]);  mm_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_18: "f32[128, 256, 3072][786432, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_6, [128, 256, 3072]);  addmm_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        mul_12: "f32[128, 256, 3072][786432, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_18, 0.7071067811865476)
        erf_1: "f32[128, 256, 3072][786432, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_12);  mul_12 = None
        add_13: "f32[128, 256, 3072][786432, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
        mul_323: "f32[128, 256, 3072][786432, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_13, 0.5);  add_13 = None
        mul_324: "f32[128, 256, 3072][786432, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_18, view_18)
        mul_325: "f32[128, 256, 3072][786432, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_324, -0.5);  mul_324 = None
        exp_11: "f32[128, 256, 3072][786432, 3072, 1]cuda:0" = torch.ops.aten.exp.default(mul_325);  mul_325 = None
        mul_326: "f32[128, 256, 3072][786432, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_11, 0.3989422804014327);  exp_11 = None
        mul_327: "f32[128, 256, 3072][786432, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_18, mul_326);  view_18 = mul_326 = None
        add_136: "f32[128, 256, 3072][786432, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_323, mul_327);  mul_323 = mul_327 = None
        mul_328: "f32[128, 256, 3072][786432, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_305, add_136);  view_305 = add_136 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_306: "f32[32768, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(mul_328, [32768, 3072]);  mul_328 = None
        permute_11: "f32[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(primals_25, [1, 0]);  primals_25 = None
        permute_288: "f32[3072, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_11, [1, 0]);  permute_11 = None
        mm_93: "f32[32768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_306, permute_288);  permute_288 = None
        permute_289: "f32[3072, 32768][1, 3072]cuda:0" = torch.ops.aten.permute.default(view_306, [1, 0])
        mm_94: "f32[3072, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_289, view_17);  permute_289 = view_17 = None
        sum_136: "f32[1, 3072][3072, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_306, [0], True);  view_306 = None
        view_307: "f32[3072][1]cuda:0" = torch.ops.aten.reshape.default(sum_136, [3072]);  sum_136 = None
        view_308: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_93, [128, 256, 768]);  mm_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_330: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_308, primals_23);  primals_23 = None
        mul_331: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_330, 768)
        sum_137: "f32[128, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_330, [2], True)
        mul_332: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_330, mul_9);  mul_330 = None
        sum_138: "f32[128, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_332, [2], True);  mul_332 = None
        mul_333: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_9, sum_138);  sum_138 = None
        sub_93: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_331, sum_137);  mul_331 = sum_137 = None
        sub_94: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_93, mul_333);  sub_93 = mul_333 = None
        mul_334: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_22, sub_94);  div_22 = sub_94 = None
        mul_335: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_308, mul_9);  mul_9 = None
        sum_139: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_335, [0, 1]);  mul_335 = None
        sum_140: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_308, [0, 1]);  view_308 = None
        add_137: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_134, mul_334);  add_134 = mul_334 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_309: "f32[32768, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(add_137, [32768, 768])
        permute_10: "f32[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(primals_21, [1, 0]);  primals_21 = None
        permute_292: "f32[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_10, [1, 0]);  permute_10 = None
        mm_95: "f32[32768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_309, permute_292);  permute_292 = None
        permute_293: "f32[768, 32768][1, 768]cuda:0" = torch.ops.aten.permute.default(view_309, [1, 0])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_9: "f32[128, 256, 12, 64][196608, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_16, [0, 2, 1, 3])
        view_14: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_9, [128, 256, 768]);  permute_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_15: "f32[32768, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_14, [32768, 768]);  view_14 = None
        mm_96: "f32[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_293, view_15);  permute_293 = view_15 = None
        sum_141: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_309, [0], True);  view_309 = None
        view_310: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_141, [768]);  sum_141 = None
        view_311: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_95, [128, 256, 768]);  mm_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        view_312: "f32[128, 256, 12, 64][196608, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_311, [128, 256, 12, 64]);  view_311 = None
        permute_296: "f32[128, 12, 256, 64][196608, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_312, [0, 2, 1, 3]);  view_312 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:120 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_backward_11 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_296, getitem_13, getitem_14, getitem_15, None, getitem_16, getitem_17, getitem_18, getitem_19, 0.0, [True, True, True, False]);  permute_296 = getitem_13 = getitem_14 = getitem_15 = getitem_16 = getitem_17 = getitem_18 = getitem_19 = None
        getitem_186: "f32[128, 12, 256, 64][196608, 64, 768, 1]cuda:0" = _scaled_dot_product_efficient_attention_backward_11[0]
        getitem_187: "f32[128, 12, 256, 64][196608, 64, 768, 1]cuda:0" = _scaled_dot_product_efficient_attention_backward_11[1]
        getitem_188: "f32[128, 12, 256, 64][196608, 64, 768, 1]cuda:0" = _scaled_dot_product_efficient_attention_backward_11[2];  _scaled_dot_product_efficient_attention_backward_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        cat_11: "f32[384, 12, 256, 64][196608, 16384, 64, 1]cuda:0" = torch.ops.aten.cat.default([getitem_186, getitem_187, getitem_188]);  getitem_186 = getitem_187 = getitem_188 = None
        view_313: "f32[3, 128, 12, 256, 64][25165824, 196608, 16384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(cat_11, [3, 128, 12, 256, 64]);  cat_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        permute_297: "f32[128, 256, 3, 12, 64][196608, 64, 25165824, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_313, [1, 3, 0, 2, 4]);  view_313 = None
        clone_52: "f32[128, 256, 3, 12, 64][589824, 2304, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_297, memory_format = torch.contiguous_format);  permute_297 = None
        view_314: "f32[128, 256, 2304][589824, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(clone_52, [128, 256, 2304]);  clone_52 = None
        view_315: "f32[32768, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(view_314, [32768, 2304]);  view_314 = None
        permute_7: "f32[768, 2304][1, 768]cuda:0" = torch.ops.aten.permute.default(primals_19, [1, 0]);  primals_19 = None
        permute_298: "f32[2304, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_7, [1, 0]);  permute_7 = None
        mm_97: "f32[32768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_315, permute_298);  permute_298 = None
        permute_299: "f32[2304, 32768][1, 2304]cuda:0" = torch.ops.aten.permute.default(view_315, [1, 0])
        mm_98: "f32[2304, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_299, view_11);  permute_299 = view_11 = None
        sum_142: "f32[1, 2304][2304, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_315, [0], True);  view_315 = None
        view_316: "f32[2304][1]cuda:0" = torch.ops.aten.reshape.default(sum_142, [2304]);  sum_142 = None
        view_317: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_97, [128, 256, 768]);  mm_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_337: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_317, primals_17);  primals_17 = None
        mul_338: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_337, 768)
        sum_143: "f32[128, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_337, [2], True)
        mul_339: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_337, mul_7);  mul_337 = None
        sum_144: "f32[128, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_339, [2], True);  mul_339 = None
        mul_340: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_7, sum_144);  sum_144 = None
        sub_96: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_338, sum_143);  mul_338 = sum_143 = None
        sub_97: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_96, mul_340);  sub_96 = mul_340 = None
        mul_341: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_23, sub_97);  div_23 = sub_97 = None
        mul_342: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_317, mul_7);  mul_7 = None
        sum_145: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_342, [0, 1]);  mul_342 = None
        sum_146: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_317, [0, 1]);  view_317 = None
        add_138: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_137, mul_341);  add_137 = mul_341 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_318: "f32[32768, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(add_138, [32768, 768])
        permute_6: "f32[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(primals_15, [1, 0]);  primals_15 = None
        permute_302: "f32[768, 3072][3072, 1]cuda:0" = torch.ops.aten.permute.default(permute_6, [1, 0]);  permute_6 = None
        mm_99: "f32[32768, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(view_318, permute_302);  permute_302 = None
        permute_303: "f32[768, 32768][1, 768]cuda:0" = torch.ops.aten.permute.default(view_318, [1, 0])
        mm_100: "f32[768, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(permute_303, view_9);  permute_303 = view_9 = None
        sum_147: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_318, [0], True);  view_318 = None
        view_319: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_147, [768]);  sum_147 = None
        view_320: "f32[128, 256, 3072][786432, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(mm_99, [128, 256, 3072]);  mm_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_8: "f32[128, 256, 3072][786432, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_2, [128, 256, 3072]);  addmm_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        mul_5: "f32[128, 256, 3072][786432, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_8, 0.7071067811865476)
        erf: "f32[128, 256, 3072][786432, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_5);  mul_5 = None
        add_6: "f32[128, 256, 3072][786432, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_344: "f32[128, 256, 3072][786432, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_6, 0.5);  add_6 = None
        mul_345: "f32[128, 256, 3072][786432, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_8, view_8)
        mul_346: "f32[128, 256, 3072][786432, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_345, -0.5);  mul_345 = None
        exp_12: "f32[128, 256, 3072][786432, 3072, 1]cuda:0" = torch.ops.aten.exp.default(mul_346);  mul_346 = None
        mul_347: "f32[128, 256, 3072][786432, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_12, 0.3989422804014327);  exp_12 = None
        mul_348: "f32[128, 256, 3072][786432, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_8, mul_347);  view_8 = mul_347 = None
        add_140: "f32[128, 256, 3072][786432, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_344, mul_348);  mul_344 = mul_348 = None
        mul_349: "f32[128, 256, 3072][786432, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_320, add_140);  view_320 = add_140 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_321: "f32[32768, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(mul_349, [32768, 3072]);  mul_349 = None
        permute_5: "f32[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(primals_13, [1, 0]);  primals_13 = None
        permute_306: "f32[3072, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_5, [1, 0]);  permute_5 = None
        mm_101: "f32[32768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_321, permute_306);  permute_306 = None
        permute_307: "f32[3072, 32768][1, 3072]cuda:0" = torch.ops.aten.permute.default(view_321, [1, 0])
        mm_102: "f32[3072, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_307, view_7);  permute_307 = view_7 = None
        sum_148: "f32[1, 3072][3072, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_321, [0], True);  view_321 = None
        view_322: "f32[3072][1]cuda:0" = torch.ops.aten.reshape.default(sum_148, [3072]);  sum_148 = None
        view_323: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_101, [128, 256, 768]);  mm_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_351: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_323, primals_11);  primals_11 = None
        mul_352: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_351, 768)
        sum_149: "f32[128, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_351, [2], True)
        mul_353: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_351, mul_2);  mul_351 = None
        sum_150: "f32[128, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_353, [2], True);  mul_353 = None
        mul_354: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2, sum_150);  sum_150 = None
        sub_99: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_352, sum_149);  mul_352 = sum_149 = None
        sub_100: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_99, mul_354);  sub_99 = mul_354 = None
        mul_355: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_24, sub_100);  div_24 = sub_100 = None
        mul_356: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_323, mul_2);  mul_2 = None
        sum_151: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_356, [0, 1]);  mul_356 = None
        sum_152: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_323, [0, 1]);  view_323 = None
        add_141: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_138, mul_355);  add_138 = mul_355 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_324: "f32[32768, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(add_141, [32768, 768])
        permute_4: "f32[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(primals_9, [1, 0]);  primals_9 = None
        permute_310: "f32[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_4, [1, 0]);  permute_4 = None
        mm_103: "f32[32768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_324, permute_310);  permute_310 = None
        permute_311: "f32[768, 32768][1, 768]cuda:0" = torch.ops.aten.permute.default(view_324, [1, 0])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_3: "f32[128, 256, 12, 64][196608, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_5, [0, 2, 1, 3])
        view_4: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_3, [128, 256, 768]);  permute_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_5: "f32[32768, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_4, [32768, 768]);  view_4 = None
        mm_104: "f32[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_311, view_5);  permute_311 = view_5 = None
        sum_153: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_324, [0], True);  view_324 = None
        view_325: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_153, [768]);  sum_153 = None
        view_326: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_103, [128, 256, 768]);  mm_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        view_327: "f32[128, 256, 12, 64][196608, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_326, [128, 256, 12, 64]);  view_326 = None
        permute_314: "f32[128, 12, 256, 64][196608, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_327, [0, 2, 1, 3]);  view_327 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:120 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_backward_12 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_314, getitem_2, getitem_3, getitem_4, None, getitem_5, getitem_6, getitem_7, getitem_8, 0.0, [True, True, True, False]);  permute_314 = getitem_2 = getitem_3 = getitem_4 = getitem_5 = getitem_6 = getitem_7 = getitem_8 = None
        getitem_190: "f32[128, 12, 256, 64][196608, 64, 768, 1]cuda:0" = _scaled_dot_product_efficient_attention_backward_12[0]
        getitem_191: "f32[128, 12, 256, 64][196608, 64, 768, 1]cuda:0" = _scaled_dot_product_efficient_attention_backward_12[1]
        getitem_192: "f32[128, 12, 256, 64][196608, 64, 768, 1]cuda:0" = _scaled_dot_product_efficient_attention_backward_12[2];  _scaled_dot_product_efficient_attention_backward_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        cat_12: "f32[384, 12, 256, 64][196608, 16384, 64, 1]cuda:0" = torch.ops.aten.cat.default([getitem_190, getitem_191, getitem_192]);  getitem_190 = getitem_191 = getitem_192 = None
        view_328: "f32[3, 128, 12, 256, 64][25165824, 196608, 16384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(cat_12, [3, 128, 12, 256, 64]);  cat_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        permute_315: "f32[128, 256, 3, 12, 64][196608, 64, 25165824, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_328, [1, 3, 0, 2, 4]);  view_328 = None
        clone_53: "f32[128, 256, 3, 12, 64][589824, 2304, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_315, memory_format = torch.contiguous_format);  permute_315 = None
        view_329: "f32[128, 256, 2304][589824, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(clone_53, [128, 256, 2304]);  clone_53 = None
        view_330: "f32[32768, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(view_329, [32768, 2304]);  view_329 = None
        permute_1: "f32[768, 2304][1, 768]cuda:0" = torch.ops.aten.permute.default(primals_7, [1, 0]);  primals_7 = None
        permute_316: "f32[2304, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_1, [1, 0]);  permute_1 = None
        mm_105: "f32[32768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_330, permute_316);  permute_316 = None
        permute_317: "f32[2304, 32768][1, 2304]cuda:0" = torch.ops.aten.permute.default(view_330, [1, 0])
        mm_106: "f32[2304, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_317, view_1);  permute_317 = view_1 = None
        sum_154: "f32[1, 2304][2304, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_330, [0], True);  view_330 = None
        view_331: "f32[2304][1]cuda:0" = torch.ops.aten.reshape.default(sum_154, [2304]);  sum_154 = None
        view_332: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_105, [128, 256, 768]);  mm_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_358: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_332, primals_5);  primals_5 = None
        mul_359: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_358, 768)
        sum_155: "f32[128, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_358, [2], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/patch_embed.py:138 in forward, code: x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
        view: "f32[128, 768, 256][196608, 1, 768]cuda:0" = torch.ops.aten.reshape.default(convolution, [128, 768, 256]);  convolution = None
        permute: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.permute.default(view, [0, 2, 1]);  view = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:1073 in _pos_embed, code: x = x + pos_embed
        add: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(permute, primals_4);  permute = primals_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        sub: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add, getitem_1);  add = getitem_1 = None
        mul: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        mul_360: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_358, mul);  mul_358 = None
        sum_156: "f32[128, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_360, [2], True);  mul_360 = None
        mul_361: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul, sum_156);  sum_156 = None
        sub_102: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_359, sum_155);  mul_359 = sum_155 = None
        sub_103: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_102, mul_361);  sub_102 = mul_361 = None
        div_25: "f32[128, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt, 768);  rsqrt = None
        mul_362: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_25, sub_103);  div_25 = sub_103 = None
        mul_363: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_332, mul);  mul = None
        sum_157: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_363, [0, 1]);  mul_363 = None
        sum_158: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_332, [0, 1]);  view_332 = None
        add_142: "f32[128, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_141, mul_362);  add_141 = mul_362 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:1073 in _pos_embed, code: x = x + pos_embed
        sum_159: "f32[1, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_142, [0], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/patch_embed.py:138 in forward, code: x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
        permute_320: "f32[128, 768, 256][196608, 1, 768]cuda:0" = torch.ops.aten.permute.default(add_142, [0, 2, 1]);  add_142 = None
        view_333: "f32[128, 768, 16, 16][196608, 1, 12288, 768]cuda:0" = torch.ops.aten.reshape.default(permute_320, [128, 768, 16, 16]);  permute_320 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/patch_embed.py:136 in forward, code: x = self.proj(x)
        sum_160: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_333, [0, 2, 3])
        convolution_backward = torch.ops.aten.convolution_backward.default(view_333, primals_1, primals_2, [768], [16, 16], [0, 0], [1, 1], False, [0, 0], 1, [False, True, False]);  view_333 = primals_1 = primals_2 = None
        getitem_195: "f32[768, 3, 16, 16][768, 1, 48, 3]cuda:0" = convolution_backward[1];  convolution_backward = None
        return (None, getitem_195, sum_160, sum_159, sum_157, sum_158, mm_106, view_331, mm_104, view_325, sum_151, sum_152, mm_102, view_322, mm_100, view_319, sum_145, sum_146, mm_98, view_316, mm_96, view_310, sum_139, sum_140, mm_94, view_307, mm_92, view_304, sum_133, sum_134, mm_90, view_301, mm_88, view_295, sum_127, sum_128, mm_86, view_292, mm_84, view_289, sum_121, sum_122, mm_82, view_286, mm_80, view_280, sum_115, sum_116, mm_78, view_277, mm_76, view_274, sum_109, sum_110, mm_74, view_271, mm_72, view_265, sum_103, sum_104, mm_70, view_262, mm_68, view_259, sum_97, sum_98, mm_66, view_256, mm_64, view_250, sum_91, sum_92, mm_62, view_247, mm_60, view_244, sum_85, sum_86, mm_58, view_241, mm_56, view_235, sum_79, sum_80, mm_54, view_232, mm_52, view_229, sum_73, sum_74, mm_50, view_226, mm_48, view_220, sum_67, sum_68, mm_46, view_217, mm_44, view_214, sum_61, sum_62, mm_42, view_211, mm_40, view_205, sum_55, sum_56, mm_38, view_202, mm_36, view_199, sum_49, sum_50, mm_34, view_196, mm_32, view_190, sum_43, sum_44, mm_30, view_187, mm_28, view_184, sum_37, sum_38, mm_26, view_181, mm_24, view_175, sum_31, sum_32, mm_22, view_172, mm_20, view_169, sum_25, sum_26, mm_18, view_166, mm_16, view_160, sum_19, sum_20, mm_14, view_157, mm_12, view_154, sum_13, sum_14, sum_10, mm_9, view_150, mm_8, view_147, mm_6, view_141, sum_5, sum_6, mm_4, view_138, mm_2, view_135)
