"""
Standalone repro captured via capture_hook.
Label: torchbench_shufflenet_v2_x1_0_train
Pattern hash: 3a41548e8ecb
Shape hash: ea21dd5b
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
    def forward(self, tangents_1: "f32[512, 1000]", sum_3: "f32[1024]", squeeze_166: "f32[1024]", sum_5: "f32[232]", squeeze_163: "f32[232]", sum_7: "f32[232]", squeeze_160: "f32[232]", sum_9: "f32[232]", squeeze_157: "f32[232]", sum_11: "f32[232]", squeeze_154: "f32[232]", sum_13: "f32[232]", squeeze_151: "f32[232]", sum_15: "f32[232]", squeeze_148: "f32[232]", sum_17: "f32[232]", squeeze_145: "f32[232]", sum_19: "f32[232]", squeeze_142: "f32[232]", sum_21: "f32[232]", squeeze_139: "f32[232]", sum_23: "f32[232]", squeeze_136: "f32[232]", sum_25: "f32[232]", squeeze_133: "f32[232]", sum_27: "f32[232]", squeeze_130: "f32[232]", sum_29: "f32[232]", squeeze_127: "f32[232]", sum_31: "f32[232]", squeeze_124: "f32[232]", sum_33: "f32[116]", squeeze_121: "f32[116]", sum_35: "f32[116]", squeeze_118: "f32[116]", sum_37: "f32[116]", squeeze_115: "f32[116]", sum_39: "f32[116]", squeeze_112: "f32[116]", sum_41: "f32[116]", squeeze_109: "f32[116]", sum_43: "f32[116]", squeeze_106: "f32[116]", sum_45: "f32[116]", squeeze_103: "f32[116]", sum_47: "f32[116]", squeeze_100: "f32[116]", sum_49: "f32[116]", squeeze_97: "f32[116]", sum_51: "f32[116]", squeeze_94: "f32[116]", sum_53: "f32[116]", squeeze_91: "f32[116]", sum_55: "f32[116]", squeeze_88: "f32[116]", sum_57: "f32[116]", squeeze_85: "f32[116]", sum_59: "f32[116]", squeeze_82: "f32[116]", sum_61: "f32[116]", squeeze_79: "f32[116]", sum_63: "f32[116]", squeeze_76: "f32[116]", sum_65: "f32[116]", squeeze_73: "f32[116]", sum_67: "f32[116]", squeeze_70: "f32[116]", sum_69: "f32[116]", squeeze_67: "f32[116]", sum_71: "f32[116]", squeeze_64: "f32[116]", sum_73: "f32[116]", squeeze_61: "f32[116]", sum_75: "f32[116]", squeeze_58: "f32[116]", sum_77: "f32[116]", squeeze_55: "f32[116]", sum_79: "f32[116]", squeeze_52: "f32[116]", sum_81: "f32[116]", squeeze_49: "f32[116]", sum_83: "f32[116]", squeeze_46: "f32[116]", sum_85: "f32[58]", squeeze_43: "f32[58]", sum_87: "f32[58]", squeeze_40: "f32[58]", sum_89: "f32[58]", squeeze_37: "f32[58]", sum_91: "f32[58]", squeeze_34: "f32[58]", sum_93: "f32[58]", squeeze_31: "f32[58]", sum_95: "f32[58]", squeeze_28: "f32[58]", sum_97: "f32[58]", squeeze_25: "f32[58]", sum_99: "f32[58]", squeeze_22: "f32[58]", sum_101: "f32[58]", squeeze_19: "f32[58]", sum_103: "f32[58]", squeeze_16: "f32[58]", sum_105: "f32[58]", squeeze_13: "f32[58]", sum_107: "f32[58]", squeeze_10: "f32[58]", sum_109: "f32[58]", squeeze_7: "f32[58]", sum_111: "f32[24]", squeeze_4: "f32[24]", sum_113: "f32[24]", squeeze_1: "f32[24]", _shape_param_0):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:162 in _forward_impl, code: x = self.fc(x)
        permute_default: "f32[1000, 512]" = torch.ops.aten.permute.default(tangents_1, [1, 0])
        sum_dim_int_list: "f32[1, 1000]" = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True);  tangents_1 = None
        reshape_default: "f32[1000]" = torch.ops.aten.reshape.default(sum_dim_int_list, _shape_param_0);  sum_dim_int_list = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:160 in _forward_impl, code: x = self.conv5(x)
        mul_tensor: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_3, squeeze_166);  sum_3 = squeeze_166 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:95 in forward, code: out = torch.cat((x1, self.branch2(x2)), dim=1)
        mul_tensor_1: "f32[232]" = torch.ops.aten.mul.Tensor(sum_5, squeeze_163);  sum_5 = squeeze_163 = None
        mul_tensor_2: "f32[232]" = torch.ops.aten.mul.Tensor(sum_7, squeeze_160);  sum_7 = squeeze_160 = None
        mul_tensor_3: "f32[232]" = torch.ops.aten.mul.Tensor(sum_9, squeeze_157);  sum_9 = squeeze_157 = None
        mul_tensor_4: "f32[232]" = torch.ops.aten.mul.Tensor(sum_11, squeeze_154);  sum_11 = squeeze_154 = None
        mul_tensor_5: "f32[232]" = torch.ops.aten.mul.Tensor(sum_13, squeeze_151);  sum_13 = squeeze_151 = None
        mul_tensor_6: "f32[232]" = torch.ops.aten.mul.Tensor(sum_15, squeeze_148);  sum_15 = squeeze_148 = None
        mul_tensor_7: "f32[232]" = torch.ops.aten.mul.Tensor(sum_17, squeeze_145);  sum_17 = squeeze_145 = None
        mul_tensor_8: "f32[232]" = torch.ops.aten.mul.Tensor(sum_19, squeeze_142);  sum_19 = squeeze_142 = None
        mul_tensor_9: "f32[232]" = torch.ops.aten.mul.Tensor(sum_21, squeeze_139);  sum_21 = squeeze_139 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:97 in forward, code: out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)
        mul_tensor_10: "f32[232]" = torch.ops.aten.mul.Tensor(sum_23, squeeze_136);  sum_23 = squeeze_136 = None
        mul_tensor_11: "f32[232]" = torch.ops.aten.mul.Tensor(sum_25, squeeze_133);  sum_25 = squeeze_133 = None
        mul_tensor_12: "f32[232]" = torch.ops.aten.mul.Tensor(sum_27, squeeze_130);  sum_27 = squeeze_130 = None
        mul_tensor_13: "f32[232]" = torch.ops.aten.mul.Tensor(sum_29, squeeze_127);  sum_29 = squeeze_127 = None
        mul_tensor_14: "f32[232]" = torch.ops.aten.mul.Tensor(sum_31, squeeze_124);  sum_31 = squeeze_124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:95 in forward, code: out = torch.cat((x1, self.branch2(x2)), dim=1)
        mul_tensor_15: "f32[116]" = torch.ops.aten.mul.Tensor(sum_33, squeeze_121);  sum_33 = squeeze_121 = None
        mul_tensor_16: "f32[116]" = torch.ops.aten.mul.Tensor(sum_35, squeeze_118);  sum_35 = squeeze_118 = None
        mul_tensor_17: "f32[116]" = torch.ops.aten.mul.Tensor(sum_37, squeeze_115);  sum_37 = squeeze_115 = None
        mul_tensor_18: "f32[116]" = torch.ops.aten.mul.Tensor(sum_39, squeeze_112);  sum_39 = squeeze_112 = None
        mul_tensor_19: "f32[116]" = torch.ops.aten.mul.Tensor(sum_41, squeeze_109);  sum_41 = squeeze_109 = None
        mul_tensor_20: "f32[116]" = torch.ops.aten.mul.Tensor(sum_43, squeeze_106);  sum_43 = squeeze_106 = None
        mul_tensor_21: "f32[116]" = torch.ops.aten.mul.Tensor(sum_45, squeeze_103);  sum_45 = squeeze_103 = None
        mul_tensor_22: "f32[116]" = torch.ops.aten.mul.Tensor(sum_47, squeeze_100);  sum_47 = squeeze_100 = None
        mul_tensor_23: "f32[116]" = torch.ops.aten.mul.Tensor(sum_49, squeeze_97);  sum_49 = squeeze_97 = None
        mul_tensor_24: "f32[116]" = torch.ops.aten.mul.Tensor(sum_51, squeeze_94);  sum_51 = squeeze_94 = None
        mul_tensor_25: "f32[116]" = torch.ops.aten.mul.Tensor(sum_53, squeeze_91);  sum_53 = squeeze_91 = None
        mul_tensor_26: "f32[116]" = torch.ops.aten.mul.Tensor(sum_55, squeeze_88);  sum_55 = squeeze_88 = None
        mul_tensor_27: "f32[116]" = torch.ops.aten.mul.Tensor(sum_57, squeeze_85);  sum_57 = squeeze_85 = None
        mul_tensor_28: "f32[116]" = torch.ops.aten.mul.Tensor(sum_59, squeeze_82);  sum_59 = squeeze_82 = None
        mul_tensor_29: "f32[116]" = torch.ops.aten.mul.Tensor(sum_61, squeeze_79);  sum_61 = squeeze_79 = None
        mul_tensor_30: "f32[116]" = torch.ops.aten.mul.Tensor(sum_63, squeeze_76);  sum_63 = squeeze_76 = None
        mul_tensor_31: "f32[116]" = torch.ops.aten.mul.Tensor(sum_65, squeeze_73);  sum_65 = squeeze_73 = None
        mul_tensor_32: "f32[116]" = torch.ops.aten.mul.Tensor(sum_67, squeeze_70);  sum_67 = squeeze_70 = None
        mul_tensor_33: "f32[116]" = torch.ops.aten.mul.Tensor(sum_69, squeeze_67);  sum_69 = squeeze_67 = None
        mul_tensor_34: "f32[116]" = torch.ops.aten.mul.Tensor(sum_71, squeeze_64);  sum_71 = squeeze_64 = None
        mul_tensor_35: "f32[116]" = torch.ops.aten.mul.Tensor(sum_73, squeeze_61);  sum_73 = squeeze_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:97 in forward, code: out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)
        mul_tensor_36: "f32[116]" = torch.ops.aten.mul.Tensor(sum_75, squeeze_58);  sum_75 = squeeze_58 = None
        mul_tensor_37: "f32[116]" = torch.ops.aten.mul.Tensor(sum_77, squeeze_55);  sum_77 = squeeze_55 = None
        mul_tensor_38: "f32[116]" = torch.ops.aten.mul.Tensor(sum_79, squeeze_52);  sum_79 = squeeze_52 = None
        mul_tensor_39: "f32[116]" = torch.ops.aten.mul.Tensor(sum_81, squeeze_49);  sum_81 = squeeze_49 = None
        mul_tensor_40: "f32[116]" = torch.ops.aten.mul.Tensor(sum_83, squeeze_46);  sum_83 = squeeze_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:95 in forward, code: out = torch.cat((x1, self.branch2(x2)), dim=1)
        mul_tensor_41: "f32[58]" = torch.ops.aten.mul.Tensor(sum_85, squeeze_43);  sum_85 = squeeze_43 = None
        mul_tensor_42: "f32[58]" = torch.ops.aten.mul.Tensor(sum_87, squeeze_40);  sum_87 = squeeze_40 = None
        mul_tensor_43: "f32[58]" = torch.ops.aten.mul.Tensor(sum_89, squeeze_37);  sum_89 = squeeze_37 = None
        mul_tensor_44: "f32[58]" = torch.ops.aten.mul.Tensor(sum_91, squeeze_34);  sum_91 = squeeze_34 = None
        mul_tensor_45: "f32[58]" = torch.ops.aten.mul.Tensor(sum_93, squeeze_31);  sum_93 = squeeze_31 = None
        mul_tensor_46: "f32[58]" = torch.ops.aten.mul.Tensor(sum_95, squeeze_28);  sum_95 = squeeze_28 = None
        mul_tensor_47: "f32[58]" = torch.ops.aten.mul.Tensor(sum_97, squeeze_25);  sum_97 = squeeze_25 = None
        mul_tensor_48: "f32[58]" = torch.ops.aten.mul.Tensor(sum_99, squeeze_22);  sum_99 = squeeze_22 = None
        mul_tensor_49: "f32[58]" = torch.ops.aten.mul.Tensor(sum_101, squeeze_19);  sum_101 = squeeze_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:97 in forward, code: out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)
        mul_tensor_50: "f32[58]" = torch.ops.aten.mul.Tensor(sum_103, squeeze_16);  sum_103 = squeeze_16 = None
        mul_tensor_51: "f32[58]" = torch.ops.aten.mul.Tensor(sum_105, squeeze_13);  sum_105 = squeeze_13 = None
        mul_tensor_52: "f32[58]" = torch.ops.aten.mul.Tensor(sum_107, squeeze_10);  sum_107 = squeeze_10 = None
        mul_tensor_53: "f32[58]" = torch.ops.aten.mul.Tensor(sum_109, squeeze_7);  sum_109 = squeeze_7 = None
        mul_tensor_54: "f32[24]" = torch.ops.aten.mul.Tensor(sum_111, squeeze_4);  sum_111 = squeeze_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:155 in _forward_impl, code: x = self.conv1(x)
        mul_tensor_55: "f32[24]" = torch.ops.aten.mul.Tensor(sum_113, squeeze_1);  sum_113 = squeeze_1 = None
        return (permute_default, reshape_default, mul_tensor, mul_tensor_1, mul_tensor_2, mul_tensor_3, mul_tensor_4, mul_tensor_5, mul_tensor_6, mul_tensor_7, mul_tensor_8, mul_tensor_9, mul_tensor_10, mul_tensor_11, mul_tensor_12, mul_tensor_13, mul_tensor_14, mul_tensor_15, mul_tensor_16, mul_tensor_17, mul_tensor_18, mul_tensor_19, mul_tensor_20, mul_tensor_21, mul_tensor_22, mul_tensor_23, mul_tensor_24, mul_tensor_25, mul_tensor_26, mul_tensor_27, mul_tensor_28, mul_tensor_29, mul_tensor_30, mul_tensor_31, mul_tensor_32, mul_tensor_33, mul_tensor_34, mul_tensor_35, mul_tensor_36, mul_tensor_37, mul_tensor_38, mul_tensor_39, mul_tensor_40, mul_tensor_41, mul_tensor_42, mul_tensor_43, mul_tensor_44, mul_tensor_45, mul_tensor_46, mul_tensor_47, mul_tensor_48, mul_tensor_49, mul_tensor_50, mul_tensor_51, mul_tensor_52, mul_tensor_53, mul_tensor_54, mul_tensor_55)


def _default_make_inputs():
    return [
    torch.randn([512, 1000], dtype=torch.float32, device='cuda'),
    torch.randn([1024], dtype=torch.float32, device='cuda'),
    torch.randn([1024], dtype=torch.float32, device='cuda'),
    torch.randn([232], dtype=torch.float32, device='cuda'),
    torch.randn([232], dtype=torch.float32, device='cuda'),
    torch.randn([232], dtype=torch.float32, device='cuda'),
    torch.randn([232], dtype=torch.float32, device='cuda'),
    torch.randn([232], dtype=torch.float32, device='cuda'),
    torch.randn([232], dtype=torch.float32, device='cuda'),
    torch.randn([232], dtype=torch.float32, device='cuda'),
    torch.randn([232], dtype=torch.float32, device='cuda'),
    torch.randn([232], dtype=torch.float32, device='cuda'),
    torch.randn([232], dtype=torch.float32, device='cuda'),
    torch.randn([232], dtype=torch.float32, device='cuda'),
    torch.randn([232], dtype=torch.float32, device='cuda'),
    torch.randn([232], dtype=torch.float32, device='cuda'),
    torch.randn([232], dtype=torch.float32, device='cuda'),
    torch.randn([232], dtype=torch.float32, device='cuda'),
    torch.randn([232], dtype=torch.float32, device='cuda'),
    torch.randn([232], dtype=torch.float32, device='cuda'),
    torch.randn([232], dtype=torch.float32, device='cuda'),
    torch.randn([232], dtype=torch.float32, device='cuda'),
    torch.randn([232], dtype=torch.float32, device='cuda'),
    torch.randn([232], dtype=torch.float32, device='cuda'),
    torch.randn([232], dtype=torch.float32, device='cuda'),
    torch.randn([232], dtype=torch.float32, device='cuda'),
    torch.randn([232], dtype=torch.float32, device='cuda'),
    torch.randn([232], dtype=torch.float32, device='cuda'),
    torch.randn([232], dtype=torch.float32, device='cuda'),
    torch.randn([232], dtype=torch.float32, device='cuda'),
    torch.randn([232], dtype=torch.float32, device='cuda'),
    torch.randn([116], dtype=torch.float32, device='cuda'),
    torch.randn([116], dtype=torch.float32, device='cuda'),
    torch.randn([116], dtype=torch.float32, device='cuda'),
    torch.randn([116], dtype=torch.float32, device='cuda'),
    torch.randn([116], dtype=torch.float32, device='cuda'),
    torch.randn([116], dtype=torch.float32, device='cuda'),
    torch.randn([116], dtype=torch.float32, device='cuda'),
    torch.randn([116], dtype=torch.float32, device='cuda'),
    torch.randn([116], dtype=torch.float32, device='cuda'),
    torch.randn([116], dtype=torch.float32, device='cuda'),
    torch.randn([116], dtype=torch.float32, device='cuda'),
    torch.randn([116], dtype=torch.float32, device='cuda'),
    torch.randn([116], dtype=torch.float32, device='cuda'),
    torch.randn([116], dtype=torch.float32, device='cuda'),
    torch.randn([116], dtype=torch.float32, device='cuda'),
    torch.randn([116], dtype=torch.float32, device='cuda'),
    torch.randn([116], dtype=torch.float32, device='cuda'),
    torch.randn([116], dtype=torch.float32, device='cuda'),
    torch.randn([116], dtype=torch.float32, device='cuda'),
    torch.randn([116], dtype=torch.float32, device='cuda'),
    torch.randn([116], dtype=torch.float32, device='cuda'),
    torch.randn([116], dtype=torch.float32, device='cuda'),
    torch.randn([116], dtype=torch.float32, device='cuda'),
    torch.randn([116], dtype=torch.float32, device='cuda'),
    torch.randn([116], dtype=torch.float32, device='cuda'),
    torch.randn([116], dtype=torch.float32, device='cuda'),
    torch.randn([116], dtype=torch.float32, device='cuda'),
    torch.randn([116], dtype=torch.float32, device='cuda'),
    torch.randn([116], dtype=torch.float32, device='cuda'),
    torch.randn([116], dtype=torch.float32, device='cuda'),
    torch.randn([116], dtype=torch.float32, device='cuda'),
    torch.randn([116], dtype=torch.float32, device='cuda'),
    torch.randn([116], dtype=torch.float32, device='cuda'),
    torch.randn([116], dtype=torch.float32, device='cuda'),
    torch.randn([116], dtype=torch.float32, device='cuda'),
    torch.randn([116], dtype=torch.float32, device='cuda'),
    torch.randn([116], dtype=torch.float32, device='cuda'),
    torch.randn([116], dtype=torch.float32, device='cuda'),
    torch.randn([116], dtype=torch.float32, device='cuda'),
    torch.randn([116], dtype=torch.float32, device='cuda'),
    torch.randn([116], dtype=torch.float32, device='cuda'),
    torch.randn([116], dtype=torch.float32, device='cuda'),
    torch.randn([116], dtype=torch.float32, device='cuda'),
    torch.randn([116], dtype=torch.float32, device='cuda'),
    torch.randn([116], dtype=torch.float32, device='cuda'),
    torch.randn([116], dtype=torch.float32, device='cuda'),
    torch.randn([116], dtype=torch.float32, device='cuda'),
    torch.randn([116], dtype=torch.float32, device='cuda'),
    torch.randn([116], dtype=torch.float32, device='cuda'),
    torch.randn([116], dtype=torch.float32, device='cuda'),
    torch.randn([116], dtype=torch.float32, device='cuda'),
    torch.randn([116], dtype=torch.float32, device='cuda'),
    torch.randn([58], dtype=torch.float32, device='cuda'),
    torch.randn([58], dtype=torch.float32, device='cuda'),
    torch.randn([58], dtype=torch.float32, device='cuda'),
    torch.randn([58], dtype=torch.float32, device='cuda'),
    torch.randn([58], dtype=torch.float32, device='cuda'),
    torch.randn([58], dtype=torch.float32, device='cuda'),
    torch.randn([58], dtype=torch.float32, device='cuda'),
    torch.randn([58], dtype=torch.float32, device='cuda'),
    torch.randn([58], dtype=torch.float32, device='cuda'),
    torch.randn([58], dtype=torch.float32, device='cuda'),
    torch.randn([58], dtype=torch.float32, device='cuda'),
    torch.randn([58], dtype=torch.float32, device='cuda'),
    torch.randn([58], dtype=torch.float32, device='cuda'),
    torch.randn([58], dtype=torch.float32, device='cuda'),
    torch.randn([58], dtype=torch.float32, device='cuda'),
    torch.randn([58], dtype=torch.float32, device='cuda'),
    torch.randn([58], dtype=torch.float32, device='cuda'),
    torch.randn([58], dtype=torch.float32, device='cuda'),
    torch.randn([58], dtype=torch.float32, device='cuda'),
    torch.randn([58], dtype=torch.float32, device='cuda'),
    torch.randn([58], dtype=torch.float32, device='cuda'),
    torch.randn([58], dtype=torch.float32, device='cuda'),
    torch.randn([58], dtype=torch.float32, device='cuda'),
    torch.randn([58], dtype=torch.float32, device='cuda'),
    torch.randn([58], dtype=torch.float32, device='cuda'),
    torch.randn([58], dtype=torch.float32, device='cuda'),
    torch.randn([24], dtype=torch.float32, device='cuda'),
    torch.randn([24], dtype=torch.float32, device='cuda'),
    torch.randn([24], dtype=torch.float32, device='cuda'),
    torch.randn([24], dtype=torch.float32, device='cuda'),
    [1000],  # _shape_param_0
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
