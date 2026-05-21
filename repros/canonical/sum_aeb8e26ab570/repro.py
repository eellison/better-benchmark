"""
Standalone repro captured via capture_hook.
Label: torchbench_resnet50_train
Pattern hash: aeb8e26ab570
Shape hash: 6f117f84
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
    def forward(self, tangents_1: "f32[128, 1000]", sum_3: "f32[2048]", squeeze_157: "f32[2048]", sum_5: "f32[512]", squeeze_154: "f32[512]", sum_7: "f32[512]", squeeze_151: "f32[512]", sum_9: "f32[2048]", squeeze_148: "f32[2048]", sum_11: "f32[512]", squeeze_145: "f32[512]", sum_13: "f32[512]", squeeze_142: "f32[512]", sum_15: "f32[2048]", squeeze_139: "f32[2048]", sum_17: "f32[2048]", squeeze_136: "f32[2048]", sum_19: "f32[512]", squeeze_133: "f32[512]", sum_21: "f32[512]", squeeze_130: "f32[512]", sum_23: "f32[1024]", squeeze_127: "f32[1024]", sum_25: "f32[256]", squeeze_124: "f32[256]", sum_27: "f32[256]", squeeze_121: "f32[256]", sum_29: "f32[1024]", squeeze_118: "f32[1024]", sum_31: "f32[256]", squeeze_115: "f32[256]", sum_33: "f32[256]", squeeze_112: "f32[256]", sum_35: "f32[1024]", squeeze_109: "f32[1024]", sum_37: "f32[256]", squeeze_106: "f32[256]", sum_39: "f32[256]", squeeze_103: "f32[256]", sum_41: "f32[1024]", squeeze_100: "f32[1024]", sum_43: "f32[256]", squeeze_97: "f32[256]", sum_45: "f32[256]", squeeze_94: "f32[256]", sum_47: "f32[1024]", squeeze_91: "f32[1024]", sum_49: "f32[256]", squeeze_88: "f32[256]", sum_51: "f32[256]", squeeze_85: "f32[256]", sum_53: "f32[1024]", squeeze_82: "f32[1024]", sum_55: "f32[1024]", squeeze_79: "f32[1024]", sum_57: "f32[256]", squeeze_76: "f32[256]", sum_59: "f32[256]", squeeze_73: "f32[256]", sum_61: "f32[512]", squeeze_70: "f32[512]", sum_63: "f32[128]", squeeze_67: "f32[128]", sum_65: "f32[128]", squeeze_64: "f32[128]", sum_67: "f32[512]", squeeze_61: "f32[512]", sum_69: "f32[128]", squeeze_58: "f32[128]", sum_71: "f32[128]", squeeze_55: "f32[128]", sum_73: "f32[512]", squeeze_52: "f32[512]", sum_75: "f32[128]", squeeze_49: "f32[128]", sum_77: "f32[128]", squeeze_46: "f32[128]", sum_79: "f32[512]", squeeze_43: "f32[512]", sum_81: "f32[512]", squeeze_40: "f32[512]", sum_83: "f32[128]", squeeze_37: "f32[128]", sum_85: "f32[128]", squeeze_34: "f32[128]", sum_87: "f32[256]", squeeze_31: "f32[256]", sum_89: "f32[64]", squeeze_28: "f32[64]", sum_91: "f32[64]", squeeze_25: "f32[64]", sum_93: "f32[256]", squeeze_22: "f32[256]", sum_95: "f32[64]", squeeze_19: "f32[64]", sum_97: "f32[64]", squeeze_16: "f32[64]", sum_99: "f32[256]", squeeze_13: "f32[256]", sum_101: "f32[256]", squeeze_10: "f32[256]", sum_103: "f32[64]", squeeze_7: "f32[64]", sum_105: "f32[64]", squeeze_4: "f32[64]", sum_107: "f32[64]", squeeze_1: "f32[64]", _shape_param_0):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:280 in _forward_impl, code: x = self.fc(x)
        permute_default: "f32[1000, 128]" = torch.ops.aten.permute.default(tangents_1, [1, 0])
        sum_dim_int_list: "f32[1, 1000]" = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True);  tangents_1 = None
        reshape_default: "f32[1000]" = torch.ops.aten.reshape.default(sum_dim_int_list, _shape_param_0);  sum_dim_int_list = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        mul_tensor: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_3, squeeze_157);  sum_3 = squeeze_157 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        mul_tensor_1: "f32[512]" = torch.ops.aten.mul.Tensor(sum_5, squeeze_154);  sum_5 = squeeze_154 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        mul_tensor_2: "f32[512]" = torch.ops.aten.mul.Tensor(sum_7, squeeze_151);  sum_7 = squeeze_151 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        mul_tensor_3: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_9, squeeze_148);  sum_9 = squeeze_148 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        mul_tensor_4: "f32[512]" = torch.ops.aten.mul.Tensor(sum_11, squeeze_145);  sum_11 = squeeze_145 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        mul_tensor_5: "f32[512]" = torch.ops.aten.mul.Tensor(sum_13, squeeze_142);  sum_13 = squeeze_142 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:158 in forward, code: identity = self.downsample(x)
        mul_tensor_6: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_15, squeeze_139);  sum_15 = squeeze_139 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        mul_tensor_7: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_17, squeeze_136);  sum_17 = squeeze_136 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        mul_tensor_8: "f32[512]" = torch.ops.aten.mul.Tensor(sum_19, squeeze_133);  sum_19 = squeeze_133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        mul_tensor_9: "f32[512]" = torch.ops.aten.mul.Tensor(sum_21, squeeze_130);  sum_21 = squeeze_130 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        mul_tensor_10: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_23, squeeze_127);  sum_23 = squeeze_127 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        mul_tensor_11: "f32[256]" = torch.ops.aten.mul.Tensor(sum_25, squeeze_124);  sum_25 = squeeze_124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        mul_tensor_12: "f32[256]" = torch.ops.aten.mul.Tensor(sum_27, squeeze_121);  sum_27 = squeeze_121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        mul_tensor_13: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_29, squeeze_118);  sum_29 = squeeze_118 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        mul_tensor_14: "f32[256]" = torch.ops.aten.mul.Tensor(sum_31, squeeze_115);  sum_31 = squeeze_115 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        mul_tensor_15: "f32[256]" = torch.ops.aten.mul.Tensor(sum_33, squeeze_112);  sum_33 = squeeze_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        mul_tensor_16: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_35, squeeze_109);  sum_35 = squeeze_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        mul_tensor_17: "f32[256]" = torch.ops.aten.mul.Tensor(sum_37, squeeze_106);  sum_37 = squeeze_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        mul_tensor_18: "f32[256]" = torch.ops.aten.mul.Tensor(sum_39, squeeze_103);  sum_39 = squeeze_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        mul_tensor_19: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_41, squeeze_100);  sum_41 = squeeze_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        mul_tensor_20: "f32[256]" = torch.ops.aten.mul.Tensor(sum_43, squeeze_97);  sum_43 = squeeze_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        mul_tensor_21: "f32[256]" = torch.ops.aten.mul.Tensor(sum_45, squeeze_94);  sum_45 = squeeze_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        mul_tensor_22: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_47, squeeze_91);  sum_47 = squeeze_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        mul_tensor_23: "f32[256]" = torch.ops.aten.mul.Tensor(sum_49, squeeze_88);  sum_49 = squeeze_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        mul_tensor_24: "f32[256]" = torch.ops.aten.mul.Tensor(sum_51, squeeze_85);  sum_51 = squeeze_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:158 in forward, code: identity = self.downsample(x)
        mul_tensor_25: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_53, squeeze_82);  sum_53 = squeeze_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        mul_tensor_26: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_55, squeeze_79);  sum_55 = squeeze_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        mul_tensor_27: "f32[256]" = torch.ops.aten.mul.Tensor(sum_57, squeeze_76);  sum_57 = squeeze_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        mul_tensor_28: "f32[256]" = torch.ops.aten.mul.Tensor(sum_59, squeeze_73);  sum_59 = squeeze_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        mul_tensor_29: "f32[512]" = torch.ops.aten.mul.Tensor(sum_61, squeeze_70);  sum_61 = squeeze_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        mul_tensor_30: "f32[128]" = torch.ops.aten.mul.Tensor(sum_63, squeeze_67);  sum_63 = squeeze_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        mul_tensor_31: "f32[128]" = torch.ops.aten.mul.Tensor(sum_65, squeeze_64);  sum_65 = squeeze_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        mul_tensor_32: "f32[512]" = torch.ops.aten.mul.Tensor(sum_67, squeeze_61);  sum_67 = squeeze_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        mul_tensor_33: "f32[128]" = torch.ops.aten.mul.Tensor(sum_69, squeeze_58);  sum_69 = squeeze_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        mul_tensor_34: "f32[128]" = torch.ops.aten.mul.Tensor(sum_71, squeeze_55);  sum_71 = squeeze_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        mul_tensor_35: "f32[512]" = torch.ops.aten.mul.Tensor(sum_73, squeeze_52);  sum_73 = squeeze_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        mul_tensor_36: "f32[128]" = torch.ops.aten.mul.Tensor(sum_75, squeeze_49);  sum_75 = squeeze_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        mul_tensor_37: "f32[128]" = torch.ops.aten.mul.Tensor(sum_77, squeeze_46);  sum_77 = squeeze_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:158 in forward, code: identity = self.downsample(x)
        mul_tensor_38: "f32[512]" = torch.ops.aten.mul.Tensor(sum_79, squeeze_43);  sum_79 = squeeze_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        mul_tensor_39: "f32[512]" = torch.ops.aten.mul.Tensor(sum_81, squeeze_40);  sum_81 = squeeze_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        mul_tensor_40: "f32[128]" = torch.ops.aten.mul.Tensor(sum_83, squeeze_37);  sum_83 = squeeze_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        mul_tensor_41: "f32[128]" = torch.ops.aten.mul.Tensor(sum_85, squeeze_34);  sum_85 = squeeze_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        mul_tensor_42: "f32[256]" = torch.ops.aten.mul.Tensor(sum_87, squeeze_31);  sum_87 = squeeze_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        mul_tensor_43: "f32[64]" = torch.ops.aten.mul.Tensor(sum_89, squeeze_28);  sum_89 = squeeze_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        mul_tensor_44: "f32[64]" = torch.ops.aten.mul.Tensor(sum_91, squeeze_25);  sum_91 = squeeze_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        mul_tensor_45: "f32[256]" = torch.ops.aten.mul.Tensor(sum_93, squeeze_22);  sum_93 = squeeze_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        mul_tensor_46: "f32[64]" = torch.ops.aten.mul.Tensor(sum_95, squeeze_19);  sum_95 = squeeze_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        mul_tensor_47: "f32[64]" = torch.ops.aten.mul.Tensor(sum_97, squeeze_16);  sum_97 = squeeze_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:158 in forward, code: identity = self.downsample(x)
        mul_tensor_48: "f32[256]" = torch.ops.aten.mul.Tensor(sum_99, squeeze_13);  sum_99 = squeeze_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        mul_tensor_49: "f32[256]" = torch.ops.aten.mul.Tensor(sum_101, squeeze_10);  sum_101 = squeeze_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        mul_tensor_50: "f32[64]" = torch.ops.aten.mul.Tensor(sum_103, squeeze_7);  sum_103 = squeeze_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        mul_tensor_51: "f32[64]" = torch.ops.aten.mul.Tensor(sum_105, squeeze_4);  sum_105 = squeeze_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:269 in _forward_impl, code: x = self.bn1(x)
        mul_tensor_52: "f32[64]" = torch.ops.aten.mul.Tensor(sum_107, squeeze_1);  sum_107 = squeeze_1 = None
        return (permute_default, reshape_default, mul_tensor, mul_tensor_1, mul_tensor_2, mul_tensor_3, mul_tensor_4, mul_tensor_5, mul_tensor_6, mul_tensor_7, mul_tensor_8, mul_tensor_9, mul_tensor_10, mul_tensor_11, mul_tensor_12, mul_tensor_13, mul_tensor_14, mul_tensor_15, mul_tensor_16, mul_tensor_17, mul_tensor_18, mul_tensor_19, mul_tensor_20, mul_tensor_21, mul_tensor_22, mul_tensor_23, mul_tensor_24, mul_tensor_25, mul_tensor_26, mul_tensor_27, mul_tensor_28, mul_tensor_29, mul_tensor_30, mul_tensor_31, mul_tensor_32, mul_tensor_33, mul_tensor_34, mul_tensor_35, mul_tensor_36, mul_tensor_37, mul_tensor_38, mul_tensor_39, mul_tensor_40, mul_tensor_41, mul_tensor_42, mul_tensor_43, mul_tensor_44, mul_tensor_45, mul_tensor_46, mul_tensor_47, mul_tensor_48, mul_tensor_49, mul_tensor_50, mul_tensor_51, mul_tensor_52)


def _default_make_inputs():
    return [
    torch.randn([128, 1000], dtype=torch.float32, device='cuda'),
    torch.randn([2048], dtype=torch.float32, device='cuda'),
    torch.randn([2048], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([2048], dtype=torch.float32, device='cuda'),
    torch.randn([2048], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([2048], dtype=torch.float32, device='cuda'),
    torch.randn([2048], dtype=torch.float32, device='cuda'),
    torch.randn([2048], dtype=torch.float32, device='cuda'),
    torch.randn([2048], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([1024], dtype=torch.float32, device='cuda'),
    torch.randn([1024], dtype=torch.float32, device='cuda'),
    torch.randn([256], dtype=torch.float32, device='cuda'),
    torch.randn([256], dtype=torch.float32, device='cuda'),
    torch.randn([256], dtype=torch.float32, device='cuda'),
    torch.randn([256], dtype=torch.float32, device='cuda'),
    torch.randn([1024], dtype=torch.float32, device='cuda'),
    torch.randn([1024], dtype=torch.float32, device='cuda'),
    torch.randn([256], dtype=torch.float32, device='cuda'),
    torch.randn([256], dtype=torch.float32, device='cuda'),
    torch.randn([256], dtype=torch.float32, device='cuda'),
    torch.randn([256], dtype=torch.float32, device='cuda'),
    torch.randn([1024], dtype=torch.float32, device='cuda'),
    torch.randn([1024], dtype=torch.float32, device='cuda'),
    torch.randn([256], dtype=torch.float32, device='cuda'),
    torch.randn([256], dtype=torch.float32, device='cuda'),
    torch.randn([256], dtype=torch.float32, device='cuda'),
    torch.randn([256], dtype=torch.float32, device='cuda'),
    torch.randn([1024], dtype=torch.float32, device='cuda'),
    torch.randn([1024], dtype=torch.float32, device='cuda'),
    torch.randn([256], dtype=torch.float32, device='cuda'),
    torch.randn([256], dtype=torch.float32, device='cuda'),
    torch.randn([256], dtype=torch.float32, device='cuda'),
    torch.randn([256], dtype=torch.float32, device='cuda'),
    torch.randn([1024], dtype=torch.float32, device='cuda'),
    torch.randn([1024], dtype=torch.float32, device='cuda'),
    torch.randn([256], dtype=torch.float32, device='cuda'),
    torch.randn([256], dtype=torch.float32, device='cuda'),
    torch.randn([256], dtype=torch.float32, device='cuda'),
    torch.randn([256], dtype=torch.float32, device='cuda'),
    torch.randn([1024], dtype=torch.float32, device='cuda'),
    torch.randn([1024], dtype=torch.float32, device='cuda'),
    torch.randn([1024], dtype=torch.float32, device='cuda'),
    torch.randn([1024], dtype=torch.float32, device='cuda'),
    torch.randn([256], dtype=torch.float32, device='cuda'),
    torch.randn([256], dtype=torch.float32, device='cuda'),
    torch.randn([256], dtype=torch.float32, device='cuda'),
    torch.randn([256], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([256], dtype=torch.float32, device='cuda'),
    torch.randn([256], dtype=torch.float32, device='cuda'),
    torch.randn([64], dtype=torch.float32, device='cuda'),
    torch.randn([64], dtype=torch.float32, device='cuda'),
    torch.randn([64], dtype=torch.float32, device='cuda'),
    torch.randn([64], dtype=torch.float32, device='cuda'),
    torch.randn([256], dtype=torch.float32, device='cuda'),
    torch.randn([256], dtype=torch.float32, device='cuda'),
    torch.randn([64], dtype=torch.float32, device='cuda'),
    torch.randn([64], dtype=torch.float32, device='cuda'),
    torch.randn([64], dtype=torch.float32, device='cuda'),
    torch.randn([64], dtype=torch.float32, device='cuda'),
    torch.randn([256], dtype=torch.float32, device='cuda'),
    torch.randn([256], dtype=torch.float32, device='cuda'),
    torch.randn([256], dtype=torch.float32, device='cuda'),
    torch.randn([256], dtype=torch.float32, device='cuda'),
    torch.randn([64], dtype=torch.float32, device='cuda'),
    torch.randn([64], dtype=torch.float32, device='cuda'),
    torch.randn([64], dtype=torch.float32, device='cuda'),
    torch.randn([64], dtype=torch.float32, device='cuda'),
    torch.randn([64], dtype=torch.float32, device='cuda'),
    torch.randn([64], dtype=torch.float32, device='cuda'),
    [1000],  # _shape_param_0
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
