"""
Standalone repro captured via capture_hook.
Label: resnet50_training
Pattern hash: edff7e8d7092
Shape hash: cb23563b
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, tangents_1: "f32[4, 1000]", _shape_param_0, sum_3: "f32[2048]", squeeze_157: "f32[2048]", sum_5: "f32[512]", squeeze_154: "f32[512]", sum_7: "f32[512]", squeeze_151: "f32[512]", sum_9: "f32[2048]", squeeze_148: "f32[2048]", sum_11: "f32[512]", squeeze_145: "f32[512]", sum_13: "f32[512]", squeeze_142: "f32[512]", sum_15: "f32[2048]", squeeze_139: "f32[2048]", sum_17: "f32[2048]", squeeze_136: "f32[2048]", sum_19: "f32[512]", squeeze_133: "f32[512]", sum_21: "f32[512]", squeeze_130: "f32[512]", sum_23: "f32[1024]", squeeze_127: "f32[1024]", sum_25: "f32[256]", squeeze_124: "f32[256]", sum_27: "f32[256]", squeeze_121: "f32[256]", sum_29: "f32[1024]", squeeze_118: "f32[1024]", sum_31: "f32[256]", squeeze_115: "f32[256]", sum_33: "f32[256]", squeeze_112: "f32[256]", sum_35: "f32[1024]", squeeze_109: "f32[1024]", sum_37: "f32[256]", squeeze_106: "f32[256]", sum_39: "f32[256]", squeeze_103: "f32[256]", sum_41: "f32[1024]", squeeze_100: "f32[1024]", sum_43: "f32[256]", squeeze_97: "f32[256]", sum_45: "f32[256]", squeeze_94: "f32[256]", sum_47: "f32[1024]", squeeze_91: "f32[1024]", sum_49: "f32[256]", squeeze_88: "f32[256]", sum_51: "f32[256]", squeeze_85: "f32[256]", sum_53: "f32[1024]", squeeze_82: "f32[1024]", sum_55: "f32[1024]", squeeze_79: "f32[1024]", sum_57: "f32[256]", squeeze_76: "f32[256]", sum_59: "f32[256]", squeeze_73: "f32[256]", sum_61: "f32[512]", squeeze_70: "f32[512]", sum_63: "f32[128]", squeeze_67: "f32[128]", sum_65: "f32[128]", squeeze_64: "f32[128]", sum_67: "f32[512]", squeeze_61: "f32[512]", sum_69: "f32[128]", squeeze_58: "f32[128]", sum_71: "f32[128]", squeeze_55: "f32[128]", sum_73: "f32[512]", squeeze_52: "f32[512]", sum_75: "f32[128]", squeeze_49: "f32[128]", sum_77: "f32[128]", squeeze_46: "f32[128]", sum_79: "f32[512]", squeeze_43: "f32[512]", sum_81: "f32[512]", squeeze_40: "f32[512]", sum_83: "f32[128]", squeeze_37: "f32[128]", sum_85: "f32[128]", squeeze_34: "f32[128]", sum_87: "f32[256]", squeeze_31: "f32[256]", sum_89: "f32[64]", squeeze_28: "f32[64]", sum_91: "f32[64]", squeeze_25: "f32[64]", sum_93: "f32[256]", squeeze_22: "f32[256]", sum_95: "f32[64]", squeeze_19: "f32[64]", sum_97: "f32[64]", squeeze_16: "f32[64]", sum_99: "f32[256]", squeeze_13: "f32[256]", sum_101: "f32[256]", squeeze_10: "f32[256]", sum_103: "f32[64]", squeeze_7: "f32[64]", sum_105: "f32[64]", squeeze_4: "f32[64]", getitem_252: "f32[4, 64, 56, 56]", getitem_261: "f32[4, 64, 56, 56]", _shape_param_1, getitem_3: "i8[4, 64, 56, 56]", _shape_param_2, _shape_param_3, convolution: "f32[4, 64, 112, 112]", getitem_1: "f32[1, 64, 1, 1]", rsqrt: "f32[1, 64, 1, 1]", primals_6: "f32[64]", primals_7: "f32[64]", full_default: "f32[]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:280 in _forward_impl, code: x = self.fc(x)
        permute_default: "f32[1000, 4]" = torch.ops.aten.permute.default(tangents_1, [1, 0])
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

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        add_tensor: "f32[4, 64, 56, 56]" = torch.ops.aten.add.Tensor(getitem_252, getitem_261);  getitem_252 = getitem_261 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:271 in _forward_impl, code: x = self.maxpool(x)
        full_default: "f32[256, 12544]" = torch.ops.aten.full.default([256, 12544], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        reshape_default_1: "f32[256, 3136]" = torch.ops.aten.reshape.default(add_tensor, _shape_param_1);  add_tensor = _shape_param_1 = None
        _low_memory_max_pool_offsets_to_indices_default: "i64[4, 64, 56, 56]" = torch.ops.prims._low_memory_max_pool_offsets_to_indices.default(getitem_3, [3, 3], [112, 112], [2, 2], [1, 1], [1, 1]);  getitem_3 = None
        reshape_default_2: "i64[256, 3136]" = torch.ops.aten.reshape.default(_low_memory_max_pool_offsets_to_indices_default, _shape_param_2);  _low_memory_max_pool_offsets_to_indices_default = _shape_param_2 = None
        scatter_add_default: "f32[256, 12544]" = torch.ops.aten.scatter_add.default(full_default, 1, reshape_default_2, reshape_default_1);  full_default = reshape_default_2 = reshape_default_1 = None
        reshape_default_3: "f32[4, 64, 112, 112]" = torch.ops.aten.reshape.default(scatter_add_default, _shape_param_3);  scatter_add_default = _shape_param_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:269 in _forward_impl, code: x = self.bn1(x)
        sub_tensor: "f32[4, 64, 112, 112]" = torch.ops.aten.sub.Tensor(convolution, getitem_1)
        mul_tensor_52: "f32[4, 64, 112, 112]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt);  sub_tensor = None
        unsqueeze_default: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_6, -1)
        unsqueeze_default_1: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        mul_tensor_53: "f32[4, 64, 112, 112]" = torch.ops.aten.mul.Tensor(mul_tensor_52, unsqueeze_default_1);  mul_tensor_52 = unsqueeze_default_1 = None
        unsqueeze_default_2: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_7, -1);  primals_7 = None
        unsqueeze_default_3: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        add_tensor_1: "f32[4, 64, 112, 112]" = torch.ops.aten.add.Tensor(mul_tensor_53, unsqueeze_default_3);  mul_tensor_53 = unsqueeze_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:270 in _forward_impl, code: x = self.relu(x)
        relu_default: "f32[4, 64, 112, 112]" = torch.ops.aten.relu.default(add_tensor_1);  add_tensor_1 = None
        le_scalar: "b8[4, 64, 112, 112]" = torch.ops.aten.le.Scalar(relu_default, 0);  relu_default = None
        full_default_1 = full_default
        where_self: "f32[4, 64, 112, 112]" = torch.ops.aten.where.self(le_scalar, full_default_1, reshape_default_3);  le_scalar = full_default_1 = reshape_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:269 in _forward_impl, code: x = self.bn1(x)
        squeeze_dims: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_1, [0, 2, 3]);  getitem_1 = None
        unsqueeze_default_4: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_dims, 0);  squeeze_dims = None
        unsqueeze_default_5: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 2);  unsqueeze_default_4 = None
        unsqueeze_default_6: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_5, 3);  unsqueeze_default_5 = None
        sum_dim_int_list_1: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_self, [0, 2, 3])
        sub_tensor_1: "f32[4, 64, 112, 112]" = torch.ops.aten.sub.Tensor(convolution, unsqueeze_default_6);  convolution = unsqueeze_default_6 = None
        mul_tensor_54: "f32[4, 64, 112, 112]" = torch.ops.aten.mul.Tensor(where_self, sub_tensor_1)
        sum_dim_int_list_2: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_tensor_54, [0, 2, 3]);  mul_tensor_54 = None
        mul_tensor_55: "f32[64]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, 1.992984693877551e-05);  sum_dim_int_list_1 = None
        unsqueeze_default_7: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_tensor_55, 0);  mul_tensor_55 = None
        unsqueeze_default_8: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_7, 2);  unsqueeze_default_7 = None
        unsqueeze_default_9: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_8, 3);  unsqueeze_default_8 = None
        mul_tensor_56: "f32[64]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_2, 1.992984693877551e-05)
        squeeze_dims_1: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt, [0, 2, 3]);  rsqrt = None
        mul_tensor_57: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_dims_1, squeeze_dims_1)
        mul_tensor_58: "f32[64]" = torch.ops.aten.mul.Tensor(mul_tensor_56, mul_tensor_57);  mul_tensor_56 = mul_tensor_57 = None
        unsqueeze_default_10: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_tensor_58, 0);  mul_tensor_58 = None
        unsqueeze_default_11: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_10, 2);  unsqueeze_default_10 = None
        unsqueeze_default_12: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_11, 3);  unsqueeze_default_11 = None
        mul_tensor_59: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_dims_1, primals_6);  primals_6 = None
        unsqueeze_default_13: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_tensor_59, 0);  mul_tensor_59 = None
        unsqueeze_default_14: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_13, 2);  unsqueeze_default_13 = None
        unsqueeze_default_15: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_14, 3);  unsqueeze_default_14 = None
        mul_tensor_60: "f32[4, 64, 112, 112]" = torch.ops.aten.mul.Tensor(sub_tensor_1, unsqueeze_default_12);  sub_tensor_1 = unsqueeze_default_12 = None
        sub_tensor_2: "f32[4, 64, 112, 112]" = torch.ops.aten.sub.Tensor(where_self, mul_tensor_60);  where_self = mul_tensor_60 = None
        sub_tensor_3: "f32[4, 64, 112, 112]" = torch.ops.aten.sub.Tensor(sub_tensor_2, unsqueeze_default_9);  sub_tensor_2 = unsqueeze_default_9 = None
        mul_tensor_61: "f32[4, 64, 112, 112]" = torch.ops.aten.mul.Tensor(sub_tensor_3, unsqueeze_default_15);  sub_tensor_3 = unsqueeze_default_15 = None
        mul_tensor_62: "f32[64]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_2, squeeze_dims_1);  sum_dim_int_list_2 = squeeze_dims_1 = None
        return (permute_default, reshape_default, mul_tensor, mul_tensor_1, mul_tensor_2, mul_tensor_3, mul_tensor_4, mul_tensor_5, mul_tensor_6, mul_tensor_7, mul_tensor_8, mul_tensor_9, mul_tensor_10, mul_tensor_11, mul_tensor_12, mul_tensor_13, mul_tensor_14, mul_tensor_15, mul_tensor_16, mul_tensor_17, mul_tensor_18, mul_tensor_19, mul_tensor_20, mul_tensor_21, mul_tensor_22, mul_tensor_23, mul_tensor_24, mul_tensor_25, mul_tensor_26, mul_tensor_27, mul_tensor_28, mul_tensor_29, mul_tensor_30, mul_tensor_31, mul_tensor_32, mul_tensor_33, mul_tensor_34, mul_tensor_35, mul_tensor_36, mul_tensor_37, mul_tensor_38, mul_tensor_39, mul_tensor_40, mul_tensor_41, mul_tensor_42, mul_tensor_43, mul_tensor_44, mul_tensor_45, mul_tensor_46, mul_tensor_47, mul_tensor_48, mul_tensor_49, mul_tensor_50, mul_tensor_51, mul_tensor_61, mul_tensor_62)


def _default_make_inputs():
    return [
    torch.randn([4, 1000], dtype=torch.float32, device='cuda'),
    [1000],  # _shape_param_0
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
    torch.randn(802816, dtype=torch.float32, device='cuda').as_strided([4, 64, 56, 56], [200704, 1, 3584, 64]),  # getitem_252
    torch.randn(802816, dtype=torch.float32, device='cuda').as_strided([4, 64, 56, 56], [200704, 1, 3584, 64]),  # getitem_261
    [256, 3136],  # _shape_param_1
    torch.randint(0, 2, (802816,), dtype=torch.int8, device='cuda').as_strided([4, 64, 56, 56], [200704, 1, 3584, 64]),  # getitem_3
    [256, 3136],  # _shape_param_2
    [4, 64, 112, 112],  # _shape_param_3
    torch.randn(3211264, dtype=torch.float32, device='cuda').as_strided([4, 64, 112, 112], [802816, 1, 7168, 64]),  # convolution
    torch.randn([1, 64, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 64, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([64], dtype=torch.float32, device='cuda'),
    torch.randn([64], dtype=torch.float32, device='cuda'),
    torch.tensor(1),  # full_default_1 (unknown shape)
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
