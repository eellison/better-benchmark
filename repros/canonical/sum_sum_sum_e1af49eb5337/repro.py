"""
Standalone repro captured via capture_hook.
Label: torchbench_mobilenet_v3_large_train
Pattern hash: e1af49eb5337
Shape hash: 5904dc70
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
    def forward(self, tangents_1: "f32[256, 1000]", where_1: "f32[256, 1280]", sum_4: "f32[960]", squeeze_136: "f32[960]", sum_6: "f32[160]", squeeze_133: "f32[160]", where_4: "f32[256, 960, 1, 1]", where_5: "f32[256, 240, 1, 1]", sum_11: "f32[960]", squeeze_130: "f32[960]", sum_13: "f32[960]", squeeze_127: "f32[960]", sum_15: "f32[160]", squeeze_124: "f32[160]", where_10: "f32[256, 960, 1, 1]", where_11: "f32[256, 240, 1, 1]", sum_20: "f32[960]", squeeze_121: "f32[960]", sum_22: "f32[960]", squeeze_118: "f32[960]", sum_24: "f32[160]", squeeze_115: "f32[160]", where_16: "f32[256, 672, 1, 1]", where_17: "f32[256, 168, 1, 1]", sum_29: "f32[672]", squeeze_112: "f32[672]", sum_31: "f32[672]", squeeze_109: "f32[672]", sum_33: "f32[112]", squeeze_106: "f32[112]", where_22: "f32[256, 672, 1, 1]", where_23: "f32[256, 168, 1, 1]", sum_38: "f32[672]", squeeze_103: "f32[672]", sum_40: "f32[672]", squeeze_100: "f32[672]", sum_42: "f32[112]", squeeze_97: "f32[112]", where_28: "f32[256, 480, 1, 1]", where_29: "f32[256, 120, 1, 1]", sum_47: "f32[480]", squeeze_94: "f32[480]", sum_49: "f32[480]", squeeze_91: "f32[480]", sum_51: "f32[80]", squeeze_88: "f32[80]", sum_53: "f32[184]", squeeze_85: "f32[184]", sum_55: "f32[184]", squeeze_82: "f32[184]", sum_57: "f32[80]", squeeze_79: "f32[80]", sum_59: "f32[184]", squeeze_76: "f32[184]", sum_61: "f32[184]", squeeze_73: "f32[184]", sum_63: "f32[80]", squeeze_70: "f32[80]", sum_65: "f32[200]", squeeze_67: "f32[200]", sum_67: "f32[200]", squeeze_64: "f32[200]", sum_69: "f32[80]", squeeze_61: "f32[80]", sum_71: "f32[240]", squeeze_58: "f32[240]", sum_73: "f32[240]", squeeze_55: "f32[240]", sum_75: "f32[40]", squeeze_52: "f32[40]", where_50: "f32[256, 120, 1, 1]", where_51: "f32[256, 32, 1, 1]", sum_80: "f32[120]", squeeze_49: "f32[120]", sum_82: "f32[120]", squeeze_46: "f32[120]", sum_84: "f32[40]", squeeze_43: "f32[40]", where_54: "f32[256, 120, 1, 1]", where_55: "f32[256, 32, 1, 1]", sum_89: "f32[120]", squeeze_40: "f32[120]", sum_91: "f32[120]", squeeze_37: "f32[120]", sum_93: "f32[40]", squeeze_34: "f32[40]", where_58: "f32[256, 72, 1, 1]", where_59: "f32[256, 24, 1, 1]", sum_98: "f32[72]", squeeze_31: "f32[72]", sum_100: "f32[72]", squeeze_28: "f32[72]", sum_102: "f32[24]", squeeze_25: "f32[24]", sum_104: "f32[72]", squeeze_22: "f32[72]", sum_106: "f32[72]", squeeze_19: "f32[72]", sum_108: "f32[24]", squeeze_16: "f32[24]", sum_110: "f32[64]", squeeze_13: "f32[64]", sum_112: "f32[64]", squeeze_10: "f32[64]", sum_114: "f32[16]", squeeze_7: "f32[16]", sum_116: "f32[16]", squeeze_4: "f32[16]", sum_118: "f32[16]", squeeze_1: "f32[16]", _shape_param_0, _shape_param_1):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/mobilenetv3.py:216 in _forward_impl, code: x = self.classifier(x)
        permute_default: "f32[1000, 256]" = torch.ops.aten.permute.default(tangents_1, [1, 0])
        sum_dim_int_list: "f32[1, 1000]" = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True);  tangents_1 = None
        reshape_default: "f32[1000]" = torch.ops.aten.reshape.default(sum_dim_int_list, _shape_param_0);  sum_dim_int_list = _shape_param_0 = None
        permute_default_1: "f32[1280, 256]" = torch.ops.aten.permute.default(where_1, [1, 0])
        sum_dim_int_list_1: "f32[1, 1280]" = torch.ops.aten.sum.dim_IntList(where_1, [0], True);  where_1 = None
        reshape_default_1: "f32[1280]" = torch.ops.aten.reshape.default(sum_dim_int_list_1, _shape_param_1);  sum_dim_int_list_1 = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/mobilenetv3.py:211 in _forward_impl, code: x = self.features(x)
        mul_tensor: "f32[960]" = torch.ops.aten.mul.Tensor(sum_4, squeeze_136);  sum_4 = squeeze_136 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/mobilenetv3.py:112 in forward, code: result = self.block(input)
        mul_tensor_1: "f32[160]" = torch.ops.aten.mul.Tensor(sum_6, squeeze_133);  sum_6 = squeeze_133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:256 in _scale, code: scale = self.fc2(scale)
        sum_dim_int_list_2: "f32[960]" = torch.ops.aten.sum.dim_IntList(where_4, [0, 2, 3]);  where_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:254 in _scale, code: scale = self.fc1(scale)
        sum_dim_int_list_3: "f32[240]" = torch.ops.aten.sum.dim_IntList(where_5, [0, 2, 3]);  where_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/mobilenetv3.py:112 in forward, code: result = self.block(input)
        mul_tensor_2: "f32[960]" = torch.ops.aten.mul.Tensor(sum_11, squeeze_130);  sum_11 = squeeze_130 = None
        mul_tensor_3: "f32[960]" = torch.ops.aten.mul.Tensor(sum_13, squeeze_127);  sum_13 = squeeze_127 = None
        mul_tensor_4: "f32[160]" = torch.ops.aten.mul.Tensor(sum_15, squeeze_124);  sum_15 = squeeze_124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:256 in _scale, code: scale = self.fc2(scale)
        sum_dim_int_list_4: "f32[960]" = torch.ops.aten.sum.dim_IntList(where_10, [0, 2, 3]);  where_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:254 in _scale, code: scale = self.fc1(scale)
        sum_dim_int_list_5: "f32[240]" = torch.ops.aten.sum.dim_IntList(where_11, [0, 2, 3]);  where_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/mobilenetv3.py:112 in forward, code: result = self.block(input)
        mul_tensor_5: "f32[960]" = torch.ops.aten.mul.Tensor(sum_20, squeeze_121);  sum_20 = squeeze_121 = None
        mul_tensor_6: "f32[960]" = torch.ops.aten.mul.Tensor(sum_22, squeeze_118);  sum_22 = squeeze_118 = None
        mul_tensor_7: "f32[160]" = torch.ops.aten.mul.Tensor(sum_24, squeeze_115);  sum_24 = squeeze_115 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:256 in _scale, code: scale = self.fc2(scale)
        sum_dim_int_list_6: "f32[672]" = torch.ops.aten.sum.dim_IntList(where_16, [0, 2, 3]);  where_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:254 in _scale, code: scale = self.fc1(scale)
        sum_dim_int_list_7: "f32[168]" = torch.ops.aten.sum.dim_IntList(where_17, [0, 2, 3]);  where_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/mobilenetv3.py:112 in forward, code: result = self.block(input)
        mul_tensor_8: "f32[672]" = torch.ops.aten.mul.Tensor(sum_29, squeeze_112);  sum_29 = squeeze_112 = None
        mul_tensor_9: "f32[672]" = torch.ops.aten.mul.Tensor(sum_31, squeeze_109);  sum_31 = squeeze_109 = None
        mul_tensor_10: "f32[112]" = torch.ops.aten.mul.Tensor(sum_33, squeeze_106);  sum_33 = squeeze_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:256 in _scale, code: scale = self.fc2(scale)
        sum_dim_int_list_8: "f32[672]" = torch.ops.aten.sum.dim_IntList(where_22, [0, 2, 3]);  where_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:254 in _scale, code: scale = self.fc1(scale)
        sum_dim_int_list_9: "f32[168]" = torch.ops.aten.sum.dim_IntList(where_23, [0, 2, 3]);  where_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/mobilenetv3.py:112 in forward, code: result = self.block(input)
        mul_tensor_11: "f32[672]" = torch.ops.aten.mul.Tensor(sum_38, squeeze_103);  sum_38 = squeeze_103 = None
        mul_tensor_12: "f32[672]" = torch.ops.aten.mul.Tensor(sum_40, squeeze_100);  sum_40 = squeeze_100 = None
        mul_tensor_13: "f32[112]" = torch.ops.aten.mul.Tensor(sum_42, squeeze_97);  sum_42 = squeeze_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:256 in _scale, code: scale = self.fc2(scale)
        sum_dim_int_list_10: "f32[480]" = torch.ops.aten.sum.dim_IntList(where_28, [0, 2, 3]);  where_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:254 in _scale, code: scale = self.fc1(scale)
        sum_dim_int_list_11: "f32[120]" = torch.ops.aten.sum.dim_IntList(where_29, [0, 2, 3]);  where_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/mobilenetv3.py:112 in forward, code: result = self.block(input)
        mul_tensor_14: "f32[480]" = torch.ops.aten.mul.Tensor(sum_47, squeeze_94);  sum_47 = squeeze_94 = None
        mul_tensor_15: "f32[480]" = torch.ops.aten.mul.Tensor(sum_49, squeeze_91);  sum_49 = squeeze_91 = None
        mul_tensor_16: "f32[80]" = torch.ops.aten.mul.Tensor(sum_51, squeeze_88);  sum_51 = squeeze_88 = None
        mul_tensor_17: "f32[184]" = torch.ops.aten.mul.Tensor(sum_53, squeeze_85);  sum_53 = squeeze_85 = None
        mul_tensor_18: "f32[184]" = torch.ops.aten.mul.Tensor(sum_55, squeeze_82);  sum_55 = squeeze_82 = None
        mul_tensor_19: "f32[80]" = torch.ops.aten.mul.Tensor(sum_57, squeeze_79);  sum_57 = squeeze_79 = None
        mul_tensor_20: "f32[184]" = torch.ops.aten.mul.Tensor(sum_59, squeeze_76);  sum_59 = squeeze_76 = None
        mul_tensor_21: "f32[184]" = torch.ops.aten.mul.Tensor(sum_61, squeeze_73);  sum_61 = squeeze_73 = None
        mul_tensor_22: "f32[80]" = torch.ops.aten.mul.Tensor(sum_63, squeeze_70);  sum_63 = squeeze_70 = None
        mul_tensor_23: "f32[200]" = torch.ops.aten.mul.Tensor(sum_65, squeeze_67);  sum_65 = squeeze_67 = None
        mul_tensor_24: "f32[200]" = torch.ops.aten.mul.Tensor(sum_67, squeeze_64);  sum_67 = squeeze_64 = None
        mul_tensor_25: "f32[80]" = torch.ops.aten.mul.Tensor(sum_69, squeeze_61);  sum_69 = squeeze_61 = None
        mul_tensor_26: "f32[240]" = torch.ops.aten.mul.Tensor(sum_71, squeeze_58);  sum_71 = squeeze_58 = None
        mul_tensor_27: "f32[240]" = torch.ops.aten.mul.Tensor(sum_73, squeeze_55);  sum_73 = squeeze_55 = None
        mul_tensor_28: "f32[40]" = torch.ops.aten.mul.Tensor(sum_75, squeeze_52);  sum_75 = squeeze_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:256 in _scale, code: scale = self.fc2(scale)
        sum_dim_int_list_12: "f32[120]" = torch.ops.aten.sum.dim_IntList(where_50, [0, 2, 3]);  where_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:254 in _scale, code: scale = self.fc1(scale)
        sum_dim_int_list_13: "f32[32]" = torch.ops.aten.sum.dim_IntList(where_51, [0, 2, 3]);  where_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/mobilenetv3.py:112 in forward, code: result = self.block(input)
        mul_tensor_29: "f32[120]" = torch.ops.aten.mul.Tensor(sum_80, squeeze_49);  sum_80 = squeeze_49 = None
        mul_tensor_30: "f32[120]" = torch.ops.aten.mul.Tensor(sum_82, squeeze_46);  sum_82 = squeeze_46 = None
        mul_tensor_31: "f32[40]" = torch.ops.aten.mul.Tensor(sum_84, squeeze_43);  sum_84 = squeeze_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:256 in _scale, code: scale = self.fc2(scale)
        sum_dim_int_list_14: "f32[120]" = torch.ops.aten.sum.dim_IntList(where_54, [0, 2, 3]);  where_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:254 in _scale, code: scale = self.fc1(scale)
        sum_dim_int_list_15: "f32[32]" = torch.ops.aten.sum.dim_IntList(where_55, [0, 2, 3]);  where_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/mobilenetv3.py:112 in forward, code: result = self.block(input)
        mul_tensor_32: "f32[120]" = torch.ops.aten.mul.Tensor(sum_89, squeeze_40);  sum_89 = squeeze_40 = None
        mul_tensor_33: "f32[120]" = torch.ops.aten.mul.Tensor(sum_91, squeeze_37);  sum_91 = squeeze_37 = None
        mul_tensor_34: "f32[40]" = torch.ops.aten.mul.Tensor(sum_93, squeeze_34);  sum_93 = squeeze_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:256 in _scale, code: scale = self.fc2(scale)
        sum_dim_int_list_16: "f32[72]" = torch.ops.aten.sum.dim_IntList(where_58, [0, 2, 3]);  where_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:254 in _scale, code: scale = self.fc1(scale)
        sum_dim_int_list_17: "f32[24]" = torch.ops.aten.sum.dim_IntList(where_59, [0, 2, 3]);  where_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/mobilenetv3.py:112 in forward, code: result = self.block(input)
        mul_tensor_35: "f32[72]" = torch.ops.aten.mul.Tensor(sum_98, squeeze_31);  sum_98 = squeeze_31 = None
        mul_tensor_36: "f32[72]" = torch.ops.aten.mul.Tensor(sum_100, squeeze_28);  sum_100 = squeeze_28 = None
        mul_tensor_37: "f32[24]" = torch.ops.aten.mul.Tensor(sum_102, squeeze_25);  sum_102 = squeeze_25 = None
        mul_tensor_38: "f32[72]" = torch.ops.aten.mul.Tensor(sum_104, squeeze_22);  sum_104 = squeeze_22 = None
        mul_tensor_39: "f32[72]" = torch.ops.aten.mul.Tensor(sum_106, squeeze_19);  sum_106 = squeeze_19 = None
        mul_tensor_40: "f32[24]" = torch.ops.aten.mul.Tensor(sum_108, squeeze_16);  sum_108 = squeeze_16 = None
        mul_tensor_41: "f32[64]" = torch.ops.aten.mul.Tensor(sum_110, squeeze_13);  sum_110 = squeeze_13 = None
        mul_tensor_42: "f32[64]" = torch.ops.aten.mul.Tensor(sum_112, squeeze_10);  sum_112 = squeeze_10 = None
        mul_tensor_43: "f32[16]" = torch.ops.aten.mul.Tensor(sum_114, squeeze_7);  sum_114 = squeeze_7 = None
        mul_tensor_44: "f32[16]" = torch.ops.aten.mul.Tensor(sum_116, squeeze_4);  sum_116 = squeeze_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/mobilenetv3.py:211 in _forward_impl, code: x = self.features(x)
        mul_tensor_45: "f32[16]" = torch.ops.aten.mul.Tensor(sum_118, squeeze_1);  sum_118 = squeeze_1 = None
        return (permute_default, reshape_default, permute_default_1, reshape_default_1, mul_tensor, mul_tensor_1, sum_dim_int_list_2, sum_dim_int_list_3, mul_tensor_2, mul_tensor_3, mul_tensor_4, sum_dim_int_list_4, sum_dim_int_list_5, mul_tensor_5, mul_tensor_6, mul_tensor_7, sum_dim_int_list_6, sum_dim_int_list_7, mul_tensor_8, mul_tensor_9, mul_tensor_10, sum_dim_int_list_8, sum_dim_int_list_9, mul_tensor_11, mul_tensor_12, mul_tensor_13, sum_dim_int_list_10, sum_dim_int_list_11, mul_tensor_14, mul_tensor_15, mul_tensor_16, mul_tensor_17, mul_tensor_18, mul_tensor_19, mul_tensor_20, mul_tensor_21, mul_tensor_22, mul_tensor_23, mul_tensor_24, mul_tensor_25, mul_tensor_26, mul_tensor_27, mul_tensor_28, sum_dim_int_list_12, sum_dim_int_list_13, mul_tensor_29, mul_tensor_30, mul_tensor_31, sum_dim_int_list_14, sum_dim_int_list_15, mul_tensor_32, mul_tensor_33, mul_tensor_34, sum_dim_int_list_16, sum_dim_int_list_17, mul_tensor_35, mul_tensor_36, mul_tensor_37, mul_tensor_38, mul_tensor_39, mul_tensor_40, mul_tensor_41, mul_tensor_42, mul_tensor_43, mul_tensor_44, mul_tensor_45)


def _default_make_inputs():
    return [
    torch.randn([256, 1000], dtype=torch.float32, device='cuda'),
    torch.randn([256, 1280], dtype=torch.float32, device='cuda'),
    torch.randn([960], dtype=torch.float32, device='cuda'),
    torch.randn([960], dtype=torch.float32, device='cuda'),
    torch.randn([160], dtype=torch.float32, device='cuda'),
    torch.randn([160], dtype=torch.float32, device='cuda'),
    torch.randn([256, 960, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([256, 240, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([960], dtype=torch.float32, device='cuda'),
    torch.randn([960], dtype=torch.float32, device='cuda'),
    torch.randn([960], dtype=torch.float32, device='cuda'),
    torch.randn([960], dtype=torch.float32, device='cuda'),
    torch.randn([160], dtype=torch.float32, device='cuda'),
    torch.randn([160], dtype=torch.float32, device='cuda'),
    torch.randn([256, 960, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([256, 240, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([960], dtype=torch.float32, device='cuda'),
    torch.randn([960], dtype=torch.float32, device='cuda'),
    torch.randn([960], dtype=torch.float32, device='cuda'),
    torch.randn([960], dtype=torch.float32, device='cuda'),
    torch.randn([160], dtype=torch.float32, device='cuda'),
    torch.randn([160], dtype=torch.float32, device='cuda'),
    torch.randn([256, 672, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([256, 168, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([672], dtype=torch.float32, device='cuda'),
    torch.randn([672], dtype=torch.float32, device='cuda'),
    torch.randn([672], dtype=torch.float32, device='cuda'),
    torch.randn([672], dtype=torch.float32, device='cuda'),
    torch.randn([112], dtype=torch.float32, device='cuda'),
    torch.randn([112], dtype=torch.float32, device='cuda'),
    torch.randn([256, 672, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([256, 168, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([672], dtype=torch.float32, device='cuda'),
    torch.randn([672], dtype=torch.float32, device='cuda'),
    torch.randn([672], dtype=torch.float32, device='cuda'),
    torch.randn([672], dtype=torch.float32, device='cuda'),
    torch.randn([112], dtype=torch.float32, device='cuda'),
    torch.randn([112], dtype=torch.float32, device='cuda'),
    torch.randn([256, 480, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([256, 120, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([480], dtype=torch.float32, device='cuda'),
    torch.randn([480], dtype=torch.float32, device='cuda'),
    torch.randn([480], dtype=torch.float32, device='cuda'),
    torch.randn([480], dtype=torch.float32, device='cuda'),
    torch.randn([80], dtype=torch.float32, device='cuda'),
    torch.randn([80], dtype=torch.float32, device='cuda'),
    torch.randn([184], dtype=torch.float32, device='cuda'),
    torch.randn([184], dtype=torch.float32, device='cuda'),
    torch.randn([184], dtype=torch.float32, device='cuda'),
    torch.randn([184], dtype=torch.float32, device='cuda'),
    torch.randn([80], dtype=torch.float32, device='cuda'),
    torch.randn([80], dtype=torch.float32, device='cuda'),
    torch.randn([184], dtype=torch.float32, device='cuda'),
    torch.randn([184], dtype=torch.float32, device='cuda'),
    torch.randn([184], dtype=torch.float32, device='cuda'),
    torch.randn([184], dtype=torch.float32, device='cuda'),
    torch.randn([80], dtype=torch.float32, device='cuda'),
    torch.randn([80], dtype=torch.float32, device='cuda'),
    torch.randn([200], dtype=torch.float32, device='cuda'),
    torch.randn([200], dtype=torch.float32, device='cuda'),
    torch.randn([200], dtype=torch.float32, device='cuda'),
    torch.randn([200], dtype=torch.float32, device='cuda'),
    torch.randn([80], dtype=torch.float32, device='cuda'),
    torch.randn([80], dtype=torch.float32, device='cuda'),
    torch.randn([240], dtype=torch.float32, device='cuda'),
    torch.randn([240], dtype=torch.float32, device='cuda'),
    torch.randn([240], dtype=torch.float32, device='cuda'),
    torch.randn([240], dtype=torch.float32, device='cuda'),
    torch.randn([40], dtype=torch.float32, device='cuda'),
    torch.randn([40], dtype=torch.float32, device='cuda'),
    torch.randn([256, 120, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([256, 32, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([120], dtype=torch.float32, device='cuda'),
    torch.randn([120], dtype=torch.float32, device='cuda'),
    torch.randn([120], dtype=torch.float32, device='cuda'),
    torch.randn([120], dtype=torch.float32, device='cuda'),
    torch.randn([40], dtype=torch.float32, device='cuda'),
    torch.randn([40], dtype=torch.float32, device='cuda'),
    torch.randn([256, 120, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([256, 32, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([120], dtype=torch.float32, device='cuda'),
    torch.randn([120], dtype=torch.float32, device='cuda'),
    torch.randn([120], dtype=torch.float32, device='cuda'),
    torch.randn([120], dtype=torch.float32, device='cuda'),
    torch.randn([40], dtype=torch.float32, device='cuda'),
    torch.randn([40], dtype=torch.float32, device='cuda'),
    torch.randn([256, 72, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([256, 24, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([72], dtype=torch.float32, device='cuda'),
    torch.randn([72], dtype=torch.float32, device='cuda'),
    torch.randn([72], dtype=torch.float32, device='cuda'),
    torch.randn([72], dtype=torch.float32, device='cuda'),
    torch.randn([24], dtype=torch.float32, device='cuda'),
    torch.randn([24], dtype=torch.float32, device='cuda'),
    torch.randn([72], dtype=torch.float32, device='cuda'),
    torch.randn([72], dtype=torch.float32, device='cuda'),
    torch.randn([72], dtype=torch.float32, device='cuda'),
    torch.randn([72], dtype=torch.float32, device='cuda'),
    torch.randn([24], dtype=torch.float32, device='cuda'),
    torch.randn([24], dtype=torch.float32, device='cuda'),
    torch.randn([64], dtype=torch.float32, device='cuda'),
    torch.randn([64], dtype=torch.float32, device='cuda'),
    torch.randn([64], dtype=torch.float32, device='cuda'),
    torch.randn([64], dtype=torch.float32, device='cuda'),
    torch.randn([16], dtype=torch.float32, device='cuda'),
    torch.randn([16], dtype=torch.float32, device='cuda'),
    torch.randn([16], dtype=torch.float32, device='cuda'),
    torch.randn([16], dtype=torch.float32, device='cuda'),
    torch.randn([16], dtype=torch.float32, device='cuda'),
    torch.randn([16], dtype=torch.float32, device='cuda'),
    [1000],  # _shape_param_0
    [1280],  # _shape_param_1
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
