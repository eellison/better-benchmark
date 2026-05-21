"""
Standalone repro captured via capture_hook.
Label: torchbench_LearningToPaint_train
Pattern hash: 1eb2f5b0bf51
Shape hash: a0bcbf20
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([], i64), T([1, 64, 1, 1], f32), T([1, 64, 1, 1], f32), T([64], f32), T([1, 64, 1, 1], f32), T([64], f32), T([], i64), T([1, 64, 1, 1], f32), T([1, 64, 1, 1], f32), T([64], f32), T([1, 64, 1, 1], f32), T([64], f32), T([], i64), T([1, 64, 1, 1], f32), T([1, 64, 1, 1], f32), T([64], f32), T([1, 64, 1, 1], f32), T([64], f32), T([], i64), T([1, 64, 1, 1], f32), T([1, 64, 1, 1], f32), T([64], f32), T([1, 64, 1, 1], f32), T([64], f32), T([], i64), T([1, 64, 1, 1], f32), T([1, 64, 1, 1], f32), T([64], f32), T([1, 64, 1, 1], f32), T([64], f32), T([], i64), T([1, 64, 1, 1], f32), T([1, 64, 1, 1], f32), T([64], f32), T([1, 64, 1, 1], f32), T([64], f32), T([], i64), T([1, 128, 1, 1], f32), T([1, 128, 1, 1], f32), T([128], f32), T([1, 128, 1, 1], f32), T([128], f32), T([], i64), T([1, 128, 1, 1], f32), T([1, 128, 1, 1], f32), T([128], f32), T([1, 128, 1, 1], f32), T([128], f32), T([], i64), T([1, 128, 1, 1], f32), T([1, 128, 1, 1], f32), T([128], f32), T([1, 128, 1, 1], f32), T([128], f32), T([], i64), T([1, 128, 1, 1], f32), T([1, 128, 1, 1], f32), T([128], f32), T([1, 128, 1, 1], f32), T([128], f32), T([], i64), T([1, 128, 1, 1], f32), T([1, 128, 1, 1], f32), T([128], f32), T([1, 128, 1, 1], f32), T([128], f32), T([], i64), T([1, 256, 1, 1], f32), T([1, 256, 1, 1], f32), T([256], f32), T([1, 256, 1, 1], f32), T([256], f32), T([], i64), T([1, 256, 1, 1], f32), T([1, 256, 1, 1], f32), T([256], f32), T([1, 256, 1, 1], f32), T([256], f32), T([], i64), T([1, 256, 1, 1], f32), T([1, 256, 1, 1], f32), T([256], f32), T([1, 256, 1, 1], f32), T([256], f32), T([], i64), T([1, 256, 1, 1], f32), T([1, 256, 1, 1], f32), T([256], f32), T([1, 256, 1, 1], f32), T([256], f32), T([], i64), T([1, 256, 1, 1], f32), T([1, 256, 1, 1], f32), T([256], f32), T([1, 256, 1, 1], f32), T([256], f32), T([], i64), T([1, 512, 1, 1], f32), T([1, 512, 1, 1], f32), T([512], f32), T([1, 512, 1, 1], f32), T([512], f32), T([], i64), T([1, 512, 1, 1], f32), T([1, 512, 1, 1], f32), T([512], f32), T([1, 512, 1, 1], f32), T([512], f32), T([], i64), T([1, 512, 1, 1], f32), T([1, 512, 1, 1], f32), T([512], f32), T([1, 512, 1, 1], f32), T([512], f32), T([], i64), T([1, 512, 1, 1], f32), T([1, 512, 1, 1], f32), T([512], f32), T([1, 512, 1, 1], f32), T([512], f32), T([], i64), T([1, 512, 1, 1], f32), T([1, 512, 1, 1], f32), T([512], f32), T([1, 512, 1, 1], f32), T([512], f32), T([1024, 65], f32))"

class Repro(torch.nn.Module):
    def forward(self, primals_3: "i64[]", getitem_1: "f32[1, 64, 1, 1]", rsqrt: "f32[1, 64, 1, 1]", primals_4: "f32[64]", getitem: "f32[1, 64, 1, 1]", primals_5: "f32[64]", primals_9: "i64[]", getitem_3: "f32[1, 64, 1, 1]", rsqrt_1: "f32[1, 64, 1, 1]", primals_10: "f32[64]", getitem_2: "f32[1, 64, 1, 1]", primals_11: "f32[64]", primals_15: "i64[]", getitem_5: "f32[1, 64, 1, 1]", rsqrt_2: "f32[1, 64, 1, 1]", primals_16: "f32[64]", getitem_4: "f32[1, 64, 1, 1]", primals_17: "f32[64]", primals_21: "i64[]", getitem_7: "f32[1, 64, 1, 1]", rsqrt_3: "f32[1, 64, 1, 1]", primals_22: "f32[64]", getitem_6: "f32[1, 64, 1, 1]", primals_23: "f32[64]", primals_27: "i64[]", getitem_9: "f32[1, 64, 1, 1]", rsqrt_4: "f32[1, 64, 1, 1]", primals_28: "f32[64]", getitem_8: "f32[1, 64, 1, 1]", primals_29: "f32[64]", primals_33: "i64[]", getitem_11: "f32[1, 64, 1, 1]", rsqrt_5: "f32[1, 64, 1, 1]", primals_34: "f32[64]", getitem_10: "f32[1, 64, 1, 1]", primals_35: "f32[64]", primals_39: "i64[]", getitem_13: "f32[1, 128, 1, 1]", rsqrt_6: "f32[1, 128, 1, 1]", primals_40: "f32[128]", getitem_12: "f32[1, 128, 1, 1]", primals_41: "f32[128]", primals_45: "i64[]", getitem_15: "f32[1, 128, 1, 1]", rsqrt_7: "f32[1, 128, 1, 1]", primals_46: "f32[128]", getitem_14: "f32[1, 128, 1, 1]", primals_47: "f32[128]", primals_51: "i64[]", getitem_17: "f32[1, 128, 1, 1]", rsqrt_8: "f32[1, 128, 1, 1]", primals_52: "f32[128]", getitem_16: "f32[1, 128, 1, 1]", primals_53: "f32[128]", primals_57: "i64[]", getitem_19: "f32[1, 128, 1, 1]", rsqrt_9: "f32[1, 128, 1, 1]", primals_58: "f32[128]", getitem_18: "f32[1, 128, 1, 1]", primals_59: "f32[128]", primals_63: "i64[]", getitem_21: "f32[1, 128, 1, 1]", rsqrt_10: "f32[1, 128, 1, 1]", primals_64: "f32[128]", getitem_20: "f32[1, 128, 1, 1]", primals_65: "f32[128]", primals_69: "i64[]", getitem_23: "f32[1, 256, 1, 1]", rsqrt_11: "f32[1, 256, 1, 1]", primals_70: "f32[256]", getitem_22: "f32[1, 256, 1, 1]", primals_71: "f32[256]", primals_75: "i64[]", getitem_25: "f32[1, 256, 1, 1]", rsqrt_12: "f32[1, 256, 1, 1]", primals_76: "f32[256]", getitem_24: "f32[1, 256, 1, 1]", primals_77: "f32[256]", primals_81: "i64[]", getitem_27: "f32[1, 256, 1, 1]", rsqrt_13: "f32[1, 256, 1, 1]", primals_82: "f32[256]", getitem_26: "f32[1, 256, 1, 1]", primals_83: "f32[256]", primals_87: "i64[]", getitem_29: "f32[1, 256, 1, 1]", rsqrt_14: "f32[1, 256, 1, 1]", primals_88: "f32[256]", getitem_28: "f32[1, 256, 1, 1]", primals_89: "f32[256]", primals_93: "i64[]", getitem_31: "f32[1, 256, 1, 1]", rsqrt_15: "f32[1, 256, 1, 1]", primals_94: "f32[256]", getitem_30: "f32[1, 256, 1, 1]", primals_95: "f32[256]", primals_99: "i64[]", getitem_33: "f32[1, 512, 1, 1]", rsqrt_16: "f32[1, 512, 1, 1]", primals_100: "f32[512]", getitem_32: "f32[1, 512, 1, 1]", primals_101: "f32[512]", primals_105: "i64[]", getitem_35: "f32[1, 512, 1, 1]", rsqrt_17: "f32[1, 512, 1, 1]", primals_106: "f32[512]", getitem_34: "f32[1, 512, 1, 1]", primals_107: "f32[512]", primals_111: "i64[]", getitem_37: "f32[1, 512, 1, 1]", rsqrt_18: "f32[1, 512, 1, 1]", primals_112: "f32[512]", getitem_36: "f32[1, 512, 1, 1]", primals_113: "f32[512]", primals_117: "i64[]", getitem_39: "f32[1, 512, 1, 1]", rsqrt_19: "f32[1, 512, 1, 1]", primals_118: "f32[512]", getitem_38: "f32[1, 512, 1, 1]", primals_119: "f32[512]", primals_123: "i64[]", getitem_41: "f32[1, 512, 1, 1]", rsqrt_20: "f32[1, 512, 1, 1]", primals_124: "f32[512]", getitem_40: "f32[1, 512, 1, 1]", primals_125: "f32[512]", addmm: "f32[1024, 65]"):
        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:128 in forward, code: x = F.relu(self.bn1(self.conv1(x)))
        add_tensor: "i64[]" = torch.ops.aten.add.Tensor(primals_3, 1)
        squeeze_dims: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_1, [0, 2, 3]);  getitem_1 = None
        squeeze_dims_1: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt, [0, 2, 3]);  rsqrt = None
        mul_tensor: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_dims, 0.1)
        mul_tensor_1: "f32[64]" = torch.ops.aten.mul.Tensor(primals_4, 0.9)
        add_tensor_1: "f32[64]" = torch.ops.aten.add.Tensor(mul_tensor, mul_tensor_1);  mul_tensor = mul_tensor_1 = None
        squeeze_dims_2: "f32[64]" = torch.ops.aten.squeeze.dims(getitem, [0, 2, 3]);  getitem = None
        mul_tensor_2: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_dims_2, 1.000000238418636);  squeeze_dims_2 = None
        mul_tensor_3: "f32[64]" = torch.ops.aten.mul.Tensor(mul_tensor_2, 0.1);  mul_tensor_2 = None
        mul_tensor_4: "f32[64]" = torch.ops.aten.mul.Tensor(primals_5, 0.9)
        add_tensor_2: "f32[64]" = torch.ops.aten.add.Tensor(mul_tensor_3, mul_tensor_4);  mul_tensor_3 = mul_tensor_4 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:54 in forward, code: out = F.relu(self.bn1(self.conv1(x)))
        add_tensor_3: "i64[]" = torch.ops.aten.add.Tensor(primals_9, 1)
        squeeze_dims_3: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_3, [0, 2, 3]);  getitem_3 = None
        squeeze_dims_4: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_1, [0, 2, 3]);  rsqrt_1 = None
        mul_tensor_5: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_dims_3, 0.1)
        mul_tensor_6: "f32[64]" = torch.ops.aten.mul.Tensor(primals_10, 0.9)
        add_tensor_4: "f32[64]" = torch.ops.aten.add.Tensor(mul_tensor_5, mul_tensor_6);  mul_tensor_5 = mul_tensor_6 = None
        squeeze_dims_5: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_2, [0, 2, 3]);  getitem_2 = None
        mul_tensor_7: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_dims_5, 1.000000953675226);  squeeze_dims_5 = None
        mul_tensor_8: "f32[64]" = torch.ops.aten.mul.Tensor(mul_tensor_7, 0.1);  mul_tensor_7 = None
        mul_tensor_9: "f32[64]" = torch.ops.aten.mul.Tensor(primals_11, 0.9)
        add_tensor_5: "f32[64]" = torch.ops.aten.add.Tensor(mul_tensor_8, mul_tensor_9);  mul_tensor_8 = mul_tensor_9 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:55 in forward, code: out = self.bn2(self.conv2(out))
        add_tensor_6: "i64[]" = torch.ops.aten.add.Tensor(primals_15, 1)
        squeeze_dims_6: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_5, [0, 2, 3]);  getitem_5 = None
        squeeze_dims_7: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_2, [0, 2, 3]);  rsqrt_2 = None
        mul_tensor_10: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_dims_6, 0.1)
        mul_tensor_11: "f32[64]" = torch.ops.aten.mul.Tensor(primals_16, 0.9)
        add_tensor_7: "f32[64]" = torch.ops.aten.add.Tensor(mul_tensor_10, mul_tensor_11);  mul_tensor_10 = mul_tensor_11 = None
        squeeze_dims_8: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_4, [0, 2, 3]);  getitem_4 = None
        mul_tensor_12: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_dims_8, 1.000000953675226);  squeeze_dims_8 = None
        mul_tensor_13: "f32[64]" = torch.ops.aten.mul.Tensor(mul_tensor_12, 0.1);  mul_tensor_12 = None
        mul_tensor_14: "f32[64]" = torch.ops.aten.mul.Tensor(primals_17, 0.9)
        add_tensor_8: "f32[64]" = torch.ops.aten.add.Tensor(mul_tensor_13, mul_tensor_14);  mul_tensor_13 = mul_tensor_14 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:56 in forward, code: out += self.shortcut(x)
        add_tensor_9: "i64[]" = torch.ops.aten.add.Tensor(primals_21, 1)
        squeeze_dims_9: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_7, [0, 2, 3]);  getitem_7 = None
        squeeze_dims_10: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_3, [0, 2, 3]);  rsqrt_3 = None
        mul_tensor_15: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_dims_9, 0.1)
        mul_tensor_16: "f32[64]" = torch.ops.aten.mul.Tensor(primals_22, 0.9)
        add_tensor_10: "f32[64]" = torch.ops.aten.add.Tensor(mul_tensor_15, mul_tensor_16);  mul_tensor_15 = mul_tensor_16 = None
        squeeze_dims_11: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_6, [0, 2, 3]);  getitem_6 = None
        mul_tensor_17: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_dims_11, 1.000000953675226);  squeeze_dims_11 = None
        mul_tensor_18: "f32[64]" = torch.ops.aten.mul.Tensor(mul_tensor_17, 0.1);  mul_tensor_17 = None
        mul_tensor_19: "f32[64]" = torch.ops.aten.mul.Tensor(primals_23, 0.9)
        add_tensor_11: "f32[64]" = torch.ops.aten.add.Tensor(mul_tensor_18, mul_tensor_19);  mul_tensor_18 = mul_tensor_19 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:54 in forward, code: out = F.relu(self.bn1(self.conv1(x)))
        add_tensor_12: "i64[]" = torch.ops.aten.add.Tensor(primals_27, 1)
        squeeze_dims_12: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_9, [0, 2, 3]);  getitem_9 = None
        squeeze_dims_13: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_4, [0, 2, 3]);  rsqrt_4 = None
        mul_tensor_20: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_dims_12, 0.1)
        mul_tensor_21: "f32[64]" = torch.ops.aten.mul.Tensor(primals_28, 0.9)
        add_tensor_13: "f32[64]" = torch.ops.aten.add.Tensor(mul_tensor_20, mul_tensor_21);  mul_tensor_20 = mul_tensor_21 = None
        squeeze_dims_14: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_8, [0, 2, 3]);  getitem_8 = None
        mul_tensor_22: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_dims_14, 1.000000953675226);  squeeze_dims_14 = None
        mul_tensor_23: "f32[64]" = torch.ops.aten.mul.Tensor(mul_tensor_22, 0.1);  mul_tensor_22 = None
        mul_tensor_24: "f32[64]" = torch.ops.aten.mul.Tensor(primals_29, 0.9)
        add_tensor_14: "f32[64]" = torch.ops.aten.add.Tensor(mul_tensor_23, mul_tensor_24);  mul_tensor_23 = mul_tensor_24 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:55 in forward, code: out = self.bn2(self.conv2(out))
        add_tensor_15: "i64[]" = torch.ops.aten.add.Tensor(primals_33, 1)
        squeeze_dims_15: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_11, [0, 2, 3]);  getitem_11 = None
        squeeze_dims_16: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_5, [0, 2, 3]);  rsqrt_5 = None
        mul_tensor_25: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_dims_15, 0.1)
        mul_tensor_26: "f32[64]" = torch.ops.aten.mul.Tensor(primals_34, 0.9)
        add_tensor_16: "f32[64]" = torch.ops.aten.add.Tensor(mul_tensor_25, mul_tensor_26);  mul_tensor_25 = mul_tensor_26 = None
        squeeze_dims_17: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_10, [0, 2, 3]);  getitem_10 = None
        mul_tensor_27: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_dims_17, 1.000000953675226);  squeeze_dims_17 = None
        mul_tensor_28: "f32[64]" = torch.ops.aten.mul.Tensor(mul_tensor_27, 0.1);  mul_tensor_27 = None
        mul_tensor_29: "f32[64]" = torch.ops.aten.mul.Tensor(primals_35, 0.9)
        add_tensor_17: "f32[64]" = torch.ops.aten.add.Tensor(mul_tensor_28, mul_tensor_29);  mul_tensor_28 = mul_tensor_29 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:54 in forward, code: out = F.relu(self.bn1(self.conv1(x)))
        add_tensor_18: "i64[]" = torch.ops.aten.add.Tensor(primals_39, 1)
        squeeze_dims_18: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_13, [0, 2, 3]);  getitem_13 = None
        squeeze_dims_19: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_6, [0, 2, 3]);  rsqrt_6 = None
        mul_tensor_30: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_dims_18, 0.1)
        mul_tensor_31: "f32[128]" = torch.ops.aten.mul.Tensor(primals_40, 0.9)
        add_tensor_19: "f32[128]" = torch.ops.aten.add.Tensor(mul_tensor_30, mul_tensor_31);  mul_tensor_30 = mul_tensor_31 = None
        squeeze_dims_20: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_12, [0, 2, 3]);  getitem_12 = None
        mul_tensor_32: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_dims_20, 1.0000038147118175);  squeeze_dims_20 = None
        mul_tensor_33: "f32[128]" = torch.ops.aten.mul.Tensor(mul_tensor_32, 0.1);  mul_tensor_32 = None
        mul_tensor_34: "f32[128]" = torch.ops.aten.mul.Tensor(primals_41, 0.9)
        add_tensor_20: "f32[128]" = torch.ops.aten.add.Tensor(mul_tensor_33, mul_tensor_34);  mul_tensor_33 = mul_tensor_34 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:55 in forward, code: out = self.bn2(self.conv2(out))
        add_tensor_21: "i64[]" = torch.ops.aten.add.Tensor(primals_45, 1)
        squeeze_dims_21: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_15, [0, 2, 3]);  getitem_15 = None
        squeeze_dims_22: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_7, [0, 2, 3]);  rsqrt_7 = None
        mul_tensor_35: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_dims_21, 0.1)
        mul_tensor_36: "f32[128]" = torch.ops.aten.mul.Tensor(primals_46, 0.9)
        add_tensor_22: "f32[128]" = torch.ops.aten.add.Tensor(mul_tensor_35, mul_tensor_36);  mul_tensor_35 = mul_tensor_36 = None
        squeeze_dims_23: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_14, [0, 2, 3]);  getitem_14 = None
        mul_tensor_37: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_dims_23, 1.0000038147118175);  squeeze_dims_23 = None
        mul_tensor_38: "f32[128]" = torch.ops.aten.mul.Tensor(mul_tensor_37, 0.1);  mul_tensor_37 = None
        mul_tensor_39: "f32[128]" = torch.ops.aten.mul.Tensor(primals_47, 0.9)
        add_tensor_23: "f32[128]" = torch.ops.aten.add.Tensor(mul_tensor_38, mul_tensor_39);  mul_tensor_38 = mul_tensor_39 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:56 in forward, code: out += self.shortcut(x)
        add_tensor_24: "i64[]" = torch.ops.aten.add.Tensor(primals_51, 1)
        squeeze_dims_24: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_17, [0, 2, 3]);  getitem_17 = None
        squeeze_dims_25: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_8, [0, 2, 3]);  rsqrt_8 = None
        mul_tensor_40: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_dims_24, 0.1)
        mul_tensor_41: "f32[128]" = torch.ops.aten.mul.Tensor(primals_52, 0.9)
        add_tensor_25: "f32[128]" = torch.ops.aten.add.Tensor(mul_tensor_40, mul_tensor_41);  mul_tensor_40 = mul_tensor_41 = None
        squeeze_dims_26: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_16, [0, 2, 3]);  getitem_16 = None
        mul_tensor_42: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_dims_26, 1.0000038147118175);  squeeze_dims_26 = None
        mul_tensor_43: "f32[128]" = torch.ops.aten.mul.Tensor(mul_tensor_42, 0.1);  mul_tensor_42 = None
        mul_tensor_44: "f32[128]" = torch.ops.aten.mul.Tensor(primals_53, 0.9)
        add_tensor_26: "f32[128]" = torch.ops.aten.add.Tensor(mul_tensor_43, mul_tensor_44);  mul_tensor_43 = mul_tensor_44 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:54 in forward, code: out = F.relu(self.bn1(self.conv1(x)))
        add_tensor_27: "i64[]" = torch.ops.aten.add.Tensor(primals_57, 1)
        squeeze_dims_27: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_19, [0, 2, 3]);  getitem_19 = None
        squeeze_dims_28: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_9, [0, 2, 3]);  rsqrt_9 = None
        mul_tensor_45: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_dims_27, 0.1)
        mul_tensor_46: "f32[128]" = torch.ops.aten.mul.Tensor(primals_58, 0.9)
        add_tensor_28: "f32[128]" = torch.ops.aten.add.Tensor(mul_tensor_45, mul_tensor_46);  mul_tensor_45 = mul_tensor_46 = None
        squeeze_dims_29: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_18, [0, 2, 3]);  getitem_18 = None
        mul_tensor_47: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_dims_29, 1.0000038147118175);  squeeze_dims_29 = None
        mul_tensor_48: "f32[128]" = torch.ops.aten.mul.Tensor(mul_tensor_47, 0.1);  mul_tensor_47 = None
        mul_tensor_49: "f32[128]" = torch.ops.aten.mul.Tensor(primals_59, 0.9)
        add_tensor_29: "f32[128]" = torch.ops.aten.add.Tensor(mul_tensor_48, mul_tensor_49);  mul_tensor_48 = mul_tensor_49 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:55 in forward, code: out = self.bn2(self.conv2(out))
        add_tensor_30: "i64[]" = torch.ops.aten.add.Tensor(primals_63, 1)
        squeeze_dims_30: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_21, [0, 2, 3]);  getitem_21 = None
        squeeze_dims_31: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_10, [0, 2, 3]);  rsqrt_10 = None
        mul_tensor_50: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_dims_30, 0.1)
        mul_tensor_51: "f32[128]" = torch.ops.aten.mul.Tensor(primals_64, 0.9)
        add_tensor_31: "f32[128]" = torch.ops.aten.add.Tensor(mul_tensor_50, mul_tensor_51);  mul_tensor_50 = mul_tensor_51 = None
        squeeze_dims_32: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_20, [0, 2, 3]);  getitem_20 = None
        mul_tensor_52: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_dims_32, 1.0000038147118175);  squeeze_dims_32 = None
        mul_tensor_53: "f32[128]" = torch.ops.aten.mul.Tensor(mul_tensor_52, 0.1);  mul_tensor_52 = None
        mul_tensor_54: "f32[128]" = torch.ops.aten.mul.Tensor(primals_65, 0.9)
        add_tensor_32: "f32[128]" = torch.ops.aten.add.Tensor(mul_tensor_53, mul_tensor_54);  mul_tensor_53 = mul_tensor_54 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:54 in forward, code: out = F.relu(self.bn1(self.conv1(x)))
        add_tensor_33: "i64[]" = torch.ops.aten.add.Tensor(primals_69, 1)
        squeeze_dims_33: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_23, [0, 2, 3]);  getitem_23 = None
        squeeze_dims_34: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_11, [0, 2, 3]);  rsqrt_11 = None
        mul_tensor_55: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_dims_33, 0.1)
        mul_tensor_56: "f32[256]" = torch.ops.aten.mul.Tensor(primals_70, 0.9)
        add_tensor_34: "f32[256]" = torch.ops.aten.add.Tensor(mul_tensor_55, mul_tensor_56);  mul_tensor_55 = mul_tensor_56 = None
        squeeze_dims_35: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_22, [0, 2, 3]);  getitem_22 = None
        mul_tensor_57: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_dims_35, 1.0000152590218967);  squeeze_dims_35 = None
        mul_tensor_58: "f32[256]" = torch.ops.aten.mul.Tensor(mul_tensor_57, 0.1);  mul_tensor_57 = None
        mul_tensor_59: "f32[256]" = torch.ops.aten.mul.Tensor(primals_71, 0.9)
        add_tensor_35: "f32[256]" = torch.ops.aten.add.Tensor(mul_tensor_58, mul_tensor_59);  mul_tensor_58 = mul_tensor_59 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:55 in forward, code: out = self.bn2(self.conv2(out))
        add_tensor_36: "i64[]" = torch.ops.aten.add.Tensor(primals_75, 1)
        squeeze_dims_36: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_25, [0, 2, 3]);  getitem_25 = None
        squeeze_dims_37: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_12, [0, 2, 3]);  rsqrt_12 = None
        mul_tensor_60: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_dims_36, 0.1)
        mul_tensor_61: "f32[256]" = torch.ops.aten.mul.Tensor(primals_76, 0.9)
        add_tensor_37: "f32[256]" = torch.ops.aten.add.Tensor(mul_tensor_60, mul_tensor_61);  mul_tensor_60 = mul_tensor_61 = None
        squeeze_dims_38: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_24, [0, 2, 3]);  getitem_24 = None
        mul_tensor_62: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_dims_38, 1.0000152590218967);  squeeze_dims_38 = None
        mul_tensor_63: "f32[256]" = torch.ops.aten.mul.Tensor(mul_tensor_62, 0.1);  mul_tensor_62 = None
        mul_tensor_64: "f32[256]" = torch.ops.aten.mul.Tensor(primals_77, 0.9)
        add_tensor_38: "f32[256]" = torch.ops.aten.add.Tensor(mul_tensor_63, mul_tensor_64);  mul_tensor_63 = mul_tensor_64 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:56 in forward, code: out += self.shortcut(x)
        add_tensor_39: "i64[]" = torch.ops.aten.add.Tensor(primals_81, 1)
        squeeze_dims_39: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_27, [0, 2, 3]);  getitem_27 = None
        squeeze_dims_40: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_13, [0, 2, 3]);  rsqrt_13 = None
        mul_tensor_65: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_dims_39, 0.1)
        mul_tensor_66: "f32[256]" = torch.ops.aten.mul.Tensor(primals_82, 0.9)
        add_tensor_40: "f32[256]" = torch.ops.aten.add.Tensor(mul_tensor_65, mul_tensor_66);  mul_tensor_65 = mul_tensor_66 = None
        squeeze_dims_41: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_26, [0, 2, 3]);  getitem_26 = None
        mul_tensor_67: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_dims_41, 1.0000152590218967);  squeeze_dims_41 = None
        mul_tensor_68: "f32[256]" = torch.ops.aten.mul.Tensor(mul_tensor_67, 0.1);  mul_tensor_67 = None
        mul_tensor_69: "f32[256]" = torch.ops.aten.mul.Tensor(primals_83, 0.9)
        add_tensor_41: "f32[256]" = torch.ops.aten.add.Tensor(mul_tensor_68, mul_tensor_69);  mul_tensor_68 = mul_tensor_69 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:54 in forward, code: out = F.relu(self.bn1(self.conv1(x)))
        add_tensor_42: "i64[]" = torch.ops.aten.add.Tensor(primals_87, 1)
        squeeze_dims_42: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_29, [0, 2, 3]);  getitem_29 = None
        squeeze_dims_43: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_14, [0, 2, 3]);  rsqrt_14 = None
        mul_tensor_70: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_dims_42, 0.1)
        mul_tensor_71: "f32[256]" = torch.ops.aten.mul.Tensor(primals_88, 0.9)
        add_tensor_43: "f32[256]" = torch.ops.aten.add.Tensor(mul_tensor_70, mul_tensor_71);  mul_tensor_70 = mul_tensor_71 = None
        squeeze_dims_44: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_28, [0, 2, 3]);  getitem_28 = None
        mul_tensor_72: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_dims_44, 1.0000152590218967);  squeeze_dims_44 = None
        mul_tensor_73: "f32[256]" = torch.ops.aten.mul.Tensor(mul_tensor_72, 0.1);  mul_tensor_72 = None
        mul_tensor_74: "f32[256]" = torch.ops.aten.mul.Tensor(primals_89, 0.9)
        add_tensor_44: "f32[256]" = torch.ops.aten.add.Tensor(mul_tensor_73, mul_tensor_74);  mul_tensor_73 = mul_tensor_74 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:55 in forward, code: out = self.bn2(self.conv2(out))
        add_tensor_45: "i64[]" = torch.ops.aten.add.Tensor(primals_93, 1)
        squeeze_dims_45: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_31, [0, 2, 3]);  getitem_31 = None
        squeeze_dims_46: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_15, [0, 2, 3]);  rsqrt_15 = None
        mul_tensor_75: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_dims_45, 0.1)
        mul_tensor_76: "f32[256]" = torch.ops.aten.mul.Tensor(primals_94, 0.9)
        add_tensor_46: "f32[256]" = torch.ops.aten.add.Tensor(mul_tensor_75, mul_tensor_76);  mul_tensor_75 = mul_tensor_76 = None
        squeeze_dims_47: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_30, [0, 2, 3]);  getitem_30 = None
        mul_tensor_77: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_dims_47, 1.0000152590218967);  squeeze_dims_47 = None
        mul_tensor_78: "f32[256]" = torch.ops.aten.mul.Tensor(mul_tensor_77, 0.1);  mul_tensor_77 = None
        mul_tensor_79: "f32[256]" = torch.ops.aten.mul.Tensor(primals_95, 0.9)
        add_tensor_47: "f32[256]" = torch.ops.aten.add.Tensor(mul_tensor_78, mul_tensor_79);  mul_tensor_78 = mul_tensor_79 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:54 in forward, code: out = F.relu(self.bn1(self.conv1(x)))
        add_tensor_48: "i64[]" = torch.ops.aten.add.Tensor(primals_99, 1)
        squeeze_dims_48: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_33, [0, 2, 3]);  getitem_33 = None
        squeeze_dims_49: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_16, [0, 2, 3]);  rsqrt_16 = None
        mul_tensor_80: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_dims_48, 0.1)
        mul_tensor_81: "f32[512]" = torch.ops.aten.mul.Tensor(primals_100, 0.9)
        add_tensor_49: "f32[512]" = torch.ops.aten.add.Tensor(mul_tensor_80, mul_tensor_81);  mul_tensor_80 = mul_tensor_81 = None
        squeeze_dims_50: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_32, [0, 2, 3]);  getitem_32 = None
        mul_tensor_82: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_dims_50, 1.0000610388817677);  squeeze_dims_50 = None
        mul_tensor_83: "f32[512]" = torch.ops.aten.mul.Tensor(mul_tensor_82, 0.1);  mul_tensor_82 = None
        mul_tensor_84: "f32[512]" = torch.ops.aten.mul.Tensor(primals_101, 0.9)
        add_tensor_50: "f32[512]" = torch.ops.aten.add.Tensor(mul_tensor_83, mul_tensor_84);  mul_tensor_83 = mul_tensor_84 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:55 in forward, code: out = self.bn2(self.conv2(out))
        add_tensor_51: "i64[]" = torch.ops.aten.add.Tensor(primals_105, 1)
        squeeze_dims_51: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_35, [0, 2, 3]);  getitem_35 = None
        squeeze_dims_52: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_17, [0, 2, 3]);  rsqrt_17 = None
        mul_tensor_85: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_dims_51, 0.1)
        mul_tensor_86: "f32[512]" = torch.ops.aten.mul.Tensor(primals_106, 0.9)
        add_tensor_52: "f32[512]" = torch.ops.aten.add.Tensor(mul_tensor_85, mul_tensor_86);  mul_tensor_85 = mul_tensor_86 = None
        squeeze_dims_53: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_34, [0, 2, 3]);  getitem_34 = None
        mul_tensor_87: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_dims_53, 1.0000610388817677);  squeeze_dims_53 = None
        mul_tensor_88: "f32[512]" = torch.ops.aten.mul.Tensor(mul_tensor_87, 0.1);  mul_tensor_87 = None
        mul_tensor_89: "f32[512]" = torch.ops.aten.mul.Tensor(primals_107, 0.9)
        add_tensor_53: "f32[512]" = torch.ops.aten.add.Tensor(mul_tensor_88, mul_tensor_89);  mul_tensor_88 = mul_tensor_89 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:56 in forward, code: out += self.shortcut(x)
        add_tensor_54: "i64[]" = torch.ops.aten.add.Tensor(primals_111, 1)
        squeeze_dims_54: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_37, [0, 2, 3]);  getitem_37 = None
        squeeze_dims_55: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_18, [0, 2, 3]);  rsqrt_18 = None
        mul_tensor_90: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_dims_54, 0.1)
        mul_tensor_91: "f32[512]" = torch.ops.aten.mul.Tensor(primals_112, 0.9)
        add_tensor_55: "f32[512]" = torch.ops.aten.add.Tensor(mul_tensor_90, mul_tensor_91);  mul_tensor_90 = mul_tensor_91 = None
        squeeze_dims_56: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_36, [0, 2, 3]);  getitem_36 = None
        mul_tensor_92: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_dims_56, 1.0000610388817677);  squeeze_dims_56 = None
        mul_tensor_93: "f32[512]" = torch.ops.aten.mul.Tensor(mul_tensor_92, 0.1);  mul_tensor_92 = None
        mul_tensor_94: "f32[512]" = torch.ops.aten.mul.Tensor(primals_113, 0.9)
        add_tensor_56: "f32[512]" = torch.ops.aten.add.Tensor(mul_tensor_93, mul_tensor_94);  mul_tensor_93 = mul_tensor_94 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:54 in forward, code: out = F.relu(self.bn1(self.conv1(x)))
        add_tensor_57: "i64[]" = torch.ops.aten.add.Tensor(primals_117, 1)
        squeeze_dims_57: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_39, [0, 2, 3]);  getitem_39 = None
        squeeze_dims_58: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_19, [0, 2, 3]);  rsqrt_19 = None
        mul_tensor_95: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_dims_57, 0.1)
        mul_tensor_96: "f32[512]" = torch.ops.aten.mul.Tensor(primals_118, 0.9)
        add_tensor_58: "f32[512]" = torch.ops.aten.add.Tensor(mul_tensor_95, mul_tensor_96);  mul_tensor_95 = mul_tensor_96 = None
        squeeze_dims_59: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_38, [0, 2, 3]);  getitem_38 = None
        mul_tensor_97: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_dims_59, 1.0000610388817677);  squeeze_dims_59 = None
        mul_tensor_98: "f32[512]" = torch.ops.aten.mul.Tensor(mul_tensor_97, 0.1);  mul_tensor_97 = None
        mul_tensor_99: "f32[512]" = torch.ops.aten.mul.Tensor(primals_119, 0.9)
        add_tensor_59: "f32[512]" = torch.ops.aten.add.Tensor(mul_tensor_98, mul_tensor_99);  mul_tensor_98 = mul_tensor_99 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:55 in forward, code: out = self.bn2(self.conv2(out))
        add_tensor_60: "i64[]" = torch.ops.aten.add.Tensor(primals_123, 1)
        squeeze_dims_60: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_41, [0, 2, 3]);  getitem_41 = None
        squeeze_dims_61: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_20, [0, 2, 3]);  rsqrt_20 = None
        mul_tensor_100: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_dims_60, 0.1)
        mul_tensor_101: "f32[512]" = torch.ops.aten.mul.Tensor(primals_124, 0.9)
        add_tensor_61: "f32[512]" = torch.ops.aten.add.Tensor(mul_tensor_100, mul_tensor_101);  mul_tensor_100 = mul_tensor_101 = None
        squeeze_dims_62: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_40, [0, 2, 3]);  getitem_40 = None
        mul_tensor_102: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_dims_62, 1.0000610388817677);  squeeze_dims_62 = None
        mul_tensor_103: "f32[512]" = torch.ops.aten.mul.Tensor(mul_tensor_102, 0.1);  mul_tensor_102 = None
        mul_tensor_104: "f32[512]" = torch.ops.aten.mul.Tensor(primals_125, 0.9)
        add_tensor_62: "f32[512]" = torch.ops.aten.add.Tensor(mul_tensor_103, mul_tensor_104);  mul_tensor_103 = mul_tensor_104 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:136 in forward, code: x = torch.sigmoid(x)
        sigmoid_default: "f32[1024, 65]" = torch.ops.aten.sigmoid.default(addmm);  addmm = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:55 in forward, code: out = self.bn2(self.conv2(out))
        unsqueeze_default: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_dims_60, 0);  squeeze_dims_60 = None
        unsqueeze_default_1: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, 2);  unsqueeze_default = None
        unsqueeze_default_2: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_1, 3);  unsqueeze_default_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:54 in forward, code: out = F.relu(self.bn1(self.conv1(x)))
        unsqueeze_default_3: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_dims_57, 0);  squeeze_dims_57 = None
        unsqueeze_default_4: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_3, 2);  unsqueeze_default_3 = None
        unsqueeze_default_5: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 3);  unsqueeze_default_4 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:56 in forward, code: out += self.shortcut(x)
        unsqueeze_default_6: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_dims_54, 0);  squeeze_dims_54 = None
        unsqueeze_default_7: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, 2);  unsqueeze_default_6 = None
        unsqueeze_default_8: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_7, 3);  unsqueeze_default_7 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:55 in forward, code: out = self.bn2(self.conv2(out))
        unsqueeze_default_9: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_dims_51, 0);  squeeze_dims_51 = None
        unsqueeze_default_10: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_9, 2);  unsqueeze_default_9 = None
        unsqueeze_default_11: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_10, 3);  unsqueeze_default_10 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:54 in forward, code: out = F.relu(self.bn1(self.conv1(x)))
        unsqueeze_default_12: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_dims_48, 0);  squeeze_dims_48 = None
        unsqueeze_default_13: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_12, 2);  unsqueeze_default_12 = None
        unsqueeze_default_14: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_13, 3);  unsqueeze_default_13 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:55 in forward, code: out = self.bn2(self.conv2(out))
        unsqueeze_default_15: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_dims_45, 0);  squeeze_dims_45 = None
        unsqueeze_default_16: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_15, 2);  unsqueeze_default_15 = None
        unsqueeze_default_17: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_16, 3);  unsqueeze_default_16 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:54 in forward, code: out = F.relu(self.bn1(self.conv1(x)))
        unsqueeze_default_18: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_dims_42, 0);  squeeze_dims_42 = None
        unsqueeze_default_19: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_18, 2);  unsqueeze_default_18 = None
        unsqueeze_default_20: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_19, 3);  unsqueeze_default_19 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:56 in forward, code: out += self.shortcut(x)
        unsqueeze_default_21: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_dims_39, 0);  squeeze_dims_39 = None
        unsqueeze_default_22: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_21, 2);  unsqueeze_default_21 = None
        unsqueeze_default_23: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_22, 3);  unsqueeze_default_22 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:55 in forward, code: out = self.bn2(self.conv2(out))
        unsqueeze_default_24: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_dims_36, 0);  squeeze_dims_36 = None
        unsqueeze_default_25: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_24, 2);  unsqueeze_default_24 = None
        unsqueeze_default_26: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_25, 3);  unsqueeze_default_25 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:54 in forward, code: out = F.relu(self.bn1(self.conv1(x)))
        unsqueeze_default_27: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_dims_33, 0);  squeeze_dims_33 = None
        unsqueeze_default_28: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_27, 2);  unsqueeze_default_27 = None
        unsqueeze_default_29: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_28, 3);  unsqueeze_default_28 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:55 in forward, code: out = self.bn2(self.conv2(out))
        unsqueeze_default_30: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_dims_30, 0);  squeeze_dims_30 = None
        unsqueeze_default_31: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_30, 2);  unsqueeze_default_30 = None
        unsqueeze_default_32: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_31, 3);  unsqueeze_default_31 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:54 in forward, code: out = F.relu(self.bn1(self.conv1(x)))
        unsqueeze_default_33: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_dims_27, 0);  squeeze_dims_27 = None
        unsqueeze_default_34: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_33, 2);  unsqueeze_default_33 = None
        unsqueeze_default_35: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_34, 3);  unsqueeze_default_34 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:56 in forward, code: out += self.shortcut(x)
        unsqueeze_default_36: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_dims_24, 0);  squeeze_dims_24 = None
        unsqueeze_default_37: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_36, 2);  unsqueeze_default_36 = None
        unsqueeze_default_38: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_37, 3);  unsqueeze_default_37 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:55 in forward, code: out = self.bn2(self.conv2(out))
        unsqueeze_default_39: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_dims_21, 0);  squeeze_dims_21 = None
        unsqueeze_default_40: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_39, 2);  unsqueeze_default_39 = None
        unsqueeze_default_41: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_40, 3);  unsqueeze_default_40 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:54 in forward, code: out = F.relu(self.bn1(self.conv1(x)))
        unsqueeze_default_42: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_dims_18, 0);  squeeze_dims_18 = None
        unsqueeze_default_43: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_42, 2);  unsqueeze_default_42 = None
        unsqueeze_default_44: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_43, 3);  unsqueeze_default_43 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:55 in forward, code: out = self.bn2(self.conv2(out))
        unsqueeze_default_45: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_dims_15, 0);  squeeze_dims_15 = None
        unsqueeze_default_46: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_45, 2);  unsqueeze_default_45 = None
        unsqueeze_default_47: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_46, 3);  unsqueeze_default_46 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:54 in forward, code: out = F.relu(self.bn1(self.conv1(x)))
        unsqueeze_default_48: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_dims_12, 0);  squeeze_dims_12 = None
        unsqueeze_default_49: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_48, 2);  unsqueeze_default_48 = None
        unsqueeze_default_50: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_49, 3);  unsqueeze_default_49 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:56 in forward, code: out += self.shortcut(x)
        unsqueeze_default_51: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_dims_9, 0);  squeeze_dims_9 = None
        unsqueeze_default_52: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_51, 2);  unsqueeze_default_51 = None
        unsqueeze_default_53: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_52, 3);  unsqueeze_default_52 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:55 in forward, code: out = self.bn2(self.conv2(out))
        unsqueeze_default_54: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_dims_6, 0);  squeeze_dims_6 = None
        unsqueeze_default_55: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_54, 2);  unsqueeze_default_54 = None
        unsqueeze_default_56: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_55, 3);  unsqueeze_default_55 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:54 in forward, code: out = F.relu(self.bn1(self.conv1(x)))
        unsqueeze_default_57: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_dims_3, 0);  squeeze_dims_3 = None
        unsqueeze_default_58: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_57, 2);  unsqueeze_default_57 = None
        unsqueeze_default_59: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_58, 3);  unsqueeze_default_58 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:128 in forward, code: x = F.relu(self.bn1(self.conv1(x)))
        unsqueeze_default_60: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_dims, 0);  squeeze_dims = None
        unsqueeze_default_61: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_60, 2);  unsqueeze_default_60 = None
        unsqueeze_default_62: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_61, 3);  unsqueeze_default_61 = None

        # No stacktrace found for following nodes
        copy__default: "i64[]" = torch.ops.aten.copy_.default(primals_3, add_tensor);  primals_3 = add_tensor = None
        copy__default_1: "f32[64]" = torch.ops.aten.copy_.default(primals_4, add_tensor_1);  primals_4 = add_tensor_1 = None
        copy__default_2: "f32[64]" = torch.ops.aten.copy_.default(primals_5, add_tensor_2);  primals_5 = add_tensor_2 = None
        copy__default_3: "i64[]" = torch.ops.aten.copy_.default(primals_9, add_tensor_3);  primals_9 = add_tensor_3 = None
        copy__default_4: "f32[64]" = torch.ops.aten.copy_.default(primals_10, add_tensor_4);  primals_10 = add_tensor_4 = None
        copy__default_5: "f32[64]" = torch.ops.aten.copy_.default(primals_11, add_tensor_5);  primals_11 = add_tensor_5 = None
        copy__default_6: "i64[]" = torch.ops.aten.copy_.default(primals_15, add_tensor_6);  primals_15 = add_tensor_6 = None
        copy__default_7: "f32[64]" = torch.ops.aten.copy_.default(primals_16, add_tensor_7);  primals_16 = add_tensor_7 = None
        copy__default_8: "f32[64]" = torch.ops.aten.copy_.default(primals_17, add_tensor_8);  primals_17 = add_tensor_8 = None
        copy__default_9: "i64[]" = torch.ops.aten.copy_.default(primals_21, add_tensor_9);  primals_21 = add_tensor_9 = None
        copy__default_10: "f32[64]" = torch.ops.aten.copy_.default(primals_22, add_tensor_10);  primals_22 = add_tensor_10 = None
        copy__default_11: "f32[64]" = torch.ops.aten.copy_.default(primals_23, add_tensor_11);  primals_23 = add_tensor_11 = None
        copy__default_12: "i64[]" = torch.ops.aten.copy_.default(primals_27, add_tensor_12);  primals_27 = add_tensor_12 = None
        copy__default_13: "f32[64]" = torch.ops.aten.copy_.default(primals_28, add_tensor_13);  primals_28 = add_tensor_13 = None
        copy__default_14: "f32[64]" = torch.ops.aten.copy_.default(primals_29, add_tensor_14);  primals_29 = add_tensor_14 = None
        copy__default_15: "i64[]" = torch.ops.aten.copy_.default(primals_33, add_tensor_15);  primals_33 = add_tensor_15 = None
        copy__default_16: "f32[64]" = torch.ops.aten.copy_.default(primals_34, add_tensor_16);  primals_34 = add_tensor_16 = None
        copy__default_17: "f32[64]" = torch.ops.aten.copy_.default(primals_35, add_tensor_17);  primals_35 = add_tensor_17 = None
        copy__default_18: "i64[]" = torch.ops.aten.copy_.default(primals_39, add_tensor_18);  primals_39 = add_tensor_18 = None
        copy__default_19: "f32[128]" = torch.ops.aten.copy_.default(primals_40, add_tensor_19);  primals_40 = add_tensor_19 = None
        copy__default_20: "f32[128]" = torch.ops.aten.copy_.default(primals_41, add_tensor_20);  primals_41 = add_tensor_20 = None
        copy__default_21: "i64[]" = torch.ops.aten.copy_.default(primals_45, add_tensor_21);  primals_45 = add_tensor_21 = None
        copy__default_22: "f32[128]" = torch.ops.aten.copy_.default(primals_46, add_tensor_22);  primals_46 = add_tensor_22 = None
        copy__default_23: "f32[128]" = torch.ops.aten.copy_.default(primals_47, add_tensor_23);  primals_47 = add_tensor_23 = None
        copy__default_24: "i64[]" = torch.ops.aten.copy_.default(primals_51, add_tensor_24);  primals_51 = add_tensor_24 = None
        copy__default_25: "f32[128]" = torch.ops.aten.copy_.default(primals_52, add_tensor_25);  primals_52 = add_tensor_25 = None
        copy__default_26: "f32[128]" = torch.ops.aten.copy_.default(primals_53, add_tensor_26);  primals_53 = add_tensor_26 = None
        copy__default_27: "i64[]" = torch.ops.aten.copy_.default(primals_57, add_tensor_27);  primals_57 = add_tensor_27 = None
        copy__default_28: "f32[128]" = torch.ops.aten.copy_.default(primals_58, add_tensor_28);  primals_58 = add_tensor_28 = None
        copy__default_29: "f32[128]" = torch.ops.aten.copy_.default(primals_59, add_tensor_29);  primals_59 = add_tensor_29 = None
        copy__default_30: "i64[]" = torch.ops.aten.copy_.default(primals_63, add_tensor_30);  primals_63 = add_tensor_30 = None
        copy__default_31: "f32[128]" = torch.ops.aten.copy_.default(primals_64, add_tensor_31);  primals_64 = add_tensor_31 = None
        copy__default_32: "f32[128]" = torch.ops.aten.copy_.default(primals_65, add_tensor_32);  primals_65 = add_tensor_32 = None
        copy__default_33: "i64[]" = torch.ops.aten.copy_.default(primals_69, add_tensor_33);  primals_69 = add_tensor_33 = None
        copy__default_34: "f32[256]" = torch.ops.aten.copy_.default(primals_70, add_tensor_34);  primals_70 = add_tensor_34 = None
        copy__default_35: "f32[256]" = torch.ops.aten.copy_.default(primals_71, add_tensor_35);  primals_71 = add_tensor_35 = None
        copy__default_36: "i64[]" = torch.ops.aten.copy_.default(primals_75, add_tensor_36);  primals_75 = add_tensor_36 = None
        copy__default_37: "f32[256]" = torch.ops.aten.copy_.default(primals_76, add_tensor_37);  primals_76 = add_tensor_37 = None
        copy__default_38: "f32[256]" = torch.ops.aten.copy_.default(primals_77, add_tensor_38);  primals_77 = add_tensor_38 = None
        copy__default_39: "i64[]" = torch.ops.aten.copy_.default(primals_81, add_tensor_39);  primals_81 = add_tensor_39 = None
        copy__default_40: "f32[256]" = torch.ops.aten.copy_.default(primals_82, add_tensor_40);  primals_82 = add_tensor_40 = None
        copy__default_41: "f32[256]" = torch.ops.aten.copy_.default(primals_83, add_tensor_41);  primals_83 = add_tensor_41 = None
        copy__default_42: "i64[]" = torch.ops.aten.copy_.default(primals_87, add_tensor_42);  primals_87 = add_tensor_42 = None
        copy__default_43: "f32[256]" = torch.ops.aten.copy_.default(primals_88, add_tensor_43);  primals_88 = add_tensor_43 = None
        copy__default_44: "f32[256]" = torch.ops.aten.copy_.default(primals_89, add_tensor_44);  primals_89 = add_tensor_44 = None
        copy__default_45: "i64[]" = torch.ops.aten.copy_.default(primals_93, add_tensor_45);  primals_93 = add_tensor_45 = None
        copy__default_46: "f32[256]" = torch.ops.aten.copy_.default(primals_94, add_tensor_46);  primals_94 = add_tensor_46 = None
        copy__default_47: "f32[256]" = torch.ops.aten.copy_.default(primals_95, add_tensor_47);  primals_95 = add_tensor_47 = None
        copy__default_48: "i64[]" = torch.ops.aten.copy_.default(primals_99, add_tensor_48);  primals_99 = add_tensor_48 = None
        copy__default_49: "f32[512]" = torch.ops.aten.copy_.default(primals_100, add_tensor_49);  primals_100 = add_tensor_49 = None
        copy__default_50: "f32[512]" = torch.ops.aten.copy_.default(primals_101, add_tensor_50);  primals_101 = add_tensor_50 = None
        copy__default_51: "i64[]" = torch.ops.aten.copy_.default(primals_105, add_tensor_51);  primals_105 = add_tensor_51 = None
        copy__default_52: "f32[512]" = torch.ops.aten.copy_.default(primals_106, add_tensor_52);  primals_106 = add_tensor_52 = None
        copy__default_53: "f32[512]" = torch.ops.aten.copy_.default(primals_107, add_tensor_53);  primals_107 = add_tensor_53 = None
        copy__default_54: "i64[]" = torch.ops.aten.copy_.default(primals_111, add_tensor_54);  primals_111 = add_tensor_54 = None
        copy__default_55: "f32[512]" = torch.ops.aten.copy_.default(primals_112, add_tensor_55);  primals_112 = add_tensor_55 = None
        copy__default_56: "f32[512]" = torch.ops.aten.copy_.default(primals_113, add_tensor_56);  primals_113 = add_tensor_56 = None
        copy__default_57: "i64[]" = torch.ops.aten.copy_.default(primals_117, add_tensor_57);  primals_117 = add_tensor_57 = None
        copy__default_58: "f32[512]" = torch.ops.aten.copy_.default(primals_118, add_tensor_58);  primals_118 = add_tensor_58 = None
        copy__default_59: "f32[512]" = torch.ops.aten.copy_.default(primals_119, add_tensor_59);  primals_119 = add_tensor_59 = None
        copy__default_60: "i64[]" = torch.ops.aten.copy_.default(primals_123, add_tensor_60);  primals_123 = add_tensor_60 = None
        copy__default_61: "f32[512]" = torch.ops.aten.copy_.default(primals_124, add_tensor_61);  primals_124 = add_tensor_61 = None
        copy__default_62: "f32[512]" = torch.ops.aten.copy_.default(primals_125, add_tensor_62);  primals_125 = add_tensor_62 = None
        return (squeeze_dims_1, squeeze_dims_4, squeeze_dims_7, squeeze_dims_10, squeeze_dims_13, squeeze_dims_16, squeeze_dims_19, squeeze_dims_22, squeeze_dims_25, squeeze_dims_28, squeeze_dims_31, squeeze_dims_34, squeeze_dims_37, squeeze_dims_40, squeeze_dims_43, squeeze_dims_46, squeeze_dims_49, squeeze_dims_52, squeeze_dims_55, squeeze_dims_58, squeeze_dims_61, sigmoid_default, unsqueeze_default_2, unsqueeze_default_5, unsqueeze_default_8, unsqueeze_default_11, unsqueeze_default_14, unsqueeze_default_17, unsqueeze_default_20, unsqueeze_default_23, unsqueeze_default_26, unsqueeze_default_29, unsqueeze_default_32, unsqueeze_default_35, unsqueeze_default_38, unsqueeze_default_41, unsqueeze_default_44, unsqueeze_default_47, unsqueeze_default_50, unsqueeze_default_53, unsqueeze_default_56, unsqueeze_default_59, unsqueeze_default_62, copy__default, copy__default_1, copy__default_2, copy__default_3, copy__default_4, copy__default_5, copy__default_6, copy__default_7, copy__default_8, copy__default_9, copy__default_10, copy__default_11, copy__default_12, copy__default_13, copy__default_14, copy__default_15, copy__default_16, copy__default_17, copy__default_18, copy__default_19, copy__default_20, copy__default_21, copy__default_22, copy__default_23, copy__default_24, copy__default_25, copy__default_26, copy__default_27, copy__default_28, copy__default_29, copy__default_30, copy__default_31, copy__default_32, copy__default_33, copy__default_34, copy__default_35, copy__default_36, copy__default_37, copy__default_38, copy__default_39, copy__default_40, copy__default_41, copy__default_42, copy__default_43, copy__default_44, copy__default_45, copy__default_46, copy__default_47, copy__default_48, copy__default_49, copy__default_50, copy__default_51, copy__default_52, copy__default_53, copy__default_54, copy__default_55, copy__default_56, copy__default_57, copy__default_58, copy__default_59, copy__default_60, copy__default_61, copy__default_62)



def _default_make_inputs():
    from repro_harness import parse_shapes_config
    return parse_shapes_config(_shapes_config)


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
