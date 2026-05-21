"""
Standalone repro captured via capture_hook.
Label: torchbench_pytorch_unet_train
Pattern hash: ed74cf0d5da7
Shape hash: b9250767
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
    def forward(self, primals_4: "i64[]", getitem_1: "f32[1, 64, 1, 1]", rsqrt: "f32[1, 64, 1, 1]", primals_5: "f32[64]", getitem: "f32[1, 64, 1, 1]", primals_6: "f32[64]", primals_11: "i64[]", getitem_3: "f32[1, 64, 1, 1]", primals_12: "f32[64]", getitem_2: "f32[1, 64, 1, 1]", primals_13: "f32[64]", primals_18: "i64[]", getitem_7: "f32[1, 128, 1, 1]", rsqrt_2: "f32[1, 128, 1, 1]", primals_19: "f32[128]", getitem_6: "f32[1, 128, 1, 1]", primals_20: "f32[128]", primals_25: "i64[]", getitem_9: "f32[1, 128, 1, 1]", primals_26: "f32[128]", getitem_8: "f32[1, 128, 1, 1]", primals_27: "f32[128]", primals_32: "i64[]", getitem_13: "f32[1, 256, 1, 1]", rsqrt_4: "f32[1, 256, 1, 1]", primals_33: "f32[256]", getitem_12: "f32[1, 256, 1, 1]", primals_34: "f32[256]", primals_39: "i64[]", getitem_15: "f32[1, 256, 1, 1]", primals_40: "f32[256]", getitem_14: "f32[1, 256, 1, 1]", primals_41: "f32[256]", primals_46: "i64[]", getitem_19: "f32[1, 512, 1, 1]", rsqrt_6: "f32[1, 512, 1, 1]", primals_47: "f32[512]", getitem_18: "f32[1, 512, 1, 1]", primals_48: "f32[512]", primals_53: "i64[]", getitem_21: "f32[1, 512, 1, 1]", primals_54: "f32[512]", getitem_20: "f32[1, 512, 1, 1]", primals_55: "f32[512]", primals_60: "i64[]", getitem_25: "f32[1, 512, 1, 1]", rsqrt_8: "f32[1, 512, 1, 1]", primals_61: "f32[512]", getitem_24: "f32[1, 512, 1, 1]", primals_62: "f32[512]", primals_67: "i64[]", getitem_27: "f32[1, 512, 1, 1]", primals_68: "f32[512]", getitem_26: "f32[1, 512, 1, 1]", primals_69: "f32[512]", primals_74: "i64[]", getitem_29: "f32[1, 512, 1, 1]", rsqrt_10: "f32[1, 512, 1, 1]", primals_75: "f32[512]", getitem_28: "f32[1, 512, 1, 1]", primals_76: "f32[512]", primals_81: "i64[]", getitem_31: "f32[1, 256, 1, 1]", primals_82: "f32[256]", getitem_30: "f32[1, 256, 1, 1]", primals_83: "f32[256]", primals_88: "i64[]", getitem_33: "f32[1, 256, 1, 1]", rsqrt_12: "f32[1, 256, 1, 1]", primals_89: "f32[256]", getitem_32: "f32[1, 256, 1, 1]", primals_90: "f32[256]", primals_95: "i64[]", getitem_35: "f32[1, 128, 1, 1]", primals_96: "f32[128]", getitem_34: "f32[1, 128, 1, 1]", primals_97: "f32[128]", primals_102: "i64[]", getitem_37: "f32[1, 128, 1, 1]", rsqrt_14: "f32[1, 128, 1, 1]", primals_103: "f32[128]", getitem_36: "f32[1, 128, 1, 1]", primals_104: "f32[128]", primals_109: "i64[]", getitem_39: "f32[1, 64, 1, 1]", primals_110: "f32[64]", getitem_38: "f32[1, 64, 1, 1]", primals_111: "f32[64]", primals_116: "i64[]", getitem_41: "f32[1, 64, 1, 1]", rsqrt_16: "f32[1, 64, 1, 1]", primals_117: "f32[64]", getitem_40: "f32[1, 64, 1, 1]", primals_118: "f32[64]", primals_123: "i64[]", convolution_17: "f32[8, 64, 640, 959]", primals_124: "f32[64]", primals_125: "f32[64]", primals_126: "f32[64]", primals_127: "f32[64]"):
        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pytorch_unet/pytorch_unet/unet/unet_parts.py:25 in forward, code: return self.double_conv(x)
        add_tensor: "i64[]" = torch.ops.aten.add.Tensor(primals_4, 1)
        squeeze_dims: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_1, [0, 2, 3]);  getitem_1 = None
        squeeze_dims_1: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt, [0, 2, 3]);  rsqrt = None
        mul_tensor: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_dims, 0.1)
        mul_tensor_1: "f32[64]" = torch.ops.aten.mul.Tensor(primals_5, 0.9)
        add_tensor_1: "f32[64]" = torch.ops.aten.add.Tensor(mul_tensor, mul_tensor_1);  mul_tensor = mul_tensor_1 = None
        squeeze_dims_2: "f32[64]" = torch.ops.aten.squeeze.dims(getitem, [0, 2, 3]);  getitem = None
        mul_tensor_2: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_dims_2, 1.000000203662711);  squeeze_dims_2 = None
        mul_tensor_3: "f32[64]" = torch.ops.aten.mul.Tensor(mul_tensor_2, 0.1);  mul_tensor_2 = None
        mul_tensor_4: "f32[64]" = torch.ops.aten.mul.Tensor(primals_6, 0.9)
        add_tensor_2: "f32[64]" = torch.ops.aten.add.Tensor(mul_tensor_3, mul_tensor_4);  mul_tensor_3 = mul_tensor_4 = None
        add_tensor_3: "i64[]" = torch.ops.aten.add.Tensor(primals_11, 1)
        squeeze_dims_3: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_3, [0, 2, 3]);  getitem_3 = None
        mul_tensor_5: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_dims_3, 0.1);  squeeze_dims_3 = None
        mul_tensor_6: "f32[64]" = torch.ops.aten.mul.Tensor(primals_12, 0.9)
        add_tensor_4: "f32[64]" = torch.ops.aten.add.Tensor(mul_tensor_5, mul_tensor_6);  mul_tensor_5 = mul_tensor_6 = None
        squeeze_dims_4: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_2, [0, 2, 3]);  getitem_2 = None
        mul_tensor_7: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_dims_4, 1.000000203662711);  squeeze_dims_4 = None
        mul_tensor_8: "f32[64]" = torch.ops.aten.mul.Tensor(mul_tensor_7, 0.1);  mul_tensor_7 = None
        mul_tensor_9: "f32[64]" = torch.ops.aten.mul.Tensor(primals_13, 0.9)
        add_tensor_5: "f32[64]" = torch.ops.aten.add.Tensor(mul_tensor_8, mul_tensor_9);  mul_tensor_8 = mul_tensor_9 = None
        add_tensor_6: "i64[]" = torch.ops.aten.add.Tensor(primals_18, 1)
        squeeze_dims_5: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_7, [0, 2, 3]);  getitem_7 = None
        squeeze_dims_6: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_2, [0, 2, 3]);  rsqrt_2 = None
        mul_tensor_10: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_dims_5, 0.1)
        mul_tensor_11: "f32[128]" = torch.ops.aten.mul.Tensor(primals_19, 0.9)
        add_tensor_7: "f32[128]" = torch.ops.aten.add.Tensor(mul_tensor_10, mul_tensor_11);  mul_tensor_10 = mul_tensor_11 = None
        squeeze_dims_7: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_6, [0, 2, 3]);  getitem_6 = None
        mul_tensor_12: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_dims_7, 1.0000008155017088);  squeeze_dims_7 = None
        mul_tensor_13: "f32[128]" = torch.ops.aten.mul.Tensor(mul_tensor_12, 0.1);  mul_tensor_12 = None
        mul_tensor_14: "f32[128]" = torch.ops.aten.mul.Tensor(primals_20, 0.9)
        add_tensor_8: "f32[128]" = torch.ops.aten.add.Tensor(mul_tensor_13, mul_tensor_14);  mul_tensor_13 = mul_tensor_14 = None
        add_tensor_9: "i64[]" = torch.ops.aten.add.Tensor(primals_25, 1)
        squeeze_dims_8: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_9, [0, 2, 3]);  getitem_9 = None
        mul_tensor_15: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_dims_8, 0.1);  squeeze_dims_8 = None
        mul_tensor_16: "f32[128]" = torch.ops.aten.mul.Tensor(primals_26, 0.9)
        add_tensor_10: "f32[128]" = torch.ops.aten.add.Tensor(mul_tensor_15, mul_tensor_16);  mul_tensor_15 = mul_tensor_16 = None
        squeeze_dims_9: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_8, [0, 2, 3]);  getitem_8 = None
        mul_tensor_17: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_dims_9, 1.0000008155017088);  squeeze_dims_9 = None
        mul_tensor_18: "f32[128]" = torch.ops.aten.mul.Tensor(mul_tensor_17, 0.1);  mul_tensor_17 = None
        mul_tensor_19: "f32[128]" = torch.ops.aten.mul.Tensor(primals_27, 0.9)
        add_tensor_11: "f32[128]" = torch.ops.aten.add.Tensor(mul_tensor_18, mul_tensor_19);  mul_tensor_18 = mul_tensor_19 = None
        add_tensor_12: "i64[]" = torch.ops.aten.add.Tensor(primals_32, 1)
        squeeze_dims_10: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_13, [0, 2, 3]);  getitem_13 = None
        squeeze_dims_11: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_4, [0, 2, 3]);  rsqrt_4 = None
        mul_tensor_20: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_dims_10, 0.1)
        mul_tensor_21: "f32[256]" = torch.ops.aten.mul.Tensor(primals_33, 0.9)
        add_tensor_13: "f32[256]" = torch.ops.aten.add.Tensor(mul_tensor_20, mul_tensor_21);  mul_tensor_20 = mul_tensor_21 = None
        squeeze_dims_12: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_12, [0, 2, 3]);  getitem_12 = None
        mul_tensor_22: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_dims_12, 1.0000032688391371);  squeeze_dims_12 = None
        mul_tensor_23: "f32[256]" = torch.ops.aten.mul.Tensor(mul_tensor_22, 0.1);  mul_tensor_22 = None
        mul_tensor_24: "f32[256]" = torch.ops.aten.mul.Tensor(primals_34, 0.9)
        add_tensor_14: "f32[256]" = torch.ops.aten.add.Tensor(mul_tensor_23, mul_tensor_24);  mul_tensor_23 = mul_tensor_24 = None
        add_tensor_15: "i64[]" = torch.ops.aten.add.Tensor(primals_39, 1)
        squeeze_dims_13: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_15, [0, 2, 3]);  getitem_15 = None
        mul_tensor_25: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_dims_13, 0.1);  squeeze_dims_13 = None
        mul_tensor_26: "f32[256]" = torch.ops.aten.mul.Tensor(primals_40, 0.9)
        add_tensor_16: "f32[256]" = torch.ops.aten.add.Tensor(mul_tensor_25, mul_tensor_26);  mul_tensor_25 = mul_tensor_26 = None
        squeeze_dims_14: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_14, [0, 2, 3]);  getitem_14 = None
        mul_tensor_27: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_dims_14, 1.0000032688391371);  squeeze_dims_14 = None
        mul_tensor_28: "f32[256]" = torch.ops.aten.mul.Tensor(mul_tensor_27, 0.1);  mul_tensor_27 = None
        mul_tensor_29: "f32[256]" = torch.ops.aten.mul.Tensor(primals_41, 0.9)
        add_tensor_17: "f32[256]" = torch.ops.aten.add.Tensor(mul_tensor_28, mul_tensor_29);  mul_tensor_28 = mul_tensor_29 = None
        add_tensor_18: "i64[]" = torch.ops.aten.add.Tensor(primals_46, 1)
        squeeze_dims_15: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_19, [0, 2, 3]);  getitem_19 = None
        squeeze_dims_16: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_6, [0, 2, 3]);  rsqrt_6 = None
        mul_tensor_30: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_dims_15, 0.1)
        mul_tensor_31: "f32[512]" = torch.ops.aten.mul.Tensor(primals_47, 0.9)
        add_tensor_19: "f32[512]" = torch.ops.aten.add.Tensor(mul_tensor_30, mul_tensor_31);  mul_tensor_30 = mul_tensor_31 = None
        squeeze_dims_17: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_18, [0, 2, 3]);  getitem_18 = None
        mul_tensor_32: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_dims_17, 1.0000131304245066);  squeeze_dims_17 = None
        mul_tensor_33: "f32[512]" = torch.ops.aten.mul.Tensor(mul_tensor_32, 0.1);  mul_tensor_32 = None
        mul_tensor_34: "f32[512]" = torch.ops.aten.mul.Tensor(primals_48, 0.9)
        add_tensor_20: "f32[512]" = torch.ops.aten.add.Tensor(mul_tensor_33, mul_tensor_34);  mul_tensor_33 = mul_tensor_34 = None
        add_tensor_21: "i64[]" = torch.ops.aten.add.Tensor(primals_53, 1)
        squeeze_dims_18: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_21, [0, 2, 3]);  getitem_21 = None
        mul_tensor_35: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_dims_18, 0.1);  squeeze_dims_18 = None
        mul_tensor_36: "f32[512]" = torch.ops.aten.mul.Tensor(primals_54, 0.9)
        add_tensor_22: "f32[512]" = torch.ops.aten.add.Tensor(mul_tensor_35, mul_tensor_36);  mul_tensor_35 = mul_tensor_36 = None
        squeeze_dims_19: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_20, [0, 2, 3]);  getitem_20 = None
        mul_tensor_37: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_dims_19, 1.0000131304245066);  squeeze_dims_19 = None
        mul_tensor_38: "f32[512]" = torch.ops.aten.mul.Tensor(mul_tensor_37, 0.1);  mul_tensor_37 = None
        mul_tensor_39: "f32[512]" = torch.ops.aten.mul.Tensor(primals_55, 0.9)
        add_tensor_23: "f32[512]" = torch.ops.aten.add.Tensor(mul_tensor_38, mul_tensor_39);  mul_tensor_38 = mul_tensor_39 = None
        add_tensor_24: "i64[]" = torch.ops.aten.add.Tensor(primals_60, 1)
        squeeze_dims_20: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_25, [0, 2, 3]);  getitem_25 = None
        squeeze_dims_21: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_8, [0, 2, 3]);  rsqrt_8 = None
        mul_tensor_40: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_dims_20, 0.1)
        mul_tensor_41: "f32[512]" = torch.ops.aten.mul.Tensor(primals_61, 0.9)
        add_tensor_25: "f32[512]" = torch.ops.aten.add.Tensor(mul_tensor_40, mul_tensor_41);  mul_tensor_40 = mul_tensor_41 = None
        squeeze_dims_22: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_24, [0, 2, 3]);  getitem_24 = None
        mul_tensor_42: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_dims_22, 1.0000529689072515);  squeeze_dims_22 = None
        mul_tensor_43: "f32[512]" = torch.ops.aten.mul.Tensor(mul_tensor_42, 0.1);  mul_tensor_42 = None
        mul_tensor_44: "f32[512]" = torch.ops.aten.mul.Tensor(primals_62, 0.9)
        add_tensor_26: "f32[512]" = torch.ops.aten.add.Tensor(mul_tensor_43, mul_tensor_44);  mul_tensor_43 = mul_tensor_44 = None
        add_tensor_27: "i64[]" = torch.ops.aten.add.Tensor(primals_67, 1)
        squeeze_dims_23: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_27, [0, 2, 3]);  getitem_27 = None
        mul_tensor_45: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_dims_23, 0.1);  squeeze_dims_23 = None
        mul_tensor_46: "f32[512]" = torch.ops.aten.mul.Tensor(primals_68, 0.9)
        add_tensor_28: "f32[512]" = torch.ops.aten.add.Tensor(mul_tensor_45, mul_tensor_46);  mul_tensor_45 = mul_tensor_46 = None
        squeeze_dims_24: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_26, [0, 2, 3]);  getitem_26 = None
        mul_tensor_47: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_dims_24, 1.0000529689072515);  squeeze_dims_24 = None
        mul_tensor_48: "f32[512]" = torch.ops.aten.mul.Tensor(mul_tensor_47, 0.1);  mul_tensor_47 = None
        mul_tensor_49: "f32[512]" = torch.ops.aten.mul.Tensor(primals_69, 0.9)
        add_tensor_29: "f32[512]" = torch.ops.aten.add.Tensor(mul_tensor_48, mul_tensor_49);  mul_tensor_48 = mul_tensor_49 = None
        add_tensor_30: "i64[]" = torch.ops.aten.add.Tensor(primals_74, 1)
        squeeze_dims_25: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_29, [0, 2, 3]);  getitem_29 = None
        squeeze_dims_26: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_10, [0, 2, 3]);  rsqrt_10 = None
        mul_tensor_50: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_dims_25, 0.1)
        mul_tensor_51: "f32[512]" = torch.ops.aten.mul.Tensor(primals_75, 0.9)
        add_tensor_31: "f32[512]" = torch.ops.aten.add.Tensor(mul_tensor_50, mul_tensor_51);  mul_tensor_50 = mul_tensor_51 = None
        squeeze_dims_27: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_28, [0, 2, 3]);  getitem_28 = None
        mul_tensor_52: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_dims_27, 1.0000131304245066);  squeeze_dims_27 = None
        mul_tensor_53: "f32[512]" = torch.ops.aten.mul.Tensor(mul_tensor_52, 0.1);  mul_tensor_52 = None
        mul_tensor_54: "f32[512]" = torch.ops.aten.mul.Tensor(primals_76, 0.9)
        add_tensor_32: "f32[512]" = torch.ops.aten.add.Tensor(mul_tensor_53, mul_tensor_54);  mul_tensor_53 = mul_tensor_54 = None
        add_tensor_33: "i64[]" = torch.ops.aten.add.Tensor(primals_81, 1)
        squeeze_dims_28: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_31, [0, 2, 3]);  getitem_31 = None
        mul_tensor_55: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_dims_28, 0.1);  squeeze_dims_28 = None
        mul_tensor_56: "f32[256]" = torch.ops.aten.mul.Tensor(primals_82, 0.9)
        add_tensor_34: "f32[256]" = torch.ops.aten.add.Tensor(mul_tensor_55, mul_tensor_56);  mul_tensor_55 = mul_tensor_56 = None
        squeeze_dims_29: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_30, [0, 2, 3]);  getitem_30 = None
        mul_tensor_57: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_dims_29, 1.0000131304245066);  squeeze_dims_29 = None
        mul_tensor_58: "f32[256]" = torch.ops.aten.mul.Tensor(mul_tensor_57, 0.1);  mul_tensor_57 = None
        mul_tensor_59: "f32[256]" = torch.ops.aten.mul.Tensor(primals_83, 0.9)
        add_tensor_35: "f32[256]" = torch.ops.aten.add.Tensor(mul_tensor_58, mul_tensor_59);  mul_tensor_58 = mul_tensor_59 = None
        add_tensor_36: "i64[]" = torch.ops.aten.add.Tensor(primals_88, 1)
        squeeze_dims_30: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_33, [0, 2, 3]);  getitem_33 = None
        squeeze_dims_31: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_12, [0, 2, 3]);  rsqrt_12 = None
        mul_tensor_60: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_dims_30, 0.1)
        mul_tensor_61: "f32[256]" = torch.ops.aten.mul.Tensor(primals_89, 0.9)
        add_tensor_37: "f32[256]" = torch.ops.aten.add.Tensor(mul_tensor_60, mul_tensor_61);  mul_tensor_60 = mul_tensor_61 = None
        squeeze_dims_32: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_32, [0, 2, 3]);  getitem_32 = None
        mul_tensor_62: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_dims_32, 1.0000032688391371);  squeeze_dims_32 = None
        mul_tensor_63: "f32[256]" = torch.ops.aten.mul.Tensor(mul_tensor_62, 0.1);  mul_tensor_62 = None
        mul_tensor_64: "f32[256]" = torch.ops.aten.mul.Tensor(primals_90, 0.9)
        add_tensor_38: "f32[256]" = torch.ops.aten.add.Tensor(mul_tensor_63, mul_tensor_64);  mul_tensor_63 = mul_tensor_64 = None
        add_tensor_39: "i64[]" = torch.ops.aten.add.Tensor(primals_95, 1)
        squeeze_dims_33: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_35, [0, 2, 3]);  getitem_35 = None
        mul_tensor_65: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_dims_33, 0.1);  squeeze_dims_33 = None
        mul_tensor_66: "f32[128]" = torch.ops.aten.mul.Tensor(primals_96, 0.9)
        add_tensor_40: "f32[128]" = torch.ops.aten.add.Tensor(mul_tensor_65, mul_tensor_66);  mul_tensor_65 = mul_tensor_66 = None
        squeeze_dims_34: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_34, [0, 2, 3]);  getitem_34 = None
        mul_tensor_67: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_dims_34, 1.0000032688391371);  squeeze_dims_34 = None
        mul_tensor_68: "f32[128]" = torch.ops.aten.mul.Tensor(mul_tensor_67, 0.1);  mul_tensor_67 = None
        mul_tensor_69: "f32[128]" = torch.ops.aten.mul.Tensor(primals_97, 0.9)
        add_tensor_41: "f32[128]" = torch.ops.aten.add.Tensor(mul_tensor_68, mul_tensor_69);  mul_tensor_68 = mul_tensor_69 = None
        add_tensor_42: "i64[]" = torch.ops.aten.add.Tensor(primals_102, 1)
        squeeze_dims_35: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_37, [0, 2, 3]);  getitem_37 = None
        squeeze_dims_36: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_14, [0, 2, 3]);  rsqrt_14 = None
        mul_tensor_70: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_dims_35, 0.1)
        mul_tensor_71: "f32[128]" = torch.ops.aten.mul.Tensor(primals_103, 0.9)
        add_tensor_43: "f32[128]" = torch.ops.aten.add.Tensor(mul_tensor_70, mul_tensor_71);  mul_tensor_70 = mul_tensor_71 = None
        squeeze_dims_37: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_36, [0, 2, 3]);  getitem_36 = None
        mul_tensor_72: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_dims_37, 1.0000008155017088);  squeeze_dims_37 = None
        mul_tensor_73: "f32[128]" = torch.ops.aten.mul.Tensor(mul_tensor_72, 0.1);  mul_tensor_72 = None
        mul_tensor_74: "f32[128]" = torch.ops.aten.mul.Tensor(primals_104, 0.9)
        add_tensor_44: "f32[128]" = torch.ops.aten.add.Tensor(mul_tensor_73, mul_tensor_74);  mul_tensor_73 = mul_tensor_74 = None
        add_tensor_45: "i64[]" = torch.ops.aten.add.Tensor(primals_109, 1)
        squeeze_dims_38: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_39, [0, 2, 3]);  getitem_39 = None
        mul_tensor_75: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_dims_38, 0.1);  squeeze_dims_38 = None
        mul_tensor_76: "f32[64]" = torch.ops.aten.mul.Tensor(primals_110, 0.9)
        add_tensor_46: "f32[64]" = torch.ops.aten.add.Tensor(mul_tensor_75, mul_tensor_76);  mul_tensor_75 = mul_tensor_76 = None
        squeeze_dims_39: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_38, [0, 2, 3]);  getitem_38 = None
        mul_tensor_77: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_dims_39, 1.0000008155017088);  squeeze_dims_39 = None
        mul_tensor_78: "f32[64]" = torch.ops.aten.mul.Tensor(mul_tensor_77, 0.1);  mul_tensor_77 = None
        mul_tensor_79: "f32[64]" = torch.ops.aten.mul.Tensor(primals_111, 0.9)
        add_tensor_47: "f32[64]" = torch.ops.aten.add.Tensor(mul_tensor_78, mul_tensor_79);  mul_tensor_78 = mul_tensor_79 = None
        add_tensor_48: "i64[]" = torch.ops.aten.add.Tensor(primals_116, 1)
        squeeze_dims_40: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_41, [0, 2, 3]);  getitem_41 = None
        squeeze_dims_41: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_16, [0, 2, 3]);  rsqrt_16 = None
        mul_tensor_80: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_dims_40, 0.1)
        mul_tensor_81: "f32[64]" = torch.ops.aten.mul.Tensor(primals_117, 0.9)
        add_tensor_49: "f32[64]" = torch.ops.aten.add.Tensor(mul_tensor_80, mul_tensor_81);  mul_tensor_80 = mul_tensor_81 = None
        squeeze_dims_42: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_40, [0, 2, 3]);  getitem_40 = None
        mul_tensor_82: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_dims_42, 1.000000203662711);  squeeze_dims_42 = None
        mul_tensor_83: "f32[64]" = torch.ops.aten.mul.Tensor(mul_tensor_82, 0.1);  mul_tensor_82 = None
        mul_tensor_84: "f32[64]" = torch.ops.aten.mul.Tensor(primals_118, 0.9)
        add_tensor_50: "f32[64]" = torch.ops.aten.add.Tensor(mul_tensor_83, mul_tensor_84);  mul_tensor_83 = mul_tensor_84 = None
        add_tensor_51: "i64[]" = torch.ops.aten.add.Tensor(primals_123, 1)
        var_mean_correction = torch.ops.aten.var_mean.correction(convolution_17, [0, 2, 3], correction = 0, keepdim = True)
        getitem_42: "f32[1, 64, 1, 1]" = var_mean_correction[0]
        getitem_43: "f32[1, 64, 1, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor_52: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem_42, 1e-05)
        rsqrt_default: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_tensor_52);  add_tensor_52 = None
        sub_tensor: "f32[8, 64, 640, 959]" = torch.ops.aten.sub.Tensor(convolution_17, getitem_43);  convolution_17 = None
        mul_tensor_85: "f32[8, 64, 640, 959]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = None
        squeeze_dims_43: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_43, [0, 2, 3]);  getitem_43 = None
        squeeze_dims_44: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_default, [0, 2, 3]);  rsqrt_default = None
        mul_tensor_86: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_dims_43, 0.1)
        mul_tensor_87: "f32[64]" = torch.ops.aten.mul.Tensor(primals_124, 0.9)
        add_tensor_53: "f32[64]" = torch.ops.aten.add.Tensor(mul_tensor_86, mul_tensor_87);  mul_tensor_86 = mul_tensor_87 = None
        squeeze_dims_45: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_42, [0, 2, 3]);  getitem_42 = None
        mul_tensor_88: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_dims_45, 1.000000203662711);  squeeze_dims_45 = None
        mul_tensor_89: "f32[64]" = torch.ops.aten.mul.Tensor(mul_tensor_88, 0.1);  mul_tensor_88 = None
        mul_tensor_90: "f32[64]" = torch.ops.aten.mul.Tensor(primals_125, 0.9)
        add_tensor_54: "f32[64]" = torch.ops.aten.add.Tensor(mul_tensor_89, mul_tensor_90);  mul_tensor_89 = mul_tensor_90 = None
        unsqueeze_default: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_126, -1);  primals_126 = None
        unsqueeze_default_1: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        mul_tensor_91: "f32[8, 64, 640, 959]" = torch.ops.aten.mul.Tensor(mul_tensor_85, unsqueeze_default_1);  mul_tensor_85 = unsqueeze_default_1 = None
        unsqueeze_default_2: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_127, -1);  primals_127 = None
        unsqueeze_default_3: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        add_tensor_55: "f32[8, 64, 640, 959]" = torch.ops.aten.add.Tensor(mul_tensor_91, unsqueeze_default_3);  mul_tensor_91 = unsqueeze_default_3 = None
        relu_default: "f32[8, 64, 640, 959]" = torch.ops.aten.relu.default(add_tensor_55);  add_tensor_55 = None
        unsqueeze_default_4: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_dims_43, 0);  squeeze_dims_43 = None
        unsqueeze_default_5: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 2);  unsqueeze_default_4 = None
        unsqueeze_default_6: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_5, 3);  unsqueeze_default_5 = None
        unsqueeze_default_7: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_dims_40, 0);  squeeze_dims_40 = None
        unsqueeze_default_8: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_7, 2);  unsqueeze_default_7 = None
        unsqueeze_default_9: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_8, 3);  unsqueeze_default_8 = None
        unsqueeze_default_10: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_dims_35, 0);  squeeze_dims_35 = None
        unsqueeze_default_11: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_10, 2);  unsqueeze_default_10 = None
        unsqueeze_default_12: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_11, 3);  unsqueeze_default_11 = None
        unsqueeze_default_13: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_dims_30, 0);  squeeze_dims_30 = None
        unsqueeze_default_14: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_13, 2);  unsqueeze_default_13 = None
        unsqueeze_default_15: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_14, 3);  unsqueeze_default_14 = None
        unsqueeze_default_16: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_dims_25, 0);  squeeze_dims_25 = None
        unsqueeze_default_17: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_16, 2);  unsqueeze_default_16 = None
        unsqueeze_default_18: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_17, 3);  unsqueeze_default_17 = None
        unsqueeze_default_19: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_dims_20, 0);  squeeze_dims_20 = None
        unsqueeze_default_20: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_19, 2);  unsqueeze_default_19 = None
        unsqueeze_default_21: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_20, 3);  unsqueeze_default_20 = None
        unsqueeze_default_22: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_dims_15, 0);  squeeze_dims_15 = None
        unsqueeze_default_23: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_22, 2);  unsqueeze_default_22 = None
        unsqueeze_default_24: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_23, 3);  unsqueeze_default_23 = None
        unsqueeze_default_25: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_dims_10, 0);  squeeze_dims_10 = None
        unsqueeze_default_26: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_25, 2);  unsqueeze_default_25 = None
        unsqueeze_default_27: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_26, 3);  unsqueeze_default_26 = None
        unsqueeze_default_28: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_dims_5, 0);  squeeze_dims_5 = None
        unsqueeze_default_29: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_28, 2);  unsqueeze_default_28 = None
        unsqueeze_default_30: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_29, 3);  unsqueeze_default_29 = None
        unsqueeze_default_31: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_dims, 0);  squeeze_dims = None
        unsqueeze_default_32: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_31, 2);  unsqueeze_default_31 = None
        unsqueeze_default_33: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_32, 3);  unsqueeze_default_32 = None

        # No stacktrace found for following nodes
        copy__default: "i64[]" = torch.ops.aten.copy_.default(primals_4, add_tensor);  primals_4 = add_tensor = None
        copy__default_1: "f32[64]" = torch.ops.aten.copy_.default(primals_5, add_tensor_1);  primals_5 = add_tensor_1 = None
        copy__default_2: "f32[64]" = torch.ops.aten.copy_.default(primals_6, add_tensor_2);  primals_6 = add_tensor_2 = None
        copy__default_3: "i64[]" = torch.ops.aten.copy_.default(primals_11, add_tensor_3);  primals_11 = add_tensor_3 = None
        copy__default_4: "f32[64]" = torch.ops.aten.copy_.default(primals_12, add_tensor_4);  primals_12 = add_tensor_4 = None
        copy__default_5: "f32[64]" = torch.ops.aten.copy_.default(primals_13, add_tensor_5);  primals_13 = add_tensor_5 = None
        copy__default_6: "i64[]" = torch.ops.aten.copy_.default(primals_18, add_tensor_6);  primals_18 = add_tensor_6 = None
        copy__default_7: "f32[128]" = torch.ops.aten.copy_.default(primals_19, add_tensor_7);  primals_19 = add_tensor_7 = None
        copy__default_8: "f32[128]" = torch.ops.aten.copy_.default(primals_20, add_tensor_8);  primals_20 = add_tensor_8 = None
        copy__default_9: "i64[]" = torch.ops.aten.copy_.default(primals_25, add_tensor_9);  primals_25 = add_tensor_9 = None
        copy__default_10: "f32[128]" = torch.ops.aten.copy_.default(primals_26, add_tensor_10);  primals_26 = add_tensor_10 = None
        copy__default_11: "f32[128]" = torch.ops.aten.copy_.default(primals_27, add_tensor_11);  primals_27 = add_tensor_11 = None
        copy__default_12: "i64[]" = torch.ops.aten.copy_.default(primals_32, add_tensor_12);  primals_32 = add_tensor_12 = None
        copy__default_13: "f32[256]" = torch.ops.aten.copy_.default(primals_33, add_tensor_13);  primals_33 = add_tensor_13 = None
        copy__default_14: "f32[256]" = torch.ops.aten.copy_.default(primals_34, add_tensor_14);  primals_34 = add_tensor_14 = None
        copy__default_15: "i64[]" = torch.ops.aten.copy_.default(primals_39, add_tensor_15);  primals_39 = add_tensor_15 = None
        copy__default_16: "f32[256]" = torch.ops.aten.copy_.default(primals_40, add_tensor_16);  primals_40 = add_tensor_16 = None
        copy__default_17: "f32[256]" = torch.ops.aten.copy_.default(primals_41, add_tensor_17);  primals_41 = add_tensor_17 = None
        copy__default_18: "i64[]" = torch.ops.aten.copy_.default(primals_46, add_tensor_18);  primals_46 = add_tensor_18 = None
        copy__default_19: "f32[512]" = torch.ops.aten.copy_.default(primals_47, add_tensor_19);  primals_47 = add_tensor_19 = None
        copy__default_20: "f32[512]" = torch.ops.aten.copy_.default(primals_48, add_tensor_20);  primals_48 = add_tensor_20 = None
        copy__default_21: "i64[]" = torch.ops.aten.copy_.default(primals_53, add_tensor_21);  primals_53 = add_tensor_21 = None
        copy__default_22: "f32[512]" = torch.ops.aten.copy_.default(primals_54, add_tensor_22);  primals_54 = add_tensor_22 = None
        copy__default_23: "f32[512]" = torch.ops.aten.copy_.default(primals_55, add_tensor_23);  primals_55 = add_tensor_23 = None
        copy__default_24: "i64[]" = torch.ops.aten.copy_.default(primals_60, add_tensor_24);  primals_60 = add_tensor_24 = None
        copy__default_25: "f32[512]" = torch.ops.aten.copy_.default(primals_61, add_tensor_25);  primals_61 = add_tensor_25 = None
        copy__default_26: "f32[512]" = torch.ops.aten.copy_.default(primals_62, add_tensor_26);  primals_62 = add_tensor_26 = None
        copy__default_27: "i64[]" = torch.ops.aten.copy_.default(primals_67, add_tensor_27);  primals_67 = add_tensor_27 = None
        copy__default_28: "f32[512]" = torch.ops.aten.copy_.default(primals_68, add_tensor_28);  primals_68 = add_tensor_28 = None
        copy__default_29: "f32[512]" = torch.ops.aten.copy_.default(primals_69, add_tensor_29);  primals_69 = add_tensor_29 = None
        copy__default_30: "i64[]" = torch.ops.aten.copy_.default(primals_74, add_tensor_30);  primals_74 = add_tensor_30 = None
        copy__default_31: "f32[512]" = torch.ops.aten.copy_.default(primals_75, add_tensor_31);  primals_75 = add_tensor_31 = None
        copy__default_32: "f32[512]" = torch.ops.aten.copy_.default(primals_76, add_tensor_32);  primals_76 = add_tensor_32 = None
        copy__default_33: "i64[]" = torch.ops.aten.copy_.default(primals_81, add_tensor_33);  primals_81 = add_tensor_33 = None
        copy__default_34: "f32[256]" = torch.ops.aten.copy_.default(primals_82, add_tensor_34);  primals_82 = add_tensor_34 = None
        copy__default_35: "f32[256]" = torch.ops.aten.copy_.default(primals_83, add_tensor_35);  primals_83 = add_tensor_35 = None
        copy__default_36: "i64[]" = torch.ops.aten.copy_.default(primals_88, add_tensor_36);  primals_88 = add_tensor_36 = None
        copy__default_37: "f32[256]" = torch.ops.aten.copy_.default(primals_89, add_tensor_37);  primals_89 = add_tensor_37 = None
        copy__default_38: "f32[256]" = torch.ops.aten.copy_.default(primals_90, add_tensor_38);  primals_90 = add_tensor_38 = None
        copy__default_39: "i64[]" = torch.ops.aten.copy_.default(primals_95, add_tensor_39);  primals_95 = add_tensor_39 = None
        copy__default_40: "f32[128]" = torch.ops.aten.copy_.default(primals_96, add_tensor_40);  primals_96 = add_tensor_40 = None
        copy__default_41: "f32[128]" = torch.ops.aten.copy_.default(primals_97, add_tensor_41);  primals_97 = add_tensor_41 = None
        copy__default_42: "i64[]" = torch.ops.aten.copy_.default(primals_102, add_tensor_42);  primals_102 = add_tensor_42 = None
        copy__default_43: "f32[128]" = torch.ops.aten.copy_.default(primals_103, add_tensor_43);  primals_103 = add_tensor_43 = None
        copy__default_44: "f32[128]" = torch.ops.aten.copy_.default(primals_104, add_tensor_44);  primals_104 = add_tensor_44 = None
        copy__default_45: "i64[]" = torch.ops.aten.copy_.default(primals_109, add_tensor_45);  primals_109 = add_tensor_45 = None
        copy__default_46: "f32[64]" = torch.ops.aten.copy_.default(primals_110, add_tensor_46);  primals_110 = add_tensor_46 = None
        copy__default_47: "f32[64]" = torch.ops.aten.copy_.default(primals_111, add_tensor_47);  primals_111 = add_tensor_47 = None
        copy__default_48: "i64[]" = torch.ops.aten.copy_.default(primals_116, add_tensor_48);  primals_116 = add_tensor_48 = None
        copy__default_49: "f32[64]" = torch.ops.aten.copy_.default(primals_117, add_tensor_49);  primals_117 = add_tensor_49 = None
        copy__default_50: "f32[64]" = torch.ops.aten.copy_.default(primals_118, add_tensor_50);  primals_118 = add_tensor_50 = None
        copy__default_51: "i64[]" = torch.ops.aten.copy_.default(primals_123, add_tensor_51);  primals_123 = add_tensor_51 = None
        copy__default_52: "f32[64]" = torch.ops.aten.copy_.default(primals_124, add_tensor_53);  primals_124 = add_tensor_53 = None
        copy__default_53: "f32[64]" = torch.ops.aten.copy_.default(primals_125, add_tensor_54);  primals_125 = add_tensor_54 = None
        return (squeeze_dims_1, squeeze_dims_6, squeeze_dims_11, squeeze_dims_16, squeeze_dims_21, squeeze_dims_26, squeeze_dims_31, squeeze_dims_36, squeeze_dims_41, squeeze_dims_44, relu_default, unsqueeze_default_6, unsqueeze_default_9, unsqueeze_default_12, unsqueeze_default_15, unsqueeze_default_18, unsqueeze_default_21, unsqueeze_default_24, unsqueeze_default_27, unsqueeze_default_30, unsqueeze_default_33, copy__default, copy__default_1, copy__default_2, copy__default_3, copy__default_4, copy__default_5, copy__default_6, copy__default_7, copy__default_8, copy__default_9, copy__default_10, copy__default_11, copy__default_12, copy__default_13, copy__default_14, copy__default_15, copy__default_16, copy__default_17, copy__default_18, copy__default_19, copy__default_20, copy__default_21, copy__default_22, copy__default_23, copy__default_24, copy__default_25, copy__default_26, copy__default_27, copy__default_28, copy__default_29, copy__default_30, copy__default_31, copy__default_32, copy__default_33, copy__default_34, copy__default_35, copy__default_36, copy__default_37, copy__default_38, copy__default_39, copy__default_40, copy__default_41, copy__default_42, copy__default_43, copy__default_44, copy__default_45, copy__default_46, copy__default_47, copy__default_48, copy__default_49, copy__default_50, copy__default_51, copy__default_52, copy__default_53)


def _default_make_inputs():
    return [
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 64, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 64, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([64], dtype=torch.float32, device='cuda'),
    torch.randn([1, 64, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([64], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 64, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([64], dtype=torch.float32, device='cuda'),
    torch.randn([1, 64, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([64], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 256, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 256, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([256], dtype=torch.float32, device='cuda'),
    torch.randn([1, 256, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([256], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 256, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([256], dtype=torch.float32, device='cuda'),
    torch.randn([1, 256, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([256], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 512, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 512, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([1, 512, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 512, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([1, 512, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 512, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 512, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([1, 512, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 512, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([1, 512, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 512, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 512, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([1, 512, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 256, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([256], dtype=torch.float32, device='cuda'),
    torch.randn([1, 256, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([256], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 256, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 256, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([256], dtype=torch.float32, device='cuda'),
    torch.randn([1, 256, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([256], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 64, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([64], dtype=torch.float32, device='cuda'),
    torch.randn([1, 64, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([64], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 64, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 64, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([64], dtype=torch.float32, device='cuda'),
    torch.randn([1, 64, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([64], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([8, 64, 640, 959], dtype=torch.float32, device='cuda'),
    torch.randn([64], dtype=torch.float32, device='cuda'),
    torch.randn([64], dtype=torch.float32, device='cuda'),
    torch.randn([64], dtype=torch.float32, device='cuda'),
    torch.randn([64], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
