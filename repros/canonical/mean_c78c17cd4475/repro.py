"""
Standalone repro captured via capture_hook.
Label: timm_visformer_small_training
Pattern hash: c78c17cd4475
Shape hash: 19a154e8
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
    def forward(self, primals_3: "i64[]", getitem_1: "f32[1, 32, 1, 1]", rsqrt: "f32[1, 32, 1, 1]", primals_4: "f32[32]", getitem: "f32[1, 32, 1, 1]", primals_5: "f32[32]", primals_10: "i64[]", getitem_3: "f32[1, 192, 1, 1]", primals_11: "f32[192]", getitem_2: "f32[1, 192, 1, 1]", primals_12: "f32[192]", primals_16: "i64[]", getitem_5: "f32[1, 192, 1, 1]", rsqrt_2: "f32[1, 192, 1, 1]", primals_17: "f32[192]", getitem_4: "f32[1, 192, 1, 1]", primals_18: "f32[192]", primals_24: "i64[]", getitem_7: "f32[1, 192, 1, 1]", rsqrt_3: "f32[1, 192, 1, 1]", primals_25: "f32[192]", getitem_6: "f32[1, 192, 1, 1]", primals_26: "f32[192]", primals_32: "i64[]", getitem_9: "f32[1, 192, 1, 1]", rsqrt_4: "f32[1, 192, 1, 1]", primals_33: "f32[192]", getitem_8: "f32[1, 192, 1, 1]", primals_34: "f32[192]", primals_40: "i64[]", getitem_11: "f32[1, 192, 1, 1]", rsqrt_5: "f32[1, 192, 1, 1]", primals_41: "f32[192]", getitem_10: "f32[1, 192, 1, 1]", primals_42: "f32[192]", primals_48: "i64[]", getitem_13: "f32[1, 192, 1, 1]", rsqrt_6: "f32[1, 192, 1, 1]", primals_49: "f32[192]", getitem_12: "f32[1, 192, 1, 1]", primals_50: "f32[192]", primals_56: "i64[]", getitem_15: "f32[1, 192, 1, 1]", rsqrt_7: "f32[1, 192, 1, 1]", primals_57: "f32[192]", getitem_14: "f32[1, 192, 1, 1]", primals_58: "f32[192]", primals_64: "i64[]", getitem_17: "f32[1, 192, 1, 1]", rsqrt_8: "f32[1, 192, 1, 1]", primals_65: "f32[192]", getitem_16: "f32[1, 192, 1, 1]", primals_66: "f32[192]", primals_74: "i64[]", getitem_19: "f32[1, 384, 1, 1]", primals_75: "f32[384]", getitem_18: "f32[1, 384, 1, 1]", primals_76: "f32[384]", primals_80: "i64[]", getitem_21: "f32[1, 384, 1, 1]", rsqrt_10: "f32[1, 384, 1, 1]", primals_81: "f32[384]", getitem_20: "f32[1, 384, 1, 1]", primals_82: "f32[384]", primals_87: "i64[]", getitem_26: "f32[1, 384, 1, 1]", rsqrt_11: "f32[1, 384, 1, 1]", primals_88: "f32[384]", getitem_25: "f32[1, 384, 1, 1]", primals_89: "f32[384]", primals_94: "i64[]", getitem_28: "f32[1, 384, 1, 1]", rsqrt_12: "f32[1, 384, 1, 1]", primals_95: "f32[384]", getitem_27: "f32[1, 384, 1, 1]", primals_96: "f32[384]", primals_101: "i64[]", getitem_33: "f32[1, 384, 1, 1]", rsqrt_13: "f32[1, 384, 1, 1]", primals_102: "f32[384]", getitem_32: "f32[1, 384, 1, 1]", primals_103: "f32[384]", primals_108: "i64[]", getitem_35: "f32[1, 384, 1, 1]", rsqrt_14: "f32[1, 384, 1, 1]", primals_109: "f32[384]", getitem_34: "f32[1, 384, 1, 1]", primals_110: "f32[384]", primals_115: "i64[]", getitem_40: "f32[1, 384, 1, 1]", rsqrt_15: "f32[1, 384, 1, 1]", primals_116: "f32[384]", getitem_39: "f32[1, 384, 1, 1]", primals_117: "f32[384]", primals_122: "i64[]", getitem_42: "f32[1, 384, 1, 1]", rsqrt_16: "f32[1, 384, 1, 1]", primals_123: "f32[384]", getitem_41: "f32[1, 384, 1, 1]", primals_124: "f32[384]", primals_129: "i64[]", getitem_47: "f32[1, 384, 1, 1]", rsqrt_17: "f32[1, 384, 1, 1]", primals_130: "f32[384]", getitem_46: "f32[1, 384, 1, 1]", primals_131: "f32[384]", primals_138: "i64[]", getitem_49: "f32[1, 768, 1, 1]", primals_139: "f32[768]", getitem_48: "f32[1, 768, 1, 1]", primals_140: "f32[768]", primals_144: "i64[]", getitem_51: "f32[1, 768, 1, 1]", rsqrt_19: "f32[1, 768, 1, 1]", primals_145: "f32[768]", getitem_50: "f32[1, 768, 1, 1]", primals_146: "f32[768]", primals_151: "i64[]", getitem_56: "f32[1, 768, 1, 1]", rsqrt_20: "f32[1, 768, 1, 1]", primals_152: "f32[768]", getitem_55: "f32[1, 768, 1, 1]", primals_153: "f32[768]", primals_158: "i64[]", getitem_58: "f32[1, 768, 1, 1]", rsqrt_21: "f32[1, 768, 1, 1]", primals_159: "f32[768]", getitem_57: "f32[1, 768, 1, 1]", primals_160: "f32[768]", primals_165: "i64[]", getitem_63: "f32[1, 768, 1, 1]", rsqrt_22: "f32[1, 768, 1, 1]", primals_166: "f32[768]", getitem_62: "f32[1, 768, 1, 1]", primals_167: "f32[768]", primals_172: "i64[]", getitem_65: "f32[1, 768, 1, 1]", rsqrt_23: "f32[1, 768, 1, 1]", primals_173: "f32[768]", getitem_64: "f32[1, 768, 1, 1]", primals_174: "f32[768]", primals_179: "i64[]", getitem_70: "f32[1, 768, 1, 1]", rsqrt_24: "f32[1, 768, 1, 1]", primals_180: "f32[768]", getitem_69: "f32[1, 768, 1, 1]", primals_181: "f32[768]", primals_186: "i64[]", getitem_72: "f32[1, 768, 1, 1]", rsqrt_25: "f32[1, 768, 1, 1]", primals_187: "f32[768]", getitem_71: "f32[1, 768, 1, 1]", primals_188: "f32[768]", primals_193: "i64[]", getitem_77: "f32[1, 768, 1, 1]", rsqrt_26: "f32[1, 768, 1, 1]", primals_194: "f32[768]", getitem_76: "f32[1, 768, 1, 1]", primals_195: "f32[768]", primals_200: "i64[]", getitem_78: "f32[1, 768, 1, 1]", add_182: "f32[32, 768, 7, 7]", getitem_79: "f32[1, 768, 1, 1]", primals_201: "f32[768]", primals_202: "f32[768]", primals_203: "f32[768]", primals_204: "f32[768]", _shape_param_0, primals_205: "f32[1000, 768]", add_175: "f32[32, 768, 7, 7]", view_61: "f32[192, 49, 128]", view_57: "f32[192, 49, 128]", view_58: "f32[192, 128, 49]", add_169: "f32[32, 768, 7, 7]", add_162: "f32[32, 768, 7, 7]", view_53: "f32[192, 49, 128]", view_49: "f32[192, 49, 128]", view_50: "f32[192, 128, 49]", add_156: "f32[32, 768, 7, 7]", add_149: "f32[32, 768, 7, 7]", view_45: "f32[192, 49, 128]", view_41: "f32[192, 49, 128]", view_42: "f32[192, 128, 49]", add_143: "f32[32, 768, 7, 7]", add_136: "f32[32, 768, 7, 7]", view_37: "f32[192, 49, 128]", view_33: "f32[192, 49, 128]", view_34: "f32[192, 128, 49]", add_117: "f32[32, 384, 14, 14]", view_29: "f32[192, 196, 64]", view_25: "f32[192, 196, 64]", view_26: "f32[192, 64, 196]", add_111: "f32[32, 384, 14, 14]", add_104: "f32[32, 384, 14, 14]", view_21: "f32[192, 196, 64]", view_17: "f32[192, 196, 64]", view_18: "f32[192, 64, 196]", add_98: "f32[32, 384, 14, 14]", add_91: "f32[32, 384, 14, 14]", view_13: "f32[192, 196, 64]", view_9: "f32[192, 196, 64]", view_10: "f32[192, 64, 196]", add_85: "f32[32, 384, 14, 14]", add_78: "f32[32, 384, 14, 14]", view_5: "f32[192, 196, 64]", view_1: "f32[192, 196, 64]", view_2: "f32[192, 64, 196]", add_58: "f32[32, 192, 28, 28]", add_50: "f32[32, 192, 28, 28]", add_42: "f32[32, 192, 28, 28]", add_34: "f32[32, 192, 28, 28]", add_26: "f32[32, 192, 28, 28]", add_18: "f32[32, 192, 28, 28]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:437 in forward_features, code: x = self.stem(x)
        add_tensor: "i64[]" = torch.ops.aten.add.Tensor(primals_3, 1)
        squeeze_dims: "f32[32]" = torch.ops.aten.squeeze.dims(getitem_1, [0, 2, 3]);  getitem_1 = None
        squeeze_dims_1: "f32[32]" = torch.ops.aten.squeeze.dims(rsqrt, [0, 2, 3]);  rsqrt = None
        mul_tensor: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_dims, 0.1)
        mul_tensor_1: "f32[32]" = torch.ops.aten.mul.Tensor(primals_4, 0.9)
        add_tensor_1: "f32[32]" = torch.ops.aten.add.Tensor(mul_tensor, mul_tensor_1);  mul_tensor = mul_tensor_1 = None
        squeeze_dims_2: "f32[32]" = torch.ops.aten.squeeze.dims(getitem, [0, 2, 3]);  getitem = None
        mul_tensor_2: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_dims_2, 1.0000024912370735);  squeeze_dims_2 = None
        mul_tensor_3: "f32[32]" = torch.ops.aten.mul.Tensor(mul_tensor_2, 0.1);  mul_tensor_2 = None
        mul_tensor_4: "f32[32]" = torch.ops.aten.mul.Tensor(primals_5, 0.9)
        add_tensor_2: "f32[32]" = torch.ops.aten.add.Tensor(mul_tensor_3, mul_tensor_4);  mul_tensor_3 = mul_tensor_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/patch_embed.py:141 in forward, code: x = self.norm(x)
        add_tensor_3: "i64[]" = torch.ops.aten.add.Tensor(primals_10, 1)
        squeeze_dims_3: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_3, [0, 2, 3]);  getitem_3 = None
        mul_tensor_5: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_dims_3, 0.1);  squeeze_dims_3 = None
        mul_tensor_6: "f32[192]" = torch.ops.aten.mul.Tensor(primals_11, 0.9)
        add_tensor_4: "f32[192]" = torch.ops.aten.add.Tensor(mul_tensor_5, mul_tensor_6);  mul_tensor_5 = mul_tensor_6 = None
        squeeze_dims_4: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_2, [0, 2, 3]);  getitem_2 = None
        mul_tensor_7: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_dims_4, 1.0000398612827361);  squeeze_dims_4 = None
        mul_tensor_8: "f32[192]" = torch.ops.aten.mul.Tensor(mul_tensor_7, 0.1);  mul_tensor_7 = None
        mul_tensor_9: "f32[192]" = torch.ops.aten.mul.Tensor(primals_12, 0.9)
        add_tensor_5: "f32[192]" = torch.ops.aten.add.Tensor(mul_tensor_8, mul_tensor_9);  mul_tensor_8 = mul_tensor_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:177 in forward, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
        add_tensor_6: "i64[]" = torch.ops.aten.add.Tensor(primals_16, 1)
        squeeze_dims_5: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_5, [0, 2, 3]);  getitem_5 = None
        squeeze_dims_6: "f32[192]" = torch.ops.aten.squeeze.dims(rsqrt_2, [0, 2, 3]);  rsqrt_2 = None
        mul_tensor_10: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_dims_5, 0.1)
        mul_tensor_11: "f32[192]" = torch.ops.aten.mul.Tensor(primals_17, 0.9)
        add_tensor_7: "f32[192]" = torch.ops.aten.add.Tensor(mul_tensor_10, mul_tensor_11);  mul_tensor_10 = mul_tensor_11 = None
        squeeze_dims_7: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_4, [0, 2, 3]);  getitem_4 = None
        mul_tensor_12: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_dims_7, 1.0000398612827361);  squeeze_dims_7 = None
        mul_tensor_13: "f32[192]" = torch.ops.aten.mul.Tensor(mul_tensor_12, 0.1);  mul_tensor_12 = None
        mul_tensor_14: "f32[192]" = torch.ops.aten.mul.Tensor(primals_18, 0.9)
        add_tensor_8: "f32[192]" = torch.ops.aten.add.Tensor(mul_tensor_13, mul_tensor_14);  mul_tensor_13 = mul_tensor_14 = None
        add_tensor_9: "i64[]" = torch.ops.aten.add.Tensor(primals_24, 1)
        squeeze_dims_8: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_7, [0, 2, 3]);  getitem_7 = None
        squeeze_dims_9: "f32[192]" = torch.ops.aten.squeeze.dims(rsqrt_3, [0, 2, 3]);  rsqrt_3 = None
        mul_tensor_15: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_dims_8, 0.1)
        mul_tensor_16: "f32[192]" = torch.ops.aten.mul.Tensor(primals_25, 0.9)
        add_tensor_10: "f32[192]" = torch.ops.aten.add.Tensor(mul_tensor_15, mul_tensor_16);  mul_tensor_15 = mul_tensor_16 = None
        squeeze_dims_10: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_6, [0, 2, 3]);  getitem_6 = None
        mul_tensor_17: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_dims_10, 1.0000398612827361);  squeeze_dims_10 = None
        mul_tensor_18: "f32[192]" = torch.ops.aten.mul.Tensor(mul_tensor_17, 0.1);  mul_tensor_17 = None
        mul_tensor_19: "f32[192]" = torch.ops.aten.mul.Tensor(primals_26, 0.9)
        add_tensor_11: "f32[192]" = torch.ops.aten.add.Tensor(mul_tensor_18, mul_tensor_19);  mul_tensor_18 = mul_tensor_19 = None
        add_tensor_12: "i64[]" = torch.ops.aten.add.Tensor(primals_32, 1)
        squeeze_dims_11: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_9, [0, 2, 3]);  getitem_9 = None
        squeeze_dims_12: "f32[192]" = torch.ops.aten.squeeze.dims(rsqrt_4, [0, 2, 3]);  rsqrt_4 = None
        mul_tensor_20: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_dims_11, 0.1)
        mul_tensor_21: "f32[192]" = torch.ops.aten.mul.Tensor(primals_33, 0.9)
        add_tensor_13: "f32[192]" = torch.ops.aten.add.Tensor(mul_tensor_20, mul_tensor_21);  mul_tensor_20 = mul_tensor_21 = None
        squeeze_dims_13: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_8, [0, 2, 3]);  getitem_8 = None
        mul_tensor_22: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_dims_13, 1.0000398612827361);  squeeze_dims_13 = None
        mul_tensor_23: "f32[192]" = torch.ops.aten.mul.Tensor(mul_tensor_22, 0.1);  mul_tensor_22 = None
        mul_tensor_24: "f32[192]" = torch.ops.aten.mul.Tensor(primals_34, 0.9)
        add_tensor_14: "f32[192]" = torch.ops.aten.add.Tensor(mul_tensor_23, mul_tensor_24);  mul_tensor_23 = mul_tensor_24 = None
        add_tensor_15: "i64[]" = torch.ops.aten.add.Tensor(primals_40, 1)
        squeeze_dims_14: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_11, [0, 2, 3]);  getitem_11 = None
        squeeze_dims_15: "f32[192]" = torch.ops.aten.squeeze.dims(rsqrt_5, [0, 2, 3]);  rsqrt_5 = None
        mul_tensor_25: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_dims_14, 0.1)
        mul_tensor_26: "f32[192]" = torch.ops.aten.mul.Tensor(primals_41, 0.9)
        add_tensor_16: "f32[192]" = torch.ops.aten.add.Tensor(mul_tensor_25, mul_tensor_26);  mul_tensor_25 = mul_tensor_26 = None
        squeeze_dims_16: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_10, [0, 2, 3]);  getitem_10 = None
        mul_tensor_27: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_dims_16, 1.0000398612827361);  squeeze_dims_16 = None
        mul_tensor_28: "f32[192]" = torch.ops.aten.mul.Tensor(mul_tensor_27, 0.1);  mul_tensor_27 = None
        mul_tensor_29: "f32[192]" = torch.ops.aten.mul.Tensor(primals_42, 0.9)
        add_tensor_17: "f32[192]" = torch.ops.aten.add.Tensor(mul_tensor_28, mul_tensor_29);  mul_tensor_28 = mul_tensor_29 = None
        add_tensor_18: "i64[]" = torch.ops.aten.add.Tensor(primals_48, 1)
        squeeze_dims_17: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_13, [0, 2, 3]);  getitem_13 = None
        squeeze_dims_18: "f32[192]" = torch.ops.aten.squeeze.dims(rsqrt_6, [0, 2, 3]);  rsqrt_6 = None
        mul_tensor_30: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_dims_17, 0.1)
        mul_tensor_31: "f32[192]" = torch.ops.aten.mul.Tensor(primals_49, 0.9)
        add_tensor_19: "f32[192]" = torch.ops.aten.add.Tensor(mul_tensor_30, mul_tensor_31);  mul_tensor_30 = mul_tensor_31 = None
        squeeze_dims_19: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_12, [0, 2, 3]);  getitem_12 = None
        mul_tensor_32: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_dims_19, 1.0000398612827361);  squeeze_dims_19 = None
        mul_tensor_33: "f32[192]" = torch.ops.aten.mul.Tensor(mul_tensor_32, 0.1);  mul_tensor_32 = None
        mul_tensor_34: "f32[192]" = torch.ops.aten.mul.Tensor(primals_50, 0.9)
        add_tensor_20: "f32[192]" = torch.ops.aten.add.Tensor(mul_tensor_33, mul_tensor_34);  mul_tensor_33 = mul_tensor_34 = None
        add_tensor_21: "i64[]" = torch.ops.aten.add.Tensor(primals_56, 1)
        squeeze_dims_20: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_15, [0, 2, 3]);  getitem_15 = None
        squeeze_dims_21: "f32[192]" = torch.ops.aten.squeeze.dims(rsqrt_7, [0, 2, 3]);  rsqrt_7 = None
        mul_tensor_35: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_dims_20, 0.1)
        mul_tensor_36: "f32[192]" = torch.ops.aten.mul.Tensor(primals_57, 0.9)
        add_tensor_22: "f32[192]" = torch.ops.aten.add.Tensor(mul_tensor_35, mul_tensor_36);  mul_tensor_35 = mul_tensor_36 = None
        squeeze_dims_22: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_14, [0, 2, 3]);  getitem_14 = None
        mul_tensor_37: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_dims_22, 1.0000398612827361);  squeeze_dims_22 = None
        mul_tensor_38: "f32[192]" = torch.ops.aten.mul.Tensor(mul_tensor_37, 0.1);  mul_tensor_37 = None
        mul_tensor_39: "f32[192]" = torch.ops.aten.mul.Tensor(primals_58, 0.9)
        add_tensor_23: "f32[192]" = torch.ops.aten.add.Tensor(mul_tensor_38, mul_tensor_39);  mul_tensor_38 = mul_tensor_39 = None
        add_tensor_24: "i64[]" = torch.ops.aten.add.Tensor(primals_64, 1)
        squeeze_dims_23: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_17, [0, 2, 3]);  getitem_17 = None
        squeeze_dims_24: "f32[192]" = torch.ops.aten.squeeze.dims(rsqrt_8, [0, 2, 3]);  rsqrt_8 = None
        mul_tensor_40: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_dims_23, 0.1)
        mul_tensor_41: "f32[192]" = torch.ops.aten.mul.Tensor(primals_65, 0.9)
        add_tensor_25: "f32[192]" = torch.ops.aten.add.Tensor(mul_tensor_40, mul_tensor_41);  mul_tensor_40 = mul_tensor_41 = None
        squeeze_dims_25: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_16, [0, 2, 3]);  getitem_16 = None
        mul_tensor_42: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_dims_25, 1.0000398612827361);  squeeze_dims_25 = None
        mul_tensor_43: "f32[192]" = torch.ops.aten.mul.Tensor(mul_tensor_42, 0.1);  mul_tensor_42 = None
        mul_tensor_44: "f32[192]" = torch.ops.aten.mul.Tensor(primals_66, 0.9)
        add_tensor_26: "f32[192]" = torch.ops.aten.add.Tensor(mul_tensor_43, mul_tensor_44);  mul_tensor_43 = mul_tensor_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/patch_embed.py:141 in forward, code: x = self.norm(x)
        add_tensor_27: "i64[]" = torch.ops.aten.add.Tensor(primals_74, 1)
        squeeze_dims_26: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_19, [0, 2, 3]);  getitem_19 = None
        mul_tensor_45: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_26, 0.1);  squeeze_dims_26 = None
        mul_tensor_46: "f32[384]" = torch.ops.aten.mul.Tensor(primals_75, 0.9)
        add_tensor_28: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_45, mul_tensor_46);  mul_tensor_45 = mul_tensor_46 = None
        squeeze_dims_27: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_18, [0, 2, 3]);  getitem_18 = None
        mul_tensor_47: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_27, 1.0001594642002871);  squeeze_dims_27 = None
        mul_tensor_48: "f32[384]" = torch.ops.aten.mul.Tensor(mul_tensor_47, 0.1);  mul_tensor_47 = None
        mul_tensor_49: "f32[384]" = torch.ops.aten.mul.Tensor(primals_76, 0.9)
        add_tensor_29: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_48, mul_tensor_49);  mul_tensor_48 = mul_tensor_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:176 in forward, code: x = x + self.drop_path(self.attn(self.norm1(x)))
        add_tensor_30: "i64[]" = torch.ops.aten.add.Tensor(primals_80, 1)
        squeeze_dims_28: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_21, [0, 2, 3]);  getitem_21 = None
        squeeze_dims_29: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_10, [0, 2, 3]);  rsqrt_10 = None
        mul_tensor_50: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_28, 0.1)
        mul_tensor_51: "f32[384]" = torch.ops.aten.mul.Tensor(primals_81, 0.9)
        add_tensor_31: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_50, mul_tensor_51);  mul_tensor_50 = mul_tensor_51 = None
        squeeze_dims_30: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_20, [0, 2, 3]);  getitem_20 = None
        mul_tensor_52: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_30, 1.0001594642002871);  squeeze_dims_30 = None
        mul_tensor_53: "f32[384]" = torch.ops.aten.mul.Tensor(mul_tensor_52, 0.1);  mul_tensor_52 = None
        mul_tensor_54: "f32[384]" = torch.ops.aten.mul.Tensor(primals_82, 0.9)
        add_tensor_32: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_53, mul_tensor_54);  mul_tensor_53 = mul_tensor_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:177 in forward, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
        add_tensor_33: "i64[]" = torch.ops.aten.add.Tensor(primals_87, 1)
        squeeze_dims_31: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_26, [0, 2, 3]);  getitem_26 = None
        squeeze_dims_32: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_11, [0, 2, 3]);  rsqrt_11 = None
        mul_tensor_55: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_31, 0.1)
        mul_tensor_56: "f32[384]" = torch.ops.aten.mul.Tensor(primals_88, 0.9)
        add_tensor_34: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_55, mul_tensor_56);  mul_tensor_55 = mul_tensor_56 = None
        squeeze_dims_33: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_25, [0, 2, 3]);  getitem_25 = None
        mul_tensor_57: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_33, 1.0001594642002871);  squeeze_dims_33 = None
        mul_tensor_58: "f32[384]" = torch.ops.aten.mul.Tensor(mul_tensor_57, 0.1);  mul_tensor_57 = None
        mul_tensor_59: "f32[384]" = torch.ops.aten.mul.Tensor(primals_89, 0.9)
        add_tensor_35: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_58, mul_tensor_59);  mul_tensor_58 = mul_tensor_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:176 in forward, code: x = x + self.drop_path(self.attn(self.norm1(x)))
        add_tensor_36: "i64[]" = torch.ops.aten.add.Tensor(primals_94, 1)
        squeeze_dims_34: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_28, [0, 2, 3]);  getitem_28 = None
        squeeze_dims_35: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_12, [0, 2, 3]);  rsqrt_12 = None
        mul_tensor_60: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_34, 0.1)
        mul_tensor_61: "f32[384]" = torch.ops.aten.mul.Tensor(primals_95, 0.9)
        add_tensor_37: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_60, mul_tensor_61);  mul_tensor_60 = mul_tensor_61 = None
        squeeze_dims_36: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_27, [0, 2, 3]);  getitem_27 = None
        mul_tensor_62: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_36, 1.0001594642002871);  squeeze_dims_36 = None
        mul_tensor_63: "f32[384]" = torch.ops.aten.mul.Tensor(mul_tensor_62, 0.1);  mul_tensor_62 = None
        mul_tensor_64: "f32[384]" = torch.ops.aten.mul.Tensor(primals_96, 0.9)
        add_tensor_38: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_63, mul_tensor_64);  mul_tensor_63 = mul_tensor_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:177 in forward, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
        add_tensor_39: "i64[]" = torch.ops.aten.add.Tensor(primals_101, 1)
        squeeze_dims_37: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_33, [0, 2, 3]);  getitem_33 = None
        squeeze_dims_38: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_13, [0, 2, 3]);  rsqrt_13 = None
        mul_tensor_65: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_37, 0.1)
        mul_tensor_66: "f32[384]" = torch.ops.aten.mul.Tensor(primals_102, 0.9)
        add_tensor_40: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_65, mul_tensor_66);  mul_tensor_65 = mul_tensor_66 = None
        squeeze_dims_39: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_32, [0, 2, 3]);  getitem_32 = None
        mul_tensor_67: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_39, 1.0001594642002871);  squeeze_dims_39 = None
        mul_tensor_68: "f32[384]" = torch.ops.aten.mul.Tensor(mul_tensor_67, 0.1);  mul_tensor_67 = None
        mul_tensor_69: "f32[384]" = torch.ops.aten.mul.Tensor(primals_103, 0.9)
        add_tensor_41: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_68, mul_tensor_69);  mul_tensor_68 = mul_tensor_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:176 in forward, code: x = x + self.drop_path(self.attn(self.norm1(x)))
        add_tensor_42: "i64[]" = torch.ops.aten.add.Tensor(primals_108, 1)
        squeeze_dims_40: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_35, [0, 2, 3]);  getitem_35 = None
        squeeze_dims_41: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_14, [0, 2, 3]);  rsqrt_14 = None
        mul_tensor_70: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_40, 0.1)
        mul_tensor_71: "f32[384]" = torch.ops.aten.mul.Tensor(primals_109, 0.9)
        add_tensor_43: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_70, mul_tensor_71);  mul_tensor_70 = mul_tensor_71 = None
        squeeze_dims_42: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_34, [0, 2, 3]);  getitem_34 = None
        mul_tensor_72: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_42, 1.0001594642002871);  squeeze_dims_42 = None
        mul_tensor_73: "f32[384]" = torch.ops.aten.mul.Tensor(mul_tensor_72, 0.1);  mul_tensor_72 = None
        mul_tensor_74: "f32[384]" = torch.ops.aten.mul.Tensor(primals_110, 0.9)
        add_tensor_44: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_73, mul_tensor_74);  mul_tensor_73 = mul_tensor_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:177 in forward, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
        add_tensor_45: "i64[]" = torch.ops.aten.add.Tensor(primals_115, 1)
        squeeze_dims_43: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_40, [0, 2, 3]);  getitem_40 = None
        squeeze_dims_44: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_15, [0, 2, 3]);  rsqrt_15 = None
        mul_tensor_75: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_43, 0.1)
        mul_tensor_76: "f32[384]" = torch.ops.aten.mul.Tensor(primals_116, 0.9)
        add_tensor_46: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_75, mul_tensor_76);  mul_tensor_75 = mul_tensor_76 = None
        squeeze_dims_45: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_39, [0, 2, 3]);  getitem_39 = None
        mul_tensor_77: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_45, 1.0001594642002871);  squeeze_dims_45 = None
        mul_tensor_78: "f32[384]" = torch.ops.aten.mul.Tensor(mul_tensor_77, 0.1);  mul_tensor_77 = None
        mul_tensor_79: "f32[384]" = torch.ops.aten.mul.Tensor(primals_117, 0.9)
        add_tensor_47: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_78, mul_tensor_79);  mul_tensor_78 = mul_tensor_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:176 in forward, code: x = x + self.drop_path(self.attn(self.norm1(x)))
        add_tensor_48: "i64[]" = torch.ops.aten.add.Tensor(primals_122, 1)
        squeeze_dims_46: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_42, [0, 2, 3]);  getitem_42 = None
        squeeze_dims_47: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_16, [0, 2, 3]);  rsqrt_16 = None
        mul_tensor_80: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_46, 0.1)
        mul_tensor_81: "f32[384]" = torch.ops.aten.mul.Tensor(primals_123, 0.9)
        add_tensor_49: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_80, mul_tensor_81);  mul_tensor_80 = mul_tensor_81 = None
        squeeze_dims_48: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_41, [0, 2, 3]);  getitem_41 = None
        mul_tensor_82: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_48, 1.0001594642002871);  squeeze_dims_48 = None
        mul_tensor_83: "f32[384]" = torch.ops.aten.mul.Tensor(mul_tensor_82, 0.1);  mul_tensor_82 = None
        mul_tensor_84: "f32[384]" = torch.ops.aten.mul.Tensor(primals_124, 0.9)
        add_tensor_50: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_83, mul_tensor_84);  mul_tensor_83 = mul_tensor_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:177 in forward, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
        add_tensor_51: "i64[]" = torch.ops.aten.add.Tensor(primals_129, 1)
        squeeze_dims_49: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_47, [0, 2, 3]);  getitem_47 = None
        squeeze_dims_50: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_17, [0, 2, 3]);  rsqrt_17 = None
        mul_tensor_85: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_49, 0.1)
        mul_tensor_86: "f32[384]" = torch.ops.aten.mul.Tensor(primals_130, 0.9)
        add_tensor_52: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_85, mul_tensor_86);  mul_tensor_85 = mul_tensor_86 = None
        squeeze_dims_51: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_46, [0, 2, 3]);  getitem_46 = None
        mul_tensor_87: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_51, 1.0001594642002871);  squeeze_dims_51 = None
        mul_tensor_88: "f32[384]" = torch.ops.aten.mul.Tensor(mul_tensor_87, 0.1);  mul_tensor_87 = None
        mul_tensor_89: "f32[384]" = torch.ops.aten.mul.Tensor(primals_131, 0.9)
        add_tensor_53: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_88, mul_tensor_89);  mul_tensor_88 = mul_tensor_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/patch_embed.py:141 in forward, code: x = self.norm(x)
        add_tensor_54: "i64[]" = torch.ops.aten.add.Tensor(primals_138, 1)
        squeeze_dims_52: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_49, [0, 2, 3]);  getitem_49 = None
        mul_tensor_90: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_dims_52, 0.1);  squeeze_dims_52 = None
        mul_tensor_91: "f32[768]" = torch.ops.aten.mul.Tensor(primals_139, 0.9)
        add_tensor_55: "f32[768]" = torch.ops.aten.add.Tensor(mul_tensor_90, mul_tensor_91);  mul_tensor_90 = mul_tensor_91 = None
        squeeze_dims_53: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_48, [0, 2, 3]);  getitem_48 = None
        mul_tensor_92: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_dims_53, 1.0006381620931717);  squeeze_dims_53 = None
        mul_tensor_93: "f32[768]" = torch.ops.aten.mul.Tensor(mul_tensor_92, 0.1);  mul_tensor_92 = None
        mul_tensor_94: "f32[768]" = torch.ops.aten.mul.Tensor(primals_140, 0.9)
        add_tensor_56: "f32[768]" = torch.ops.aten.add.Tensor(mul_tensor_93, mul_tensor_94);  mul_tensor_93 = mul_tensor_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:176 in forward, code: x = x + self.drop_path(self.attn(self.norm1(x)))
        add_tensor_57: "i64[]" = torch.ops.aten.add.Tensor(primals_144, 1)
        squeeze_dims_54: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_51, [0, 2, 3]);  getitem_51 = None
        squeeze_dims_55: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_19, [0, 2, 3]);  rsqrt_19 = None
        mul_tensor_95: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_dims_54, 0.1)
        mul_tensor_96: "f32[768]" = torch.ops.aten.mul.Tensor(primals_145, 0.9)
        add_tensor_58: "f32[768]" = torch.ops.aten.add.Tensor(mul_tensor_95, mul_tensor_96);  mul_tensor_95 = mul_tensor_96 = None
        squeeze_dims_56: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_50, [0, 2, 3]);  getitem_50 = None
        mul_tensor_97: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_dims_56, 1.0006381620931717);  squeeze_dims_56 = None
        mul_tensor_98: "f32[768]" = torch.ops.aten.mul.Tensor(mul_tensor_97, 0.1);  mul_tensor_97 = None
        mul_tensor_99: "f32[768]" = torch.ops.aten.mul.Tensor(primals_146, 0.9)
        add_tensor_59: "f32[768]" = torch.ops.aten.add.Tensor(mul_tensor_98, mul_tensor_99);  mul_tensor_98 = mul_tensor_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:177 in forward, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
        add_tensor_60: "i64[]" = torch.ops.aten.add.Tensor(primals_151, 1)
        squeeze_dims_57: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_56, [0, 2, 3]);  getitem_56 = None
        squeeze_dims_58: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_20, [0, 2, 3]);  rsqrt_20 = None
        mul_tensor_100: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_dims_57, 0.1)
        mul_tensor_101: "f32[768]" = torch.ops.aten.mul.Tensor(primals_152, 0.9)
        add_tensor_61: "f32[768]" = torch.ops.aten.add.Tensor(mul_tensor_100, mul_tensor_101);  mul_tensor_100 = mul_tensor_101 = None
        squeeze_dims_59: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_55, [0, 2, 3]);  getitem_55 = None
        mul_tensor_102: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_dims_59, 1.0006381620931717);  squeeze_dims_59 = None
        mul_tensor_103: "f32[768]" = torch.ops.aten.mul.Tensor(mul_tensor_102, 0.1);  mul_tensor_102 = None
        mul_tensor_104: "f32[768]" = torch.ops.aten.mul.Tensor(primals_153, 0.9)
        add_tensor_62: "f32[768]" = torch.ops.aten.add.Tensor(mul_tensor_103, mul_tensor_104);  mul_tensor_103 = mul_tensor_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:176 in forward, code: x = x + self.drop_path(self.attn(self.norm1(x)))
        add_tensor_63: "i64[]" = torch.ops.aten.add.Tensor(primals_158, 1)
        squeeze_dims_60: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_58, [0, 2, 3]);  getitem_58 = None
        squeeze_dims_61: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_21, [0, 2, 3]);  rsqrt_21 = None
        mul_tensor_105: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_dims_60, 0.1)
        mul_tensor_106: "f32[768]" = torch.ops.aten.mul.Tensor(primals_159, 0.9)
        add_tensor_64: "f32[768]" = torch.ops.aten.add.Tensor(mul_tensor_105, mul_tensor_106);  mul_tensor_105 = mul_tensor_106 = None
        squeeze_dims_62: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_57, [0, 2, 3]);  getitem_57 = None
        mul_tensor_107: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_dims_62, 1.0006381620931717);  squeeze_dims_62 = None
        mul_tensor_108: "f32[768]" = torch.ops.aten.mul.Tensor(mul_tensor_107, 0.1);  mul_tensor_107 = None
        mul_tensor_109: "f32[768]" = torch.ops.aten.mul.Tensor(primals_160, 0.9)
        add_tensor_65: "f32[768]" = torch.ops.aten.add.Tensor(mul_tensor_108, mul_tensor_109);  mul_tensor_108 = mul_tensor_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:177 in forward, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
        add_tensor_66: "i64[]" = torch.ops.aten.add.Tensor(primals_165, 1)
        squeeze_dims_63: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_63, [0, 2, 3]);  getitem_63 = None
        squeeze_dims_64: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_22, [0, 2, 3]);  rsqrt_22 = None
        mul_tensor_110: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_dims_63, 0.1)
        mul_tensor_111: "f32[768]" = torch.ops.aten.mul.Tensor(primals_166, 0.9)
        add_tensor_67: "f32[768]" = torch.ops.aten.add.Tensor(mul_tensor_110, mul_tensor_111);  mul_tensor_110 = mul_tensor_111 = None
        squeeze_dims_65: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_62, [0, 2, 3]);  getitem_62 = None
        mul_tensor_112: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_dims_65, 1.0006381620931717);  squeeze_dims_65 = None
        mul_tensor_113: "f32[768]" = torch.ops.aten.mul.Tensor(mul_tensor_112, 0.1);  mul_tensor_112 = None
        mul_tensor_114: "f32[768]" = torch.ops.aten.mul.Tensor(primals_167, 0.9)
        add_tensor_68: "f32[768]" = torch.ops.aten.add.Tensor(mul_tensor_113, mul_tensor_114);  mul_tensor_113 = mul_tensor_114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:176 in forward, code: x = x + self.drop_path(self.attn(self.norm1(x)))
        add_tensor_69: "i64[]" = torch.ops.aten.add.Tensor(primals_172, 1)
        squeeze_dims_66: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_65, [0, 2, 3]);  getitem_65 = None
        squeeze_dims_67: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_23, [0, 2, 3]);  rsqrt_23 = None
        mul_tensor_115: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_dims_66, 0.1)
        mul_tensor_116: "f32[768]" = torch.ops.aten.mul.Tensor(primals_173, 0.9)
        add_tensor_70: "f32[768]" = torch.ops.aten.add.Tensor(mul_tensor_115, mul_tensor_116);  mul_tensor_115 = mul_tensor_116 = None
        squeeze_dims_68: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_64, [0, 2, 3]);  getitem_64 = None
        mul_tensor_117: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_dims_68, 1.0006381620931717);  squeeze_dims_68 = None
        mul_tensor_118: "f32[768]" = torch.ops.aten.mul.Tensor(mul_tensor_117, 0.1);  mul_tensor_117 = None
        mul_tensor_119: "f32[768]" = torch.ops.aten.mul.Tensor(primals_174, 0.9)
        add_tensor_71: "f32[768]" = torch.ops.aten.add.Tensor(mul_tensor_118, mul_tensor_119);  mul_tensor_118 = mul_tensor_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:177 in forward, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
        add_tensor_72: "i64[]" = torch.ops.aten.add.Tensor(primals_179, 1)
        squeeze_dims_69: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_70, [0, 2, 3]);  getitem_70 = None
        squeeze_dims_70: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_24, [0, 2, 3]);  rsqrt_24 = None
        mul_tensor_120: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_dims_69, 0.1)
        mul_tensor_121: "f32[768]" = torch.ops.aten.mul.Tensor(primals_180, 0.9)
        add_tensor_73: "f32[768]" = torch.ops.aten.add.Tensor(mul_tensor_120, mul_tensor_121);  mul_tensor_120 = mul_tensor_121 = None
        squeeze_dims_71: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_69, [0, 2, 3]);  getitem_69 = None
        mul_tensor_122: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_dims_71, 1.0006381620931717);  squeeze_dims_71 = None
        mul_tensor_123: "f32[768]" = torch.ops.aten.mul.Tensor(mul_tensor_122, 0.1);  mul_tensor_122 = None
        mul_tensor_124: "f32[768]" = torch.ops.aten.mul.Tensor(primals_181, 0.9)
        add_tensor_74: "f32[768]" = torch.ops.aten.add.Tensor(mul_tensor_123, mul_tensor_124);  mul_tensor_123 = mul_tensor_124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:176 in forward, code: x = x + self.drop_path(self.attn(self.norm1(x)))
        add_tensor_75: "i64[]" = torch.ops.aten.add.Tensor(primals_186, 1)
        squeeze_dims_72: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_72, [0, 2, 3]);  getitem_72 = None
        squeeze_dims_73: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_25, [0, 2, 3]);  rsqrt_25 = None
        mul_tensor_125: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_dims_72, 0.1)
        mul_tensor_126: "f32[768]" = torch.ops.aten.mul.Tensor(primals_187, 0.9)
        add_tensor_76: "f32[768]" = torch.ops.aten.add.Tensor(mul_tensor_125, mul_tensor_126);  mul_tensor_125 = mul_tensor_126 = None
        squeeze_dims_74: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_71, [0, 2, 3]);  getitem_71 = None
        mul_tensor_127: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_dims_74, 1.0006381620931717);  squeeze_dims_74 = None
        mul_tensor_128: "f32[768]" = torch.ops.aten.mul.Tensor(mul_tensor_127, 0.1);  mul_tensor_127 = None
        mul_tensor_129: "f32[768]" = torch.ops.aten.mul.Tensor(primals_188, 0.9)
        add_tensor_77: "f32[768]" = torch.ops.aten.add.Tensor(mul_tensor_128, mul_tensor_129);  mul_tensor_128 = mul_tensor_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:177 in forward, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
        add_tensor_78: "i64[]" = torch.ops.aten.add.Tensor(primals_193, 1)
        squeeze_dims_75: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_77, [0, 2, 3]);  getitem_77 = None
        squeeze_dims_76: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_26, [0, 2, 3]);  rsqrt_26 = None
        mul_tensor_130: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_dims_75, 0.1)
        mul_tensor_131: "f32[768]" = torch.ops.aten.mul.Tensor(primals_194, 0.9)
        add_tensor_79: "f32[768]" = torch.ops.aten.add.Tensor(mul_tensor_130, mul_tensor_131);  mul_tensor_130 = mul_tensor_131 = None
        squeeze_dims_77: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_76, [0, 2, 3]);  getitem_76 = None
        mul_tensor_132: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_dims_77, 1.0006381620931717);  squeeze_dims_77 = None
        mul_tensor_133: "f32[768]" = torch.ops.aten.mul.Tensor(mul_tensor_132, 0.1);  mul_tensor_132 = None
        mul_tensor_134: "f32[768]" = torch.ops.aten.mul.Tensor(primals_195, 0.9)
        add_tensor_80: "f32[768]" = torch.ops.aten.add.Tensor(mul_tensor_133, mul_tensor_134);  mul_tensor_133 = mul_tensor_134 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:468 in forward_features, code: x = self.norm(x)
        add_tensor_81: "i64[]" = torch.ops.aten.add.Tensor(primals_200, 1)
        add_tensor_82: "f32[1, 768, 1, 1]" = torch.ops.aten.add.Tensor(getitem_78, 1e-05)
        rsqrt_default: "f32[1, 768, 1, 1]" = torch.ops.aten.rsqrt.default(add_tensor_82);  add_tensor_82 = None
        sub_tensor: "f32[32, 768, 7, 7]" = torch.ops.aten.sub.Tensor(add_182, getitem_79)
        mul_tensor_135: "f32[32, 768, 7, 7]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = None
        squeeze_dims_78: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_79, [0, 2, 3]);  getitem_79 = None
        squeeze_dims_79: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_default, [0, 2, 3]);  rsqrt_default = None
        mul_tensor_136: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_dims_78, 0.1)
        mul_tensor_137: "f32[768]" = torch.ops.aten.mul.Tensor(primals_201, 0.9)
        add_tensor_83: "f32[768]" = torch.ops.aten.add.Tensor(mul_tensor_136, mul_tensor_137);  mul_tensor_136 = mul_tensor_137 = None
        squeeze_dims_80: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_78, [0, 2, 3]);  getitem_78 = None
        mul_tensor_138: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_dims_80, 1.0006381620931717);  squeeze_dims_80 = None
        mul_tensor_139: "f32[768]" = torch.ops.aten.mul.Tensor(mul_tensor_138, 0.1);  mul_tensor_138 = None
        mul_tensor_140: "f32[768]" = torch.ops.aten.mul.Tensor(primals_202, 0.9)
        add_tensor_84: "f32[768]" = torch.ops.aten.add.Tensor(mul_tensor_139, mul_tensor_140);  mul_tensor_139 = mul_tensor_140 = None
        unsqueeze_default: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_203, -1);  primals_203 = None
        unsqueeze_default_1: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        mul_tensor_141: "f32[32, 768, 7, 7]" = torch.ops.aten.mul.Tensor(mul_tensor_135, unsqueeze_default_1);  mul_tensor_135 = unsqueeze_default_1 = None
        unsqueeze_default_2: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_204, -1);  primals_204 = None
        unsqueeze_default_3: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        add_tensor_85: "f32[32, 768, 7, 7]" = torch.ops.aten.add.Tensor(mul_tensor_141, unsqueeze_default_3);  mul_tensor_141 = unsqueeze_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/adaptive_avgmax_pool.py:172 in forward, code: x = self.pool(x)
        mean_dim: "f32[32, 768, 1, 1]" = torch.ops.aten.mean.dim(add_tensor_85, [-1, -2], True);  add_tensor_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/adaptive_avgmax_pool.py:173 in forward, code: x = self.flatten(x)
        reshape_default: "f32[32, 768]" = torch.ops.aten.reshape.default(mean_dim, _shape_param_0);  mean_dim = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:474 in forward_head, code: return x if pre_logits else self.head(x)
        permute_default: "f32[768, 1000]" = torch.ops.aten.permute.default(primals_205, [1, 0]);  primals_205 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:468 in forward_features, code: x = self.norm(x)
        unsqueeze_default_4: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_dims_78, 0);  squeeze_dims_78 = None
        unsqueeze_default_5: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 2);  unsqueeze_default_4 = None
        unsqueeze_default_6: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_5, 3);  unsqueeze_default_5 = None
        sub_tensor_1: "f32[32, 768, 7, 7]" = torch.ops.aten.sub.Tensor(add_182, unsqueeze_default_6);  add_182 = unsqueeze_default_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:177 in forward, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
        unsqueeze_default_7: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_dims_75, 0);  squeeze_dims_75 = None
        unsqueeze_default_8: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_7, 2);  unsqueeze_default_7 = None
        unsqueeze_default_9: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_8, 3);  unsqueeze_default_8 = None
        sub_tensor_2: "f32[32, 768, 7, 7]" = torch.ops.aten.sub.Tensor(add_175, unsqueeze_default_9);  add_175 = unsqueeze_default_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:119 in forward, code: x = attn @ v
        permute_default_1: "f32[192, 128, 49]" = torch.ops.aten.permute.default(view_61, [0, 2, 1]);  view_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:116 in forward, code: attn = (q @ k.transpose(-2, -1)) * self.scale
        permute_default_2: "f32[192, 128, 49]" = torch.ops.aten.permute.default(view_57, [0, 2, 1]);  view_57 = None
        permute_default_3: "f32[192, 49, 128]" = torch.ops.aten.permute.default(view_58, [0, 2, 1]);  view_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:176 in forward, code: x = x + self.drop_path(self.attn(self.norm1(x)))
        unsqueeze_default_10: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_dims_72, 0);  squeeze_dims_72 = None
        unsqueeze_default_11: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_10, 2);  unsqueeze_default_10 = None
        unsqueeze_default_12: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_11, 3);  unsqueeze_default_11 = None
        sub_tensor_3: "f32[32, 768, 7, 7]" = torch.ops.aten.sub.Tensor(add_169, unsqueeze_default_12);  add_169 = unsqueeze_default_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:177 in forward, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
        unsqueeze_default_13: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_dims_69, 0);  squeeze_dims_69 = None
        unsqueeze_default_14: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_13, 2);  unsqueeze_default_13 = None
        unsqueeze_default_15: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_14, 3);  unsqueeze_default_14 = None
        sub_tensor_4: "f32[32, 768, 7, 7]" = torch.ops.aten.sub.Tensor(add_162, unsqueeze_default_15);  add_162 = unsqueeze_default_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:119 in forward, code: x = attn @ v
        permute_default_4: "f32[192, 128, 49]" = torch.ops.aten.permute.default(view_53, [0, 2, 1]);  view_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:116 in forward, code: attn = (q @ k.transpose(-2, -1)) * self.scale
        permute_default_5: "f32[192, 128, 49]" = torch.ops.aten.permute.default(view_49, [0, 2, 1]);  view_49 = None
        permute_default_6: "f32[192, 49, 128]" = torch.ops.aten.permute.default(view_50, [0, 2, 1]);  view_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:176 in forward, code: x = x + self.drop_path(self.attn(self.norm1(x)))
        unsqueeze_default_16: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_dims_66, 0);  squeeze_dims_66 = None
        unsqueeze_default_17: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_16, 2);  unsqueeze_default_16 = None
        unsqueeze_default_18: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_17, 3);  unsqueeze_default_17 = None
        sub_tensor_5: "f32[32, 768, 7, 7]" = torch.ops.aten.sub.Tensor(add_156, unsqueeze_default_18);  add_156 = unsqueeze_default_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:177 in forward, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
        unsqueeze_default_19: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_dims_63, 0);  squeeze_dims_63 = None
        unsqueeze_default_20: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_19, 2);  unsqueeze_default_19 = None
        unsqueeze_default_21: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_20, 3);  unsqueeze_default_20 = None
        sub_tensor_6: "f32[32, 768, 7, 7]" = torch.ops.aten.sub.Tensor(add_149, unsqueeze_default_21);  add_149 = unsqueeze_default_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:119 in forward, code: x = attn @ v
        permute_default_7: "f32[192, 128, 49]" = torch.ops.aten.permute.default(view_45, [0, 2, 1]);  view_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:116 in forward, code: attn = (q @ k.transpose(-2, -1)) * self.scale
        permute_default_8: "f32[192, 128, 49]" = torch.ops.aten.permute.default(view_41, [0, 2, 1]);  view_41 = None
        permute_default_9: "f32[192, 49, 128]" = torch.ops.aten.permute.default(view_42, [0, 2, 1]);  view_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:176 in forward, code: x = x + self.drop_path(self.attn(self.norm1(x)))
        unsqueeze_default_22: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_dims_60, 0);  squeeze_dims_60 = None
        unsqueeze_default_23: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_22, 2);  unsqueeze_default_22 = None
        unsqueeze_default_24: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_23, 3);  unsqueeze_default_23 = None
        sub_tensor_7: "f32[32, 768, 7, 7]" = torch.ops.aten.sub.Tensor(add_143, unsqueeze_default_24);  add_143 = unsqueeze_default_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:177 in forward, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
        unsqueeze_default_25: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_dims_57, 0);  squeeze_dims_57 = None
        unsqueeze_default_26: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_25, 2);  unsqueeze_default_25 = None
        unsqueeze_default_27: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_26, 3);  unsqueeze_default_26 = None
        sub_tensor_8: "f32[32, 768, 7, 7]" = torch.ops.aten.sub.Tensor(add_136, unsqueeze_default_27);  add_136 = unsqueeze_default_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:119 in forward, code: x = attn @ v
        permute_default_10: "f32[192, 128, 49]" = torch.ops.aten.permute.default(view_37, [0, 2, 1]);  view_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:116 in forward, code: attn = (q @ k.transpose(-2, -1)) * self.scale
        permute_default_11: "f32[192, 128, 49]" = torch.ops.aten.permute.default(view_33, [0, 2, 1]);  view_33 = None
        permute_default_12: "f32[192, 49, 128]" = torch.ops.aten.permute.default(view_34, [0, 2, 1]);  view_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:176 in forward, code: x = x + self.drop_path(self.attn(self.norm1(x)))
        unsqueeze_default_28: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_dims_54, 0);  squeeze_dims_54 = None
        unsqueeze_default_29: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_28, 2);  unsqueeze_default_28 = None
        unsqueeze_default_30: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_29, 3);  unsqueeze_default_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:177 in forward, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
        unsqueeze_default_31: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_dims_49, 0);  squeeze_dims_49 = None
        unsqueeze_default_32: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_31, 2);  unsqueeze_default_31 = None
        unsqueeze_default_33: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_32, 3);  unsqueeze_default_32 = None
        sub_tensor_9: "f32[32, 384, 14, 14]" = torch.ops.aten.sub.Tensor(add_117, unsqueeze_default_33);  add_117 = unsqueeze_default_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:119 in forward, code: x = attn @ v
        permute_default_13: "f32[192, 64, 196]" = torch.ops.aten.permute.default(view_29, [0, 2, 1]);  view_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:116 in forward, code: attn = (q @ k.transpose(-2, -1)) * self.scale
        permute_default_14: "f32[192, 64, 196]" = torch.ops.aten.permute.default(view_25, [0, 2, 1]);  view_25 = None
        permute_default_15: "f32[192, 196, 64]" = torch.ops.aten.permute.default(view_26, [0, 2, 1]);  view_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:176 in forward, code: x = x + self.drop_path(self.attn(self.norm1(x)))
        unsqueeze_default_34: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_dims_46, 0);  squeeze_dims_46 = None
        unsqueeze_default_35: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_34, 2);  unsqueeze_default_34 = None
        unsqueeze_default_36: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_35, 3);  unsqueeze_default_35 = None
        sub_tensor_10: "f32[32, 384, 14, 14]" = torch.ops.aten.sub.Tensor(add_111, unsqueeze_default_36);  add_111 = unsqueeze_default_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:177 in forward, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
        unsqueeze_default_37: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_dims_43, 0);  squeeze_dims_43 = None
        unsqueeze_default_38: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_37, 2);  unsqueeze_default_37 = None
        unsqueeze_default_39: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_38, 3);  unsqueeze_default_38 = None
        sub_tensor_11: "f32[32, 384, 14, 14]" = torch.ops.aten.sub.Tensor(add_104, unsqueeze_default_39);  add_104 = unsqueeze_default_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:119 in forward, code: x = attn @ v
        permute_default_16: "f32[192, 64, 196]" = torch.ops.aten.permute.default(view_21, [0, 2, 1]);  view_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:116 in forward, code: attn = (q @ k.transpose(-2, -1)) * self.scale
        permute_default_17: "f32[192, 64, 196]" = torch.ops.aten.permute.default(view_17, [0, 2, 1]);  view_17 = None
        permute_default_18: "f32[192, 196, 64]" = torch.ops.aten.permute.default(view_18, [0, 2, 1]);  view_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:176 in forward, code: x = x + self.drop_path(self.attn(self.norm1(x)))
        unsqueeze_default_40: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_dims_40, 0);  squeeze_dims_40 = None
        unsqueeze_default_41: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_40, 2);  unsqueeze_default_40 = None
        unsqueeze_default_42: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_41, 3);  unsqueeze_default_41 = None
        sub_tensor_12: "f32[32, 384, 14, 14]" = torch.ops.aten.sub.Tensor(add_98, unsqueeze_default_42);  add_98 = unsqueeze_default_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:177 in forward, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
        unsqueeze_default_43: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_dims_37, 0);  squeeze_dims_37 = None
        unsqueeze_default_44: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_43, 2);  unsqueeze_default_43 = None
        unsqueeze_default_45: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_44, 3);  unsqueeze_default_44 = None
        sub_tensor_13: "f32[32, 384, 14, 14]" = torch.ops.aten.sub.Tensor(add_91, unsqueeze_default_45);  add_91 = unsqueeze_default_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:119 in forward, code: x = attn @ v
        permute_default_19: "f32[192, 64, 196]" = torch.ops.aten.permute.default(view_13, [0, 2, 1]);  view_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:116 in forward, code: attn = (q @ k.transpose(-2, -1)) * self.scale
        permute_default_20: "f32[192, 64, 196]" = torch.ops.aten.permute.default(view_9, [0, 2, 1]);  view_9 = None
        permute_default_21: "f32[192, 196, 64]" = torch.ops.aten.permute.default(view_10, [0, 2, 1]);  view_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:176 in forward, code: x = x + self.drop_path(self.attn(self.norm1(x)))
        unsqueeze_default_46: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_dims_34, 0);  squeeze_dims_34 = None
        unsqueeze_default_47: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_46, 2);  unsqueeze_default_46 = None
        unsqueeze_default_48: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_47, 3);  unsqueeze_default_47 = None
        sub_tensor_14: "f32[32, 384, 14, 14]" = torch.ops.aten.sub.Tensor(add_85, unsqueeze_default_48);  add_85 = unsqueeze_default_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:177 in forward, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
        unsqueeze_default_49: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_dims_31, 0);  squeeze_dims_31 = None
        unsqueeze_default_50: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_49, 2);  unsqueeze_default_49 = None
        unsqueeze_default_51: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_50, 3);  unsqueeze_default_50 = None
        sub_tensor_15: "f32[32, 384, 14, 14]" = torch.ops.aten.sub.Tensor(add_78, unsqueeze_default_51);  add_78 = unsqueeze_default_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:119 in forward, code: x = attn @ v
        permute_default_22: "f32[192, 64, 196]" = torch.ops.aten.permute.default(view_5, [0, 2, 1]);  view_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:116 in forward, code: attn = (q @ k.transpose(-2, -1)) * self.scale
        permute_default_23: "f32[192, 64, 196]" = torch.ops.aten.permute.default(view_1, [0, 2, 1]);  view_1 = None
        permute_default_24: "f32[192, 196, 64]" = torch.ops.aten.permute.default(view_2, [0, 2, 1]);  view_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:176 in forward, code: x = x + self.drop_path(self.attn(self.norm1(x)))
        unsqueeze_default_52: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_dims_28, 0);  squeeze_dims_28 = None
        unsqueeze_default_53: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_52, 2);  unsqueeze_default_52 = None
        unsqueeze_default_54: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_53, 3);  unsqueeze_default_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:177 in forward, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
        unsqueeze_default_55: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_dims_23, 0);  squeeze_dims_23 = None
        unsqueeze_default_56: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_55, 2);  unsqueeze_default_55 = None
        unsqueeze_default_57: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_56, 3);  unsqueeze_default_56 = None
        sub_tensor_16: "f32[32, 192, 28, 28]" = torch.ops.aten.sub.Tensor(add_58, unsqueeze_default_57);  add_58 = unsqueeze_default_57 = None
        unsqueeze_default_58: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_dims_20, 0);  squeeze_dims_20 = None
        unsqueeze_default_59: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_58, 2);  unsqueeze_default_58 = None
        unsqueeze_default_60: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_59, 3);  unsqueeze_default_59 = None
        sub_tensor_17: "f32[32, 192, 28, 28]" = torch.ops.aten.sub.Tensor(add_50, unsqueeze_default_60);  add_50 = unsqueeze_default_60 = None
        unsqueeze_default_61: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_dims_17, 0);  squeeze_dims_17 = None
        unsqueeze_default_62: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_61, 2);  unsqueeze_default_61 = None
        unsqueeze_default_63: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_62, 3);  unsqueeze_default_62 = None
        sub_tensor_18: "f32[32, 192, 28, 28]" = torch.ops.aten.sub.Tensor(add_42, unsqueeze_default_63);  add_42 = unsqueeze_default_63 = None
        unsqueeze_default_64: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_dims_14, 0);  squeeze_dims_14 = None
        unsqueeze_default_65: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_64, 2);  unsqueeze_default_64 = None
        unsqueeze_default_66: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_65, 3);  unsqueeze_default_65 = None
        sub_tensor_19: "f32[32, 192, 28, 28]" = torch.ops.aten.sub.Tensor(add_34, unsqueeze_default_66);  add_34 = unsqueeze_default_66 = None
        unsqueeze_default_67: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_dims_11, 0);  squeeze_dims_11 = None
        unsqueeze_default_68: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_67, 2);  unsqueeze_default_67 = None
        unsqueeze_default_69: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_68, 3);  unsqueeze_default_68 = None
        sub_tensor_20: "f32[32, 192, 28, 28]" = torch.ops.aten.sub.Tensor(add_26, unsqueeze_default_69);  add_26 = unsqueeze_default_69 = None
        unsqueeze_default_70: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_dims_8, 0);  squeeze_dims_8 = None
        unsqueeze_default_71: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_70, 2);  unsqueeze_default_70 = None
        unsqueeze_default_72: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_71, 3);  unsqueeze_default_71 = None
        sub_tensor_21: "f32[32, 192, 28, 28]" = torch.ops.aten.sub.Tensor(add_18, unsqueeze_default_72);  add_18 = unsqueeze_default_72 = None
        unsqueeze_default_73: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_dims_5, 0);  squeeze_dims_5 = None
        unsqueeze_default_74: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_73, 2);  unsqueeze_default_73 = None
        unsqueeze_default_75: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_74, 3);  unsqueeze_default_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:437 in forward_features, code: x = self.stem(x)
        unsqueeze_default_76: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(squeeze_dims, 0);  squeeze_dims = None
        unsqueeze_default_77: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_76, 2);  unsqueeze_default_76 = None
        unsqueeze_default_78: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_77, 3);  unsqueeze_default_77 = None

        # No stacktrace found for following nodes
        copy__default: "i64[]" = torch.ops.aten.copy_.default(primals_3, add_tensor);  primals_3 = add_tensor = None
        copy__default_1: "f32[32]" = torch.ops.aten.copy_.default(primals_4, add_tensor_1);  primals_4 = add_tensor_1 = None
        copy__default_2: "f32[32]" = torch.ops.aten.copy_.default(primals_5, add_tensor_2);  primals_5 = add_tensor_2 = None
        copy__default_3: "i64[]" = torch.ops.aten.copy_.default(primals_10, add_tensor_3);  primals_10 = add_tensor_3 = None
        copy__default_4: "f32[192]" = torch.ops.aten.copy_.default(primals_11, add_tensor_4);  primals_11 = add_tensor_4 = None
        copy__default_5: "f32[192]" = torch.ops.aten.copy_.default(primals_12, add_tensor_5);  primals_12 = add_tensor_5 = None
        copy__default_6: "i64[]" = torch.ops.aten.copy_.default(primals_16, add_tensor_6);  primals_16 = add_tensor_6 = None
        copy__default_7: "f32[192]" = torch.ops.aten.copy_.default(primals_17, add_tensor_7);  primals_17 = add_tensor_7 = None
        copy__default_8: "f32[192]" = torch.ops.aten.copy_.default(primals_18, add_tensor_8);  primals_18 = add_tensor_8 = None
        copy__default_9: "i64[]" = torch.ops.aten.copy_.default(primals_24, add_tensor_9);  primals_24 = add_tensor_9 = None
        copy__default_10: "f32[192]" = torch.ops.aten.copy_.default(primals_25, add_tensor_10);  primals_25 = add_tensor_10 = None
        copy__default_11: "f32[192]" = torch.ops.aten.copy_.default(primals_26, add_tensor_11);  primals_26 = add_tensor_11 = None
        copy__default_12: "i64[]" = torch.ops.aten.copy_.default(primals_32, add_tensor_12);  primals_32 = add_tensor_12 = None
        copy__default_13: "f32[192]" = torch.ops.aten.copy_.default(primals_33, add_tensor_13);  primals_33 = add_tensor_13 = None
        copy__default_14: "f32[192]" = torch.ops.aten.copy_.default(primals_34, add_tensor_14);  primals_34 = add_tensor_14 = None
        copy__default_15: "i64[]" = torch.ops.aten.copy_.default(primals_40, add_tensor_15);  primals_40 = add_tensor_15 = None
        copy__default_16: "f32[192]" = torch.ops.aten.copy_.default(primals_41, add_tensor_16);  primals_41 = add_tensor_16 = None
        copy__default_17: "f32[192]" = torch.ops.aten.copy_.default(primals_42, add_tensor_17);  primals_42 = add_tensor_17 = None
        copy__default_18: "i64[]" = torch.ops.aten.copy_.default(primals_48, add_tensor_18);  primals_48 = add_tensor_18 = None
        copy__default_19: "f32[192]" = torch.ops.aten.copy_.default(primals_49, add_tensor_19);  primals_49 = add_tensor_19 = None
        copy__default_20: "f32[192]" = torch.ops.aten.copy_.default(primals_50, add_tensor_20);  primals_50 = add_tensor_20 = None
        copy__default_21: "i64[]" = torch.ops.aten.copy_.default(primals_56, add_tensor_21);  primals_56 = add_tensor_21 = None
        copy__default_22: "f32[192]" = torch.ops.aten.copy_.default(primals_57, add_tensor_22);  primals_57 = add_tensor_22 = None
        copy__default_23: "f32[192]" = torch.ops.aten.copy_.default(primals_58, add_tensor_23);  primals_58 = add_tensor_23 = None
        copy__default_24: "i64[]" = torch.ops.aten.copy_.default(primals_64, add_tensor_24);  primals_64 = add_tensor_24 = None
        copy__default_25: "f32[192]" = torch.ops.aten.copy_.default(primals_65, add_tensor_25);  primals_65 = add_tensor_25 = None
        copy__default_26: "f32[192]" = torch.ops.aten.copy_.default(primals_66, add_tensor_26);  primals_66 = add_tensor_26 = None
        copy__default_27: "i64[]" = torch.ops.aten.copy_.default(primals_74, add_tensor_27);  primals_74 = add_tensor_27 = None
        copy__default_28: "f32[384]" = torch.ops.aten.copy_.default(primals_75, add_tensor_28);  primals_75 = add_tensor_28 = None
        copy__default_29: "f32[384]" = torch.ops.aten.copy_.default(primals_76, add_tensor_29);  primals_76 = add_tensor_29 = None
        copy__default_30: "i64[]" = torch.ops.aten.copy_.default(primals_80, add_tensor_30);  primals_80 = add_tensor_30 = None
        copy__default_31: "f32[384]" = torch.ops.aten.copy_.default(primals_81, add_tensor_31);  primals_81 = add_tensor_31 = None
        copy__default_32: "f32[384]" = torch.ops.aten.copy_.default(primals_82, add_tensor_32);  primals_82 = add_tensor_32 = None
        copy__default_33: "i64[]" = torch.ops.aten.copy_.default(primals_87, add_tensor_33);  primals_87 = add_tensor_33 = None
        copy__default_34: "f32[384]" = torch.ops.aten.copy_.default(primals_88, add_tensor_34);  primals_88 = add_tensor_34 = None
        copy__default_35: "f32[384]" = torch.ops.aten.copy_.default(primals_89, add_tensor_35);  primals_89 = add_tensor_35 = None
        copy__default_36: "i64[]" = torch.ops.aten.copy_.default(primals_94, add_tensor_36);  primals_94 = add_tensor_36 = None
        copy__default_37: "f32[384]" = torch.ops.aten.copy_.default(primals_95, add_tensor_37);  primals_95 = add_tensor_37 = None
        copy__default_38: "f32[384]" = torch.ops.aten.copy_.default(primals_96, add_tensor_38);  primals_96 = add_tensor_38 = None
        copy__default_39: "i64[]" = torch.ops.aten.copy_.default(primals_101, add_tensor_39);  primals_101 = add_tensor_39 = None
        copy__default_40: "f32[384]" = torch.ops.aten.copy_.default(primals_102, add_tensor_40);  primals_102 = add_tensor_40 = None
        copy__default_41: "f32[384]" = torch.ops.aten.copy_.default(primals_103, add_tensor_41);  primals_103 = add_tensor_41 = None
        copy__default_42: "i64[]" = torch.ops.aten.copy_.default(primals_108, add_tensor_42);  primals_108 = add_tensor_42 = None
        copy__default_43: "f32[384]" = torch.ops.aten.copy_.default(primals_109, add_tensor_43);  primals_109 = add_tensor_43 = None
        copy__default_44: "f32[384]" = torch.ops.aten.copy_.default(primals_110, add_tensor_44);  primals_110 = add_tensor_44 = None
        copy__default_45: "i64[]" = torch.ops.aten.copy_.default(primals_115, add_tensor_45);  primals_115 = add_tensor_45 = None
        copy__default_46: "f32[384]" = torch.ops.aten.copy_.default(primals_116, add_tensor_46);  primals_116 = add_tensor_46 = None
        copy__default_47: "f32[384]" = torch.ops.aten.copy_.default(primals_117, add_tensor_47);  primals_117 = add_tensor_47 = None
        copy__default_48: "i64[]" = torch.ops.aten.copy_.default(primals_122, add_tensor_48);  primals_122 = add_tensor_48 = None
        copy__default_49: "f32[384]" = torch.ops.aten.copy_.default(primals_123, add_tensor_49);  primals_123 = add_tensor_49 = None
        copy__default_50: "f32[384]" = torch.ops.aten.copy_.default(primals_124, add_tensor_50);  primals_124 = add_tensor_50 = None
        copy__default_51: "i64[]" = torch.ops.aten.copy_.default(primals_129, add_tensor_51);  primals_129 = add_tensor_51 = None
        copy__default_52: "f32[384]" = torch.ops.aten.copy_.default(primals_130, add_tensor_52);  primals_130 = add_tensor_52 = None
        copy__default_53: "f32[384]" = torch.ops.aten.copy_.default(primals_131, add_tensor_53);  primals_131 = add_tensor_53 = None
        copy__default_54: "i64[]" = torch.ops.aten.copy_.default(primals_138, add_tensor_54);  primals_138 = add_tensor_54 = None
        copy__default_55: "f32[768]" = torch.ops.aten.copy_.default(primals_139, add_tensor_55);  primals_139 = add_tensor_55 = None
        copy__default_56: "f32[768]" = torch.ops.aten.copy_.default(primals_140, add_tensor_56);  primals_140 = add_tensor_56 = None
        copy__default_57: "i64[]" = torch.ops.aten.copy_.default(primals_144, add_tensor_57);  primals_144 = add_tensor_57 = None
        copy__default_58: "f32[768]" = torch.ops.aten.copy_.default(primals_145, add_tensor_58);  primals_145 = add_tensor_58 = None
        copy__default_59: "f32[768]" = torch.ops.aten.copy_.default(primals_146, add_tensor_59);  primals_146 = add_tensor_59 = None
        copy__default_60: "i64[]" = torch.ops.aten.copy_.default(primals_151, add_tensor_60);  primals_151 = add_tensor_60 = None
        copy__default_61: "f32[768]" = torch.ops.aten.copy_.default(primals_152, add_tensor_61);  primals_152 = add_tensor_61 = None
        copy__default_62: "f32[768]" = torch.ops.aten.copy_.default(primals_153, add_tensor_62);  primals_153 = add_tensor_62 = None
        copy__default_63: "i64[]" = torch.ops.aten.copy_.default(primals_158, add_tensor_63);  primals_158 = add_tensor_63 = None
        copy__default_64: "f32[768]" = torch.ops.aten.copy_.default(primals_159, add_tensor_64);  primals_159 = add_tensor_64 = None
        copy__default_65: "f32[768]" = torch.ops.aten.copy_.default(primals_160, add_tensor_65);  primals_160 = add_tensor_65 = None
        copy__default_66: "i64[]" = torch.ops.aten.copy_.default(primals_165, add_tensor_66);  primals_165 = add_tensor_66 = None
        copy__default_67: "f32[768]" = torch.ops.aten.copy_.default(primals_166, add_tensor_67);  primals_166 = add_tensor_67 = None
        copy__default_68: "f32[768]" = torch.ops.aten.copy_.default(primals_167, add_tensor_68);  primals_167 = add_tensor_68 = None
        copy__default_69: "i64[]" = torch.ops.aten.copy_.default(primals_172, add_tensor_69);  primals_172 = add_tensor_69 = None
        copy__default_70: "f32[768]" = torch.ops.aten.copy_.default(primals_173, add_tensor_70);  primals_173 = add_tensor_70 = None
        copy__default_71: "f32[768]" = torch.ops.aten.copy_.default(primals_174, add_tensor_71);  primals_174 = add_tensor_71 = None
        copy__default_72: "i64[]" = torch.ops.aten.copy_.default(primals_179, add_tensor_72);  primals_179 = add_tensor_72 = None
        copy__default_73: "f32[768]" = torch.ops.aten.copy_.default(primals_180, add_tensor_73);  primals_180 = add_tensor_73 = None
        copy__default_74: "f32[768]" = torch.ops.aten.copy_.default(primals_181, add_tensor_74);  primals_181 = add_tensor_74 = None
        copy__default_75: "i64[]" = torch.ops.aten.copy_.default(primals_186, add_tensor_75);  primals_186 = add_tensor_75 = None
        copy__default_76: "f32[768]" = torch.ops.aten.copy_.default(primals_187, add_tensor_76);  primals_187 = add_tensor_76 = None
        copy__default_77: "f32[768]" = torch.ops.aten.copy_.default(primals_188, add_tensor_77);  primals_188 = add_tensor_77 = None
        copy__default_78: "i64[]" = torch.ops.aten.copy_.default(primals_193, add_tensor_78);  primals_193 = add_tensor_78 = None
        copy__default_79: "f32[768]" = torch.ops.aten.copy_.default(primals_194, add_tensor_79);  primals_194 = add_tensor_79 = None
        copy__default_80: "f32[768]" = torch.ops.aten.copy_.default(primals_195, add_tensor_80);  primals_195 = add_tensor_80 = None
        copy__default_81: "i64[]" = torch.ops.aten.copy_.default(primals_200, add_tensor_81);  primals_200 = add_tensor_81 = None
        copy__default_82: "f32[768]" = torch.ops.aten.copy_.default(primals_201, add_tensor_83);  primals_201 = add_tensor_83 = None
        copy__default_83: "f32[768]" = torch.ops.aten.copy_.default(primals_202, add_tensor_84);  primals_202 = add_tensor_84 = None
        return (squeeze_dims_1, squeeze_dims_6, squeeze_dims_9, squeeze_dims_12, squeeze_dims_15, squeeze_dims_18, squeeze_dims_21, squeeze_dims_24, squeeze_dims_29, squeeze_dims_32, squeeze_dims_35, squeeze_dims_38, squeeze_dims_41, squeeze_dims_44, squeeze_dims_47, squeeze_dims_50, squeeze_dims_55, squeeze_dims_58, squeeze_dims_61, squeeze_dims_64, squeeze_dims_67, squeeze_dims_70, squeeze_dims_73, squeeze_dims_76, squeeze_dims_79, reshape_default, permute_default, sub_tensor_1, sub_tensor_2, permute_default_1, permute_default_2, permute_default_3, sub_tensor_3, sub_tensor_4, permute_default_4, permute_default_5, permute_default_6, sub_tensor_5, sub_tensor_6, permute_default_7, permute_default_8, permute_default_9, sub_tensor_7, sub_tensor_8, permute_default_10, permute_default_11, permute_default_12, unsqueeze_default_30, sub_tensor_9, permute_default_13, permute_default_14, permute_default_15, sub_tensor_10, sub_tensor_11, permute_default_16, permute_default_17, permute_default_18, sub_tensor_12, sub_tensor_13, permute_default_19, permute_default_20, permute_default_21, sub_tensor_14, sub_tensor_15, permute_default_22, permute_default_23, permute_default_24, unsqueeze_default_54, sub_tensor_16, sub_tensor_17, sub_tensor_18, sub_tensor_19, sub_tensor_20, sub_tensor_21, unsqueeze_default_75, unsqueeze_default_78, copy__default, copy__default_1, copy__default_2, copy__default_3, copy__default_4, copy__default_5, copy__default_6, copy__default_7, copy__default_8, copy__default_9, copy__default_10, copy__default_11, copy__default_12, copy__default_13, copy__default_14, copy__default_15, copy__default_16, copy__default_17, copy__default_18, copy__default_19, copy__default_20, copy__default_21, copy__default_22, copy__default_23, copy__default_24, copy__default_25, copy__default_26, copy__default_27, copy__default_28, copy__default_29, copy__default_30, copy__default_31, copy__default_32, copy__default_33, copy__default_34, copy__default_35, copy__default_36, copy__default_37, copy__default_38, copy__default_39, copy__default_40, copy__default_41, copy__default_42, copy__default_43, copy__default_44, copy__default_45, copy__default_46, copy__default_47, copy__default_48, copy__default_49, copy__default_50, copy__default_51, copy__default_52, copy__default_53, copy__default_54, copy__default_55, copy__default_56, copy__default_57, copy__default_58, copy__default_59, copy__default_60, copy__default_61, copy__default_62, copy__default_63, copy__default_64, copy__default_65, copy__default_66, copy__default_67, copy__default_68, copy__default_69, copy__default_70, copy__default_71, copy__default_72, copy__default_73, copy__default_74, copy__default_75, copy__default_76, copy__default_77, copy__default_78, copy__default_79, copy__default_80, copy__default_81, copy__default_82, copy__default_83)


def _default_make_inputs():
    return [
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 32, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 32, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([32], dtype=torch.float32, device='cuda'),
    torch.randn([1, 32, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([32], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 192, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([1, 192, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 192, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 192, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([1, 192, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 192, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 192, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([1, 192, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 192, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 192, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([1, 192, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 192, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 192, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([1, 192, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 192, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 192, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([1, 192, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 192, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 192, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([1, 192, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 192, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 192, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([1, 192, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 384, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([1, 384, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 384, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 384, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([1, 384, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 384, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 384, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([1, 384, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 384, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 384, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([1, 384, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 384, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 384, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([1, 384, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 384, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 384, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([1, 384, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 384, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 384, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([1, 384, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 384, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 384, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([1, 384, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 384, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 384, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([1, 384, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 768, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([1, 768, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 768, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 768, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([1, 768, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 768, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 768, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([1, 768, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 768, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 768, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([1, 768, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 768, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 768, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([1, 768, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 768, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 768, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([1, 768, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 768, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 768, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([1, 768, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 768, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 768, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([1, 768, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 768, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 768, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([1, 768, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 768, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([32, 768, 7, 7], dtype=torch.float32, device='cuda'),
    torch.randn([1, 768, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    [32, 768],  # _shape_param_0
    torch.randn([1000, 768], dtype=torch.float32, device='cuda'),
    torch.randn([32, 768, 7, 7], dtype=torch.float32, device='cuda'),
    torch.randn([192, 49, 128], dtype=torch.float32, device='cuda'),
    torch.randn([192, 49, 128], dtype=torch.float32, device='cuda'),
    torch.randn([192, 128, 49], dtype=torch.float32, device='cuda'),
    torch.randn([32, 768, 7, 7], dtype=torch.float32, device='cuda'),
    torch.randn([32, 768, 7, 7], dtype=torch.float32, device='cuda'),
    torch.randn([192, 49, 128], dtype=torch.float32, device='cuda'),
    torch.randn([192, 49, 128], dtype=torch.float32, device='cuda'),
    torch.randn([192, 128, 49], dtype=torch.float32, device='cuda'),
    torch.randn([32, 768, 7, 7], dtype=torch.float32, device='cuda'),
    torch.randn([32, 768, 7, 7], dtype=torch.float32, device='cuda'),
    torch.randn([192, 49, 128], dtype=torch.float32, device='cuda'),
    torch.randn([192, 49, 128], dtype=torch.float32, device='cuda'),
    torch.randn([192, 128, 49], dtype=torch.float32, device='cuda'),
    torch.randn([32, 768, 7, 7], dtype=torch.float32, device='cuda'),
    torch.randn([32, 768, 7, 7], dtype=torch.float32, device='cuda'),
    torch.randn([192, 49, 128], dtype=torch.float32, device='cuda'),
    torch.randn([192, 49, 128], dtype=torch.float32, device='cuda'),
    torch.randn([192, 128, 49], dtype=torch.float32, device='cuda'),
    torch.randn([32, 384, 14, 14], dtype=torch.float32, device='cuda'),
    torch.randn([192, 196, 64], dtype=torch.float32, device='cuda'),
    torch.randn([192, 196, 64], dtype=torch.float32, device='cuda'),
    torch.randn([192, 64, 196], dtype=torch.float32, device='cuda'),
    torch.randn([32, 384, 14, 14], dtype=torch.float32, device='cuda'),
    torch.randn([32, 384, 14, 14], dtype=torch.float32, device='cuda'),
    torch.randn([192, 196, 64], dtype=torch.float32, device='cuda'),
    torch.randn([192, 196, 64], dtype=torch.float32, device='cuda'),
    torch.randn([192, 64, 196], dtype=torch.float32, device='cuda'),
    torch.randn([32, 384, 14, 14], dtype=torch.float32, device='cuda'),
    torch.randn([32, 384, 14, 14], dtype=torch.float32, device='cuda'),
    torch.randn([192, 196, 64], dtype=torch.float32, device='cuda'),
    torch.randn([192, 196, 64], dtype=torch.float32, device='cuda'),
    torch.randn([192, 64, 196], dtype=torch.float32, device='cuda'),
    torch.randn([32, 384, 14, 14], dtype=torch.float32, device='cuda'),
    torch.randn([32, 384, 14, 14], dtype=torch.float32, device='cuda'),
    torch.randn([192, 196, 64], dtype=torch.float32, device='cuda'),
    torch.randn([192, 196, 64], dtype=torch.float32, device='cuda'),
    torch.randn([192, 64, 196], dtype=torch.float32, device='cuda'),
    torch.randn([32, 192, 28, 28], dtype=torch.float32, device='cuda'),
    torch.randn([32, 192, 28, 28], dtype=torch.float32, device='cuda'),
    torch.randn([32, 192, 28, 28], dtype=torch.float32, device='cuda'),
    torch.randn([32, 192, 28, 28], dtype=torch.float32, device='cuda'),
    torch.randn([32, 192, 28, 28], dtype=torch.float32, device='cuda'),
    torch.randn([32, 192, 28, 28], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
