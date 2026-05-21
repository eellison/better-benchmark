"""
Standalone repro captured via capture_hook.
Label: timm_mobilevit_s_train
Pattern hash: eb919cd7e75e
Shape hash: f78f0295
"""
_shapes_config = "(T([], i64), T([1, 16, 1, 1], f32), T([16], f32), T([1, 16, 1, 1], f32), T([16], f32), T([], i64), T([1, 64, 1, 1], f32), T([64], f32), T([1, 64, 1, 1], f32), T([64], f32), T([], i64), T([1, 64, 1, 1], f32), T([64], f32), T([1, 64, 1, 1], f32), T([64], f32), T([], i64), T([1, 32, 1, 1], f32), T([1, 32, 1, 1], f32), T([32], f32), T([1, 32, 1, 1], f32), T([32], f32), T([], i64), T([1, 128, 1, 1], f32), T([128], f32), T([1, 128, 1, 1], f32), T([128], f32), T([], i64), T([1, 128, 1, 1], f32), T([128], f32), T([1, 128, 1, 1], f32), T([128], f32), T([], i64), T([1, 64, 1, 1], f32), T([1, 64, 1, 1], f32), T([64], f32), T([1, 64, 1, 1], f32), T([64], f32), T([], i64), T([1, 256, 1, 1], f32), T([256], f32), T([1, 256, 1, 1], f32), T([256], f32), T([], i64), T([1, 256, 1, 1], f32), T([256], f32), T([1, 256, 1, 1], f32), T([256], f32), T([], i64), T([1, 64, 1, 1], f32), T([1, 64, 1, 1], f32), T([64], f32), T([1, 64, 1, 1], f32), T([64], f32), T([], i64), T([1, 256, 1, 1], f32), T([256], f32), T([1, 256, 1, 1], f32), T([256], f32), T([], i64), T([1, 256, 1, 1], f32), T([256], f32), T([1, 256, 1, 1], f32), T([256], f32), T([], i64), T([1, 64, 1, 1], f32), T([1, 64, 1, 1], f32), T([64], f32), T([1, 64, 1, 1], f32), T([64], f32), T([], i64), T([1, 256, 1, 1], f32), T([256], f32), T([1, 256, 1, 1], f32), T([256], f32), T([], i64), T([1, 256, 1, 1], f32), T([256], f32), T([1, 256, 1, 1], f32), T([256], f32), T([], i64), T([1, 96, 1, 1], f32), T([1, 96, 1, 1], f32), T([96], f32), T([1, 96, 1, 1], f32), T([96], f32), T([], i64), T([1, 96, 1, 1], f32), T([96], f32), T([1, 96, 1, 1], f32), T([96], f32), T([], i64), T([1, 96, 1, 1], f32), T([96], f32), T([1, 96, 1, 1], f32), T([96], f32), T([], i64), T([1, 96, 1, 1], f32), T([96], f32), T([1, 96, 1, 1], f32), T([96], f32), T([], i64), T([1, 384, 1, 1], f32), T([384], f32), T([1, 384, 1, 1], f32), T([384], f32), T([], i64), T([1, 384, 1, 1], f32), T([384], f32), T([1, 384, 1, 1], f32), T([384], f32), T([], i64), T([1, 128, 1, 1], f32), T([1, 128, 1, 1], f32), T([128], f32), T([1, 128, 1, 1], f32), T([128], f32), T([], i64), T([1, 128, 1, 1], f32), T([128], f32), T([1, 128, 1, 1], f32), T([128], f32), T([], i64), T([1, 128, 1, 1], f32), T([128], f32), T([1, 128, 1, 1], f32), T([128], f32), T([], i64), T([1, 128, 1, 1], f32), T([128], f32), T([1, 128, 1, 1], f32), T([128], f32), T([], i64), T([1, 512, 1, 1], f32), T([512], f32), T([1, 512, 1, 1], f32), T([512], f32), T([], i64), T([1, 512, 1, 1], f32), T([512], f32), T([1, 512, 1, 1], f32), T([512], f32), T([], i64), T([1, 160, 1, 1], f32), T([1, 160, 1, 1], f32), T([160], f32), T([1, 160, 1, 1], f32), T([160], f32), T([], i64), T([1, 160, 1, 1], f32), T([160], f32), T([1, 160, 1, 1], f32), T([160], f32), T([], i64), T([1, 160, 1, 1], f32), T([160], f32), T([1, 160, 1, 1], f32), T([160], f32), T([], i64), T([1, 160, 1, 1], f32), T([160], f32), T([1, 160, 1, 1], f32), T([160], f32), T([], i64), T([128, 640, 8, 8], f32, stride=(40960, 1, 5120, 640)), T([640], f32), T([640], f32), T([640], f32), T([640], f32), T([1000, 640], f32), T([512, 16, 1], f32), T([512, 16, 1], f32), T([512, 16, 1], f32), T([512, 16, 1], f32), T([512, 16, 1], f32), T([512, 16, 1], f32), T([512, 16, 1], f32), T([512, 64, 1], f32), T([512, 64, 1], f32), T([512, 64, 1], f32), T([512, 64, 1], f32), T([512, 64, 1], f32), T([512, 64, 1], f32), T([512, 64, 1], f32), T([512, 64, 1], f32), T([512, 64, 1], f32), T([512, 256, 1], f32), T([512, 256, 1], f32), T([512, 256, 1], f32), T([512, 256, 1], f32), T([512, 256, 1], f32), S([128, 640]))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, primals_3: "i64[]", getitem_1: "f32[1, 16, 1, 1]", primals_4: "f32[16]", getitem: "f32[1, 16, 1, 1]", primals_5: "f32[16]", primals_9: "i64[]", getitem_3: "f32[1, 64, 1, 1]", primals_10: "f32[64]", getitem_2: "f32[1, 64, 1, 1]", primals_11: "f32[64]", primals_15: "i64[]", getitem_5: "f32[1, 64, 1, 1]", primals_16: "f32[64]", getitem_4: "f32[1, 64, 1, 1]", primals_17: "f32[64]", primals_21: "i64[]", getitem_7: "f32[1, 32, 1, 1]", rsqrt_3: "f32[1, 32, 1, 1]", primals_22: "f32[32]", getitem_6: "f32[1, 32, 1, 1]", primals_23: "f32[32]", primals_27: "i64[]", getitem_9: "f32[1, 128, 1, 1]", primals_28: "f32[128]", getitem_8: "f32[1, 128, 1, 1]", primals_29: "f32[128]", primals_33: "i64[]", getitem_11: "f32[1, 128, 1, 1]", primals_34: "f32[128]", getitem_10: "f32[1, 128, 1, 1]", primals_35: "f32[128]", primals_39: "i64[]", getitem_13: "f32[1, 64, 1, 1]", rsqrt_6: "f32[1, 64, 1, 1]", primals_40: "f32[64]", getitem_12: "f32[1, 64, 1, 1]", primals_41: "f32[64]", primals_45: "i64[]", getitem_15: "f32[1, 256, 1, 1]", primals_46: "f32[256]", getitem_14: "f32[1, 256, 1, 1]", primals_47: "f32[256]", primals_51: "i64[]", getitem_17: "f32[1, 256, 1, 1]", primals_52: "f32[256]", getitem_16: "f32[1, 256, 1, 1]", primals_53: "f32[256]", primals_57: "i64[]", getitem_19: "f32[1, 64, 1, 1]", rsqrt_9: "f32[1, 64, 1, 1]", primals_58: "f32[64]", getitem_18: "f32[1, 64, 1, 1]", primals_59: "f32[64]", primals_63: "i64[]", getitem_21: "f32[1, 256, 1, 1]", primals_64: "f32[256]", getitem_20: "f32[1, 256, 1, 1]", primals_65: "f32[256]", primals_69: "i64[]", getitem_23: "f32[1, 256, 1, 1]", primals_70: "f32[256]", getitem_22: "f32[1, 256, 1, 1]", primals_71: "f32[256]", primals_75: "i64[]", getitem_25: "f32[1, 64, 1, 1]", rsqrt_12: "f32[1, 64, 1, 1]", primals_76: "f32[64]", getitem_24: "f32[1, 64, 1, 1]", primals_77: "f32[64]", primals_81: "i64[]", getitem_27: "f32[1, 256, 1, 1]", primals_82: "f32[256]", getitem_26: "f32[1, 256, 1, 1]", primals_83: "f32[256]", primals_87: "i64[]", getitem_29: "f32[1, 256, 1, 1]", primals_88: "f32[256]", getitem_28: "f32[1, 256, 1, 1]", primals_89: "f32[256]", primals_93: "i64[]", getitem_31: "f32[1, 96, 1, 1]", rsqrt_15: "f32[1, 96, 1, 1]", primals_94: "f32[96]", getitem_30: "f32[1, 96, 1, 1]", primals_95: "f32[96]", primals_99: "i64[]", getitem_33: "f32[1, 96, 1, 1]", primals_100: "f32[96]", getitem_32: "f32[1, 96, 1, 1]", primals_101: "f32[96]", primals_132: "i64[]", getitem_59: "f32[1, 96, 1, 1]", primals_133: "f32[96]", getitem_58: "f32[1, 96, 1, 1]", primals_134: "f32[96]", primals_138: "i64[]", getitem_61: "f32[1, 96, 1, 1]", primals_139: "f32[96]", getitem_60: "f32[1, 96, 1, 1]", primals_140: "f32[96]", primals_144: "i64[]", getitem_63: "f32[1, 384, 1, 1]", primals_145: "f32[384]", getitem_62: "f32[1, 384, 1, 1]", primals_146: "f32[384]", primals_150: "i64[]", getitem_65: "f32[1, 384, 1, 1]", primals_151: "f32[384]", getitem_64: "f32[1, 384, 1, 1]", primals_152: "f32[384]", primals_156: "i64[]", getitem_67: "f32[1, 128, 1, 1]", rsqrt_26: "f32[1, 128, 1, 1]", primals_157: "f32[128]", getitem_66: "f32[1, 128, 1, 1]", primals_158: "f32[128]", primals_162: "i64[]", getitem_69: "f32[1, 128, 1, 1]", primals_163: "f32[128]", getitem_68: "f32[1, 128, 1, 1]", primals_164: "f32[128]", primals_219: "i64[]", getitem_117: "f32[1, 128, 1, 1]", primals_220: "f32[128]", getitem_116: "f32[1, 128, 1, 1]", primals_221: "f32[128]", primals_225: "i64[]", getitem_119: "f32[1, 128, 1, 1]", primals_226: "f32[128]", getitem_118: "f32[1, 128, 1, 1]", primals_227: "f32[128]", primals_231: "i64[]", getitem_121: "f32[1, 512, 1, 1]", primals_232: "f32[512]", getitem_120: "f32[1, 512, 1, 1]", primals_233: "f32[512]", primals_237: "i64[]", getitem_123: "f32[1, 512, 1, 1]", primals_238: "f32[512]", getitem_122: "f32[1, 512, 1, 1]", primals_239: "f32[512]", primals_243: "i64[]", getitem_125: "f32[1, 160, 1, 1]", rsqrt_41: "f32[1, 160, 1, 1]", primals_244: "f32[160]", getitem_124: "f32[1, 160, 1, 1]", primals_245: "f32[160]", primals_249: "i64[]", getitem_127: "f32[1, 160, 1, 1]", primals_250: "f32[160]", getitem_126: "f32[1, 160, 1, 1]", primals_251: "f32[160]", primals_294: "i64[]", getitem_164: "f32[1, 160, 1, 1]", primals_295: "f32[160]", getitem_163: "f32[1, 160, 1, 1]", primals_296: "f32[160]", primals_300: "i64[]", getitem_166: "f32[1, 160, 1, 1]", primals_301: "f32[160]", getitem_165: "f32[1, 160, 1, 1]", primals_302: "f32[160]", primals_306: "i64[]", convolution_34: "f32[128, 640, 8, 8]", primals_307: "f32[640]", primals_308: "f32[640]", primals_309: "f32[640]", primals_310: "f32[640]", primals_311: "f32[1000, 640]", rsqrt_49: "f32[512, 16, 1]", rsqrt_48: "f32[512, 16, 1]", rsqrt_47: "f32[512, 16, 1]", rsqrt_46: "f32[512, 16, 1]", rsqrt_45: "f32[512, 16, 1]", rsqrt_44: "f32[512, 16, 1]", rsqrt_43: "f32[512, 16, 1]", rsqrt_36: "f32[512, 64, 1]", rsqrt_35: "f32[512, 64, 1]", rsqrt_34: "f32[512, 64, 1]", rsqrt_33: "f32[512, 64, 1]", rsqrt_32: "f32[512, 64, 1]", rsqrt_31: "f32[512, 64, 1]", rsqrt_30: "f32[512, 64, 1]", rsqrt_29: "f32[512, 64, 1]", rsqrt_28: "f32[512, 64, 1]", rsqrt_21: "f32[512, 256, 1]", rsqrt_20: "f32[512, 256, 1]", rsqrt_19: "f32[512, 256, 1]", rsqrt_18: "f32[512, 256, 1]", rsqrt_17: "f32[512, 256, 1]", _shape_param_0):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_tensor: "i64[]" = torch.ops.aten.add.Tensor(primals_3, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims: "f32[16]" = torch.ops.aten.squeeze.dims(getitem_1, [0, 2, 3]);  getitem_1 = None
        mul_tensor: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_dims, 0.1);  squeeze_dims = None
        mul_tensor_1: "f32[16]" = torch.ops.aten.mul.Tensor(primals_4, 0.9)
        add_tensor_1: "f32[16]" = torch.ops.aten.add.Tensor(mul_tensor, mul_tensor_1);  mul_tensor = mul_tensor_1 = None
        squeeze_dims_1: "f32[16]" = torch.ops.aten.squeeze.dims(getitem, [0, 2, 3]);  getitem = None
        mul_tensor_2: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_dims_1, 1.0000004768373856);  squeeze_dims_1 = None
        mul_tensor_3: "f32[16]" = torch.ops.aten.mul.Tensor(mul_tensor_2, 0.1);  mul_tensor_2 = None
        mul_tensor_4: "f32[16]" = torch.ops.aten.mul.Tensor(primals_5, 0.9)
        add_tensor_2: "f32[16]" = torch.ops.aten.add.Tensor(mul_tensor_3, mul_tensor_4);  mul_tensor_3 = mul_tensor_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_tensor_3: "i64[]" = torch.ops.aten.add.Tensor(primals_9, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims_2: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_3, [0, 2, 3]);  getitem_3 = None
        mul_tensor_5: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_dims_2, 0.1);  squeeze_dims_2 = None
        mul_tensor_6: "f32[64]" = torch.ops.aten.mul.Tensor(primals_10, 0.9)
        add_tensor_4: "f32[64]" = torch.ops.aten.add.Tensor(mul_tensor_5, mul_tensor_6);  mul_tensor_5 = mul_tensor_6 = None
        squeeze_dims_3: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_2, [0, 2, 3]);  getitem_2 = None
        mul_tensor_7: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_dims_3, 1.0000004768373856);  squeeze_dims_3 = None
        mul_tensor_8: "f32[64]" = torch.ops.aten.mul.Tensor(mul_tensor_7, 0.1);  mul_tensor_7 = None
        mul_tensor_9: "f32[64]" = torch.ops.aten.mul.Tensor(primals_11, 0.9)
        add_tensor_5: "f32[64]" = torch.ops.aten.add.Tensor(mul_tensor_8, mul_tensor_9);  mul_tensor_8 = mul_tensor_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_tensor_6: "i64[]" = torch.ops.aten.add.Tensor(primals_15, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims_4: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_5, [0, 2, 3]);  getitem_5 = None
        mul_tensor_10: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_dims_4, 0.1);  squeeze_dims_4 = None
        mul_tensor_11: "f32[64]" = torch.ops.aten.mul.Tensor(primals_16, 0.9)
        add_tensor_7: "f32[64]" = torch.ops.aten.add.Tensor(mul_tensor_10, mul_tensor_11);  mul_tensor_10 = mul_tensor_11 = None
        squeeze_dims_5: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_4, [0, 2, 3]);  getitem_4 = None
        mul_tensor_12: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_dims_5, 1.0000004768373856);  squeeze_dims_5 = None
        mul_tensor_13: "f32[64]" = torch.ops.aten.mul.Tensor(mul_tensor_12, 0.1);  mul_tensor_12 = None
        mul_tensor_14: "f32[64]" = torch.ops.aten.mul.Tensor(primals_17, 0.9)
        add_tensor_8: "f32[64]" = torch.ops.aten.add.Tensor(mul_tensor_13, mul_tensor_14);  mul_tensor_13 = mul_tensor_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_tensor_9: "i64[]" = torch.ops.aten.add.Tensor(primals_21, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims_6: "f32[32]" = torch.ops.aten.squeeze.dims(getitem_7, [0, 2, 3]);  getitem_7 = None
        squeeze_dims_7: "f32[32]" = torch.ops.aten.squeeze.dims(rsqrt_3, [0, 2, 3]);  rsqrt_3 = None
        mul_tensor_15: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_dims_6, 0.1)
        mul_tensor_16: "f32[32]" = torch.ops.aten.mul.Tensor(primals_22, 0.9)
        add_tensor_10: "f32[32]" = torch.ops.aten.add.Tensor(mul_tensor_15, mul_tensor_16);  mul_tensor_15 = mul_tensor_16 = None
        squeeze_dims_8: "f32[32]" = torch.ops.aten.squeeze.dims(getitem_6, [0, 2, 3]);  getitem_6 = None
        mul_tensor_17: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_dims_8, 1.0000004768373856);  squeeze_dims_8 = None
        mul_tensor_18: "f32[32]" = torch.ops.aten.mul.Tensor(mul_tensor_17, 0.1);  mul_tensor_17 = None
        mul_tensor_19: "f32[32]" = torch.ops.aten.mul.Tensor(primals_23, 0.9)
        add_tensor_11: "f32[32]" = torch.ops.aten.add.Tensor(mul_tensor_18, mul_tensor_19);  mul_tensor_18 = mul_tensor_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_tensor_12: "i64[]" = torch.ops.aten.add.Tensor(primals_27, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims_9: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_9, [0, 2, 3]);  getitem_9 = None
        mul_tensor_20: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_dims_9, 0.1);  squeeze_dims_9 = None
        mul_tensor_21: "f32[128]" = torch.ops.aten.mul.Tensor(primals_28, 0.9)
        add_tensor_13: "f32[128]" = torch.ops.aten.add.Tensor(mul_tensor_20, mul_tensor_21);  mul_tensor_20 = mul_tensor_21 = None
        squeeze_dims_10: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_8, [0, 2, 3]);  getitem_8 = None
        mul_tensor_22: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_dims_10, 1.0000004768373856);  squeeze_dims_10 = None
        mul_tensor_23: "f32[128]" = torch.ops.aten.mul.Tensor(mul_tensor_22, 0.1);  mul_tensor_22 = None
        mul_tensor_24: "f32[128]" = torch.ops.aten.mul.Tensor(primals_29, 0.9)
        add_tensor_14: "f32[128]" = torch.ops.aten.add.Tensor(mul_tensor_23, mul_tensor_24);  mul_tensor_23 = mul_tensor_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_tensor_15: "i64[]" = torch.ops.aten.add.Tensor(primals_33, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims_11: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_11, [0, 2, 3]);  getitem_11 = None
        mul_tensor_25: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_dims_11, 0.1);  squeeze_dims_11 = None
        mul_tensor_26: "f32[128]" = torch.ops.aten.mul.Tensor(primals_34, 0.9)
        add_tensor_16: "f32[128]" = torch.ops.aten.add.Tensor(mul_tensor_25, mul_tensor_26);  mul_tensor_25 = mul_tensor_26 = None
        squeeze_dims_12: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_10, [0, 2, 3]);  getitem_10 = None
        mul_tensor_27: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_dims_12, 1.0000019073522708);  squeeze_dims_12 = None
        mul_tensor_28: "f32[128]" = torch.ops.aten.mul.Tensor(mul_tensor_27, 0.1);  mul_tensor_27 = None
        mul_tensor_29: "f32[128]" = torch.ops.aten.mul.Tensor(primals_35, 0.9)
        add_tensor_17: "f32[128]" = torch.ops.aten.add.Tensor(mul_tensor_28, mul_tensor_29);  mul_tensor_28 = mul_tensor_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_tensor_18: "i64[]" = torch.ops.aten.add.Tensor(primals_39, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims_13: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_13, [0, 2, 3]);  getitem_13 = None
        squeeze_dims_14: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_6, [0, 2, 3]);  rsqrt_6 = None
        mul_tensor_30: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_dims_13, 0.1)
        mul_tensor_31: "f32[64]" = torch.ops.aten.mul.Tensor(primals_40, 0.9)
        add_tensor_19: "f32[64]" = torch.ops.aten.add.Tensor(mul_tensor_30, mul_tensor_31);  mul_tensor_30 = mul_tensor_31 = None
        squeeze_dims_15: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_12, [0, 2, 3]);  getitem_12 = None
        mul_tensor_32: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_dims_15, 1.0000019073522708);  squeeze_dims_15 = None
        mul_tensor_33: "f32[64]" = torch.ops.aten.mul.Tensor(mul_tensor_32, 0.1);  mul_tensor_32 = None
        mul_tensor_34: "f32[64]" = torch.ops.aten.mul.Tensor(primals_41, 0.9)
        add_tensor_20: "f32[64]" = torch.ops.aten.add.Tensor(mul_tensor_33, mul_tensor_34);  mul_tensor_33 = mul_tensor_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_tensor_21: "i64[]" = torch.ops.aten.add.Tensor(primals_45, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims_16: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_15, [0, 2, 3]);  getitem_15 = None
        mul_tensor_35: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_dims_16, 0.1);  squeeze_dims_16 = None
        mul_tensor_36: "f32[256]" = torch.ops.aten.mul.Tensor(primals_46, 0.9)
        add_tensor_22: "f32[256]" = torch.ops.aten.add.Tensor(mul_tensor_35, mul_tensor_36);  mul_tensor_35 = mul_tensor_36 = None
        squeeze_dims_17: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_14, [0, 2, 3]);  getitem_14 = None
        mul_tensor_37: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_dims_17, 1.0000019073522708);  squeeze_dims_17 = None
        mul_tensor_38: "f32[256]" = torch.ops.aten.mul.Tensor(mul_tensor_37, 0.1);  mul_tensor_37 = None
        mul_tensor_39: "f32[256]" = torch.ops.aten.mul.Tensor(primals_47, 0.9)
        add_tensor_23: "f32[256]" = torch.ops.aten.add.Tensor(mul_tensor_38, mul_tensor_39);  mul_tensor_38 = mul_tensor_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_tensor_24: "i64[]" = torch.ops.aten.add.Tensor(primals_51, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims_18: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_17, [0, 2, 3]);  getitem_17 = None
        mul_tensor_40: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_dims_18, 0.1);  squeeze_dims_18 = None
        mul_tensor_41: "f32[256]" = torch.ops.aten.mul.Tensor(primals_52, 0.9)
        add_tensor_25: "f32[256]" = torch.ops.aten.add.Tensor(mul_tensor_40, mul_tensor_41);  mul_tensor_40 = mul_tensor_41 = None
        squeeze_dims_19: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_16, [0, 2, 3]);  getitem_16 = None
        mul_tensor_42: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_dims_19, 1.0000019073522708);  squeeze_dims_19 = None
        mul_tensor_43: "f32[256]" = torch.ops.aten.mul.Tensor(mul_tensor_42, 0.1);  mul_tensor_42 = None
        mul_tensor_44: "f32[256]" = torch.ops.aten.mul.Tensor(primals_53, 0.9)
        add_tensor_26: "f32[256]" = torch.ops.aten.add.Tensor(mul_tensor_43, mul_tensor_44);  mul_tensor_43 = mul_tensor_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_tensor_27: "i64[]" = torch.ops.aten.add.Tensor(primals_57, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims_20: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_19, [0, 2, 3]);  getitem_19 = None
        squeeze_dims_21: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_9, [0, 2, 3]);  rsqrt_9 = None
        mul_tensor_45: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_dims_20, 0.1)
        mul_tensor_46: "f32[64]" = torch.ops.aten.mul.Tensor(primals_58, 0.9)
        add_tensor_28: "f32[64]" = torch.ops.aten.add.Tensor(mul_tensor_45, mul_tensor_46);  mul_tensor_45 = mul_tensor_46 = None
        squeeze_dims_22: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_18, [0, 2, 3]);  getitem_18 = None
        mul_tensor_47: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_dims_22, 1.0000019073522708);  squeeze_dims_22 = None
        mul_tensor_48: "f32[64]" = torch.ops.aten.mul.Tensor(mul_tensor_47, 0.1);  mul_tensor_47 = None
        mul_tensor_49: "f32[64]" = torch.ops.aten.mul.Tensor(primals_59, 0.9)
        add_tensor_29: "f32[64]" = torch.ops.aten.add.Tensor(mul_tensor_48, mul_tensor_49);  mul_tensor_48 = mul_tensor_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_tensor_30: "i64[]" = torch.ops.aten.add.Tensor(primals_63, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims_23: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_21, [0, 2, 3]);  getitem_21 = None
        mul_tensor_50: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_dims_23, 0.1);  squeeze_dims_23 = None
        mul_tensor_51: "f32[256]" = torch.ops.aten.mul.Tensor(primals_64, 0.9)
        add_tensor_31: "f32[256]" = torch.ops.aten.add.Tensor(mul_tensor_50, mul_tensor_51);  mul_tensor_50 = mul_tensor_51 = None
        squeeze_dims_24: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_20, [0, 2, 3]);  getitem_20 = None
        mul_tensor_52: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_dims_24, 1.0000019073522708);  squeeze_dims_24 = None
        mul_tensor_53: "f32[256]" = torch.ops.aten.mul.Tensor(mul_tensor_52, 0.1);  mul_tensor_52 = None
        mul_tensor_54: "f32[256]" = torch.ops.aten.mul.Tensor(primals_65, 0.9)
        add_tensor_32: "f32[256]" = torch.ops.aten.add.Tensor(mul_tensor_53, mul_tensor_54);  mul_tensor_53 = mul_tensor_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_tensor_33: "i64[]" = torch.ops.aten.add.Tensor(primals_69, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims_25: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_23, [0, 2, 3]);  getitem_23 = None
        mul_tensor_55: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_dims_25, 0.1);  squeeze_dims_25 = None
        mul_tensor_56: "f32[256]" = torch.ops.aten.mul.Tensor(primals_70, 0.9)
        add_tensor_34: "f32[256]" = torch.ops.aten.add.Tensor(mul_tensor_55, mul_tensor_56);  mul_tensor_55 = mul_tensor_56 = None
        squeeze_dims_26: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_22, [0, 2, 3]);  getitem_22 = None
        mul_tensor_57: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_dims_26, 1.0000019073522708);  squeeze_dims_26 = None
        mul_tensor_58: "f32[256]" = torch.ops.aten.mul.Tensor(mul_tensor_57, 0.1);  mul_tensor_57 = None
        mul_tensor_59: "f32[256]" = torch.ops.aten.mul.Tensor(primals_71, 0.9)
        add_tensor_35: "f32[256]" = torch.ops.aten.add.Tensor(mul_tensor_58, mul_tensor_59);  mul_tensor_58 = mul_tensor_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_tensor_36: "i64[]" = torch.ops.aten.add.Tensor(primals_75, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims_27: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_25, [0, 2, 3]);  getitem_25 = None
        squeeze_dims_28: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_12, [0, 2, 3]);  rsqrt_12 = None
        mul_tensor_60: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_dims_27, 0.1)
        mul_tensor_61: "f32[64]" = torch.ops.aten.mul.Tensor(primals_76, 0.9)
        add_tensor_37: "f32[64]" = torch.ops.aten.add.Tensor(mul_tensor_60, mul_tensor_61);  mul_tensor_60 = mul_tensor_61 = None
        squeeze_dims_29: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_24, [0, 2, 3]);  getitem_24 = None
        mul_tensor_62: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_dims_29, 1.0000019073522708);  squeeze_dims_29 = None
        mul_tensor_63: "f32[64]" = torch.ops.aten.mul.Tensor(mul_tensor_62, 0.1);  mul_tensor_62 = None
        mul_tensor_64: "f32[64]" = torch.ops.aten.mul.Tensor(primals_77, 0.9)
        add_tensor_38: "f32[64]" = torch.ops.aten.add.Tensor(mul_tensor_63, mul_tensor_64);  mul_tensor_63 = mul_tensor_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_tensor_39: "i64[]" = torch.ops.aten.add.Tensor(primals_81, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims_30: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_27, [0, 2, 3]);  getitem_27 = None
        mul_tensor_65: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_dims_30, 0.1);  squeeze_dims_30 = None
        mul_tensor_66: "f32[256]" = torch.ops.aten.mul.Tensor(primals_82, 0.9)
        add_tensor_40: "f32[256]" = torch.ops.aten.add.Tensor(mul_tensor_65, mul_tensor_66);  mul_tensor_65 = mul_tensor_66 = None
        squeeze_dims_31: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_26, [0, 2, 3]);  getitem_26 = None
        mul_tensor_67: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_dims_31, 1.0000019073522708);  squeeze_dims_31 = None
        mul_tensor_68: "f32[256]" = torch.ops.aten.mul.Tensor(mul_tensor_67, 0.1);  mul_tensor_67 = None
        mul_tensor_69: "f32[256]" = torch.ops.aten.mul.Tensor(primals_83, 0.9)
        add_tensor_41: "f32[256]" = torch.ops.aten.add.Tensor(mul_tensor_68, mul_tensor_69);  mul_tensor_68 = mul_tensor_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_tensor_42: "i64[]" = torch.ops.aten.add.Tensor(primals_87, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims_32: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_29, [0, 2, 3]);  getitem_29 = None
        mul_tensor_70: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_dims_32, 0.1);  squeeze_dims_32 = None
        mul_tensor_71: "f32[256]" = torch.ops.aten.mul.Tensor(primals_88, 0.9)
        add_tensor_43: "f32[256]" = torch.ops.aten.add.Tensor(mul_tensor_70, mul_tensor_71);  mul_tensor_70 = mul_tensor_71 = None
        squeeze_dims_33: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_28, [0, 2, 3]);  getitem_28 = None
        mul_tensor_72: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_dims_33, 1.0000076294527394);  squeeze_dims_33 = None
        mul_tensor_73: "f32[256]" = torch.ops.aten.mul.Tensor(mul_tensor_72, 0.1);  mul_tensor_72 = None
        mul_tensor_74: "f32[256]" = torch.ops.aten.mul.Tensor(primals_89, 0.9)
        add_tensor_44: "f32[256]" = torch.ops.aten.add.Tensor(mul_tensor_73, mul_tensor_74);  mul_tensor_73 = mul_tensor_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_tensor_45: "i64[]" = torch.ops.aten.add.Tensor(primals_93, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims_34: "f32[96]" = torch.ops.aten.squeeze.dims(getitem_31, [0, 2, 3]);  getitem_31 = None
        squeeze_dims_35: "f32[96]" = torch.ops.aten.squeeze.dims(rsqrt_15, [0, 2, 3]);  rsqrt_15 = None
        mul_tensor_75: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_dims_34, 0.1)
        mul_tensor_76: "f32[96]" = torch.ops.aten.mul.Tensor(primals_94, 0.9)
        add_tensor_46: "f32[96]" = torch.ops.aten.add.Tensor(mul_tensor_75, mul_tensor_76);  mul_tensor_75 = mul_tensor_76 = None
        squeeze_dims_36: "f32[96]" = torch.ops.aten.squeeze.dims(getitem_30, [0, 2, 3]);  getitem_30 = None
        mul_tensor_77: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_dims_36, 1.0000076294527394);  squeeze_dims_36 = None
        mul_tensor_78: "f32[96]" = torch.ops.aten.mul.Tensor(mul_tensor_77, 0.1);  mul_tensor_77 = None
        mul_tensor_79: "f32[96]" = torch.ops.aten.mul.Tensor(primals_95, 0.9)
        add_tensor_47: "f32[96]" = torch.ops.aten.add.Tensor(mul_tensor_78, mul_tensor_79);  mul_tensor_78 = mul_tensor_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_tensor_48: "i64[]" = torch.ops.aten.add.Tensor(primals_99, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims_37: "f32[96]" = torch.ops.aten.squeeze.dims(getitem_33, [0, 2, 3]);  getitem_33 = None
        mul_tensor_80: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_dims_37, 0.1);  squeeze_dims_37 = None
        mul_tensor_81: "f32[96]" = torch.ops.aten.mul.Tensor(primals_100, 0.9)
        add_tensor_49: "f32[96]" = torch.ops.aten.add.Tensor(mul_tensor_80, mul_tensor_81);  mul_tensor_80 = mul_tensor_81 = None
        squeeze_dims_38: "f32[96]" = torch.ops.aten.squeeze.dims(getitem_32, [0, 2, 3]);  getitem_32 = None
        mul_tensor_82: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_dims_38, 1.0000076294527394);  squeeze_dims_38 = None
        mul_tensor_83: "f32[96]" = torch.ops.aten.mul.Tensor(mul_tensor_82, 0.1);  mul_tensor_82 = None
        mul_tensor_84: "f32[96]" = torch.ops.aten.mul.Tensor(primals_101, 0.9)
        add_tensor_50: "f32[96]" = torch.ops.aten.add.Tensor(mul_tensor_83, mul_tensor_84);  mul_tensor_83 = mul_tensor_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_tensor_51: "i64[]" = torch.ops.aten.add.Tensor(primals_132, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims_39: "f32[96]" = torch.ops.aten.squeeze.dims(getitem_59, [0, 2, 3]);  getitem_59 = None
        mul_tensor_85: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_dims_39, 0.1);  squeeze_dims_39 = None
        mul_tensor_86: "f32[96]" = torch.ops.aten.mul.Tensor(primals_133, 0.9)
        add_tensor_52: "f32[96]" = torch.ops.aten.add.Tensor(mul_tensor_85, mul_tensor_86);  mul_tensor_85 = mul_tensor_86 = None
        squeeze_dims_40: "f32[96]" = torch.ops.aten.squeeze.dims(getitem_58, [0, 2, 3]);  getitem_58 = None
        mul_tensor_87: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_dims_40, 1.0000076294527394);  squeeze_dims_40 = None
        mul_tensor_88: "f32[96]" = torch.ops.aten.mul.Tensor(mul_tensor_87, 0.1);  mul_tensor_87 = None
        mul_tensor_89: "f32[96]" = torch.ops.aten.mul.Tensor(primals_134, 0.9)
        add_tensor_53: "f32[96]" = torch.ops.aten.add.Tensor(mul_tensor_88, mul_tensor_89);  mul_tensor_88 = mul_tensor_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_tensor_54: "i64[]" = torch.ops.aten.add.Tensor(primals_138, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims_41: "f32[96]" = torch.ops.aten.squeeze.dims(getitem_61, [0, 2, 3]);  getitem_61 = None
        mul_tensor_90: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_dims_41, 0.1);  squeeze_dims_41 = None
        mul_tensor_91: "f32[96]" = torch.ops.aten.mul.Tensor(primals_139, 0.9)
        add_tensor_55: "f32[96]" = torch.ops.aten.add.Tensor(mul_tensor_90, mul_tensor_91);  mul_tensor_90 = mul_tensor_91 = None
        squeeze_dims_42: "f32[96]" = torch.ops.aten.squeeze.dims(getitem_60, [0, 2, 3]);  getitem_60 = None
        mul_tensor_92: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_dims_42, 1.0000076294527394);  squeeze_dims_42 = None
        mul_tensor_93: "f32[96]" = torch.ops.aten.mul.Tensor(mul_tensor_92, 0.1);  mul_tensor_92 = None
        mul_tensor_94: "f32[96]" = torch.ops.aten.mul.Tensor(primals_140, 0.9)
        add_tensor_56: "f32[96]" = torch.ops.aten.add.Tensor(mul_tensor_93, mul_tensor_94);  mul_tensor_93 = mul_tensor_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_tensor_57: "i64[]" = torch.ops.aten.add.Tensor(primals_144, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims_43: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_63, [0, 2, 3]);  getitem_63 = None
        mul_tensor_95: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_43, 0.1);  squeeze_dims_43 = None
        mul_tensor_96: "f32[384]" = torch.ops.aten.mul.Tensor(primals_145, 0.9)
        add_tensor_58: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_95, mul_tensor_96);  mul_tensor_95 = mul_tensor_96 = None
        squeeze_dims_44: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_62, [0, 2, 3]);  getitem_62 = None
        mul_tensor_97: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_44, 1.0000076294527394);  squeeze_dims_44 = None
        mul_tensor_98: "f32[384]" = torch.ops.aten.mul.Tensor(mul_tensor_97, 0.1);  mul_tensor_97 = None
        mul_tensor_99: "f32[384]" = torch.ops.aten.mul.Tensor(primals_146, 0.9)
        add_tensor_59: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_98, mul_tensor_99);  mul_tensor_98 = mul_tensor_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_tensor_60: "i64[]" = torch.ops.aten.add.Tensor(primals_150, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims_45: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_65, [0, 2, 3]);  getitem_65 = None
        mul_tensor_100: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_45, 0.1);  squeeze_dims_45 = None
        mul_tensor_101: "f32[384]" = torch.ops.aten.mul.Tensor(primals_151, 0.9)
        add_tensor_61: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_100, mul_tensor_101);  mul_tensor_100 = mul_tensor_101 = None
        squeeze_dims_46: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_64, [0, 2, 3]);  getitem_64 = None
        mul_tensor_102: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_46, 1.000030518509476);  squeeze_dims_46 = None
        mul_tensor_103: "f32[384]" = torch.ops.aten.mul.Tensor(mul_tensor_102, 0.1);  mul_tensor_102 = None
        mul_tensor_104: "f32[384]" = torch.ops.aten.mul.Tensor(primals_152, 0.9)
        add_tensor_62: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_103, mul_tensor_104);  mul_tensor_103 = mul_tensor_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_tensor_63: "i64[]" = torch.ops.aten.add.Tensor(primals_156, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims_47: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_67, [0, 2, 3]);  getitem_67 = None
        squeeze_dims_48: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_26, [0, 2, 3]);  rsqrt_26 = None
        mul_tensor_105: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_dims_47, 0.1)
        mul_tensor_106: "f32[128]" = torch.ops.aten.mul.Tensor(primals_157, 0.9)
        add_tensor_64: "f32[128]" = torch.ops.aten.add.Tensor(mul_tensor_105, mul_tensor_106);  mul_tensor_105 = mul_tensor_106 = None
        squeeze_dims_49: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_66, [0, 2, 3]);  getitem_66 = None
        mul_tensor_107: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_dims_49, 1.000030518509476);  squeeze_dims_49 = None
        mul_tensor_108: "f32[128]" = torch.ops.aten.mul.Tensor(mul_tensor_107, 0.1);  mul_tensor_107 = None
        mul_tensor_109: "f32[128]" = torch.ops.aten.mul.Tensor(primals_158, 0.9)
        add_tensor_65: "f32[128]" = torch.ops.aten.add.Tensor(mul_tensor_108, mul_tensor_109);  mul_tensor_108 = mul_tensor_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_tensor_66: "i64[]" = torch.ops.aten.add.Tensor(primals_162, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims_50: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_69, [0, 2, 3]);  getitem_69 = None
        mul_tensor_110: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_dims_50, 0.1);  squeeze_dims_50 = None
        mul_tensor_111: "f32[128]" = torch.ops.aten.mul.Tensor(primals_163, 0.9)
        add_tensor_67: "f32[128]" = torch.ops.aten.add.Tensor(mul_tensor_110, mul_tensor_111);  mul_tensor_110 = mul_tensor_111 = None
        squeeze_dims_51: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_68, [0, 2, 3]);  getitem_68 = None
        mul_tensor_112: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_dims_51, 1.000030518509476);  squeeze_dims_51 = None
        mul_tensor_113: "f32[128]" = torch.ops.aten.mul.Tensor(mul_tensor_112, 0.1);  mul_tensor_112 = None
        mul_tensor_114: "f32[128]" = torch.ops.aten.mul.Tensor(primals_164, 0.9)
        add_tensor_68: "f32[128]" = torch.ops.aten.add.Tensor(mul_tensor_113, mul_tensor_114);  mul_tensor_113 = mul_tensor_114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_tensor_69: "i64[]" = torch.ops.aten.add.Tensor(primals_219, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims_52: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_117, [0, 2, 3]);  getitem_117 = None
        mul_tensor_115: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_dims_52, 0.1);  squeeze_dims_52 = None
        mul_tensor_116: "f32[128]" = torch.ops.aten.mul.Tensor(primals_220, 0.9)
        add_tensor_70: "f32[128]" = torch.ops.aten.add.Tensor(mul_tensor_115, mul_tensor_116);  mul_tensor_115 = mul_tensor_116 = None
        squeeze_dims_53: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_116, [0, 2, 3]);  getitem_116 = None
        mul_tensor_117: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_dims_53, 1.000030518509476);  squeeze_dims_53 = None
        mul_tensor_118: "f32[128]" = torch.ops.aten.mul.Tensor(mul_tensor_117, 0.1);  mul_tensor_117 = None
        mul_tensor_119: "f32[128]" = torch.ops.aten.mul.Tensor(primals_221, 0.9)
        add_tensor_71: "f32[128]" = torch.ops.aten.add.Tensor(mul_tensor_118, mul_tensor_119);  mul_tensor_118 = mul_tensor_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_tensor_72: "i64[]" = torch.ops.aten.add.Tensor(primals_225, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims_54: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_119, [0, 2, 3]);  getitem_119 = None
        mul_tensor_120: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_dims_54, 0.1);  squeeze_dims_54 = None
        mul_tensor_121: "f32[128]" = torch.ops.aten.mul.Tensor(primals_226, 0.9)
        add_tensor_73: "f32[128]" = torch.ops.aten.add.Tensor(mul_tensor_120, mul_tensor_121);  mul_tensor_120 = mul_tensor_121 = None
        squeeze_dims_55: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_118, [0, 2, 3]);  getitem_118 = None
        mul_tensor_122: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_dims_55, 1.000030518509476);  squeeze_dims_55 = None
        mul_tensor_123: "f32[128]" = torch.ops.aten.mul.Tensor(mul_tensor_122, 0.1);  mul_tensor_122 = None
        mul_tensor_124: "f32[128]" = torch.ops.aten.mul.Tensor(primals_227, 0.9)
        add_tensor_74: "f32[128]" = torch.ops.aten.add.Tensor(mul_tensor_123, mul_tensor_124);  mul_tensor_123 = mul_tensor_124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_tensor_75: "i64[]" = torch.ops.aten.add.Tensor(primals_231, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims_56: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_121, [0, 2, 3]);  getitem_121 = None
        mul_tensor_125: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_dims_56, 0.1);  squeeze_dims_56 = None
        mul_tensor_126: "f32[512]" = torch.ops.aten.mul.Tensor(primals_232, 0.9)
        add_tensor_76: "f32[512]" = torch.ops.aten.add.Tensor(mul_tensor_125, mul_tensor_126);  mul_tensor_125 = mul_tensor_126 = None
        squeeze_dims_57: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_120, [0, 2, 3]);  getitem_120 = None
        mul_tensor_127: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_dims_57, 1.000030518509476);  squeeze_dims_57 = None
        mul_tensor_128: "f32[512]" = torch.ops.aten.mul.Tensor(mul_tensor_127, 0.1);  mul_tensor_127 = None
        mul_tensor_129: "f32[512]" = torch.ops.aten.mul.Tensor(primals_233, 0.9)
        add_tensor_77: "f32[512]" = torch.ops.aten.add.Tensor(mul_tensor_128, mul_tensor_129);  mul_tensor_128 = mul_tensor_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_tensor_78: "i64[]" = torch.ops.aten.add.Tensor(primals_237, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims_58: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_123, [0, 2, 3]);  getitem_123 = None
        mul_tensor_130: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_dims_58, 0.1);  squeeze_dims_58 = None
        mul_tensor_131: "f32[512]" = torch.ops.aten.mul.Tensor(primals_238, 0.9)
        add_tensor_79: "f32[512]" = torch.ops.aten.add.Tensor(mul_tensor_130, mul_tensor_131);  mul_tensor_130 = mul_tensor_131 = None
        squeeze_dims_59: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_122, [0, 2, 3]);  getitem_122 = None
        mul_tensor_132: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_dims_59, 1.0001220852154804);  squeeze_dims_59 = None
        mul_tensor_133: "f32[512]" = torch.ops.aten.mul.Tensor(mul_tensor_132, 0.1);  mul_tensor_132 = None
        mul_tensor_134: "f32[512]" = torch.ops.aten.mul.Tensor(primals_239, 0.9)
        add_tensor_80: "f32[512]" = torch.ops.aten.add.Tensor(mul_tensor_133, mul_tensor_134);  mul_tensor_133 = mul_tensor_134 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_tensor_81: "i64[]" = torch.ops.aten.add.Tensor(primals_243, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims_60: "f32[160]" = torch.ops.aten.squeeze.dims(getitem_125, [0, 2, 3]);  getitem_125 = None
        squeeze_dims_61: "f32[160]" = torch.ops.aten.squeeze.dims(rsqrt_41, [0, 2, 3]);  rsqrt_41 = None
        mul_tensor_135: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_dims_60, 0.1)
        mul_tensor_136: "f32[160]" = torch.ops.aten.mul.Tensor(primals_244, 0.9)
        add_tensor_82: "f32[160]" = torch.ops.aten.add.Tensor(mul_tensor_135, mul_tensor_136);  mul_tensor_135 = mul_tensor_136 = None
        squeeze_dims_62: "f32[160]" = torch.ops.aten.squeeze.dims(getitem_124, [0, 2, 3]);  getitem_124 = None
        mul_tensor_137: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_dims_62, 1.0001220852154804);  squeeze_dims_62 = None
        mul_tensor_138: "f32[160]" = torch.ops.aten.mul.Tensor(mul_tensor_137, 0.1);  mul_tensor_137 = None
        mul_tensor_139: "f32[160]" = torch.ops.aten.mul.Tensor(primals_245, 0.9)
        add_tensor_83: "f32[160]" = torch.ops.aten.add.Tensor(mul_tensor_138, mul_tensor_139);  mul_tensor_138 = mul_tensor_139 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_tensor_84: "i64[]" = torch.ops.aten.add.Tensor(primals_249, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims_63: "f32[160]" = torch.ops.aten.squeeze.dims(getitem_127, [0, 2, 3]);  getitem_127 = None
        mul_tensor_140: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_dims_63, 0.1);  squeeze_dims_63 = None
        mul_tensor_141: "f32[160]" = torch.ops.aten.mul.Tensor(primals_250, 0.9)
        add_tensor_85: "f32[160]" = torch.ops.aten.add.Tensor(mul_tensor_140, mul_tensor_141);  mul_tensor_140 = mul_tensor_141 = None
        squeeze_dims_64: "f32[160]" = torch.ops.aten.squeeze.dims(getitem_126, [0, 2, 3]);  getitem_126 = None
        mul_tensor_142: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_dims_64, 1.0001220852154804);  squeeze_dims_64 = None
        mul_tensor_143: "f32[160]" = torch.ops.aten.mul.Tensor(mul_tensor_142, 0.1);  mul_tensor_142 = None
        mul_tensor_144: "f32[160]" = torch.ops.aten.mul.Tensor(primals_251, 0.9)
        add_tensor_86: "f32[160]" = torch.ops.aten.add.Tensor(mul_tensor_143, mul_tensor_144);  mul_tensor_143 = mul_tensor_144 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_tensor_87: "i64[]" = torch.ops.aten.add.Tensor(primals_294, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims_65: "f32[160]" = torch.ops.aten.squeeze.dims(getitem_164, [0, 2, 3]);  getitem_164 = None
        mul_tensor_145: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_dims_65, 0.1);  squeeze_dims_65 = None
        mul_tensor_146: "f32[160]" = torch.ops.aten.mul.Tensor(primals_295, 0.9)
        add_tensor_88: "f32[160]" = torch.ops.aten.add.Tensor(mul_tensor_145, mul_tensor_146);  mul_tensor_145 = mul_tensor_146 = None
        squeeze_dims_66: "f32[160]" = torch.ops.aten.squeeze.dims(getitem_163, [0, 2, 3]);  getitem_163 = None
        mul_tensor_147: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_dims_66, 1.0001220852154804);  squeeze_dims_66 = None
        mul_tensor_148: "f32[160]" = torch.ops.aten.mul.Tensor(mul_tensor_147, 0.1);  mul_tensor_147 = None
        mul_tensor_149: "f32[160]" = torch.ops.aten.mul.Tensor(primals_296, 0.9)
        add_tensor_89: "f32[160]" = torch.ops.aten.add.Tensor(mul_tensor_148, mul_tensor_149);  mul_tensor_148 = mul_tensor_149 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_tensor_90: "i64[]" = torch.ops.aten.add.Tensor(primals_300, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims_67: "f32[160]" = torch.ops.aten.squeeze.dims(getitem_166, [0, 2, 3]);  getitem_166 = None
        mul_tensor_150: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_dims_67, 0.1);  squeeze_dims_67 = None
        mul_tensor_151: "f32[160]" = torch.ops.aten.mul.Tensor(primals_301, 0.9)
        add_tensor_91: "f32[160]" = torch.ops.aten.add.Tensor(mul_tensor_150, mul_tensor_151);  mul_tensor_150 = mul_tensor_151 = None
        squeeze_dims_68: "f32[160]" = torch.ops.aten.squeeze.dims(getitem_165, [0, 2, 3]);  getitem_165 = None
        mul_tensor_152: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_dims_68, 1.0001220852154804);  squeeze_dims_68 = None
        mul_tensor_153: "f32[160]" = torch.ops.aten.mul.Tensor(mul_tensor_152, 0.1);  mul_tensor_152 = None
        mul_tensor_154: "f32[160]" = torch.ops.aten.mul.Tensor(primals_302, 0.9)
        add_tensor_92: "f32[160]" = torch.ops.aten.add.Tensor(mul_tensor_153, mul_tensor_154);  mul_tensor_153 = mul_tensor_154 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_tensor_93: "i64[]" = torch.ops.aten.add.Tensor(primals_306, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_correction = torch.ops.aten.var_mean.correction(convolution_34, [0, 2, 3], correction = 0, keepdim = True)
        getitem_167: "f32[1, 640, 1, 1]" = var_mean_correction[0]
        getitem_168: "f32[1, 640, 1, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor_94: "f32[1, 640, 1, 1]" = torch.ops.aten.add.Tensor(getitem_167, 1e-05)
        rsqrt_default: "f32[1, 640, 1, 1]" = torch.ops.aten.rsqrt.default(add_tensor_94);  add_tensor_94 = None
        sub_tensor: "f32[128, 640, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_34, getitem_168);  convolution_34 = None
        mul_tensor_155: "f32[128, 640, 8, 8]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        squeeze_dims_69: "f32[640]" = torch.ops.aten.squeeze.dims(getitem_168, [0, 2, 3]);  getitem_168 = None
        mul_tensor_156: "f32[640]" = torch.ops.aten.mul.Tensor(squeeze_dims_69, 0.1);  squeeze_dims_69 = None
        mul_tensor_157: "f32[640]" = torch.ops.aten.mul.Tensor(primals_307, 0.9)
        add_tensor_95: "f32[640]" = torch.ops.aten.add.Tensor(mul_tensor_156, mul_tensor_157);  mul_tensor_156 = mul_tensor_157 = None
        squeeze_dims_70: "f32[640]" = torch.ops.aten.squeeze.dims(getitem_167, [0, 2, 3]);  getitem_167 = None
        mul_tensor_158: "f32[640]" = torch.ops.aten.mul.Tensor(squeeze_dims_70, 1.0001220852154804);  squeeze_dims_70 = None
        mul_tensor_159: "f32[640]" = torch.ops.aten.mul.Tensor(mul_tensor_158, 0.1);  mul_tensor_158 = None
        mul_tensor_160: "f32[640]" = torch.ops.aten.mul.Tensor(primals_308, 0.9)
        add_tensor_96: "f32[640]" = torch.ops.aten.add.Tensor(mul_tensor_159, mul_tensor_160);  mul_tensor_159 = mul_tensor_160 = None
        unsqueeze_default: "f32[640, 1]" = torch.ops.aten.unsqueeze.default(primals_309, -1);  primals_309 = None
        unsqueeze_default_1: "f32[640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        mul_tensor_161: "f32[128, 640, 8, 8]" = torch.ops.aten.mul.Tensor(mul_tensor_155, unsqueeze_default_1);  mul_tensor_155 = unsqueeze_default_1 = None
        unsqueeze_default_2: "f32[640, 1]" = torch.ops.aten.unsqueeze.default(primals_310, -1);  primals_310 = None
        unsqueeze_default_3: "f32[640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        add_tensor_97: "f32[128, 640, 8, 8]" = torch.ops.aten.add.Tensor(mul_tensor_161, unsqueeze_default_3);  mul_tensor_161 = unsqueeze_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        neg_default: "f32[128, 640, 8, 8]" = torch.ops.aten.neg.default(add_tensor_97)
        exp_default: "f32[128, 640, 8, 8]" = torch.ops.aten.exp.default(neg_default);  neg_default = None
        add_tensor_98: "f32[128, 640, 8, 8]" = torch.ops.aten.add.Tensor(exp_default, 1);  exp_default = None
        div_tensor: "f32[128, 640, 8, 8]" = torch.ops.aten.div.Tensor(add_tensor_97, add_tensor_98);  add_tensor_97 = add_tensor_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/adaptive_avgmax_pool.py:172 in forward, code: x = self.pool(x)
        mean_dim: "f32[128, 640, 1, 1]" = torch.ops.aten.mean.dim(div_tensor, [-1, -2], True);  div_tensor = None
        as_strided_default: "f32[128, 640, 1, 1]" = torch.ops.aten.as_strided.default(mean_dim, [128, 640, 1, 1], [640, 1, 640, 640]);  mean_dim = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/adaptive_avgmax_pool.py:173 in forward, code: x = self.flatten(x)
        reshape_default: "f32[128, 640]" = torch.ops.aten.reshape.default(as_strided_default, _shape_param_0);  as_strided_default = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/classifier.py:141 in forward, code: x = self.fc(x)
        permute_default: "f32[640, 1000]" = torch.ops.aten.permute.default(primals_311, [1, 0]);  primals_311 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/mobilevit.py:264 in forward, code: x = self.norm(x)
        div_tensor_1: "f32[512, 16, 1]" = torch.ops.aten.div.Tensor(rsqrt_49, 240);  rsqrt_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        div_tensor_2: "f32[512, 16, 1]" = torch.ops.aten.div.Tensor(rsqrt_48, 240);  rsqrt_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        div_tensor_3: "f32[512, 16, 1]" = torch.ops.aten.div.Tensor(rsqrt_47, 240);  rsqrt_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        div_tensor_4: "f32[512, 16, 1]" = torch.ops.aten.div.Tensor(rsqrt_46, 240);  rsqrt_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        div_tensor_5: "f32[512, 16, 1]" = torch.ops.aten.div.Tensor(rsqrt_45, 240);  rsqrt_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        div_tensor_6: "f32[512, 16, 1]" = torch.ops.aten.div.Tensor(rsqrt_44, 240);  rsqrt_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        div_tensor_7: "f32[512, 16, 1]" = torch.ops.aten.div.Tensor(rsqrt_43, 240);  rsqrt_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        unsqueeze_default_4: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(squeeze_dims_60, 0);  squeeze_dims_60 = None
        unsqueeze_default_5: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 2);  unsqueeze_default_4 = None
        unsqueeze_default_6: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_5, 3);  unsqueeze_default_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/mobilevit.py:264 in forward, code: x = self.norm(x)
        div_tensor_8: "f32[512, 64, 1]" = torch.ops.aten.div.Tensor(rsqrt_36, 192);  rsqrt_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        div_tensor_9: "f32[512, 64, 1]" = torch.ops.aten.div.Tensor(rsqrt_35, 192);  rsqrt_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        div_tensor_10: "f32[512, 64, 1]" = torch.ops.aten.div.Tensor(rsqrt_34, 192);  rsqrt_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        div_tensor_11: "f32[512, 64, 1]" = torch.ops.aten.div.Tensor(rsqrt_33, 192);  rsqrt_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        div_tensor_12: "f32[512, 64, 1]" = torch.ops.aten.div.Tensor(rsqrt_32, 192);  rsqrt_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        div_tensor_13: "f32[512, 64, 1]" = torch.ops.aten.div.Tensor(rsqrt_31, 192);  rsqrt_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        div_tensor_14: "f32[512, 64, 1]" = torch.ops.aten.div.Tensor(rsqrt_30, 192);  rsqrt_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        div_tensor_15: "f32[512, 64, 1]" = torch.ops.aten.div.Tensor(rsqrt_29, 192);  rsqrt_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        div_tensor_16: "f32[512, 64, 1]" = torch.ops.aten.div.Tensor(rsqrt_28, 192);  rsqrt_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        unsqueeze_default_7: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_dims_47, 0);  squeeze_dims_47 = None
        unsqueeze_default_8: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_7, 2);  unsqueeze_default_7 = None
        unsqueeze_default_9: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_8, 3);  unsqueeze_default_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/mobilevit.py:264 in forward, code: x = self.norm(x)
        div_tensor_17: "f32[512, 256, 1]" = torch.ops.aten.div.Tensor(rsqrt_21, 144);  rsqrt_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        div_tensor_18: "f32[512, 256, 1]" = torch.ops.aten.div.Tensor(rsqrt_20, 144);  rsqrt_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        div_tensor_19: "f32[512, 256, 1]" = torch.ops.aten.div.Tensor(rsqrt_19, 144);  rsqrt_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        div_tensor_20: "f32[512, 256, 1]" = torch.ops.aten.div.Tensor(rsqrt_18, 144);  rsqrt_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        div_tensor_21: "f32[512, 256, 1]" = torch.ops.aten.div.Tensor(rsqrt_17, 144);  rsqrt_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        unsqueeze_default_10: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(squeeze_dims_34, 0);  squeeze_dims_34 = None
        unsqueeze_default_11: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_10, 2);  unsqueeze_default_10 = None
        unsqueeze_default_12: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_11, 3);  unsqueeze_default_11 = None
        unsqueeze_default_13: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_dims_27, 0);  squeeze_dims_27 = None
        unsqueeze_default_14: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_13, 2);  unsqueeze_default_13 = None
        unsqueeze_default_15: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_14, 3);  unsqueeze_default_14 = None
        unsqueeze_default_16: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_dims_20, 0);  squeeze_dims_20 = None
        unsqueeze_default_17: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_16, 2);  unsqueeze_default_16 = None
        unsqueeze_default_18: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_17, 3);  unsqueeze_default_17 = None
        unsqueeze_default_19: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_dims_13, 0);  squeeze_dims_13 = None
        unsqueeze_default_20: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_19, 2);  unsqueeze_default_19 = None
        unsqueeze_default_21: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_20, 3);  unsqueeze_default_20 = None
        unsqueeze_default_22: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(squeeze_dims_6, 0);  squeeze_dims_6 = None
        unsqueeze_default_23: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_22, 2);  unsqueeze_default_22 = None
        unsqueeze_default_24: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_23, 3);  unsqueeze_default_23 = None

        # No stacktrace found for following nodes
        copy__default: "i64[]" = torch.ops.aten.copy_.default(primals_3, add_tensor);  primals_3 = add_tensor = None
        copy__default_1: "f32[16]" = torch.ops.aten.copy_.default(primals_4, add_tensor_1);  primals_4 = add_tensor_1 = None
        copy__default_2: "f32[16]" = torch.ops.aten.copy_.default(primals_5, add_tensor_2);  primals_5 = add_tensor_2 = None
        copy__default_3: "i64[]" = torch.ops.aten.copy_.default(primals_9, add_tensor_3);  primals_9 = add_tensor_3 = None
        copy__default_4: "f32[64]" = torch.ops.aten.copy_.default(primals_10, add_tensor_4);  primals_10 = add_tensor_4 = None
        copy__default_5: "f32[64]" = torch.ops.aten.copy_.default(primals_11, add_tensor_5);  primals_11 = add_tensor_5 = None
        copy__default_6: "i64[]" = torch.ops.aten.copy_.default(primals_15, add_tensor_6);  primals_15 = add_tensor_6 = None
        copy__default_7: "f32[64]" = torch.ops.aten.copy_.default(primals_16, add_tensor_7);  primals_16 = add_tensor_7 = None
        copy__default_8: "f32[64]" = torch.ops.aten.copy_.default(primals_17, add_tensor_8);  primals_17 = add_tensor_8 = None
        copy__default_9: "i64[]" = torch.ops.aten.copy_.default(primals_21, add_tensor_9);  primals_21 = add_tensor_9 = None
        copy__default_10: "f32[32]" = torch.ops.aten.copy_.default(primals_22, add_tensor_10);  primals_22 = add_tensor_10 = None
        copy__default_11: "f32[32]" = torch.ops.aten.copy_.default(primals_23, add_tensor_11);  primals_23 = add_tensor_11 = None
        copy__default_12: "i64[]" = torch.ops.aten.copy_.default(primals_27, add_tensor_12);  primals_27 = add_tensor_12 = None
        copy__default_13: "f32[128]" = torch.ops.aten.copy_.default(primals_28, add_tensor_13);  primals_28 = add_tensor_13 = None
        copy__default_14: "f32[128]" = torch.ops.aten.copy_.default(primals_29, add_tensor_14);  primals_29 = add_tensor_14 = None
        copy__default_15: "i64[]" = torch.ops.aten.copy_.default(primals_33, add_tensor_15);  primals_33 = add_tensor_15 = None
        copy__default_16: "f32[128]" = torch.ops.aten.copy_.default(primals_34, add_tensor_16);  primals_34 = add_tensor_16 = None
        copy__default_17: "f32[128]" = torch.ops.aten.copy_.default(primals_35, add_tensor_17);  primals_35 = add_tensor_17 = None
        copy__default_18: "i64[]" = torch.ops.aten.copy_.default(primals_39, add_tensor_18);  primals_39 = add_tensor_18 = None
        copy__default_19: "f32[64]" = torch.ops.aten.copy_.default(primals_40, add_tensor_19);  primals_40 = add_tensor_19 = None
        copy__default_20: "f32[64]" = torch.ops.aten.copy_.default(primals_41, add_tensor_20);  primals_41 = add_tensor_20 = None
        copy__default_21: "i64[]" = torch.ops.aten.copy_.default(primals_45, add_tensor_21);  primals_45 = add_tensor_21 = None
        copy__default_22: "f32[256]" = torch.ops.aten.copy_.default(primals_46, add_tensor_22);  primals_46 = add_tensor_22 = None
        copy__default_23: "f32[256]" = torch.ops.aten.copy_.default(primals_47, add_tensor_23);  primals_47 = add_tensor_23 = None
        copy__default_24: "i64[]" = torch.ops.aten.copy_.default(primals_51, add_tensor_24);  primals_51 = add_tensor_24 = None
        copy__default_25: "f32[256]" = torch.ops.aten.copy_.default(primals_52, add_tensor_25);  primals_52 = add_tensor_25 = None
        copy__default_26: "f32[256]" = torch.ops.aten.copy_.default(primals_53, add_tensor_26);  primals_53 = add_tensor_26 = None
        copy__default_27: "i64[]" = torch.ops.aten.copy_.default(primals_57, add_tensor_27);  primals_57 = add_tensor_27 = None
        copy__default_28: "f32[64]" = torch.ops.aten.copy_.default(primals_58, add_tensor_28);  primals_58 = add_tensor_28 = None
        copy__default_29: "f32[64]" = torch.ops.aten.copy_.default(primals_59, add_tensor_29);  primals_59 = add_tensor_29 = None
        copy__default_30: "i64[]" = torch.ops.aten.copy_.default(primals_63, add_tensor_30);  primals_63 = add_tensor_30 = None
        copy__default_31: "f32[256]" = torch.ops.aten.copy_.default(primals_64, add_tensor_31);  primals_64 = add_tensor_31 = None
        copy__default_32: "f32[256]" = torch.ops.aten.copy_.default(primals_65, add_tensor_32);  primals_65 = add_tensor_32 = None
        copy__default_33: "i64[]" = torch.ops.aten.copy_.default(primals_69, add_tensor_33);  primals_69 = add_tensor_33 = None
        copy__default_34: "f32[256]" = torch.ops.aten.copy_.default(primals_70, add_tensor_34);  primals_70 = add_tensor_34 = None
        copy__default_35: "f32[256]" = torch.ops.aten.copy_.default(primals_71, add_tensor_35);  primals_71 = add_tensor_35 = None
        copy__default_36: "i64[]" = torch.ops.aten.copy_.default(primals_75, add_tensor_36);  primals_75 = add_tensor_36 = None
        copy__default_37: "f32[64]" = torch.ops.aten.copy_.default(primals_76, add_tensor_37);  primals_76 = add_tensor_37 = None
        copy__default_38: "f32[64]" = torch.ops.aten.copy_.default(primals_77, add_tensor_38);  primals_77 = add_tensor_38 = None
        copy__default_39: "i64[]" = torch.ops.aten.copy_.default(primals_81, add_tensor_39);  primals_81 = add_tensor_39 = None
        copy__default_40: "f32[256]" = torch.ops.aten.copy_.default(primals_82, add_tensor_40);  primals_82 = add_tensor_40 = None
        copy__default_41: "f32[256]" = torch.ops.aten.copy_.default(primals_83, add_tensor_41);  primals_83 = add_tensor_41 = None
        copy__default_42: "i64[]" = torch.ops.aten.copy_.default(primals_87, add_tensor_42);  primals_87 = add_tensor_42 = None
        copy__default_43: "f32[256]" = torch.ops.aten.copy_.default(primals_88, add_tensor_43);  primals_88 = add_tensor_43 = None
        copy__default_44: "f32[256]" = torch.ops.aten.copy_.default(primals_89, add_tensor_44);  primals_89 = add_tensor_44 = None
        copy__default_45: "i64[]" = torch.ops.aten.copy_.default(primals_93, add_tensor_45);  primals_93 = add_tensor_45 = None
        copy__default_46: "f32[96]" = torch.ops.aten.copy_.default(primals_94, add_tensor_46);  primals_94 = add_tensor_46 = None
        copy__default_47: "f32[96]" = torch.ops.aten.copy_.default(primals_95, add_tensor_47);  primals_95 = add_tensor_47 = None
        copy__default_48: "i64[]" = torch.ops.aten.copy_.default(primals_99, add_tensor_48);  primals_99 = add_tensor_48 = None
        copy__default_49: "f32[96]" = torch.ops.aten.copy_.default(primals_100, add_tensor_49);  primals_100 = add_tensor_49 = None
        copy__default_50: "f32[96]" = torch.ops.aten.copy_.default(primals_101, add_tensor_50);  primals_101 = add_tensor_50 = None
        copy__default_51: "i64[]" = torch.ops.aten.copy_.default(primals_132, add_tensor_51);  primals_132 = add_tensor_51 = None
        copy__default_52: "f32[96]" = torch.ops.aten.copy_.default(primals_133, add_tensor_52);  primals_133 = add_tensor_52 = None
        copy__default_53: "f32[96]" = torch.ops.aten.copy_.default(primals_134, add_tensor_53);  primals_134 = add_tensor_53 = None
        copy__default_54: "i64[]" = torch.ops.aten.copy_.default(primals_138, add_tensor_54);  primals_138 = add_tensor_54 = None
        copy__default_55: "f32[96]" = torch.ops.aten.copy_.default(primals_139, add_tensor_55);  primals_139 = add_tensor_55 = None
        copy__default_56: "f32[96]" = torch.ops.aten.copy_.default(primals_140, add_tensor_56);  primals_140 = add_tensor_56 = None
        copy__default_57: "i64[]" = torch.ops.aten.copy_.default(primals_144, add_tensor_57);  primals_144 = add_tensor_57 = None
        copy__default_58: "f32[384]" = torch.ops.aten.copy_.default(primals_145, add_tensor_58);  primals_145 = add_tensor_58 = None
        copy__default_59: "f32[384]" = torch.ops.aten.copy_.default(primals_146, add_tensor_59);  primals_146 = add_tensor_59 = None
        copy__default_60: "i64[]" = torch.ops.aten.copy_.default(primals_150, add_tensor_60);  primals_150 = add_tensor_60 = None
        copy__default_61: "f32[384]" = torch.ops.aten.copy_.default(primals_151, add_tensor_61);  primals_151 = add_tensor_61 = None
        copy__default_62: "f32[384]" = torch.ops.aten.copy_.default(primals_152, add_tensor_62);  primals_152 = add_tensor_62 = None
        copy__default_63: "i64[]" = torch.ops.aten.copy_.default(primals_156, add_tensor_63);  primals_156 = add_tensor_63 = None
        copy__default_64: "f32[128]" = torch.ops.aten.copy_.default(primals_157, add_tensor_64);  primals_157 = add_tensor_64 = None
        copy__default_65: "f32[128]" = torch.ops.aten.copy_.default(primals_158, add_tensor_65);  primals_158 = add_tensor_65 = None
        copy__default_66: "i64[]" = torch.ops.aten.copy_.default(primals_162, add_tensor_66);  primals_162 = add_tensor_66 = None
        copy__default_67: "f32[128]" = torch.ops.aten.copy_.default(primals_163, add_tensor_67);  primals_163 = add_tensor_67 = None
        copy__default_68: "f32[128]" = torch.ops.aten.copy_.default(primals_164, add_tensor_68);  primals_164 = add_tensor_68 = None
        copy__default_69: "i64[]" = torch.ops.aten.copy_.default(primals_219, add_tensor_69);  primals_219 = add_tensor_69 = None
        copy__default_70: "f32[128]" = torch.ops.aten.copy_.default(primals_220, add_tensor_70);  primals_220 = add_tensor_70 = None
        copy__default_71: "f32[128]" = torch.ops.aten.copy_.default(primals_221, add_tensor_71);  primals_221 = add_tensor_71 = None
        copy__default_72: "i64[]" = torch.ops.aten.copy_.default(primals_225, add_tensor_72);  primals_225 = add_tensor_72 = None
        copy__default_73: "f32[128]" = torch.ops.aten.copy_.default(primals_226, add_tensor_73);  primals_226 = add_tensor_73 = None
        copy__default_74: "f32[128]" = torch.ops.aten.copy_.default(primals_227, add_tensor_74);  primals_227 = add_tensor_74 = None
        copy__default_75: "i64[]" = torch.ops.aten.copy_.default(primals_231, add_tensor_75);  primals_231 = add_tensor_75 = None
        copy__default_76: "f32[512]" = torch.ops.aten.copy_.default(primals_232, add_tensor_76);  primals_232 = add_tensor_76 = None
        copy__default_77: "f32[512]" = torch.ops.aten.copy_.default(primals_233, add_tensor_77);  primals_233 = add_tensor_77 = None
        copy__default_78: "i64[]" = torch.ops.aten.copy_.default(primals_237, add_tensor_78);  primals_237 = add_tensor_78 = None
        copy__default_79: "f32[512]" = torch.ops.aten.copy_.default(primals_238, add_tensor_79);  primals_238 = add_tensor_79 = None
        copy__default_80: "f32[512]" = torch.ops.aten.copy_.default(primals_239, add_tensor_80);  primals_239 = add_tensor_80 = None
        copy__default_81: "i64[]" = torch.ops.aten.copy_.default(primals_243, add_tensor_81);  primals_243 = add_tensor_81 = None
        copy__default_82: "f32[160]" = torch.ops.aten.copy_.default(primals_244, add_tensor_82);  primals_244 = add_tensor_82 = None
        copy__default_83: "f32[160]" = torch.ops.aten.copy_.default(primals_245, add_tensor_83);  primals_245 = add_tensor_83 = None
        copy__default_84: "i64[]" = torch.ops.aten.copy_.default(primals_249, add_tensor_84);  primals_249 = add_tensor_84 = None
        copy__default_85: "f32[160]" = torch.ops.aten.copy_.default(primals_250, add_tensor_85);  primals_250 = add_tensor_85 = None
        copy__default_86: "f32[160]" = torch.ops.aten.copy_.default(primals_251, add_tensor_86);  primals_251 = add_tensor_86 = None
        copy__default_87: "i64[]" = torch.ops.aten.copy_.default(primals_294, add_tensor_87);  primals_294 = add_tensor_87 = None
        copy__default_88: "f32[160]" = torch.ops.aten.copy_.default(primals_295, add_tensor_88);  primals_295 = add_tensor_88 = None
        copy__default_89: "f32[160]" = torch.ops.aten.copy_.default(primals_296, add_tensor_89);  primals_296 = add_tensor_89 = None
        copy__default_90: "i64[]" = torch.ops.aten.copy_.default(primals_300, add_tensor_90);  primals_300 = add_tensor_90 = None
        copy__default_91: "f32[160]" = torch.ops.aten.copy_.default(primals_301, add_tensor_91);  primals_301 = add_tensor_91 = None
        copy__default_92: "f32[160]" = torch.ops.aten.copy_.default(primals_302, add_tensor_92);  primals_302 = add_tensor_92 = None
        copy__default_93: "i64[]" = torch.ops.aten.copy_.default(primals_306, add_tensor_93);  primals_306 = add_tensor_93 = None
        copy__default_94: "f32[640]" = torch.ops.aten.copy_.default(primals_307, add_tensor_95);  primals_307 = add_tensor_95 = None
        copy__default_95: "f32[640]" = torch.ops.aten.copy_.default(primals_308, add_tensor_96);  primals_308 = add_tensor_96 = None
        return (squeeze_dims_7, squeeze_dims_14, squeeze_dims_21, squeeze_dims_28, squeeze_dims_35, squeeze_dims_48, squeeze_dims_61, reshape_default, permute_default, div_tensor_1, div_tensor_2, div_tensor_3, div_tensor_4, div_tensor_5, div_tensor_6, div_tensor_7, unsqueeze_default_6, div_tensor_8, div_tensor_9, div_tensor_10, div_tensor_11, div_tensor_12, div_tensor_13, div_tensor_14, div_tensor_15, div_tensor_16, unsqueeze_default_9, div_tensor_17, div_tensor_18, div_tensor_19, div_tensor_20, div_tensor_21, unsqueeze_default_12, unsqueeze_default_15, unsqueeze_default_18, unsqueeze_default_21, unsqueeze_default_24, copy__default, copy__default_1, copy__default_2, copy__default_3, copy__default_4, copy__default_5, copy__default_6, copy__default_7, copy__default_8, copy__default_9, copy__default_10, copy__default_11, copy__default_12, copy__default_13, copy__default_14, copy__default_15, copy__default_16, copy__default_17, copy__default_18, copy__default_19, copy__default_20, copy__default_21, copy__default_22, copy__default_23, copy__default_24, copy__default_25, copy__default_26, copy__default_27, copy__default_28, copy__default_29, copy__default_30, copy__default_31, copy__default_32, copy__default_33, copy__default_34, copy__default_35, copy__default_36, copy__default_37, copy__default_38, copy__default_39, copy__default_40, copy__default_41, copy__default_42, copy__default_43, copy__default_44, copy__default_45, copy__default_46, copy__default_47, copy__default_48, copy__default_49, copy__default_50, copy__default_51, copy__default_52, copy__default_53, copy__default_54, copy__default_55, copy__default_56, copy__default_57, copy__default_58, copy__default_59, copy__default_60, copy__default_61, copy__default_62, copy__default_63, copy__default_64, copy__default_65, copy__default_66, copy__default_67, copy__default_68, copy__default_69, copy__default_70, copy__default_71, copy__default_72, copy__default_73, copy__default_74, copy__default_75, copy__default_76, copy__default_77, copy__default_78, copy__default_79, copy__default_80, copy__default_81, copy__default_82, copy__default_83, copy__default_84, copy__default_85, copy__default_86, copy__default_87, copy__default_88, copy__default_89, copy__default_90, copy__default_91, copy__default_92, copy__default_93, copy__default_94, copy__default_95)



def make_inputs():
    return [
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 16, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([16], dtype=torch.float32, device='cuda'),
    torch.randn([1, 16, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([16], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
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
    torch.randn([1, 32, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 32, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([32], dtype=torch.float32, device='cuda'),
    torch.randn([1, 32, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([32], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
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
    torch.randn([1, 64, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 64, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([64], dtype=torch.float32, device='cuda'),
    torch.randn([1, 64, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([64], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
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
    torch.randn([1, 64, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 64, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([64], dtype=torch.float32, device='cuda'),
    torch.randn([1, 64, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([64], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
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
    torch.randn([1, 64, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 64, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([64], dtype=torch.float32, device='cuda'),
    torch.randn([1, 64, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([64], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
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
    torch.randn([1, 96, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 96, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([96], dtype=torch.float32, device='cuda'),
    torch.randn([1, 96, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([96], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 96, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([96], dtype=torch.float32, device='cuda'),
    torch.randn([1, 96, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([96], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 96, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([96], dtype=torch.float32, device='cuda'),
    torch.randn([1, 96, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([96], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 96, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([96], dtype=torch.float32, device='cuda'),
    torch.randn([1, 96, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([96], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 384, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([1, 384, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 384, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([1, 384, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
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
    torch.randn([1, 160, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 160, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([160], dtype=torch.float32, device='cuda'),
    torch.randn([1, 160, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([160], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 160, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([160], dtype=torch.float32, device='cuda'),
    torch.randn([1, 160, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([160], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 160, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([160], dtype=torch.float32, device='cuda'),
    torch.randn([1, 160, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([160], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 160, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([160], dtype=torch.float32, device='cuda'),
    torch.randn([1, 160, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([160], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn(5242880, dtype=torch.float32, device='cuda').as_strided([128, 640, 8, 8], [40960, 1, 5120, 640]),  # convolution_34
    torch.randn([640], dtype=torch.float32, device='cuda'),
    torch.randn([640], dtype=torch.float32, device='cuda'),
    torch.randn([640], dtype=torch.float32, device='cuda'),
    torch.randn([640], dtype=torch.float32, device='cuda'),
    torch.randn([1000, 640], dtype=torch.float32, device='cuda'),
    torch.randn([512, 16, 1], dtype=torch.float32, device='cuda'),
    torch.randn([512, 16, 1], dtype=torch.float32, device='cuda'),
    torch.randn([512, 16, 1], dtype=torch.float32, device='cuda'),
    torch.randn([512, 16, 1], dtype=torch.float32, device='cuda'),
    torch.randn([512, 16, 1], dtype=torch.float32, device='cuda'),
    torch.randn([512, 16, 1], dtype=torch.float32, device='cuda'),
    torch.randn([512, 16, 1], dtype=torch.float32, device='cuda'),
    torch.randn([512, 64, 1], dtype=torch.float32, device='cuda'),
    torch.randn([512, 64, 1], dtype=torch.float32, device='cuda'),
    torch.randn([512, 64, 1], dtype=torch.float32, device='cuda'),
    torch.randn([512, 64, 1], dtype=torch.float32, device='cuda'),
    torch.randn([512, 64, 1], dtype=torch.float32, device='cuda'),
    torch.randn([512, 64, 1], dtype=torch.float32, device='cuda'),
    torch.randn([512, 64, 1], dtype=torch.float32, device='cuda'),
    torch.randn([512, 64, 1], dtype=torch.float32, device='cuda'),
    torch.randn([512, 64, 1], dtype=torch.float32, device='cuda'),
    torch.randn([512, 256, 1], dtype=torch.float32, device='cuda'),
    torch.randn([512, 256, 1], dtype=torch.float32, device='cuda'),
    torch.randn([512, 256, 1], dtype=torch.float32, device='cuda'),
    torch.randn([512, 256, 1], dtype=torch.float32, device='cuda'),
    torch.randn([512, 256, 1], dtype=torch.float32, device='cuda'),
    [128, 640],  # _shape_param_0
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
