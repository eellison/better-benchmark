import torch
import torch._inductor.inductor_prims  # registers inductor prims ops
from torch import device
from math import inf, nan

class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "f32[64, 3, 3, 3]", arg1_1: "f32[64]", arg2_1: "f32[8, 3, 640, 959]", arg3_1: "f32[64]", arg4_1: "f32[64]", arg5_1: "f32[64]", arg6_1: "f32[64]", arg7_1: "f32[64, 64, 3, 3]", arg8_1: "f32[64]", arg9_1: "f32[64]", arg10_1: "f32[64]", arg11_1: "f32[64]", arg12_1: "f32[64]", arg13_1: "f32[128, 64, 3, 3]", arg14_1: "f32[128]", arg15_1: "f32[128]", arg16_1: "f32[128]", arg17_1: "f32[128]", arg18_1: "f32[128]", arg19_1: "f32[128, 128, 3, 3]", arg20_1: "f32[128]", arg21_1: "f32[128]", arg22_1: "f32[128]", arg23_1: "f32[128]", arg24_1: "f32[128]", arg25_1: "f32[256, 128, 3, 3]", arg26_1: "f32[256]", arg27_1: "f32[256]", arg28_1: "f32[256]", arg29_1: "f32[256]", arg30_1: "f32[256]", arg31_1: "f32[256, 256, 3, 3]", arg32_1: "f32[256]", arg33_1: "f32[256]", arg34_1: "f32[256]", arg35_1: "f32[256]", arg36_1: "f32[256]", arg37_1: "f32[512, 256, 3, 3]", arg38_1: "f32[512]", arg39_1: "f32[512]", arg40_1: "f32[512]", arg41_1: "f32[512]", arg42_1: "f32[512]", arg43_1: "f32[512, 512, 3, 3]", arg44_1: "f32[512]", arg45_1: "f32[512]", arg46_1: "f32[512]", arg47_1: "f32[512]", arg48_1: "f32[512]", arg49_1: "f32[512, 512, 3, 3]", arg50_1: "f32[512]", arg51_1: "f32[512]", arg52_1: "f32[512]", arg53_1: "f32[512]", arg54_1: "f32[512]", arg55_1: "f32[512, 512, 3, 3]", arg56_1: "f32[512]", arg57_1: "f32[512]", arg58_1: "f32[512]", arg59_1: "f32[512]", arg60_1: "f32[512]", arg61_1: "f32[512, 1024, 3, 3]", arg62_1: "f32[512]", arg63_1: "f32[512]", arg64_1: "f32[512]", arg65_1: "f32[512]", arg66_1: "f32[512]", arg67_1: "f32[256, 512, 3, 3]", arg68_1: "f32[256]", arg69_1: "f32[256]", arg70_1: "f32[256]", arg71_1: "f32[256]", arg72_1: "f32[256]", arg73_1: "f32[256, 512, 3, 3]", arg74_1: "f32[256]", arg75_1: "f32[256]", arg76_1: "f32[256]", arg77_1: "f32[256]", arg78_1: "f32[256]", arg79_1: "f32[128, 256, 3, 3]", arg80_1: "f32[128]", arg81_1: "f32[128]", arg82_1: "f32[128]", arg83_1: "f32[128]", arg84_1: "f32[128]", arg85_1: "f32[128, 256, 3, 3]", arg86_1: "f32[128]", arg87_1: "f32[128]", arg88_1: "f32[128]", arg89_1: "f32[128]", arg90_1: "f32[128]", arg91_1: "f32[64, 128, 3, 3]", arg92_1: "f32[64]", arg93_1: "f32[64]", arg94_1: "f32[64]", arg95_1: "f32[64]", arg96_1: "f32[64]", arg97_1: "f32[64, 128, 3, 3]", arg98_1: "f32[64]", arg99_1: "f32[64]", arg100_1: "f32[64]", arg101_1: "f32[64]", arg102_1: "f32[64]", arg103_1: "f32[64, 64, 3, 3]", arg104_1: "f32[64]", arg105_1: "f32[64]", arg106_1: "f32[64]", arg107_1: "f32[64]", arg108_1: "f32[64]", arg109_1: "f32[2, 64, 1, 1]", arg110_1: "f32[2]"):
        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pytorch_unet/pytorch_unet/unet/unet_parts.py:25 in forward, code: return self.double_conv(x)
        convolution: "f32[8, 64, 640, 959]" = torch.ops.aten.convolution.default(arg2_1, arg0_1, arg1_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg2_1 = arg0_1 = arg1_1 = None
        unsqueeze: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg3_1, -1);  arg3_1 = None
        unsqueeze_1: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, -1);  unsqueeze = None
        sub: "f32[8, 64, 640, 959]" = torch.ops.aten.sub.Tensor(convolution, unsqueeze_1);  convolution = unsqueeze_1 = None
        add: "f32[64]" = torch.ops.aten.add.Tensor(arg4_1, 1e-05);  arg4_1 = None
        sqrt: "f32[64]" = torch.ops.aten.sqrt.default(add);  add = None
        reciprocal: "f32[64]" = torch.ops.aten.reciprocal.default(sqrt);  sqrt = None
        mul: "f32[64]" = torch.ops.aten.mul.Tensor(reciprocal, 1);  reciprocal = None
        unsqueeze_2: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(mul, -1);  mul = None
        unsqueeze_3: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, -1);  unsqueeze_2 = None
        mul_1: "f32[8, 64, 640, 959]" = torch.ops.aten.mul.Tensor(sub, unsqueeze_3);  sub = unsqueeze_3 = None
        unsqueeze_4: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg5_1, -1);  arg5_1 = None
        unsqueeze_5: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, -1);  unsqueeze_4 = None
        mul_2: "f32[8, 64, 640, 959]" = torch.ops.aten.mul.Tensor(mul_1, unsqueeze_5);  mul_1 = unsqueeze_5 = None
        unsqueeze_6: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg6_1, -1);  arg6_1 = None
        unsqueeze_7: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_6, -1);  unsqueeze_6 = None
        add_1: "f32[8, 64, 640, 959]" = torch.ops.aten.add.Tensor(mul_2, unsqueeze_7);  mul_2 = unsqueeze_7 = None
        relu: "f32[8, 64, 640, 959]" = torch.ops.aten.relu.default(add_1);  add_1 = None
        convolution_1: "f32[8, 64, 640, 959]" = torch.ops.aten.convolution.default(relu, arg7_1, arg8_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu = arg7_1 = arg8_1 = None
        unsqueeze_8: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg9_1, -1);  arg9_1 = None
        unsqueeze_9: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_8, -1);  unsqueeze_8 = None
        sub_1: "f32[8, 64, 640, 959]" = torch.ops.aten.sub.Tensor(convolution_1, unsqueeze_9);  convolution_1 = unsqueeze_9 = None
        add_2: "f32[64]" = torch.ops.aten.add.Tensor(arg10_1, 1e-05);  arg10_1 = None
        sqrt_1: "f32[64]" = torch.ops.aten.sqrt.default(add_2);  add_2 = None
        reciprocal_1: "f32[64]" = torch.ops.aten.reciprocal.default(sqrt_1);  sqrt_1 = None
        mul_3: "f32[64]" = torch.ops.aten.mul.Tensor(reciprocal_1, 1);  reciprocal_1 = None
        unsqueeze_10: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(mul_3, -1);  mul_3 = None
        unsqueeze_11: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_10, -1);  unsqueeze_10 = None
        mul_4: "f32[8, 64, 640, 959]" = torch.ops.aten.mul.Tensor(sub_1, unsqueeze_11);  sub_1 = unsqueeze_11 = None
        unsqueeze_12: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg11_1, -1);  arg11_1 = None
        unsqueeze_13: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_12, -1);  unsqueeze_12 = None
        mul_5: "f32[8, 64, 640, 959]" = torch.ops.aten.mul.Tensor(mul_4, unsqueeze_13);  mul_4 = unsqueeze_13 = None
        unsqueeze_14: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg12_1, -1);  arg12_1 = None
        unsqueeze_15: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_14, -1);  unsqueeze_14 = None
        add_3: "f32[8, 64, 640, 959]" = torch.ops.aten.add.Tensor(mul_5, unsqueeze_15);  mul_5 = unsqueeze_15 = None
        relu_1: "f32[8, 64, 640, 959]" = torch.ops.aten.relu.default(add_3);  add_3 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pytorch_unet/pytorch_unet/unet/unet_parts.py:38 in forward, code: return self.maxpool_conv(x)
        _low_memory_max_pool_with_offsets = torch.ops.prims._low_memory_max_pool_with_offsets.default(relu_1, [2, 2], [2, 2], [0, 0], [1, 1], False)
        getitem: "f32[8, 64, 320, 479]" = _low_memory_max_pool_with_offsets[0];  _low_memory_max_pool_with_offsets = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pytorch_unet/pytorch_unet/unet/unet_parts.py:25 in forward, code: return self.double_conv(x)
        convolution_2: "f32[8, 128, 320, 479]" = torch.ops.aten.convolution.default(getitem, arg13_1, arg14_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  getitem = arg13_1 = arg14_1 = None
        unsqueeze_16: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg15_1, -1);  arg15_1 = None
        unsqueeze_17: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_16, -1);  unsqueeze_16 = None
        sub_2: "f32[8, 128, 320, 479]" = torch.ops.aten.sub.Tensor(convolution_2, unsqueeze_17);  convolution_2 = unsqueeze_17 = None
        add_4: "f32[128]" = torch.ops.aten.add.Tensor(arg16_1, 1e-05);  arg16_1 = None
        sqrt_2: "f32[128]" = torch.ops.aten.sqrt.default(add_4);  add_4 = None
        reciprocal_2: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_2);  sqrt_2 = None
        mul_6: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_2, 1);  reciprocal_2 = None
        unsqueeze_18: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_6, -1);  mul_6 = None
        unsqueeze_19: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_18, -1);  unsqueeze_18 = None
        mul_7: "f32[8, 128, 320, 479]" = torch.ops.aten.mul.Tensor(sub_2, unsqueeze_19);  sub_2 = unsqueeze_19 = None
        unsqueeze_20: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg17_1, -1);  arg17_1 = None
        unsqueeze_21: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_20, -1);  unsqueeze_20 = None
        mul_8: "f32[8, 128, 320, 479]" = torch.ops.aten.mul.Tensor(mul_7, unsqueeze_21);  mul_7 = unsqueeze_21 = None
        unsqueeze_22: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg18_1, -1);  arg18_1 = None
        unsqueeze_23: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_22, -1);  unsqueeze_22 = None
        add_5: "f32[8, 128, 320, 479]" = torch.ops.aten.add.Tensor(mul_8, unsqueeze_23);  mul_8 = unsqueeze_23 = None
        relu_2: "f32[8, 128, 320, 479]" = torch.ops.aten.relu.default(add_5);  add_5 = None
        convolution_3: "f32[8, 128, 320, 479]" = torch.ops.aten.convolution.default(relu_2, arg19_1, arg20_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_2 = arg19_1 = arg20_1 = None
        unsqueeze_24: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg21_1, -1);  arg21_1 = None
        unsqueeze_25: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_24, -1);  unsqueeze_24 = None
        sub_3: "f32[8, 128, 320, 479]" = torch.ops.aten.sub.Tensor(convolution_3, unsqueeze_25);  convolution_3 = unsqueeze_25 = None
        add_6: "f32[128]" = torch.ops.aten.add.Tensor(arg22_1, 1e-05);  arg22_1 = None
        sqrt_3: "f32[128]" = torch.ops.aten.sqrt.default(add_6);  add_6 = None
        reciprocal_3: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_3);  sqrt_3 = None
        mul_9: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_3, 1);  reciprocal_3 = None
        unsqueeze_26: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_9, -1);  mul_9 = None
        unsqueeze_27: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_26, -1);  unsqueeze_26 = None
        mul_10: "f32[8, 128, 320, 479]" = torch.ops.aten.mul.Tensor(sub_3, unsqueeze_27);  sub_3 = unsqueeze_27 = None
        unsqueeze_28: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg23_1, -1);  arg23_1 = None
        unsqueeze_29: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_28, -1);  unsqueeze_28 = None
        mul_11: "f32[8, 128, 320, 479]" = torch.ops.aten.mul.Tensor(mul_10, unsqueeze_29);  mul_10 = unsqueeze_29 = None
        unsqueeze_30: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg24_1, -1);  arg24_1 = None
        unsqueeze_31: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_30, -1);  unsqueeze_30 = None
        add_7: "f32[8, 128, 320, 479]" = torch.ops.aten.add.Tensor(mul_11, unsqueeze_31);  mul_11 = unsqueeze_31 = None
        relu_3: "f32[8, 128, 320, 479]" = torch.ops.aten.relu.default(add_7);  add_7 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pytorch_unet/pytorch_unet/unet/unet_parts.py:38 in forward, code: return self.maxpool_conv(x)
        _low_memory_max_pool_with_offsets_1 = torch.ops.prims._low_memory_max_pool_with_offsets.default(relu_3, [2, 2], [2, 2], [0, 0], [1, 1], False)
        getitem_2: "f32[8, 128, 160, 239]" = _low_memory_max_pool_with_offsets_1[0];  _low_memory_max_pool_with_offsets_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pytorch_unet/pytorch_unet/unet/unet_parts.py:25 in forward, code: return self.double_conv(x)
        convolution_4: "f32[8, 256, 160, 239]" = torch.ops.aten.convolution.default(getitem_2, arg25_1, arg26_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  getitem_2 = arg25_1 = arg26_1 = None
        unsqueeze_32: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg27_1, -1);  arg27_1 = None
        unsqueeze_33: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_32, -1);  unsqueeze_32 = None
        sub_4: "f32[8, 256, 160, 239]" = torch.ops.aten.sub.Tensor(convolution_4, unsqueeze_33);  convolution_4 = unsqueeze_33 = None
        add_8: "f32[256]" = torch.ops.aten.add.Tensor(arg28_1, 1e-05);  arg28_1 = None
        sqrt_4: "f32[256]" = torch.ops.aten.sqrt.default(add_8);  add_8 = None
        reciprocal_4: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_4);  sqrt_4 = None
        mul_12: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_4, 1);  reciprocal_4 = None
        unsqueeze_34: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_12, -1);  mul_12 = None
        unsqueeze_35: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_34, -1);  unsqueeze_34 = None
        mul_13: "f32[8, 256, 160, 239]" = torch.ops.aten.mul.Tensor(sub_4, unsqueeze_35);  sub_4 = unsqueeze_35 = None
        unsqueeze_36: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg29_1, -1);  arg29_1 = None
        unsqueeze_37: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_36, -1);  unsqueeze_36 = None
        mul_14: "f32[8, 256, 160, 239]" = torch.ops.aten.mul.Tensor(mul_13, unsqueeze_37);  mul_13 = unsqueeze_37 = None
        unsqueeze_38: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg30_1, -1);  arg30_1 = None
        unsqueeze_39: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_38, -1);  unsqueeze_38 = None
        add_9: "f32[8, 256, 160, 239]" = torch.ops.aten.add.Tensor(mul_14, unsqueeze_39);  mul_14 = unsqueeze_39 = None
        relu_4: "f32[8, 256, 160, 239]" = torch.ops.aten.relu.default(add_9);  add_9 = None
        convolution_5: "f32[8, 256, 160, 239]" = torch.ops.aten.convolution.default(relu_4, arg31_1, arg32_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_4 = arg31_1 = arg32_1 = None
        unsqueeze_40: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg33_1, -1);  arg33_1 = None
        unsqueeze_41: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_40, -1);  unsqueeze_40 = None
        sub_5: "f32[8, 256, 160, 239]" = torch.ops.aten.sub.Tensor(convolution_5, unsqueeze_41);  convolution_5 = unsqueeze_41 = None
        add_10: "f32[256]" = torch.ops.aten.add.Tensor(arg34_1, 1e-05);  arg34_1 = None
        sqrt_5: "f32[256]" = torch.ops.aten.sqrt.default(add_10);  add_10 = None
        reciprocal_5: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_5);  sqrt_5 = None
        mul_15: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_5, 1);  reciprocal_5 = None
        unsqueeze_42: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_15, -1);  mul_15 = None
        unsqueeze_43: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_42, -1);  unsqueeze_42 = None
        mul_16: "f32[8, 256, 160, 239]" = torch.ops.aten.mul.Tensor(sub_5, unsqueeze_43);  sub_5 = unsqueeze_43 = None
        unsqueeze_44: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg35_1, -1);  arg35_1 = None
        unsqueeze_45: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_44, -1);  unsqueeze_44 = None
        mul_17: "f32[8, 256, 160, 239]" = torch.ops.aten.mul.Tensor(mul_16, unsqueeze_45);  mul_16 = unsqueeze_45 = None
        unsqueeze_46: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg36_1, -1);  arg36_1 = None
        unsqueeze_47: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_46, -1);  unsqueeze_46 = None
        add_11: "f32[8, 256, 160, 239]" = torch.ops.aten.add.Tensor(mul_17, unsqueeze_47);  mul_17 = unsqueeze_47 = None
        relu_5: "f32[8, 256, 160, 239]" = torch.ops.aten.relu.default(add_11);  add_11 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pytorch_unet/pytorch_unet/unet/unet_parts.py:38 in forward, code: return self.maxpool_conv(x)
        _low_memory_max_pool_with_offsets_2 = torch.ops.prims._low_memory_max_pool_with_offsets.default(relu_5, [2, 2], [2, 2], [0, 0], [1, 1], False)
        getitem_4: "f32[8, 256, 80, 119]" = _low_memory_max_pool_with_offsets_2[0];  _low_memory_max_pool_with_offsets_2 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pytorch_unet/pytorch_unet/unet/unet_parts.py:25 in forward, code: return self.double_conv(x)
        convolution_6: "f32[8, 512, 80, 119]" = torch.ops.aten.convolution.default(getitem_4, arg37_1, arg38_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  getitem_4 = arg37_1 = arg38_1 = None
        unsqueeze_48: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg39_1, -1);  arg39_1 = None
        unsqueeze_49: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_48, -1);  unsqueeze_48 = None
        sub_6: "f32[8, 512, 80, 119]" = torch.ops.aten.sub.Tensor(convolution_6, unsqueeze_49);  convolution_6 = unsqueeze_49 = None
        add_12: "f32[512]" = torch.ops.aten.add.Tensor(arg40_1, 1e-05);  arg40_1 = None
        sqrt_6: "f32[512]" = torch.ops.aten.sqrt.default(add_12);  add_12 = None
        reciprocal_6: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_6);  sqrt_6 = None
        mul_18: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_6, 1);  reciprocal_6 = None
        unsqueeze_50: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_18, -1);  mul_18 = None
        unsqueeze_51: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_50, -1);  unsqueeze_50 = None
        mul_19: "f32[8, 512, 80, 119]" = torch.ops.aten.mul.Tensor(sub_6, unsqueeze_51);  sub_6 = unsqueeze_51 = None
        unsqueeze_52: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg41_1, -1);  arg41_1 = None
        unsqueeze_53: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_52, -1);  unsqueeze_52 = None
        mul_20: "f32[8, 512, 80, 119]" = torch.ops.aten.mul.Tensor(mul_19, unsqueeze_53);  mul_19 = unsqueeze_53 = None
        unsqueeze_54: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg42_1, -1);  arg42_1 = None
        unsqueeze_55: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_54, -1);  unsqueeze_54 = None
        add_13: "f32[8, 512, 80, 119]" = torch.ops.aten.add.Tensor(mul_20, unsqueeze_55);  mul_20 = unsqueeze_55 = None
        relu_6: "f32[8, 512, 80, 119]" = torch.ops.aten.relu.default(add_13);  add_13 = None
        convolution_7: "f32[8, 512, 80, 119]" = torch.ops.aten.convolution.default(relu_6, arg43_1, arg44_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_6 = arg43_1 = arg44_1 = None
        unsqueeze_56: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg45_1, -1);  arg45_1 = None
        unsqueeze_57: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_56, -1);  unsqueeze_56 = None
        sub_7: "f32[8, 512, 80, 119]" = torch.ops.aten.sub.Tensor(convolution_7, unsqueeze_57);  convolution_7 = unsqueeze_57 = None
        add_14: "f32[512]" = torch.ops.aten.add.Tensor(arg46_1, 1e-05);  arg46_1 = None
        sqrt_7: "f32[512]" = torch.ops.aten.sqrt.default(add_14);  add_14 = None
        reciprocal_7: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_7);  sqrt_7 = None
        mul_21: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_7, 1);  reciprocal_7 = None
        unsqueeze_58: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_21, -1);  mul_21 = None
        unsqueeze_59: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_58, -1);  unsqueeze_58 = None
        mul_22: "f32[8, 512, 80, 119]" = torch.ops.aten.mul.Tensor(sub_7, unsqueeze_59);  sub_7 = unsqueeze_59 = None
        unsqueeze_60: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg47_1, -1);  arg47_1 = None
        unsqueeze_61: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_60, -1);  unsqueeze_60 = None
        mul_23: "f32[8, 512, 80, 119]" = torch.ops.aten.mul.Tensor(mul_22, unsqueeze_61);  mul_22 = unsqueeze_61 = None
        unsqueeze_62: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg48_1, -1);  arg48_1 = None
        unsqueeze_63: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_62, -1);  unsqueeze_62 = None
        add_15: "f32[8, 512, 80, 119]" = torch.ops.aten.add.Tensor(mul_23, unsqueeze_63);  mul_23 = unsqueeze_63 = None
        relu_7: "f32[8, 512, 80, 119]" = torch.ops.aten.relu.default(add_15);  add_15 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pytorch_unet/pytorch_unet/unet/unet_parts.py:38 in forward, code: return self.maxpool_conv(x)
        _low_memory_max_pool_with_offsets_3 = torch.ops.prims._low_memory_max_pool_with_offsets.default(relu_7, [2, 2], [2, 2], [0, 0], [1, 1], False)
        getitem_6: "f32[8, 512, 40, 59]" = _low_memory_max_pool_with_offsets_3[0];  _low_memory_max_pool_with_offsets_3 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pytorch_unet/pytorch_unet/unet/unet_parts.py:25 in forward, code: return self.double_conv(x)
        convolution_8: "f32[8, 512, 40, 59]" = torch.ops.aten.convolution.default(getitem_6, arg49_1, arg50_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  getitem_6 = arg49_1 = arg50_1 = None
        unsqueeze_64: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg51_1, -1);  arg51_1 = None
        unsqueeze_65: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_64, -1);  unsqueeze_64 = None
        sub_8: "f32[8, 512, 40, 59]" = torch.ops.aten.sub.Tensor(convolution_8, unsqueeze_65);  convolution_8 = unsqueeze_65 = None
        add_16: "f32[512]" = torch.ops.aten.add.Tensor(arg52_1, 1e-05);  arg52_1 = None
        sqrt_8: "f32[512]" = torch.ops.aten.sqrt.default(add_16);  add_16 = None
        reciprocal_8: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_8);  sqrt_8 = None
        mul_24: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_8, 1);  reciprocal_8 = None
        unsqueeze_66: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_24, -1);  mul_24 = None
        unsqueeze_67: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_66, -1);  unsqueeze_66 = None
        mul_25: "f32[8, 512, 40, 59]" = torch.ops.aten.mul.Tensor(sub_8, unsqueeze_67);  sub_8 = unsqueeze_67 = None
        unsqueeze_68: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg53_1, -1);  arg53_1 = None
        unsqueeze_69: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_68, -1);  unsqueeze_68 = None
        mul_26: "f32[8, 512, 40, 59]" = torch.ops.aten.mul.Tensor(mul_25, unsqueeze_69);  mul_25 = unsqueeze_69 = None
        unsqueeze_70: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg54_1, -1);  arg54_1 = None
        unsqueeze_71: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_70, -1);  unsqueeze_70 = None
        add_17: "f32[8, 512, 40, 59]" = torch.ops.aten.add.Tensor(mul_26, unsqueeze_71);  mul_26 = unsqueeze_71 = None
        relu_8: "f32[8, 512, 40, 59]" = torch.ops.aten.relu.default(add_17);  add_17 = None
        convolution_9: "f32[8, 512, 40, 59]" = torch.ops.aten.convolution.default(relu_8, arg55_1, arg56_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_8 = arg55_1 = arg56_1 = None
        unsqueeze_72: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg57_1, -1);  arg57_1 = None
        unsqueeze_73: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_72, -1);  unsqueeze_72 = None
        sub_9: "f32[8, 512, 40, 59]" = torch.ops.aten.sub.Tensor(convolution_9, unsqueeze_73);  convolution_9 = unsqueeze_73 = None
        add_18: "f32[512]" = torch.ops.aten.add.Tensor(arg58_1, 1e-05);  arg58_1 = None
        sqrt_9: "f32[512]" = torch.ops.aten.sqrt.default(add_18);  add_18 = None
        reciprocal_9: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_9);  sqrt_9 = None
        mul_27: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_9, 1);  reciprocal_9 = None
        unsqueeze_74: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_27, -1);  mul_27 = None
        unsqueeze_75: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_74, -1);  unsqueeze_74 = None
        mul_28: "f32[8, 512, 40, 59]" = torch.ops.aten.mul.Tensor(sub_9, unsqueeze_75);  sub_9 = unsqueeze_75 = None
        unsqueeze_76: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg59_1, -1);  arg59_1 = None
        unsqueeze_77: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_76, -1);  unsqueeze_76 = None
        mul_29: "f32[8, 512, 40, 59]" = torch.ops.aten.mul.Tensor(mul_28, unsqueeze_77);  mul_28 = unsqueeze_77 = None
        unsqueeze_78: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg60_1, -1);  arg60_1 = None
        unsqueeze_79: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_78, -1);  unsqueeze_78 = None
        add_19: "f32[8, 512, 40, 59]" = torch.ops.aten.add.Tensor(mul_29, unsqueeze_79);  mul_29 = unsqueeze_79 = None
        relu_9: "f32[8, 512, 40, 59]" = torch.ops.aten.relu.default(add_19);  add_19 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pytorch_unet/pytorch_unet/unet/unet_parts.py:58 in forward, code: x1 = self.up(x1)
        iota: "i64[80]" = torch.ops.prims.iota.default(80, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        convert_element_type_20: "f32[80]" = torch.ops.prims.convert_element_type.default(iota, torch.float32);  iota = None
        mul_30: "f32[80]" = torch.ops.aten.mul.Tensor(convert_element_type_20, 0.4936708860759494);  convert_element_type_20 = None
        clamp_min: "f32[80]" = torch.ops.aten.clamp_min.default(mul_30, 0.0);  mul_30 = None
        view: "f32[80, 1]" = torch.ops.aten.reshape.default(clamp_min, [80, 1]);  clamp_min = None
        convert_element_type_21: "i64[80, 1]" = torch.ops.prims.convert_element_type.default(view, torch.int64)
        add_20: "i64[80, 1]" = torch.ops.aten.add.Tensor(convert_element_type_21, 1)
        clamp_max: "i64[80, 1]" = torch.ops.aten.clamp_max.default(add_20, 39);  add_20 = None
        iota_1: "i64[118]" = torch.ops.prims.iota.default(118, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        convert_element_type_22: "f32[118]" = torch.ops.prims.convert_element_type.default(iota_1, torch.float32);  iota_1 = None
        mul_31: "f32[118]" = torch.ops.aten.mul.Tensor(convert_element_type_22, 0.49572649572649574);  convert_element_type_22 = None
        clamp_min_1: "f32[118]" = torch.ops.aten.clamp_min.default(mul_31, 0.0);  mul_31 = None
        convert_element_type_23: "i64[118]" = torch.ops.prims.convert_element_type.default(clamp_min_1, torch.int64)
        add_21: "i64[118]" = torch.ops.aten.add.Tensor(convert_element_type_23, 1)
        clamp_max_1: "i64[118]" = torch.ops.aten.clamp_max.default(add_21, 58);  add_21 = None
        _unsafe_index_3: "f32[8, 512, 80, 118]" = torch.ops.aten._unsafe_index.Tensor(relu_9, [None, None, clamp_max, clamp_max_1])
        _unsafe_index_2: "f32[8, 512, 80, 118]" = torch.ops.aten._unsafe_index.Tensor(relu_9, [None, None, clamp_max, convert_element_type_23]);  clamp_max = None
        sub_12: "f32[8, 512, 80, 118]" = torch.ops.aten.sub.Tensor(_unsafe_index_3, _unsafe_index_2);  _unsafe_index_3 = None
        sub_10: "f32[118]" = torch.ops.aten.sub.Tensor(clamp_min_1, convert_element_type_23);  clamp_min_1 = None
        clamp_min_2: "f32[118]" = torch.ops.aten.clamp_min.default(sub_10, 0.0);  sub_10 = None
        clamp_max_2: "f32[118]" = torch.ops.aten.clamp_max.default(clamp_min_2, 1.0);  clamp_min_2 = None
        mul_33: "f32[8, 512, 80, 118]" = torch.ops.aten.mul.Tensor(sub_12, clamp_max_2);  sub_12 = None
        add_23: "f32[8, 512, 80, 118]" = torch.ops.aten.add.Tensor(_unsafe_index_2, mul_33);  _unsafe_index_2 = mul_33 = None
        _unsafe_index_1: "f32[8, 512, 80, 118]" = torch.ops.aten._unsafe_index.Tensor(relu_9, [None, None, convert_element_type_21, clamp_max_1]);  clamp_max_1 = None
        _unsafe_index: "f32[8, 512, 80, 118]" = torch.ops.aten._unsafe_index.Tensor(relu_9, [None, None, convert_element_type_21, convert_element_type_23]);  relu_9 = convert_element_type_23 = None
        sub_11: "f32[8, 512, 80, 118]" = torch.ops.aten.sub.Tensor(_unsafe_index_1, _unsafe_index);  _unsafe_index_1 = None
        mul_32: "f32[8, 512, 80, 118]" = torch.ops.aten.mul.Tensor(sub_11, clamp_max_2);  sub_11 = clamp_max_2 = None
        add_22: "f32[8, 512, 80, 118]" = torch.ops.aten.add.Tensor(_unsafe_index, mul_32);  _unsafe_index = mul_32 = None
        sub_14: "f32[8, 512, 80, 118]" = torch.ops.aten.sub.Tensor(add_23, add_22);  add_23 = None
        sub_13: "f32[80, 1]" = torch.ops.aten.sub.Tensor(view, convert_element_type_21);  view = convert_element_type_21 = None
        clamp_min_3: "f32[80, 1]" = torch.ops.aten.clamp_min.default(sub_13, 0.0);  sub_13 = None
        clamp_max_3: "f32[80, 1]" = torch.ops.aten.clamp_max.default(clamp_min_3, 1.0);  clamp_min_3 = None
        mul_34: "f32[8, 512, 80, 118]" = torch.ops.aten.mul.Tensor(sub_14, clamp_max_3);  sub_14 = clamp_max_3 = None
        add_24: "f32[8, 512, 80, 118]" = torch.ops.aten.add.Tensor(add_22, mul_34);  add_22 = mul_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:5462 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd: "f32[8, 512, 80, 119]" = torch.ops.aten.constant_pad_nd.default(add_24, [0, 1, 0, 0], 0.0);  add_24 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pytorch_unet/pytorch_unet/unet/unet_parts.py:67 in forward, code: x = torch.cat([x2, x1], dim=1)
        cat: "f32[8, 1024, 80, 119]" = torch.ops.aten.cat.default([relu_7, constant_pad_nd], 1);  relu_7 = constant_pad_nd = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pytorch_unet/pytorch_unet/unet/unet_parts.py:25 in forward, code: return self.double_conv(x)
        convolution_10: "f32[8, 512, 80, 119]" = torch.ops.aten.convolution.default(cat, arg61_1, arg62_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  cat = arg61_1 = arg62_1 = None
        unsqueeze_80: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg63_1, -1);  arg63_1 = None
        unsqueeze_81: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_80, -1);  unsqueeze_80 = None
        sub_15: "f32[8, 512, 80, 119]" = torch.ops.aten.sub.Tensor(convolution_10, unsqueeze_81);  convolution_10 = unsqueeze_81 = None
        add_25: "f32[512]" = torch.ops.aten.add.Tensor(arg64_1, 1e-05);  arg64_1 = None
        sqrt_10: "f32[512]" = torch.ops.aten.sqrt.default(add_25);  add_25 = None
        reciprocal_10: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_10);  sqrt_10 = None
        mul_35: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_10, 1);  reciprocal_10 = None
        unsqueeze_82: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_35, -1);  mul_35 = None
        unsqueeze_83: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_82, -1);  unsqueeze_82 = None
        mul_36: "f32[8, 512, 80, 119]" = torch.ops.aten.mul.Tensor(sub_15, unsqueeze_83);  sub_15 = unsqueeze_83 = None
        unsqueeze_84: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg65_1, -1);  arg65_1 = None
        unsqueeze_85: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_84, -1);  unsqueeze_84 = None
        mul_37: "f32[8, 512, 80, 119]" = torch.ops.aten.mul.Tensor(mul_36, unsqueeze_85);  mul_36 = unsqueeze_85 = None
        unsqueeze_86: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg66_1, -1);  arg66_1 = None
        unsqueeze_87: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_86, -1);  unsqueeze_86 = None
        add_26: "f32[8, 512, 80, 119]" = torch.ops.aten.add.Tensor(mul_37, unsqueeze_87);  mul_37 = unsqueeze_87 = None
        relu_10: "f32[8, 512, 80, 119]" = torch.ops.aten.relu.default(add_26);  add_26 = None
        convolution_11: "f32[8, 256, 80, 119]" = torch.ops.aten.convolution.default(relu_10, arg67_1, arg68_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_10 = arg67_1 = arg68_1 = None
        unsqueeze_88: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg69_1, -1);  arg69_1 = None
        unsqueeze_89: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_88, -1);  unsqueeze_88 = None
        sub_16: "f32[8, 256, 80, 119]" = torch.ops.aten.sub.Tensor(convolution_11, unsqueeze_89);  convolution_11 = unsqueeze_89 = None
        add_27: "f32[256]" = torch.ops.aten.add.Tensor(arg70_1, 1e-05);  arg70_1 = None
        sqrt_11: "f32[256]" = torch.ops.aten.sqrt.default(add_27);  add_27 = None
        reciprocal_11: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_11);  sqrt_11 = None
        mul_38: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_11, 1);  reciprocal_11 = None
        unsqueeze_90: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_38, -1);  mul_38 = None
        unsqueeze_91: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_90, -1);  unsqueeze_90 = None
        mul_39: "f32[8, 256, 80, 119]" = torch.ops.aten.mul.Tensor(sub_16, unsqueeze_91);  sub_16 = unsqueeze_91 = None
        unsqueeze_92: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg71_1, -1);  arg71_1 = None
        unsqueeze_93: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_92, -1);  unsqueeze_92 = None
        mul_40: "f32[8, 256, 80, 119]" = torch.ops.aten.mul.Tensor(mul_39, unsqueeze_93);  mul_39 = unsqueeze_93 = None
        unsqueeze_94: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg72_1, -1);  arg72_1 = None
        unsqueeze_95: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_94, -1);  unsqueeze_94 = None
        add_28: "f32[8, 256, 80, 119]" = torch.ops.aten.add.Tensor(mul_40, unsqueeze_95);  mul_40 = unsqueeze_95 = None
        relu_11: "f32[8, 256, 80, 119]" = torch.ops.aten.relu.default(add_28);  add_28 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pytorch_unet/pytorch_unet/unet/unet_parts.py:58 in forward, code: x1 = self.up(x1)
        iota_2: "i64[160]" = torch.ops.prims.iota.default(160, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        convert_element_type_28: "f32[160]" = torch.ops.prims.convert_element_type.default(iota_2, torch.float32);  iota_2 = None
        mul_41: "f32[160]" = torch.ops.aten.mul.Tensor(convert_element_type_28, 0.4968553459119497);  convert_element_type_28 = None
        clamp_min_4: "f32[160]" = torch.ops.aten.clamp_min.default(mul_41, 0.0);  mul_41 = None
        view_2: "f32[160, 1]" = torch.ops.aten.reshape.default(clamp_min_4, [160, 1]);  clamp_min_4 = None
        convert_element_type_29: "i64[160, 1]" = torch.ops.prims.convert_element_type.default(view_2, torch.int64)
        add_29: "i64[160, 1]" = torch.ops.aten.add.Tensor(convert_element_type_29, 1)
        clamp_max_4: "i64[160, 1]" = torch.ops.aten.clamp_max.default(add_29, 79);  add_29 = None
        iota_3: "i64[238]" = torch.ops.prims.iota.default(238, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        convert_element_type_30: "f32[238]" = torch.ops.prims.convert_element_type.default(iota_3, torch.float32);  iota_3 = None
        mul_42: "f32[238]" = torch.ops.aten.mul.Tensor(convert_element_type_30, 0.4978902953586498);  convert_element_type_30 = None
        clamp_min_5: "f32[238]" = torch.ops.aten.clamp_min.default(mul_42, 0.0);  mul_42 = None
        convert_element_type_31: "i64[238]" = torch.ops.prims.convert_element_type.default(clamp_min_5, torch.int64)
        add_30: "i64[238]" = torch.ops.aten.add.Tensor(convert_element_type_31, 1)
        clamp_max_5: "i64[238]" = torch.ops.aten.clamp_max.default(add_30, 118);  add_30 = None
        _unsafe_index_7: "f32[8, 256, 160, 238]" = torch.ops.aten._unsafe_index.Tensor(relu_11, [None, None, clamp_max_4, clamp_max_5])
        _unsafe_index_6: "f32[8, 256, 160, 238]" = torch.ops.aten._unsafe_index.Tensor(relu_11, [None, None, clamp_max_4, convert_element_type_31]);  clamp_max_4 = None
        sub_19: "f32[8, 256, 160, 238]" = torch.ops.aten.sub.Tensor(_unsafe_index_7, _unsafe_index_6);  _unsafe_index_7 = None
        sub_17: "f32[238]" = torch.ops.aten.sub.Tensor(clamp_min_5, convert_element_type_31);  clamp_min_5 = None
        clamp_min_6: "f32[238]" = torch.ops.aten.clamp_min.default(sub_17, 0.0);  sub_17 = None
        clamp_max_6: "f32[238]" = torch.ops.aten.clamp_max.default(clamp_min_6, 1.0);  clamp_min_6 = None
        mul_44: "f32[8, 256, 160, 238]" = torch.ops.aten.mul.Tensor(sub_19, clamp_max_6);  sub_19 = None
        add_32: "f32[8, 256, 160, 238]" = torch.ops.aten.add.Tensor(_unsafe_index_6, mul_44);  _unsafe_index_6 = mul_44 = None
        _unsafe_index_5: "f32[8, 256, 160, 238]" = torch.ops.aten._unsafe_index.Tensor(relu_11, [None, None, convert_element_type_29, clamp_max_5]);  clamp_max_5 = None
        _unsafe_index_4: "f32[8, 256, 160, 238]" = torch.ops.aten._unsafe_index.Tensor(relu_11, [None, None, convert_element_type_29, convert_element_type_31]);  relu_11 = convert_element_type_31 = None
        sub_18: "f32[8, 256, 160, 238]" = torch.ops.aten.sub.Tensor(_unsafe_index_5, _unsafe_index_4);  _unsafe_index_5 = None
        mul_43: "f32[8, 256, 160, 238]" = torch.ops.aten.mul.Tensor(sub_18, clamp_max_6);  sub_18 = clamp_max_6 = None
        add_31: "f32[8, 256, 160, 238]" = torch.ops.aten.add.Tensor(_unsafe_index_4, mul_43);  _unsafe_index_4 = mul_43 = None
        sub_21: "f32[8, 256, 160, 238]" = torch.ops.aten.sub.Tensor(add_32, add_31);  add_32 = None
        sub_20: "f32[160, 1]" = torch.ops.aten.sub.Tensor(view_2, convert_element_type_29);  view_2 = convert_element_type_29 = None
        clamp_min_7: "f32[160, 1]" = torch.ops.aten.clamp_min.default(sub_20, 0.0);  sub_20 = None
        clamp_max_7: "f32[160, 1]" = torch.ops.aten.clamp_max.default(clamp_min_7, 1.0);  clamp_min_7 = None
        mul_45: "f32[8, 256, 160, 238]" = torch.ops.aten.mul.Tensor(sub_21, clamp_max_7);  sub_21 = clamp_max_7 = None
        add_33: "f32[8, 256, 160, 238]" = torch.ops.aten.add.Tensor(add_31, mul_45);  add_31 = mul_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:5462 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_1: "f32[8, 256, 160, 239]" = torch.ops.aten.constant_pad_nd.default(add_33, [0, 1, 0, 0], 0.0);  add_33 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pytorch_unet/pytorch_unet/unet/unet_parts.py:67 in forward, code: x = torch.cat([x2, x1], dim=1)
        cat_1: "f32[8, 512, 160, 239]" = torch.ops.aten.cat.default([relu_5, constant_pad_nd_1], 1);  relu_5 = constant_pad_nd_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pytorch_unet/pytorch_unet/unet/unet_parts.py:25 in forward, code: return self.double_conv(x)
        convolution_12: "f32[8, 256, 160, 239]" = torch.ops.aten.convolution.default(cat_1, arg73_1, arg74_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  cat_1 = arg73_1 = arg74_1 = None
        unsqueeze_96: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg75_1, -1);  arg75_1 = None
        unsqueeze_97: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_96, -1);  unsqueeze_96 = None
        sub_22: "f32[8, 256, 160, 239]" = torch.ops.aten.sub.Tensor(convolution_12, unsqueeze_97);  convolution_12 = unsqueeze_97 = None
        add_34: "f32[256]" = torch.ops.aten.add.Tensor(arg76_1, 1e-05);  arg76_1 = None
        sqrt_12: "f32[256]" = torch.ops.aten.sqrt.default(add_34);  add_34 = None
        reciprocal_12: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_12);  sqrt_12 = None
        mul_46: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_12, 1);  reciprocal_12 = None
        unsqueeze_98: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_46, -1);  mul_46 = None
        unsqueeze_99: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_98, -1);  unsqueeze_98 = None
        mul_47: "f32[8, 256, 160, 239]" = torch.ops.aten.mul.Tensor(sub_22, unsqueeze_99);  sub_22 = unsqueeze_99 = None
        unsqueeze_100: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg77_1, -1);  arg77_1 = None
        unsqueeze_101: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_100, -1);  unsqueeze_100 = None
        mul_48: "f32[8, 256, 160, 239]" = torch.ops.aten.mul.Tensor(mul_47, unsqueeze_101);  mul_47 = unsqueeze_101 = None
        unsqueeze_102: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg78_1, -1);  arg78_1 = None
        unsqueeze_103: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_102, -1);  unsqueeze_102 = None
        add_35: "f32[8, 256, 160, 239]" = torch.ops.aten.add.Tensor(mul_48, unsqueeze_103);  mul_48 = unsqueeze_103 = None
        relu_12: "f32[8, 256, 160, 239]" = torch.ops.aten.relu.default(add_35);  add_35 = None
        convolution_13: "f32[8, 128, 160, 239]" = torch.ops.aten.convolution.default(relu_12, arg79_1, arg80_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_12 = arg79_1 = arg80_1 = None
        unsqueeze_104: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg81_1, -1);  arg81_1 = None
        unsqueeze_105: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_104, -1);  unsqueeze_104 = None
        sub_23: "f32[8, 128, 160, 239]" = torch.ops.aten.sub.Tensor(convolution_13, unsqueeze_105);  convolution_13 = unsqueeze_105 = None
        add_36: "f32[128]" = torch.ops.aten.add.Tensor(arg82_1, 1e-05);  arg82_1 = None
        sqrt_13: "f32[128]" = torch.ops.aten.sqrt.default(add_36);  add_36 = None
        reciprocal_13: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_13);  sqrt_13 = None
        mul_49: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_13, 1);  reciprocal_13 = None
        unsqueeze_106: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_49, -1);  mul_49 = None
        unsqueeze_107: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_106, -1);  unsqueeze_106 = None
        mul_50: "f32[8, 128, 160, 239]" = torch.ops.aten.mul.Tensor(sub_23, unsqueeze_107);  sub_23 = unsqueeze_107 = None
        unsqueeze_108: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg83_1, -1);  arg83_1 = None
        unsqueeze_109: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_108, -1);  unsqueeze_108 = None
        mul_51: "f32[8, 128, 160, 239]" = torch.ops.aten.mul.Tensor(mul_50, unsqueeze_109);  mul_50 = unsqueeze_109 = None
        unsqueeze_110: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg84_1, -1);  arg84_1 = None
        unsqueeze_111: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_110, -1);  unsqueeze_110 = None
        add_37: "f32[8, 128, 160, 239]" = torch.ops.aten.add.Tensor(mul_51, unsqueeze_111);  mul_51 = unsqueeze_111 = None
        relu_13: "f32[8, 128, 160, 239]" = torch.ops.aten.relu.default(add_37);  add_37 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pytorch_unet/pytorch_unet/unet/unet_parts.py:58 in forward, code: x1 = self.up(x1)
        iota_4: "i64[320]" = torch.ops.prims.iota.default(320, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        convert_element_type_36: "f32[320]" = torch.ops.prims.convert_element_type.default(iota_4, torch.float32);  iota_4 = None
        mul_52: "f32[320]" = torch.ops.aten.mul.Tensor(convert_element_type_36, 0.49843260188087773);  convert_element_type_36 = None
        clamp_min_8: "f32[320]" = torch.ops.aten.clamp_min.default(mul_52, 0.0);  mul_52 = None
        view_4: "f32[320, 1]" = torch.ops.aten.reshape.default(clamp_min_8, [320, 1]);  clamp_min_8 = None
        convert_element_type_37: "i64[320, 1]" = torch.ops.prims.convert_element_type.default(view_4, torch.int64)
        add_38: "i64[320, 1]" = torch.ops.aten.add.Tensor(convert_element_type_37, 1)
        clamp_max_8: "i64[320, 1]" = torch.ops.aten.clamp_max.default(add_38, 159);  add_38 = None
        iota_5: "i64[478]" = torch.ops.prims.iota.default(478, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        convert_element_type_38: "f32[478]" = torch.ops.prims.convert_element_type.default(iota_5, torch.float32);  iota_5 = None
        mul_53: "f32[478]" = torch.ops.aten.mul.Tensor(convert_element_type_38, 0.4989517819706499);  convert_element_type_38 = None
        clamp_min_9: "f32[478]" = torch.ops.aten.clamp_min.default(mul_53, 0.0);  mul_53 = None
        convert_element_type_39: "i64[478]" = torch.ops.prims.convert_element_type.default(clamp_min_9, torch.int64)
        add_39: "i64[478]" = torch.ops.aten.add.Tensor(convert_element_type_39, 1)
        clamp_max_9: "i64[478]" = torch.ops.aten.clamp_max.default(add_39, 238);  add_39 = None
        _unsafe_index_11: "f32[8, 128, 320, 478]" = torch.ops.aten._unsafe_index.Tensor(relu_13, [None, None, clamp_max_8, clamp_max_9])
        _unsafe_index_10: "f32[8, 128, 320, 478]" = torch.ops.aten._unsafe_index.Tensor(relu_13, [None, None, clamp_max_8, convert_element_type_39]);  clamp_max_8 = None
        sub_26: "f32[8, 128, 320, 478]" = torch.ops.aten.sub.Tensor(_unsafe_index_11, _unsafe_index_10);  _unsafe_index_11 = None
        sub_24: "f32[478]" = torch.ops.aten.sub.Tensor(clamp_min_9, convert_element_type_39);  clamp_min_9 = None
        clamp_min_10: "f32[478]" = torch.ops.aten.clamp_min.default(sub_24, 0.0);  sub_24 = None
        clamp_max_10: "f32[478]" = torch.ops.aten.clamp_max.default(clamp_min_10, 1.0);  clamp_min_10 = None
        mul_55: "f32[8, 128, 320, 478]" = torch.ops.aten.mul.Tensor(sub_26, clamp_max_10);  sub_26 = None
        add_41: "f32[8, 128, 320, 478]" = torch.ops.aten.add.Tensor(_unsafe_index_10, mul_55);  _unsafe_index_10 = mul_55 = None
        _unsafe_index_9: "f32[8, 128, 320, 478]" = torch.ops.aten._unsafe_index.Tensor(relu_13, [None, None, convert_element_type_37, clamp_max_9]);  clamp_max_9 = None
        _unsafe_index_8: "f32[8, 128, 320, 478]" = torch.ops.aten._unsafe_index.Tensor(relu_13, [None, None, convert_element_type_37, convert_element_type_39]);  relu_13 = convert_element_type_39 = None
        sub_25: "f32[8, 128, 320, 478]" = torch.ops.aten.sub.Tensor(_unsafe_index_9, _unsafe_index_8);  _unsafe_index_9 = None
        mul_54: "f32[8, 128, 320, 478]" = torch.ops.aten.mul.Tensor(sub_25, clamp_max_10);  sub_25 = clamp_max_10 = None
        add_40: "f32[8, 128, 320, 478]" = torch.ops.aten.add.Tensor(_unsafe_index_8, mul_54);  _unsafe_index_8 = mul_54 = None
        sub_28: "f32[8, 128, 320, 478]" = torch.ops.aten.sub.Tensor(add_41, add_40);  add_41 = None
        sub_27: "f32[320, 1]" = torch.ops.aten.sub.Tensor(view_4, convert_element_type_37);  view_4 = convert_element_type_37 = None
        clamp_min_11: "f32[320, 1]" = torch.ops.aten.clamp_min.default(sub_27, 0.0);  sub_27 = None
        clamp_max_11: "f32[320, 1]" = torch.ops.aten.clamp_max.default(clamp_min_11, 1.0);  clamp_min_11 = None
        mul_56: "f32[8, 128, 320, 478]" = torch.ops.aten.mul.Tensor(sub_28, clamp_max_11);  sub_28 = clamp_max_11 = None
        add_42: "f32[8, 128, 320, 478]" = torch.ops.aten.add.Tensor(add_40, mul_56);  add_40 = mul_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:5462 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_2: "f32[8, 128, 320, 479]" = torch.ops.aten.constant_pad_nd.default(add_42, [0, 1, 0, 0], 0.0);  add_42 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pytorch_unet/pytorch_unet/unet/unet_parts.py:67 in forward, code: x = torch.cat([x2, x1], dim=1)
        cat_2: "f32[8, 256, 320, 479]" = torch.ops.aten.cat.default([relu_3, constant_pad_nd_2], 1);  relu_3 = constant_pad_nd_2 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pytorch_unet/pytorch_unet/unet/unet_parts.py:25 in forward, code: return self.double_conv(x)
        convolution_14: "f32[8, 128, 320, 479]" = torch.ops.aten.convolution.default(cat_2, arg85_1, arg86_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  cat_2 = arg85_1 = arg86_1 = None
        unsqueeze_112: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg87_1, -1);  arg87_1 = None
        unsqueeze_113: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_112, -1);  unsqueeze_112 = None
        sub_29: "f32[8, 128, 320, 479]" = torch.ops.aten.sub.Tensor(convolution_14, unsqueeze_113);  convolution_14 = unsqueeze_113 = None
        add_43: "f32[128]" = torch.ops.aten.add.Tensor(arg88_1, 1e-05);  arg88_1 = None
        sqrt_14: "f32[128]" = torch.ops.aten.sqrt.default(add_43);  add_43 = None
        reciprocal_14: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_14);  sqrt_14 = None
        mul_57: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_14, 1);  reciprocal_14 = None
        unsqueeze_114: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_57, -1);  mul_57 = None
        unsqueeze_115: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_114, -1);  unsqueeze_114 = None
        mul_58: "f32[8, 128, 320, 479]" = torch.ops.aten.mul.Tensor(sub_29, unsqueeze_115);  sub_29 = unsqueeze_115 = None
        unsqueeze_116: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg89_1, -1);  arg89_1 = None
        unsqueeze_117: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_116, -1);  unsqueeze_116 = None
        mul_59: "f32[8, 128, 320, 479]" = torch.ops.aten.mul.Tensor(mul_58, unsqueeze_117);  mul_58 = unsqueeze_117 = None
        unsqueeze_118: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg90_1, -1);  arg90_1 = None
        unsqueeze_119: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_118, -1);  unsqueeze_118 = None
        add_44: "f32[8, 128, 320, 479]" = torch.ops.aten.add.Tensor(mul_59, unsqueeze_119);  mul_59 = unsqueeze_119 = None
        relu_14: "f32[8, 128, 320, 479]" = torch.ops.aten.relu.default(add_44);  add_44 = None
        convolution_15: "f32[8, 64, 320, 479]" = torch.ops.aten.convolution.default(relu_14, arg91_1, arg92_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_14 = arg91_1 = arg92_1 = None
        unsqueeze_120: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg93_1, -1);  arg93_1 = None
        unsqueeze_121: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_120, -1);  unsqueeze_120 = None
        sub_30: "f32[8, 64, 320, 479]" = torch.ops.aten.sub.Tensor(convolution_15, unsqueeze_121);  convolution_15 = unsqueeze_121 = None
        add_45: "f32[64]" = torch.ops.aten.add.Tensor(arg94_1, 1e-05);  arg94_1 = None
        sqrt_15: "f32[64]" = torch.ops.aten.sqrt.default(add_45);  add_45 = None
        reciprocal_15: "f32[64]" = torch.ops.aten.reciprocal.default(sqrt_15);  sqrt_15 = None
        mul_60: "f32[64]" = torch.ops.aten.mul.Tensor(reciprocal_15, 1);  reciprocal_15 = None
        unsqueeze_122: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(mul_60, -1);  mul_60 = None
        unsqueeze_123: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_122, -1);  unsqueeze_122 = None
        mul_61: "f32[8, 64, 320, 479]" = torch.ops.aten.mul.Tensor(sub_30, unsqueeze_123);  sub_30 = unsqueeze_123 = None
        unsqueeze_124: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg95_1, -1);  arg95_1 = None
        unsqueeze_125: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_124, -1);  unsqueeze_124 = None
        mul_62: "f32[8, 64, 320, 479]" = torch.ops.aten.mul.Tensor(mul_61, unsqueeze_125);  mul_61 = unsqueeze_125 = None
        unsqueeze_126: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg96_1, -1);  arg96_1 = None
        unsqueeze_127: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_126, -1);  unsqueeze_126 = None
        add_46: "f32[8, 64, 320, 479]" = torch.ops.aten.add.Tensor(mul_62, unsqueeze_127);  mul_62 = unsqueeze_127 = None
        relu_15: "f32[8, 64, 320, 479]" = torch.ops.aten.relu.default(add_46);  add_46 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pytorch_unet/pytorch_unet/unet/unet_parts.py:58 in forward, code: x1 = self.up(x1)
        iota_6: "i64[640]" = torch.ops.prims.iota.default(640, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        convert_element_type_44: "f32[640]" = torch.ops.prims.convert_element_type.default(iota_6, torch.float32);  iota_6 = None
        mul_63: "f32[640]" = torch.ops.aten.mul.Tensor(convert_element_type_44, 0.49921752738654146);  convert_element_type_44 = None
        clamp_min_12: "f32[640]" = torch.ops.aten.clamp_min.default(mul_63, 0.0);  mul_63 = None
        view_6: "f32[640, 1]" = torch.ops.aten.reshape.default(clamp_min_12, [640, 1]);  clamp_min_12 = None
        convert_element_type_45: "i64[640, 1]" = torch.ops.prims.convert_element_type.default(view_6, torch.int64)
        add_47: "i64[640, 1]" = torch.ops.aten.add.Tensor(convert_element_type_45, 1)
        clamp_max_12: "i64[640, 1]" = torch.ops.aten.clamp_max.default(add_47, 319);  add_47 = None
        iota_7: "i64[958]" = torch.ops.prims.iota.default(958, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        convert_element_type_46: "f32[958]" = torch.ops.prims.convert_element_type.default(iota_7, torch.float32);  iota_7 = None
        mul_64: "f32[958]" = torch.ops.aten.mul.Tensor(convert_element_type_46, 0.4994775339602926);  convert_element_type_46 = None
        clamp_min_13: "f32[958]" = torch.ops.aten.clamp_min.default(mul_64, 0.0);  mul_64 = None
        convert_element_type_47: "i64[958]" = torch.ops.prims.convert_element_type.default(clamp_min_13, torch.int64)
        add_48: "i64[958]" = torch.ops.aten.add.Tensor(convert_element_type_47, 1)
        clamp_max_13: "i64[958]" = torch.ops.aten.clamp_max.default(add_48, 478);  add_48 = None
        _unsafe_index_15: "f32[8, 64, 640, 958]" = torch.ops.aten._unsafe_index.Tensor(relu_15, [None, None, clamp_max_12, clamp_max_13])
        _unsafe_index_14: "f32[8, 64, 640, 958]" = torch.ops.aten._unsafe_index.Tensor(relu_15, [None, None, clamp_max_12, convert_element_type_47]);  clamp_max_12 = None
        sub_33: "f32[8, 64, 640, 958]" = torch.ops.aten.sub.Tensor(_unsafe_index_15, _unsafe_index_14);  _unsafe_index_15 = None
        sub_31: "f32[958]" = torch.ops.aten.sub.Tensor(clamp_min_13, convert_element_type_47);  clamp_min_13 = None
        clamp_min_14: "f32[958]" = torch.ops.aten.clamp_min.default(sub_31, 0.0);  sub_31 = None
        clamp_max_14: "f32[958]" = torch.ops.aten.clamp_max.default(clamp_min_14, 1.0);  clamp_min_14 = None
        mul_66: "f32[8, 64, 640, 958]" = torch.ops.aten.mul.Tensor(sub_33, clamp_max_14);  sub_33 = None
        add_50: "f32[8, 64, 640, 958]" = torch.ops.aten.add.Tensor(_unsafe_index_14, mul_66);  _unsafe_index_14 = mul_66 = None
        _unsafe_index_13: "f32[8, 64, 640, 958]" = torch.ops.aten._unsafe_index.Tensor(relu_15, [None, None, convert_element_type_45, clamp_max_13]);  clamp_max_13 = None
        _unsafe_index_12: "f32[8, 64, 640, 958]" = torch.ops.aten._unsafe_index.Tensor(relu_15, [None, None, convert_element_type_45, convert_element_type_47]);  relu_15 = convert_element_type_47 = None
        sub_32: "f32[8, 64, 640, 958]" = torch.ops.aten.sub.Tensor(_unsafe_index_13, _unsafe_index_12);  _unsafe_index_13 = None
        mul_65: "f32[8, 64, 640, 958]" = torch.ops.aten.mul.Tensor(sub_32, clamp_max_14);  sub_32 = clamp_max_14 = None
        add_49: "f32[8, 64, 640, 958]" = torch.ops.aten.add.Tensor(_unsafe_index_12, mul_65);  _unsafe_index_12 = mul_65 = None
        sub_35: "f32[8, 64, 640, 958]" = torch.ops.aten.sub.Tensor(add_50, add_49);  add_50 = None
        sub_34: "f32[640, 1]" = torch.ops.aten.sub.Tensor(view_6, convert_element_type_45);  view_6 = convert_element_type_45 = None
        clamp_min_15: "f32[640, 1]" = torch.ops.aten.clamp_min.default(sub_34, 0.0);  sub_34 = None
        clamp_max_15: "f32[640, 1]" = torch.ops.aten.clamp_max.default(clamp_min_15, 1.0);  clamp_min_15 = None
        mul_67: "f32[8, 64, 640, 958]" = torch.ops.aten.mul.Tensor(sub_35, clamp_max_15);  sub_35 = clamp_max_15 = None
        add_51: "f32[8, 64, 640, 958]" = torch.ops.aten.add.Tensor(add_49, mul_67);  add_49 = mul_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:5462 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_3: "f32[8, 64, 640, 959]" = torch.ops.aten.constant_pad_nd.default(add_51, [0, 1, 0, 0], 0.0);  add_51 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pytorch_unet/pytorch_unet/unet/unet_parts.py:67 in forward, code: x = torch.cat([x2, x1], dim=1)
        cat_3: "f32[8, 128, 640, 959]" = torch.ops.aten.cat.default([relu_1, constant_pad_nd_3], 1);  relu_1 = constant_pad_nd_3 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pytorch_unet/pytorch_unet/unet/unet_parts.py:25 in forward, code: return self.double_conv(x)
        convolution_16: "f32[8, 64, 640, 959]" = torch.ops.aten.convolution.default(cat_3, arg97_1, arg98_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  cat_3 = arg97_1 = arg98_1 = None
        unsqueeze_128: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg99_1, -1);  arg99_1 = None
        unsqueeze_129: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_128, -1);  unsqueeze_128 = None
        sub_36: "f32[8, 64, 640, 959]" = torch.ops.aten.sub.Tensor(convolution_16, unsqueeze_129);  convolution_16 = unsqueeze_129 = None
        add_52: "f32[64]" = torch.ops.aten.add.Tensor(arg100_1, 1e-05);  arg100_1 = None
        sqrt_16: "f32[64]" = torch.ops.aten.sqrt.default(add_52);  add_52 = None
        reciprocal_16: "f32[64]" = torch.ops.aten.reciprocal.default(sqrt_16);  sqrt_16 = None
        mul_68: "f32[64]" = torch.ops.aten.mul.Tensor(reciprocal_16, 1);  reciprocal_16 = None
        unsqueeze_130: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(mul_68, -1);  mul_68 = None
        unsqueeze_131: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_130, -1);  unsqueeze_130 = None
        mul_69: "f32[8, 64, 640, 959]" = torch.ops.aten.mul.Tensor(sub_36, unsqueeze_131);  sub_36 = unsqueeze_131 = None
        unsqueeze_132: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg101_1, -1);  arg101_1 = None
        unsqueeze_133: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_132, -1);  unsqueeze_132 = None
        mul_70: "f32[8, 64, 640, 959]" = torch.ops.aten.mul.Tensor(mul_69, unsqueeze_133);  mul_69 = unsqueeze_133 = None
        unsqueeze_134: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg102_1, -1);  arg102_1 = None
        unsqueeze_135: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_134, -1);  unsqueeze_134 = None
        add_53: "f32[8, 64, 640, 959]" = torch.ops.aten.add.Tensor(mul_70, unsqueeze_135);  mul_70 = unsqueeze_135 = None
        relu_16: "f32[8, 64, 640, 959]" = torch.ops.aten.relu.default(add_53);  add_53 = None
        convolution_17: "f32[8, 64, 640, 959]" = torch.ops.aten.convolution.default(relu_16, arg103_1, arg104_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_16 = arg103_1 = arg104_1 = None
        unsqueeze_136: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg105_1, -1);  arg105_1 = None
        unsqueeze_137: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_136, -1);  unsqueeze_136 = None
        sub_37: "f32[8, 64, 640, 959]" = torch.ops.aten.sub.Tensor(convolution_17, unsqueeze_137);  convolution_17 = unsqueeze_137 = None
        add_54: "f32[64]" = torch.ops.aten.add.Tensor(arg106_1, 1e-05);  arg106_1 = None
        sqrt_17: "f32[64]" = torch.ops.aten.sqrt.default(add_54);  add_54 = None
        reciprocal_17: "f32[64]" = torch.ops.aten.reciprocal.default(sqrt_17);  sqrt_17 = None
        mul_71: "f32[64]" = torch.ops.aten.mul.Tensor(reciprocal_17, 1);  reciprocal_17 = None
        unsqueeze_138: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(mul_71, -1);  mul_71 = None
        unsqueeze_139: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_138, -1);  unsqueeze_138 = None
        mul_72: "f32[8, 64, 640, 959]" = torch.ops.aten.mul.Tensor(sub_37, unsqueeze_139);  sub_37 = unsqueeze_139 = None
        unsqueeze_140: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg107_1, -1);  arg107_1 = None
        unsqueeze_141: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_140, -1);  unsqueeze_140 = None
        mul_73: "f32[8, 64, 640, 959]" = torch.ops.aten.mul.Tensor(mul_72, unsqueeze_141);  mul_72 = unsqueeze_141 = None
        unsqueeze_142: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg108_1, -1);  arg108_1 = None
        unsqueeze_143: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_142, -1);  unsqueeze_142 = None
        add_55: "f32[8, 64, 640, 959]" = torch.ops.aten.add.Tensor(mul_73, unsqueeze_143);  mul_73 = unsqueeze_143 = None
        relu_17: "f32[8, 64, 640, 959]" = torch.ops.aten.relu.default(add_55);  add_55 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pytorch_unet/pytorch_unet/unet/unet_parts.py:77 in forward, code: return self.conv(x)
        convolution_18: "f32[8, 2, 640, 959]" = torch.ops.aten.convolution.default(relu_17, arg109_1, arg110_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_17 = arg109_1 = arg110_1 = None
        return (convolution_18,)
