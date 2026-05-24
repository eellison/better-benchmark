import torch
import torch._inductor.inductor_prims  # registers inductor prims ops
from torch import device
from math import inf, nan

class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "f32[64, 3, 3, 3]", arg1_1: "f32[64]", arg2_1: "f32[1, 3, 32, 128]", arg3_1: "f32[64]", arg4_1: "f32[64]", arg5_1: "f32[64]", arg6_1: "f32[64]", arg7_1: "f32[64, 64, 3, 3]", arg8_1: "f32[64]", arg9_1: "f32[64]", arg10_1: "f32[64]", arg11_1: "f32[64]", arg12_1: "f32[64]", arg13_1: "f32[128, 64, 3, 3]", arg14_1: "f32[128]", arg15_1: "f32[128]", arg16_1: "f32[128]", arg17_1: "f32[128]", arg18_1: "f32[128]", arg19_1: "f32[128, 128, 3, 3]", arg20_1: "f32[128]", arg21_1: "f32[128]", arg22_1: "f32[128]", arg23_1: "f32[128]", arg24_1: "f32[128]", arg25_1: "f32[256, 128, 3, 3]", arg26_1: "f32[256]", arg27_1: "f32[256]", arg28_1: "f32[256]", arg29_1: "f32[256]", arg30_1: "f32[256]", arg31_1: "f32[256, 256, 3, 3]", arg32_1: "f32[256]", arg33_1: "f32[256]", arg34_1: "f32[256]", arg35_1: "f32[256]", arg36_1: "f32[256]", arg37_1: "f32[256, 256, 3, 3]", arg38_1: "f32[256]", arg39_1: "f32[256]", arg40_1: "f32[256]", arg41_1: "f32[256]", arg42_1: "f32[256]", arg43_1: "f32[512, 256, 3, 3]", arg44_1: "f32[512]", arg45_1: "f32[512]", arg46_1: "f32[512]", arg47_1: "f32[512]", arg48_1: "f32[512]", arg49_1: "f32[512, 512, 3, 3]", arg50_1: "f32[512]", arg51_1: "f32[512]", arg52_1: "f32[512]", arg53_1: "f32[512]", arg54_1: "f32[512]", arg55_1: "f32[512, 512, 3, 3]", arg56_1: "f32[512]", arg57_1: "f32[512]", arg58_1: "f32[512]", arg59_1: "f32[512]", arg60_1: "f32[512]", arg61_1: "f32[512, 512, 3, 3]", arg62_1: "f32[512]", arg63_1: "f32[512]", arg64_1: "f32[512]", arg65_1: "f32[512]", arg66_1: "f32[512]", arg67_1: "f32[512, 512, 3, 3]", arg68_1: "f32[512]", arg69_1: "f32[512]", arg70_1: "f32[512]", arg71_1: "f32[512]", arg72_1: "f32[512]", arg73_1: "f32[512, 512, 3, 3]", arg74_1: "f32[512]", arg75_1: "f32[512]", arg76_1: "f32[512]", arg77_1: "f32[512]", arg78_1: "f32[512]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/doctr/models/recognition/crnn/pytorch.py:208 in forward, code: features = self.feat_extractor(x)
        convolution: "f32[1, 64, 32, 128]" = torch.ops.aten.convolution.default(arg2_1, arg0_1, arg1_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg2_1 = arg0_1 = arg1_1 = None
        unsqueeze: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg3_1, -1);  arg3_1 = None
        unsqueeze_1: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, -1);  unsqueeze = None
        sub: "f32[1, 64, 32, 128]" = torch.ops.aten.sub.Tensor(convolution, unsqueeze_1);  convolution = unsqueeze_1 = None
        add: "f32[64]" = torch.ops.aten.add.Tensor(arg4_1, 1e-05);  arg4_1 = None
        sqrt: "f32[64]" = torch.ops.aten.sqrt.default(add);  add = None
        reciprocal: "f32[64]" = torch.ops.aten.reciprocal.default(sqrt);  sqrt = None
        mul: "f32[64]" = torch.ops.aten.mul.Tensor(reciprocal, 1);  reciprocal = None
        unsqueeze_2: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(mul, -1);  mul = None
        unsqueeze_3: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, -1);  unsqueeze_2 = None
        mul_1: "f32[1, 64, 32, 128]" = torch.ops.aten.mul.Tensor(sub, unsqueeze_3);  sub = unsqueeze_3 = None
        unsqueeze_4: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg5_1, -1);  arg5_1 = None
        unsqueeze_5: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, -1);  unsqueeze_4 = None
        mul_2: "f32[1, 64, 32, 128]" = torch.ops.aten.mul.Tensor(mul_1, unsqueeze_5);  mul_1 = unsqueeze_5 = None
        unsqueeze_6: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg6_1, -1);  arg6_1 = None
        unsqueeze_7: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_6, -1);  unsqueeze_6 = None
        add_1: "f32[1, 64, 32, 128]" = torch.ops.aten.add.Tensor(mul_2, unsqueeze_7);  mul_2 = unsqueeze_7 = None
        relu: "f32[1, 64, 32, 128]" = torch.ops.aten.relu.default(add_1);  add_1 = None
        convolution_1: "f32[1, 64, 32, 128]" = torch.ops.aten.convolution.default(relu, arg7_1, arg8_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu = arg7_1 = arg8_1 = None
        unsqueeze_8: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg9_1, -1);  arg9_1 = None
        unsqueeze_9: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_8, -1);  unsqueeze_8 = None
        sub_1: "f32[1, 64, 32, 128]" = torch.ops.aten.sub.Tensor(convolution_1, unsqueeze_9);  convolution_1 = unsqueeze_9 = None
        add_2: "f32[64]" = torch.ops.aten.add.Tensor(arg10_1, 1e-05);  arg10_1 = None
        sqrt_1: "f32[64]" = torch.ops.aten.sqrt.default(add_2);  add_2 = None
        reciprocal_1: "f32[64]" = torch.ops.aten.reciprocal.default(sqrt_1);  sqrt_1 = None
        mul_3: "f32[64]" = torch.ops.aten.mul.Tensor(reciprocal_1, 1);  reciprocal_1 = None
        unsqueeze_10: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(mul_3, -1);  mul_3 = None
        unsqueeze_11: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_10, -1);  unsqueeze_10 = None
        mul_4: "f32[1, 64, 32, 128]" = torch.ops.aten.mul.Tensor(sub_1, unsqueeze_11);  sub_1 = unsqueeze_11 = None
        unsqueeze_12: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg11_1, -1);  arg11_1 = None
        unsqueeze_13: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_12, -1);  unsqueeze_12 = None
        mul_5: "f32[1, 64, 32, 128]" = torch.ops.aten.mul.Tensor(mul_4, unsqueeze_13);  mul_4 = unsqueeze_13 = None
        unsqueeze_14: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg12_1, -1);  arg12_1 = None
        unsqueeze_15: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_14, -1);  unsqueeze_14 = None
        add_3: "f32[1, 64, 32, 128]" = torch.ops.aten.add.Tensor(mul_5, unsqueeze_15);  mul_5 = unsqueeze_15 = None
        relu_1: "f32[1, 64, 32, 128]" = torch.ops.aten.relu.default(add_3);  add_3 = None
        _low_memory_max_pool_with_offsets = torch.ops.prims._low_memory_max_pool_with_offsets.default(relu_1, [2, 2], [2, 2], [0, 0], [1, 1], False);  relu_1 = None
        getitem: "f32[1, 64, 16, 64]" = _low_memory_max_pool_with_offsets[0];  _low_memory_max_pool_with_offsets = None
        convolution_2: "f32[1, 128, 16, 64]" = torch.ops.aten.convolution.default(getitem, arg13_1, arg14_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  getitem = arg13_1 = arg14_1 = None
        unsqueeze_16: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg15_1, -1);  arg15_1 = None
        unsqueeze_17: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_16, -1);  unsqueeze_16 = None
        sub_2: "f32[1, 128, 16, 64]" = torch.ops.aten.sub.Tensor(convolution_2, unsqueeze_17);  convolution_2 = unsqueeze_17 = None
        add_4: "f32[128]" = torch.ops.aten.add.Tensor(arg16_1, 1e-05);  arg16_1 = None
        sqrt_2: "f32[128]" = torch.ops.aten.sqrt.default(add_4);  add_4 = None
        reciprocal_2: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_2);  sqrt_2 = None
        mul_6: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_2, 1);  reciprocal_2 = None
        unsqueeze_18: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_6, -1);  mul_6 = None
        unsqueeze_19: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_18, -1);  unsqueeze_18 = None
        mul_7: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(sub_2, unsqueeze_19);  sub_2 = unsqueeze_19 = None
        unsqueeze_20: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg17_1, -1);  arg17_1 = None
        unsqueeze_21: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_20, -1);  unsqueeze_20 = None
        mul_8: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(mul_7, unsqueeze_21);  mul_7 = unsqueeze_21 = None
        unsqueeze_22: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg18_1, -1);  arg18_1 = None
        unsqueeze_23: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_22, -1);  unsqueeze_22 = None
        add_5: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_8, unsqueeze_23);  mul_8 = unsqueeze_23 = None
        relu_2: "f32[1, 128, 16, 64]" = torch.ops.aten.relu.default(add_5);  add_5 = None
        convolution_3: "f32[1, 128, 16, 64]" = torch.ops.aten.convolution.default(relu_2, arg19_1, arg20_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_2 = arg19_1 = arg20_1 = None
        unsqueeze_24: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg21_1, -1);  arg21_1 = None
        unsqueeze_25: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_24, -1);  unsqueeze_24 = None
        sub_3: "f32[1, 128, 16, 64]" = torch.ops.aten.sub.Tensor(convolution_3, unsqueeze_25);  convolution_3 = unsqueeze_25 = None
        add_6: "f32[128]" = torch.ops.aten.add.Tensor(arg22_1, 1e-05);  arg22_1 = None
        sqrt_3: "f32[128]" = torch.ops.aten.sqrt.default(add_6);  add_6 = None
        reciprocal_3: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_3);  sqrt_3 = None
        mul_9: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_3, 1);  reciprocal_3 = None
        unsqueeze_26: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_9, -1);  mul_9 = None
        unsqueeze_27: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_26, -1);  unsqueeze_26 = None
        mul_10: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(sub_3, unsqueeze_27);  sub_3 = unsqueeze_27 = None
        unsqueeze_28: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg23_1, -1);  arg23_1 = None
        unsqueeze_29: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_28, -1);  unsqueeze_28 = None
        mul_11: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(mul_10, unsqueeze_29);  mul_10 = unsqueeze_29 = None
        unsqueeze_30: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg24_1, -1);  arg24_1 = None
        unsqueeze_31: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_30, -1);  unsqueeze_30 = None
        add_7: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_11, unsqueeze_31);  mul_11 = unsqueeze_31 = None
        relu_3: "f32[1, 128, 16, 64]" = torch.ops.aten.relu.default(add_7);  add_7 = None
        _low_memory_max_pool_with_offsets_1 = torch.ops.prims._low_memory_max_pool_with_offsets.default(relu_3, [2, 2], [2, 2], [0, 0], [1, 1], False);  relu_3 = None
        getitem_2: "f32[1, 128, 8, 32]" = _low_memory_max_pool_with_offsets_1[0];  _low_memory_max_pool_with_offsets_1 = None
        convolution_4: "f32[1, 256, 8, 32]" = torch.ops.aten.convolution.default(getitem_2, arg25_1, arg26_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  getitem_2 = arg25_1 = arg26_1 = None
        unsqueeze_32: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg27_1, -1);  arg27_1 = None
        unsqueeze_33: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_32, -1);  unsqueeze_32 = None
        sub_4: "f32[1, 256, 8, 32]" = torch.ops.aten.sub.Tensor(convolution_4, unsqueeze_33);  convolution_4 = unsqueeze_33 = None
        add_8: "f32[256]" = torch.ops.aten.add.Tensor(arg28_1, 1e-05);  arg28_1 = None
        sqrt_4: "f32[256]" = torch.ops.aten.sqrt.default(add_8);  add_8 = None
        reciprocal_4: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_4);  sqrt_4 = None
        mul_12: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_4, 1);  reciprocal_4 = None
        unsqueeze_34: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_12, -1);  mul_12 = None
        unsqueeze_35: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_34, -1);  unsqueeze_34 = None
        mul_13: "f32[1, 256, 8, 32]" = torch.ops.aten.mul.Tensor(sub_4, unsqueeze_35);  sub_4 = unsqueeze_35 = None
        unsqueeze_36: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg29_1, -1);  arg29_1 = None
        unsqueeze_37: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_36, -1);  unsqueeze_36 = None
        mul_14: "f32[1, 256, 8, 32]" = torch.ops.aten.mul.Tensor(mul_13, unsqueeze_37);  mul_13 = unsqueeze_37 = None
        unsqueeze_38: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg30_1, -1);  arg30_1 = None
        unsqueeze_39: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_38, -1);  unsqueeze_38 = None
        add_9: "f32[1, 256, 8, 32]" = torch.ops.aten.add.Tensor(mul_14, unsqueeze_39);  mul_14 = unsqueeze_39 = None
        relu_4: "f32[1, 256, 8, 32]" = torch.ops.aten.relu.default(add_9);  add_9 = None
        convolution_5: "f32[1, 256, 8, 32]" = torch.ops.aten.convolution.default(relu_4, arg31_1, arg32_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_4 = arg31_1 = arg32_1 = None
        unsqueeze_40: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg33_1, -1);  arg33_1 = None
        unsqueeze_41: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_40, -1);  unsqueeze_40 = None
        sub_5: "f32[1, 256, 8, 32]" = torch.ops.aten.sub.Tensor(convolution_5, unsqueeze_41);  convolution_5 = unsqueeze_41 = None
        add_10: "f32[256]" = torch.ops.aten.add.Tensor(arg34_1, 1e-05);  arg34_1 = None
        sqrt_5: "f32[256]" = torch.ops.aten.sqrt.default(add_10);  add_10 = None
        reciprocal_5: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_5);  sqrt_5 = None
        mul_15: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_5, 1);  reciprocal_5 = None
        unsqueeze_42: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_15, -1);  mul_15 = None
        unsqueeze_43: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_42, -1);  unsqueeze_42 = None
        mul_16: "f32[1, 256, 8, 32]" = torch.ops.aten.mul.Tensor(sub_5, unsqueeze_43);  sub_5 = unsqueeze_43 = None
        unsqueeze_44: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg35_1, -1);  arg35_1 = None
        unsqueeze_45: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_44, -1);  unsqueeze_44 = None
        mul_17: "f32[1, 256, 8, 32]" = torch.ops.aten.mul.Tensor(mul_16, unsqueeze_45);  mul_16 = unsqueeze_45 = None
        unsqueeze_46: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg36_1, -1);  arg36_1 = None
        unsqueeze_47: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_46, -1);  unsqueeze_46 = None
        add_11: "f32[1, 256, 8, 32]" = torch.ops.aten.add.Tensor(mul_17, unsqueeze_47);  mul_17 = unsqueeze_47 = None
        relu_5: "f32[1, 256, 8, 32]" = torch.ops.aten.relu.default(add_11);  add_11 = None
        convolution_6: "f32[1, 256, 8, 32]" = torch.ops.aten.convolution.default(relu_5, arg37_1, arg38_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_5 = arg37_1 = arg38_1 = None
        unsqueeze_48: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg39_1, -1);  arg39_1 = None
        unsqueeze_49: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_48, -1);  unsqueeze_48 = None
        sub_6: "f32[1, 256, 8, 32]" = torch.ops.aten.sub.Tensor(convolution_6, unsqueeze_49);  convolution_6 = unsqueeze_49 = None
        add_12: "f32[256]" = torch.ops.aten.add.Tensor(arg40_1, 1e-05);  arg40_1 = None
        sqrt_6: "f32[256]" = torch.ops.aten.sqrt.default(add_12);  add_12 = None
        reciprocal_6: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_6);  sqrt_6 = None
        mul_18: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_6, 1);  reciprocal_6 = None
        unsqueeze_50: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_18, -1);  mul_18 = None
        unsqueeze_51: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_50, -1);  unsqueeze_50 = None
        mul_19: "f32[1, 256, 8, 32]" = torch.ops.aten.mul.Tensor(sub_6, unsqueeze_51);  sub_6 = unsqueeze_51 = None
        unsqueeze_52: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg41_1, -1);  arg41_1 = None
        unsqueeze_53: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_52, -1);  unsqueeze_52 = None
        mul_20: "f32[1, 256, 8, 32]" = torch.ops.aten.mul.Tensor(mul_19, unsqueeze_53);  mul_19 = unsqueeze_53 = None
        unsqueeze_54: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg42_1, -1);  arg42_1 = None
        unsqueeze_55: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_54, -1);  unsqueeze_54 = None
        add_13: "f32[1, 256, 8, 32]" = torch.ops.aten.add.Tensor(mul_20, unsqueeze_55);  mul_20 = unsqueeze_55 = None
        relu_6: "f32[1, 256, 8, 32]" = torch.ops.aten.relu.default(add_13);  add_13 = None
        _low_memory_max_pool_with_offsets_2 = torch.ops.prims._low_memory_max_pool_with_offsets.default(relu_6, [2, 1], [2, 1], [0, 0], [1, 1], False);  relu_6 = None
        getitem_4: "f32[1, 256, 4, 32]" = _low_memory_max_pool_with_offsets_2[0];  _low_memory_max_pool_with_offsets_2 = None
        convolution_7: "f32[1, 512, 4, 32]" = torch.ops.aten.convolution.default(getitem_4, arg43_1, arg44_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  getitem_4 = arg43_1 = arg44_1 = None
        unsqueeze_56: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg45_1, -1);  arg45_1 = None
        unsqueeze_57: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_56, -1);  unsqueeze_56 = None
        sub_7: "f32[1, 512, 4, 32]" = torch.ops.aten.sub.Tensor(convolution_7, unsqueeze_57);  convolution_7 = unsqueeze_57 = None
        add_14: "f32[512]" = torch.ops.aten.add.Tensor(arg46_1, 1e-05);  arg46_1 = None
        sqrt_7: "f32[512]" = torch.ops.aten.sqrt.default(add_14);  add_14 = None
        reciprocal_7: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_7);  sqrt_7 = None
        mul_21: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_7, 1);  reciprocal_7 = None
        unsqueeze_58: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_21, -1);  mul_21 = None
        unsqueeze_59: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_58, -1);  unsqueeze_58 = None
        mul_22: "f32[1, 512, 4, 32]" = torch.ops.aten.mul.Tensor(sub_7, unsqueeze_59);  sub_7 = unsqueeze_59 = None
        unsqueeze_60: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg47_1, -1);  arg47_1 = None
        unsqueeze_61: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_60, -1);  unsqueeze_60 = None
        mul_23: "f32[1, 512, 4, 32]" = torch.ops.aten.mul.Tensor(mul_22, unsqueeze_61);  mul_22 = unsqueeze_61 = None
        unsqueeze_62: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg48_1, -1);  arg48_1 = None
        unsqueeze_63: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_62, -1);  unsqueeze_62 = None
        add_15: "f32[1, 512, 4, 32]" = torch.ops.aten.add.Tensor(mul_23, unsqueeze_63);  mul_23 = unsqueeze_63 = None
        relu_7: "f32[1, 512, 4, 32]" = torch.ops.aten.relu.default(add_15);  add_15 = None
        convolution_8: "f32[1, 512, 4, 32]" = torch.ops.aten.convolution.default(relu_7, arg49_1, arg50_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_7 = arg49_1 = arg50_1 = None
        unsqueeze_64: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg51_1, -1);  arg51_1 = None
        unsqueeze_65: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_64, -1);  unsqueeze_64 = None
        sub_8: "f32[1, 512, 4, 32]" = torch.ops.aten.sub.Tensor(convolution_8, unsqueeze_65);  convolution_8 = unsqueeze_65 = None
        add_16: "f32[512]" = torch.ops.aten.add.Tensor(arg52_1, 1e-05);  arg52_1 = None
        sqrt_8: "f32[512]" = torch.ops.aten.sqrt.default(add_16);  add_16 = None
        reciprocal_8: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_8);  sqrt_8 = None
        mul_24: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_8, 1);  reciprocal_8 = None
        unsqueeze_66: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_24, -1);  mul_24 = None
        unsqueeze_67: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_66, -1);  unsqueeze_66 = None
        mul_25: "f32[1, 512, 4, 32]" = torch.ops.aten.mul.Tensor(sub_8, unsqueeze_67);  sub_8 = unsqueeze_67 = None
        unsqueeze_68: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg53_1, -1);  arg53_1 = None
        unsqueeze_69: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_68, -1);  unsqueeze_68 = None
        mul_26: "f32[1, 512, 4, 32]" = torch.ops.aten.mul.Tensor(mul_25, unsqueeze_69);  mul_25 = unsqueeze_69 = None
        unsqueeze_70: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg54_1, -1);  arg54_1 = None
        unsqueeze_71: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_70, -1);  unsqueeze_70 = None
        add_17: "f32[1, 512, 4, 32]" = torch.ops.aten.add.Tensor(mul_26, unsqueeze_71);  mul_26 = unsqueeze_71 = None
        relu_8: "f32[1, 512, 4, 32]" = torch.ops.aten.relu.default(add_17);  add_17 = None
        convolution_9: "f32[1, 512, 4, 32]" = torch.ops.aten.convolution.default(relu_8, arg55_1, arg56_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_8 = arg55_1 = arg56_1 = None
        unsqueeze_72: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg57_1, -1);  arg57_1 = None
        unsqueeze_73: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_72, -1);  unsqueeze_72 = None
        sub_9: "f32[1, 512, 4, 32]" = torch.ops.aten.sub.Tensor(convolution_9, unsqueeze_73);  convolution_9 = unsqueeze_73 = None
        add_18: "f32[512]" = torch.ops.aten.add.Tensor(arg58_1, 1e-05);  arg58_1 = None
        sqrt_9: "f32[512]" = torch.ops.aten.sqrt.default(add_18);  add_18 = None
        reciprocal_9: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_9);  sqrt_9 = None
        mul_27: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_9, 1);  reciprocal_9 = None
        unsqueeze_74: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_27, -1);  mul_27 = None
        unsqueeze_75: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_74, -1);  unsqueeze_74 = None
        mul_28: "f32[1, 512, 4, 32]" = torch.ops.aten.mul.Tensor(sub_9, unsqueeze_75);  sub_9 = unsqueeze_75 = None
        unsqueeze_76: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg59_1, -1);  arg59_1 = None
        unsqueeze_77: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_76, -1);  unsqueeze_76 = None
        mul_29: "f32[1, 512, 4, 32]" = torch.ops.aten.mul.Tensor(mul_28, unsqueeze_77);  mul_28 = unsqueeze_77 = None
        unsqueeze_78: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg60_1, -1);  arg60_1 = None
        unsqueeze_79: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_78, -1);  unsqueeze_78 = None
        add_19: "f32[1, 512, 4, 32]" = torch.ops.aten.add.Tensor(mul_29, unsqueeze_79);  mul_29 = unsqueeze_79 = None
        relu_9: "f32[1, 512, 4, 32]" = torch.ops.aten.relu.default(add_19);  add_19 = None
        _low_memory_max_pool_with_offsets_3 = torch.ops.prims._low_memory_max_pool_with_offsets.default(relu_9, [2, 1], [2, 1], [0, 0], [1, 1], False);  relu_9 = None
        getitem_6: "f32[1, 512, 2, 32]" = _low_memory_max_pool_with_offsets_3[0];  _low_memory_max_pool_with_offsets_3 = None
        convolution_10: "f32[1, 512, 2, 32]" = torch.ops.aten.convolution.default(getitem_6, arg61_1, arg62_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  getitem_6 = arg61_1 = arg62_1 = None
        unsqueeze_80: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg63_1, -1);  arg63_1 = None
        unsqueeze_81: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_80, -1);  unsqueeze_80 = None
        sub_10: "f32[1, 512, 2, 32]" = torch.ops.aten.sub.Tensor(convolution_10, unsqueeze_81);  convolution_10 = unsqueeze_81 = None
        add_20: "f32[512]" = torch.ops.aten.add.Tensor(arg64_1, 1e-05);  arg64_1 = None
        sqrt_10: "f32[512]" = torch.ops.aten.sqrt.default(add_20);  add_20 = None
        reciprocal_10: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_10);  sqrt_10 = None
        mul_30: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_10, 1);  reciprocal_10 = None
        unsqueeze_82: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_30, -1);  mul_30 = None
        unsqueeze_83: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_82, -1);  unsqueeze_82 = None
        mul_31: "f32[1, 512, 2, 32]" = torch.ops.aten.mul.Tensor(sub_10, unsqueeze_83);  sub_10 = unsqueeze_83 = None
        unsqueeze_84: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg65_1, -1);  arg65_1 = None
        unsqueeze_85: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_84, -1);  unsqueeze_84 = None
        mul_32: "f32[1, 512, 2, 32]" = torch.ops.aten.mul.Tensor(mul_31, unsqueeze_85);  mul_31 = unsqueeze_85 = None
        unsqueeze_86: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg66_1, -1);  arg66_1 = None
        unsqueeze_87: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_86, -1);  unsqueeze_86 = None
        add_21: "f32[1, 512, 2, 32]" = torch.ops.aten.add.Tensor(mul_32, unsqueeze_87);  mul_32 = unsqueeze_87 = None
        relu_10: "f32[1, 512, 2, 32]" = torch.ops.aten.relu.default(add_21);  add_21 = None
        convolution_11: "f32[1, 512, 2, 32]" = torch.ops.aten.convolution.default(relu_10, arg67_1, arg68_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_10 = arg67_1 = arg68_1 = None
        unsqueeze_88: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg69_1, -1);  arg69_1 = None
        unsqueeze_89: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_88, -1);  unsqueeze_88 = None
        sub_11: "f32[1, 512, 2, 32]" = torch.ops.aten.sub.Tensor(convolution_11, unsqueeze_89);  convolution_11 = unsqueeze_89 = None
        add_22: "f32[512]" = torch.ops.aten.add.Tensor(arg70_1, 1e-05);  arg70_1 = None
        sqrt_11: "f32[512]" = torch.ops.aten.sqrt.default(add_22);  add_22 = None
        reciprocal_11: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_11);  sqrt_11 = None
        mul_33: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_11, 1);  reciprocal_11 = None
        unsqueeze_90: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_33, -1);  mul_33 = None
        unsqueeze_91: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_90, -1);  unsqueeze_90 = None
        mul_34: "f32[1, 512, 2, 32]" = torch.ops.aten.mul.Tensor(sub_11, unsqueeze_91);  sub_11 = unsqueeze_91 = None
        unsqueeze_92: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg71_1, -1);  arg71_1 = None
        unsqueeze_93: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_92, -1);  unsqueeze_92 = None
        mul_35: "f32[1, 512, 2, 32]" = torch.ops.aten.mul.Tensor(mul_34, unsqueeze_93);  mul_34 = unsqueeze_93 = None
        unsqueeze_94: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg72_1, -1);  arg72_1 = None
        unsqueeze_95: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_94, -1);  unsqueeze_94 = None
        add_23: "f32[1, 512, 2, 32]" = torch.ops.aten.add.Tensor(mul_35, unsqueeze_95);  mul_35 = unsqueeze_95 = None
        relu_11: "f32[1, 512, 2, 32]" = torch.ops.aten.relu.default(add_23);  add_23 = None
        convolution_12: "f32[1, 512, 2, 32]" = torch.ops.aten.convolution.default(relu_11, arg73_1, arg74_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_11 = arg73_1 = arg74_1 = None
        unsqueeze_96: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg75_1, -1);  arg75_1 = None
        unsqueeze_97: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_96, -1);  unsqueeze_96 = None
        sub_12: "f32[1, 512, 2, 32]" = torch.ops.aten.sub.Tensor(convolution_12, unsqueeze_97);  convolution_12 = unsqueeze_97 = None
        add_24: "f32[512]" = torch.ops.aten.add.Tensor(arg76_1, 1e-05);  arg76_1 = None
        sqrt_12: "f32[512]" = torch.ops.aten.sqrt.default(add_24);  add_24 = None
        reciprocal_12: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_12);  sqrt_12 = None
        mul_36: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_12, 1);  reciprocal_12 = None
        unsqueeze_98: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_36, -1);  mul_36 = None
        unsqueeze_99: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_98, -1);  unsqueeze_98 = None
        mul_37: "f32[1, 512, 2, 32]" = torch.ops.aten.mul.Tensor(sub_12, unsqueeze_99);  sub_12 = unsqueeze_99 = None
        unsqueeze_100: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg77_1, -1);  arg77_1 = None
        unsqueeze_101: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_100, -1);  unsqueeze_100 = None
        mul_38: "f32[1, 512, 2, 32]" = torch.ops.aten.mul.Tensor(mul_37, unsqueeze_101);  mul_37 = unsqueeze_101 = None
        unsqueeze_102: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg78_1, -1);  arg78_1 = None
        unsqueeze_103: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_102, -1);  unsqueeze_102 = None
        add_25: "f32[1, 512, 2, 32]" = torch.ops.aten.add.Tensor(mul_38, unsqueeze_103);  mul_38 = unsqueeze_103 = None
        relu_12: "f32[1, 512, 2, 32]" = torch.ops.aten.relu.default(add_25);  add_25 = None
        _low_memory_max_pool_with_offsets_4 = torch.ops.prims._low_memory_max_pool_with_offsets.default(relu_12, [2, 1], [2, 1], [0, 0], [1, 1], False);  relu_12 = None
        getitem_8: "f32[1, 512, 1, 32]" = _low_memory_max_pool_with_offsets_4[0];  _low_memory_max_pool_with_offsets_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/doctr/models/recognition/crnn/pytorch.py:211 in forward, code: features_seq = torch.reshape(features, shape=(-1, h * c, w))
        view: "f32[1, 512, 32]" = torch.ops.aten.reshape.default(getitem_8, [-1, 512, 32]);  getitem_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/doctr/models/recognition/crnn/pytorch.py:212 in forward, code: features_seq = torch.transpose(features_seq, 1, 2)
        permute: "f32[1, 32, 512]" = torch.ops.aten.permute.default(view, [0, 2, 1]);  view = None
        return (permute,)
