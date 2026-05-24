import torch
from torch import device
from math import inf, nan

class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "f32[32, 512, 28, 28]", arg1_1: "f32[1024, 512, 1, 1]", arg2_1: "f32[1024]", arg3_1: "f32[1024]", arg4_1: "f32[1024]", arg5_1: "f32[1024]", arg6_1: "f32[32, 1024, 14, 14]", arg7_1: "f32[256, 1024, 1, 1]", arg8_1: "f32[256]", arg9_1: "f32[256]", arg10_1: "f32[256]", arg11_1: "f32[256]", arg12_1: "f32[256, 256, 3, 3]", arg13_1: "f32[256]", arg14_1: "f32[256]", arg15_1: "f32[256]", arg16_1: "f32[256]", arg17_1: "f32[1024, 256, 1, 1]", arg18_1: "f32[1024]", arg19_1: "f32[1024]", arg20_1: "f32[1024]", arg21_1: "f32[1024]", arg22_1: "f32[256, 1024, 1, 1]", arg23_1: "f32[256]", arg24_1: "f32[256]", arg25_1: "f32[256]", arg26_1: "f32[256]", arg27_1: "f32[256, 256, 3, 3]", arg28_1: "f32[256]", arg29_1: "f32[256]", arg30_1: "f32[256]", arg31_1: "f32[256]", arg32_1: "f32[1024, 256, 1, 1]", arg33_1: "f32[1024]", arg34_1: "f32[1024]", arg35_1: "f32[1024]", arg36_1: "f32[1024]", arg37_1: "f32[256, 1024, 1, 1]", arg38_1: "f32[256]", arg39_1: "f32[256]", arg40_1: "f32[256]", arg41_1: "f32[256]", arg42_1: "f32[256, 256, 3, 3]", arg43_1: "f32[256]", arg44_1: "f32[256]", arg45_1: "f32[256]", arg46_1: "f32[256]", arg47_1: "f32[1024, 256, 1, 1]", arg48_1: "f32[1024]", arg49_1: "f32[1024]", arg50_1: "f32[1024]", arg51_1: "f32[1024]", arg52_1: "f32[256, 1024, 1, 1]", arg53_1: "f32[256]", arg54_1: "f32[256]", arg55_1: "f32[256]", arg56_1: "f32[256]", arg57_1: "f32[256, 256, 3, 3]", arg58_1: "f32[256]", arg59_1: "f32[256]", arg60_1: "f32[256]", arg61_1: "f32[256]", arg62_1: "f32[1024, 256, 1, 1]", arg63_1: "f32[1024]", arg64_1: "f32[1024]", arg65_1: "f32[1024]", arg66_1: "f32[1024]", arg67_1: "f32[256, 1024, 1, 1]", arg68_1: "f32[256]", arg69_1: "f32[256]", arg70_1: "f32[256]", arg71_1: "f32[256]", arg72_1: "f32[256, 256, 3, 3]", arg73_1: "f32[256]", arg74_1: "f32[256]", arg75_1: "f32[256]", arg76_1: "f32[256]", arg77_1: "f32[1024, 256, 1, 1]", arg78_1: "f32[1024]", arg79_1: "f32[1024]", arg80_1: "f32[1024]", arg81_1: "f32[1024]", arg82_1: "f32[512, 1024, 1, 1]", arg83_1: "f32[512]", arg84_1: "f32[512]", arg85_1: "f32[512]", arg86_1: "f32[512]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:158 in forward, code: identity = self.downsample(x)
        convolution: "f32[32, 1024, 14, 14]" = torch.ops.aten.convolution.default(arg0_1, arg1_1, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  arg0_1 = arg1_1 = None
        unsqueeze: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg2_1, -1);  arg2_1 = None
        unsqueeze_1: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, -1);  unsqueeze = None
        sub: "f32[32, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution, unsqueeze_1);  convolution = unsqueeze_1 = None
        add: "f32[1024]" = torch.ops.aten.add.Tensor(arg3_1, 1e-05);  arg3_1 = None
        sqrt: "f32[1024]" = torch.ops.aten.sqrt.default(add);  add = None
        reciprocal: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt);  sqrt = None
        mul: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal, 1);  reciprocal = None
        unsqueeze_2: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul, -1);  mul = None
        unsqueeze_3: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, -1);  unsqueeze_2 = None
        mul_1: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub, unsqueeze_3);  sub = unsqueeze_3 = None
        unsqueeze_4: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg4_1, -1);  arg4_1 = None
        unsqueeze_5: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, -1);  unsqueeze_4 = None
        mul_2: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_1, unsqueeze_5);  mul_1 = unsqueeze_5 = None
        unsqueeze_6: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg5_1, -1);  arg5_1 = None
        unsqueeze_7: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_6, -1);  unsqueeze_6 = None
        add_1: "f32[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_2, unsqueeze_7);  mul_2 = unsqueeze_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:160 in forward, code: out += identity
        add_2: "f32[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(arg6_1, add_1);  add_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        relu: "f32[32, 1024, 14, 14]" = torch.ops.aten.relu.default(add_2);  add_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_1: "f32[32, 256, 14, 14]" = torch.ops.aten.convolution.default(relu, arg7_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg7_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        unsqueeze_8: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg8_1, -1);  arg8_1 = None
        unsqueeze_9: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_8, -1);  unsqueeze_8 = None
        sub_1: "f32[32, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_1, unsqueeze_9);  convolution_1 = unsqueeze_9 = None
        add_3: "f32[256]" = torch.ops.aten.add.Tensor(arg9_1, 1e-05);  arg9_1 = None
        sqrt_1: "f32[256]" = torch.ops.aten.sqrt.default(add_3);  add_3 = None
        reciprocal_1: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_1);  sqrt_1 = None
        mul_3: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_1, 1);  reciprocal_1 = None
        unsqueeze_10: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_3, -1);  mul_3 = None
        unsqueeze_11: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_10, -1);  unsqueeze_10 = None
        mul_4: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_1, unsqueeze_11);  sub_1 = unsqueeze_11 = None
        unsqueeze_12: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg10_1, -1);  arg10_1 = None
        unsqueeze_13: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_12, -1);  unsqueeze_12 = None
        mul_5: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_4, unsqueeze_13);  mul_4 = unsqueeze_13 = None
        unsqueeze_14: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg11_1, -1);  arg11_1 = None
        unsqueeze_15: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_14, -1);  unsqueeze_14 = None
        add_4: "f32[32, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_5, unsqueeze_15);  mul_5 = unsqueeze_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        relu_1: "f32[32, 256, 14, 14]" = torch.ops.aten.relu.default(add_4);  add_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convolution_2: "f32[32, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_1, arg12_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_1 = arg12_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        unsqueeze_16: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg13_1, -1);  arg13_1 = None
        unsqueeze_17: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_16, -1);  unsqueeze_16 = None
        sub_2: "f32[32, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_2, unsqueeze_17);  convolution_2 = unsqueeze_17 = None
        add_5: "f32[256]" = torch.ops.aten.add.Tensor(arg14_1, 1e-05);  arg14_1 = None
        sqrt_2: "f32[256]" = torch.ops.aten.sqrt.default(add_5);  add_5 = None
        reciprocal_2: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_2);  sqrt_2 = None
        mul_6: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_2, 1);  reciprocal_2 = None
        unsqueeze_18: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_6, -1);  mul_6 = None
        unsqueeze_19: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_18, -1);  unsqueeze_18 = None
        mul_7: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_2, unsqueeze_19);  sub_2 = unsqueeze_19 = None
        unsqueeze_20: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg15_1, -1);  arg15_1 = None
        unsqueeze_21: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_20, -1);  unsqueeze_20 = None
        mul_8: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_7, unsqueeze_21);  mul_7 = unsqueeze_21 = None
        unsqueeze_22: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg16_1, -1);  arg16_1 = None
        unsqueeze_23: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_22, -1);  unsqueeze_22 = None
        add_6: "f32[32, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_8, unsqueeze_23);  mul_8 = unsqueeze_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        relu_2: "f32[32, 256, 14, 14]" = torch.ops.aten.relu.default(add_6);  add_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convolution_3: "f32[32, 1024, 14, 14]" = torch.ops.aten.convolution.default(relu_2, arg17_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_2 = arg17_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        unsqueeze_24: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg18_1, -1);  arg18_1 = None
        unsqueeze_25: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_24, -1);  unsqueeze_24 = None
        sub_3: "f32[32, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_3, unsqueeze_25);  convolution_3 = unsqueeze_25 = None
        add_7: "f32[1024]" = torch.ops.aten.add.Tensor(arg19_1, 1e-05);  arg19_1 = None
        sqrt_3: "f32[1024]" = torch.ops.aten.sqrt.default(add_7);  add_7 = None
        reciprocal_3: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_3);  sqrt_3 = None
        mul_9: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_3, 1);  reciprocal_3 = None
        unsqueeze_26: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_9, -1);  mul_9 = None
        unsqueeze_27: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_26, -1);  unsqueeze_26 = None
        mul_10: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_3, unsqueeze_27);  sub_3 = unsqueeze_27 = None
        unsqueeze_28: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg20_1, -1);  arg20_1 = None
        unsqueeze_29: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_28, -1);  unsqueeze_28 = None
        mul_11: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_10, unsqueeze_29);  mul_10 = unsqueeze_29 = None
        unsqueeze_30: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg21_1, -1);  arg21_1 = None
        unsqueeze_31: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_30, -1);  unsqueeze_30 = None
        add_8: "f32[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_11, unsqueeze_31);  mul_11 = unsqueeze_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:160 in forward, code: out += identity
        add_9: "f32[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_8, relu);  add_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        relu_3: "f32[32, 1024, 14, 14]" = torch.ops.aten.relu.default(add_9);  add_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_4: "f32[32, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_3, arg22_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg22_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        unsqueeze_32: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg23_1, -1);  arg23_1 = None
        unsqueeze_33: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_32, -1);  unsqueeze_32 = None
        sub_4: "f32[32, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_4, unsqueeze_33);  convolution_4 = unsqueeze_33 = None
        add_10: "f32[256]" = torch.ops.aten.add.Tensor(arg24_1, 1e-05);  arg24_1 = None
        sqrt_4: "f32[256]" = torch.ops.aten.sqrt.default(add_10);  add_10 = None
        reciprocal_4: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_4);  sqrt_4 = None
        mul_12: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_4, 1);  reciprocal_4 = None
        unsqueeze_34: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_12, -1);  mul_12 = None
        unsqueeze_35: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_34, -1);  unsqueeze_34 = None
        mul_13: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_4, unsqueeze_35);  sub_4 = unsqueeze_35 = None
        unsqueeze_36: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg25_1, -1);  arg25_1 = None
        unsqueeze_37: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_36, -1);  unsqueeze_36 = None
        mul_14: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_13, unsqueeze_37);  mul_13 = unsqueeze_37 = None
        unsqueeze_38: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg26_1, -1);  arg26_1 = None
        unsqueeze_39: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_38, -1);  unsqueeze_38 = None
        add_11: "f32[32, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_14, unsqueeze_39);  mul_14 = unsqueeze_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        relu_4: "f32[32, 256, 14, 14]" = torch.ops.aten.relu.default(add_11);  add_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convolution_5: "f32[32, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_4, arg27_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_4 = arg27_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        unsqueeze_40: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg28_1, -1);  arg28_1 = None
        unsqueeze_41: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_40, -1);  unsqueeze_40 = None
        sub_5: "f32[32, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_5, unsqueeze_41);  convolution_5 = unsqueeze_41 = None
        add_12: "f32[256]" = torch.ops.aten.add.Tensor(arg29_1, 1e-05);  arg29_1 = None
        sqrt_5: "f32[256]" = torch.ops.aten.sqrt.default(add_12);  add_12 = None
        reciprocal_5: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_5);  sqrt_5 = None
        mul_15: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_5, 1);  reciprocal_5 = None
        unsqueeze_42: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_15, -1);  mul_15 = None
        unsqueeze_43: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_42, -1);  unsqueeze_42 = None
        mul_16: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_5, unsqueeze_43);  sub_5 = unsqueeze_43 = None
        unsqueeze_44: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg30_1, -1);  arg30_1 = None
        unsqueeze_45: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_44, -1);  unsqueeze_44 = None
        mul_17: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_16, unsqueeze_45);  mul_16 = unsqueeze_45 = None
        unsqueeze_46: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg31_1, -1);  arg31_1 = None
        unsqueeze_47: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_46, -1);  unsqueeze_46 = None
        add_13: "f32[32, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_17, unsqueeze_47);  mul_17 = unsqueeze_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        relu_5: "f32[32, 256, 14, 14]" = torch.ops.aten.relu.default(add_13);  add_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convolution_6: "f32[32, 1024, 14, 14]" = torch.ops.aten.convolution.default(relu_5, arg32_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_5 = arg32_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        unsqueeze_48: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg33_1, -1);  arg33_1 = None
        unsqueeze_49: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_48, -1);  unsqueeze_48 = None
        sub_6: "f32[32, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_6, unsqueeze_49);  convolution_6 = unsqueeze_49 = None
        add_14: "f32[1024]" = torch.ops.aten.add.Tensor(arg34_1, 1e-05);  arg34_1 = None
        sqrt_6: "f32[1024]" = torch.ops.aten.sqrt.default(add_14);  add_14 = None
        reciprocal_6: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_6);  sqrt_6 = None
        mul_18: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_6, 1);  reciprocal_6 = None
        unsqueeze_50: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_18, -1);  mul_18 = None
        unsqueeze_51: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_50, -1);  unsqueeze_50 = None
        mul_19: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_6, unsqueeze_51);  sub_6 = unsqueeze_51 = None
        unsqueeze_52: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg35_1, -1);  arg35_1 = None
        unsqueeze_53: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_52, -1);  unsqueeze_52 = None
        mul_20: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_19, unsqueeze_53);  mul_19 = unsqueeze_53 = None
        unsqueeze_54: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg36_1, -1);  arg36_1 = None
        unsqueeze_55: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_54, -1);  unsqueeze_54 = None
        add_15: "f32[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_20, unsqueeze_55);  mul_20 = unsqueeze_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:160 in forward, code: out += identity
        add_16: "f32[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_15, relu_3);  add_15 = relu_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        relu_6: "f32[32, 1024, 14, 14]" = torch.ops.aten.relu.default(add_16);  add_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_7: "f32[32, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_6, arg37_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg37_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        unsqueeze_56: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg38_1, -1);  arg38_1 = None
        unsqueeze_57: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_56, -1);  unsqueeze_56 = None
        sub_7: "f32[32, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_7, unsqueeze_57);  convolution_7 = unsqueeze_57 = None
        add_17: "f32[256]" = torch.ops.aten.add.Tensor(arg39_1, 1e-05);  arg39_1 = None
        sqrt_7: "f32[256]" = torch.ops.aten.sqrt.default(add_17);  add_17 = None
        reciprocal_7: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_7);  sqrt_7 = None
        mul_21: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_7, 1);  reciprocal_7 = None
        unsqueeze_58: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_21, -1);  mul_21 = None
        unsqueeze_59: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_58, -1);  unsqueeze_58 = None
        mul_22: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_7, unsqueeze_59);  sub_7 = unsqueeze_59 = None
        unsqueeze_60: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg40_1, -1);  arg40_1 = None
        unsqueeze_61: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_60, -1);  unsqueeze_60 = None
        mul_23: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_22, unsqueeze_61);  mul_22 = unsqueeze_61 = None
        unsqueeze_62: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg41_1, -1);  arg41_1 = None
        unsqueeze_63: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_62, -1);  unsqueeze_62 = None
        add_18: "f32[32, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_23, unsqueeze_63);  mul_23 = unsqueeze_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        relu_7: "f32[32, 256, 14, 14]" = torch.ops.aten.relu.default(add_18);  add_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convolution_8: "f32[32, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_7, arg42_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_7 = arg42_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        unsqueeze_64: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg43_1, -1);  arg43_1 = None
        unsqueeze_65: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_64, -1);  unsqueeze_64 = None
        sub_8: "f32[32, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_8, unsqueeze_65);  convolution_8 = unsqueeze_65 = None
        add_19: "f32[256]" = torch.ops.aten.add.Tensor(arg44_1, 1e-05);  arg44_1 = None
        sqrt_8: "f32[256]" = torch.ops.aten.sqrt.default(add_19);  add_19 = None
        reciprocal_8: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_8);  sqrt_8 = None
        mul_24: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_8, 1);  reciprocal_8 = None
        unsqueeze_66: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_24, -1);  mul_24 = None
        unsqueeze_67: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_66, -1);  unsqueeze_66 = None
        mul_25: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_8, unsqueeze_67);  sub_8 = unsqueeze_67 = None
        unsqueeze_68: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg45_1, -1);  arg45_1 = None
        unsqueeze_69: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_68, -1);  unsqueeze_68 = None
        mul_26: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_25, unsqueeze_69);  mul_25 = unsqueeze_69 = None
        unsqueeze_70: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg46_1, -1);  arg46_1 = None
        unsqueeze_71: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_70, -1);  unsqueeze_70 = None
        add_20: "f32[32, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_26, unsqueeze_71);  mul_26 = unsqueeze_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        relu_8: "f32[32, 256, 14, 14]" = torch.ops.aten.relu.default(add_20);  add_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convolution_9: "f32[32, 1024, 14, 14]" = torch.ops.aten.convolution.default(relu_8, arg47_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_8 = arg47_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        unsqueeze_72: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg48_1, -1);  arg48_1 = None
        unsqueeze_73: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_72, -1);  unsqueeze_72 = None
        sub_9: "f32[32, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_9, unsqueeze_73);  convolution_9 = unsqueeze_73 = None
        add_21: "f32[1024]" = torch.ops.aten.add.Tensor(arg49_1, 1e-05);  arg49_1 = None
        sqrt_9: "f32[1024]" = torch.ops.aten.sqrt.default(add_21);  add_21 = None
        reciprocal_9: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_9);  sqrt_9 = None
        mul_27: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_9, 1);  reciprocal_9 = None
        unsqueeze_74: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_27, -1);  mul_27 = None
        unsqueeze_75: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_74, -1);  unsqueeze_74 = None
        mul_28: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_9, unsqueeze_75);  sub_9 = unsqueeze_75 = None
        unsqueeze_76: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg50_1, -1);  arg50_1 = None
        unsqueeze_77: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_76, -1);  unsqueeze_76 = None
        mul_29: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_28, unsqueeze_77);  mul_28 = unsqueeze_77 = None
        unsqueeze_78: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg51_1, -1);  arg51_1 = None
        unsqueeze_79: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_78, -1);  unsqueeze_78 = None
        add_22: "f32[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_29, unsqueeze_79);  mul_29 = unsqueeze_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:160 in forward, code: out += identity
        add_23: "f32[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_22, relu_6);  add_22 = relu_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        relu_9: "f32[32, 1024, 14, 14]" = torch.ops.aten.relu.default(add_23);  add_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_10: "f32[32, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_9, arg52_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg52_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        unsqueeze_80: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg53_1, -1);  arg53_1 = None
        unsqueeze_81: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_80, -1);  unsqueeze_80 = None
        sub_10: "f32[32, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_10, unsqueeze_81);  convolution_10 = unsqueeze_81 = None
        add_24: "f32[256]" = torch.ops.aten.add.Tensor(arg54_1, 1e-05);  arg54_1 = None
        sqrt_10: "f32[256]" = torch.ops.aten.sqrt.default(add_24);  add_24 = None
        reciprocal_10: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_10);  sqrt_10 = None
        mul_30: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_10, 1);  reciprocal_10 = None
        unsqueeze_82: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_30, -1);  mul_30 = None
        unsqueeze_83: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_82, -1);  unsqueeze_82 = None
        mul_31: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_10, unsqueeze_83);  sub_10 = unsqueeze_83 = None
        unsqueeze_84: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg55_1, -1);  arg55_1 = None
        unsqueeze_85: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_84, -1);  unsqueeze_84 = None
        mul_32: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_31, unsqueeze_85);  mul_31 = unsqueeze_85 = None
        unsqueeze_86: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg56_1, -1);  arg56_1 = None
        unsqueeze_87: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_86, -1);  unsqueeze_86 = None
        add_25: "f32[32, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_32, unsqueeze_87);  mul_32 = unsqueeze_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        relu_10: "f32[32, 256, 14, 14]" = torch.ops.aten.relu.default(add_25);  add_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convolution_11: "f32[32, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_10, arg57_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_10 = arg57_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        unsqueeze_88: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg58_1, -1);  arg58_1 = None
        unsqueeze_89: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_88, -1);  unsqueeze_88 = None
        sub_11: "f32[32, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_11, unsqueeze_89);  convolution_11 = unsqueeze_89 = None
        add_26: "f32[256]" = torch.ops.aten.add.Tensor(arg59_1, 1e-05);  arg59_1 = None
        sqrt_11: "f32[256]" = torch.ops.aten.sqrt.default(add_26);  add_26 = None
        reciprocal_11: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_11);  sqrt_11 = None
        mul_33: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_11, 1);  reciprocal_11 = None
        unsqueeze_90: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_33, -1);  mul_33 = None
        unsqueeze_91: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_90, -1);  unsqueeze_90 = None
        mul_34: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_11, unsqueeze_91);  sub_11 = unsqueeze_91 = None
        unsqueeze_92: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg60_1, -1);  arg60_1 = None
        unsqueeze_93: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_92, -1);  unsqueeze_92 = None
        mul_35: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_34, unsqueeze_93);  mul_34 = unsqueeze_93 = None
        unsqueeze_94: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg61_1, -1);  arg61_1 = None
        unsqueeze_95: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_94, -1);  unsqueeze_94 = None
        add_27: "f32[32, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_35, unsqueeze_95);  mul_35 = unsqueeze_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        relu_11: "f32[32, 256, 14, 14]" = torch.ops.aten.relu.default(add_27);  add_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convolution_12: "f32[32, 1024, 14, 14]" = torch.ops.aten.convolution.default(relu_11, arg62_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_11 = arg62_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        unsqueeze_96: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg63_1, -1);  arg63_1 = None
        unsqueeze_97: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_96, -1);  unsqueeze_96 = None
        sub_12: "f32[32, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_12, unsqueeze_97);  convolution_12 = unsqueeze_97 = None
        add_28: "f32[1024]" = torch.ops.aten.add.Tensor(arg64_1, 1e-05);  arg64_1 = None
        sqrt_12: "f32[1024]" = torch.ops.aten.sqrt.default(add_28);  add_28 = None
        reciprocal_12: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_12);  sqrt_12 = None
        mul_36: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_12, 1);  reciprocal_12 = None
        unsqueeze_98: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_36, -1);  mul_36 = None
        unsqueeze_99: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_98, -1);  unsqueeze_98 = None
        mul_37: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_12, unsqueeze_99);  sub_12 = unsqueeze_99 = None
        unsqueeze_100: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg65_1, -1);  arg65_1 = None
        unsqueeze_101: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_100, -1);  unsqueeze_100 = None
        mul_38: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_37, unsqueeze_101);  mul_37 = unsqueeze_101 = None
        unsqueeze_102: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg66_1, -1);  arg66_1 = None
        unsqueeze_103: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_102, -1);  unsqueeze_102 = None
        add_29: "f32[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_38, unsqueeze_103);  mul_38 = unsqueeze_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:160 in forward, code: out += identity
        add_30: "f32[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_29, relu_9);  add_29 = relu_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        relu_12: "f32[32, 1024, 14, 14]" = torch.ops.aten.relu.default(add_30);  add_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_13: "f32[32, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_12, arg67_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg67_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        unsqueeze_104: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg68_1, -1);  arg68_1 = None
        unsqueeze_105: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_104, -1);  unsqueeze_104 = None
        sub_13: "f32[32, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_13, unsqueeze_105);  convolution_13 = unsqueeze_105 = None
        add_31: "f32[256]" = torch.ops.aten.add.Tensor(arg69_1, 1e-05);  arg69_1 = None
        sqrt_13: "f32[256]" = torch.ops.aten.sqrt.default(add_31);  add_31 = None
        reciprocal_13: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_13);  sqrt_13 = None
        mul_39: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_13, 1);  reciprocal_13 = None
        unsqueeze_106: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_39, -1);  mul_39 = None
        unsqueeze_107: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_106, -1);  unsqueeze_106 = None
        mul_40: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_13, unsqueeze_107);  sub_13 = unsqueeze_107 = None
        unsqueeze_108: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg70_1, -1);  arg70_1 = None
        unsqueeze_109: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_108, -1);  unsqueeze_108 = None
        mul_41: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_40, unsqueeze_109);  mul_40 = unsqueeze_109 = None
        unsqueeze_110: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg71_1, -1);  arg71_1 = None
        unsqueeze_111: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_110, -1);  unsqueeze_110 = None
        add_32: "f32[32, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_41, unsqueeze_111);  mul_41 = unsqueeze_111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        relu_13: "f32[32, 256, 14, 14]" = torch.ops.aten.relu.default(add_32);  add_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convolution_14: "f32[32, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_13, arg72_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_13 = arg72_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        unsqueeze_112: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg73_1, -1);  arg73_1 = None
        unsqueeze_113: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_112, -1);  unsqueeze_112 = None
        sub_14: "f32[32, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_14, unsqueeze_113);  convolution_14 = unsqueeze_113 = None
        add_33: "f32[256]" = torch.ops.aten.add.Tensor(arg74_1, 1e-05);  arg74_1 = None
        sqrt_14: "f32[256]" = torch.ops.aten.sqrt.default(add_33);  add_33 = None
        reciprocal_14: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_14);  sqrt_14 = None
        mul_42: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_14, 1);  reciprocal_14 = None
        unsqueeze_114: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_42, -1);  mul_42 = None
        unsqueeze_115: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_114, -1);  unsqueeze_114 = None
        mul_43: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_14, unsqueeze_115);  sub_14 = unsqueeze_115 = None
        unsqueeze_116: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg75_1, -1);  arg75_1 = None
        unsqueeze_117: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_116, -1);  unsqueeze_116 = None
        mul_44: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_43, unsqueeze_117);  mul_43 = unsqueeze_117 = None
        unsqueeze_118: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg76_1, -1);  arg76_1 = None
        unsqueeze_119: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_118, -1);  unsqueeze_118 = None
        add_34: "f32[32, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_44, unsqueeze_119);  mul_44 = unsqueeze_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        relu_14: "f32[32, 256, 14, 14]" = torch.ops.aten.relu.default(add_34);  add_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convolution_15: "f32[32, 1024, 14, 14]" = torch.ops.aten.convolution.default(relu_14, arg77_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_14 = arg77_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        unsqueeze_120: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg78_1, -1);  arg78_1 = None
        unsqueeze_121: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_120, -1);  unsqueeze_120 = None
        sub_15: "f32[32, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_15, unsqueeze_121);  convolution_15 = unsqueeze_121 = None
        add_35: "f32[1024]" = torch.ops.aten.add.Tensor(arg79_1, 1e-05);  arg79_1 = None
        sqrt_15: "f32[1024]" = torch.ops.aten.sqrt.default(add_35);  add_35 = None
        reciprocal_15: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_15);  sqrt_15 = None
        mul_45: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_15, 1);  reciprocal_15 = None
        unsqueeze_122: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_45, -1);  mul_45 = None
        unsqueeze_123: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_122, -1);  unsqueeze_122 = None
        mul_46: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_15, unsqueeze_123);  sub_15 = unsqueeze_123 = None
        unsqueeze_124: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg80_1, -1);  arg80_1 = None
        unsqueeze_125: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_124, -1);  unsqueeze_124 = None
        mul_47: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_46, unsqueeze_125);  mul_46 = unsqueeze_125 = None
        unsqueeze_126: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg81_1, -1);  arg81_1 = None
        unsqueeze_127: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_126, -1);  unsqueeze_126 = None
        add_36: "f32[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_47, unsqueeze_127);  mul_47 = unsqueeze_127 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:160 in forward, code: out += identity
        add_37: "f32[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_36, relu_12);  add_36 = relu_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        relu_15: "f32[32, 1024, 14, 14]" = torch.ops.aten.relu.default(add_37);  add_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_16: "f32[32, 512, 14, 14]" = torch.ops.aten.convolution.default(relu_15, arg82_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg82_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        unsqueeze_128: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg83_1, -1);  arg83_1 = None
        unsqueeze_129: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_128, -1);  unsqueeze_128 = None
        sub_16: "f32[32, 512, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_16, unsqueeze_129);  convolution_16 = unsqueeze_129 = None
        add_38: "f32[512]" = torch.ops.aten.add.Tensor(arg84_1, 1e-05);  arg84_1 = None
        sqrt_16: "f32[512]" = torch.ops.aten.sqrt.default(add_38);  add_38 = None
        reciprocal_16: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_16);  sqrt_16 = None
        mul_48: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_16, 1);  reciprocal_16 = None
        unsqueeze_130: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_48, -1);  mul_48 = None
        unsqueeze_131: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_130, -1);  unsqueeze_130 = None
        mul_49: "f32[32, 512, 14, 14]" = torch.ops.aten.mul.Tensor(sub_16, unsqueeze_131);  sub_16 = unsqueeze_131 = None
        unsqueeze_132: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg85_1, -1);  arg85_1 = None
        unsqueeze_133: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_132, -1);  unsqueeze_132 = None
        mul_50: "f32[32, 512, 14, 14]" = torch.ops.aten.mul.Tensor(mul_49, unsqueeze_133);  mul_49 = unsqueeze_133 = None
        unsqueeze_134: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg86_1, -1);  arg86_1 = None
        unsqueeze_135: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_134, -1);  unsqueeze_134 = None
        add_39: "f32[32, 512, 14, 14]" = torch.ops.aten.add.Tensor(mul_50, unsqueeze_135);  mul_50 = unsqueeze_135 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        relu_16: "f32[32, 512, 14, 14]" = torch.ops.aten.relu.default(add_39);  add_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        copy_: "f32[32, 1024, 14, 14]" = torch.ops.aten.copy_.default(arg6_1, relu);  arg6_1 = relu = copy_ = None
        return (relu_16, relu_15)
