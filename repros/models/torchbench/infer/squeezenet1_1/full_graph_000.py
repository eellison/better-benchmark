import torch
import torch._inductor.inductor_prims  # registers inductor prims ops
from torch import device
from math import inf, nan

class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "f16[64, 3, 3, 3]", arg1_1: "f16[64]", arg2_1: "f16[512, 3, 224, 224]", arg3_1: "f16[16, 64, 1, 1]", arg4_1: "f16[16]", arg5_1: "f16[64, 16, 1, 1]", arg6_1: "f16[64]", arg7_1: "f16[64, 16, 3, 3]", arg8_1: "f16[64]", arg9_1: "f16[16, 128, 1, 1]", arg10_1: "f16[16]", arg11_1: "f16[64, 16, 1, 1]", arg12_1: "f16[64]", arg13_1: "f16[64, 16, 3, 3]", arg14_1: "f16[64]", arg15_1: "f16[32, 128, 1, 1]", arg16_1: "f16[32]", arg17_1: "f16[128, 32, 1, 1]", arg18_1: "f16[128]", arg19_1: "f16[128, 32, 3, 3]", arg20_1: "f16[128]", arg21_1: "f16[32, 256, 1, 1]", arg22_1: "f16[32]", arg23_1: "f16[128, 32, 1, 1]", arg24_1: "f16[128]", arg25_1: "f16[128, 32, 3, 3]", arg26_1: "f16[128]", arg27_1: "f16[48, 256, 1, 1]", arg28_1: "f16[48]", arg29_1: "f16[192, 48, 1, 1]", arg30_1: "f16[192]", arg31_1: "f16[192, 48, 3, 3]", arg32_1: "f16[192]", arg33_1: "f16[48, 384, 1, 1]", arg34_1: "f16[48]", arg35_1: "f16[192, 48, 1, 1]", arg36_1: "f16[192]", arg37_1: "f16[192, 48, 3, 3]", arg38_1: "f16[192]", arg39_1: "f16[64, 384, 1, 1]", arg40_1: "f16[64]", arg41_1: "f16[256, 64, 1, 1]", arg42_1: "f16[256]", arg43_1: "f16[256, 64, 3, 3]", arg44_1: "f16[256]", arg45_1: "f16[64, 512, 1, 1]", arg46_1: "f16[64]", arg47_1: "f16[256, 64, 1, 1]", arg48_1: "f16[256]", arg49_1: "f16[256, 64, 3, 3]", arg50_1: "f16[256]", arg51_1: "f16[1000, 512, 1, 1]", arg52_1: "f16[1000]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:95 in forward, code: x = self.features(x)
        convolution: "f16[512, 64, 111, 111]" = torch.ops.aten.convolution.default(arg2_1, arg0_1, arg1_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  arg2_1 = arg0_1 = arg1_1 = None
        relu: "f16[512, 64, 111, 111]" = torch.ops.aten.relu.default(convolution);  convolution = None
        _low_memory_max_pool_with_offsets = torch.ops.prims._low_memory_max_pool_with_offsets.default(relu, [3, 3], [2, 2], [0, 0], [1, 1], True);  relu = None
        getitem: "f16[512, 64, 55, 55]" = _low_memory_max_pool_with_offsets[0];  _low_memory_max_pool_with_offsets = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:30 in forward, code: x = self.squeeze_activation(self.squeeze(x))
        convolution_1: "f16[512, 16, 55, 55]" = torch.ops.aten.convolution.default(getitem, arg3_1, arg4_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem = arg3_1 = arg4_1 = None
        relu_1: "f16[512, 16, 55, 55]" = torch.ops.aten.relu.default(convolution_1);  convolution_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:32 in forward, code: [self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1
        convolution_2: "f16[512, 64, 55, 55]" = torch.ops.aten.convolution.default(relu_1, arg5_1, arg6_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg5_1 = arg6_1 = None
        relu_2: "f16[512, 64, 55, 55]" = torch.ops.aten.relu.default(convolution_2);  convolution_2 = None
        convolution_3: "f16[512, 64, 55, 55]" = torch.ops.aten.convolution.default(relu_1, arg7_1, arg8_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_1 = arg7_1 = arg8_1 = None
        relu_3: "f16[512, 64, 55, 55]" = torch.ops.aten.relu.default(convolution_3);  convolution_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:31 in forward, code: return torch.cat(
        cat: "f16[512, 128, 55, 55]" = torch.ops.aten.cat.default([relu_2, relu_3], 1);  relu_2 = relu_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:30 in forward, code: x = self.squeeze_activation(self.squeeze(x))
        convolution_4: "f16[512, 16, 55, 55]" = torch.ops.aten.convolution.default(cat, arg9_1, arg10_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat = arg9_1 = arg10_1 = None
        relu_4: "f16[512, 16, 55, 55]" = torch.ops.aten.relu.default(convolution_4);  convolution_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:32 in forward, code: [self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1
        convolution_5: "f16[512, 64, 55, 55]" = torch.ops.aten.convolution.default(relu_4, arg11_1, arg12_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg11_1 = arg12_1 = None
        relu_5: "f16[512, 64, 55, 55]" = torch.ops.aten.relu.default(convolution_5);  convolution_5 = None
        convolution_6: "f16[512, 64, 55, 55]" = torch.ops.aten.convolution.default(relu_4, arg13_1, arg14_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_4 = arg13_1 = arg14_1 = None
        relu_6: "f16[512, 64, 55, 55]" = torch.ops.aten.relu.default(convolution_6);  convolution_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:31 in forward, code: return torch.cat(
        cat_1: "f16[512, 128, 55, 55]" = torch.ops.aten.cat.default([relu_5, relu_6], 1);  relu_5 = relu_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:95 in forward, code: x = self.features(x)
        _low_memory_max_pool_with_offsets_1 = torch.ops.prims._low_memory_max_pool_with_offsets.default(cat_1, [3, 3], [2, 2], [0, 0], [1, 1], True);  cat_1 = None
        getitem_2: "f16[512, 128, 27, 27]" = _low_memory_max_pool_with_offsets_1[0];  _low_memory_max_pool_with_offsets_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:30 in forward, code: x = self.squeeze_activation(self.squeeze(x))
        convolution_7: "f16[512, 32, 27, 27]" = torch.ops.aten.convolution.default(getitem_2, arg15_1, arg16_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_2 = arg15_1 = arg16_1 = None
        relu_7: "f16[512, 32, 27, 27]" = torch.ops.aten.relu.default(convolution_7);  convolution_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:32 in forward, code: [self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1
        convolution_8: "f16[512, 128, 27, 27]" = torch.ops.aten.convolution.default(relu_7, arg17_1, arg18_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg17_1 = arg18_1 = None
        relu_8: "f16[512, 128, 27, 27]" = torch.ops.aten.relu.default(convolution_8);  convolution_8 = None
        convolution_9: "f16[512, 128, 27, 27]" = torch.ops.aten.convolution.default(relu_7, arg19_1, arg20_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_7 = arg19_1 = arg20_1 = None
        relu_9: "f16[512, 128, 27, 27]" = torch.ops.aten.relu.default(convolution_9);  convolution_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:31 in forward, code: return torch.cat(
        cat_2: "f16[512, 256, 27, 27]" = torch.ops.aten.cat.default([relu_8, relu_9], 1);  relu_8 = relu_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:30 in forward, code: x = self.squeeze_activation(self.squeeze(x))
        convolution_10: "f16[512, 32, 27, 27]" = torch.ops.aten.convolution.default(cat_2, arg21_1, arg22_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_2 = arg21_1 = arg22_1 = None
        relu_10: "f16[512, 32, 27, 27]" = torch.ops.aten.relu.default(convolution_10);  convolution_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:32 in forward, code: [self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1
        convolution_11: "f16[512, 128, 27, 27]" = torch.ops.aten.convolution.default(relu_10, arg23_1, arg24_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg23_1 = arg24_1 = None
        relu_11: "f16[512, 128, 27, 27]" = torch.ops.aten.relu.default(convolution_11);  convolution_11 = None
        convolution_12: "f16[512, 128, 27, 27]" = torch.ops.aten.convolution.default(relu_10, arg25_1, arg26_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_10 = arg25_1 = arg26_1 = None
        relu_12: "f16[512, 128, 27, 27]" = torch.ops.aten.relu.default(convolution_12);  convolution_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:31 in forward, code: return torch.cat(
        cat_3: "f16[512, 256, 27, 27]" = torch.ops.aten.cat.default([relu_11, relu_12], 1);  relu_11 = relu_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:95 in forward, code: x = self.features(x)
        _low_memory_max_pool_with_offsets_2 = torch.ops.prims._low_memory_max_pool_with_offsets.default(cat_3, [3, 3], [2, 2], [0, 0], [1, 1], True);  cat_3 = None
        getitem_4: "f16[512, 256, 13, 13]" = _low_memory_max_pool_with_offsets_2[0];  _low_memory_max_pool_with_offsets_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:30 in forward, code: x = self.squeeze_activation(self.squeeze(x))
        convolution_13: "f16[512, 48, 13, 13]" = torch.ops.aten.convolution.default(getitem_4, arg27_1, arg28_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_4 = arg27_1 = arg28_1 = None
        relu_13: "f16[512, 48, 13, 13]" = torch.ops.aten.relu.default(convolution_13);  convolution_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:32 in forward, code: [self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1
        convolution_14: "f16[512, 192, 13, 13]" = torch.ops.aten.convolution.default(relu_13, arg29_1, arg30_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg29_1 = arg30_1 = None
        relu_14: "f16[512, 192, 13, 13]" = torch.ops.aten.relu.default(convolution_14);  convolution_14 = None
        convolution_15: "f16[512, 192, 13, 13]" = torch.ops.aten.convolution.default(relu_13, arg31_1, arg32_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_13 = arg31_1 = arg32_1 = None
        relu_15: "f16[512, 192, 13, 13]" = torch.ops.aten.relu.default(convolution_15);  convolution_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:31 in forward, code: return torch.cat(
        cat_4: "f16[512, 384, 13, 13]" = torch.ops.aten.cat.default([relu_14, relu_15], 1);  relu_14 = relu_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:30 in forward, code: x = self.squeeze_activation(self.squeeze(x))
        convolution_16: "f16[512, 48, 13, 13]" = torch.ops.aten.convolution.default(cat_4, arg33_1, arg34_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_4 = arg33_1 = arg34_1 = None
        relu_16: "f16[512, 48, 13, 13]" = torch.ops.aten.relu.default(convolution_16);  convolution_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:32 in forward, code: [self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1
        convolution_17: "f16[512, 192, 13, 13]" = torch.ops.aten.convolution.default(relu_16, arg35_1, arg36_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg35_1 = arg36_1 = None
        relu_17: "f16[512, 192, 13, 13]" = torch.ops.aten.relu.default(convolution_17);  convolution_17 = None
        convolution_18: "f16[512, 192, 13, 13]" = torch.ops.aten.convolution.default(relu_16, arg37_1, arg38_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_16 = arg37_1 = arg38_1 = None
        relu_18: "f16[512, 192, 13, 13]" = torch.ops.aten.relu.default(convolution_18);  convolution_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:31 in forward, code: return torch.cat(
        cat_5: "f16[512, 384, 13, 13]" = torch.ops.aten.cat.default([relu_17, relu_18], 1);  relu_17 = relu_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:30 in forward, code: x = self.squeeze_activation(self.squeeze(x))
        convolution_19: "f16[512, 64, 13, 13]" = torch.ops.aten.convolution.default(cat_5, arg39_1, arg40_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_5 = arg39_1 = arg40_1 = None
        relu_19: "f16[512, 64, 13, 13]" = torch.ops.aten.relu.default(convolution_19);  convolution_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:32 in forward, code: [self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1
        convolution_20: "f16[512, 256, 13, 13]" = torch.ops.aten.convolution.default(relu_19, arg41_1, arg42_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg41_1 = arg42_1 = None
        relu_20: "f16[512, 256, 13, 13]" = torch.ops.aten.relu.default(convolution_20);  convolution_20 = None
        convolution_21: "f16[512, 256, 13, 13]" = torch.ops.aten.convolution.default(relu_19, arg43_1, arg44_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_19 = arg43_1 = arg44_1 = None
        relu_21: "f16[512, 256, 13, 13]" = torch.ops.aten.relu.default(convolution_21);  convolution_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:31 in forward, code: return torch.cat(
        cat_6: "f16[512, 512, 13, 13]" = torch.ops.aten.cat.default([relu_20, relu_21], 1);  relu_20 = relu_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:30 in forward, code: x = self.squeeze_activation(self.squeeze(x))
        convolution_22: "f16[512, 64, 13, 13]" = torch.ops.aten.convolution.default(cat_6, arg45_1, arg46_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_6 = arg45_1 = arg46_1 = None
        relu_22: "f16[512, 64, 13, 13]" = torch.ops.aten.relu.default(convolution_22);  convolution_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:32 in forward, code: [self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1
        convolution_23: "f16[512, 256, 13, 13]" = torch.ops.aten.convolution.default(relu_22, arg47_1, arg48_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg47_1 = arg48_1 = None
        relu_23: "f16[512, 256, 13, 13]" = torch.ops.aten.relu.default(convolution_23);  convolution_23 = None
        convolution_24: "f16[512, 256, 13, 13]" = torch.ops.aten.convolution.default(relu_22, arg49_1, arg50_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_22 = arg49_1 = arg50_1 = None
        relu_24: "f16[512, 256, 13, 13]" = torch.ops.aten.relu.default(convolution_24);  convolution_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:31 in forward, code: return torch.cat(
        cat_7: "f16[512, 512, 13, 13]" = torch.ops.aten.cat.default([relu_23, relu_24], 1);  relu_23 = relu_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:96 in forward, code: x = self.classifier(x)
        convolution_25: "f16[512, 1000, 13, 13]" = torch.ops.aten.convolution.default(cat_7, arg51_1, arg52_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_7 = arg51_1 = arg52_1 = None
        relu_25: "f16[512, 1000, 13, 13]" = torch.ops.aten.relu.default(convolution_25);  convolution_25 = None
        mean: "f16[512, 1000, 1, 1]" = torch.ops.aten.mean.dim(relu_25, [-1, -2], True);  relu_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:97 in forward, code: return torch.flatten(x, 1)
        view: "f16[512, 1000]" = torch.ops.aten.reshape.default(mean, [512, 1000]);  mean = None
        return (view,)
