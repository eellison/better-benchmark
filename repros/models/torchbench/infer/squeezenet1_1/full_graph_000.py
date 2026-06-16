class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "bf16[64, 3, 3, 3][27, 9, 3, 1]cuda:0", arg1_1: "bf16[64][1]cuda:0", arg2_1: "bf16[16, 3, 224, 224][150528, 50176, 224, 1]cuda:0", arg3_1: "bf16[16, 64, 1, 1][64, 1, 1, 1]cuda:0", arg4_1: "bf16[16][1]cuda:0", arg5_1: "bf16[64, 16, 1, 1][16, 1, 1, 1]cuda:0", arg6_1: "bf16[64][1]cuda:0", arg7_1: "bf16[64, 16, 3, 3][144, 9, 3, 1]cuda:0", arg8_1: "bf16[64][1]cuda:0", arg9_1: "bf16[16, 128, 1, 1][128, 1, 1, 1]cuda:0", arg10_1: "bf16[16][1]cuda:0", arg11_1: "bf16[64, 16, 1, 1][16, 1, 1, 1]cuda:0", arg12_1: "bf16[64][1]cuda:0", arg13_1: "bf16[64, 16, 3, 3][144, 9, 3, 1]cuda:0", arg14_1: "bf16[64][1]cuda:0", arg15_1: "bf16[32, 128, 1, 1][128, 1, 1, 1]cuda:0", arg16_1: "bf16[32][1]cuda:0", arg17_1: "bf16[128, 32, 1, 1][32, 1, 1, 1]cuda:0", arg18_1: "bf16[128][1]cuda:0", arg19_1: "bf16[128, 32, 3, 3][288, 9, 3, 1]cuda:0", arg20_1: "bf16[128][1]cuda:0", arg21_1: "bf16[32, 256, 1, 1][256, 1, 1, 1]cuda:0", arg22_1: "bf16[32][1]cuda:0", arg23_1: "bf16[128, 32, 1, 1][32, 1, 1, 1]cuda:0", arg24_1: "bf16[128][1]cuda:0", arg25_1: "bf16[128, 32, 3, 3][288, 9, 3, 1]cuda:0", arg26_1: "bf16[128][1]cuda:0", arg27_1: "bf16[48, 256, 1, 1][256, 1, 1, 1]cuda:0", arg28_1: "bf16[48][1]cuda:0", arg29_1: "bf16[192, 48, 1, 1][48, 1, 1, 1]cuda:0", arg30_1: "bf16[192][1]cuda:0", arg31_1: "bf16[192, 48, 3, 3][432, 9, 3, 1]cuda:0", arg32_1: "bf16[192][1]cuda:0", arg33_1: "bf16[48, 384, 1, 1][384, 1, 1, 1]cuda:0", arg34_1: "bf16[48][1]cuda:0", arg35_1: "bf16[192, 48, 1, 1][48, 1, 1, 1]cuda:0", arg36_1: "bf16[192][1]cuda:0", arg37_1: "bf16[192, 48, 3, 3][432, 9, 3, 1]cuda:0", arg38_1: "bf16[192][1]cuda:0", arg39_1: "bf16[64, 384, 1, 1][384, 1, 1, 1]cuda:0", arg40_1: "bf16[64][1]cuda:0", arg41_1: "bf16[256, 64, 1, 1][64, 1, 1, 1]cuda:0", arg42_1: "bf16[256][1]cuda:0", arg43_1: "bf16[256, 64, 3, 3][576, 9, 3, 1]cuda:0", arg44_1: "bf16[256][1]cuda:0", arg45_1: "bf16[64, 512, 1, 1][512, 1, 1, 1]cuda:0", arg46_1: "bf16[64][1]cuda:0", arg47_1: "bf16[256, 64, 1, 1][64, 1, 1, 1]cuda:0", arg48_1: "bf16[256][1]cuda:0", arg49_1: "bf16[256, 64, 3, 3][576, 9, 3, 1]cuda:0", arg50_1: "bf16[256][1]cuda:0", arg51_1: "bf16[1000, 512, 1, 1][512, 1, 1, 1]cuda:0", arg52_1: "bf16[1000][1]cuda:0"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:95 in forward, code: x = self.features(x)
        convolution: "bf16[16, 64, 111, 111][788544, 12321, 111, 1]cuda:0" = torch.ops.aten.convolution.default(arg2_1, arg0_1, arg1_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  arg2_1 = arg0_1 = arg1_1 = None
        relu: "bf16[16, 64, 111, 111][788544, 12321, 111, 1]cuda:0" = torch.ops.aten.relu.default(convolution);  convolution = None
        _low_memory_max_pool_with_offsets = torch.ops.prims._low_memory_max_pool_with_offsets.default(relu, [3, 3], [2, 2], [0, 0], [1, 1], True);  relu = None
        getitem: "bf16[16, 64, 55, 55][193600, 3025, 55, 1]cuda:0" = _low_memory_max_pool_with_offsets[0];  _low_memory_max_pool_with_offsets = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:30 in forward, code: x = self.squeeze_activation(self.squeeze(x))
        convolution_1: "bf16[16, 16, 55, 55][48400, 3025, 55, 1]cuda:0" = torch.ops.aten.convolution.default(getitem, arg3_1, arg4_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem = arg3_1 = arg4_1 = None
        relu_1: "bf16[16, 16, 55, 55][48400, 3025, 55, 1]cuda:0" = torch.ops.aten.relu.default(convolution_1);  convolution_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:32 in forward, code: [self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1
        convolution_2: "bf16[16, 64, 55, 55][193600, 3025, 55, 1]cuda:0" = torch.ops.aten.convolution.default(relu_1, arg5_1, arg6_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg5_1 = arg6_1 = None
        relu_2: "bf16[16, 64, 55, 55][193600, 3025, 55, 1]cuda:0" = torch.ops.aten.relu.default(convolution_2);  convolution_2 = None
        convolution_3: "bf16[16, 64, 55, 55][193600, 3025, 55, 1]cuda:0" = torch.ops.aten.convolution.default(relu_1, arg7_1, arg8_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_1 = arg7_1 = arg8_1 = None
        relu_3: "bf16[16, 64, 55, 55][193600, 3025, 55, 1]cuda:0" = torch.ops.aten.relu.default(convolution_3);  convolution_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:31 in forward, code: return torch.cat(
        cat: "bf16[16, 128, 55, 55][387200, 3025, 55, 1]cuda:0" = torch.ops.aten.cat.default([relu_2, relu_3], 1);  relu_2 = relu_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:30 in forward, code: x = self.squeeze_activation(self.squeeze(x))
        convolution_4: "bf16[16, 16, 55, 55][48400, 3025, 55, 1]cuda:0" = torch.ops.aten.convolution.default(cat, arg9_1, arg10_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat = arg9_1 = arg10_1 = None
        relu_4: "bf16[16, 16, 55, 55][48400, 3025, 55, 1]cuda:0" = torch.ops.aten.relu.default(convolution_4);  convolution_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:32 in forward, code: [self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1
        convolution_5: "bf16[16, 64, 55, 55][193600, 3025, 55, 1]cuda:0" = torch.ops.aten.convolution.default(relu_4, arg11_1, arg12_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg11_1 = arg12_1 = None
        relu_5: "bf16[16, 64, 55, 55][193600, 3025, 55, 1]cuda:0" = torch.ops.aten.relu.default(convolution_5);  convolution_5 = None
        convolution_6: "bf16[16, 64, 55, 55][193600, 3025, 55, 1]cuda:0" = torch.ops.aten.convolution.default(relu_4, arg13_1, arg14_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_4 = arg13_1 = arg14_1 = None
        relu_6: "bf16[16, 64, 55, 55][193600, 3025, 55, 1]cuda:0" = torch.ops.aten.relu.default(convolution_6);  convolution_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:31 in forward, code: return torch.cat(
        cat_1: "bf16[16, 128, 55, 55][387200, 3025, 55, 1]cuda:0" = torch.ops.aten.cat.default([relu_5, relu_6], 1);  relu_5 = relu_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:95 in forward, code: x = self.features(x)
        _low_memory_max_pool_with_offsets_1 = torch.ops.prims._low_memory_max_pool_with_offsets.default(cat_1, [3, 3], [2, 2], [0, 0], [1, 1], True);  cat_1 = None
        getitem_2: "bf16[16, 128, 27, 27][93312, 729, 27, 1]cuda:0" = _low_memory_max_pool_with_offsets_1[0];  _low_memory_max_pool_with_offsets_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:30 in forward, code: x = self.squeeze_activation(self.squeeze(x))
        convolution_7: "bf16[16, 32, 27, 27][23328, 729, 27, 1]cuda:0" = torch.ops.aten.convolution.default(getitem_2, arg15_1, arg16_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_2 = arg15_1 = arg16_1 = None
        relu_7: "bf16[16, 32, 27, 27][23328, 729, 27, 1]cuda:0" = torch.ops.aten.relu.default(convolution_7);  convolution_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:32 in forward, code: [self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1
        convolution_8: "bf16[16, 128, 27, 27][93312, 729, 27, 1]cuda:0" = torch.ops.aten.convolution.default(relu_7, arg17_1, arg18_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg17_1 = arg18_1 = None
        relu_8: "bf16[16, 128, 27, 27][93312, 729, 27, 1]cuda:0" = torch.ops.aten.relu.default(convolution_8);  convolution_8 = None
        convolution_9: "bf16[16, 128, 27, 27][93312, 729, 27, 1]cuda:0" = torch.ops.aten.convolution.default(relu_7, arg19_1, arg20_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_7 = arg19_1 = arg20_1 = None
        relu_9: "bf16[16, 128, 27, 27][93312, 729, 27, 1]cuda:0" = torch.ops.aten.relu.default(convolution_9);  convolution_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:31 in forward, code: return torch.cat(
        cat_2: "bf16[16, 256, 27, 27][186624, 729, 27, 1]cuda:0" = torch.ops.aten.cat.default([relu_8, relu_9], 1);  relu_8 = relu_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:30 in forward, code: x = self.squeeze_activation(self.squeeze(x))
        convolution_10: "bf16[16, 32, 27, 27][23328, 729, 27, 1]cuda:0" = torch.ops.aten.convolution.default(cat_2, arg21_1, arg22_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_2 = arg21_1 = arg22_1 = None
        relu_10: "bf16[16, 32, 27, 27][23328, 729, 27, 1]cuda:0" = torch.ops.aten.relu.default(convolution_10);  convolution_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:32 in forward, code: [self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1
        convolution_11: "bf16[16, 128, 27, 27][93312, 729, 27, 1]cuda:0" = torch.ops.aten.convolution.default(relu_10, arg23_1, arg24_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg23_1 = arg24_1 = None
        relu_11: "bf16[16, 128, 27, 27][93312, 729, 27, 1]cuda:0" = torch.ops.aten.relu.default(convolution_11);  convolution_11 = None
        convolution_12: "bf16[16, 128, 27, 27][93312, 729, 27, 1]cuda:0" = torch.ops.aten.convolution.default(relu_10, arg25_1, arg26_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_10 = arg25_1 = arg26_1 = None
        relu_12: "bf16[16, 128, 27, 27][93312, 729, 27, 1]cuda:0" = torch.ops.aten.relu.default(convolution_12);  convolution_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:31 in forward, code: return torch.cat(
        cat_3: "bf16[16, 256, 27, 27][186624, 729, 27, 1]cuda:0" = torch.ops.aten.cat.default([relu_11, relu_12], 1);  relu_11 = relu_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:95 in forward, code: x = self.features(x)
        _low_memory_max_pool_with_offsets_2 = torch.ops.prims._low_memory_max_pool_with_offsets.default(cat_3, [3, 3], [2, 2], [0, 0], [1, 1], True);  cat_3 = None
        getitem_4: "bf16[16, 256, 13, 13][43264, 169, 13, 1]cuda:0" = _low_memory_max_pool_with_offsets_2[0];  _low_memory_max_pool_with_offsets_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:30 in forward, code: x = self.squeeze_activation(self.squeeze(x))
        convolution_13: "bf16[16, 48, 13, 13][8112, 169, 13, 1]cuda:0" = torch.ops.aten.convolution.default(getitem_4, arg27_1, arg28_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_4 = arg27_1 = arg28_1 = None
        relu_13: "bf16[16, 48, 13, 13][8112, 169, 13, 1]cuda:0" = torch.ops.aten.relu.default(convolution_13);  convolution_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:32 in forward, code: [self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1
        convolution_14: "bf16[16, 192, 13, 13][32448, 169, 13, 1]cuda:0" = torch.ops.aten.convolution.default(relu_13, arg29_1, arg30_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg29_1 = arg30_1 = None
        relu_14: "bf16[16, 192, 13, 13][32448, 169, 13, 1]cuda:0" = torch.ops.aten.relu.default(convolution_14);  convolution_14 = None
        convolution_15: "bf16[16, 192, 13, 13][32448, 169, 13, 1]cuda:0" = torch.ops.aten.convolution.default(relu_13, arg31_1, arg32_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_13 = arg31_1 = arg32_1 = None
        relu_15: "bf16[16, 192, 13, 13][32448, 169, 13, 1]cuda:0" = torch.ops.aten.relu.default(convolution_15);  convolution_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:31 in forward, code: return torch.cat(
        cat_4: "bf16[16, 384, 13, 13][64896, 169, 13, 1]cuda:0" = torch.ops.aten.cat.default([relu_14, relu_15], 1);  relu_14 = relu_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:30 in forward, code: x = self.squeeze_activation(self.squeeze(x))
        convolution_16: "bf16[16, 48, 13, 13][8112, 169, 13, 1]cuda:0" = torch.ops.aten.convolution.default(cat_4, arg33_1, arg34_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_4 = arg33_1 = arg34_1 = None
        relu_16: "bf16[16, 48, 13, 13][8112, 169, 13, 1]cuda:0" = torch.ops.aten.relu.default(convolution_16);  convolution_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:32 in forward, code: [self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1
        convolution_17: "bf16[16, 192, 13, 13][32448, 169, 13, 1]cuda:0" = torch.ops.aten.convolution.default(relu_16, arg35_1, arg36_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg35_1 = arg36_1 = None
        relu_17: "bf16[16, 192, 13, 13][32448, 169, 13, 1]cuda:0" = torch.ops.aten.relu.default(convolution_17);  convolution_17 = None
        convolution_18: "bf16[16, 192, 13, 13][32448, 169, 13, 1]cuda:0" = torch.ops.aten.convolution.default(relu_16, arg37_1, arg38_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_16 = arg37_1 = arg38_1 = None
        relu_18: "bf16[16, 192, 13, 13][32448, 169, 13, 1]cuda:0" = torch.ops.aten.relu.default(convolution_18);  convolution_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:31 in forward, code: return torch.cat(
        cat_5: "bf16[16, 384, 13, 13][64896, 169, 13, 1]cuda:0" = torch.ops.aten.cat.default([relu_17, relu_18], 1);  relu_17 = relu_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:30 in forward, code: x = self.squeeze_activation(self.squeeze(x))
        convolution_19: "bf16[16, 64, 13, 13][10816, 169, 13, 1]cuda:0" = torch.ops.aten.convolution.default(cat_5, arg39_1, arg40_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_5 = arg39_1 = arg40_1 = None
        relu_19: "bf16[16, 64, 13, 13][10816, 169, 13, 1]cuda:0" = torch.ops.aten.relu.default(convolution_19);  convolution_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:32 in forward, code: [self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1
        convolution_20: "bf16[16, 256, 13, 13][43264, 169, 13, 1]cuda:0" = torch.ops.aten.convolution.default(relu_19, arg41_1, arg42_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg41_1 = arg42_1 = None
        relu_20: "bf16[16, 256, 13, 13][43264, 169, 13, 1]cuda:0" = torch.ops.aten.relu.default(convolution_20);  convolution_20 = None
        convolution_21: "bf16[16, 256, 13, 13][43264, 169, 13, 1]cuda:0" = torch.ops.aten.convolution.default(relu_19, arg43_1, arg44_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_19 = arg43_1 = arg44_1 = None
        relu_21: "bf16[16, 256, 13, 13][43264, 169, 13, 1]cuda:0" = torch.ops.aten.relu.default(convolution_21);  convolution_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:31 in forward, code: return torch.cat(
        cat_6: "bf16[16, 512, 13, 13][86528, 169, 13, 1]cuda:0" = torch.ops.aten.cat.default([relu_20, relu_21], 1);  relu_20 = relu_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:30 in forward, code: x = self.squeeze_activation(self.squeeze(x))
        convolution_22: "bf16[16, 64, 13, 13][10816, 169, 13, 1]cuda:0" = torch.ops.aten.convolution.default(cat_6, arg45_1, arg46_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_6 = arg45_1 = arg46_1 = None
        relu_22: "bf16[16, 64, 13, 13][10816, 169, 13, 1]cuda:0" = torch.ops.aten.relu.default(convolution_22);  convolution_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:32 in forward, code: [self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1
        convolution_23: "bf16[16, 256, 13, 13][43264, 169, 13, 1]cuda:0" = torch.ops.aten.convolution.default(relu_22, arg47_1, arg48_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg47_1 = arg48_1 = None
        relu_23: "bf16[16, 256, 13, 13][43264, 169, 13, 1]cuda:0" = torch.ops.aten.relu.default(convolution_23);  convolution_23 = None
        convolution_24: "bf16[16, 256, 13, 13][43264, 169, 13, 1]cuda:0" = torch.ops.aten.convolution.default(relu_22, arg49_1, arg50_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_22 = arg49_1 = arg50_1 = None
        relu_24: "bf16[16, 256, 13, 13][43264, 169, 13, 1]cuda:0" = torch.ops.aten.relu.default(convolution_24);  convolution_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:31 in forward, code: return torch.cat(
        cat_7: "bf16[16, 512, 13, 13][86528, 169, 13, 1]cuda:0" = torch.ops.aten.cat.default([relu_23, relu_24], 1);  relu_23 = relu_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:96 in forward, code: x = self.classifier(x)
        convolution_25: "bf16[16, 1000, 13, 13][169000, 169, 13, 1]cuda:0" = torch.ops.aten.convolution.default(cat_7, arg51_1, arg52_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_7 = arg51_1 = arg52_1 = None
        relu_25: "bf16[16, 1000, 13, 13][169000, 169, 13, 1]cuda:0" = torch.ops.aten.relu.default(convolution_25);  convolution_25 = None
        mean: "bf16[16, 1000, 1, 1][1000, 1, 1, 1]cuda:0" = torch.ops.aten.mean.dim(relu_25, [-1, -2], True);  relu_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:97 in forward, code: return torch.flatten(x, 1)
        view: "bf16[16, 1000][1000, 1]cuda:0" = torch.ops.aten.reshape.default(mean, [16, 1000]);  mean = None
        return (view,)
