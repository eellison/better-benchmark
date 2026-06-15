class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[64, 3, 3, 3][27, 9, 3, 1]cuda:0", primals_2: "f32[64][1]cuda:0", primals_3: "f32[32, 3, 224, 224][150528, 50176, 224, 1]cuda:0", primals_4: "f32[16, 64, 1, 1][64, 1, 1, 1]cuda:0", primals_5: "f32[16][1]cuda:0", primals_6: "f32[64, 16, 1, 1][16, 1, 1, 1]cuda:0", primals_7: "f32[64][1]cuda:0", primals_8: "f32[64, 16, 3, 3][144, 9, 3, 1]cuda:0", primals_9: "f32[64][1]cuda:0", primals_10: "f32[16, 128, 1, 1][128, 1, 1, 1]cuda:0", primals_11: "f32[16][1]cuda:0", primals_12: "f32[64, 16, 1, 1][16, 1, 1, 1]cuda:0", primals_13: "f32[64][1]cuda:0", primals_14: "f32[64, 16, 3, 3][144, 9, 3, 1]cuda:0", primals_15: "f32[64][1]cuda:0", primals_16: "f32[32, 128, 1, 1][128, 1, 1, 1]cuda:0", primals_17: "f32[32][1]cuda:0", primals_18: "f32[128, 32, 1, 1][32, 1, 1, 1]cuda:0", primals_19: "f32[128][1]cuda:0", primals_20: "f32[128, 32, 3, 3][288, 9, 3, 1]cuda:0", primals_21: "f32[128][1]cuda:0", primals_22: "f32[32, 256, 1, 1][256, 1, 1, 1]cuda:0", primals_23: "f32[32][1]cuda:0", primals_24: "f32[128, 32, 1, 1][32, 1, 1, 1]cuda:0", primals_25: "f32[128][1]cuda:0", primals_26: "f32[128, 32, 3, 3][288, 9, 3, 1]cuda:0", primals_27: "f32[128][1]cuda:0", primals_28: "f32[48, 256, 1, 1][256, 1, 1, 1]cuda:0", primals_29: "f32[48][1]cuda:0", primals_30: "f32[192, 48, 1, 1][48, 1, 1, 1]cuda:0", primals_31: "f32[192][1]cuda:0", primals_32: "f32[192, 48, 3, 3][432, 9, 3, 1]cuda:0", primals_33: "f32[192][1]cuda:0", primals_34: "f32[48, 384, 1, 1][384, 1, 1, 1]cuda:0", primals_35: "f32[48][1]cuda:0", primals_36: "f32[192, 48, 1, 1][48, 1, 1, 1]cuda:0", primals_37: "f32[192][1]cuda:0", primals_38: "f32[192, 48, 3, 3][432, 9, 3, 1]cuda:0", primals_39: "f32[192][1]cuda:0", primals_40: "f32[64, 384, 1, 1][384, 1, 1, 1]cuda:0", primals_41: "f32[64][1]cuda:0", primals_42: "f32[256, 64, 1, 1][64, 1, 1, 1]cuda:0", primals_43: "f32[256][1]cuda:0", primals_44: "f32[256, 64, 3, 3][576, 9, 3, 1]cuda:0", primals_45: "f32[256][1]cuda:0", primals_46: "f32[64, 512, 1, 1][512, 1, 1, 1]cuda:0", primals_47: "f32[64][1]cuda:0", primals_48: "f32[256, 64, 1, 1][64, 1, 1, 1]cuda:0", primals_49: "f32[256][1]cuda:0", primals_50: "f32[256, 64, 3, 3][576, 9, 3, 1]cuda:0", primals_51: "f32[256][1]cuda:0", primals_52: "f32[1000, 512, 1, 1][512, 1, 1, 1]cuda:0", primals_53: "f32[1000][1]cuda:0"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:95 in forward, code: x = self.features(x)
        convert_element_type: "bf16[64][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_2, torch.bfloat16);  primals_2 = None
        convert_element_type_1: "bf16[64, 3, 3, 3][27, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_1, torch.bfloat16);  primals_1 = None
        convert_element_type_2: "bf16[32, 3, 224, 224][150528, 50176, 224, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_3, torch.bfloat16);  primals_3 = None
        convolution: "bf16[32, 64, 111, 111][788544, 12321, 111, 1]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_2, convert_element_type_1, convert_element_type, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type = None
        relu: "bf16[32, 64, 111, 111][788544, 12321, 111, 1]cuda:0" = torch.ops.aten.relu.default(convolution);  convolution = None
        _low_memory_max_pool_with_offsets = torch.ops.prims._low_memory_max_pool_with_offsets.default(relu, [3, 3], [2, 2], [0, 0], [1, 1], True)
        getitem: "bf16[32, 64, 55, 55][193600, 3025, 55, 1]cuda:0" = _low_memory_max_pool_with_offsets[0]
        getitem_1: "i8[32, 64, 55, 55][193600, 3025, 55, 1]cuda:0" = _low_memory_max_pool_with_offsets[1];  _low_memory_max_pool_with_offsets = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:30 in forward, code: x = self.squeeze_activation(self.squeeze(x))
        convert_element_type_3: "bf16[16][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_5, torch.bfloat16);  primals_5 = None
        convert_element_type_4: "bf16[16, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_4, torch.bfloat16);  primals_4 = None
        convolution_1: "bf16[32, 16, 55, 55][48400, 3025, 55, 1]cuda:0" = torch.ops.aten.convolution.default(getitem, convert_element_type_4, convert_element_type_3, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_3 = None
        relu_1: "bf16[32, 16, 55, 55][48400, 3025, 55, 1]cuda:0" = torch.ops.aten.relu.default(convolution_1);  convolution_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:32 in forward, code: [self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1
        convert_element_type_5: "bf16[64][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_7, torch.bfloat16);  primals_7 = None
        convert_element_type_6: "bf16[64, 16, 1, 1][16, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_6, torch.bfloat16);  primals_6 = None
        convolution_2: "bf16[32, 64, 55, 55][193600, 3025, 55, 1]cuda:0" = torch.ops.aten.convolution.default(relu_1, convert_element_type_6, convert_element_type_5, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_5 = None
        relu_2: "bf16[32, 64, 55, 55][193600, 3025, 55, 1]cuda:0" = torch.ops.aten.relu.default(convolution_2);  convolution_2 = None
        convert_element_type_7: "bf16[64][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_9, torch.bfloat16);  primals_9 = None
        convert_element_type_8: "bf16[64, 16, 3, 3][144, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_8, torch.bfloat16);  primals_8 = None
        convolution_3: "bf16[32, 64, 55, 55][193600, 3025, 55, 1]cuda:0" = torch.ops.aten.convolution.default(relu_1, convert_element_type_8, convert_element_type_7, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  convert_element_type_7 = None
        relu_3: "bf16[32, 64, 55, 55][193600, 3025, 55, 1]cuda:0" = torch.ops.aten.relu.default(convolution_3);  convolution_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:31 in forward, code: return torch.cat(
        cat: "bf16[32, 128, 55, 55][387200, 3025, 55, 1]cuda:0" = torch.ops.aten.cat.default([relu_2, relu_3], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:30 in forward, code: x = self.squeeze_activation(self.squeeze(x))
        convert_element_type_9: "bf16[16][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_11, torch.bfloat16);  primals_11 = None
        convert_element_type_10: "bf16[16, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_10, torch.bfloat16);  primals_10 = None
        convolution_4: "bf16[32, 16, 55, 55][48400, 3025, 55, 1]cuda:0" = torch.ops.aten.convolution.default(cat, convert_element_type_10, convert_element_type_9, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_9 = None
        relu_4: "bf16[32, 16, 55, 55][48400, 3025, 55, 1]cuda:0" = torch.ops.aten.relu.default(convolution_4);  convolution_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:32 in forward, code: [self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1
        convert_element_type_11: "bf16[64][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_13, torch.bfloat16);  primals_13 = None
        convert_element_type_12: "bf16[64, 16, 1, 1][16, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_12, torch.bfloat16);  primals_12 = None
        convolution_5: "bf16[32, 64, 55, 55][193600, 3025, 55, 1]cuda:0" = torch.ops.aten.convolution.default(relu_4, convert_element_type_12, convert_element_type_11, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_11 = None
        relu_5: "bf16[32, 64, 55, 55][193600, 3025, 55, 1]cuda:0" = torch.ops.aten.relu.default(convolution_5);  convolution_5 = None
        convert_element_type_13: "bf16[64][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_15, torch.bfloat16);  primals_15 = None
        convert_element_type_14: "bf16[64, 16, 3, 3][144, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_14, torch.bfloat16);  primals_14 = None
        convolution_6: "bf16[32, 64, 55, 55][193600, 3025, 55, 1]cuda:0" = torch.ops.aten.convolution.default(relu_4, convert_element_type_14, convert_element_type_13, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  convert_element_type_13 = None
        relu_6: "bf16[32, 64, 55, 55][193600, 3025, 55, 1]cuda:0" = torch.ops.aten.relu.default(convolution_6);  convolution_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:31 in forward, code: return torch.cat(
        cat_1: "bf16[32, 128, 55, 55][387200, 3025, 55, 1]cuda:0" = torch.ops.aten.cat.default([relu_5, relu_6], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:95 in forward, code: x = self.features(x)
        _low_memory_max_pool_with_offsets_1 = torch.ops.prims._low_memory_max_pool_with_offsets.default(cat_1, [3, 3], [2, 2], [0, 0], [1, 1], True);  cat_1 = None
        getitem_2: "bf16[32, 128, 27, 27][93312, 729, 27, 1]cuda:0" = _low_memory_max_pool_with_offsets_1[0]
        getitem_3: "i8[32, 128, 27, 27][93312, 729, 27, 1]cuda:0" = _low_memory_max_pool_with_offsets_1[1];  _low_memory_max_pool_with_offsets_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:30 in forward, code: x = self.squeeze_activation(self.squeeze(x))
        convert_element_type_15: "bf16[32][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_17, torch.bfloat16);  primals_17 = None
        convert_element_type_16: "bf16[32, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_16, torch.bfloat16);  primals_16 = None
        convolution_7: "bf16[32, 32, 27, 27][23328, 729, 27, 1]cuda:0" = torch.ops.aten.convolution.default(getitem_2, convert_element_type_16, convert_element_type_15, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_15 = None
        relu_7: "bf16[32, 32, 27, 27][23328, 729, 27, 1]cuda:0" = torch.ops.aten.relu.default(convolution_7);  convolution_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:32 in forward, code: [self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1
        convert_element_type_17: "bf16[128][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_19, torch.bfloat16);  primals_19 = None
        convert_element_type_18: "bf16[128, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_18, torch.bfloat16);  primals_18 = None
        convolution_8: "bf16[32, 128, 27, 27][93312, 729, 27, 1]cuda:0" = torch.ops.aten.convolution.default(relu_7, convert_element_type_18, convert_element_type_17, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_17 = None
        relu_8: "bf16[32, 128, 27, 27][93312, 729, 27, 1]cuda:0" = torch.ops.aten.relu.default(convolution_8);  convolution_8 = None
        convert_element_type_19: "bf16[128][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_21, torch.bfloat16);  primals_21 = None
        convert_element_type_20: "bf16[128, 32, 3, 3][288, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_20, torch.bfloat16);  primals_20 = None
        convolution_9: "bf16[32, 128, 27, 27][93312, 729, 27, 1]cuda:0" = torch.ops.aten.convolution.default(relu_7, convert_element_type_20, convert_element_type_19, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  convert_element_type_19 = None
        relu_9: "bf16[32, 128, 27, 27][93312, 729, 27, 1]cuda:0" = torch.ops.aten.relu.default(convolution_9);  convolution_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:31 in forward, code: return torch.cat(
        cat_2: "bf16[32, 256, 27, 27][186624, 729, 27, 1]cuda:0" = torch.ops.aten.cat.default([relu_8, relu_9], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:30 in forward, code: x = self.squeeze_activation(self.squeeze(x))
        convert_element_type_21: "bf16[32][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_23, torch.bfloat16);  primals_23 = None
        convert_element_type_22: "bf16[32, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_22, torch.bfloat16);  primals_22 = None
        convolution_10: "bf16[32, 32, 27, 27][23328, 729, 27, 1]cuda:0" = torch.ops.aten.convolution.default(cat_2, convert_element_type_22, convert_element_type_21, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_21 = None
        relu_10: "bf16[32, 32, 27, 27][23328, 729, 27, 1]cuda:0" = torch.ops.aten.relu.default(convolution_10);  convolution_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:32 in forward, code: [self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1
        convert_element_type_23: "bf16[128][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_25, torch.bfloat16);  primals_25 = None
        convert_element_type_24: "bf16[128, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_24, torch.bfloat16);  primals_24 = None
        convolution_11: "bf16[32, 128, 27, 27][93312, 729, 27, 1]cuda:0" = torch.ops.aten.convolution.default(relu_10, convert_element_type_24, convert_element_type_23, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_23 = None
        relu_11: "bf16[32, 128, 27, 27][93312, 729, 27, 1]cuda:0" = torch.ops.aten.relu.default(convolution_11);  convolution_11 = None
        convert_element_type_25: "bf16[128][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_27, torch.bfloat16);  primals_27 = None
        convert_element_type_26: "bf16[128, 32, 3, 3][288, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_26, torch.bfloat16);  primals_26 = None
        convolution_12: "bf16[32, 128, 27, 27][93312, 729, 27, 1]cuda:0" = torch.ops.aten.convolution.default(relu_10, convert_element_type_26, convert_element_type_25, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  convert_element_type_25 = None
        relu_12: "bf16[32, 128, 27, 27][93312, 729, 27, 1]cuda:0" = torch.ops.aten.relu.default(convolution_12);  convolution_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:31 in forward, code: return torch.cat(
        cat_3: "bf16[32, 256, 27, 27][186624, 729, 27, 1]cuda:0" = torch.ops.aten.cat.default([relu_11, relu_12], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:95 in forward, code: x = self.features(x)
        _low_memory_max_pool_with_offsets_2 = torch.ops.prims._low_memory_max_pool_with_offsets.default(cat_3, [3, 3], [2, 2], [0, 0], [1, 1], True);  cat_3 = None
        getitem_4: "bf16[32, 256, 13, 13][43264, 169, 13, 1]cuda:0" = _low_memory_max_pool_with_offsets_2[0]
        getitem_5: "i8[32, 256, 13, 13][43264, 169, 13, 1]cuda:0" = _low_memory_max_pool_with_offsets_2[1];  _low_memory_max_pool_with_offsets_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:30 in forward, code: x = self.squeeze_activation(self.squeeze(x))
        convert_element_type_27: "bf16[48][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_29, torch.bfloat16);  primals_29 = None
        convert_element_type_28: "bf16[48, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_28, torch.bfloat16);  primals_28 = None
        convolution_13: "bf16[32, 48, 13, 13][8112, 169, 13, 1]cuda:0" = torch.ops.aten.convolution.default(getitem_4, convert_element_type_28, convert_element_type_27, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_27 = None
        relu_13: "bf16[32, 48, 13, 13][8112, 169, 13, 1]cuda:0" = torch.ops.aten.relu.default(convolution_13);  convolution_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:32 in forward, code: [self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1
        convert_element_type_29: "bf16[192][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_31, torch.bfloat16);  primals_31 = None
        convert_element_type_30: "bf16[192, 48, 1, 1][48, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_30, torch.bfloat16);  primals_30 = None
        convolution_14: "bf16[32, 192, 13, 13][32448, 169, 13, 1]cuda:0" = torch.ops.aten.convolution.default(relu_13, convert_element_type_30, convert_element_type_29, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_29 = None
        relu_14: "bf16[32, 192, 13, 13][32448, 169, 13, 1]cuda:0" = torch.ops.aten.relu.default(convolution_14);  convolution_14 = None
        convert_element_type_31: "bf16[192][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_33, torch.bfloat16);  primals_33 = None
        convert_element_type_32: "bf16[192, 48, 3, 3][432, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_32, torch.bfloat16);  primals_32 = None
        convolution_15: "bf16[32, 192, 13, 13][32448, 169, 13, 1]cuda:0" = torch.ops.aten.convolution.default(relu_13, convert_element_type_32, convert_element_type_31, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  convert_element_type_31 = None
        relu_15: "bf16[32, 192, 13, 13][32448, 169, 13, 1]cuda:0" = torch.ops.aten.relu.default(convolution_15);  convolution_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:31 in forward, code: return torch.cat(
        cat_4: "bf16[32, 384, 13, 13][64896, 169, 13, 1]cuda:0" = torch.ops.aten.cat.default([relu_14, relu_15], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:30 in forward, code: x = self.squeeze_activation(self.squeeze(x))
        convert_element_type_33: "bf16[48][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_35, torch.bfloat16);  primals_35 = None
        convert_element_type_34: "bf16[48, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_34, torch.bfloat16);  primals_34 = None
        convolution_16: "bf16[32, 48, 13, 13][8112, 169, 13, 1]cuda:0" = torch.ops.aten.convolution.default(cat_4, convert_element_type_34, convert_element_type_33, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_33 = None
        relu_16: "bf16[32, 48, 13, 13][8112, 169, 13, 1]cuda:0" = torch.ops.aten.relu.default(convolution_16);  convolution_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:32 in forward, code: [self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1
        convert_element_type_35: "bf16[192][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_37, torch.bfloat16);  primals_37 = None
        convert_element_type_36: "bf16[192, 48, 1, 1][48, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_36, torch.bfloat16);  primals_36 = None
        convolution_17: "bf16[32, 192, 13, 13][32448, 169, 13, 1]cuda:0" = torch.ops.aten.convolution.default(relu_16, convert_element_type_36, convert_element_type_35, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_35 = None
        relu_17: "bf16[32, 192, 13, 13][32448, 169, 13, 1]cuda:0" = torch.ops.aten.relu.default(convolution_17);  convolution_17 = None
        convert_element_type_37: "bf16[192][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_39, torch.bfloat16);  primals_39 = None
        convert_element_type_38: "bf16[192, 48, 3, 3][432, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_38, torch.bfloat16);  primals_38 = None
        convolution_18: "bf16[32, 192, 13, 13][32448, 169, 13, 1]cuda:0" = torch.ops.aten.convolution.default(relu_16, convert_element_type_38, convert_element_type_37, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  convert_element_type_37 = None
        relu_18: "bf16[32, 192, 13, 13][32448, 169, 13, 1]cuda:0" = torch.ops.aten.relu.default(convolution_18);  convolution_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:31 in forward, code: return torch.cat(
        cat_5: "bf16[32, 384, 13, 13][64896, 169, 13, 1]cuda:0" = torch.ops.aten.cat.default([relu_17, relu_18], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:30 in forward, code: x = self.squeeze_activation(self.squeeze(x))
        convert_element_type_39: "bf16[64][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_41, torch.bfloat16);  primals_41 = None
        convert_element_type_40: "bf16[64, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_40, torch.bfloat16);  primals_40 = None
        convolution_19: "bf16[32, 64, 13, 13][10816, 169, 13, 1]cuda:0" = torch.ops.aten.convolution.default(cat_5, convert_element_type_40, convert_element_type_39, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_39 = None
        relu_19: "bf16[32, 64, 13, 13][10816, 169, 13, 1]cuda:0" = torch.ops.aten.relu.default(convolution_19);  convolution_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:32 in forward, code: [self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1
        convert_element_type_41: "bf16[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_43, torch.bfloat16);  primals_43 = None
        convert_element_type_42: "bf16[256, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_42, torch.bfloat16);  primals_42 = None
        convolution_20: "bf16[32, 256, 13, 13][43264, 169, 13, 1]cuda:0" = torch.ops.aten.convolution.default(relu_19, convert_element_type_42, convert_element_type_41, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_41 = None
        relu_20: "bf16[32, 256, 13, 13][43264, 169, 13, 1]cuda:0" = torch.ops.aten.relu.default(convolution_20);  convolution_20 = None
        convert_element_type_43: "bf16[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_45, torch.bfloat16);  primals_45 = None
        convert_element_type_44: "bf16[256, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_44, torch.bfloat16);  primals_44 = None
        convolution_21: "bf16[32, 256, 13, 13][43264, 169, 13, 1]cuda:0" = torch.ops.aten.convolution.default(relu_19, convert_element_type_44, convert_element_type_43, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  convert_element_type_43 = None
        relu_21: "bf16[32, 256, 13, 13][43264, 169, 13, 1]cuda:0" = torch.ops.aten.relu.default(convolution_21);  convolution_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:31 in forward, code: return torch.cat(
        cat_6: "bf16[32, 512, 13, 13][86528, 169, 13, 1]cuda:0" = torch.ops.aten.cat.default([relu_20, relu_21], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:30 in forward, code: x = self.squeeze_activation(self.squeeze(x))
        convert_element_type_45: "bf16[64][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_47, torch.bfloat16);  primals_47 = None
        convert_element_type_46: "bf16[64, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_46, torch.bfloat16);  primals_46 = None
        convolution_22: "bf16[32, 64, 13, 13][10816, 169, 13, 1]cuda:0" = torch.ops.aten.convolution.default(cat_6, convert_element_type_46, convert_element_type_45, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_45 = None
        relu_22: "bf16[32, 64, 13, 13][10816, 169, 13, 1]cuda:0" = torch.ops.aten.relu.default(convolution_22);  convolution_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:32 in forward, code: [self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1
        convert_element_type_47: "bf16[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_49, torch.bfloat16);  primals_49 = None
        convert_element_type_48: "bf16[256, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_48, torch.bfloat16);  primals_48 = None
        convolution_23: "bf16[32, 256, 13, 13][43264, 169, 13, 1]cuda:0" = torch.ops.aten.convolution.default(relu_22, convert_element_type_48, convert_element_type_47, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_47 = None
        relu_23: "bf16[32, 256, 13, 13][43264, 169, 13, 1]cuda:0" = torch.ops.aten.relu.default(convolution_23);  convolution_23 = None
        convert_element_type_49: "bf16[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_51, torch.bfloat16);  primals_51 = None
        convert_element_type_50: "bf16[256, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_50, torch.bfloat16);  primals_50 = None
        convolution_24: "bf16[32, 256, 13, 13][43264, 169, 13, 1]cuda:0" = torch.ops.aten.convolution.default(relu_22, convert_element_type_50, convert_element_type_49, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  convert_element_type_49 = None
        relu_24: "bf16[32, 256, 13, 13][43264, 169, 13, 1]cuda:0" = torch.ops.aten.relu.default(convolution_24);  convolution_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:31 in forward, code: return torch.cat(
        cat_7: "bf16[32, 512, 13, 13][86528, 169, 13, 1]cuda:0" = torch.ops.aten.cat.default([relu_23, relu_24], 1)

        # No stacktrace found for following nodes
        inductor_seeds_default: "i64[1][1]cuda:0" = torch.ops.prims.inductor_seeds.default(1, device(type='cuda', index=0))

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:96 in forward, code: x = self.classifier(x)
        inductor_lookup_seed_default: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 0);  inductor_seeds_default = None
        inductor_random_default: "f32[32, 512, 13, 13][86528, 169, 13, 1]cuda:0" = torch.ops.prims.inductor_random.default([32, 512, 13, 13], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        convert_element_type_default: "bf16[32, 512, 13, 13][86528, 169, 13, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default, torch.bfloat16);  inductor_random_default = None
        gt: "b8[32, 512, 13, 13][86528, 169, 13, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default, 0.5);  convert_element_type_default = None
        mul: "bf16[32, 512, 13, 13][86528, 169, 13, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt, cat_7);  cat_7 = None
        mul_1: "bf16[32, 512, 13, 13][86528, 169, 13, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul, 2.0);  mul = None
        convert_element_type_51: "bf16[1000][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_53, torch.bfloat16);  primals_53 = None
        convert_element_type_52: "bf16[1000, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_52, torch.bfloat16);  primals_52 = None
        convolution_25: "bf16[32, 1000, 13, 13][169000, 169, 13, 1]cuda:0" = torch.ops.aten.convolution.default(mul_1, convert_element_type_52, convert_element_type_51, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_51 = None
        relu_25: "bf16[32, 1000, 13, 13][169000, 169, 13, 1]cuda:0" = torch.ops.aten.relu.default(convolution_25);  convolution_25 = None
        mean: "bf16[32, 1000, 1, 1][1000, 1, 1, 1]cuda:0" = torch.ops.aten.mean.dim(relu_25, [-1, -2], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:97 in forward, code: return torch.flatten(x, 1)
        view: "bf16[32, 1000][1000, 1]cuda:0" = torch.ops.aten.reshape.default(mean, [32, 1000]);  mean = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:96 in forward, code: x = self.classifier(x)
        le: "b8[32, 1000, 13, 13][169000, 169, 13, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_25, 0);  relu_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:32 in forward, code: [self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1
        le_1: "b8[32, 256, 13, 13][43264, 169, 13, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_24, 0);  relu_24 = None
        le_2: "b8[32, 256, 13, 13][43264, 169, 13, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_23, 0);  relu_23 = None
        le_4: "b8[32, 256, 13, 13][43264, 169, 13, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_21, 0);  relu_21 = None
        le_5: "b8[32, 256, 13, 13][43264, 169, 13, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_20, 0);  relu_20 = None
        le_7: "b8[32, 192, 13, 13][32448, 169, 13, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_18, 0);  relu_18 = None
        le_8: "b8[32, 192, 13, 13][32448, 169, 13, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_17, 0);  relu_17 = None
        le_10: "b8[32, 192, 13, 13][32448, 169, 13, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_15, 0);  relu_15 = None
        le_11: "b8[32, 192, 13, 13][32448, 169, 13, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_14, 0);  relu_14 = None
        le_13: "b8[32, 128, 27, 27][93312, 729, 27, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_12, 0);  relu_12 = None
        le_14: "b8[32, 128, 27, 27][93312, 729, 27, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_11, 0);  relu_11 = None
        le_16: "b8[32, 128, 27, 27][93312, 729, 27, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_9, 0);  relu_9 = None
        le_17: "b8[32, 128, 27, 27][93312, 729, 27, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_8, 0);  relu_8 = None
        le_19: "b8[32, 64, 55, 55][193600, 3025, 55, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_6, 0);  relu_6 = None
        le_20: "b8[32, 64, 55, 55][193600, 3025, 55, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_5, 0);  relu_5 = None
        le_22: "b8[32, 64, 55, 55][193600, 3025, 55, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_3, 0);  relu_3 = None
        le_23: "b8[32, 64, 55, 55][193600, 3025, 55, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_2, 0);  relu_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:95 in forward, code: x = self.features(x)
        le_25: "b8[32, 64, 111, 111][788544, 12321, 111, 1]cuda:0" = torch.ops.aten.le.Scalar(relu, 0);  relu = None
        return (view, convert_element_type_1, convert_element_type_2, getitem, getitem_1, convert_element_type_4, relu_1, convert_element_type_6, convert_element_type_8, cat, convert_element_type_10, relu_4, convert_element_type_12, convert_element_type_14, getitem_2, getitem_3, convert_element_type_16, relu_7, convert_element_type_18, convert_element_type_20, cat_2, convert_element_type_22, relu_10, convert_element_type_24, convert_element_type_26, getitem_4, getitem_5, convert_element_type_28, relu_13, convert_element_type_30, convert_element_type_32, cat_4, convert_element_type_34, relu_16, convert_element_type_36, convert_element_type_38, cat_5, convert_element_type_40, relu_19, convert_element_type_42, convert_element_type_44, cat_6, convert_element_type_46, relu_22, convert_element_type_48, convert_element_type_50, gt, mul_1, convert_element_type_52, le, le_1, le_2, le_4, le_5, le_7, le_8, le_10, le_11, le_13, le_14, le_16, le_17, le_19, le_20, le_22, le_23, le_25)
