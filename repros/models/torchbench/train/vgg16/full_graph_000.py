class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[64, 3, 3, 3][27, 9, 3, 1]cuda:0", primals_2: "f32[64][1]cuda:0", primals_3: "f32[64, 3, 224, 224][150528, 50176, 224, 1]cuda:0", primals_4: "f32[64, 64, 3, 3][576, 9, 3, 1]cuda:0", primals_5: "f32[64][1]cuda:0", primals_6: "f32[128, 64, 3, 3][576, 9, 3, 1]cuda:0", primals_7: "f32[128][1]cuda:0", primals_8: "f32[128, 128, 3, 3][1152, 9, 3, 1]cuda:0", primals_9: "f32[128][1]cuda:0", primals_10: "f32[256, 128, 3, 3][1152, 9, 3, 1]cuda:0", primals_11: "f32[256][1]cuda:0", primals_12: "f32[256, 256, 3, 3][2304, 9, 3, 1]cuda:0", primals_13: "f32[256][1]cuda:0", primals_14: "f32[256, 256, 3, 3][2304, 9, 3, 1]cuda:0", primals_15: "f32[256][1]cuda:0", primals_16: "f32[512, 256, 3, 3][2304, 9, 3, 1]cuda:0", primals_17: "f32[512][1]cuda:0", primals_18: "f32[512, 512, 3, 3][4608, 9, 3, 1]cuda:0", primals_19: "f32[512][1]cuda:0", primals_20: "f32[512, 512, 3, 3][4608, 9, 3, 1]cuda:0", primals_21: "f32[512][1]cuda:0", primals_22: "f32[512, 512, 3, 3][4608, 9, 3, 1]cuda:0", primals_23: "f32[512][1]cuda:0", primals_24: "f32[512, 512, 3, 3][4608, 9, 3, 1]cuda:0", primals_25: "f32[512][1]cuda:0", primals_26: "f32[512, 512, 3, 3][4608, 9, 3, 1]cuda:0", primals_27: "f32[512][1]cuda:0", primals_28: "f32[4096, 25088][25088, 1]cuda:0", primals_29: "f32[4096][1]cuda:0", primals_30: "f32[4096, 4096][4096, 1]cuda:0", primals_31: "f32[4096][1]cuda:0", primals_32: "f32[1000, 4096][4096, 1]cuda:0", primals_33: "f32[1000][1]cuda:0"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vgg.py:66 in forward, code: x = self.features(x)
        convert_element_type: "bf16[64][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_2, torch.bfloat16);  primals_2 = None
        convert_element_type_1: "bf16[64, 3, 3, 3][27, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_1, torch.bfloat16);  primals_1 = None
        convert_element_type_2: "bf16[64, 3, 224, 224][150528, 50176, 224, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_3, torch.bfloat16);  primals_3 = None
        convolution: "bf16[64, 64, 224, 224][3211264, 50176, 224, 1]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_2, convert_element_type_1, convert_element_type, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  convert_element_type = None
        relu: "bf16[64, 64, 224, 224][3211264, 50176, 224, 1]cuda:0" = torch.ops.aten.relu.default(convolution);  convolution = None
        convert_element_type_3: "bf16[64][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_5, torch.bfloat16);  primals_5 = None
        convert_element_type_4: "bf16[64, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_4, torch.bfloat16);  primals_4 = None
        convolution_1: "bf16[64, 64, 224, 224][3211264, 50176, 224, 1]cuda:0" = torch.ops.aten.convolution.default(relu, convert_element_type_4, convert_element_type_3, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  convert_element_type_3 = None
        relu_1: "bf16[64, 64, 224, 224][3211264, 50176, 224, 1]cuda:0" = torch.ops.aten.relu.default(convolution_1);  convolution_1 = None
        _low_memory_max_pool_with_offsets = torch.ops.prims._low_memory_max_pool_with_offsets.default(relu_1, [2, 2], [2, 2], [0, 0], [1, 1], False)
        getitem: "bf16[64, 64, 112, 112][802816, 12544, 112, 1]cuda:0" = _low_memory_max_pool_with_offsets[0]
        getitem_1: "i8[64, 64, 112, 112][802816, 12544, 112, 1]cuda:0" = _low_memory_max_pool_with_offsets[1];  _low_memory_max_pool_with_offsets = None
        convert_element_type_5: "bf16[128][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_7, torch.bfloat16);  primals_7 = None
        convert_element_type_6: "bf16[128, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_6, torch.bfloat16);  primals_6 = None
        convolution_2: "bf16[64, 128, 112, 112][1605632, 12544, 112, 1]cuda:0" = torch.ops.aten.convolution.default(getitem, convert_element_type_6, convert_element_type_5, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  convert_element_type_5 = None
        relu_2: "bf16[64, 128, 112, 112][1605632, 12544, 112, 1]cuda:0" = torch.ops.aten.relu.default(convolution_2);  convolution_2 = None
        convert_element_type_7: "bf16[128][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_9, torch.bfloat16);  primals_9 = None
        convert_element_type_8: "bf16[128, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_8, torch.bfloat16);  primals_8 = None
        convolution_3: "bf16[64, 128, 112, 112][1605632, 12544, 112, 1]cuda:0" = torch.ops.aten.convolution.default(relu_2, convert_element_type_8, convert_element_type_7, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  convert_element_type_7 = None
        relu_3: "bf16[64, 128, 112, 112][1605632, 12544, 112, 1]cuda:0" = torch.ops.aten.relu.default(convolution_3);  convolution_3 = None
        _low_memory_max_pool_with_offsets_1 = torch.ops.prims._low_memory_max_pool_with_offsets.default(relu_3, [2, 2], [2, 2], [0, 0], [1, 1], False)
        getitem_2: "bf16[64, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = _low_memory_max_pool_with_offsets_1[0]
        getitem_3: "i8[64, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = _low_memory_max_pool_with_offsets_1[1];  _low_memory_max_pool_with_offsets_1 = None
        convert_element_type_9: "bf16[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_11, torch.bfloat16);  primals_11 = None
        convert_element_type_10: "bf16[256, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_10, torch.bfloat16);  primals_10 = None
        convolution_4: "bf16[64, 256, 56, 56][802816, 3136, 56, 1]cuda:0" = torch.ops.aten.convolution.default(getitem_2, convert_element_type_10, convert_element_type_9, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  convert_element_type_9 = None
        relu_4: "bf16[64, 256, 56, 56][802816, 3136, 56, 1]cuda:0" = torch.ops.aten.relu.default(convolution_4);  convolution_4 = None
        convert_element_type_11: "bf16[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_13, torch.bfloat16);  primals_13 = None
        convert_element_type_12: "bf16[256, 256, 3, 3][2304, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_12, torch.bfloat16);  primals_12 = None
        convolution_5: "bf16[64, 256, 56, 56][802816, 3136, 56, 1]cuda:0" = torch.ops.aten.convolution.default(relu_4, convert_element_type_12, convert_element_type_11, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  convert_element_type_11 = None
        relu_5: "bf16[64, 256, 56, 56][802816, 3136, 56, 1]cuda:0" = torch.ops.aten.relu.default(convolution_5);  convolution_5 = None
        convert_element_type_13: "bf16[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_15, torch.bfloat16);  primals_15 = None
        convert_element_type_14: "bf16[256, 256, 3, 3][2304, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_14, torch.bfloat16);  primals_14 = None
        convolution_6: "bf16[64, 256, 56, 56][802816, 3136, 56, 1]cuda:0" = torch.ops.aten.convolution.default(relu_5, convert_element_type_14, convert_element_type_13, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  convert_element_type_13 = None
        relu_6: "bf16[64, 256, 56, 56][802816, 3136, 56, 1]cuda:0" = torch.ops.aten.relu.default(convolution_6);  convolution_6 = None
        _low_memory_max_pool_with_offsets_2 = torch.ops.prims._low_memory_max_pool_with_offsets.default(relu_6, [2, 2], [2, 2], [0, 0], [1, 1], False)
        getitem_4: "bf16[64, 256, 28, 28][200704, 784, 28, 1]cuda:0" = _low_memory_max_pool_with_offsets_2[0]
        getitem_5: "i8[64, 256, 28, 28][200704, 784, 28, 1]cuda:0" = _low_memory_max_pool_with_offsets_2[1];  _low_memory_max_pool_with_offsets_2 = None
        convert_element_type_15: "bf16[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_17, torch.bfloat16);  primals_17 = None
        convert_element_type_16: "bf16[512, 256, 3, 3][2304, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_16, torch.bfloat16);  primals_16 = None
        convolution_7: "bf16[64, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.aten.convolution.default(getitem_4, convert_element_type_16, convert_element_type_15, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  convert_element_type_15 = None
        relu_7: "bf16[64, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.aten.relu.default(convolution_7);  convolution_7 = None
        convert_element_type_17: "bf16[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_19, torch.bfloat16);  primals_19 = None
        convert_element_type_18: "bf16[512, 512, 3, 3][4608, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_18, torch.bfloat16);  primals_18 = None
        convolution_8: "bf16[64, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.aten.convolution.default(relu_7, convert_element_type_18, convert_element_type_17, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  convert_element_type_17 = None
        relu_8: "bf16[64, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.aten.relu.default(convolution_8);  convolution_8 = None
        convert_element_type_19: "bf16[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_21, torch.bfloat16);  primals_21 = None
        convert_element_type_20: "bf16[512, 512, 3, 3][4608, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_20, torch.bfloat16);  primals_20 = None
        convolution_9: "bf16[64, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.aten.convolution.default(relu_8, convert_element_type_20, convert_element_type_19, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  convert_element_type_19 = None
        relu_9: "bf16[64, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.aten.relu.default(convolution_9);  convolution_9 = None
        _low_memory_max_pool_with_offsets_3 = torch.ops.prims._low_memory_max_pool_with_offsets.default(relu_9, [2, 2], [2, 2], [0, 0], [1, 1], False)
        getitem_6: "bf16[64, 512, 14, 14][100352, 196, 14, 1]cuda:0" = _low_memory_max_pool_with_offsets_3[0]
        getitem_7: "i8[64, 512, 14, 14][100352, 196, 14, 1]cuda:0" = _low_memory_max_pool_with_offsets_3[1];  _low_memory_max_pool_with_offsets_3 = None
        convert_element_type_21: "bf16[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_23, torch.bfloat16);  primals_23 = None
        convert_element_type_22: "bf16[512, 512, 3, 3][4608, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_22, torch.bfloat16);  primals_22 = None
        convolution_10: "bf16[64, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(getitem_6, convert_element_type_22, convert_element_type_21, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  convert_element_type_21 = None
        relu_10: "bf16[64, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.relu.default(convolution_10);  convolution_10 = None
        convert_element_type_23: "bf16[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_25, torch.bfloat16);  primals_25 = None
        convert_element_type_24: "bf16[512, 512, 3, 3][4608, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_24, torch.bfloat16);  primals_24 = None
        convolution_11: "bf16[64, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(relu_10, convert_element_type_24, convert_element_type_23, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  convert_element_type_23 = None
        relu_11: "bf16[64, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.relu.default(convolution_11);  convolution_11 = None
        convert_element_type_25: "bf16[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_27, torch.bfloat16);  primals_27 = None
        convert_element_type_26: "bf16[512, 512, 3, 3][4608, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_26, torch.bfloat16);  primals_26 = None
        convolution_12: "bf16[64, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(relu_11, convert_element_type_26, convert_element_type_25, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  convert_element_type_25 = None
        relu_12: "bf16[64, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.relu.default(convolution_12);  convolution_12 = None
        _low_memory_max_pool_with_offsets_4 = torch.ops.prims._low_memory_max_pool_with_offsets.default(relu_12, [2, 2], [2, 2], [0, 0], [1, 1], False)
        getitem_8: "bf16[64, 512, 7, 7][25088, 49, 7, 1]cuda:0" = _low_memory_max_pool_with_offsets_4[0]
        getitem_9: "i8[64, 512, 7, 7][25088, 49, 7, 1]cuda:0" = _low_memory_max_pool_with_offsets_4[1];  _low_memory_max_pool_with_offsets_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vgg.py:67 in forward, code: x = self.avgpool(x)
        _adaptive_avg_pool2d: "bf16[64, 512, 7, 7][25088, 49, 7, 1]cuda:0" = torch.ops.aten._adaptive_avg_pool2d.default(getitem_8, [7, 7])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vgg.py:68 in forward, code: x = torch.flatten(x, 1)
        view: "bf16[64, 25088][25088, 1]cuda:0" = torch.ops.aten.reshape.default(_adaptive_avg_pool2d, [64, 25088]);  _adaptive_avg_pool2d = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vgg.py:69 in forward, code: x = self.classifier(x)
        convert_element_type_27: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_29, torch.bfloat16);  primals_29 = None
        convert_element_type_28: "bf16[4096, 25088][25088, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_28, torch.bfloat16);  primals_28 = None
        permute: "bf16[25088, 4096][1, 25088]cuda:0" = torch.ops.aten.permute.default(convert_element_type_28, [1, 0]);  convert_element_type_28 = None
        addmm: "bf16[64, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_27, view, permute);  convert_element_type_27 = None
        relu_13: "bf16[64, 4096][4096, 1]cuda:0" = torch.ops.aten.relu.default(addmm);  addmm = None

        # No stacktrace found for following nodes
        inductor_seeds_default: "i64[2][1]cuda:0" = torch.ops.prims.inductor_seeds.default(2, device(type='cuda', index=0))

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vgg.py:69 in forward, code: x = self.classifier(x)
        inductor_lookup_seed_default: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 0)
        inductor_random_default_1: "f32[64, 4096][4096, 1]cuda:0" = torch.ops.prims.inductor_random.default([64, 4096], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        convert_element_type_default_1: "bf16[64, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_1, torch.bfloat16);  inductor_random_default_1 = None
        gt: "b8[64, 4096][4096, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_1, 0.5);  convert_element_type_default_1 = None
        mul: "bf16[64, 4096][4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt, relu_13)
        mul_1: "bf16[64, 4096][4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul, 2.0);  mul = None
        convert_element_type_32: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_31, torch.bfloat16);  primals_31 = None
        convert_element_type_33: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_30, torch.bfloat16);  primals_30 = None
        permute_1: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_33, [1, 0]);  convert_element_type_33 = None
        addmm_1: "bf16[64, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_32, mul_1, permute_1);  convert_element_type_32 = None
        relu_14: "bf16[64, 4096][4096, 1]cuda:0" = torch.ops.aten.relu.default(addmm_1);  addmm_1 = None
        inductor_lookup_seed_default_1: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 1);  inductor_seeds_default = None
        inductor_random_default: "f32[64, 4096][4096, 1]cuda:0" = torch.ops.prims.inductor_random.default([64, 4096], inductor_lookup_seed_default_1, 'rand');  inductor_lookup_seed_default_1 = None
        convert_element_type_default: "bf16[64, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default, torch.bfloat16);  inductor_random_default = None
        gt_1: "b8[64, 4096][4096, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default, 0.5);  convert_element_type_default = None
        mul_2: "bf16[64, 4096][4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_1, relu_14)
        mul_3: "bf16[64, 4096][4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2, 2.0);  mul_2 = None
        convert_element_type_37: "bf16[1000][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_33, torch.bfloat16);  primals_33 = None
        convert_element_type_38: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_32, torch.bfloat16);  primals_32 = None
        permute_2: "bf16[4096, 1000][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_38, [1, 0]);  convert_element_type_38 = None
        addmm_2: "bf16[64, 1000][1000, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_37, mul_3, permute_2);  convert_element_type_37 = None
        permute_3: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None
        le: "b8[64, 4096][4096, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_14, 0);  relu_14 = None
        permute_7: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_1, [1, 0]);  permute_1 = None
        le_1: "b8[64, 4096][4096, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_13, 0);  relu_13 = None
        permute_11: "bf16[4096, 25088][25088, 1]cuda:0" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vgg.py:66 in forward, code: x = self.features(x)
        le_2: "b8[64, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_12, 0);  relu_12 = None
        le_5: "b8[64, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_9, 0);  relu_9 = None
        le_8: "b8[64, 256, 56, 56][802816, 3136, 56, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_6, 0);  relu_6 = None
        le_11: "b8[64, 128, 112, 112][1605632, 12544, 112, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_3, 0);  relu_3 = None
        le_13: "b8[64, 64, 224, 224][3211264, 50176, 224, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_1, 0);  relu_1 = None
        return (addmm_2, convert_element_type_1, convert_element_type_2, relu, convert_element_type_4, getitem, getitem_1, convert_element_type_6, relu_2, convert_element_type_8, getitem_2, getitem_3, convert_element_type_10, relu_4, convert_element_type_12, relu_5, convert_element_type_14, getitem_4, getitem_5, convert_element_type_16, relu_7, convert_element_type_18, relu_8, convert_element_type_20, getitem_6, getitem_7, convert_element_type_22, relu_10, convert_element_type_24, relu_11, convert_element_type_26, getitem_8, getitem_9, view, gt, mul_1, gt_1, mul_3, permute_3, le, permute_7, le_1, permute_11, le_2, le_5, le_8, le_11, le_13)
