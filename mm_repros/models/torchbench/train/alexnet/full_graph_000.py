class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[64, 3, 11, 11][363, 121, 11, 1]cuda:0", primals_2: "f32[64][1]cuda:0", primals_3: "f32[128, 3, 224, 224][150528, 50176, 224, 1]cuda:0", primals_4: "f32[192, 64, 5, 5][1600, 25, 5, 1]cuda:0", primals_5: "f32[192][1]cuda:0", primals_6: "f32[384, 192, 3, 3][1728, 9, 3, 1]cuda:0", primals_7: "f32[384][1]cuda:0", primals_8: "f32[256, 384, 3, 3][3456, 9, 3, 1]cuda:0", primals_9: "f32[256][1]cuda:0", primals_10: "f32[256, 256, 3, 3][2304, 9, 3, 1]cuda:0", primals_11: "f32[256][1]cuda:0", primals_12: "f32[4096, 9216][9216, 1]cuda:0", primals_13: "f32[4096][1]cuda:0", primals_14: "f32[4096, 4096][4096, 1]cuda:0", primals_15: "f32[4096][1]cuda:0", primals_16: "f32[1000, 4096][4096, 1]cuda:0", primals_17: "f32[1000][1]cuda:0"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/alexnet.py:48 in forward, code: x = self.features(x)
        convert_element_type: "bf16[64][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_2, torch.bfloat16);  primals_2 = None
        convert_element_type_1: "bf16[64, 3, 11, 11][363, 121, 11, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_1, torch.bfloat16);  primals_1 = None
        convert_element_type_2: "bf16[128, 3, 224, 224][150528, 50176, 224, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_3, torch.bfloat16);  primals_3 = None
        convolution: "bf16[128, 64, 55, 55][193600, 3025, 55, 1]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_2, convert_element_type_1, convert_element_type, [4, 4], [2, 2], [1, 1], False, [0, 0], 1);  convert_element_type = None
        relu: "bf16[128, 64, 55, 55][193600, 3025, 55, 1]cuda:0" = torch.ops.aten.relu.default(convolution);  convolution = None
        _low_memory_max_pool_with_offsets = torch.ops.prims._low_memory_max_pool_with_offsets.default(relu, [3, 3], [2, 2], [0, 0], [1, 1], False)
        getitem: "bf16[128, 64, 27, 27][46656, 729, 27, 1]cuda:0" = _low_memory_max_pool_with_offsets[0]
        getitem_1: "i8[128, 64, 27, 27][46656, 729, 27, 1]cuda:0" = _low_memory_max_pool_with_offsets[1];  _low_memory_max_pool_with_offsets = None
        convert_element_type_3: "bf16[192][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_5, torch.bfloat16);  primals_5 = None
        convert_element_type_4: "bf16[192, 64, 5, 5][1600, 25, 5, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_4, torch.bfloat16);  primals_4 = None
        convolution_1: "bf16[128, 192, 27, 27][139968, 729, 27, 1]cuda:0" = torch.ops.aten.convolution.default(getitem, convert_element_type_4, convert_element_type_3, [1, 1], [2, 2], [1, 1], False, [0, 0], 1);  convert_element_type_3 = None
        relu_1: "bf16[128, 192, 27, 27][139968, 729, 27, 1]cuda:0" = torch.ops.aten.relu.default(convolution_1);  convolution_1 = None
        _low_memory_max_pool_with_offsets_1 = torch.ops.prims._low_memory_max_pool_with_offsets.default(relu_1, [3, 3], [2, 2], [0, 0], [1, 1], False)
        getitem_2: "bf16[128, 192, 13, 13][32448, 169, 13, 1]cuda:0" = _low_memory_max_pool_with_offsets_1[0]
        getitem_3: "i8[128, 192, 13, 13][32448, 169, 13, 1]cuda:0" = _low_memory_max_pool_with_offsets_1[1];  _low_memory_max_pool_with_offsets_1 = None
        convert_element_type_5: "bf16[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_7, torch.bfloat16);  primals_7 = None
        convert_element_type_6: "bf16[384, 192, 3, 3][1728, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_6, torch.bfloat16);  primals_6 = None
        convolution_2: "bf16[128, 384, 13, 13][64896, 169, 13, 1]cuda:0" = torch.ops.aten.convolution.default(getitem_2, convert_element_type_6, convert_element_type_5, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  convert_element_type_5 = None
        relu_2: "bf16[128, 384, 13, 13][64896, 169, 13, 1]cuda:0" = torch.ops.aten.relu.default(convolution_2);  convolution_2 = None
        convert_element_type_7: "bf16[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_9, torch.bfloat16);  primals_9 = None
        convert_element_type_8: "bf16[256, 384, 3, 3][3456, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_8, torch.bfloat16);  primals_8 = None
        convolution_3: "bf16[128, 256, 13, 13][43264, 169, 13, 1]cuda:0" = torch.ops.aten.convolution.default(relu_2, convert_element_type_8, convert_element_type_7, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  convert_element_type_7 = None
        relu_3: "bf16[128, 256, 13, 13][43264, 169, 13, 1]cuda:0" = torch.ops.aten.relu.default(convolution_3);  convolution_3 = None
        convert_element_type_9: "bf16[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_11, torch.bfloat16);  primals_11 = None
        convert_element_type_10: "bf16[256, 256, 3, 3][2304, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_10, torch.bfloat16);  primals_10 = None
        convolution_4: "bf16[128, 256, 13, 13][43264, 169, 13, 1]cuda:0" = torch.ops.aten.convolution.default(relu_3, convert_element_type_10, convert_element_type_9, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  convert_element_type_9 = None
        relu_4: "bf16[128, 256, 13, 13][43264, 169, 13, 1]cuda:0" = torch.ops.aten.relu.default(convolution_4);  convolution_4 = None
        _low_memory_max_pool_with_offsets_2 = torch.ops.prims._low_memory_max_pool_with_offsets.default(relu_4, [3, 3], [2, 2], [0, 0], [1, 1], False)
        getitem_4: "bf16[128, 256, 6, 6][9216, 36, 6, 1]cuda:0" = _low_memory_max_pool_with_offsets_2[0]
        getitem_5: "i8[128, 256, 6, 6][9216, 36, 6, 1]cuda:0" = _low_memory_max_pool_with_offsets_2[1];  _low_memory_max_pool_with_offsets_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/alexnet.py:49 in forward, code: x = self.avgpool(x)
        _adaptive_avg_pool2d: "bf16[128, 256, 6, 6][9216, 36, 6, 1]cuda:0" = torch.ops.aten._adaptive_avg_pool2d.default(getitem_4, [6, 6])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/alexnet.py:50 in forward, code: x = torch.flatten(x, 1)
        view: "bf16[128, 9216][9216, 1]cuda:0" = torch.ops.aten.reshape.default(_adaptive_avg_pool2d, [128, 9216]);  _adaptive_avg_pool2d = None

        # No stacktrace found for following nodes
        inductor_seeds_default: "i64[2][1]cuda:0" = torch.ops.prims.inductor_seeds.default(2, device(type='cuda', index=0))

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/alexnet.py:51 in forward, code: x = self.classifier(x)
        inductor_lookup_seed_default: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 0)
        inductor_random_default_1: "f32[128, 9216][9216, 1]cuda:0" = torch.ops.prims.inductor_random.default([128, 9216], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        convert_element_type_default_1: "bf16[128, 9216][9216, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_1, torch.bfloat16);  inductor_random_default_1 = None
        gt: "b8[128, 9216][9216, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_1, 0.5);  convert_element_type_default_1 = None
        mul: "bf16[128, 9216][9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt, view);  view = None
        mul_1: "bf16[128, 9216][9216, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul, 2.0);  mul = None
        convert_element_type_11: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_13, torch.bfloat16);  primals_13 = None
        convert_element_type_12: "bf16[4096, 9216][9216, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_12, torch.bfloat16);  primals_12 = None
        permute: "bf16[9216, 4096][1, 9216]cuda:0" = torch.ops.aten.permute.default(convert_element_type_12, [1, 0]);  convert_element_type_12 = None
        addmm: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_11, mul_1, permute);  convert_element_type_11 = None
        relu_5: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.relu.default(addmm);  addmm = None
        inductor_lookup_seed_default_1: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 1);  inductor_seeds_default = None
        inductor_random_default: "f32[128, 4096][4096, 1]cuda:0" = torch.ops.prims.inductor_random.default([128, 4096], inductor_lookup_seed_default_1, 'rand');  inductor_lookup_seed_default_1 = None
        convert_element_type_default: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default, torch.bfloat16);  inductor_random_default = None
        gt_1: "b8[128, 4096][4096, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default, 0.5);  convert_element_type_default = None
        mul_2: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_1, relu_5)
        mul_3: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2, 2.0);  mul_2 = None
        convert_element_type_16: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_15, torch.bfloat16);  primals_15 = None
        convert_element_type_17: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_14, torch.bfloat16);  primals_14 = None
        permute_1: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_17, [1, 0]);  convert_element_type_17 = None
        addmm_1: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_16, mul_3, permute_1);  convert_element_type_16 = None
        relu_6: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.relu.default(addmm_1);  addmm_1 = None
        convert_element_type_21: "bf16[1000][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_17, torch.bfloat16);  primals_17 = None
        convert_element_type_22: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_16, torch.bfloat16);  primals_16 = None
        permute_2: "bf16[4096, 1000][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_22, [1, 0]);  convert_element_type_22 = None
        addmm_2: "bf16[128, 1000][1000, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_21, relu_6, permute_2);  convert_element_type_21 = None
        permute_3: "bf16[1000, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None
        permute_7: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_1, [1, 0]);  permute_1 = None
        le_1: "b8[128, 4096][4096, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_5, 0);  relu_5 = None
        permute_11: "bf16[4096, 9216][9216, 1]cuda:0" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/alexnet.py:48 in forward, code: x = self.features(x)
        le_2: "b8[128, 256, 13, 13][43264, 169, 13, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_4, 0);  relu_4 = None
        le_5: "b8[128, 192, 27, 27][139968, 729, 27, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_1, 0);  relu_1 = None
        le_6: "b8[128, 64, 55, 55][193600, 3025, 55, 1]cuda:0" = torch.ops.aten.le.Scalar(relu, 0);  relu = None
        return (addmm_2, convert_element_type_1, convert_element_type_2, getitem, getitem_1, convert_element_type_4, getitem_2, getitem_3, convert_element_type_6, relu_2, convert_element_type_8, relu_3, convert_element_type_10, getitem_4, getitem_5, gt, mul_1, gt_1, mul_3, relu_6, permute_3, permute_7, le_1, permute_11, le_2, le_5, le_6)
