class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[64, 3, 3, 3]", primals_2: "f32[64]", primals_3: "f32[128, 3, 224, 224]", primals_4: "f32[64, 64, 3, 3]", primals_5: "f32[64]", primals_6: "f32[128, 64, 3, 3]", primals_7: "f32[128]", primals_8: "f32[128, 128, 3, 3]", primals_9: "f32[128]", primals_10: "f32[256, 128, 3, 3]", primals_11: "f32[256]", primals_12: "f32[256, 256, 3, 3]", primals_13: "f32[256]", primals_14: "f32[256, 256, 3, 3]", primals_15: "f32[256]", primals_16: "f32[512, 256, 3, 3]", primals_17: "f32[512]", primals_18: "f32[512, 512, 3, 3]", primals_19: "f32[512]", primals_20: "f32[512, 512, 3, 3]", primals_21: "f32[512]", primals_22: "f32[512, 512, 3, 3]", primals_23: "f32[512]", primals_24: "f32[512, 512, 3, 3]", primals_25: "f32[512]", primals_26: "f32[512, 512, 3, 3]", primals_27: "f32[512]", primals_28: "f32[4096, 25088]", primals_29: "f32[4096]", primals_30: "f32[4096, 4096]", primals_31: "f32[4096]", primals_32: "f32[1000, 4096]", primals_33: "f32[1000]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vgg.py:66 in forward, code: x = self.features(x)
        convolution: "f32[128, 64, 224, 224]" = torch.ops.aten.convolution.default(primals_3, primals_1, primals_2, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_2 = None
        relu: "f32[128, 64, 224, 224]" = torch.ops.aten.relu.default(convolution);  convolution = None
        convolution_1: "f32[128, 64, 224, 224]" = torch.ops.aten.convolution.default(relu, primals_4, primals_5, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_5 = None
        relu_1: "f32[128, 64, 224, 224]" = torch.ops.aten.relu.default(convolution_1);  convolution_1 = None
        _low_memory_max_pool_with_offsets = torch.ops.prims._low_memory_max_pool_with_offsets.default(relu_1, [2, 2], [2, 2], [0, 0], [1, 1], False)
        getitem: "f32[128, 64, 112, 112]" = _low_memory_max_pool_with_offsets[0]
        getitem_1: "i8[128, 64, 112, 112]" = _low_memory_max_pool_with_offsets[1];  _low_memory_max_pool_with_offsets = None
        convolution_2: "f32[128, 128, 112, 112]" = torch.ops.aten.convolution.default(getitem, primals_6, primals_7, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_7 = None
        relu_2: "f32[128, 128, 112, 112]" = torch.ops.aten.relu.default(convolution_2);  convolution_2 = None
        convolution_3: "f32[128, 128, 112, 112]" = torch.ops.aten.convolution.default(relu_2, primals_8, primals_9, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_9 = None
        relu_3: "f32[128, 128, 112, 112]" = torch.ops.aten.relu.default(convolution_3);  convolution_3 = None
        _low_memory_max_pool_with_offsets_1 = torch.ops.prims._low_memory_max_pool_with_offsets.default(relu_3, [2, 2], [2, 2], [0, 0], [1, 1], False)
        getitem_2: "f32[128, 128, 56, 56]" = _low_memory_max_pool_with_offsets_1[0]
        getitem_3: "i8[128, 128, 56, 56]" = _low_memory_max_pool_with_offsets_1[1];  _low_memory_max_pool_with_offsets_1 = None
        convolution_4: "f32[128, 256, 56, 56]" = torch.ops.aten.convolution.default(getitem_2, primals_10, primals_11, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_11 = None
        relu_4: "f32[128, 256, 56, 56]" = torch.ops.aten.relu.default(convolution_4);  convolution_4 = None
        convolution_5: "f32[128, 256, 56, 56]" = torch.ops.aten.convolution.default(relu_4, primals_12, primals_13, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_13 = None
        relu_5: "f32[128, 256, 56, 56]" = torch.ops.aten.relu.default(convolution_5);  convolution_5 = None
        convolution_6: "f32[128, 256, 56, 56]" = torch.ops.aten.convolution.default(relu_5, primals_14, primals_15, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_15 = None
        relu_6: "f32[128, 256, 56, 56]" = torch.ops.aten.relu.default(convolution_6);  convolution_6 = None
        _low_memory_max_pool_with_offsets_2 = torch.ops.prims._low_memory_max_pool_with_offsets.default(relu_6, [2, 2], [2, 2], [0, 0], [1, 1], False)
        getitem_4: "f32[128, 256, 28, 28]" = _low_memory_max_pool_with_offsets_2[0]
        getitem_5: "i8[128, 256, 28, 28]" = _low_memory_max_pool_with_offsets_2[1];  _low_memory_max_pool_with_offsets_2 = None
        convolution_7: "f32[128, 512, 28, 28]" = torch.ops.aten.convolution.default(getitem_4, primals_16, primals_17, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_17 = None
        relu_7: "f32[128, 512, 28, 28]" = torch.ops.aten.relu.default(convolution_7);  convolution_7 = None
        convolution_8: "f32[128, 512, 28, 28]" = torch.ops.aten.convolution.default(relu_7, primals_18, primals_19, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_19 = None
        relu_8: "f32[128, 512, 28, 28]" = torch.ops.aten.relu.default(convolution_8);  convolution_8 = None
        convolution_9: "f32[128, 512, 28, 28]" = torch.ops.aten.convolution.default(relu_8, primals_20, primals_21, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_21 = None
        relu_9: "f32[128, 512, 28, 28]" = torch.ops.aten.relu.default(convolution_9);  convolution_9 = None
        _low_memory_max_pool_with_offsets_3 = torch.ops.prims._low_memory_max_pool_with_offsets.default(relu_9, [2, 2], [2, 2], [0, 0], [1, 1], False)
        getitem_6: "f32[128, 512, 14, 14]" = _low_memory_max_pool_with_offsets_3[0]
        getitem_7: "i8[128, 512, 14, 14]" = _low_memory_max_pool_with_offsets_3[1];  _low_memory_max_pool_with_offsets_3 = None
        convolution_10: "f32[128, 512, 14, 14]" = torch.ops.aten.convolution.default(getitem_6, primals_22, primals_23, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_23 = None
        relu_10: "f32[128, 512, 14, 14]" = torch.ops.aten.relu.default(convolution_10);  convolution_10 = None
        convolution_11: "f32[128, 512, 14, 14]" = torch.ops.aten.convolution.default(relu_10, primals_24, primals_25, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_25 = None
        relu_11: "f32[128, 512, 14, 14]" = torch.ops.aten.relu.default(convolution_11);  convolution_11 = None
        convolution_12: "f32[128, 512, 14, 14]" = torch.ops.aten.convolution.default(relu_11, primals_26, primals_27, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_27 = None
        relu_12: "f32[128, 512, 14, 14]" = torch.ops.aten.relu.default(convolution_12);  convolution_12 = None
        _low_memory_max_pool_with_offsets_4 = torch.ops.prims._low_memory_max_pool_with_offsets.default(relu_12, [2, 2], [2, 2], [0, 0], [1, 1], False)
        getitem_8: "f32[128, 512, 7, 7]" = _low_memory_max_pool_with_offsets_4[0]
        getitem_9: "i8[128, 512, 7, 7]" = _low_memory_max_pool_with_offsets_4[1];  _low_memory_max_pool_with_offsets_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vgg.py:67 in forward, code: x = self.avgpool(x)
        _adaptive_avg_pool2d: "f32[128, 512, 7, 7]" = torch.ops.aten._adaptive_avg_pool2d.default(getitem_8, [7, 7])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vgg.py:68 in forward, code: x = torch.flatten(x, 1)
        view: "f32[128, 25088]" = torch.ops.aten.reshape.default(_adaptive_avg_pool2d, [128, 25088]);  _adaptive_avg_pool2d = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vgg.py:69 in forward, code: x = self.classifier(x)
        permute: "f32[25088, 4096]" = torch.ops.aten.permute.default(primals_28, [1, 0])
        addmm: "f32[128, 4096]" = torch.ops.aten.addmm.default(primals_29, view, permute);  primals_29 = permute = None
        relu_13: "f32[128, 4096]" = torch.ops.aten.relu.default(addmm);  addmm = None

        # No stacktrace found for following nodes
        inductor_seeds_default: "i64[2]" = torch.ops.prims.inductor_seeds.default(2, device(type='cuda', index=0))

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vgg.py:69 in forward, code: x = self.classifier(x)
        inductor_lookup_seed_default: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 0)
        inductor_random_default_1: "f32[128, 4096]" = torch.ops.prims.inductor_random.default([128, 4096], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        gt: "b8[128, 4096]" = torch.ops.aten.gt.Scalar(inductor_random_default_1, 0.5);  inductor_random_default_1 = None
        mul: "f32[128, 4096]" = torch.ops.aten.mul.Tensor(gt, relu_13)
        mul_1: "f32[128, 4096]" = torch.ops.aten.mul.Tensor(mul, 2.0);  mul = None
        permute_1: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_30, [1, 0])
        addmm_1: "f32[128, 4096]" = torch.ops.aten.addmm.default(primals_31, mul_1, permute_1);  primals_31 = permute_1 = None
        relu_14: "f32[128, 4096]" = torch.ops.aten.relu.default(addmm_1);  addmm_1 = None
        inductor_lookup_seed_default_1: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 1);  inductor_seeds_default = None
        inductor_random_default: "f32[128, 4096]" = torch.ops.prims.inductor_random.default([128, 4096], inductor_lookup_seed_default_1, 'rand');  inductor_lookup_seed_default_1 = None
        gt_1: "b8[128, 4096]" = torch.ops.aten.gt.Scalar(inductor_random_default, 0.5);  inductor_random_default = None
        mul_2: "f32[128, 4096]" = torch.ops.aten.mul.Tensor(gt_1, relu_14)
        mul_3: "f32[128, 4096]" = torch.ops.aten.mul.Tensor(mul_2, 2.0);  mul_2 = None
        permute_2: "f32[4096, 1000]" = torch.ops.aten.permute.default(primals_32, [1, 0])
        addmm_2: "f32[128, 1000]" = torch.ops.aten.addmm.default(primals_33, mul_3, permute_2);  primals_33 = permute_2 = None
        le: "b8[128, 4096]" = torch.ops.aten.le.Scalar(relu_14, 0);  relu_14 = None
        le_1: "b8[128, 4096]" = torch.ops.aten.le.Scalar(relu_13, 0);  relu_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vgg.py:66 in forward, code: x = self.features(x)
        le_2: "b8[128, 512, 14, 14]" = torch.ops.aten.le.Scalar(relu_12, 0);  relu_12 = None
        le_5: "b8[128, 512, 28, 28]" = torch.ops.aten.le.Scalar(relu_9, 0);  relu_9 = None
        le_8: "b8[128, 256, 56, 56]" = torch.ops.aten.le.Scalar(relu_6, 0);  relu_6 = None
        le_11: "b8[128, 128, 112, 112]" = torch.ops.aten.le.Scalar(relu_3, 0);  relu_3 = None
        le_13: "b8[128, 64, 224, 224]" = torch.ops.aten.le.Scalar(relu_1, 0);  relu_1 = None
        return (addmm_2, primals_1, primals_3, primals_4, primals_6, primals_8, primals_10, primals_12, primals_14, primals_16, primals_18, primals_20, primals_22, primals_24, primals_26, primals_28, primals_30, primals_32, relu, getitem, getitem_1, relu_2, getitem_2, getitem_3, relu_4, relu_5, getitem_4, getitem_5, relu_7, relu_8, getitem_6, getitem_7, relu_10, relu_11, getitem_8, getitem_9, view, gt, mul_1, gt_1, mul_3, le, le_1, le_2, le_5, le_8, le_11, le_13)
