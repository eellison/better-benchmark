class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[64, 3, 11, 11]", primals_2: "f32[64]", primals_3: "f32[1024, 3, 224, 224]", primals_4: "f32[192, 64, 5, 5]", primals_5: "f32[192]", primals_6: "f32[384, 192, 3, 3]", primals_7: "f32[384]", primals_8: "f32[256, 384, 3, 3]", primals_9: "f32[256]", primals_10: "f32[256, 256, 3, 3]", primals_11: "f32[256]", primals_12: "f32[4096, 9216]", primals_13: "f32[4096]", primals_14: "f32[4096, 4096]", primals_15: "f32[4096]", primals_16: "f32[1000, 4096]", primals_17: "f32[1000]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/alexnet.py:48 in forward, code: x = self.features(x)
        convolution: "f32[1024, 64, 55, 55]" = torch.ops.aten.convolution.default(primals_3, primals_1, primals_2, [4, 4], [2, 2], [1, 1], False, [0, 0], 1);  primals_2 = None
        relu: "f32[1024, 64, 55, 55]" = torch.ops.aten.relu.default(convolution);  convolution = None
        _low_memory_max_pool_with_offsets = torch.ops.prims._low_memory_max_pool_with_offsets.default(relu, [3, 3], [2, 2], [0, 0], [1, 1], False)
        getitem: "f32[1024, 64, 27, 27]" = _low_memory_max_pool_with_offsets[0]
        getitem_1: "i8[1024, 64, 27, 27]" = _low_memory_max_pool_with_offsets[1];  _low_memory_max_pool_with_offsets = None
        convolution_1: "f32[1024, 192, 27, 27]" = torch.ops.aten.convolution.default(getitem, primals_4, primals_5, [1, 1], [2, 2], [1, 1], False, [0, 0], 1);  primals_5 = None
        relu_1: "f32[1024, 192, 27, 27]" = torch.ops.aten.relu.default(convolution_1);  convolution_1 = None
        _low_memory_max_pool_with_offsets_1 = torch.ops.prims._low_memory_max_pool_with_offsets.default(relu_1, [3, 3], [2, 2], [0, 0], [1, 1], False)
        getitem_2: "f32[1024, 192, 13, 13]" = _low_memory_max_pool_with_offsets_1[0]
        getitem_3: "i8[1024, 192, 13, 13]" = _low_memory_max_pool_with_offsets_1[1];  _low_memory_max_pool_with_offsets_1 = None
        convolution_2: "f32[1024, 384, 13, 13]" = torch.ops.aten.convolution.default(getitem_2, primals_6, primals_7, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_7 = None
        relu_2: "f32[1024, 384, 13, 13]" = torch.ops.aten.relu.default(convolution_2);  convolution_2 = None
        convolution_3: "f32[1024, 256, 13, 13]" = torch.ops.aten.convolution.default(relu_2, primals_8, primals_9, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_9 = None
        relu_3: "f32[1024, 256, 13, 13]" = torch.ops.aten.relu.default(convolution_3);  convolution_3 = None
        convolution_4: "f32[1024, 256, 13, 13]" = torch.ops.aten.convolution.default(relu_3, primals_10, primals_11, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_11 = None
        relu_4: "f32[1024, 256, 13, 13]" = torch.ops.aten.relu.default(convolution_4);  convolution_4 = None
        _low_memory_max_pool_with_offsets_2 = torch.ops.prims._low_memory_max_pool_with_offsets.default(relu_4, [3, 3], [2, 2], [0, 0], [1, 1], False)
        getitem_4: "f32[1024, 256, 6, 6]" = _low_memory_max_pool_with_offsets_2[0]
        getitem_5: "i8[1024, 256, 6, 6]" = _low_memory_max_pool_with_offsets_2[1];  _low_memory_max_pool_with_offsets_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/alexnet.py:49 in forward, code: x = self.avgpool(x)
        _adaptive_avg_pool2d: "f32[1024, 256, 6, 6]" = torch.ops.aten._adaptive_avg_pool2d.default(getitem_4, [6, 6])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/alexnet.py:50 in forward, code: x = torch.flatten(x, 1)
        view: "f32[1024, 9216]" = torch.ops.aten.reshape.default(_adaptive_avg_pool2d, [1024, 9216]);  _adaptive_avg_pool2d = None

        # No stacktrace found for following nodes
        inductor_seeds_default: "i64[2]" = torch.ops.prims.inductor_seeds.default(2, device(type='cuda', index=0))

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/alexnet.py:51 in forward, code: x = self.classifier(x)
        inductor_lookup_seed_default: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 0)
        inductor_random_default_1: "f32[1024, 9216]" = torch.ops.prims.inductor_random.default([1024, 9216], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        gt: "b8[1024, 9216]" = torch.ops.aten.gt.Scalar(inductor_random_default_1, 0.5);  inductor_random_default_1 = None
        mul: "f32[1024, 9216]" = torch.ops.aten.mul.Tensor(gt, view);  view = None
        mul_1: "f32[1024, 9216]" = torch.ops.aten.mul.Tensor(mul, 2.0);  mul = None
        permute: "f32[9216, 4096]" = torch.ops.aten.permute.default(primals_12, [1, 0])
        addmm: "f32[1024, 4096]" = torch.ops.aten.addmm.default(primals_13, mul_1, permute);  primals_13 = permute = None
        relu_5: "f32[1024, 4096]" = torch.ops.aten.relu.default(addmm);  addmm = None
        inductor_lookup_seed_default_1: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 1);  inductor_seeds_default = None
        inductor_random_default: "f32[1024, 4096]" = torch.ops.prims.inductor_random.default([1024, 4096], inductor_lookup_seed_default_1, 'rand');  inductor_lookup_seed_default_1 = None
        gt_1: "b8[1024, 4096]" = torch.ops.aten.gt.Scalar(inductor_random_default, 0.5);  inductor_random_default = None
        mul_2: "f32[1024, 4096]" = torch.ops.aten.mul.Tensor(gt_1, relu_5)
        mul_3: "f32[1024, 4096]" = torch.ops.aten.mul.Tensor(mul_2, 2.0);  mul_2 = None
        permute_1: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_14, [1, 0])
        addmm_1: "f32[1024, 4096]" = torch.ops.aten.addmm.default(primals_15, mul_3, permute_1);  primals_15 = permute_1 = None
        relu_6: "f32[1024, 4096]" = torch.ops.aten.relu.default(addmm_1);  addmm_1 = None
        permute_2: "f32[4096, 1000]" = torch.ops.aten.permute.default(primals_16, [1, 0])
        addmm_2: "f32[1024, 1000]" = torch.ops.aten.addmm.default(primals_17, relu_6, permute_2);  primals_17 = permute_2 = None
        le_1: "b8[1024, 4096]" = torch.ops.aten.le.Scalar(relu_5, 0);  relu_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/alexnet.py:48 in forward, code: x = self.features(x)
        le_2: "b8[1024, 256, 13, 13]" = torch.ops.aten.le.Scalar(relu_4, 0);  relu_4 = None
        le_5: "b8[1024, 192, 27, 27]" = torch.ops.aten.le.Scalar(relu_1, 0);  relu_1 = None
        le_6: "b8[1024, 64, 55, 55]" = torch.ops.aten.le.Scalar(relu, 0);  relu = None
        return (addmm_2, primals_1, primals_3, primals_4, primals_6, primals_8, primals_10, primals_12, primals_14, primals_16, getitem, getitem_1, getitem_2, getitem_3, relu_2, relu_3, getitem_4, getitem_5, gt, mul_1, gt_1, mul_3, relu_6, le_1, le_2, le_5, le_6)
