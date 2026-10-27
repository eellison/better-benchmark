class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "bf16[64, 3, 3, 3][27, 9, 3, 1]cuda:0", arg1_1: "bf16[64][1]cuda:0", arg2_1: "bf16[4, 3, 224, 224][150528, 50176, 224, 1]cuda:0", arg3_1: "bf16[64, 64, 3, 3][576, 9, 3, 1]cuda:0", arg4_1: "bf16[64][1]cuda:0", arg5_1: "bf16[128, 64, 3, 3][576, 9, 3, 1]cuda:0", arg6_1: "bf16[128][1]cuda:0", arg7_1: "bf16[128, 128, 3, 3][1152, 9, 3, 1]cuda:0", arg8_1: "bf16[128][1]cuda:0", arg9_1: "bf16[256, 128, 3, 3][1152, 9, 3, 1]cuda:0", arg10_1: "bf16[256][1]cuda:0", arg11_1: "bf16[256, 256, 3, 3][2304, 9, 3, 1]cuda:0", arg12_1: "bf16[256][1]cuda:0", arg13_1: "bf16[256, 256, 3, 3][2304, 9, 3, 1]cuda:0", arg14_1: "bf16[256][1]cuda:0", arg15_1: "bf16[512, 256, 3, 3][2304, 9, 3, 1]cuda:0", arg16_1: "bf16[512][1]cuda:0", arg17_1: "bf16[512, 512, 3, 3][4608, 9, 3, 1]cuda:0", arg18_1: "bf16[512][1]cuda:0", arg19_1: "bf16[512, 512, 3, 3][4608, 9, 3, 1]cuda:0", arg20_1: "bf16[512][1]cuda:0", arg21_1: "bf16[512, 512, 3, 3][4608, 9, 3, 1]cuda:0", arg22_1: "bf16[512][1]cuda:0", arg23_1: "bf16[512, 512, 3, 3][4608, 9, 3, 1]cuda:0", arg24_1: "bf16[512][1]cuda:0", arg25_1: "bf16[512, 512, 3, 3][4608, 9, 3, 1]cuda:0", arg26_1: "bf16[512][1]cuda:0", arg27_1: "bf16[4096, 25088][25088, 1]cuda:0", arg28_1: "bf16[4096][1]cuda:0", arg29_1: "bf16[4096, 4096][4096, 1]cuda:0", arg30_1: "bf16[4096][1]cuda:0", arg31_1: "bf16[1000, 4096][4096, 1]cuda:0", arg32_1: "bf16[1000][1]cuda:0"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vgg.py:66 in forward, code: x = self.features(x)
        convolution: "bf16[4, 64, 224, 224][3211264, 50176, 224, 1]cuda:0" = torch.ops.aten.convolution.default(arg2_1, arg0_1, arg1_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg2_1 = arg0_1 = arg1_1 = None
        relu: "bf16[4, 64, 224, 224][3211264, 50176, 224, 1]cuda:0" = torch.ops.aten.relu.default(convolution);  convolution = None
        convolution_1: "bf16[4, 64, 224, 224][3211264, 50176, 224, 1]cuda:0" = torch.ops.aten.convolution.default(relu, arg3_1, arg4_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu = arg3_1 = arg4_1 = None
        relu_1: "bf16[4, 64, 224, 224][3211264, 50176, 224, 1]cuda:0" = torch.ops.aten.relu.default(convolution_1);  convolution_1 = None
        _low_memory_max_pool_with_offsets = torch.ops.prims._low_memory_max_pool_with_offsets.default(relu_1, [2, 2], [2, 2], [0, 0], [1, 1], False);  relu_1 = None
        getitem: "bf16[4, 64, 112, 112][802816, 12544, 112, 1]cuda:0" = _low_memory_max_pool_with_offsets[0];  _low_memory_max_pool_with_offsets = None
        convolution_2: "bf16[4, 128, 112, 112][1605632, 12544, 112, 1]cuda:0" = torch.ops.aten.convolution.default(getitem, arg5_1, arg6_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  getitem = arg5_1 = arg6_1 = None
        relu_2: "bf16[4, 128, 112, 112][1605632, 12544, 112, 1]cuda:0" = torch.ops.aten.relu.default(convolution_2);  convolution_2 = None
        convolution_3: "bf16[4, 128, 112, 112][1605632, 12544, 112, 1]cuda:0" = torch.ops.aten.convolution.default(relu_2, arg7_1, arg8_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_2 = arg7_1 = arg8_1 = None
        relu_3: "bf16[4, 128, 112, 112][1605632, 12544, 112, 1]cuda:0" = torch.ops.aten.relu.default(convolution_3);  convolution_3 = None
        _low_memory_max_pool_with_offsets_1 = torch.ops.prims._low_memory_max_pool_with_offsets.default(relu_3, [2, 2], [2, 2], [0, 0], [1, 1], False);  relu_3 = None
        getitem_2: "bf16[4, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = _low_memory_max_pool_with_offsets_1[0];  _low_memory_max_pool_with_offsets_1 = None
        convolution_4: "bf16[4, 256, 56, 56][802816, 3136, 56, 1]cuda:0" = torch.ops.aten.convolution.default(getitem_2, arg9_1, arg10_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  getitem_2 = arg9_1 = arg10_1 = None
        relu_4: "bf16[4, 256, 56, 56][802816, 3136, 56, 1]cuda:0" = torch.ops.aten.relu.default(convolution_4);  convolution_4 = None
        convolution_5: "bf16[4, 256, 56, 56][802816, 3136, 56, 1]cuda:0" = torch.ops.aten.convolution.default(relu_4, arg11_1, arg12_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_4 = arg11_1 = arg12_1 = None
        relu_5: "bf16[4, 256, 56, 56][802816, 3136, 56, 1]cuda:0" = torch.ops.aten.relu.default(convolution_5);  convolution_5 = None
        convolution_6: "bf16[4, 256, 56, 56][802816, 3136, 56, 1]cuda:0" = torch.ops.aten.convolution.default(relu_5, arg13_1, arg14_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_5 = arg13_1 = arg14_1 = None
        relu_6: "bf16[4, 256, 56, 56][802816, 3136, 56, 1]cuda:0" = torch.ops.aten.relu.default(convolution_6);  convolution_6 = None
        _low_memory_max_pool_with_offsets_2 = torch.ops.prims._low_memory_max_pool_with_offsets.default(relu_6, [2, 2], [2, 2], [0, 0], [1, 1], False);  relu_6 = None
        getitem_4: "bf16[4, 256, 28, 28][200704, 784, 28, 1]cuda:0" = _low_memory_max_pool_with_offsets_2[0];  _low_memory_max_pool_with_offsets_2 = None
        convolution_7: "bf16[4, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.aten.convolution.default(getitem_4, arg15_1, arg16_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  getitem_4 = arg15_1 = arg16_1 = None
        relu_7: "bf16[4, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.aten.relu.default(convolution_7);  convolution_7 = None
        convolution_8: "bf16[4, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.aten.convolution.default(relu_7, arg17_1, arg18_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_7 = arg17_1 = arg18_1 = None
        relu_8: "bf16[4, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.aten.relu.default(convolution_8);  convolution_8 = None
        convolution_9: "bf16[4, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.aten.convolution.default(relu_8, arg19_1, arg20_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_8 = arg19_1 = arg20_1 = None
        relu_9: "bf16[4, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.aten.relu.default(convolution_9);  convolution_9 = None
        _low_memory_max_pool_with_offsets_3 = torch.ops.prims._low_memory_max_pool_with_offsets.default(relu_9, [2, 2], [2, 2], [0, 0], [1, 1], False);  relu_9 = None
        getitem_6: "bf16[4, 512, 14, 14][100352, 196, 14, 1]cuda:0" = _low_memory_max_pool_with_offsets_3[0];  _low_memory_max_pool_with_offsets_3 = None
        convolution_10: "bf16[4, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(getitem_6, arg21_1, arg22_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  getitem_6 = arg21_1 = arg22_1 = None
        relu_10: "bf16[4, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.relu.default(convolution_10);  convolution_10 = None
        convolution_11: "bf16[4, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(relu_10, arg23_1, arg24_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_10 = arg23_1 = arg24_1 = None
        relu_11: "bf16[4, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.relu.default(convolution_11);  convolution_11 = None
        convolution_12: "bf16[4, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(relu_11, arg25_1, arg26_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_11 = arg25_1 = arg26_1 = None
        relu_12: "bf16[4, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.relu.default(convolution_12);  convolution_12 = None
        _low_memory_max_pool_with_offsets_4 = torch.ops.prims._low_memory_max_pool_with_offsets.default(relu_12, [2, 2], [2, 2], [0, 0], [1, 1], False);  relu_12 = None
        getitem_8: "bf16[4, 512, 7, 7][25088, 49, 7, 1]cuda:0" = _low_memory_max_pool_with_offsets_4[0];  _low_memory_max_pool_with_offsets_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vgg.py:67 in forward, code: x = self.avgpool(x)
        _adaptive_avg_pool2d: "bf16[4, 512, 7, 7][25088, 49, 7, 1]cuda:0" = torch.ops.aten._adaptive_avg_pool2d.default(getitem_8, [7, 7]);  getitem_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vgg.py:68 in forward, code: x = torch.flatten(x, 1)
        view: "bf16[4, 25088][25088, 1]cuda:0" = torch.ops.aten.reshape.default(_adaptive_avg_pool2d, [4, 25088]);  _adaptive_avg_pool2d = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vgg.py:69 in forward, code: x = self.classifier(x)
        permute: "bf16[25088, 4096][1, 25088]cuda:0" = torch.ops.aten.permute.default(arg27_1, [1, 0]);  arg27_1 = None
        addmm: "bf16[4, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(arg28_1, view, permute);  arg28_1 = view = permute = None
        relu_13: "bf16[4, 4096][4096, 1]cuda:0" = torch.ops.aten.relu.default(addmm);  addmm = None
        permute_1: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg29_1, [1, 0]);  arg29_1 = None
        addmm_1: "bf16[4, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(arg30_1, relu_13, permute_1);  arg30_1 = relu_13 = permute_1 = None
        relu_14: "bf16[4, 4096][4096, 1]cuda:0" = torch.ops.aten.relu.default(addmm_1);  addmm_1 = None
        permute_2: "bf16[4096, 1000][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg31_1, [1, 0]);  arg31_1 = None
        addmm_2: "bf16[4, 1000][1000, 1]cuda:0" = torch.ops.aten.addmm.default(arg32_1, relu_14, permute_2);  arg32_1 = relu_14 = permute_2 = None
        return (addmm_2,)
