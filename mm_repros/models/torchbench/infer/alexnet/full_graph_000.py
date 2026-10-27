class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "bf16[64, 3, 11, 11][363, 121, 11, 1]cuda:0", arg1_1: "bf16[64][1]cuda:0", arg2_1: "bf16[128, 3, 224, 224][150528, 50176, 224, 1]cuda:0", arg3_1: "bf16[192, 64, 5, 5][1600, 25, 5, 1]cuda:0", arg4_1: "bf16[192][1]cuda:0", arg5_1: "bf16[384, 192, 3, 3][1728, 9, 3, 1]cuda:0", arg6_1: "bf16[384][1]cuda:0", arg7_1: "bf16[256, 384, 3, 3][3456, 9, 3, 1]cuda:0", arg8_1: "bf16[256][1]cuda:0", arg9_1: "bf16[256, 256, 3, 3][2304, 9, 3, 1]cuda:0", arg10_1: "bf16[256][1]cuda:0", arg11_1: "bf16[4096, 9216][9216, 1]cuda:0", arg12_1: "bf16[4096][1]cuda:0", arg13_1: "bf16[4096, 4096][4096, 1]cuda:0", arg14_1: "bf16[4096][1]cuda:0", arg15_1: "bf16[1000, 4096][4096, 1]cuda:0", arg16_1: "bf16[1000][1]cuda:0"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/alexnet.py:48 in forward, code: x = self.features(x)
        convolution: "bf16[128, 64, 55, 55][193600, 3025, 55, 1]cuda:0" = torch.ops.aten.convolution.default(arg2_1, arg0_1, arg1_1, [4, 4], [2, 2], [1, 1], False, [0, 0], 1);  arg2_1 = arg0_1 = arg1_1 = None
        relu: "bf16[128, 64, 55, 55][193600, 3025, 55, 1]cuda:0" = torch.ops.aten.relu.default(convolution);  convolution = None
        _low_memory_max_pool_with_offsets = torch.ops.prims._low_memory_max_pool_with_offsets.default(relu, [3, 3], [2, 2], [0, 0], [1, 1], False);  relu = None
        getitem: "bf16[128, 64, 27, 27][46656, 729, 27, 1]cuda:0" = _low_memory_max_pool_with_offsets[0];  _low_memory_max_pool_with_offsets = None
        convolution_1: "bf16[128, 192, 27, 27][139968, 729, 27, 1]cuda:0" = torch.ops.aten.convolution.default(getitem, arg3_1, arg4_1, [1, 1], [2, 2], [1, 1], False, [0, 0], 1);  getitem = arg3_1 = arg4_1 = None
        relu_1: "bf16[128, 192, 27, 27][139968, 729, 27, 1]cuda:0" = torch.ops.aten.relu.default(convolution_1);  convolution_1 = None
        _low_memory_max_pool_with_offsets_1 = torch.ops.prims._low_memory_max_pool_with_offsets.default(relu_1, [3, 3], [2, 2], [0, 0], [1, 1], False);  relu_1 = None
        getitem_2: "bf16[128, 192, 13, 13][32448, 169, 13, 1]cuda:0" = _low_memory_max_pool_with_offsets_1[0];  _low_memory_max_pool_with_offsets_1 = None
        convolution_2: "bf16[128, 384, 13, 13][64896, 169, 13, 1]cuda:0" = torch.ops.aten.convolution.default(getitem_2, arg5_1, arg6_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  getitem_2 = arg5_1 = arg6_1 = None
        relu_2: "bf16[128, 384, 13, 13][64896, 169, 13, 1]cuda:0" = torch.ops.aten.relu.default(convolution_2);  convolution_2 = None
        convolution_3: "bf16[128, 256, 13, 13][43264, 169, 13, 1]cuda:0" = torch.ops.aten.convolution.default(relu_2, arg7_1, arg8_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_2 = arg7_1 = arg8_1 = None
        relu_3: "bf16[128, 256, 13, 13][43264, 169, 13, 1]cuda:0" = torch.ops.aten.relu.default(convolution_3);  convolution_3 = None
        convolution_4: "bf16[128, 256, 13, 13][43264, 169, 13, 1]cuda:0" = torch.ops.aten.convolution.default(relu_3, arg9_1, arg10_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_3 = arg9_1 = arg10_1 = None
        relu_4: "bf16[128, 256, 13, 13][43264, 169, 13, 1]cuda:0" = torch.ops.aten.relu.default(convolution_4);  convolution_4 = None
        _low_memory_max_pool_with_offsets_2 = torch.ops.prims._low_memory_max_pool_with_offsets.default(relu_4, [3, 3], [2, 2], [0, 0], [1, 1], False);  relu_4 = None
        getitem_4: "bf16[128, 256, 6, 6][9216, 36, 6, 1]cuda:0" = _low_memory_max_pool_with_offsets_2[0];  _low_memory_max_pool_with_offsets_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/alexnet.py:49 in forward, code: x = self.avgpool(x)
        _adaptive_avg_pool2d: "bf16[128, 256, 6, 6][9216, 36, 6, 1]cuda:0" = torch.ops.aten._adaptive_avg_pool2d.default(getitem_4, [6, 6]);  getitem_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/alexnet.py:50 in forward, code: x = torch.flatten(x, 1)
        view: "bf16[128, 9216][9216, 1]cuda:0" = torch.ops.aten.reshape.default(_adaptive_avg_pool2d, [128, 9216]);  _adaptive_avg_pool2d = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/alexnet.py:51 in forward, code: x = self.classifier(x)
        permute: "bf16[9216, 4096][1, 9216]cuda:0" = torch.ops.aten.permute.default(arg11_1, [1, 0]);  arg11_1 = None
        addmm: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(arg12_1, view, permute);  arg12_1 = view = permute = None
        relu_5: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.relu.default(addmm);  addmm = None
        permute_1: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg13_1, [1, 0]);  arg13_1 = None
        addmm_1: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(arg14_1, relu_5, permute_1);  arg14_1 = relu_5 = permute_1 = None
        relu_6: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.relu.default(addmm_1);  addmm_1 = None
        permute_2: "bf16[4096, 1000][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg15_1, [1, 0]);  arg15_1 = None
        addmm_2: "bf16[128, 1000][1000, 1]cuda:0" = torch.ops.aten.addmm.default(arg16_1, relu_6, permute_2);  arg16_1 = relu_6 = permute_2 = None
        return (addmm_2,)
