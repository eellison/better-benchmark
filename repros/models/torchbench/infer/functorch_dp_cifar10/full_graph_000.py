class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "bf16[64, 3, 7, 7][147, 49, 7, 1]cuda:0", arg1_1: "bf16[128, 3, 32, 32][3072, 1024, 32, 1]cuda:0", arg2_1: "bf16[64][1]cuda:0", arg3_1: "bf16[64][1]cuda:0", arg4_1: "bf16[64, 64, 3, 3][576, 9, 3, 1]cuda:0", arg5_1: "bf16[64][1]cuda:0", arg6_1: "bf16[64][1]cuda:0", arg7_1: "bf16[64, 64, 3, 3][576, 9, 3, 1]cuda:0", arg8_1: "bf16[64][1]cuda:0", arg9_1: "bf16[64][1]cuda:0", arg10_1: "bf16[64, 64, 3, 3][576, 9, 3, 1]cuda:0", arg11_1: "bf16[64][1]cuda:0", arg12_1: "bf16[64][1]cuda:0", arg13_1: "bf16[64, 64, 3, 3][576, 9, 3, 1]cuda:0", arg14_1: "bf16[64][1]cuda:0", arg15_1: "bf16[64][1]cuda:0", arg16_1: "bf16[128, 64, 3, 3][576, 9, 3, 1]cuda:0", arg17_1: "bf16[128][1]cuda:0", arg18_1: "bf16[128][1]cuda:0", arg19_1: "bf16[128, 128, 3, 3][1152, 9, 3, 1]cuda:0", arg20_1: "bf16[128][1]cuda:0", arg21_1: "bf16[128][1]cuda:0", arg22_1: "bf16[128, 64, 1, 1][64, 1, 1, 1]cuda:0", arg23_1: "bf16[128][1]cuda:0", arg24_1: "bf16[128][1]cuda:0", arg25_1: "bf16[128, 128, 3, 3][1152, 9, 3, 1]cuda:0", arg26_1: "bf16[128][1]cuda:0", arg27_1: "bf16[128][1]cuda:0", arg28_1: "bf16[128, 128, 3, 3][1152, 9, 3, 1]cuda:0", arg29_1: "bf16[128][1]cuda:0", arg30_1: "bf16[128][1]cuda:0", arg31_1: "bf16[256, 128, 3, 3][1152, 9, 3, 1]cuda:0", arg32_1: "bf16[256][1]cuda:0", arg33_1: "bf16[256][1]cuda:0", arg34_1: "bf16[256, 256, 3, 3][2304, 9, 3, 1]cuda:0", arg35_1: "bf16[256][1]cuda:0", arg36_1: "bf16[256][1]cuda:0", arg37_1: "bf16[256, 128, 1, 1][128, 1, 1, 1]cuda:0", arg38_1: "bf16[256][1]cuda:0", arg39_1: "bf16[256][1]cuda:0", arg40_1: "bf16[256, 256, 3, 3][2304, 9, 3, 1]cuda:0", arg41_1: "bf16[256][1]cuda:0", arg42_1: "bf16[256][1]cuda:0", arg43_1: "bf16[256, 256, 3, 3][2304, 9, 3, 1]cuda:0", arg44_1: "bf16[256][1]cuda:0", arg45_1: "bf16[256][1]cuda:0", arg46_1: "bf16[512, 256, 3, 3][2304, 9, 3, 1]cuda:0", arg47_1: "bf16[512][1]cuda:0", arg48_1: "bf16[512][1]cuda:0", arg49_1: "bf16[512, 512, 3, 3][4608, 9, 3, 1]cuda:0", arg50_1: "bf16[512][1]cuda:0", arg51_1: "bf16[512][1]cuda:0", arg52_1: "bf16[512, 256, 1, 1][256, 1, 1, 1]cuda:0", arg53_1: "bf16[512][1]cuda:0", arg54_1: "bf16[512][1]cuda:0", arg55_1: "bf16[512, 512, 3, 3][4608, 9, 3, 1]cuda:0", arg56_1: "bf16[512][1]cuda:0", arg57_1: "bf16[512][1]cuda:0", arg58_1: "bf16[512, 512, 3, 3][4608, 9, 3, 1]cuda:0", arg59_1: "bf16[512][1]cuda:0", arg60_1: "bf16[512][1]cuda:0", arg61_1: "bf16[1000, 512][512, 1]cuda:0", arg62_1: "bf16[1000][1]cuda:0"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:268 in _forward_impl, code: x = self.conv1(x)
        convolution: "bf16[128, 64, 16, 16][16384, 256, 16, 1]cuda:0" = torch.ops.aten.convolution.default(arg1_1, arg0_1, None, [2, 2], [3, 3], [1, 1], False, [0, 0], 1);  arg1_1 = arg0_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:269 in _forward_impl, code: x = self.bn1(x)
        convert_element_type: "f32[128, 64, 16, 16][16384, 256, 16, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution, torch.float32);  convolution = None
        view: "f32[128, 32, 2, 256][16384, 512, 256, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type, [128, 32, 2, 256]);  convert_element_type = None
        var_mean = torch.ops.aten.var_mean.correction(view, [2, 3], correction = 0, keepdim = True)
        getitem: "f32[128, 32, 1, 1][32, 1, 1, 1]cuda:0" = var_mean[0]
        getitem_1: "f32[128, 32, 1, 1][32, 1, 1, 1]cuda:0" = var_mean[1];  var_mean = None
        sub: "f32[128, 32, 2, 256][16384, 512, 256, 1]cuda:0" = torch.ops.aten.sub.Tensor(view, getitem_1);  view = getitem_1 = None
        add: "f32[128, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt: "f32[128, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add);  add = None
        mul: "f32[128, 32, 2, 256][16384, 512, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
        view_1: "f32[128, 64, 16, 16][16384, 256, 16, 1]cuda:0" = torch.ops.aten.reshape.default(mul, [128, 64, 16, 16]);  mul = None
        unsqueeze: "bf16[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg2_1, 0);  arg2_1 = None
        unsqueeze_1: "bf16[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze, 2);  unsqueeze = None
        unsqueeze_2: "bf16[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1, 3);  unsqueeze_1 = None
        mul_1: "f32[128, 64, 16, 16][16384, 256, 16, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_1, unsqueeze_2);  view_1 = unsqueeze_2 = None
        unsqueeze_3: "bf16[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg3_1, 0);  arg3_1 = None
        unsqueeze_4: "bf16[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_3, 2);  unsqueeze_3 = None
        unsqueeze_5: "bf16[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_4, 3);  unsqueeze_4 = None
        add_1: "f32[128, 64, 16, 16][16384, 256, 16, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_1, unsqueeze_5);  mul_1 = unsqueeze_5 = None
        convert_element_type_1: "bf16[128, 64, 16, 16][16384, 256, 16, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_1, torch.bfloat16);  add_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:270 in _forward_impl, code: x = self.relu(x)
        relu: "bf16[128, 64, 16, 16][16384, 256, 16, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_1);  convert_element_type_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:271 in _forward_impl, code: x = self.maxpool(x)
        _low_memory_max_pool_with_offsets = torch.ops.prims._low_memory_max_pool_with_offsets.default(relu, [3, 3], [2, 2], [1, 1], [1, 1], False);  relu = None
        getitem_2: "bf16[128, 64, 8, 8][4096, 64, 8, 1]cuda:0" = _low_memory_max_pool_with_offsets[0];  _low_memory_max_pool_with_offsets = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:92 in forward, code: out = self.conv1(x)
        convolution_1: "bf16[128, 64, 8, 8][4096, 64, 8, 1]cuda:0" = torch.ops.aten.convolution.default(getitem_2, arg4_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg4_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:93 in forward, code: out = self.bn1(out)
        convert_element_type_4: "f32[128, 64, 8, 8][4096, 64, 8, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_1, torch.float32);  convolution_1 = None
        view_2: "f32[128, 32, 2, 64][4096, 128, 64, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_4, [128, 32, 2, 64]);  convert_element_type_4 = None
        var_mean_1 = torch.ops.aten.var_mean.correction(view_2, [2, 3], correction = 0, keepdim = True)
        getitem_4: "f32[128, 32, 1, 1][32, 1, 1, 1]cuda:0" = var_mean_1[0]
        getitem_5: "f32[128, 32, 1, 1][32, 1, 1, 1]cuda:0" = var_mean_1[1];  var_mean_1 = None
        sub_1: "f32[128, 32, 2, 64][4096, 128, 64, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_2, getitem_5);  view_2 = getitem_5 = None
        add_2: "f32[128, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_4, 1e-05);  getitem_4 = None
        rsqrt_1: "f32[128, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_2);  add_2 = None
        mul_2: "f32[128, 32, 2, 64][4096, 128, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = rsqrt_1 = None
        view_3: "f32[128, 64, 8, 8][4096, 64, 8, 1]cuda:0" = torch.ops.aten.reshape.default(mul_2, [128, 64, 8, 8]);  mul_2 = None
        unsqueeze_6: "bf16[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg5_1, 0);  arg5_1 = None
        unsqueeze_7: "bf16[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_6, 2);  unsqueeze_6 = None
        unsqueeze_8: "bf16[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_7, 3);  unsqueeze_7 = None
        mul_3: "f32[128, 64, 8, 8][4096, 64, 8, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_3, unsqueeze_8);  view_3 = unsqueeze_8 = None
        unsqueeze_9: "bf16[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg6_1, 0);  arg6_1 = None
        unsqueeze_10: "bf16[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_9, 2);  unsqueeze_9 = None
        unsqueeze_11: "bf16[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_10, 3);  unsqueeze_10 = None
        add_3: "f32[128, 64, 8, 8][4096, 64, 8, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_3, unsqueeze_11);  mul_3 = unsqueeze_11 = None
        convert_element_type_5: "bf16[128, 64, 8, 8][4096, 64, 8, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_3, torch.bfloat16);  add_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:94 in forward, code: out = self.relu(out)
        relu_1: "bf16[128, 64, 8, 8][4096, 64, 8, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_5);  convert_element_type_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:96 in forward, code: out = self.conv2(out)
        convolution_2: "bf16[128, 64, 8, 8][4096, 64, 8, 1]cuda:0" = torch.ops.aten.convolution.default(relu_1, arg7_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_1 = arg7_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:97 in forward, code: out = self.bn2(out)
        convert_element_type_8: "f32[128, 64, 8, 8][4096, 64, 8, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_2, torch.float32);  convolution_2 = None
        view_4: "f32[128, 32, 2, 64][4096, 128, 64, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_8, [128, 32, 2, 64]);  convert_element_type_8 = None
        var_mean_2 = torch.ops.aten.var_mean.correction(view_4, [2, 3], correction = 0, keepdim = True)
        getitem_6: "f32[128, 32, 1, 1][32, 1, 1, 1]cuda:0" = var_mean_2[0]
        getitem_7: "f32[128, 32, 1, 1][32, 1, 1, 1]cuda:0" = var_mean_2[1];  var_mean_2 = None
        sub_2: "f32[128, 32, 2, 64][4096, 128, 64, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_4, getitem_7);  view_4 = getitem_7 = None
        add_4: "f32[128, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_6, 1e-05);  getitem_6 = None
        rsqrt_2: "f32[128, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_4);  add_4 = None
        mul_4: "f32[128, 32, 2, 64][4096, 128, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_2);  sub_2 = rsqrt_2 = None
        view_5: "f32[128, 64, 8, 8][4096, 64, 8, 1]cuda:0" = torch.ops.aten.reshape.default(mul_4, [128, 64, 8, 8]);  mul_4 = None
        unsqueeze_12: "bf16[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg8_1, 0);  arg8_1 = None
        unsqueeze_13: "bf16[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_12, 2);  unsqueeze_12 = None
        unsqueeze_14: "bf16[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_13, 3);  unsqueeze_13 = None
        mul_5: "f32[128, 64, 8, 8][4096, 64, 8, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_5, unsqueeze_14);  view_5 = unsqueeze_14 = None
        unsqueeze_15: "bf16[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg9_1, 0);  arg9_1 = None
        unsqueeze_16: "bf16[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_15, 2);  unsqueeze_15 = None
        unsqueeze_17: "bf16[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_16, 3);  unsqueeze_16 = None
        add_5: "f32[128, 64, 8, 8][4096, 64, 8, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_5, unsqueeze_17);  mul_5 = unsqueeze_17 = None
        convert_element_type_9: "bf16[128, 64, 8, 8][4096, 64, 8, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_5, torch.bfloat16);  add_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:102 in forward, code: out += identity
        add_6: "bf16[128, 64, 8, 8][4096, 64, 8, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_9, getitem_2);  convert_element_type_9 = getitem_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:103 in forward, code: out = self.relu(out)
        relu_2: "bf16[128, 64, 8, 8][4096, 64, 8, 1]cuda:0" = torch.ops.aten.relu.default(add_6);  add_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:92 in forward, code: out = self.conv1(x)
        convolution_3: "bf16[128, 64, 8, 8][4096, 64, 8, 1]cuda:0" = torch.ops.aten.convolution.default(relu_2, arg10_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg10_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:93 in forward, code: out = self.bn1(out)
        convert_element_type_12: "f32[128, 64, 8, 8][4096, 64, 8, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_3, torch.float32);  convolution_3 = None
        view_6: "f32[128, 32, 2, 64][4096, 128, 64, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_12, [128, 32, 2, 64]);  convert_element_type_12 = None
        var_mean_3 = torch.ops.aten.var_mean.correction(view_6, [2, 3], correction = 0, keepdim = True)
        getitem_8: "f32[128, 32, 1, 1][32, 1, 1, 1]cuda:0" = var_mean_3[0]
        getitem_9: "f32[128, 32, 1, 1][32, 1, 1, 1]cuda:0" = var_mean_3[1];  var_mean_3 = None
        sub_3: "f32[128, 32, 2, 64][4096, 128, 64, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_6, getitem_9);  view_6 = getitem_9 = None
        add_7: "f32[128, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_8, 1e-05);  getitem_8 = None
        rsqrt_3: "f32[128, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_7);  add_7 = None
        mul_6: "f32[128, 32, 2, 64][4096, 128, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_3);  sub_3 = rsqrt_3 = None
        view_7: "f32[128, 64, 8, 8][4096, 64, 8, 1]cuda:0" = torch.ops.aten.reshape.default(mul_6, [128, 64, 8, 8]);  mul_6 = None
        unsqueeze_18: "bf16[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg11_1, 0);  arg11_1 = None
        unsqueeze_19: "bf16[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_18, 2);  unsqueeze_18 = None
        unsqueeze_20: "bf16[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_19, 3);  unsqueeze_19 = None
        mul_7: "f32[128, 64, 8, 8][4096, 64, 8, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_7, unsqueeze_20);  view_7 = unsqueeze_20 = None
        unsqueeze_21: "bf16[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg12_1, 0);  arg12_1 = None
        unsqueeze_22: "bf16[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_21, 2);  unsqueeze_21 = None
        unsqueeze_23: "bf16[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_22, 3);  unsqueeze_22 = None
        add_8: "f32[128, 64, 8, 8][4096, 64, 8, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_7, unsqueeze_23);  mul_7 = unsqueeze_23 = None
        convert_element_type_13: "bf16[128, 64, 8, 8][4096, 64, 8, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_8, torch.bfloat16);  add_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:94 in forward, code: out = self.relu(out)
        relu_3: "bf16[128, 64, 8, 8][4096, 64, 8, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_13);  convert_element_type_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:96 in forward, code: out = self.conv2(out)
        convolution_4: "bf16[128, 64, 8, 8][4096, 64, 8, 1]cuda:0" = torch.ops.aten.convolution.default(relu_3, arg13_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_3 = arg13_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:97 in forward, code: out = self.bn2(out)
        convert_element_type_16: "f32[128, 64, 8, 8][4096, 64, 8, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_4, torch.float32);  convolution_4 = None
        view_8: "f32[128, 32, 2, 64][4096, 128, 64, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_16, [128, 32, 2, 64]);  convert_element_type_16 = None
        var_mean_4 = torch.ops.aten.var_mean.correction(view_8, [2, 3], correction = 0, keepdim = True)
        getitem_10: "f32[128, 32, 1, 1][32, 1, 1, 1]cuda:0" = var_mean_4[0]
        getitem_11: "f32[128, 32, 1, 1][32, 1, 1, 1]cuda:0" = var_mean_4[1];  var_mean_4 = None
        sub_4: "f32[128, 32, 2, 64][4096, 128, 64, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_8, getitem_11);  view_8 = getitem_11 = None
        add_9: "f32[128, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_10, 1e-05);  getitem_10 = None
        rsqrt_4: "f32[128, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_9);  add_9 = None
        mul_8: "f32[128, 32, 2, 64][4096, 128, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_4);  sub_4 = rsqrt_4 = None
        view_9: "f32[128, 64, 8, 8][4096, 64, 8, 1]cuda:0" = torch.ops.aten.reshape.default(mul_8, [128, 64, 8, 8]);  mul_8 = None
        unsqueeze_24: "bf16[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg14_1, 0);  arg14_1 = None
        unsqueeze_25: "bf16[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_24, 2);  unsqueeze_24 = None
        unsqueeze_26: "bf16[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_25, 3);  unsqueeze_25 = None
        mul_9: "f32[128, 64, 8, 8][4096, 64, 8, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_9, unsqueeze_26);  view_9 = unsqueeze_26 = None
        unsqueeze_27: "bf16[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg15_1, 0);  arg15_1 = None
        unsqueeze_28: "bf16[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_27, 2);  unsqueeze_27 = None
        unsqueeze_29: "bf16[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_28, 3);  unsqueeze_28 = None
        add_10: "f32[128, 64, 8, 8][4096, 64, 8, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_9, unsqueeze_29);  mul_9 = unsqueeze_29 = None
        convert_element_type_17: "bf16[128, 64, 8, 8][4096, 64, 8, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_10, torch.bfloat16);  add_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:102 in forward, code: out += identity
        add_11: "bf16[128, 64, 8, 8][4096, 64, 8, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_17, relu_2);  convert_element_type_17 = relu_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:103 in forward, code: out = self.relu(out)
        relu_4: "bf16[128, 64, 8, 8][4096, 64, 8, 1]cuda:0" = torch.ops.aten.relu.default(add_11);  add_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:92 in forward, code: out = self.conv1(x)
        convolution_5: "bf16[128, 128, 4, 4][2048, 16, 4, 1]cuda:0" = torch.ops.aten.convolution.default(relu_4, arg16_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  arg16_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:93 in forward, code: out = self.bn1(out)
        convert_element_type_20: "f32[128, 128, 4, 4][2048, 16, 4, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_5, torch.float32);  convolution_5 = None
        view_10: "f32[128, 32, 4, 16][2048, 64, 16, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_20, [128, 32, 4, 16]);  convert_element_type_20 = None
        var_mean_5 = torch.ops.aten.var_mean.correction(view_10, [2, 3], correction = 0, keepdim = True)
        getitem_12: "f32[128, 32, 1, 1][32, 1, 1, 1]cuda:0" = var_mean_5[0]
        getitem_13: "f32[128, 32, 1, 1][32, 1, 1, 1]cuda:0" = var_mean_5[1];  var_mean_5 = None
        sub_5: "f32[128, 32, 4, 16][2048, 64, 16, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_10, getitem_13);  view_10 = getitem_13 = None
        add_12: "f32[128, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_12, 1e-05);  getitem_12 = None
        rsqrt_5: "f32[128, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_12);  add_12 = None
        mul_10: "f32[128, 32, 4, 16][2048, 64, 16, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_5);  sub_5 = rsqrt_5 = None
        view_11: "f32[128, 128, 4, 4][2048, 16, 4, 1]cuda:0" = torch.ops.aten.reshape.default(mul_10, [128, 128, 4, 4]);  mul_10 = None
        unsqueeze_30: "bf16[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg17_1, 0);  arg17_1 = None
        unsqueeze_31: "bf16[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_30, 2);  unsqueeze_30 = None
        unsqueeze_32: "bf16[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_31, 3);  unsqueeze_31 = None
        mul_11: "f32[128, 128, 4, 4][2048, 16, 4, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_11, unsqueeze_32);  view_11 = unsqueeze_32 = None
        unsqueeze_33: "bf16[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg18_1, 0);  arg18_1 = None
        unsqueeze_34: "bf16[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_33, 2);  unsqueeze_33 = None
        unsqueeze_35: "bf16[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_34, 3);  unsqueeze_34 = None
        add_13: "f32[128, 128, 4, 4][2048, 16, 4, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_11, unsqueeze_35);  mul_11 = unsqueeze_35 = None
        convert_element_type_21: "bf16[128, 128, 4, 4][2048, 16, 4, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_13, torch.bfloat16);  add_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:94 in forward, code: out = self.relu(out)
        relu_5: "bf16[128, 128, 4, 4][2048, 16, 4, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_21);  convert_element_type_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:96 in forward, code: out = self.conv2(out)
        convolution_6: "bf16[128, 128, 4, 4][2048, 16, 4, 1]cuda:0" = torch.ops.aten.convolution.default(relu_5, arg19_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_5 = arg19_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:97 in forward, code: out = self.bn2(out)
        convert_element_type_24: "f32[128, 128, 4, 4][2048, 16, 4, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_6, torch.float32);  convolution_6 = None
        view_12: "f32[128, 32, 4, 16][2048, 64, 16, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_24, [128, 32, 4, 16]);  convert_element_type_24 = None
        var_mean_6 = torch.ops.aten.var_mean.correction(view_12, [2, 3], correction = 0, keepdim = True)
        getitem_14: "f32[128, 32, 1, 1][32, 1, 1, 1]cuda:0" = var_mean_6[0]
        getitem_15: "f32[128, 32, 1, 1][32, 1, 1, 1]cuda:0" = var_mean_6[1];  var_mean_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:100 in forward, code: identity = self.downsample(x)
        convolution_7: "bf16[128, 128, 4, 4][2048, 16, 4, 1]cuda:0" = torch.ops.aten.convolution.default(relu_4, arg22_1, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  relu_4 = arg22_1 = None
        convert_element_type_28: "f32[128, 128, 4, 4][2048, 16, 4, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_7, torch.float32);  convolution_7 = None
        view_14: "f32[128, 32, 4, 16][2048, 64, 16, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_28, [128, 32, 4, 16]);  convert_element_type_28 = None
        var_mean_7 = torch.ops.aten.var_mean.correction(view_14, [2, 3], correction = 0, keepdim = True)
        getitem_16: "f32[128, 32, 1, 1][32, 1, 1, 1]cuda:0" = var_mean_7[0]
        getitem_17: "f32[128, 32, 1, 1][32, 1, 1, 1]cuda:0" = var_mean_7[1];  var_mean_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:97 in forward, code: out = self.bn2(out)
        sub_6: "f32[128, 32, 4, 16][2048, 64, 16, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_12, getitem_15);  view_12 = getitem_15 = None
        add_14: "f32[128, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_14, 1e-05);  getitem_14 = None
        rsqrt_6: "f32[128, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_14);  add_14 = None
        mul_12: "f32[128, 32, 4, 16][2048, 64, 16, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_6);  sub_6 = rsqrt_6 = None
        view_13: "f32[128, 128, 4, 4][2048, 16, 4, 1]cuda:0" = torch.ops.aten.reshape.default(mul_12, [128, 128, 4, 4]);  mul_12 = None
        unsqueeze_36: "bf16[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg20_1, 0);  arg20_1 = None
        unsqueeze_37: "bf16[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_36, 2);  unsqueeze_36 = None
        unsqueeze_38: "bf16[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_37, 3);  unsqueeze_37 = None
        mul_13: "f32[128, 128, 4, 4][2048, 16, 4, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_13, unsqueeze_38);  view_13 = unsqueeze_38 = None
        unsqueeze_39: "bf16[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg21_1, 0);  arg21_1 = None
        unsqueeze_40: "bf16[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_39, 2);  unsqueeze_39 = None
        unsqueeze_41: "bf16[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_40, 3);  unsqueeze_40 = None
        add_15: "f32[128, 128, 4, 4][2048, 16, 4, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_13, unsqueeze_41);  mul_13 = unsqueeze_41 = None
        convert_element_type_25: "bf16[128, 128, 4, 4][2048, 16, 4, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_15, torch.bfloat16);  add_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:100 in forward, code: identity = self.downsample(x)
        sub_7: "f32[128, 32, 4, 16][2048, 64, 16, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_14, getitem_17);  view_14 = getitem_17 = None
        add_16: "f32[128, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_16, 1e-05);  getitem_16 = None
        rsqrt_7: "f32[128, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_16);  add_16 = None
        mul_14: "f32[128, 32, 4, 16][2048, 64, 16, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_7);  sub_7 = rsqrt_7 = None
        view_15: "f32[128, 128, 4, 4][2048, 16, 4, 1]cuda:0" = torch.ops.aten.reshape.default(mul_14, [128, 128, 4, 4]);  mul_14 = None
        unsqueeze_42: "bf16[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg23_1, 0);  arg23_1 = None
        unsqueeze_43: "bf16[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_42, 2);  unsqueeze_42 = None
        unsqueeze_44: "bf16[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_43, 3);  unsqueeze_43 = None
        mul_15: "f32[128, 128, 4, 4][2048, 16, 4, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_15, unsqueeze_44);  view_15 = unsqueeze_44 = None
        unsqueeze_45: "bf16[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg24_1, 0);  arg24_1 = None
        unsqueeze_46: "bf16[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_45, 2);  unsqueeze_45 = None
        unsqueeze_47: "bf16[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_46, 3);  unsqueeze_46 = None
        add_17: "f32[128, 128, 4, 4][2048, 16, 4, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_15, unsqueeze_47);  mul_15 = unsqueeze_47 = None
        convert_element_type_29: "bf16[128, 128, 4, 4][2048, 16, 4, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_17, torch.bfloat16);  add_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:102 in forward, code: out += identity
        add_18: "bf16[128, 128, 4, 4][2048, 16, 4, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_25, convert_element_type_29);  convert_element_type_25 = convert_element_type_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:103 in forward, code: out = self.relu(out)
        relu_6: "bf16[128, 128, 4, 4][2048, 16, 4, 1]cuda:0" = torch.ops.aten.relu.default(add_18);  add_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:92 in forward, code: out = self.conv1(x)
        convolution_8: "bf16[128, 128, 4, 4][2048, 16, 4, 1]cuda:0" = torch.ops.aten.convolution.default(relu_6, arg25_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg25_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:93 in forward, code: out = self.bn1(out)
        convert_element_type_32: "f32[128, 128, 4, 4][2048, 16, 4, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_8, torch.float32);  convolution_8 = None
        view_16: "f32[128, 32, 4, 16][2048, 64, 16, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_32, [128, 32, 4, 16]);  convert_element_type_32 = None
        var_mean_8 = torch.ops.aten.var_mean.correction(view_16, [2, 3], correction = 0, keepdim = True)
        getitem_18: "f32[128, 32, 1, 1][32, 1, 1, 1]cuda:0" = var_mean_8[0]
        getitem_19: "f32[128, 32, 1, 1][32, 1, 1, 1]cuda:0" = var_mean_8[1];  var_mean_8 = None
        sub_8: "f32[128, 32, 4, 16][2048, 64, 16, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_16, getitem_19);  view_16 = getitem_19 = None
        add_19: "f32[128, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_18, 1e-05);  getitem_18 = None
        rsqrt_8: "f32[128, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_19);  add_19 = None
        mul_16: "f32[128, 32, 4, 16][2048, 64, 16, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_8);  sub_8 = rsqrt_8 = None
        view_17: "f32[128, 128, 4, 4][2048, 16, 4, 1]cuda:0" = torch.ops.aten.reshape.default(mul_16, [128, 128, 4, 4]);  mul_16 = None
        unsqueeze_48: "bf16[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg26_1, 0);  arg26_1 = None
        unsqueeze_49: "bf16[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_48, 2);  unsqueeze_48 = None
        unsqueeze_50: "bf16[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_49, 3);  unsqueeze_49 = None
        mul_17: "f32[128, 128, 4, 4][2048, 16, 4, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_17, unsqueeze_50);  view_17 = unsqueeze_50 = None
        unsqueeze_51: "bf16[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg27_1, 0);  arg27_1 = None
        unsqueeze_52: "bf16[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_51, 2);  unsqueeze_51 = None
        unsqueeze_53: "bf16[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_52, 3);  unsqueeze_52 = None
        add_20: "f32[128, 128, 4, 4][2048, 16, 4, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_17, unsqueeze_53);  mul_17 = unsqueeze_53 = None
        convert_element_type_33: "bf16[128, 128, 4, 4][2048, 16, 4, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_20, torch.bfloat16);  add_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:94 in forward, code: out = self.relu(out)
        relu_7: "bf16[128, 128, 4, 4][2048, 16, 4, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_33);  convert_element_type_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:96 in forward, code: out = self.conv2(out)
        convolution_9: "bf16[128, 128, 4, 4][2048, 16, 4, 1]cuda:0" = torch.ops.aten.convolution.default(relu_7, arg28_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_7 = arg28_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:97 in forward, code: out = self.bn2(out)
        convert_element_type_36: "f32[128, 128, 4, 4][2048, 16, 4, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_9, torch.float32);  convolution_9 = None
        view_18: "f32[128, 32, 4, 16][2048, 64, 16, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_36, [128, 32, 4, 16]);  convert_element_type_36 = None
        var_mean_9 = torch.ops.aten.var_mean.correction(view_18, [2, 3], correction = 0, keepdim = True)
        getitem_20: "f32[128, 32, 1, 1][32, 1, 1, 1]cuda:0" = var_mean_9[0]
        getitem_21: "f32[128, 32, 1, 1][32, 1, 1, 1]cuda:0" = var_mean_9[1];  var_mean_9 = None
        sub_9: "f32[128, 32, 4, 16][2048, 64, 16, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_18, getitem_21);  view_18 = getitem_21 = None
        add_21: "f32[128, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_20, 1e-05);  getitem_20 = None
        rsqrt_9: "f32[128, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_21);  add_21 = None
        mul_18: "f32[128, 32, 4, 16][2048, 64, 16, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_9);  sub_9 = rsqrt_9 = None
        view_19: "f32[128, 128, 4, 4][2048, 16, 4, 1]cuda:0" = torch.ops.aten.reshape.default(mul_18, [128, 128, 4, 4]);  mul_18 = None
        unsqueeze_54: "bf16[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg29_1, 0);  arg29_1 = None
        unsqueeze_55: "bf16[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_54, 2);  unsqueeze_54 = None
        unsqueeze_56: "bf16[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_55, 3);  unsqueeze_55 = None
        mul_19: "f32[128, 128, 4, 4][2048, 16, 4, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_19, unsqueeze_56);  view_19 = unsqueeze_56 = None
        unsqueeze_57: "bf16[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg30_1, 0);  arg30_1 = None
        unsqueeze_58: "bf16[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_57, 2);  unsqueeze_57 = None
        unsqueeze_59: "bf16[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_58, 3);  unsqueeze_58 = None
        add_22: "f32[128, 128, 4, 4][2048, 16, 4, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_19, unsqueeze_59);  mul_19 = unsqueeze_59 = None
        convert_element_type_37: "bf16[128, 128, 4, 4][2048, 16, 4, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_22, torch.bfloat16);  add_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:102 in forward, code: out += identity
        add_23: "bf16[128, 128, 4, 4][2048, 16, 4, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_37, relu_6);  convert_element_type_37 = relu_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:103 in forward, code: out = self.relu(out)
        relu_8: "bf16[128, 128, 4, 4][2048, 16, 4, 1]cuda:0" = torch.ops.aten.relu.default(add_23);  add_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:92 in forward, code: out = self.conv1(x)
        convolution_10: "bf16[128, 256, 2, 2][1024, 4, 2, 1]cuda:0" = torch.ops.aten.convolution.default(relu_8, arg31_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  arg31_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:93 in forward, code: out = self.bn1(out)
        convert_element_type_40: "f32[128, 256, 2, 2][1024, 4, 2, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_10, torch.float32);  convolution_10 = None
        view_20: "f32[128, 32, 8, 4][1024, 32, 4, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_40, [128, 32, 8, 4]);  convert_element_type_40 = None
        var_mean_10 = torch.ops.aten.var_mean.correction(view_20, [2, 3], correction = 0, keepdim = True)
        getitem_22: "f32[128, 32, 1, 1][32, 1, 1, 1]cuda:0" = var_mean_10[0]
        getitem_23: "f32[128, 32, 1, 1][32, 1, 1, 1]cuda:0" = var_mean_10[1];  var_mean_10 = None
        sub_10: "f32[128, 32, 8, 4][1024, 32, 4, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_20, getitem_23);  view_20 = getitem_23 = None
        add_24: "f32[128, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_22, 1e-05);  getitem_22 = None
        rsqrt_10: "f32[128, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_24);  add_24 = None
        mul_20: "f32[128, 32, 8, 4][1024, 32, 4, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_10);  sub_10 = rsqrt_10 = None
        view_21: "f32[128, 256, 2, 2][1024, 4, 2, 1]cuda:0" = torch.ops.aten.reshape.default(mul_20, [128, 256, 2, 2]);  mul_20 = None
        unsqueeze_60: "bf16[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg32_1, 0);  arg32_1 = None
        unsqueeze_61: "bf16[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_60, 2);  unsqueeze_60 = None
        unsqueeze_62: "bf16[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_61, 3);  unsqueeze_61 = None
        mul_21: "f32[128, 256, 2, 2][1024, 4, 2, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_21, unsqueeze_62);  view_21 = unsqueeze_62 = None
        unsqueeze_63: "bf16[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg33_1, 0);  arg33_1 = None
        unsqueeze_64: "bf16[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_63, 2);  unsqueeze_63 = None
        unsqueeze_65: "bf16[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_64, 3);  unsqueeze_64 = None
        add_25: "f32[128, 256, 2, 2][1024, 4, 2, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_21, unsqueeze_65);  mul_21 = unsqueeze_65 = None
        convert_element_type_41: "bf16[128, 256, 2, 2][1024, 4, 2, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_25, torch.bfloat16);  add_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:94 in forward, code: out = self.relu(out)
        relu_9: "bf16[128, 256, 2, 2][1024, 4, 2, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_41);  convert_element_type_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:96 in forward, code: out = self.conv2(out)
        convolution_11: "bf16[128, 256, 2, 2][1024, 4, 2, 1]cuda:0" = torch.ops.aten.convolution.default(relu_9, arg34_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_9 = arg34_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:97 in forward, code: out = self.bn2(out)
        convert_element_type_44: "f32[128, 256, 2, 2][1024, 4, 2, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_11, torch.float32);  convolution_11 = None
        view_22: "f32[128, 32, 8, 4][1024, 32, 4, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_44, [128, 32, 8, 4]);  convert_element_type_44 = None
        var_mean_11 = torch.ops.aten.var_mean.correction(view_22, [2, 3], correction = 0, keepdim = True)
        getitem_24: "f32[128, 32, 1, 1][32, 1, 1, 1]cuda:0" = var_mean_11[0]
        getitem_25: "f32[128, 32, 1, 1][32, 1, 1, 1]cuda:0" = var_mean_11[1];  var_mean_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:100 in forward, code: identity = self.downsample(x)
        convolution_12: "bf16[128, 256, 2, 2][1024, 4, 2, 1]cuda:0" = torch.ops.aten.convolution.default(relu_8, arg37_1, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  relu_8 = arg37_1 = None
        convert_element_type_48: "f32[128, 256, 2, 2][1024, 4, 2, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_12, torch.float32);  convolution_12 = None
        view_24: "f32[128, 32, 8, 4][1024, 32, 4, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_48, [128, 32, 8, 4]);  convert_element_type_48 = None
        var_mean_12 = torch.ops.aten.var_mean.correction(view_24, [2, 3], correction = 0, keepdim = True)
        getitem_26: "f32[128, 32, 1, 1][32, 1, 1, 1]cuda:0" = var_mean_12[0]
        getitem_27: "f32[128, 32, 1, 1][32, 1, 1, 1]cuda:0" = var_mean_12[1];  var_mean_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:97 in forward, code: out = self.bn2(out)
        sub_11: "f32[128, 32, 8, 4][1024, 32, 4, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_22, getitem_25);  view_22 = getitem_25 = None
        add_26: "f32[128, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_24, 1e-05);  getitem_24 = None
        rsqrt_11: "f32[128, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_26);  add_26 = None
        mul_22: "f32[128, 32, 8, 4][1024, 32, 4, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_11);  sub_11 = rsqrt_11 = None
        view_23: "f32[128, 256, 2, 2][1024, 4, 2, 1]cuda:0" = torch.ops.aten.reshape.default(mul_22, [128, 256, 2, 2]);  mul_22 = None
        unsqueeze_66: "bf16[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg35_1, 0);  arg35_1 = None
        unsqueeze_67: "bf16[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_66, 2);  unsqueeze_66 = None
        unsqueeze_68: "bf16[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_67, 3);  unsqueeze_67 = None
        mul_23: "f32[128, 256, 2, 2][1024, 4, 2, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_23, unsqueeze_68);  view_23 = unsqueeze_68 = None
        unsqueeze_69: "bf16[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg36_1, 0);  arg36_1 = None
        unsqueeze_70: "bf16[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_69, 2);  unsqueeze_69 = None
        unsqueeze_71: "bf16[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_70, 3);  unsqueeze_70 = None
        add_27: "f32[128, 256, 2, 2][1024, 4, 2, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_23, unsqueeze_71);  mul_23 = unsqueeze_71 = None
        convert_element_type_45: "bf16[128, 256, 2, 2][1024, 4, 2, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_27, torch.bfloat16);  add_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:100 in forward, code: identity = self.downsample(x)
        sub_12: "f32[128, 32, 8, 4][1024, 32, 4, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_24, getitem_27);  view_24 = getitem_27 = None
        add_28: "f32[128, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_26, 1e-05);  getitem_26 = None
        rsqrt_12: "f32[128, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_28);  add_28 = None
        mul_24: "f32[128, 32, 8, 4][1024, 32, 4, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_12);  sub_12 = rsqrt_12 = None
        view_25: "f32[128, 256, 2, 2][1024, 4, 2, 1]cuda:0" = torch.ops.aten.reshape.default(mul_24, [128, 256, 2, 2]);  mul_24 = None
        unsqueeze_72: "bf16[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg38_1, 0);  arg38_1 = None
        unsqueeze_73: "bf16[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_72, 2);  unsqueeze_72 = None
        unsqueeze_74: "bf16[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_73, 3);  unsqueeze_73 = None
        mul_25: "f32[128, 256, 2, 2][1024, 4, 2, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_25, unsqueeze_74);  view_25 = unsqueeze_74 = None
        unsqueeze_75: "bf16[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg39_1, 0);  arg39_1 = None
        unsqueeze_76: "bf16[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_75, 2);  unsqueeze_75 = None
        unsqueeze_77: "bf16[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_76, 3);  unsqueeze_76 = None
        add_29: "f32[128, 256, 2, 2][1024, 4, 2, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_25, unsqueeze_77);  mul_25 = unsqueeze_77 = None
        convert_element_type_49: "bf16[128, 256, 2, 2][1024, 4, 2, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_29, torch.bfloat16);  add_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:102 in forward, code: out += identity
        add_30: "bf16[128, 256, 2, 2][1024, 4, 2, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_45, convert_element_type_49);  convert_element_type_45 = convert_element_type_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:103 in forward, code: out = self.relu(out)
        relu_10: "bf16[128, 256, 2, 2][1024, 4, 2, 1]cuda:0" = torch.ops.aten.relu.default(add_30);  add_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:92 in forward, code: out = self.conv1(x)
        convolution_13: "bf16[128, 256, 2, 2][1024, 4, 2, 1]cuda:0" = torch.ops.aten.convolution.default(relu_10, arg40_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg40_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:93 in forward, code: out = self.bn1(out)
        convert_element_type_52: "f32[128, 256, 2, 2][1024, 4, 2, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_13, torch.float32);  convolution_13 = None
        view_26: "f32[128, 32, 8, 4][1024, 32, 4, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_52, [128, 32, 8, 4]);  convert_element_type_52 = None
        var_mean_13 = torch.ops.aten.var_mean.correction(view_26, [2, 3], correction = 0, keepdim = True)
        getitem_28: "f32[128, 32, 1, 1][32, 1, 1, 1]cuda:0" = var_mean_13[0]
        getitem_29: "f32[128, 32, 1, 1][32, 1, 1, 1]cuda:0" = var_mean_13[1];  var_mean_13 = None
        sub_13: "f32[128, 32, 8, 4][1024, 32, 4, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_26, getitem_29);  view_26 = getitem_29 = None
        add_31: "f32[128, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_28, 1e-05);  getitem_28 = None
        rsqrt_13: "f32[128, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_31);  add_31 = None
        mul_26: "f32[128, 32, 8, 4][1024, 32, 4, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_13);  sub_13 = rsqrt_13 = None
        view_27: "f32[128, 256, 2, 2][1024, 4, 2, 1]cuda:0" = torch.ops.aten.reshape.default(mul_26, [128, 256, 2, 2]);  mul_26 = None
        unsqueeze_78: "bf16[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg41_1, 0);  arg41_1 = None
        unsqueeze_79: "bf16[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_78, 2);  unsqueeze_78 = None
        unsqueeze_80: "bf16[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_79, 3);  unsqueeze_79 = None
        mul_27: "f32[128, 256, 2, 2][1024, 4, 2, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_27, unsqueeze_80);  view_27 = unsqueeze_80 = None
        unsqueeze_81: "bf16[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg42_1, 0);  arg42_1 = None
        unsqueeze_82: "bf16[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_81, 2);  unsqueeze_81 = None
        unsqueeze_83: "bf16[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_82, 3);  unsqueeze_82 = None
        add_32: "f32[128, 256, 2, 2][1024, 4, 2, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_27, unsqueeze_83);  mul_27 = unsqueeze_83 = None
        convert_element_type_53: "bf16[128, 256, 2, 2][1024, 4, 2, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_32, torch.bfloat16);  add_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:94 in forward, code: out = self.relu(out)
        relu_11: "bf16[128, 256, 2, 2][1024, 4, 2, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_53);  convert_element_type_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:96 in forward, code: out = self.conv2(out)
        convolution_14: "bf16[128, 256, 2, 2][1024, 4, 2, 1]cuda:0" = torch.ops.aten.convolution.default(relu_11, arg43_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_11 = arg43_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:97 in forward, code: out = self.bn2(out)
        convert_element_type_56: "f32[128, 256, 2, 2][1024, 4, 2, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_14, torch.float32);  convolution_14 = None
        view_28: "f32[128, 32, 8, 4][1024, 32, 4, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_56, [128, 32, 8, 4]);  convert_element_type_56 = None
        var_mean_14 = torch.ops.aten.var_mean.correction(view_28, [2, 3], correction = 0, keepdim = True)
        getitem_30: "f32[128, 32, 1, 1][32, 1, 1, 1]cuda:0" = var_mean_14[0]
        getitem_31: "f32[128, 32, 1, 1][32, 1, 1, 1]cuda:0" = var_mean_14[1];  var_mean_14 = None
        sub_14: "f32[128, 32, 8, 4][1024, 32, 4, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_28, getitem_31);  view_28 = getitem_31 = None
        add_33: "f32[128, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_30, 1e-05);  getitem_30 = None
        rsqrt_14: "f32[128, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_33);  add_33 = None
        mul_28: "f32[128, 32, 8, 4][1024, 32, 4, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_14);  sub_14 = rsqrt_14 = None
        view_29: "f32[128, 256, 2, 2][1024, 4, 2, 1]cuda:0" = torch.ops.aten.reshape.default(mul_28, [128, 256, 2, 2]);  mul_28 = None
        unsqueeze_84: "bf16[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg44_1, 0);  arg44_1 = None
        unsqueeze_85: "bf16[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_84, 2);  unsqueeze_84 = None
        unsqueeze_86: "bf16[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_85, 3);  unsqueeze_85 = None
        mul_29: "f32[128, 256, 2, 2][1024, 4, 2, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_29, unsqueeze_86);  view_29 = unsqueeze_86 = None
        unsqueeze_87: "bf16[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg45_1, 0);  arg45_1 = None
        unsqueeze_88: "bf16[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_87, 2);  unsqueeze_87 = None
        unsqueeze_89: "bf16[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_88, 3);  unsqueeze_88 = None
        add_34: "f32[128, 256, 2, 2][1024, 4, 2, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_29, unsqueeze_89);  mul_29 = unsqueeze_89 = None
        convert_element_type_57: "bf16[128, 256, 2, 2][1024, 4, 2, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_34, torch.bfloat16);  add_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:102 in forward, code: out += identity
        add_35: "bf16[128, 256, 2, 2][1024, 4, 2, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_57, relu_10);  convert_element_type_57 = relu_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:103 in forward, code: out = self.relu(out)
        relu_12: "bf16[128, 256, 2, 2][1024, 4, 2, 1]cuda:0" = torch.ops.aten.relu.default(add_35);  add_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:92 in forward, code: out = self.conv1(x)
        convolution_15: "bf16[128, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.convolution.default(relu_12, arg46_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  arg46_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:93 in forward, code: out = self.bn1(out)
        convert_element_type_60: "f32[128, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_15, torch.float32);  convolution_15 = None
        view_30: "f32[128, 32, 16, 1][512, 16, 1, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_60, [128, 32, 16, 1]);  convert_element_type_60 = None
        var_mean_15 = torch.ops.aten.var_mean.correction(view_30, [2, 3], correction = 0, keepdim = True)
        getitem_32: "f32[128, 32, 1, 1][32, 1, 1, 1]cuda:0" = var_mean_15[0]
        getitem_33: "f32[128, 32, 1, 1][32, 1, 1, 1]cuda:0" = var_mean_15[1];  var_mean_15 = None
        sub_15: "f32[128, 32, 16, 1][512, 16, 1, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_30, getitem_33);  view_30 = getitem_33 = None
        add_36: "f32[128, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_32, 1e-05);  getitem_32 = None
        rsqrt_15: "f32[128, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_36);  add_36 = None
        mul_30: "f32[128, 32, 16, 1][512, 16, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_15);  sub_15 = rsqrt_15 = None
        view_31: "f32[128, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_30, [128, 512, 1, 1]);  mul_30 = None
        unsqueeze_90: "bf16[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg47_1, 0);  arg47_1 = None
        unsqueeze_91: "bf16[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_90, 2);  unsqueeze_90 = None
        unsqueeze_92: "bf16[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_91, 3);  unsqueeze_91 = None
        mul_31: "f32[128, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_31, unsqueeze_92);  view_31 = unsqueeze_92 = None
        unsqueeze_93: "bf16[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg48_1, 0);  arg48_1 = None
        unsqueeze_94: "bf16[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_93, 2);  unsqueeze_93 = None
        unsqueeze_95: "bf16[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_94, 3);  unsqueeze_94 = None
        add_37: "f32[128, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_31, unsqueeze_95);  mul_31 = unsqueeze_95 = None
        convert_element_type_61: "bf16[128, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_37, torch.bfloat16);  add_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:94 in forward, code: out = self.relu(out)
        relu_13: "bf16[128, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_61);  convert_element_type_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:96 in forward, code: out = self.conv2(out)
        convolution_16: "bf16[128, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.convolution.default(relu_13, arg49_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_13 = arg49_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:97 in forward, code: out = self.bn2(out)
        convert_element_type_64: "f32[128, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_16, torch.float32);  convolution_16 = None
        view_32: "f32[128, 32, 16, 1][512, 16, 1, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_64, [128, 32, 16, 1]);  convert_element_type_64 = None
        var_mean_16 = torch.ops.aten.var_mean.correction(view_32, [2, 3], correction = 0, keepdim = True)
        getitem_34: "f32[128, 32, 1, 1][32, 1, 1, 1]cuda:0" = var_mean_16[0]
        getitem_35: "f32[128, 32, 1, 1][32, 1, 1, 1]cuda:0" = var_mean_16[1];  var_mean_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:100 in forward, code: identity = self.downsample(x)
        convolution_17: "bf16[128, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.convolution.default(relu_12, arg52_1, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  relu_12 = arg52_1 = None
        convert_element_type_68: "f32[128, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_17, torch.float32);  convolution_17 = None
        view_34: "f32[128, 32, 16, 1][512, 16, 1, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_68, [128, 32, 16, 1]);  convert_element_type_68 = None
        var_mean_17 = torch.ops.aten.var_mean.correction(view_34, [2, 3], correction = 0, keepdim = True)
        getitem_36: "f32[128, 32, 1, 1][32, 1, 1, 1]cuda:0" = var_mean_17[0]
        getitem_37: "f32[128, 32, 1, 1][32, 1, 1, 1]cuda:0" = var_mean_17[1];  var_mean_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:97 in forward, code: out = self.bn2(out)
        sub_16: "f32[128, 32, 16, 1][512, 16, 1, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_32, getitem_35);  view_32 = getitem_35 = None
        add_38: "f32[128, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_34, 1e-05);  getitem_34 = None
        rsqrt_16: "f32[128, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_38);  add_38 = None
        mul_32: "f32[128, 32, 16, 1][512, 16, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_16);  sub_16 = rsqrt_16 = None
        view_33: "f32[128, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_32, [128, 512, 1, 1]);  mul_32 = None
        unsqueeze_96: "bf16[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg50_1, 0);  arg50_1 = None
        unsqueeze_97: "bf16[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_96, 2);  unsqueeze_96 = None
        unsqueeze_98: "bf16[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_97, 3);  unsqueeze_97 = None
        mul_33: "f32[128, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_33, unsqueeze_98);  view_33 = unsqueeze_98 = None
        unsqueeze_99: "bf16[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg51_1, 0);  arg51_1 = None
        unsqueeze_100: "bf16[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_99, 2);  unsqueeze_99 = None
        unsqueeze_101: "bf16[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_100, 3);  unsqueeze_100 = None
        add_39: "f32[128, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_33, unsqueeze_101);  mul_33 = unsqueeze_101 = None
        convert_element_type_65: "bf16[128, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_39, torch.bfloat16);  add_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:100 in forward, code: identity = self.downsample(x)
        sub_17: "f32[128, 32, 16, 1][512, 16, 1, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_34, getitem_37);  view_34 = getitem_37 = None
        add_40: "f32[128, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_36, 1e-05);  getitem_36 = None
        rsqrt_17: "f32[128, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_40);  add_40 = None
        mul_34: "f32[128, 32, 16, 1][512, 16, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_17);  sub_17 = rsqrt_17 = None
        view_35: "f32[128, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_34, [128, 512, 1, 1]);  mul_34 = None
        unsqueeze_102: "bf16[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg53_1, 0);  arg53_1 = None
        unsqueeze_103: "bf16[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_102, 2);  unsqueeze_102 = None
        unsqueeze_104: "bf16[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_103, 3);  unsqueeze_103 = None
        mul_35: "f32[128, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_35, unsqueeze_104);  view_35 = unsqueeze_104 = None
        unsqueeze_105: "bf16[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg54_1, 0);  arg54_1 = None
        unsqueeze_106: "bf16[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_105, 2);  unsqueeze_105 = None
        unsqueeze_107: "bf16[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_106, 3);  unsqueeze_106 = None
        add_41: "f32[128, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_35, unsqueeze_107);  mul_35 = unsqueeze_107 = None
        convert_element_type_69: "bf16[128, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_41, torch.bfloat16);  add_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:102 in forward, code: out += identity
        add_42: "bf16[128, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_65, convert_element_type_69);  convert_element_type_65 = convert_element_type_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:103 in forward, code: out = self.relu(out)
        relu_14: "bf16[128, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.relu.default(add_42);  add_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:92 in forward, code: out = self.conv1(x)
        convolution_18: "bf16[128, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.convolution.default(relu_14, arg55_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg55_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:93 in forward, code: out = self.bn1(out)
        convert_element_type_72: "f32[128, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_18, torch.float32);  convolution_18 = None
        view_36: "f32[128, 32, 16, 1][512, 16, 1, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_72, [128, 32, 16, 1]);  convert_element_type_72 = None
        var_mean_18 = torch.ops.aten.var_mean.correction(view_36, [2, 3], correction = 0, keepdim = True)
        getitem_38: "f32[128, 32, 1, 1][32, 1, 1, 1]cuda:0" = var_mean_18[0]
        getitem_39: "f32[128, 32, 1, 1][32, 1, 1, 1]cuda:0" = var_mean_18[1];  var_mean_18 = None
        sub_18: "f32[128, 32, 16, 1][512, 16, 1, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_36, getitem_39);  view_36 = getitem_39 = None
        add_43: "f32[128, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_38, 1e-05);  getitem_38 = None
        rsqrt_18: "f32[128, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_43);  add_43 = None
        mul_36: "f32[128, 32, 16, 1][512, 16, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_18);  sub_18 = rsqrt_18 = None
        view_37: "f32[128, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_36, [128, 512, 1, 1]);  mul_36 = None
        unsqueeze_108: "bf16[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg56_1, 0);  arg56_1 = None
        unsqueeze_109: "bf16[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_108, 2);  unsqueeze_108 = None
        unsqueeze_110: "bf16[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_109, 3);  unsqueeze_109 = None
        mul_37: "f32[128, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_37, unsqueeze_110);  view_37 = unsqueeze_110 = None
        unsqueeze_111: "bf16[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg57_1, 0);  arg57_1 = None
        unsqueeze_112: "bf16[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_111, 2);  unsqueeze_111 = None
        unsqueeze_113: "bf16[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_112, 3);  unsqueeze_112 = None
        add_44: "f32[128, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_37, unsqueeze_113);  mul_37 = unsqueeze_113 = None
        convert_element_type_73: "bf16[128, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_44, torch.bfloat16);  add_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:94 in forward, code: out = self.relu(out)
        relu_15: "bf16[128, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_73);  convert_element_type_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:96 in forward, code: out = self.conv2(out)
        convolution_19: "bf16[128, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.convolution.default(relu_15, arg58_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_15 = arg58_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:97 in forward, code: out = self.bn2(out)
        convert_element_type_76: "f32[128, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_19, torch.float32);  convolution_19 = None
        view_38: "f32[128, 32, 16, 1][512, 16, 1, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_76, [128, 32, 16, 1]);  convert_element_type_76 = None
        var_mean_19 = torch.ops.aten.var_mean.correction(view_38, [2, 3], correction = 0, keepdim = True)
        getitem_40: "f32[128, 32, 1, 1][32, 1, 1, 1]cuda:0" = var_mean_19[0]
        getitem_41: "f32[128, 32, 1, 1][32, 1, 1, 1]cuda:0" = var_mean_19[1];  var_mean_19 = None
        sub_19: "f32[128, 32, 16, 1][512, 16, 1, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_38, getitem_41);  view_38 = getitem_41 = None
        add_45: "f32[128, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_40, 1e-05);  getitem_40 = None
        rsqrt_19: "f32[128, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_45);  add_45 = None
        mul_38: "f32[128, 32, 16, 1][512, 16, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_19);  sub_19 = rsqrt_19 = None
        view_39: "f32[128, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_38, [128, 512, 1, 1]);  mul_38 = None
        unsqueeze_114: "bf16[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg59_1, 0);  arg59_1 = None
        unsqueeze_115: "bf16[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_114, 2);  unsqueeze_114 = None
        unsqueeze_116: "bf16[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_115, 3);  unsqueeze_115 = None
        mul_39: "f32[128, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_39, unsqueeze_116);  view_39 = unsqueeze_116 = None
        unsqueeze_117: "bf16[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg60_1, 0);  arg60_1 = None
        unsqueeze_118: "bf16[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_117, 2);  unsqueeze_117 = None
        unsqueeze_119: "bf16[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_118, 3);  unsqueeze_118 = None
        add_46: "f32[128, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_39, unsqueeze_119);  mul_39 = unsqueeze_119 = None
        convert_element_type_77: "bf16[128, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_46, torch.bfloat16);  add_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:102 in forward, code: out += identity
        add_47: "bf16[128, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_77, relu_14);  convert_element_type_77 = relu_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:103 in forward, code: out = self.relu(out)
        relu_16: "bf16[128, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.relu.default(add_47);  add_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:278 in _forward_impl, code: x = self.avgpool(x)
        mean: "bf16[128, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.mean.dim(relu_16, [-1, -2], True);  relu_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:279 in _forward_impl, code: x = torch.flatten(x, 1)
        view_40: "bf16[128, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mean, [128, 512]);  mean = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:280 in _forward_impl, code: x = self.fc(x)
        permute: "bf16[512, 1000][1, 512]cuda:0" = torch.ops.aten.permute.default(arg61_1, [1, 0]);  arg61_1 = None
        addmm: "bf16[128, 1000][1000, 1]cuda:0" = torch.ops.aten.addmm.default(arg62_1, view_40, permute);  arg62_1 = view_40 = permute = None
        return (addmm,)
