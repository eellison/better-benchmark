class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "bf16[1, 3, 256, 256][196608, 65536, 256, 1]cuda:0", arg1_1: "bf16[64, 3, 7, 7][147, 49, 7, 1]cuda:0", arg2_1: "bf16[64][1]cuda:0", arg3_1: "bf16[128, 64, 3, 3][576, 9, 3, 1]cuda:0", arg4_1: "bf16[128][1]cuda:0", arg5_1: "bf16[256, 128, 3, 3][1152, 9, 3, 1]cuda:0", arg6_1: "bf16[256][1]cuda:0", arg7_1: "bf16[256, 256, 3, 3][2304, 9, 3, 1]cuda:0", arg8_1: "bf16[256][1]cuda:0", arg9_1: "bf16[256, 256, 3, 3][2304, 9, 3, 1]cuda:0", arg10_1: "bf16[256][1]cuda:0", arg11_1: "bf16[256, 256, 3, 3][2304, 9, 3, 1]cuda:0", arg12_1: "bf16[256][1]cuda:0", arg13_1: "bf16[256, 256, 3, 3][2304, 9, 3, 1]cuda:0", arg14_1: "bf16[256][1]cuda:0", arg15_1: "bf16[256, 256, 3, 3][2304, 9, 3, 1]cuda:0", arg16_1: "bf16[256][1]cuda:0", arg17_1: "bf16[256, 256, 3, 3][2304, 9, 3, 1]cuda:0", arg18_1: "bf16[256][1]cuda:0", arg19_1: "bf16[256, 256, 3, 3][2304, 9, 3, 1]cuda:0", arg20_1: "bf16[256][1]cuda:0", arg21_1: "bf16[256, 256, 3, 3][2304, 9, 3, 1]cuda:0", arg22_1: "bf16[256][1]cuda:0", arg23_1: "bf16[256, 256, 3, 3][2304, 9, 3, 1]cuda:0", arg24_1: "bf16[256][1]cuda:0", arg25_1: "bf16[256, 256, 3, 3][2304, 9, 3, 1]cuda:0", arg26_1: "bf16[256][1]cuda:0", arg27_1: "bf16[256, 256, 3, 3][2304, 9, 3, 1]cuda:0", arg28_1: "bf16[256][1]cuda:0", arg29_1: "bf16[256, 256, 3, 3][2304, 9, 3, 1]cuda:0", arg30_1: "bf16[256][1]cuda:0", arg31_1: "bf16[256, 256, 3, 3][2304, 9, 3, 1]cuda:0", arg32_1: "bf16[256][1]cuda:0", arg33_1: "bf16[256, 256, 3, 3][2304, 9, 3, 1]cuda:0", arg34_1: "bf16[256][1]cuda:0", arg35_1: "bf16[256, 256, 3, 3][2304, 9, 3, 1]cuda:0", arg36_1: "bf16[256][1]cuda:0", arg37_1: "bf16[256, 256, 3, 3][2304, 9, 3, 1]cuda:0", arg38_1: "bf16[256][1]cuda:0", arg39_1: "bf16[256, 256, 3, 3][2304, 9, 3, 1]cuda:0", arg40_1: "bf16[256][1]cuda:0", arg41_1: "bf16[256, 256, 3, 3][2304, 9, 3, 1]cuda:0", arg42_1: "bf16[256][1]cuda:0", arg43_1: "bf16[256, 128, 3, 3][1152, 9, 3, 1]cuda:0", arg44_1: "bf16[128][1]cuda:0", arg45_1: "bf16[128, 64, 3, 3][576, 9, 3, 1]cuda:0", arg46_1: "bf16[64][1]cuda:0", arg47_1: "bf16[3, 64, 7, 7][3136, 49, 7, 1]cuda:0", arg48_1: "bf16[3][1]cuda:0"):
        # File: /tmp/pytorch-work/torch/nn/functional.py:5737 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        convert_element_type: "f32[1, 3, 256, 256][196608, 65536, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(arg0_1, torch.float32);  arg0_1 = None
        iota: "i64[262][1]cuda:0" = torch.ops.prims.iota.default(262, start = -3, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        abs_1: "i64[262][1]cuda:0" = torch.ops.aten.abs.default(iota);  iota = None
        sub: "i64[262][1]cuda:0" = torch.ops.aten.sub.Tensor(255, abs_1);  abs_1 = None
        abs_2: "i64[262][1]cuda:0" = torch.ops.aten.abs.default(sub);  sub = None
        sub_1: "i64[262][1]cuda:0" = torch.ops.aten.sub.Tensor(255, abs_2);  abs_2 = None
        _unsafe_index: "f32[1, 3, 262, 256][201216, 67072, 256, 1]cuda:0" = torch.ops.aten._unsafe_index.Tensor(convert_element_type, [None, None, sub_1, None]);  convert_element_type = sub_1 = None
        iota_1: "i64[262][1]cuda:0" = torch.ops.prims.iota.default(262, start = -3, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        abs_3: "i64[262][1]cuda:0" = torch.ops.aten.abs.default(iota_1);  iota_1 = None
        sub_2: "i64[262][1]cuda:0" = torch.ops.aten.sub.Tensor(255, abs_3);  abs_3 = None
        abs_4: "i64[262][1]cuda:0" = torch.ops.aten.abs.default(sub_2);  sub_2 = None
        sub_3: "i64[262][1]cuda:0" = torch.ops.aten.sub.Tensor(255, abs_4);  abs_4 = None
        _unsafe_index_1: "f32[1, 3, 262, 262][205932, 68644, 262, 1]cuda:0" = torch.ops.aten._unsafe_index.Tensor(_unsafe_index, [None, None, None, sub_3]);  _unsafe_index = sub_3 = None
        convert_element_type_1: "bf16[1, 3, 262, 262][205932, 68644, 262, 1]cuda:0" = torch.ops.prims.convert_element_type.default(_unsafe_index_1, torch.bfloat16);  _unsafe_index_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pytorch_CycleGAN_and_pix2pix/models/networks.py:496 in forward, code: return self.model(input)
        convolution: "bf16[1, 64, 256, 256][4194304, 65536, 256, 1]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_1, arg1_1, arg2_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_1 = arg1_1 = arg2_1 = None
        convert_element_type_2: "f32[1, 64, 256, 256][4194304, 65536, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution, torch.float32)
        var_mean = torch.ops.aten.var_mean.correction(convert_element_type_2, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_2 = None
        getitem: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = var_mean[0]
        getitem_1: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = var_mean[1];  var_mean = None
        sub_4: "f32[1, 64, 256, 256][4194304, 65536, 256, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution, getitem_1);  convolution = getitem_1 = None
        add: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add);  add = None
        mul: "f32[1, 64, 256, 256][4194304, 65536, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_4, rsqrt);  sub_4 = rsqrt = None
        convert_element_type_3: "bf16[1, 64, 256, 256][4194304, 65536, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul, torch.bfloat16);  mul = None
        relu: "bf16[1, 64, 256, 256][4194304, 65536, 256, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_3);  convert_element_type_3 = None
        convolution_1: "bf16[1, 128, 128, 128][2097152, 16384, 128, 1]cuda:0" = torch.ops.aten.convolution.default(relu, arg3_1, arg4_1, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  relu = arg3_1 = arg4_1 = None
        convert_element_type_4: "f32[1, 128, 128, 128][2097152, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_1, torch.float32)
        var_mean_1 = torch.ops.aten.var_mean.correction(convert_element_type_4, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_4 = None
        getitem_2: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_1[0]
        getitem_3: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_1[1];  var_mean_1 = None
        sub_5: "f32[1, 128, 128, 128][2097152, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_1, getitem_3);  convolution_1 = getitem_3 = None
        add_1: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_2, 1e-05);  getitem_2 = None
        rsqrt_1: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
        mul_1: "f32[1, 128, 128, 128][2097152, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_1);  sub_5 = rsqrt_1 = None
        convert_element_type_5: "bf16[1, 128, 128, 128][2097152, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1, torch.bfloat16);  mul_1 = None
        relu_1: "bf16[1, 128, 128, 128][2097152, 16384, 128, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_5);  convert_element_type_5 = None
        convolution_2: "bf16[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0" = torch.ops.aten.convolution.default(relu_1, arg5_1, arg6_1, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  relu_1 = arg5_1 = arg6_1 = None
        convert_element_type_6: "f32[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_2, torch.float32)
        var_mean_2 = torch.ops.aten.var_mean.correction(convert_element_type_6, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_6 = None
        getitem_4: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = var_mean_2[0]
        getitem_5: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = var_mean_2[1];  var_mean_2 = None
        sub_6: "f32[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_2, getitem_5);  convolution_2 = getitem_5 = None
        add_2: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_4, 1e-05);  getitem_4 = None
        rsqrt_2: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_2);  add_2 = None
        mul_2: "f32[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_2);  sub_6 = rsqrt_2 = None
        convert_element_type_7: "bf16[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_2, torch.bfloat16);  mul_2 = None
        relu_2: "bf16[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_7);  convert_element_type_7 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5737 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        convert_element_type_8: "f32[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(relu_2, torch.float32)
        iota_2: "i64[66][1]cuda:0" = torch.ops.prims.iota.default(66, start = -1, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        abs_5: "i64[66][1]cuda:0" = torch.ops.aten.abs.default(iota_2);  iota_2 = None
        sub_7: "i64[66][1]cuda:0" = torch.ops.aten.sub.Tensor(63, abs_5);  abs_5 = None
        abs_6: "i64[66][1]cuda:0" = torch.ops.aten.abs.default(sub_7);  sub_7 = None
        sub_8: "i64[66][1]cuda:0" = torch.ops.aten.sub.Tensor(63, abs_6);  abs_6 = None
        _unsafe_index_2: "f32[1, 256, 66, 64][1081344, 4224, 64, 1]cuda:0" = torch.ops.aten._unsafe_index.Tensor(convert_element_type_8, [None, None, sub_8, None]);  convert_element_type_8 = sub_8 = None
        iota_3: "i64[66][1]cuda:0" = torch.ops.prims.iota.default(66, start = -1, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        abs_7: "i64[66][1]cuda:0" = torch.ops.aten.abs.default(iota_3);  iota_3 = None
        sub_9: "i64[66][1]cuda:0" = torch.ops.aten.sub.Tensor(63, abs_7);  abs_7 = None
        abs_8: "i64[66][1]cuda:0" = torch.ops.aten.abs.default(sub_9);  sub_9 = None
        sub_10: "i64[66][1]cuda:0" = torch.ops.aten.sub.Tensor(63, abs_8);  abs_8 = None
        _unsafe_index_3: "f32[1, 256, 66, 66][1115136, 4356, 66, 1]cuda:0" = torch.ops.aten._unsafe_index.Tensor(_unsafe_index_2, [None, None, None, sub_10]);  _unsafe_index_2 = sub_10 = None
        convert_element_type_9: "bf16[1, 256, 66, 66][1115136, 4356, 66, 1]cuda:0" = torch.ops.prims.convert_element_type.default(_unsafe_index_3, torch.bfloat16);  _unsafe_index_3 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pytorch_CycleGAN_and_pix2pix/models/networks.py:564 in forward, code: out = x + self.conv_block(x)  # add skip connections
        convolution_3: "bf16[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_9, arg7_1, arg8_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_9 = arg7_1 = arg8_1 = None
        convert_element_type_10: "f32[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_3, torch.float32)
        var_mean_3 = torch.ops.aten.var_mean.correction(convert_element_type_10, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_10 = None
        getitem_6: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = var_mean_3[0]
        getitem_7: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = var_mean_3[1];  var_mean_3 = None
        sub_11: "f32[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_3, getitem_7);  convolution_3 = getitem_7 = None
        add_3: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_6, 1e-05);  getitem_6 = None
        rsqrt_3: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_3);  add_3 = None
        mul_3: "f32[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_3);  sub_11 = rsqrt_3 = None
        convert_element_type_11: "bf16[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_3, torch.bfloat16);  mul_3 = None
        relu_3: "bf16[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_11);  convert_element_type_11 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5737 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        convert_element_type_12: "f32[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(relu_3, torch.float32);  relu_3 = None
        iota_4: "i64[66][1]cuda:0" = torch.ops.prims.iota.default(66, start = -1, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        abs_9: "i64[66][1]cuda:0" = torch.ops.aten.abs.default(iota_4);  iota_4 = None
        sub_12: "i64[66][1]cuda:0" = torch.ops.aten.sub.Tensor(63, abs_9);  abs_9 = None
        abs_10: "i64[66][1]cuda:0" = torch.ops.aten.abs.default(sub_12);  sub_12 = None
        sub_13: "i64[66][1]cuda:0" = torch.ops.aten.sub.Tensor(63, abs_10);  abs_10 = None
        _unsafe_index_4: "f32[1, 256, 66, 64][1081344, 4224, 64, 1]cuda:0" = torch.ops.aten._unsafe_index.Tensor(convert_element_type_12, [None, None, sub_13, None]);  convert_element_type_12 = sub_13 = None
        iota_5: "i64[66][1]cuda:0" = torch.ops.prims.iota.default(66, start = -1, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        abs_11: "i64[66][1]cuda:0" = torch.ops.aten.abs.default(iota_5);  iota_5 = None
        sub_14: "i64[66][1]cuda:0" = torch.ops.aten.sub.Tensor(63, abs_11);  abs_11 = None
        abs_12: "i64[66][1]cuda:0" = torch.ops.aten.abs.default(sub_14);  sub_14 = None
        sub_15: "i64[66][1]cuda:0" = torch.ops.aten.sub.Tensor(63, abs_12);  abs_12 = None
        _unsafe_index_5: "f32[1, 256, 66, 66][1115136, 4356, 66, 1]cuda:0" = torch.ops.aten._unsafe_index.Tensor(_unsafe_index_4, [None, None, None, sub_15]);  _unsafe_index_4 = sub_15 = None
        convert_element_type_13: "bf16[1, 256, 66, 66][1115136, 4356, 66, 1]cuda:0" = torch.ops.prims.convert_element_type.default(_unsafe_index_5, torch.bfloat16);  _unsafe_index_5 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pytorch_CycleGAN_and_pix2pix/models/networks.py:564 in forward, code: out = x + self.conv_block(x)  # add skip connections
        convolution_4: "bf16[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_13, arg9_1, arg10_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_13 = arg9_1 = arg10_1 = None
        convert_element_type_14: "f32[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_4, torch.float32)
        var_mean_4 = torch.ops.aten.var_mean.correction(convert_element_type_14, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_14 = None
        getitem_8: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = var_mean_4[0]
        getitem_9: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = var_mean_4[1];  var_mean_4 = None
        sub_16: "f32[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_4, getitem_9);  convolution_4 = getitem_9 = None
        add_4: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_8, 1e-05);  getitem_8 = None
        rsqrt_4: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_4);  add_4 = None
        mul_4: "f32[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_4);  sub_16 = rsqrt_4 = None
        convert_element_type_15: "bf16[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_4, torch.bfloat16);  mul_4 = None
        add_5: "bf16[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(relu_2, convert_element_type_15);  relu_2 = convert_element_type_15 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5737 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        convert_element_type_16: "f32[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_5, torch.float32)
        iota_6: "i64[66][1]cuda:0" = torch.ops.prims.iota.default(66, start = -1, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        abs_13: "i64[66][1]cuda:0" = torch.ops.aten.abs.default(iota_6);  iota_6 = None
        sub_17: "i64[66][1]cuda:0" = torch.ops.aten.sub.Tensor(63, abs_13);  abs_13 = None
        abs_14: "i64[66][1]cuda:0" = torch.ops.aten.abs.default(sub_17);  sub_17 = None
        sub_18: "i64[66][1]cuda:0" = torch.ops.aten.sub.Tensor(63, abs_14);  abs_14 = None
        _unsafe_index_6: "f32[1, 256, 66, 64][1081344, 4224, 64, 1]cuda:0" = torch.ops.aten._unsafe_index.Tensor(convert_element_type_16, [None, None, sub_18, None]);  convert_element_type_16 = sub_18 = None
        iota_7: "i64[66][1]cuda:0" = torch.ops.prims.iota.default(66, start = -1, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        abs_15: "i64[66][1]cuda:0" = torch.ops.aten.abs.default(iota_7);  iota_7 = None
        sub_19: "i64[66][1]cuda:0" = torch.ops.aten.sub.Tensor(63, abs_15);  abs_15 = None
        abs_16: "i64[66][1]cuda:0" = torch.ops.aten.abs.default(sub_19);  sub_19 = None
        sub_20: "i64[66][1]cuda:0" = torch.ops.aten.sub.Tensor(63, abs_16);  abs_16 = None
        _unsafe_index_7: "f32[1, 256, 66, 66][1115136, 4356, 66, 1]cuda:0" = torch.ops.aten._unsafe_index.Tensor(_unsafe_index_6, [None, None, None, sub_20]);  _unsafe_index_6 = sub_20 = None
        convert_element_type_17: "bf16[1, 256, 66, 66][1115136, 4356, 66, 1]cuda:0" = torch.ops.prims.convert_element_type.default(_unsafe_index_7, torch.bfloat16);  _unsafe_index_7 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pytorch_CycleGAN_and_pix2pix/models/networks.py:564 in forward, code: out = x + self.conv_block(x)  # add skip connections
        convolution_5: "bf16[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_17, arg11_1, arg12_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_17 = arg11_1 = arg12_1 = None
        convert_element_type_18: "f32[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_5, torch.float32)
        var_mean_5 = torch.ops.aten.var_mean.correction(convert_element_type_18, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_18 = None
        getitem_10: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = var_mean_5[0]
        getitem_11: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = var_mean_5[1];  var_mean_5 = None
        sub_21: "f32[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_5, getitem_11);  convolution_5 = getitem_11 = None
        add_6: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_10, 1e-05);  getitem_10 = None
        rsqrt_5: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_6);  add_6 = None
        mul_5: "f32[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_5);  sub_21 = rsqrt_5 = None
        convert_element_type_19: "bf16[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_5, torch.bfloat16);  mul_5 = None
        relu_4: "bf16[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_19);  convert_element_type_19 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5737 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        convert_element_type_20: "f32[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(relu_4, torch.float32);  relu_4 = None
        iota_8: "i64[66][1]cuda:0" = torch.ops.prims.iota.default(66, start = -1, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        abs_17: "i64[66][1]cuda:0" = torch.ops.aten.abs.default(iota_8);  iota_8 = None
        sub_22: "i64[66][1]cuda:0" = torch.ops.aten.sub.Tensor(63, abs_17);  abs_17 = None
        abs_18: "i64[66][1]cuda:0" = torch.ops.aten.abs.default(sub_22);  sub_22 = None
        sub_23: "i64[66][1]cuda:0" = torch.ops.aten.sub.Tensor(63, abs_18);  abs_18 = None
        _unsafe_index_8: "f32[1, 256, 66, 64][1081344, 4224, 64, 1]cuda:0" = torch.ops.aten._unsafe_index.Tensor(convert_element_type_20, [None, None, sub_23, None]);  convert_element_type_20 = sub_23 = None
        iota_9: "i64[66][1]cuda:0" = torch.ops.prims.iota.default(66, start = -1, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        abs_19: "i64[66][1]cuda:0" = torch.ops.aten.abs.default(iota_9);  iota_9 = None
        sub_24: "i64[66][1]cuda:0" = torch.ops.aten.sub.Tensor(63, abs_19);  abs_19 = None
        abs_20: "i64[66][1]cuda:0" = torch.ops.aten.abs.default(sub_24);  sub_24 = None
        sub_25: "i64[66][1]cuda:0" = torch.ops.aten.sub.Tensor(63, abs_20);  abs_20 = None
        _unsafe_index_9: "f32[1, 256, 66, 66][1115136, 4356, 66, 1]cuda:0" = torch.ops.aten._unsafe_index.Tensor(_unsafe_index_8, [None, None, None, sub_25]);  _unsafe_index_8 = sub_25 = None
        convert_element_type_21: "bf16[1, 256, 66, 66][1115136, 4356, 66, 1]cuda:0" = torch.ops.prims.convert_element_type.default(_unsafe_index_9, torch.bfloat16);  _unsafe_index_9 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pytorch_CycleGAN_and_pix2pix/models/networks.py:564 in forward, code: out = x + self.conv_block(x)  # add skip connections
        convolution_6: "bf16[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_21, arg13_1, arg14_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_21 = arg13_1 = arg14_1 = None
        convert_element_type_22: "f32[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_6, torch.float32)
        var_mean_6 = torch.ops.aten.var_mean.correction(convert_element_type_22, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_22 = None
        getitem_12: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = var_mean_6[0]
        getitem_13: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = var_mean_6[1];  var_mean_6 = None
        sub_26: "f32[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_6, getitem_13);  convolution_6 = getitem_13 = None
        add_7: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_12, 1e-05);  getitem_12 = None
        rsqrt_6: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_7);  add_7 = None
        mul_6: "f32[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_6);  sub_26 = rsqrt_6 = None
        convert_element_type_23: "bf16[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_6, torch.bfloat16);  mul_6 = None
        add_8: "bf16[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(add_5, convert_element_type_23);  add_5 = convert_element_type_23 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5737 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        convert_element_type_24: "f32[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_8, torch.float32)
        iota_10: "i64[66][1]cuda:0" = torch.ops.prims.iota.default(66, start = -1, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        abs_21: "i64[66][1]cuda:0" = torch.ops.aten.abs.default(iota_10);  iota_10 = None
        sub_27: "i64[66][1]cuda:0" = torch.ops.aten.sub.Tensor(63, abs_21);  abs_21 = None
        abs_22: "i64[66][1]cuda:0" = torch.ops.aten.abs.default(sub_27);  sub_27 = None
        sub_28: "i64[66][1]cuda:0" = torch.ops.aten.sub.Tensor(63, abs_22);  abs_22 = None
        _unsafe_index_10: "f32[1, 256, 66, 64][1081344, 4224, 64, 1]cuda:0" = torch.ops.aten._unsafe_index.Tensor(convert_element_type_24, [None, None, sub_28, None]);  convert_element_type_24 = sub_28 = None
        iota_11: "i64[66][1]cuda:0" = torch.ops.prims.iota.default(66, start = -1, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        abs_23: "i64[66][1]cuda:0" = torch.ops.aten.abs.default(iota_11);  iota_11 = None
        sub_29: "i64[66][1]cuda:0" = torch.ops.aten.sub.Tensor(63, abs_23);  abs_23 = None
        abs_24: "i64[66][1]cuda:0" = torch.ops.aten.abs.default(sub_29);  sub_29 = None
        sub_30: "i64[66][1]cuda:0" = torch.ops.aten.sub.Tensor(63, abs_24);  abs_24 = None
        _unsafe_index_11: "f32[1, 256, 66, 66][1115136, 4356, 66, 1]cuda:0" = torch.ops.aten._unsafe_index.Tensor(_unsafe_index_10, [None, None, None, sub_30]);  _unsafe_index_10 = sub_30 = None
        convert_element_type_25: "bf16[1, 256, 66, 66][1115136, 4356, 66, 1]cuda:0" = torch.ops.prims.convert_element_type.default(_unsafe_index_11, torch.bfloat16);  _unsafe_index_11 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pytorch_CycleGAN_and_pix2pix/models/networks.py:564 in forward, code: out = x + self.conv_block(x)  # add skip connections
        convolution_7: "bf16[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_25, arg15_1, arg16_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_25 = arg15_1 = arg16_1 = None
        convert_element_type_26: "f32[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_7, torch.float32)
        var_mean_7 = torch.ops.aten.var_mean.correction(convert_element_type_26, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_26 = None
        getitem_14: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = var_mean_7[0]
        getitem_15: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = var_mean_7[1];  var_mean_7 = None
        sub_31: "f32[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_7, getitem_15);  convolution_7 = getitem_15 = None
        add_9: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_14, 1e-05);  getitem_14 = None
        rsqrt_7: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_9);  add_9 = None
        mul_7: "f32[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_31, rsqrt_7);  sub_31 = rsqrt_7 = None
        convert_element_type_27: "bf16[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_7, torch.bfloat16);  mul_7 = None
        relu_5: "bf16[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_27);  convert_element_type_27 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5737 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        convert_element_type_28: "f32[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(relu_5, torch.float32);  relu_5 = None
        iota_12: "i64[66][1]cuda:0" = torch.ops.prims.iota.default(66, start = -1, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        abs_25: "i64[66][1]cuda:0" = torch.ops.aten.abs.default(iota_12);  iota_12 = None
        sub_32: "i64[66][1]cuda:0" = torch.ops.aten.sub.Tensor(63, abs_25);  abs_25 = None
        abs_26: "i64[66][1]cuda:0" = torch.ops.aten.abs.default(sub_32);  sub_32 = None
        sub_33: "i64[66][1]cuda:0" = torch.ops.aten.sub.Tensor(63, abs_26);  abs_26 = None
        _unsafe_index_12: "f32[1, 256, 66, 64][1081344, 4224, 64, 1]cuda:0" = torch.ops.aten._unsafe_index.Tensor(convert_element_type_28, [None, None, sub_33, None]);  convert_element_type_28 = sub_33 = None
        iota_13: "i64[66][1]cuda:0" = torch.ops.prims.iota.default(66, start = -1, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        abs_27: "i64[66][1]cuda:0" = torch.ops.aten.abs.default(iota_13);  iota_13 = None
        sub_34: "i64[66][1]cuda:0" = torch.ops.aten.sub.Tensor(63, abs_27);  abs_27 = None
        abs_28: "i64[66][1]cuda:0" = torch.ops.aten.abs.default(sub_34);  sub_34 = None
        sub_35: "i64[66][1]cuda:0" = torch.ops.aten.sub.Tensor(63, abs_28);  abs_28 = None
        _unsafe_index_13: "f32[1, 256, 66, 66][1115136, 4356, 66, 1]cuda:0" = torch.ops.aten._unsafe_index.Tensor(_unsafe_index_12, [None, None, None, sub_35]);  _unsafe_index_12 = sub_35 = None
        convert_element_type_29: "bf16[1, 256, 66, 66][1115136, 4356, 66, 1]cuda:0" = torch.ops.prims.convert_element_type.default(_unsafe_index_13, torch.bfloat16);  _unsafe_index_13 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pytorch_CycleGAN_and_pix2pix/models/networks.py:564 in forward, code: out = x + self.conv_block(x)  # add skip connections
        convolution_8: "bf16[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_29, arg17_1, arg18_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_29 = arg17_1 = arg18_1 = None
        convert_element_type_30: "f32[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_8, torch.float32)
        var_mean_8 = torch.ops.aten.var_mean.correction(convert_element_type_30, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_30 = None
        getitem_16: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = var_mean_8[0]
        getitem_17: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = var_mean_8[1];  var_mean_8 = None
        sub_36: "f32[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_8, getitem_17);  convolution_8 = getitem_17 = None
        add_10: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_16, 1e-05);  getitem_16 = None
        rsqrt_8: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_10);  add_10 = None
        mul_8: "f32[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_8);  sub_36 = rsqrt_8 = None
        convert_element_type_31: "bf16[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_8, torch.bfloat16);  mul_8 = None
        add_11: "bf16[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(add_8, convert_element_type_31);  add_8 = convert_element_type_31 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5737 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        convert_element_type_32: "f32[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_11, torch.float32)
        iota_14: "i64[66][1]cuda:0" = torch.ops.prims.iota.default(66, start = -1, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        abs_29: "i64[66][1]cuda:0" = torch.ops.aten.abs.default(iota_14);  iota_14 = None
        sub_37: "i64[66][1]cuda:0" = torch.ops.aten.sub.Tensor(63, abs_29);  abs_29 = None
        abs_30: "i64[66][1]cuda:0" = torch.ops.aten.abs.default(sub_37);  sub_37 = None
        sub_38: "i64[66][1]cuda:0" = torch.ops.aten.sub.Tensor(63, abs_30);  abs_30 = None
        _unsafe_index_14: "f32[1, 256, 66, 64][1081344, 4224, 64, 1]cuda:0" = torch.ops.aten._unsafe_index.Tensor(convert_element_type_32, [None, None, sub_38, None]);  convert_element_type_32 = sub_38 = None
        iota_15: "i64[66][1]cuda:0" = torch.ops.prims.iota.default(66, start = -1, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        abs_31: "i64[66][1]cuda:0" = torch.ops.aten.abs.default(iota_15);  iota_15 = None
        sub_39: "i64[66][1]cuda:0" = torch.ops.aten.sub.Tensor(63, abs_31);  abs_31 = None
        abs_32: "i64[66][1]cuda:0" = torch.ops.aten.abs.default(sub_39);  sub_39 = None
        sub_40: "i64[66][1]cuda:0" = torch.ops.aten.sub.Tensor(63, abs_32);  abs_32 = None
        _unsafe_index_15: "f32[1, 256, 66, 66][1115136, 4356, 66, 1]cuda:0" = torch.ops.aten._unsafe_index.Tensor(_unsafe_index_14, [None, None, None, sub_40]);  _unsafe_index_14 = sub_40 = None
        convert_element_type_33: "bf16[1, 256, 66, 66][1115136, 4356, 66, 1]cuda:0" = torch.ops.prims.convert_element_type.default(_unsafe_index_15, torch.bfloat16);  _unsafe_index_15 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pytorch_CycleGAN_and_pix2pix/models/networks.py:564 in forward, code: out = x + self.conv_block(x)  # add skip connections
        convolution_9: "bf16[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_33, arg19_1, arg20_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_33 = arg19_1 = arg20_1 = None
        convert_element_type_34: "f32[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_9, torch.float32)
        var_mean_9 = torch.ops.aten.var_mean.correction(convert_element_type_34, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_34 = None
        getitem_18: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = var_mean_9[0]
        getitem_19: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = var_mean_9[1];  var_mean_9 = None
        sub_41: "f32[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_9, getitem_19);  convolution_9 = getitem_19 = None
        add_12: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_18, 1e-05);  getitem_18 = None
        rsqrt_9: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_12);  add_12 = None
        mul_9: "f32[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_41, rsqrt_9);  sub_41 = rsqrt_9 = None
        convert_element_type_35: "bf16[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_9, torch.bfloat16);  mul_9 = None
        relu_6: "bf16[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_35);  convert_element_type_35 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5737 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        convert_element_type_36: "f32[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(relu_6, torch.float32);  relu_6 = None
        iota_16: "i64[66][1]cuda:0" = torch.ops.prims.iota.default(66, start = -1, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        abs_33: "i64[66][1]cuda:0" = torch.ops.aten.abs.default(iota_16);  iota_16 = None
        sub_42: "i64[66][1]cuda:0" = torch.ops.aten.sub.Tensor(63, abs_33);  abs_33 = None
        abs_34: "i64[66][1]cuda:0" = torch.ops.aten.abs.default(sub_42);  sub_42 = None
        sub_43: "i64[66][1]cuda:0" = torch.ops.aten.sub.Tensor(63, abs_34);  abs_34 = None
        _unsafe_index_16: "f32[1, 256, 66, 64][1081344, 4224, 64, 1]cuda:0" = torch.ops.aten._unsafe_index.Tensor(convert_element_type_36, [None, None, sub_43, None]);  convert_element_type_36 = sub_43 = None
        iota_17: "i64[66][1]cuda:0" = torch.ops.prims.iota.default(66, start = -1, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        abs_35: "i64[66][1]cuda:0" = torch.ops.aten.abs.default(iota_17);  iota_17 = None
        sub_44: "i64[66][1]cuda:0" = torch.ops.aten.sub.Tensor(63, abs_35);  abs_35 = None
        abs_36: "i64[66][1]cuda:0" = torch.ops.aten.abs.default(sub_44);  sub_44 = None
        sub_45: "i64[66][1]cuda:0" = torch.ops.aten.sub.Tensor(63, abs_36);  abs_36 = None
        _unsafe_index_17: "f32[1, 256, 66, 66][1115136, 4356, 66, 1]cuda:0" = torch.ops.aten._unsafe_index.Tensor(_unsafe_index_16, [None, None, None, sub_45]);  _unsafe_index_16 = sub_45 = None
        convert_element_type_37: "bf16[1, 256, 66, 66][1115136, 4356, 66, 1]cuda:0" = torch.ops.prims.convert_element_type.default(_unsafe_index_17, torch.bfloat16);  _unsafe_index_17 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pytorch_CycleGAN_and_pix2pix/models/networks.py:564 in forward, code: out = x + self.conv_block(x)  # add skip connections
        convolution_10: "bf16[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_37, arg21_1, arg22_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_37 = arg21_1 = arg22_1 = None
        convert_element_type_38: "f32[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_10, torch.float32)
        var_mean_10 = torch.ops.aten.var_mean.correction(convert_element_type_38, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_38 = None
        getitem_20: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = var_mean_10[0]
        getitem_21: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = var_mean_10[1];  var_mean_10 = None
        sub_46: "f32[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_10, getitem_21);  convolution_10 = getitem_21 = None
        add_13: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_20, 1e-05);  getitem_20 = None
        rsqrt_10: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_13);  add_13 = None
        mul_10: "f32[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_46, rsqrt_10);  sub_46 = rsqrt_10 = None
        convert_element_type_39: "bf16[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_10, torch.bfloat16);  mul_10 = None
        add_14: "bf16[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(add_11, convert_element_type_39);  add_11 = convert_element_type_39 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5737 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        convert_element_type_40: "f32[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_14, torch.float32)
        iota_18: "i64[66][1]cuda:0" = torch.ops.prims.iota.default(66, start = -1, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        abs_37: "i64[66][1]cuda:0" = torch.ops.aten.abs.default(iota_18);  iota_18 = None
        sub_47: "i64[66][1]cuda:0" = torch.ops.aten.sub.Tensor(63, abs_37);  abs_37 = None
        abs_38: "i64[66][1]cuda:0" = torch.ops.aten.abs.default(sub_47);  sub_47 = None
        sub_48: "i64[66][1]cuda:0" = torch.ops.aten.sub.Tensor(63, abs_38);  abs_38 = None
        _unsafe_index_18: "f32[1, 256, 66, 64][1081344, 4224, 64, 1]cuda:0" = torch.ops.aten._unsafe_index.Tensor(convert_element_type_40, [None, None, sub_48, None]);  convert_element_type_40 = sub_48 = None
        iota_19: "i64[66][1]cuda:0" = torch.ops.prims.iota.default(66, start = -1, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        abs_39: "i64[66][1]cuda:0" = torch.ops.aten.abs.default(iota_19);  iota_19 = None
        sub_49: "i64[66][1]cuda:0" = torch.ops.aten.sub.Tensor(63, abs_39);  abs_39 = None
        abs_40: "i64[66][1]cuda:0" = torch.ops.aten.abs.default(sub_49);  sub_49 = None
        sub_50: "i64[66][1]cuda:0" = torch.ops.aten.sub.Tensor(63, abs_40);  abs_40 = None
        _unsafe_index_19: "f32[1, 256, 66, 66][1115136, 4356, 66, 1]cuda:0" = torch.ops.aten._unsafe_index.Tensor(_unsafe_index_18, [None, None, None, sub_50]);  _unsafe_index_18 = sub_50 = None
        convert_element_type_41: "bf16[1, 256, 66, 66][1115136, 4356, 66, 1]cuda:0" = torch.ops.prims.convert_element_type.default(_unsafe_index_19, torch.bfloat16);  _unsafe_index_19 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pytorch_CycleGAN_and_pix2pix/models/networks.py:564 in forward, code: out = x + self.conv_block(x)  # add skip connections
        convolution_11: "bf16[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_41, arg23_1, arg24_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_41 = arg23_1 = arg24_1 = None
        convert_element_type_42: "f32[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_11, torch.float32)
        var_mean_11 = torch.ops.aten.var_mean.correction(convert_element_type_42, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_42 = None
        getitem_22: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = var_mean_11[0]
        getitem_23: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = var_mean_11[1];  var_mean_11 = None
        sub_51: "f32[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_11, getitem_23);  convolution_11 = getitem_23 = None
        add_15: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_22, 1e-05);  getitem_22 = None
        rsqrt_11: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_15);  add_15 = None
        mul_11: "f32[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_51, rsqrt_11);  sub_51 = rsqrt_11 = None
        convert_element_type_43: "bf16[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_11, torch.bfloat16);  mul_11 = None
        relu_7: "bf16[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_43);  convert_element_type_43 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5737 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        convert_element_type_44: "f32[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(relu_7, torch.float32);  relu_7 = None
        iota_20: "i64[66][1]cuda:0" = torch.ops.prims.iota.default(66, start = -1, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        abs_41: "i64[66][1]cuda:0" = torch.ops.aten.abs.default(iota_20);  iota_20 = None
        sub_52: "i64[66][1]cuda:0" = torch.ops.aten.sub.Tensor(63, abs_41);  abs_41 = None
        abs_42: "i64[66][1]cuda:0" = torch.ops.aten.abs.default(sub_52);  sub_52 = None
        sub_53: "i64[66][1]cuda:0" = torch.ops.aten.sub.Tensor(63, abs_42);  abs_42 = None
        _unsafe_index_20: "f32[1, 256, 66, 64][1081344, 4224, 64, 1]cuda:0" = torch.ops.aten._unsafe_index.Tensor(convert_element_type_44, [None, None, sub_53, None]);  convert_element_type_44 = sub_53 = None
        iota_21: "i64[66][1]cuda:0" = torch.ops.prims.iota.default(66, start = -1, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        abs_43: "i64[66][1]cuda:0" = torch.ops.aten.abs.default(iota_21);  iota_21 = None
        sub_54: "i64[66][1]cuda:0" = torch.ops.aten.sub.Tensor(63, abs_43);  abs_43 = None
        abs_44: "i64[66][1]cuda:0" = torch.ops.aten.abs.default(sub_54);  sub_54 = None
        sub_55: "i64[66][1]cuda:0" = torch.ops.aten.sub.Tensor(63, abs_44);  abs_44 = None
        _unsafe_index_21: "f32[1, 256, 66, 66][1115136, 4356, 66, 1]cuda:0" = torch.ops.aten._unsafe_index.Tensor(_unsafe_index_20, [None, None, None, sub_55]);  _unsafe_index_20 = sub_55 = None
        convert_element_type_45: "bf16[1, 256, 66, 66][1115136, 4356, 66, 1]cuda:0" = torch.ops.prims.convert_element_type.default(_unsafe_index_21, torch.bfloat16);  _unsafe_index_21 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pytorch_CycleGAN_and_pix2pix/models/networks.py:564 in forward, code: out = x + self.conv_block(x)  # add skip connections
        convolution_12: "bf16[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_45, arg25_1, arg26_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_45 = arg25_1 = arg26_1 = None
        convert_element_type_46: "f32[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_12, torch.float32)
        var_mean_12 = torch.ops.aten.var_mean.correction(convert_element_type_46, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_46 = None
        getitem_24: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = var_mean_12[0]
        getitem_25: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = var_mean_12[1];  var_mean_12 = None
        sub_56: "f32[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_12, getitem_25);  convolution_12 = getitem_25 = None
        add_16: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_24, 1e-05);  getitem_24 = None
        rsqrt_12: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_16);  add_16 = None
        mul_12: "f32[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_56, rsqrt_12);  sub_56 = rsqrt_12 = None
        convert_element_type_47: "bf16[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_12, torch.bfloat16);  mul_12 = None
        add_17: "bf16[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(add_14, convert_element_type_47);  add_14 = convert_element_type_47 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5737 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        convert_element_type_48: "f32[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_17, torch.float32)
        iota_22: "i64[66][1]cuda:0" = torch.ops.prims.iota.default(66, start = -1, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        abs_45: "i64[66][1]cuda:0" = torch.ops.aten.abs.default(iota_22);  iota_22 = None
        sub_57: "i64[66][1]cuda:0" = torch.ops.aten.sub.Tensor(63, abs_45);  abs_45 = None
        abs_46: "i64[66][1]cuda:0" = torch.ops.aten.abs.default(sub_57);  sub_57 = None
        sub_58: "i64[66][1]cuda:0" = torch.ops.aten.sub.Tensor(63, abs_46);  abs_46 = None
        _unsafe_index_22: "f32[1, 256, 66, 64][1081344, 4224, 64, 1]cuda:0" = torch.ops.aten._unsafe_index.Tensor(convert_element_type_48, [None, None, sub_58, None]);  convert_element_type_48 = sub_58 = None
        iota_23: "i64[66][1]cuda:0" = torch.ops.prims.iota.default(66, start = -1, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        abs_47: "i64[66][1]cuda:0" = torch.ops.aten.abs.default(iota_23);  iota_23 = None
        sub_59: "i64[66][1]cuda:0" = torch.ops.aten.sub.Tensor(63, abs_47);  abs_47 = None
        abs_48: "i64[66][1]cuda:0" = torch.ops.aten.abs.default(sub_59);  sub_59 = None
        sub_60: "i64[66][1]cuda:0" = torch.ops.aten.sub.Tensor(63, abs_48);  abs_48 = None
        _unsafe_index_23: "f32[1, 256, 66, 66][1115136, 4356, 66, 1]cuda:0" = torch.ops.aten._unsafe_index.Tensor(_unsafe_index_22, [None, None, None, sub_60]);  _unsafe_index_22 = sub_60 = None
        convert_element_type_49: "bf16[1, 256, 66, 66][1115136, 4356, 66, 1]cuda:0" = torch.ops.prims.convert_element_type.default(_unsafe_index_23, torch.bfloat16);  _unsafe_index_23 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pytorch_CycleGAN_and_pix2pix/models/networks.py:564 in forward, code: out = x + self.conv_block(x)  # add skip connections
        convolution_13: "bf16[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_49, arg27_1, arg28_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_49 = arg27_1 = arg28_1 = None
        convert_element_type_50: "f32[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_13, torch.float32)
        var_mean_13 = torch.ops.aten.var_mean.correction(convert_element_type_50, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_50 = None
        getitem_26: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = var_mean_13[0]
        getitem_27: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = var_mean_13[1];  var_mean_13 = None
        sub_61: "f32[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_13, getitem_27);  convolution_13 = getitem_27 = None
        add_18: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_26, 1e-05);  getitem_26 = None
        rsqrt_13: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_18);  add_18 = None
        mul_13: "f32[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_61, rsqrt_13);  sub_61 = rsqrt_13 = None
        convert_element_type_51: "bf16[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_13, torch.bfloat16);  mul_13 = None
        relu_8: "bf16[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_51);  convert_element_type_51 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5737 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        convert_element_type_52: "f32[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(relu_8, torch.float32);  relu_8 = None
        iota_24: "i64[66][1]cuda:0" = torch.ops.prims.iota.default(66, start = -1, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        abs_49: "i64[66][1]cuda:0" = torch.ops.aten.abs.default(iota_24);  iota_24 = None
        sub_62: "i64[66][1]cuda:0" = torch.ops.aten.sub.Tensor(63, abs_49);  abs_49 = None
        abs_50: "i64[66][1]cuda:0" = torch.ops.aten.abs.default(sub_62);  sub_62 = None
        sub_63: "i64[66][1]cuda:0" = torch.ops.aten.sub.Tensor(63, abs_50);  abs_50 = None
        _unsafe_index_24: "f32[1, 256, 66, 64][1081344, 4224, 64, 1]cuda:0" = torch.ops.aten._unsafe_index.Tensor(convert_element_type_52, [None, None, sub_63, None]);  convert_element_type_52 = sub_63 = None
        iota_25: "i64[66][1]cuda:0" = torch.ops.prims.iota.default(66, start = -1, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        abs_51: "i64[66][1]cuda:0" = torch.ops.aten.abs.default(iota_25);  iota_25 = None
        sub_64: "i64[66][1]cuda:0" = torch.ops.aten.sub.Tensor(63, abs_51);  abs_51 = None
        abs_52: "i64[66][1]cuda:0" = torch.ops.aten.abs.default(sub_64);  sub_64 = None
        sub_65: "i64[66][1]cuda:0" = torch.ops.aten.sub.Tensor(63, abs_52);  abs_52 = None
        _unsafe_index_25: "f32[1, 256, 66, 66][1115136, 4356, 66, 1]cuda:0" = torch.ops.aten._unsafe_index.Tensor(_unsafe_index_24, [None, None, None, sub_65]);  _unsafe_index_24 = sub_65 = None
        convert_element_type_53: "bf16[1, 256, 66, 66][1115136, 4356, 66, 1]cuda:0" = torch.ops.prims.convert_element_type.default(_unsafe_index_25, torch.bfloat16);  _unsafe_index_25 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pytorch_CycleGAN_and_pix2pix/models/networks.py:564 in forward, code: out = x + self.conv_block(x)  # add skip connections
        convolution_14: "bf16[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_53, arg29_1, arg30_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_53 = arg29_1 = arg30_1 = None
        convert_element_type_54: "f32[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_14, torch.float32)
        var_mean_14 = torch.ops.aten.var_mean.correction(convert_element_type_54, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_54 = None
        getitem_28: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = var_mean_14[0]
        getitem_29: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = var_mean_14[1];  var_mean_14 = None
        sub_66: "f32[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_14, getitem_29);  convolution_14 = getitem_29 = None
        add_19: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_28, 1e-05);  getitem_28 = None
        rsqrt_14: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_19);  add_19 = None
        mul_14: "f32[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_66, rsqrt_14);  sub_66 = rsqrt_14 = None
        convert_element_type_55: "bf16[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_14, torch.bfloat16);  mul_14 = None
        add_20: "bf16[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(add_17, convert_element_type_55);  add_17 = convert_element_type_55 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5737 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        convert_element_type_56: "f32[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_20, torch.float32)
        iota_26: "i64[66][1]cuda:0" = torch.ops.prims.iota.default(66, start = -1, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        abs_53: "i64[66][1]cuda:0" = torch.ops.aten.abs.default(iota_26);  iota_26 = None
        sub_67: "i64[66][1]cuda:0" = torch.ops.aten.sub.Tensor(63, abs_53);  abs_53 = None
        abs_54: "i64[66][1]cuda:0" = torch.ops.aten.abs.default(sub_67);  sub_67 = None
        sub_68: "i64[66][1]cuda:0" = torch.ops.aten.sub.Tensor(63, abs_54);  abs_54 = None
        _unsafe_index_26: "f32[1, 256, 66, 64][1081344, 4224, 64, 1]cuda:0" = torch.ops.aten._unsafe_index.Tensor(convert_element_type_56, [None, None, sub_68, None]);  convert_element_type_56 = sub_68 = None
        iota_27: "i64[66][1]cuda:0" = torch.ops.prims.iota.default(66, start = -1, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        abs_55: "i64[66][1]cuda:0" = torch.ops.aten.abs.default(iota_27);  iota_27 = None
        sub_69: "i64[66][1]cuda:0" = torch.ops.aten.sub.Tensor(63, abs_55);  abs_55 = None
        abs_56: "i64[66][1]cuda:0" = torch.ops.aten.abs.default(sub_69);  sub_69 = None
        sub_70: "i64[66][1]cuda:0" = torch.ops.aten.sub.Tensor(63, abs_56);  abs_56 = None
        _unsafe_index_27: "f32[1, 256, 66, 66][1115136, 4356, 66, 1]cuda:0" = torch.ops.aten._unsafe_index.Tensor(_unsafe_index_26, [None, None, None, sub_70]);  _unsafe_index_26 = sub_70 = None
        convert_element_type_57: "bf16[1, 256, 66, 66][1115136, 4356, 66, 1]cuda:0" = torch.ops.prims.convert_element_type.default(_unsafe_index_27, torch.bfloat16);  _unsafe_index_27 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pytorch_CycleGAN_and_pix2pix/models/networks.py:564 in forward, code: out = x + self.conv_block(x)  # add skip connections
        convolution_15: "bf16[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_57, arg31_1, arg32_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_57 = arg31_1 = arg32_1 = None
        convert_element_type_58: "f32[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_15, torch.float32)
        var_mean_15 = torch.ops.aten.var_mean.correction(convert_element_type_58, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_58 = None
        getitem_30: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = var_mean_15[0]
        getitem_31: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = var_mean_15[1];  var_mean_15 = None
        sub_71: "f32[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_15, getitem_31);  convolution_15 = getitem_31 = None
        add_21: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_30, 1e-05);  getitem_30 = None
        rsqrt_15: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_21);  add_21 = None
        mul_15: "f32[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_71, rsqrt_15);  sub_71 = rsqrt_15 = None
        convert_element_type_59: "bf16[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_15, torch.bfloat16);  mul_15 = None
        relu_9: "bf16[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_59);  convert_element_type_59 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5737 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        convert_element_type_60: "f32[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(relu_9, torch.float32);  relu_9 = None
        iota_28: "i64[66][1]cuda:0" = torch.ops.prims.iota.default(66, start = -1, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        abs_57: "i64[66][1]cuda:0" = torch.ops.aten.abs.default(iota_28);  iota_28 = None
        sub_72: "i64[66][1]cuda:0" = torch.ops.aten.sub.Tensor(63, abs_57);  abs_57 = None
        abs_58: "i64[66][1]cuda:0" = torch.ops.aten.abs.default(sub_72);  sub_72 = None
        sub_73: "i64[66][1]cuda:0" = torch.ops.aten.sub.Tensor(63, abs_58);  abs_58 = None
        _unsafe_index_28: "f32[1, 256, 66, 64][1081344, 4224, 64, 1]cuda:0" = torch.ops.aten._unsafe_index.Tensor(convert_element_type_60, [None, None, sub_73, None]);  convert_element_type_60 = sub_73 = None
        iota_29: "i64[66][1]cuda:0" = torch.ops.prims.iota.default(66, start = -1, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        abs_59: "i64[66][1]cuda:0" = torch.ops.aten.abs.default(iota_29);  iota_29 = None
        sub_74: "i64[66][1]cuda:0" = torch.ops.aten.sub.Tensor(63, abs_59);  abs_59 = None
        abs_60: "i64[66][1]cuda:0" = torch.ops.aten.abs.default(sub_74);  sub_74 = None
        sub_75: "i64[66][1]cuda:0" = torch.ops.aten.sub.Tensor(63, abs_60);  abs_60 = None
        _unsafe_index_29: "f32[1, 256, 66, 66][1115136, 4356, 66, 1]cuda:0" = torch.ops.aten._unsafe_index.Tensor(_unsafe_index_28, [None, None, None, sub_75]);  _unsafe_index_28 = sub_75 = None
        convert_element_type_61: "bf16[1, 256, 66, 66][1115136, 4356, 66, 1]cuda:0" = torch.ops.prims.convert_element_type.default(_unsafe_index_29, torch.bfloat16);  _unsafe_index_29 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pytorch_CycleGAN_and_pix2pix/models/networks.py:564 in forward, code: out = x + self.conv_block(x)  # add skip connections
        convolution_16: "bf16[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_61, arg33_1, arg34_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_61 = arg33_1 = arg34_1 = None
        convert_element_type_62: "f32[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_16, torch.float32)
        var_mean_16 = torch.ops.aten.var_mean.correction(convert_element_type_62, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_62 = None
        getitem_32: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = var_mean_16[0]
        getitem_33: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = var_mean_16[1];  var_mean_16 = None
        sub_76: "f32[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_16, getitem_33);  convolution_16 = getitem_33 = None
        add_22: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_32, 1e-05);  getitem_32 = None
        rsqrt_16: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_22);  add_22 = None
        mul_16: "f32[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_76, rsqrt_16);  sub_76 = rsqrt_16 = None
        convert_element_type_63: "bf16[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_16, torch.bfloat16);  mul_16 = None
        add_23: "bf16[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(add_20, convert_element_type_63);  add_20 = convert_element_type_63 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5737 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        convert_element_type_64: "f32[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_23, torch.float32)
        iota_30: "i64[66][1]cuda:0" = torch.ops.prims.iota.default(66, start = -1, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        abs_61: "i64[66][1]cuda:0" = torch.ops.aten.abs.default(iota_30);  iota_30 = None
        sub_77: "i64[66][1]cuda:0" = torch.ops.aten.sub.Tensor(63, abs_61);  abs_61 = None
        abs_62: "i64[66][1]cuda:0" = torch.ops.aten.abs.default(sub_77);  sub_77 = None
        sub_78: "i64[66][1]cuda:0" = torch.ops.aten.sub.Tensor(63, abs_62);  abs_62 = None
        _unsafe_index_30: "f32[1, 256, 66, 64][1081344, 4224, 64, 1]cuda:0" = torch.ops.aten._unsafe_index.Tensor(convert_element_type_64, [None, None, sub_78, None]);  convert_element_type_64 = sub_78 = None
        iota_31: "i64[66][1]cuda:0" = torch.ops.prims.iota.default(66, start = -1, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        abs_63: "i64[66][1]cuda:0" = torch.ops.aten.abs.default(iota_31);  iota_31 = None
        sub_79: "i64[66][1]cuda:0" = torch.ops.aten.sub.Tensor(63, abs_63);  abs_63 = None
        abs_64: "i64[66][1]cuda:0" = torch.ops.aten.abs.default(sub_79);  sub_79 = None
        sub_80: "i64[66][1]cuda:0" = torch.ops.aten.sub.Tensor(63, abs_64);  abs_64 = None
        _unsafe_index_31: "f32[1, 256, 66, 66][1115136, 4356, 66, 1]cuda:0" = torch.ops.aten._unsafe_index.Tensor(_unsafe_index_30, [None, None, None, sub_80]);  _unsafe_index_30 = sub_80 = None
        convert_element_type_65: "bf16[1, 256, 66, 66][1115136, 4356, 66, 1]cuda:0" = torch.ops.prims.convert_element_type.default(_unsafe_index_31, torch.bfloat16);  _unsafe_index_31 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pytorch_CycleGAN_and_pix2pix/models/networks.py:564 in forward, code: out = x + self.conv_block(x)  # add skip connections
        convolution_17: "bf16[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_65, arg35_1, arg36_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_65 = arg35_1 = arg36_1 = None
        convert_element_type_66: "f32[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_17, torch.float32)
        var_mean_17 = torch.ops.aten.var_mean.correction(convert_element_type_66, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_66 = None
        getitem_34: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = var_mean_17[0]
        getitem_35: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = var_mean_17[1];  var_mean_17 = None
        sub_81: "f32[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_17, getitem_35);  convolution_17 = getitem_35 = None
        add_24: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_34, 1e-05);  getitem_34 = None
        rsqrt_17: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_24);  add_24 = None
        mul_17: "f32[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_81, rsqrt_17);  sub_81 = rsqrt_17 = None
        convert_element_type_67: "bf16[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_17, torch.bfloat16);  mul_17 = None
        relu_10: "bf16[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_67);  convert_element_type_67 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5737 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        convert_element_type_68: "f32[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(relu_10, torch.float32);  relu_10 = None
        iota_32: "i64[66][1]cuda:0" = torch.ops.prims.iota.default(66, start = -1, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        abs_65: "i64[66][1]cuda:0" = torch.ops.aten.abs.default(iota_32);  iota_32 = None
        sub_82: "i64[66][1]cuda:0" = torch.ops.aten.sub.Tensor(63, abs_65);  abs_65 = None
        abs_66: "i64[66][1]cuda:0" = torch.ops.aten.abs.default(sub_82);  sub_82 = None
        sub_83: "i64[66][1]cuda:0" = torch.ops.aten.sub.Tensor(63, abs_66);  abs_66 = None
        _unsafe_index_32: "f32[1, 256, 66, 64][1081344, 4224, 64, 1]cuda:0" = torch.ops.aten._unsafe_index.Tensor(convert_element_type_68, [None, None, sub_83, None]);  convert_element_type_68 = sub_83 = None
        iota_33: "i64[66][1]cuda:0" = torch.ops.prims.iota.default(66, start = -1, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        abs_67: "i64[66][1]cuda:0" = torch.ops.aten.abs.default(iota_33);  iota_33 = None
        sub_84: "i64[66][1]cuda:0" = torch.ops.aten.sub.Tensor(63, abs_67);  abs_67 = None
        abs_68: "i64[66][1]cuda:0" = torch.ops.aten.abs.default(sub_84);  sub_84 = None
        sub_85: "i64[66][1]cuda:0" = torch.ops.aten.sub.Tensor(63, abs_68);  abs_68 = None
        _unsafe_index_33: "f32[1, 256, 66, 66][1115136, 4356, 66, 1]cuda:0" = torch.ops.aten._unsafe_index.Tensor(_unsafe_index_32, [None, None, None, sub_85]);  _unsafe_index_32 = sub_85 = None
        convert_element_type_69: "bf16[1, 256, 66, 66][1115136, 4356, 66, 1]cuda:0" = torch.ops.prims.convert_element_type.default(_unsafe_index_33, torch.bfloat16);  _unsafe_index_33 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pytorch_CycleGAN_and_pix2pix/models/networks.py:564 in forward, code: out = x + self.conv_block(x)  # add skip connections
        convolution_18: "bf16[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_69, arg37_1, arg38_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_69 = arg37_1 = arg38_1 = None
        convert_element_type_70: "f32[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_18, torch.float32)
        var_mean_18 = torch.ops.aten.var_mean.correction(convert_element_type_70, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_70 = None
        getitem_36: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = var_mean_18[0]
        getitem_37: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = var_mean_18[1];  var_mean_18 = None
        sub_86: "f32[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_18, getitem_37);  convolution_18 = getitem_37 = None
        add_25: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_36, 1e-05);  getitem_36 = None
        rsqrt_18: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_25);  add_25 = None
        mul_18: "f32[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_86, rsqrt_18);  sub_86 = rsqrt_18 = None
        convert_element_type_71: "bf16[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_18, torch.bfloat16);  mul_18 = None
        add_26: "bf16[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(add_23, convert_element_type_71);  add_23 = convert_element_type_71 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5737 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        convert_element_type_72: "f32[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_26, torch.float32)
        iota_34: "i64[66][1]cuda:0" = torch.ops.prims.iota.default(66, start = -1, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        abs_69: "i64[66][1]cuda:0" = torch.ops.aten.abs.default(iota_34);  iota_34 = None
        sub_87: "i64[66][1]cuda:0" = torch.ops.aten.sub.Tensor(63, abs_69);  abs_69 = None
        abs_70: "i64[66][1]cuda:0" = torch.ops.aten.abs.default(sub_87);  sub_87 = None
        sub_88: "i64[66][1]cuda:0" = torch.ops.aten.sub.Tensor(63, abs_70);  abs_70 = None
        _unsafe_index_34: "f32[1, 256, 66, 64][1081344, 4224, 64, 1]cuda:0" = torch.ops.aten._unsafe_index.Tensor(convert_element_type_72, [None, None, sub_88, None]);  convert_element_type_72 = sub_88 = None
        iota_35: "i64[66][1]cuda:0" = torch.ops.prims.iota.default(66, start = -1, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        abs_71: "i64[66][1]cuda:0" = torch.ops.aten.abs.default(iota_35);  iota_35 = None
        sub_89: "i64[66][1]cuda:0" = torch.ops.aten.sub.Tensor(63, abs_71);  abs_71 = None
        abs_72: "i64[66][1]cuda:0" = torch.ops.aten.abs.default(sub_89);  sub_89 = None
        sub_90: "i64[66][1]cuda:0" = torch.ops.aten.sub.Tensor(63, abs_72);  abs_72 = None
        _unsafe_index_35: "f32[1, 256, 66, 66][1115136, 4356, 66, 1]cuda:0" = torch.ops.aten._unsafe_index.Tensor(_unsafe_index_34, [None, None, None, sub_90]);  _unsafe_index_34 = sub_90 = None
        convert_element_type_73: "bf16[1, 256, 66, 66][1115136, 4356, 66, 1]cuda:0" = torch.ops.prims.convert_element_type.default(_unsafe_index_35, torch.bfloat16);  _unsafe_index_35 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pytorch_CycleGAN_and_pix2pix/models/networks.py:564 in forward, code: out = x + self.conv_block(x)  # add skip connections
        convolution_19: "bf16[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_73, arg39_1, arg40_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_73 = arg39_1 = arg40_1 = None
        convert_element_type_74: "f32[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_19, torch.float32)
        var_mean_19 = torch.ops.aten.var_mean.correction(convert_element_type_74, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_74 = None
        getitem_38: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = var_mean_19[0]
        getitem_39: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = var_mean_19[1];  var_mean_19 = None
        sub_91: "f32[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_19, getitem_39);  convolution_19 = getitem_39 = None
        add_27: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_38, 1e-05);  getitem_38 = None
        rsqrt_19: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_27);  add_27 = None
        mul_19: "f32[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_91, rsqrt_19);  sub_91 = rsqrt_19 = None
        convert_element_type_75: "bf16[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_19, torch.bfloat16);  mul_19 = None
        relu_11: "bf16[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_75);  convert_element_type_75 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5737 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        convert_element_type_76: "f32[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(relu_11, torch.float32);  relu_11 = None
        iota_36: "i64[66][1]cuda:0" = torch.ops.prims.iota.default(66, start = -1, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        abs_73: "i64[66][1]cuda:0" = torch.ops.aten.abs.default(iota_36);  iota_36 = None
        sub_92: "i64[66][1]cuda:0" = torch.ops.aten.sub.Tensor(63, abs_73);  abs_73 = None
        abs_74: "i64[66][1]cuda:0" = torch.ops.aten.abs.default(sub_92);  sub_92 = None
        sub_93: "i64[66][1]cuda:0" = torch.ops.aten.sub.Tensor(63, abs_74);  abs_74 = None
        _unsafe_index_36: "f32[1, 256, 66, 64][1081344, 4224, 64, 1]cuda:0" = torch.ops.aten._unsafe_index.Tensor(convert_element_type_76, [None, None, sub_93, None]);  convert_element_type_76 = sub_93 = None
        iota_37: "i64[66][1]cuda:0" = torch.ops.prims.iota.default(66, start = -1, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        abs_75: "i64[66][1]cuda:0" = torch.ops.aten.abs.default(iota_37);  iota_37 = None
        sub_94: "i64[66][1]cuda:0" = torch.ops.aten.sub.Tensor(63, abs_75);  abs_75 = None
        abs_76: "i64[66][1]cuda:0" = torch.ops.aten.abs.default(sub_94);  sub_94 = None
        sub_95: "i64[66][1]cuda:0" = torch.ops.aten.sub.Tensor(63, abs_76);  abs_76 = None
        _unsafe_index_37: "f32[1, 256, 66, 66][1115136, 4356, 66, 1]cuda:0" = torch.ops.aten._unsafe_index.Tensor(_unsafe_index_36, [None, None, None, sub_95]);  _unsafe_index_36 = sub_95 = None
        convert_element_type_77: "bf16[1, 256, 66, 66][1115136, 4356, 66, 1]cuda:0" = torch.ops.prims.convert_element_type.default(_unsafe_index_37, torch.bfloat16);  _unsafe_index_37 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pytorch_CycleGAN_and_pix2pix/models/networks.py:564 in forward, code: out = x + self.conv_block(x)  # add skip connections
        convolution_20: "bf16[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_77, arg41_1, arg42_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_77 = arg41_1 = arg42_1 = None
        convert_element_type_78: "f32[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_20, torch.float32)
        var_mean_20 = torch.ops.aten.var_mean.correction(convert_element_type_78, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_78 = None
        getitem_40: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = var_mean_20[0]
        getitem_41: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = var_mean_20[1];  var_mean_20 = None
        sub_96: "f32[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_20, getitem_41);  convolution_20 = getitem_41 = None
        add_28: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_40, 1e-05);  getitem_40 = None
        rsqrt_20: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_28);  add_28 = None
        mul_20: "f32[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_96, rsqrt_20);  sub_96 = rsqrt_20 = None
        convert_element_type_79: "bf16[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_20, torch.bfloat16);  mul_20 = None
        add_29: "bf16[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(add_26, convert_element_type_79);  add_26 = convert_element_type_79 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pytorch_CycleGAN_and_pix2pix/models/networks.py:496 in forward, code: return self.model(input)
        convolution_21: "bf16[1, 128, 128, 128][2097152, 16384, 128, 1]cuda:0" = torch.ops.aten.convolution.default(add_29, arg43_1, arg44_1, [2, 2], [1, 1], [1, 1], True, [1, 1], 1);  add_29 = arg43_1 = arg44_1 = None
        convert_element_type_80: "f32[1, 128, 128, 128][2097152, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_21, torch.float32)
        var_mean_21 = torch.ops.aten.var_mean.correction(convert_element_type_80, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_80 = None
        getitem_42: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_21[0]
        getitem_43: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_21[1];  var_mean_21 = None
        sub_97: "f32[1, 128, 128, 128][2097152, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_21, getitem_43);  convolution_21 = getitem_43 = None
        add_30: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_42, 1e-05);  getitem_42 = None
        rsqrt_21: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_30);  add_30 = None
        mul_21: "f32[1, 128, 128, 128][2097152, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_97, rsqrt_21);  sub_97 = rsqrt_21 = None
        convert_element_type_81: "bf16[1, 128, 128, 128][2097152, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_21, torch.bfloat16);  mul_21 = None
        relu_12: "bf16[1, 128, 128, 128][2097152, 16384, 128, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_81);  convert_element_type_81 = None
        convolution_22: "bf16[1, 64, 256, 256][4194304, 65536, 256, 1]cuda:0" = torch.ops.aten.convolution.default(relu_12, arg45_1, arg46_1, [2, 2], [1, 1], [1, 1], True, [1, 1], 1);  relu_12 = arg45_1 = arg46_1 = None
        convert_element_type_82: "f32[1, 64, 256, 256][4194304, 65536, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_22, torch.float32)
        var_mean_22 = torch.ops.aten.var_mean.correction(convert_element_type_82, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_82 = None
        getitem_44: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = var_mean_22[0]
        getitem_45: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = var_mean_22[1];  var_mean_22 = None
        sub_98: "f32[1, 64, 256, 256][4194304, 65536, 256, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_22, getitem_45);  convolution_22 = getitem_45 = None
        add_31: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_44, 1e-05);  getitem_44 = None
        rsqrt_22: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_31);  add_31 = None
        mul_22: "f32[1, 64, 256, 256][4194304, 65536, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_98, rsqrt_22);  sub_98 = rsqrt_22 = None
        convert_element_type_83: "bf16[1, 64, 256, 256][4194304, 65536, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_22, torch.bfloat16);  mul_22 = None
        relu_13: "bf16[1, 64, 256, 256][4194304, 65536, 256, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_83);  convert_element_type_83 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5737 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        convert_element_type_84: "f32[1, 64, 256, 256][4194304, 65536, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(relu_13, torch.float32);  relu_13 = None
        iota_38: "i64[262][1]cuda:0" = torch.ops.prims.iota.default(262, start = -3, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        abs_77: "i64[262][1]cuda:0" = torch.ops.aten.abs.default(iota_38);  iota_38 = None
        sub_99: "i64[262][1]cuda:0" = torch.ops.aten.sub.Tensor(255, abs_77);  abs_77 = None
        abs_78: "i64[262][1]cuda:0" = torch.ops.aten.abs.default(sub_99);  sub_99 = None
        sub_100: "i64[262][1]cuda:0" = torch.ops.aten.sub.Tensor(255, abs_78);  abs_78 = None
        _unsafe_index_38: "f32[1, 64, 262, 256][4292608, 67072, 256, 1]cuda:0" = torch.ops.aten._unsafe_index.Tensor(convert_element_type_84, [None, None, sub_100, None]);  convert_element_type_84 = sub_100 = None
        iota_39: "i64[262][1]cuda:0" = torch.ops.prims.iota.default(262, start = -3, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        abs_79: "i64[262][1]cuda:0" = torch.ops.aten.abs.default(iota_39);  iota_39 = None
        sub_101: "i64[262][1]cuda:0" = torch.ops.aten.sub.Tensor(255, abs_79);  abs_79 = None
        abs_80: "i64[262][1]cuda:0" = torch.ops.aten.abs.default(sub_101);  sub_101 = None
        sub_102: "i64[262][1]cuda:0" = torch.ops.aten.sub.Tensor(255, abs_80);  abs_80 = None
        _unsafe_index_39: "f32[1, 64, 262, 262][4393216, 68644, 262, 1]cuda:0" = torch.ops.aten._unsafe_index.Tensor(_unsafe_index_38, [None, None, None, sub_102]);  _unsafe_index_38 = sub_102 = None
        convert_element_type_85: "bf16[1, 64, 262, 262][4393216, 68644, 262, 1]cuda:0" = torch.ops.prims.convert_element_type.default(_unsafe_index_39, torch.bfloat16);  _unsafe_index_39 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pytorch_CycleGAN_and_pix2pix/models/networks.py:496 in forward, code: return self.model(input)
        convolution_23: "bf16[1, 3, 256, 256][196608, 65536, 256, 1]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_85, arg47_1, arg48_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_85 = arg47_1 = arg48_1 = None
        tanh: "bf16[1, 3, 256, 256][196608, 65536, 256, 1]cuda:0" = torch.ops.aten.tanh.default(convolution_23);  convolution_23 = None
        return (tanh,)
