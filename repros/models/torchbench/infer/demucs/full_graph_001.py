class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "bf16[64, 2, 8][16, 8, 1]cuda:0", arg1_1: "bf16[64][1]cuda:0", arg2_1: "bf16[8, 2, 382788][765576, 382788, 1]cuda:0", arg3_1: "bf16[128, 64, 1][64, 1, 1]cuda:0", arg4_1: "bf16[128][1]cuda:0", arg5_1: "bf16[128, 64, 8][512, 8, 1]cuda:0", arg6_1: "bf16[128][1]cuda:0", arg7_1: "bf16[256, 128, 1][128, 1, 1]cuda:0", arg8_1: "bf16[256][1]cuda:0", arg9_1: "bf16[256, 128, 8][1024, 8, 1]cuda:0", arg10_1: "bf16[256][1]cuda:0", arg11_1: "bf16[512, 256, 1][256, 1, 1]cuda:0", arg12_1: "bf16[512][1]cuda:0", arg13_1: "bf16[512, 256, 8][2048, 8, 1]cuda:0", arg14_1: "bf16[512][1]cuda:0", arg15_1: "bf16[1024, 512, 1][512, 1, 1]cuda:0", arg16_1: "bf16[1024][1]cuda:0", arg17_1: "bf16[1024, 512, 8][4096, 8, 1]cuda:0", arg18_1: "bf16[1024][1]cuda:0", arg19_1: "bf16[2048, 1024, 1][1024, 1, 1]cuda:0", arg20_1: "bf16[2048][1]cuda:0", arg21_1: "bf16[2048, 1024, 8][8192, 8, 1]cuda:0", arg22_1: "bf16[2048][1]cuda:0", arg23_1: "bf16[4096, 2048, 1][2048, 1, 1]cuda:0", arg24_1: "bf16[4096][1]cuda:0"):
        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/demucs/demucs/model.py:215 in forward, code: x = encode(x)
        convolution: "bf16[8, 64, 95696][6124544, 95696, 1]cuda:0" = torch.ops.aten.convolution.default(arg2_1, arg0_1, arg1_1, [4], [0], [1], False, [0], 1);  arg2_1 = arg0_1 = arg1_1 = None
        relu: "bf16[8, 64, 95696][6124544, 95696, 1]cuda:0" = torch.ops.aten.relu.default(convolution);  convolution = None
        convolution_1: "bf16[8, 128, 95696][12249088, 95696, 1]cuda:0" = torch.ops.aten.convolution.default(relu, arg3_1, arg4_1, [1], [0], [1], False, [0], 1);  relu = arg3_1 = arg4_1 = None
        glu: "bf16[8, 64, 95696][6124544, 95696, 1]cuda:0" = torch.ops.aten.glu.default(convolution_1, 1);  convolution_1 = None
        convolution_2: "bf16[8, 128, 23923][3062144, 23923, 1]cuda:0" = torch.ops.aten.convolution.default(glu, arg5_1, arg6_1, [4], [0], [1], False, [0], 1);  arg5_1 = arg6_1 = None
        relu_1: "bf16[8, 128, 23923][3062144, 23923, 1]cuda:0" = torch.ops.aten.relu.default(convolution_2);  convolution_2 = None
        convolution_3: "bf16[8, 256, 23923][6124288, 23923, 1]cuda:0" = torch.ops.aten.convolution.default(relu_1, arg7_1, arg8_1, [1], [0], [1], False, [0], 1);  relu_1 = arg7_1 = arg8_1 = None
        glu_1: "bf16[8, 128, 23923][3062144, 23923, 1]cuda:0" = torch.ops.aten.glu.default(convolution_3, 1);  convolution_3 = None
        convolution_4: "bf16[8, 256, 5979][1530624, 5979, 1]cuda:0" = torch.ops.aten.convolution.default(glu_1, arg9_1, arg10_1, [4], [0], [1], False, [0], 1);  arg9_1 = arg10_1 = None
        relu_2: "bf16[8, 256, 5979][1530624, 5979, 1]cuda:0" = torch.ops.aten.relu.default(convolution_4);  convolution_4 = None
        convolution_5: "bf16[8, 512, 5979][3061248, 5979, 1]cuda:0" = torch.ops.aten.convolution.default(relu_2, arg11_1, arg12_1, [1], [0], [1], False, [0], 1);  relu_2 = arg11_1 = arg12_1 = None
        glu_2: "bf16[8, 256, 5979][1530624, 5979, 1]cuda:0" = torch.ops.aten.glu.default(convolution_5, 1);  convolution_5 = None
        convolution_6: "bf16[8, 512, 1493][764416, 1493, 1]cuda:0" = torch.ops.aten.convolution.default(glu_2, arg13_1, arg14_1, [4], [0], [1], False, [0], 1);  arg13_1 = arg14_1 = None
        relu_3: "bf16[8, 512, 1493][764416, 1493, 1]cuda:0" = torch.ops.aten.relu.default(convolution_6);  convolution_6 = None
        convolution_7: "bf16[8, 1024, 1493][1528832, 1493, 1]cuda:0" = torch.ops.aten.convolution.default(relu_3, arg15_1, arg16_1, [1], [0], [1], False, [0], 1);  relu_3 = arg15_1 = arg16_1 = None
        glu_3: "bf16[8, 512, 1493][764416, 1493, 1]cuda:0" = torch.ops.aten.glu.default(convolution_7, 1);  convolution_7 = None
        convolution_8: "bf16[8, 1024, 372][380928, 372, 1]cuda:0" = torch.ops.aten.convolution.default(glu_3, arg17_1, arg18_1, [4], [0], [1], False, [0], 1);  arg17_1 = arg18_1 = None
        relu_4: "bf16[8, 1024, 372][380928, 372, 1]cuda:0" = torch.ops.aten.relu.default(convolution_8);  convolution_8 = None
        convolution_9: "bf16[8, 2048, 372][761856, 372, 1]cuda:0" = torch.ops.aten.convolution.default(relu_4, arg19_1, arg20_1, [1], [0], [1], False, [0], 1);  relu_4 = arg19_1 = arg20_1 = None
        glu_4: "bf16[8, 1024, 372][380928, 372, 1]cuda:0" = torch.ops.aten.glu.default(convolution_9, 1);  convolution_9 = None
        convolution_10: "bf16[8, 2048, 92][188416, 92, 1]cuda:0" = torch.ops.aten.convolution.default(glu_4, arg21_1, arg22_1, [4], [0], [1], False, [0], 1);  arg21_1 = arg22_1 = None
        relu_5: "bf16[8, 2048, 92][188416, 92, 1]cuda:0" = torch.ops.aten.relu.default(convolution_10);  convolution_10 = None
        convolution_11: "bf16[8, 4096, 92][376832, 92, 1]cuda:0" = torch.ops.aten.convolution.default(relu_5, arg23_1, arg24_1, [1], [0], [1], False, [0], 1);  relu_5 = arg23_1 = arg24_1 = None
        glu_5: "bf16[8, 2048, 92][188416, 92, 1]cuda:0" = torch.ops.aten.glu.default(convolution_11, 1);  convolution_11 = None
        return (glu_5, glu, glu_1, glu_2, glu_3, glu_4)
