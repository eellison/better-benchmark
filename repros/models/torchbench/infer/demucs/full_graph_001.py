class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "f32[64, 2, 8]", arg1_1: "f32[64]", arg2_1: "f32[8, 2, 382788]", arg3_1: "f32[128, 64, 1]", arg4_1: "f32[128]", arg5_1: "f32[128, 64, 8]", arg6_1: "f32[128]", arg7_1: "f32[256, 128, 1]", arg8_1: "f32[256]", arg9_1: "f32[256, 128, 8]", arg10_1: "f32[256]", arg11_1: "f32[512, 256, 1]", arg12_1: "f32[512]", arg13_1: "f32[512, 256, 8]", arg14_1: "f32[512]", arg15_1: "f32[1024, 512, 1]", arg16_1: "f32[1024]", arg17_1: "f32[1024, 512, 8]", arg18_1: "f32[1024]", arg19_1: "f32[2048, 1024, 1]", arg20_1: "f32[2048]", arg21_1: "f32[2048, 1024, 8]", arg22_1: "f32[2048]", arg23_1: "f32[4096, 2048, 1]", arg24_1: "f32[4096]"):
        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/demucs/demucs/model.py:215 in forward, code: x = encode(x)
        convolution: "f32[8, 64, 95696]" = torch.ops.aten.convolution.default(arg2_1, arg0_1, arg1_1, [4], [0], [1], False, [0], 1);  arg2_1 = arg0_1 = arg1_1 = None
        relu: "f32[8, 64, 95696]" = torch.ops.aten.relu.default(convolution);  convolution = None
        convolution_1: "f32[8, 128, 95696]" = torch.ops.aten.convolution.default(relu, arg3_1, arg4_1, [1], [0], [1], False, [0], 1);  relu = arg3_1 = arg4_1 = None
        glu: "f32[8, 64, 95696]" = torch.ops.aten.glu.default(convolution_1, 1);  convolution_1 = None
        convolution_2: "f32[8, 128, 23923]" = torch.ops.aten.convolution.default(glu, arg5_1, arg6_1, [4], [0], [1], False, [0], 1);  arg5_1 = arg6_1 = None
        relu_1: "f32[8, 128, 23923]" = torch.ops.aten.relu.default(convolution_2);  convolution_2 = None
        convolution_3: "f32[8, 256, 23923]" = torch.ops.aten.convolution.default(relu_1, arg7_1, arg8_1, [1], [0], [1], False, [0], 1);  relu_1 = arg7_1 = arg8_1 = None
        glu_1: "f32[8, 128, 23923]" = torch.ops.aten.glu.default(convolution_3, 1);  convolution_3 = None
        convolution_4: "f32[8, 256, 5979]" = torch.ops.aten.convolution.default(glu_1, arg9_1, arg10_1, [4], [0], [1], False, [0], 1);  arg9_1 = arg10_1 = None
        relu_2: "f32[8, 256, 5979]" = torch.ops.aten.relu.default(convolution_4);  convolution_4 = None
        convolution_5: "f32[8, 512, 5979]" = torch.ops.aten.convolution.default(relu_2, arg11_1, arg12_1, [1], [0], [1], False, [0], 1);  relu_2 = arg11_1 = arg12_1 = None
        glu_2: "f32[8, 256, 5979]" = torch.ops.aten.glu.default(convolution_5, 1);  convolution_5 = None
        convolution_6: "f32[8, 512, 1493]" = torch.ops.aten.convolution.default(glu_2, arg13_1, arg14_1, [4], [0], [1], False, [0], 1);  arg13_1 = arg14_1 = None
        relu_3: "f32[8, 512, 1493]" = torch.ops.aten.relu.default(convolution_6);  convolution_6 = None
        convolution_7: "f32[8, 1024, 1493]" = torch.ops.aten.convolution.default(relu_3, arg15_1, arg16_1, [1], [0], [1], False, [0], 1);  relu_3 = arg15_1 = arg16_1 = None
        glu_3: "f32[8, 512, 1493]" = torch.ops.aten.glu.default(convolution_7, 1);  convolution_7 = None
        convolution_8: "f32[8, 1024, 372]" = torch.ops.aten.convolution.default(glu_3, arg17_1, arg18_1, [4], [0], [1], False, [0], 1);  arg17_1 = arg18_1 = None
        relu_4: "f32[8, 1024, 372]" = torch.ops.aten.relu.default(convolution_8);  convolution_8 = None
        convolution_9: "f32[8, 2048, 372]" = torch.ops.aten.convolution.default(relu_4, arg19_1, arg20_1, [1], [0], [1], False, [0], 1);  relu_4 = arg19_1 = arg20_1 = None
        glu_4: "f32[8, 1024, 372]" = torch.ops.aten.glu.default(convolution_9, 1);  convolution_9 = None
        convolution_10: "f32[8, 2048, 92]" = torch.ops.aten.convolution.default(glu_4, arg21_1, arg22_1, [4], [0], [1], False, [0], 1);  arg21_1 = arg22_1 = None
        relu_5: "f32[8, 2048, 92]" = torch.ops.aten.relu.default(convolution_10);  convolution_10 = None
        convolution_11: "f32[8, 4096, 92]" = torch.ops.aten.convolution.default(relu_5, arg23_1, arg24_1, [1], [0], [1], False, [0], 1);  relu_5 = arg23_1 = arg24_1 = None
        glu_5: "f32[8, 2048, 92]" = torch.ops.aten.glu.default(convolution_11, 1);  convolution_11 = None
        return (glu_5, glu, glu_1, glu_2, glu_3, glu_4)
