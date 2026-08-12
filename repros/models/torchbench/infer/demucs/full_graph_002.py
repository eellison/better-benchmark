class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "bf16[8, 2048, 92][2048, 1, 16384]cuda:0", arg1_1: "bf16[8, 2048, 92][188416, 92, 1]cuda:0", arg2_1: "bf16[4096, 2048, 3][6144, 3, 1]cuda:0", arg3_1: "bf16[4096][1]cuda:0", arg4_1: "bf16[2048, 1024, 8][8192, 8, 1]cuda:0", arg5_1: "bf16[1024][1]cuda:0", arg6_1: "bf16[8, 1024, 372][380928, 372, 1]cuda:0", arg7_1: "bf16[2048, 1024, 3][3072, 3, 1]cuda:0", arg8_1: "bf16[2048][1]cuda:0", arg9_1: "bf16[1024, 512, 8][4096, 8, 1]cuda:0", arg10_1: "bf16[512][1]cuda:0", arg11_1: "bf16[8, 512, 1493][764416, 1493, 1]cuda:0", arg12_1: "bf16[1024, 512, 3][1536, 3, 1]cuda:0", arg13_1: "bf16[1024][1]cuda:0", arg14_1: "bf16[512, 256, 8][2048, 8, 1]cuda:0", arg15_1: "bf16[256][1]cuda:0", arg16_1: "bf16[8, 256, 5979][1530624, 5979, 1]cuda:0", arg17_1: "bf16[512, 256, 3][768, 3, 1]cuda:0", arg18_1: "bf16[512][1]cuda:0", arg19_1: "bf16[256, 128, 8][1024, 8, 1]cuda:0", arg20_1: "bf16[128][1]cuda:0", arg21_1: "bf16[8, 128, 23923][3062144, 23923, 1]cuda:0", arg22_1: "bf16[256, 128, 3][384, 3, 1]cuda:0", arg23_1: "bf16[256][1]cuda:0", arg24_1: "bf16[128, 64, 8][512, 8, 1]cuda:0", arg25_1: "bf16[64][1]cuda:0", arg26_1: "bf16[8, 64, 95696][6124544, 95696, 1]cuda:0", arg27_1: "bf16[128, 64, 3][192, 3, 1]cuda:0", arg28_1: "bf16[128][1]cuda:0", arg29_1: "bf16[64, 8, 8][64, 8, 1]cuda:0", arg30_1: "bf16[8][1]cuda:0"):
        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/demucs/demucs/model.py:225 in torch_dynamo_resume_in_forward_at_220, code: x = x + skip
        add: "bf16[8, 2048, 92][2048, 1, 16384]cuda:0" = torch.ops.aten.add.Tensor(arg0_1, arg1_1);  arg0_1 = arg1_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/demucs/demucs/model.py:226 in torch_dynamo_resume_in_forward_at_220, code: x = decode(x)
        convolution: "bf16[8, 4096, 90][368640, 90, 1]cuda:0" = torch.ops.aten.convolution.default(add, arg2_1, arg3_1, [1], [0], [1], False, [0], 1);  add = arg2_1 = arg3_1 = None
        glu: "bf16[8, 2048, 90][184320, 90, 1]cuda:0" = torch.ops.aten.glu.default(convolution, 1);  convolution = None
        convolution_1: "bf16[8, 1024, 364][372736, 364, 1]cuda:0" = torch.ops.aten.convolution.default(glu, arg4_1, arg5_1, [4], [0], [1], True, [0], 1);  glu = arg4_1 = arg5_1 = None
        relu: "bf16[8, 1024, 364][372736, 364, 1]cuda:0" = torch.ops.aten.relu.default(convolution_1);  convolution_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/demucs/demucs/utils.py:35 in center_trim, code: tensor = tensor[..., delta // 2 : -(delta - delta // 2)]
        slice_1: "bf16[8, 1024, 364][380928, 372, 1]cuda:0" = torch.ops.aten.slice.Tensor(arg6_1, 2, 4, -4);  arg6_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/demucs/demucs/model.py:225 in torch_dynamo_resume_in_forward_at_220, code: x = x + skip
        add_1: "bf16[8, 1024, 364][372736, 364, 1]cuda:0" = torch.ops.aten.add.Tensor(relu, slice_1);  relu = slice_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/demucs/demucs/model.py:226 in torch_dynamo_resume_in_forward_at_220, code: x = decode(x)
        convolution_2: "bf16[8, 2048, 362][741376, 362, 1]cuda:0" = torch.ops.aten.convolution.default(add_1, arg7_1, arg8_1, [1], [0], [1], False, [0], 1);  add_1 = arg7_1 = arg8_1 = None
        glu_1: "bf16[8, 1024, 362][370688, 362, 1]cuda:0" = torch.ops.aten.glu.default(convolution_2, 1);  convolution_2 = None
        convolution_3: "bf16[8, 512, 1452][743424, 1452, 1]cuda:0" = torch.ops.aten.convolution.default(glu_1, arg9_1, arg10_1, [4], [0], [1], True, [0], 1);  glu_1 = arg9_1 = arg10_1 = None
        relu_1: "bf16[8, 512, 1452][743424, 1452, 1]cuda:0" = torch.ops.aten.relu.default(convolution_3);  convolution_3 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/demucs/demucs/utils.py:35 in center_trim, code: tensor = tensor[..., delta // 2 : -(delta - delta // 2)]
        slice_2: "bf16[8, 512, 1452][764416, 1493, 1]cuda:0" = torch.ops.aten.slice.Tensor(arg11_1, 2, 20, -21);  arg11_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/demucs/demucs/model.py:225 in torch_dynamo_resume_in_forward_at_220, code: x = x + skip
        add_2: "bf16[8, 512, 1452][743424, 1452, 1]cuda:0" = torch.ops.aten.add.Tensor(relu_1, slice_2);  relu_1 = slice_2 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/demucs/demucs/model.py:226 in torch_dynamo_resume_in_forward_at_220, code: x = decode(x)
        convolution_4: "bf16[8, 1024, 1450][1484800, 1450, 1]cuda:0" = torch.ops.aten.convolution.default(add_2, arg12_1, arg13_1, [1], [0], [1], False, [0], 1);  add_2 = arg12_1 = arg13_1 = None
        glu_2: "bf16[8, 512, 1450][742400, 1450, 1]cuda:0" = torch.ops.aten.glu.default(convolution_4, 1);  convolution_4 = None
        convolution_5: "bf16[8, 256, 5804][1485824, 5804, 1]cuda:0" = torch.ops.aten.convolution.default(glu_2, arg14_1, arg15_1, [4], [0], [1], True, [0], 1);  glu_2 = arg14_1 = arg15_1 = None
        relu_2: "bf16[8, 256, 5804][1485824, 5804, 1]cuda:0" = torch.ops.aten.relu.default(convolution_5);  convolution_5 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/demucs/demucs/utils.py:35 in center_trim, code: tensor = tensor[..., delta // 2 : -(delta - delta // 2)]
        slice_3: "bf16[8, 256, 5804][1530624, 5979, 1]cuda:0" = torch.ops.aten.slice.Tensor(arg16_1, 2, 87, -88);  arg16_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/demucs/demucs/model.py:225 in torch_dynamo_resume_in_forward_at_220, code: x = x + skip
        add_3: "bf16[8, 256, 5804][1485824, 5804, 1]cuda:0" = torch.ops.aten.add.Tensor(relu_2, slice_3);  relu_2 = slice_3 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/demucs/demucs/model.py:226 in torch_dynamo_resume_in_forward_at_220, code: x = decode(x)
        convolution_6: "bf16[8, 512, 5802][2970624, 5802, 1]cuda:0" = torch.ops.aten.convolution.default(add_3, arg17_1, arg18_1, [1], [0], [1], False, [0], 1);  add_3 = arg17_1 = arg18_1 = None
        glu_3: "bf16[8, 256, 5802][1485312, 5802, 1]cuda:0" = torch.ops.aten.glu.default(convolution_6, 1);  convolution_6 = None
        convolution_7: "bf16[8, 128, 23212][2971136, 23212, 1]cuda:0" = torch.ops.aten.convolution.default(glu_3, arg19_1, arg20_1, [4], [0], [1], True, [0], 1);  glu_3 = arg19_1 = arg20_1 = None
        relu_3: "bf16[8, 128, 23212][2971136, 23212, 1]cuda:0" = torch.ops.aten.relu.default(convolution_7);  convolution_7 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/demucs/demucs/utils.py:35 in center_trim, code: tensor = tensor[..., delta // 2 : -(delta - delta // 2)]
        slice_4: "bf16[8, 128, 23212][3062144, 23923, 1]cuda:0" = torch.ops.aten.slice.Tensor(arg21_1, 2, 355, -356);  arg21_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/demucs/demucs/model.py:225 in torch_dynamo_resume_in_forward_at_220, code: x = x + skip
        add_4: "bf16[8, 128, 23212][2971136, 23212, 1]cuda:0" = torch.ops.aten.add.Tensor(relu_3, slice_4);  relu_3 = slice_4 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/demucs/demucs/model.py:226 in torch_dynamo_resume_in_forward_at_220, code: x = decode(x)
        convolution_8: "bf16[8, 256, 23210][5941760, 23210, 1]cuda:0" = torch.ops.aten.convolution.default(add_4, arg22_1, arg23_1, [1], [0], [1], False, [0], 1);  add_4 = arg22_1 = arg23_1 = None
        glu_4: "bf16[8, 128, 23210][2970880, 23210, 1]cuda:0" = torch.ops.aten.glu.default(convolution_8, 1);  convolution_8 = None
        convolution_9: "bf16[8, 64, 92844][5942016, 92844, 1]cuda:0" = torch.ops.aten.convolution.default(glu_4, arg24_1, arg25_1, [4], [0], [1], True, [0], 1);  glu_4 = arg24_1 = arg25_1 = None
        relu_4: "bf16[8, 64, 92844][5942016, 92844, 1]cuda:0" = torch.ops.aten.relu.default(convolution_9);  convolution_9 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/demucs/demucs/utils.py:35 in center_trim, code: tensor = tensor[..., delta // 2 : -(delta - delta // 2)]
        slice_5: "bf16[8, 64, 92844][6124544, 95696, 1]cuda:0" = torch.ops.aten.slice.Tensor(arg26_1, 2, 1426, -1426);  arg26_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/demucs/demucs/model.py:225 in torch_dynamo_resume_in_forward_at_220, code: x = x + skip
        add_5: "bf16[8, 64, 92844][5942016, 92844, 1]cuda:0" = torch.ops.aten.add.Tensor(relu_4, slice_5);  relu_4 = slice_5 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/demucs/demucs/model.py:226 in torch_dynamo_resume_in_forward_at_220, code: x = decode(x)
        convolution_10: "bf16[8, 128, 92842][11883776, 92842, 1]cuda:0" = torch.ops.aten.convolution.default(add_5, arg27_1, arg28_1, [1], [0], [1], False, [0], 1);  add_5 = arg27_1 = arg28_1 = None
        glu_5: "bf16[8, 64, 92842][5941888, 92842, 1]cuda:0" = torch.ops.aten.glu.default(convolution_10, 1);  convolution_10 = None
        convolution_11: "bf16[8, 8, 371372][2970976, 371372, 1]cuda:0" = torch.ops.aten.convolution.default(glu_5, arg29_1, arg30_1, [4], [0], [1], True, [0], 1);  glu_5 = arg29_1 = arg30_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/demucs/demucs/model.py:232 in torch_dynamo_resume_in_forward_at_220, code: x = x.view(x.size(0), self.sources, self.audio_channels, x.size(-1))
        view: "bf16[8, 4, 2, 371372][2970976, 742744, 371372, 1]cuda:0" = torch.ops.aten.reshape.default(convolution_11, [8, 4, 2, 371372]);  convolution_11 = None
        return (view,)
