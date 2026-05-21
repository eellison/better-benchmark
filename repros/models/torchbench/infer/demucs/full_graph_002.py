class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "f32[8, 2048, 92]", arg1_1: "f32[8, 2048, 92]", arg2_1: "f32[4096, 2048, 3]", arg3_1: "f32[4096]", arg4_1: "f32[2048, 1024, 8]", arg5_1: "f32[1024]", arg6_1: "f32[8, 1024, 372]", arg7_1: "f32[2048, 1024, 3]", arg8_1: "f32[2048]", arg9_1: "f32[1024, 512, 8]", arg10_1: "f32[512]", arg11_1: "f32[8, 512, 1493]", arg12_1: "f32[1024, 512, 3]", arg13_1: "f32[1024]", arg14_1: "f32[512, 256, 8]", arg15_1: "f32[256]", arg16_1: "f32[8, 256, 5979]", arg17_1: "f32[512, 256, 3]", arg18_1: "f32[512]", arg19_1: "f32[256, 128, 8]", arg20_1: "f32[128]", arg21_1: "f32[8, 128, 23923]", arg22_1: "f32[256, 128, 3]", arg23_1: "f32[256]", arg24_1: "f32[128, 64, 8]", arg25_1: "f32[64]", arg26_1: "f32[8, 64, 95696]", arg27_1: "f32[128, 64, 3]", arg28_1: "f32[128]", arg29_1: "f32[64, 8, 8]", arg30_1: "f32[8]"):
        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/demucs/demucs/model.py:225 in torch_dynamo_resume_in_forward_at_220, code: x = x + skip
        add: "f32[8, 2048, 92]" = torch.ops.aten.add.Tensor(arg0_1, arg1_1);  arg0_1 = arg1_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/demucs/demucs/model.py:226 in torch_dynamo_resume_in_forward_at_220, code: x = decode(x)
        convolution: "f32[8, 4096, 90]" = torch.ops.aten.convolution.default(add, arg2_1, arg3_1, [1], [0], [1], False, [0], 1);  add = arg2_1 = arg3_1 = None
        glu: "f32[8, 2048, 90]" = torch.ops.aten.glu.default(convolution, 1);  convolution = None
        convolution_1: "f32[8, 1024, 364]" = torch.ops.aten.convolution.default(glu, arg4_1, arg5_1, [4], [0], [1], True, [0], 1);  glu = arg4_1 = arg5_1 = None
        relu: "f32[8, 1024, 364]" = torch.ops.aten.relu.default(convolution_1);  convolution_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/demucs/demucs/utils.py:35 in center_trim, code: tensor = tensor[..., delta // 2 : -(delta - delta // 2)]
        slice_1: "f32[8, 1024, 364]" = torch.ops.aten.slice.Tensor(arg6_1, 2, 4, -4);  arg6_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/demucs/demucs/model.py:225 in torch_dynamo_resume_in_forward_at_220, code: x = x + skip
        add_1: "f32[8, 1024, 364]" = torch.ops.aten.add.Tensor(relu, slice_1);  relu = slice_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/demucs/demucs/model.py:226 in torch_dynamo_resume_in_forward_at_220, code: x = decode(x)
        convolution_2: "f32[8, 2048, 362]" = torch.ops.aten.convolution.default(add_1, arg7_1, arg8_1, [1], [0], [1], False, [0], 1);  add_1 = arg7_1 = arg8_1 = None
        glu_1: "f32[8, 1024, 362]" = torch.ops.aten.glu.default(convolution_2, 1);  convolution_2 = None
        convolution_3: "f32[8, 512, 1452]" = torch.ops.aten.convolution.default(glu_1, arg9_1, arg10_1, [4], [0], [1], True, [0], 1);  glu_1 = arg9_1 = arg10_1 = None
        relu_1: "f32[8, 512, 1452]" = torch.ops.aten.relu.default(convolution_3);  convolution_3 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/demucs/demucs/utils.py:35 in center_trim, code: tensor = tensor[..., delta // 2 : -(delta - delta // 2)]
        slice_2: "f32[8, 512, 1452]" = torch.ops.aten.slice.Tensor(arg11_1, 2, 20, -21);  arg11_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/demucs/demucs/model.py:225 in torch_dynamo_resume_in_forward_at_220, code: x = x + skip
        add_2: "f32[8, 512, 1452]" = torch.ops.aten.add.Tensor(relu_1, slice_2);  relu_1 = slice_2 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/demucs/demucs/model.py:226 in torch_dynamo_resume_in_forward_at_220, code: x = decode(x)
        convolution_4: "f32[8, 1024, 1450]" = torch.ops.aten.convolution.default(add_2, arg12_1, arg13_1, [1], [0], [1], False, [0], 1);  add_2 = arg12_1 = arg13_1 = None
        glu_2: "f32[8, 512, 1450]" = torch.ops.aten.glu.default(convolution_4, 1);  convolution_4 = None
        convolution_5: "f32[8, 256, 5804]" = torch.ops.aten.convolution.default(glu_2, arg14_1, arg15_1, [4], [0], [1], True, [0], 1);  glu_2 = arg14_1 = arg15_1 = None
        relu_2: "f32[8, 256, 5804]" = torch.ops.aten.relu.default(convolution_5);  convolution_5 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/demucs/demucs/utils.py:35 in center_trim, code: tensor = tensor[..., delta // 2 : -(delta - delta // 2)]
        slice_3: "f32[8, 256, 5804]" = torch.ops.aten.slice.Tensor(arg16_1, 2, 87, -88);  arg16_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/demucs/demucs/model.py:225 in torch_dynamo_resume_in_forward_at_220, code: x = x + skip
        add_3: "f32[8, 256, 5804]" = torch.ops.aten.add.Tensor(relu_2, slice_3);  relu_2 = slice_3 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/demucs/demucs/model.py:226 in torch_dynamo_resume_in_forward_at_220, code: x = decode(x)
        convolution_6: "f32[8, 512, 5802]" = torch.ops.aten.convolution.default(add_3, arg17_1, arg18_1, [1], [0], [1], False, [0], 1);  add_3 = arg17_1 = arg18_1 = None
        glu_3: "f32[8, 256, 5802]" = torch.ops.aten.glu.default(convolution_6, 1);  convolution_6 = None
        convolution_7: "f32[8, 128, 23212]" = torch.ops.aten.convolution.default(glu_3, arg19_1, arg20_1, [4], [0], [1], True, [0], 1);  glu_3 = arg19_1 = arg20_1 = None
        relu_3: "f32[8, 128, 23212]" = torch.ops.aten.relu.default(convolution_7);  convolution_7 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/demucs/demucs/utils.py:35 in center_trim, code: tensor = tensor[..., delta // 2 : -(delta - delta // 2)]
        slice_4: "f32[8, 128, 23212]" = torch.ops.aten.slice.Tensor(arg21_1, 2, 355, -356);  arg21_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/demucs/demucs/model.py:225 in torch_dynamo_resume_in_forward_at_220, code: x = x + skip
        add_4: "f32[8, 128, 23212]" = torch.ops.aten.add.Tensor(relu_3, slice_4);  relu_3 = slice_4 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/demucs/demucs/model.py:226 in torch_dynamo_resume_in_forward_at_220, code: x = decode(x)
        convolution_8: "f32[8, 256, 23210]" = torch.ops.aten.convolution.default(add_4, arg22_1, arg23_1, [1], [0], [1], False, [0], 1);  add_4 = arg22_1 = arg23_1 = None
        glu_4: "f32[8, 128, 23210]" = torch.ops.aten.glu.default(convolution_8, 1);  convolution_8 = None
        convolution_9: "f32[8, 64, 92844]" = torch.ops.aten.convolution.default(glu_4, arg24_1, arg25_1, [4], [0], [1], True, [0], 1);  glu_4 = arg24_1 = arg25_1 = None
        relu_4: "f32[8, 64, 92844]" = torch.ops.aten.relu.default(convolution_9);  convolution_9 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/demucs/demucs/utils.py:35 in center_trim, code: tensor = tensor[..., delta // 2 : -(delta - delta // 2)]
        slice_5: "f32[8, 64, 92844]" = torch.ops.aten.slice.Tensor(arg26_1, 2, 1426, -1426);  arg26_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/demucs/demucs/model.py:225 in torch_dynamo_resume_in_forward_at_220, code: x = x + skip
        add_5: "f32[8, 64, 92844]" = torch.ops.aten.add.Tensor(relu_4, slice_5);  relu_4 = slice_5 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/demucs/demucs/model.py:226 in torch_dynamo_resume_in_forward_at_220, code: x = decode(x)
        convolution_10: "f32[8, 128, 92842]" = torch.ops.aten.convolution.default(add_5, arg27_1, arg28_1, [1], [0], [1], False, [0], 1);  add_5 = arg27_1 = arg28_1 = None
        glu_5: "f32[8, 64, 92842]" = torch.ops.aten.glu.default(convolution_10, 1);  convolution_10 = None
        convolution_11: "f32[8, 8, 371372]" = torch.ops.aten.convolution.default(glu_5, arg29_1, arg30_1, [4], [0], [1], True, [0], 1);  glu_5 = arg29_1 = arg30_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/demucs/demucs/model.py:232 in torch_dynamo_resume_in_forward_at_220, code: x = x.view(x.size(0), self.sources, self.audio_channels, x.size(-1))
        view: "f32[8, 4, 2, 371372]" = torch.ops.aten.reshape.default(convolution_11, [8, 4, 2, 371372]);  convolution_11 = None
        return (view,)
