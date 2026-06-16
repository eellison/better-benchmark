class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "bf16[4, 2048, 92][2048, 1, 8192]cuda:0", primals_2: "bf16[4, 2048, 92][188416, 92, 1]cuda:0", primals_3: "f32[4096, 2048, 3][6144, 3, 1]cuda:0", primals_4: "f32[4096][1]cuda:0", primals_5: "f32[2048, 1024, 8][8192, 8, 1]cuda:0", primals_6: "f32[1024][1]cuda:0", primals_7: "bf16[4, 1024, 372][380928, 372, 1]cuda:0", primals_8: "f32[2048, 1024, 3][3072, 3, 1]cuda:0", primals_9: "f32[2048][1]cuda:0", primals_10: "f32[1024, 512, 8][4096, 8, 1]cuda:0", primals_11: "f32[512][1]cuda:0", primals_12: "bf16[4, 512, 1493][764416, 1493, 1]cuda:0", primals_13: "f32[1024, 512, 3][1536, 3, 1]cuda:0", primals_14: "f32[1024][1]cuda:0", primals_15: "f32[512, 256, 8][2048, 8, 1]cuda:0", primals_16: "f32[256][1]cuda:0", primals_17: "bf16[4, 256, 5979][1530624, 5979, 1]cuda:0", primals_18: "f32[512, 256, 3][768, 3, 1]cuda:0", primals_19: "f32[512][1]cuda:0", primals_20: "f32[256, 128, 8][1024, 8, 1]cuda:0", primals_21: "f32[128][1]cuda:0", primals_22: "bf16[4, 128, 23923][3062144, 23923, 1]cuda:0", primals_23: "f32[256, 128, 3][384, 3, 1]cuda:0", primals_24: "f32[256][1]cuda:0", primals_25: "f32[128, 64, 8][512, 8, 1]cuda:0", primals_26: "f32[64][1]cuda:0", primals_27: "bf16[4, 64, 95696][6124544, 95696, 1]cuda:0", primals_28: "f32[128, 64, 3][192, 3, 1]cuda:0", primals_29: "f32[128][1]cuda:0", primals_30: "f32[64, 8, 8][64, 8, 1]cuda:0", primals_31: "f32[8][1]cuda:0"):
        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/demucs/demucs/model.py:225 in torch_dynamo_resume_in_forward_at_220, code: x = x + skip
        add: "bf16[4, 2048, 92][2048, 1, 8192]cuda:0" = torch.ops.aten.add.Tensor(primals_1, primals_2);  primals_1 = primals_2 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/demucs/demucs/model.py:226 in torch_dynamo_resume_in_forward_at_220, code: x = decode(x)
        convert_element_type: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_4, torch.bfloat16);  primals_4 = None
        convert_element_type_1: "bf16[4096, 2048, 3][6144, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_3, torch.bfloat16);  primals_3 = None
        convolution: "bf16[4, 4096, 90][368640, 90, 1]cuda:0" = torch.ops.aten.convolution.default(add, convert_element_type_1, convert_element_type, [1], [0], [1], False, [0], 1);  convert_element_type = None
        glu: "bf16[4, 2048, 90][184320, 90, 1]cuda:0" = torch.ops.aten.glu.default(convolution, 1)
        convert_element_type_2: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_6, torch.bfloat16);  primals_6 = None
        convert_element_type_3: "bf16[2048, 1024, 8][8192, 8, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_5, torch.bfloat16);  primals_5 = None
        convolution_1: "bf16[4, 1024, 364][372736, 364, 1]cuda:0" = torch.ops.aten.convolution.default(glu, convert_element_type_3, convert_element_type_2, [4], [0], [1], True, [0], 1);  convert_element_type_2 = None
        relu: "bf16[4, 1024, 364][372736, 364, 1]cuda:0" = torch.ops.aten.relu.default(convolution_1);  convolution_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/demucs/demucs/utils.py:35 in center_trim, code: tensor = tensor[..., delta // 2 : -(delta - delta // 2)]
        slice_1: "bf16[4, 1024, 364][380928, 372, 1]cuda:0" = torch.ops.aten.slice.Tensor(primals_7, 2, 4, -4);  primals_7 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/demucs/demucs/model.py:225 in torch_dynamo_resume_in_forward_at_220, code: x = x + skip
        add_1: "bf16[4, 1024, 364][372736, 364, 1]cuda:0" = torch.ops.aten.add.Tensor(relu, slice_1);  slice_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/demucs/demucs/model.py:226 in torch_dynamo_resume_in_forward_at_220, code: x = decode(x)
        convert_element_type_4: "bf16[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_9, torch.bfloat16);  primals_9 = None
        convert_element_type_5: "bf16[2048, 1024, 3][3072, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_8, torch.bfloat16);  primals_8 = None
        convolution_2: "bf16[4, 2048, 362][741376, 362, 1]cuda:0" = torch.ops.aten.convolution.default(add_1, convert_element_type_5, convert_element_type_4, [1], [0], [1], False, [0], 1);  convert_element_type_4 = None
        glu_1: "bf16[4, 1024, 362][370688, 362, 1]cuda:0" = torch.ops.aten.glu.default(convolution_2, 1)
        convert_element_type_6: "bf16[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_11, torch.bfloat16);  primals_11 = None
        convert_element_type_7: "bf16[1024, 512, 8][4096, 8, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_10, torch.bfloat16);  primals_10 = None
        convolution_3: "bf16[4, 512, 1452][743424, 1452, 1]cuda:0" = torch.ops.aten.convolution.default(glu_1, convert_element_type_7, convert_element_type_6, [4], [0], [1], True, [0], 1);  convert_element_type_6 = None
        relu_1: "bf16[4, 512, 1452][743424, 1452, 1]cuda:0" = torch.ops.aten.relu.default(convolution_3);  convolution_3 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/demucs/demucs/utils.py:35 in center_trim, code: tensor = tensor[..., delta // 2 : -(delta - delta // 2)]
        slice_2: "bf16[4, 512, 1452][764416, 1493, 1]cuda:0" = torch.ops.aten.slice.Tensor(primals_12, 2, 20, -21);  primals_12 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/demucs/demucs/model.py:225 in torch_dynamo_resume_in_forward_at_220, code: x = x + skip
        add_2: "bf16[4, 512, 1452][743424, 1452, 1]cuda:0" = torch.ops.aten.add.Tensor(relu_1, slice_2);  slice_2 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/demucs/demucs/model.py:226 in torch_dynamo_resume_in_forward_at_220, code: x = decode(x)
        convert_element_type_8: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_14, torch.bfloat16);  primals_14 = None
        convert_element_type_9: "bf16[1024, 512, 3][1536, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_13, torch.bfloat16);  primals_13 = None
        convolution_4: "bf16[4, 1024, 1450][1484800, 1450, 1]cuda:0" = torch.ops.aten.convolution.default(add_2, convert_element_type_9, convert_element_type_8, [1], [0], [1], False, [0], 1);  convert_element_type_8 = None
        glu_2: "bf16[4, 512, 1450][742400, 1450, 1]cuda:0" = torch.ops.aten.glu.default(convolution_4, 1)
        convert_element_type_10: "bf16[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_16, torch.bfloat16);  primals_16 = None
        convert_element_type_11: "bf16[512, 256, 8][2048, 8, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_15, torch.bfloat16);  primals_15 = None
        convolution_5: "bf16[4, 256, 5804][1485824, 5804, 1]cuda:0" = torch.ops.aten.convolution.default(glu_2, convert_element_type_11, convert_element_type_10, [4], [0], [1], True, [0], 1);  convert_element_type_10 = None
        relu_2: "bf16[4, 256, 5804][1485824, 5804, 1]cuda:0" = torch.ops.aten.relu.default(convolution_5);  convolution_5 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/demucs/demucs/utils.py:35 in center_trim, code: tensor = tensor[..., delta // 2 : -(delta - delta // 2)]
        slice_3: "bf16[4, 256, 5804][1530624, 5979, 1]cuda:0" = torch.ops.aten.slice.Tensor(primals_17, 2, 87, -88);  primals_17 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/demucs/demucs/model.py:225 in torch_dynamo_resume_in_forward_at_220, code: x = x + skip
        add_3: "bf16[4, 256, 5804][1485824, 5804, 1]cuda:0" = torch.ops.aten.add.Tensor(relu_2, slice_3);  slice_3 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/demucs/demucs/model.py:226 in torch_dynamo_resume_in_forward_at_220, code: x = decode(x)
        convert_element_type_12: "bf16[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_19, torch.bfloat16);  primals_19 = None
        convert_element_type_13: "bf16[512, 256, 3][768, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_18, torch.bfloat16);  primals_18 = None
        convolution_6: "bf16[4, 512, 5802][2970624, 5802, 1]cuda:0" = torch.ops.aten.convolution.default(add_3, convert_element_type_13, convert_element_type_12, [1], [0], [1], False, [0], 1);  convert_element_type_12 = None
        glu_3: "bf16[4, 256, 5802][1485312, 5802, 1]cuda:0" = torch.ops.aten.glu.default(convolution_6, 1)
        convert_element_type_14: "bf16[128][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_21, torch.bfloat16);  primals_21 = None
        convert_element_type_15: "bf16[256, 128, 8][1024, 8, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_20, torch.bfloat16);  primals_20 = None
        convolution_7: "bf16[4, 128, 23212][2971136, 23212, 1]cuda:0" = torch.ops.aten.convolution.default(glu_3, convert_element_type_15, convert_element_type_14, [4], [0], [1], True, [0], 1);  convert_element_type_14 = None
        relu_3: "bf16[4, 128, 23212][2971136, 23212, 1]cuda:0" = torch.ops.aten.relu.default(convolution_7);  convolution_7 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/demucs/demucs/utils.py:35 in center_trim, code: tensor = tensor[..., delta // 2 : -(delta - delta // 2)]
        slice_4: "bf16[4, 128, 23212][3062144, 23923, 1]cuda:0" = torch.ops.aten.slice.Tensor(primals_22, 2, 355, -356);  primals_22 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/demucs/demucs/model.py:225 in torch_dynamo_resume_in_forward_at_220, code: x = x + skip
        add_4: "bf16[4, 128, 23212][2971136, 23212, 1]cuda:0" = torch.ops.aten.add.Tensor(relu_3, slice_4);  slice_4 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/demucs/demucs/model.py:226 in torch_dynamo_resume_in_forward_at_220, code: x = decode(x)
        convert_element_type_16: "bf16[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_24, torch.bfloat16);  primals_24 = None
        convert_element_type_17: "bf16[256, 128, 3][384, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_23, torch.bfloat16);  primals_23 = None
        convolution_8: "bf16[4, 256, 23210][5941760, 23210, 1]cuda:0" = torch.ops.aten.convolution.default(add_4, convert_element_type_17, convert_element_type_16, [1], [0], [1], False, [0], 1);  convert_element_type_16 = None
        glu_4: "bf16[4, 128, 23210][2970880, 23210, 1]cuda:0" = torch.ops.aten.glu.default(convolution_8, 1)
        convert_element_type_18: "bf16[64][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_26, torch.bfloat16);  primals_26 = None
        convert_element_type_19: "bf16[128, 64, 8][512, 8, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_25, torch.bfloat16);  primals_25 = None
        convolution_9: "bf16[4, 64, 92844][5942016, 92844, 1]cuda:0" = torch.ops.aten.convolution.default(glu_4, convert_element_type_19, convert_element_type_18, [4], [0], [1], True, [0], 1);  convert_element_type_18 = None
        relu_4: "bf16[4, 64, 92844][5942016, 92844, 1]cuda:0" = torch.ops.aten.relu.default(convolution_9);  convolution_9 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/demucs/demucs/utils.py:35 in center_trim, code: tensor = tensor[..., delta // 2 : -(delta - delta // 2)]
        slice_5: "bf16[4, 64, 92844][6124544, 95696, 1]cuda:0" = torch.ops.aten.slice.Tensor(primals_27, 2, 1426, -1426);  primals_27 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/demucs/demucs/model.py:225 in torch_dynamo_resume_in_forward_at_220, code: x = x + skip
        add_5: "bf16[4, 64, 92844][5942016, 92844, 1]cuda:0" = torch.ops.aten.add.Tensor(relu_4, slice_5);  slice_5 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/demucs/demucs/model.py:226 in torch_dynamo_resume_in_forward_at_220, code: x = decode(x)
        convert_element_type_20: "bf16[128][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_29, torch.bfloat16);  primals_29 = None
        convert_element_type_21: "bf16[128, 64, 3][192, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_28, torch.bfloat16);  primals_28 = None
        convolution_10: "bf16[4, 128, 92842][11883776, 92842, 1]cuda:0" = torch.ops.aten.convolution.default(add_5, convert_element_type_21, convert_element_type_20, [1], [0], [1], False, [0], 1);  convert_element_type_20 = None
        glu_5: "bf16[4, 64, 92842][5941888, 92842, 1]cuda:0" = torch.ops.aten.glu.default(convolution_10, 1)
        convert_element_type_22: "bf16[8][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_31, torch.bfloat16);  primals_31 = None
        convert_element_type_23: "bf16[64, 8, 8][64, 8, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_30, torch.bfloat16);  primals_30 = None
        convolution_11: "bf16[4, 8, 371372][2970976, 371372, 1]cuda:0" = torch.ops.aten.convolution.default(glu_5, convert_element_type_23, convert_element_type_22, [4], [0], [1], True, [0], 1);  convert_element_type_22 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/demucs/demucs/model.py:232 in torch_dynamo_resume_in_forward_at_220, code: x = x.view(x.size(0), self.sources, self.audio_channels, x.size(-1))
        view: "bf16[4, 4, 2, 371372][2970976, 742744, 371372, 1]cuda:0" = torch.ops.aten.reshape.default(convolution_11, [4, 4, 2, 371372]);  convolution_11 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/demucs/demucs/model.py:226 in torch_dynamo_resume_in_forward_at_220, code: x = decode(x)
        le: "b8[4, 64, 92844][5942016, 92844, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_4, 0);  relu_4 = None
        le_1: "b8[4, 128, 23212][2971136, 23212, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_3, 0);  relu_3 = None
        le_2: "b8[4, 256, 5804][1485824, 5804, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_2, 0);  relu_2 = None
        le_3: "b8[4, 512, 1452][743424, 1452, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_1, 0);  relu_1 = None
        le_4: "b8[4, 1024, 364][372736, 364, 1]cuda:0" = torch.ops.aten.le.Scalar(relu, 0);  relu = None
        return (view, add, convert_element_type_1, convolution, glu, convert_element_type_3, add_1, convert_element_type_5, convolution_2, glu_1, convert_element_type_7, add_2, convert_element_type_9, convolution_4, glu_2, convert_element_type_11, add_3, convert_element_type_13, convolution_6, glu_3, convert_element_type_15, add_4, convert_element_type_17, convolution_8, glu_4, convert_element_type_19, add_5, convert_element_type_21, convolution_10, glu_5, convert_element_type_23, le, le_1, le_2, le_3, le_4)
