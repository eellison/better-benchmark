class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[64, 2, 8][16, 8, 1]cuda:0", primals_2: "f32[64][1]cuda:0", primals_3: "f32[4, 2, 382788][765576, 382788, 1]cuda:0", primals_4: "f32[128, 64, 1][64, 1, 1]cuda:0", primals_5: "f32[128][1]cuda:0", primals_6: "f32[128, 64, 8][512, 8, 1]cuda:0", primals_7: "f32[128][1]cuda:0", primals_8: "f32[256, 128, 1][128, 1, 1]cuda:0", primals_9: "f32[256][1]cuda:0", primals_10: "f32[256, 128, 8][1024, 8, 1]cuda:0", primals_11: "f32[256][1]cuda:0", primals_12: "f32[512, 256, 1][256, 1, 1]cuda:0", primals_13: "f32[512][1]cuda:0", primals_14: "f32[512, 256, 8][2048, 8, 1]cuda:0", primals_15: "f32[512][1]cuda:0", primals_16: "f32[1024, 512, 1][512, 1, 1]cuda:0", primals_17: "f32[1024][1]cuda:0", primals_18: "f32[1024, 512, 8][4096, 8, 1]cuda:0", primals_19: "f32[1024][1]cuda:0", primals_20: "f32[2048, 1024, 1][1024, 1, 1]cuda:0", primals_21: "f32[2048][1]cuda:0", primals_22: "f32[2048, 1024, 8][8192, 8, 1]cuda:0", primals_23: "f32[2048][1]cuda:0", primals_24: "f32[4096, 2048, 1][2048, 1, 1]cuda:0", primals_25: "f32[4096][1]cuda:0"):
        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/demucs/demucs/model.py:215 in forward, code: x = encode(x)
        convert_element_type: "bf16[64][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_2, torch.bfloat16);  primals_2 = None
        convert_element_type_1: "bf16[64, 2, 8][16, 8, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_1, torch.bfloat16);  primals_1 = None
        convert_element_type_2: "bf16[4, 2, 382788][765576, 382788, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_3, torch.bfloat16);  primals_3 = None
        convolution: "bf16[4, 64, 95696][6124544, 95696, 1]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_2, convert_element_type_1, convert_element_type, [4], [0], [1], False, [0], 1);  convert_element_type = None
        relu: "bf16[4, 64, 95696][6124544, 95696, 1]cuda:0" = torch.ops.aten.relu.default(convolution);  convolution = None
        convert_element_type_3: "bf16[128][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_5, torch.bfloat16);  primals_5 = None
        convert_element_type_4: "bf16[128, 64, 1][64, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_4, torch.bfloat16);  primals_4 = None
        convolution_1: "bf16[4, 128, 95696][12249088, 95696, 1]cuda:0" = torch.ops.aten.convolution.default(relu, convert_element_type_4, convert_element_type_3, [1], [0], [1], False, [0], 1);  convert_element_type_3 = None
        glu: "bf16[4, 64, 95696][6124544, 95696, 1]cuda:0" = torch.ops.aten.glu.default(convolution_1, 1)
        convert_element_type_5: "bf16[128][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_7, torch.bfloat16);  primals_7 = None
        convert_element_type_6: "bf16[128, 64, 8][512, 8, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_6, torch.bfloat16);  primals_6 = None
        convolution_2: "bf16[4, 128, 23923][3062144, 23923, 1]cuda:0" = torch.ops.aten.convolution.default(glu, convert_element_type_6, convert_element_type_5, [4], [0], [1], False, [0], 1);  convert_element_type_5 = None
        relu_1: "bf16[4, 128, 23923][3062144, 23923, 1]cuda:0" = torch.ops.aten.relu.default(convolution_2);  convolution_2 = None
        convert_element_type_7: "bf16[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_9, torch.bfloat16);  primals_9 = None
        convert_element_type_8: "bf16[256, 128, 1][128, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_8, torch.bfloat16);  primals_8 = None
        convolution_3: "bf16[4, 256, 23923][6124288, 23923, 1]cuda:0" = torch.ops.aten.convolution.default(relu_1, convert_element_type_8, convert_element_type_7, [1], [0], [1], False, [0], 1);  convert_element_type_7 = None
        glu_1: "bf16[4, 128, 23923][3062144, 23923, 1]cuda:0" = torch.ops.aten.glu.default(convolution_3, 1)
        convert_element_type_9: "bf16[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_11, torch.bfloat16);  primals_11 = None
        convert_element_type_10: "bf16[256, 128, 8][1024, 8, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_10, torch.bfloat16);  primals_10 = None
        convolution_4: "bf16[4, 256, 5979][1530624, 5979, 1]cuda:0" = torch.ops.aten.convolution.default(glu_1, convert_element_type_10, convert_element_type_9, [4], [0], [1], False, [0], 1);  convert_element_type_9 = None
        relu_2: "bf16[4, 256, 5979][1530624, 5979, 1]cuda:0" = torch.ops.aten.relu.default(convolution_4);  convolution_4 = None
        convert_element_type_11: "bf16[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_13, torch.bfloat16);  primals_13 = None
        convert_element_type_12: "bf16[512, 256, 1][256, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_12, torch.bfloat16);  primals_12 = None
        convolution_5: "bf16[4, 512, 5979][3061248, 5979, 1]cuda:0" = torch.ops.aten.convolution.default(relu_2, convert_element_type_12, convert_element_type_11, [1], [0], [1], False, [0], 1);  convert_element_type_11 = None
        glu_2: "bf16[4, 256, 5979][1530624, 5979, 1]cuda:0" = torch.ops.aten.glu.default(convolution_5, 1)
        convert_element_type_13: "bf16[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_15, torch.bfloat16);  primals_15 = None
        convert_element_type_14: "bf16[512, 256, 8][2048, 8, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_14, torch.bfloat16);  primals_14 = None
        convolution_6: "bf16[4, 512, 1493][764416, 1493, 1]cuda:0" = torch.ops.aten.convolution.default(glu_2, convert_element_type_14, convert_element_type_13, [4], [0], [1], False, [0], 1);  convert_element_type_13 = None
        relu_3: "bf16[4, 512, 1493][764416, 1493, 1]cuda:0" = torch.ops.aten.relu.default(convolution_6);  convolution_6 = None
        convert_element_type_15: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_17, torch.bfloat16);  primals_17 = None
        convert_element_type_16: "bf16[1024, 512, 1][512, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_16, torch.bfloat16);  primals_16 = None
        convolution_7: "bf16[4, 1024, 1493][1528832, 1493, 1]cuda:0" = torch.ops.aten.convolution.default(relu_3, convert_element_type_16, convert_element_type_15, [1], [0], [1], False, [0], 1);  convert_element_type_15 = None
        glu_3: "bf16[4, 512, 1493][764416, 1493, 1]cuda:0" = torch.ops.aten.glu.default(convolution_7, 1)
        convert_element_type_17: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_19, torch.bfloat16);  primals_19 = None
        convert_element_type_18: "bf16[1024, 512, 8][4096, 8, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_18, torch.bfloat16);  primals_18 = None
        convolution_8: "bf16[4, 1024, 372][380928, 372, 1]cuda:0" = torch.ops.aten.convolution.default(glu_3, convert_element_type_18, convert_element_type_17, [4], [0], [1], False, [0], 1);  convert_element_type_17 = None
        relu_4: "bf16[4, 1024, 372][380928, 372, 1]cuda:0" = torch.ops.aten.relu.default(convolution_8);  convolution_8 = None
        convert_element_type_19: "bf16[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_21, torch.bfloat16);  primals_21 = None
        convert_element_type_20: "bf16[2048, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_20, torch.bfloat16);  primals_20 = None
        convolution_9: "bf16[4, 2048, 372][761856, 372, 1]cuda:0" = torch.ops.aten.convolution.default(relu_4, convert_element_type_20, convert_element_type_19, [1], [0], [1], False, [0], 1);  convert_element_type_19 = None
        glu_4: "bf16[4, 1024, 372][380928, 372, 1]cuda:0" = torch.ops.aten.glu.default(convolution_9, 1)
        convert_element_type_21: "bf16[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_23, torch.bfloat16);  primals_23 = None
        convert_element_type_22: "bf16[2048, 1024, 8][8192, 8, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_22, torch.bfloat16);  primals_22 = None
        convolution_10: "bf16[4, 2048, 92][188416, 92, 1]cuda:0" = torch.ops.aten.convolution.default(glu_4, convert_element_type_22, convert_element_type_21, [4], [0], [1], False, [0], 1);  convert_element_type_21 = None
        relu_5: "bf16[4, 2048, 92][188416, 92, 1]cuda:0" = torch.ops.aten.relu.default(convolution_10);  convolution_10 = None
        convert_element_type_23: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_25, torch.bfloat16);  primals_25 = None
        convert_element_type_24: "bf16[4096, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_24, torch.bfloat16);  primals_24 = None
        convolution_11: "bf16[4, 4096, 92][376832, 92, 1]cuda:0" = torch.ops.aten.convolution.default(relu_5, convert_element_type_24, convert_element_type_23, [1], [0], [1], False, [0], 1);  convert_element_type_23 = None
        glu_5: "bf16[4, 2048, 92][188416, 92, 1]cuda:0" = torch.ops.aten.glu.default(convolution_11, 1)
        return (glu_5, glu, glu_1, glu_2, glu_3, glu_4, convert_element_type_1, convert_element_type_2, relu, convert_element_type_4, convolution_1, glu, convert_element_type_6, relu_1, convert_element_type_8, convolution_3, glu_1, convert_element_type_10, relu_2, convert_element_type_12, convolution_5, glu_2, convert_element_type_14, relu_3, convert_element_type_16, convolution_7, glu_3, convert_element_type_18, relu_4, convert_element_type_20, convolution_9, glu_4, convert_element_type_22, relu_5, convert_element_type_24, convolution_11)
