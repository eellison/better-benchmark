class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[64, 2, 8]", primals_2: "f32[64]", primals_3: "f32[64, 2, 382788]", primals_4: "f32[128, 64, 1]", primals_5: "f32[128]", primals_6: "f32[128, 64, 8]", primals_7: "f32[128]", primals_8: "f32[256, 128, 1]", primals_9: "f32[256]", primals_10: "f32[256, 128, 8]", primals_11: "f32[256]", primals_12: "f32[512, 256, 1]", primals_13: "f32[512]", primals_14: "f32[512, 256, 8]", primals_15: "f32[512]", primals_16: "f32[1024, 512, 1]", primals_17: "f32[1024]", primals_18: "f32[1024, 512, 8]", primals_19: "f32[1024]", primals_20: "f32[2048, 1024, 1]", primals_21: "f32[2048]", primals_22: "f32[2048, 1024, 8]", primals_23: "f32[2048]", primals_24: "f32[4096, 2048, 1]", primals_25: "f32[4096]"):
        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/demucs/demucs/model.py:215 in forward, code: x = encode(x)
        convolution: "f32[64, 64, 95696]" = torch.ops.aten.convolution.default(primals_3, primals_1, primals_2, [4], [0], [1], False, [0], 1);  primals_2 = None
        relu: "f32[64, 64, 95696]" = torch.ops.aten.relu.default(convolution);  convolution = None
        convolution_1: "f32[64, 128, 95696]" = torch.ops.aten.convolution.default(relu, primals_4, primals_5, [1], [0], [1], False, [0], 1);  primals_5 = None
        glu: "f32[64, 64, 95696]" = torch.ops.aten.glu.default(convolution_1, 1)
        convolution_2: "f32[64, 128, 23923]" = torch.ops.aten.convolution.default(glu, primals_6, primals_7, [4], [0], [1], False, [0], 1);  primals_7 = None
        relu_1: "f32[64, 128, 23923]" = torch.ops.aten.relu.default(convolution_2);  convolution_2 = None
        convolution_3: "f32[64, 256, 23923]" = torch.ops.aten.convolution.default(relu_1, primals_8, primals_9, [1], [0], [1], False, [0], 1);  primals_9 = None
        glu_1: "f32[64, 128, 23923]" = torch.ops.aten.glu.default(convolution_3, 1)
        convolution_4: "f32[64, 256, 5979]" = torch.ops.aten.convolution.default(glu_1, primals_10, primals_11, [4], [0], [1], False, [0], 1);  primals_11 = None
        relu_2: "f32[64, 256, 5979]" = torch.ops.aten.relu.default(convolution_4);  convolution_4 = None
        convolution_5: "f32[64, 512, 5979]" = torch.ops.aten.convolution.default(relu_2, primals_12, primals_13, [1], [0], [1], False, [0], 1);  primals_13 = None
        glu_2: "f32[64, 256, 5979]" = torch.ops.aten.glu.default(convolution_5, 1)
        convolution_6: "f32[64, 512, 1493]" = torch.ops.aten.convolution.default(glu_2, primals_14, primals_15, [4], [0], [1], False, [0], 1);  primals_15 = None
        relu_3: "f32[64, 512, 1493]" = torch.ops.aten.relu.default(convolution_6);  convolution_6 = None
        convolution_7: "f32[64, 1024, 1493]" = torch.ops.aten.convolution.default(relu_3, primals_16, primals_17, [1], [0], [1], False, [0], 1);  primals_17 = None
        glu_3: "f32[64, 512, 1493]" = torch.ops.aten.glu.default(convolution_7, 1)
        convolution_8: "f32[64, 1024, 372]" = torch.ops.aten.convolution.default(glu_3, primals_18, primals_19, [4], [0], [1], False, [0], 1);  primals_19 = None
        relu_4: "f32[64, 1024, 372]" = torch.ops.aten.relu.default(convolution_8);  convolution_8 = None
        convolution_9: "f32[64, 2048, 372]" = torch.ops.aten.convolution.default(relu_4, primals_20, primals_21, [1], [0], [1], False, [0], 1);  primals_21 = None
        glu_4: "f32[64, 1024, 372]" = torch.ops.aten.glu.default(convolution_9, 1)
        convolution_10: "f32[64, 2048, 92]" = torch.ops.aten.convolution.default(glu_4, primals_22, primals_23, [4], [0], [1], False, [0], 1);  primals_23 = None
        relu_5: "f32[64, 2048, 92]" = torch.ops.aten.relu.default(convolution_10);  convolution_10 = None
        convolution_11: "f32[64, 4096, 92]" = torch.ops.aten.convolution.default(relu_5, primals_24, primals_25, [1], [0], [1], False, [0], 1);  primals_25 = None
        glu_5: "f32[64, 2048, 92]" = torch.ops.aten.glu.default(convolution_11, 1)
        return (glu_5, glu, glu_1, glu_2, glu_3, glu_4, primals_1, primals_3, primals_4, primals_6, primals_8, primals_10, primals_12, primals_14, primals_16, primals_18, primals_20, primals_22, primals_24, relu, convolution_1, glu, relu_1, convolution_3, glu_1, relu_2, convolution_5, glu_2, relu_3, convolution_7, glu_3, relu_4, convolution_9, glu_4, relu_5, convolution_11)
