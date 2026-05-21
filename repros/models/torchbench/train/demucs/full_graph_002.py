class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[64, 2048, 92]", primals_2: "f32[64, 2048, 92]", primals_3: "f32[4096, 2048, 3]", primals_4: "f32[4096]", primals_5: "f32[2048, 1024, 8]", primals_6: "f32[1024]", primals_7: "f32[64, 1024, 372]", primals_8: "f32[2048, 1024, 3]", primals_9: "f32[2048]", primals_10: "f32[1024, 512, 8]", primals_11: "f32[512]", primals_12: "f32[64, 512, 1493]", primals_13: "f32[1024, 512, 3]", primals_14: "f32[1024]", primals_15: "f32[512, 256, 8]", primals_16: "f32[256]", primals_17: "f32[64, 256, 5979]", primals_18: "f32[512, 256, 3]", primals_19: "f32[512]", primals_20: "f32[256, 128, 8]", primals_21: "f32[128]", primals_22: "f32[64, 128, 23923]", primals_23: "f32[256, 128, 3]", primals_24: "f32[256]", primals_25: "f32[128, 64, 8]", primals_26: "f32[64]", primals_27: "f32[64, 64, 95696]", primals_28: "f32[128, 64, 3]", primals_29: "f32[128]", primals_30: "f32[64, 8, 8]", primals_31: "f32[8]"):
        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/demucs/demucs/model.py:225 in torch_dynamo_resume_in_forward_at_220, code: x = x + skip
        add: "f32[64, 2048, 92]" = torch.ops.aten.add.Tensor(primals_1, primals_2);  primals_1 = primals_2 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/demucs/demucs/model.py:226 in torch_dynamo_resume_in_forward_at_220, code: x = decode(x)
        convolution: "f32[64, 4096, 90]" = torch.ops.aten.convolution.default(add, primals_3, primals_4, [1], [0], [1], False, [0], 1);  primals_4 = None
        glu: "f32[64, 2048, 90]" = torch.ops.aten.glu.default(convolution, 1)
        convolution_1: "f32[64, 1024, 364]" = torch.ops.aten.convolution.default(glu, primals_5, primals_6, [4], [0], [1], True, [0], 1);  primals_6 = None
        relu: "f32[64, 1024, 364]" = torch.ops.aten.relu.default(convolution_1);  convolution_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/demucs/demucs/utils.py:35 in center_trim, code: tensor = tensor[..., delta // 2 : -(delta - delta // 2)]
        slice_1: "f32[64, 1024, 364]" = torch.ops.aten.slice.Tensor(primals_7, 2, 4, -4);  primals_7 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/demucs/demucs/model.py:225 in torch_dynamo_resume_in_forward_at_220, code: x = x + skip
        add_1: "f32[64, 1024, 364]" = torch.ops.aten.add.Tensor(relu, slice_1);  slice_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/demucs/demucs/model.py:226 in torch_dynamo_resume_in_forward_at_220, code: x = decode(x)
        convolution_2: "f32[64, 2048, 362]" = torch.ops.aten.convolution.default(add_1, primals_8, primals_9, [1], [0], [1], False, [0], 1);  primals_9 = None
        glu_1: "f32[64, 1024, 362]" = torch.ops.aten.glu.default(convolution_2, 1)
        convolution_3: "f32[64, 512, 1452]" = torch.ops.aten.convolution.default(glu_1, primals_10, primals_11, [4], [0], [1], True, [0], 1);  primals_11 = None
        relu_1: "f32[64, 512, 1452]" = torch.ops.aten.relu.default(convolution_3);  convolution_3 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/demucs/demucs/utils.py:35 in center_trim, code: tensor = tensor[..., delta // 2 : -(delta - delta // 2)]
        slice_2: "f32[64, 512, 1452]" = torch.ops.aten.slice.Tensor(primals_12, 2, 20, -21);  primals_12 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/demucs/demucs/model.py:225 in torch_dynamo_resume_in_forward_at_220, code: x = x + skip
        add_2: "f32[64, 512, 1452]" = torch.ops.aten.add.Tensor(relu_1, slice_2);  slice_2 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/demucs/demucs/model.py:226 in torch_dynamo_resume_in_forward_at_220, code: x = decode(x)
        convolution_4: "f32[64, 1024, 1450]" = torch.ops.aten.convolution.default(add_2, primals_13, primals_14, [1], [0], [1], False, [0], 1);  primals_14 = None
        glu_2: "f32[64, 512, 1450]" = torch.ops.aten.glu.default(convolution_4, 1)
        convolution_5: "f32[64, 256, 5804]" = torch.ops.aten.convolution.default(glu_2, primals_15, primals_16, [4], [0], [1], True, [0], 1);  primals_16 = None
        relu_2: "f32[64, 256, 5804]" = torch.ops.aten.relu.default(convolution_5);  convolution_5 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/demucs/demucs/utils.py:35 in center_trim, code: tensor = tensor[..., delta // 2 : -(delta - delta // 2)]
        slice_3: "f32[64, 256, 5804]" = torch.ops.aten.slice.Tensor(primals_17, 2, 87, -88);  primals_17 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/demucs/demucs/model.py:225 in torch_dynamo_resume_in_forward_at_220, code: x = x + skip
        add_3: "f32[64, 256, 5804]" = torch.ops.aten.add.Tensor(relu_2, slice_3);  slice_3 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/demucs/demucs/model.py:226 in torch_dynamo_resume_in_forward_at_220, code: x = decode(x)
        convolution_6: "f32[64, 512, 5802]" = torch.ops.aten.convolution.default(add_3, primals_18, primals_19, [1], [0], [1], False, [0], 1);  primals_19 = None
        glu_3: "f32[64, 256, 5802]" = torch.ops.aten.glu.default(convolution_6, 1)
        convolution_7: "f32[64, 128, 23212]" = torch.ops.aten.convolution.default(glu_3, primals_20, primals_21, [4], [0], [1], True, [0], 1);  primals_21 = None
        relu_3: "f32[64, 128, 23212]" = torch.ops.aten.relu.default(convolution_7);  convolution_7 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/demucs/demucs/utils.py:35 in center_trim, code: tensor = tensor[..., delta // 2 : -(delta - delta // 2)]
        slice_4: "f32[64, 128, 23212]" = torch.ops.aten.slice.Tensor(primals_22, 2, 355, -356);  primals_22 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/demucs/demucs/model.py:225 in torch_dynamo_resume_in_forward_at_220, code: x = x + skip
        add_4: "f32[64, 128, 23212]" = torch.ops.aten.add.Tensor(relu_3, slice_4);  slice_4 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/demucs/demucs/model.py:226 in torch_dynamo_resume_in_forward_at_220, code: x = decode(x)
        convolution_8: "f32[64, 256, 23210]" = torch.ops.aten.convolution.default(add_4, primals_23, primals_24, [1], [0], [1], False, [0], 1);  primals_24 = None
        glu_4: "f32[64, 128, 23210]" = torch.ops.aten.glu.default(convolution_8, 1)
        convolution_9: "f32[64, 64, 92844]" = torch.ops.aten.convolution.default(glu_4, primals_25, primals_26, [4], [0], [1], True, [0], 1);  primals_26 = None
        relu_4: "f32[64, 64, 92844]" = torch.ops.aten.relu.default(convolution_9);  convolution_9 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/demucs/demucs/utils.py:35 in center_trim, code: tensor = tensor[..., delta // 2 : -(delta - delta // 2)]
        slice_5: "f32[64, 64, 92844]" = torch.ops.aten.slice.Tensor(primals_27, 2, 1426, -1426);  primals_27 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/demucs/demucs/model.py:225 in torch_dynamo_resume_in_forward_at_220, code: x = x + skip
        add_5: "f32[64, 64, 92844]" = torch.ops.aten.add.Tensor(relu_4, slice_5);  slice_5 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/demucs/demucs/model.py:226 in torch_dynamo_resume_in_forward_at_220, code: x = decode(x)
        convolution_10: "f32[64, 128, 92842]" = torch.ops.aten.convolution.default(add_5, primals_28, primals_29, [1], [0], [1], False, [0], 1);  primals_29 = None
        glu_5: "f32[64, 64, 92842]" = torch.ops.aten.glu.default(convolution_10, 1)
        convolution_11: "f32[64, 8, 371372]" = torch.ops.aten.convolution.default(glu_5, primals_30, primals_31, [4], [0], [1], True, [0], 1);  primals_31 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/demucs/demucs/model.py:232 in torch_dynamo_resume_in_forward_at_220, code: x = x.view(x.size(0), self.sources, self.audio_channels, x.size(-1))
        view: "f32[64, 4, 2, 371372]" = torch.ops.aten.reshape.default(convolution_11, [64, 4, 2, 371372]);  convolution_11 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/demucs/demucs/model.py:226 in torch_dynamo_resume_in_forward_at_220, code: x = decode(x)
        le: "b8[64, 64, 92844]" = torch.ops.aten.le.Scalar(relu_4, 0);  relu_4 = None
        le_1: "b8[64, 128, 23212]" = torch.ops.aten.le.Scalar(relu_3, 0);  relu_3 = None
        le_2: "b8[64, 256, 5804]" = torch.ops.aten.le.Scalar(relu_2, 0);  relu_2 = None
        le_3: "b8[64, 512, 1452]" = torch.ops.aten.le.Scalar(relu_1, 0);  relu_1 = None
        le_4: "b8[64, 1024, 364]" = torch.ops.aten.le.Scalar(relu, 0);  relu = None
        return (view, primals_3, primals_5, primals_8, primals_10, primals_13, primals_15, primals_18, primals_20, primals_23, primals_25, primals_28, primals_30, add, convolution, glu, add_1, convolution_2, glu_1, add_2, convolution_4, glu_2, add_3, convolution_6, glu_3, add_4, convolution_8, glu_4, add_5, convolution_10, glu_5, le, le_1, le_2, le_3, le_4)
