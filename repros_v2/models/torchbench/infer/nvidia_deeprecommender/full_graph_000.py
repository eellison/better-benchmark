class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "bf16[512, 197951][197951, 1]cuda:0", arg1_1: "bf16[512][1]cuda:0", arg2_1: "bf16[256, 197951][197951, 1]cuda:0", arg3_1: "bf16[512, 512][512, 1]cuda:0", arg4_1: "bf16[512][1]cuda:0", arg5_1: "bf16[1024, 512][512, 1]cuda:0", arg6_1: "bf16[1024][1]cuda:0", arg7_1: "bf16[512, 1024][1024, 1]cuda:0", arg8_1: "bf16[512][1]cuda:0", arg9_1: "bf16[512, 512][512, 1]cuda:0", arg10_1: "bf16[512][1]cuda:0", arg11_1: "bf16[197951, 512][512, 1]cuda:0", arg12_1: "bf16[197951][1]cuda:0"):
        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nvidia_deeprecommender/reco_encoder/model/model.py:132 in encode, code: input=F.linear(input=x, weight=w, bias=self.encode_b[ind]),
        permute: "bf16[197951, 512][1, 197951]cuda:0" = torch.ops.aten.permute.default(arg0_1, [1, 0]);  arg0_1 = None
        addmm: "bf16[256, 512][512, 1]cuda:0" = torch.ops.aten.addmm.default(arg1_1, arg2_1, permute);  arg1_1 = arg2_1 = permute = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nvidia_deeprecommender/reco_encoder/model/model.py:12 in activation, code: return F.selu(input)
        convert_element_type_3: "f32[256, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(addmm, torch.float32);  addmm = None
        gt: "b8[256, 512][512, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_3, 0)
        mul: "f32[256, 512][512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_3, 1.0507009873554805)
        mul_1: "f32[256, 512][512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_3, 1.0);  convert_element_type_3 = None
        expm1: "f32[256, 512][512, 1]cuda:0" = torch.ops.aten.expm1.default(mul_1);  mul_1 = None
        mul_2: "f32[256, 512][512, 1]cuda:0" = torch.ops.aten.mul.Tensor(expm1, 1.7580993408473766);  expm1 = None
        where: "f32[256, 512][512, 1]cuda:0" = torch.ops.aten.where.self(gt, mul, mul_2);  gt = mul = mul_2 = None
        convert_element_type_4: "bf16[256, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where, torch.bfloat16);  where = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nvidia_deeprecommender/reco_encoder/model/model.py:132 in encode, code: input=F.linear(input=x, weight=w, bias=self.encode_b[ind]),
        permute_1: "bf16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(arg3_1, [1, 0]);  arg3_1 = None
        addmm_1: "bf16[256, 512][512, 1]cuda:0" = torch.ops.aten.addmm.default(arg4_1, convert_element_type_4, permute_1);  arg4_1 = convert_element_type_4 = permute_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nvidia_deeprecommender/reco_encoder/model/model.py:12 in activation, code: return F.selu(input)
        convert_element_type_8: "f32[256, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(addmm_1, torch.float32);  addmm_1 = None
        gt_1: "b8[256, 512][512, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_8, 0)
        mul_3: "f32[256, 512][512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_8, 1.0507009873554805)
        mul_4: "f32[256, 512][512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_8, 1.0);  convert_element_type_8 = None
        expm1_1: "f32[256, 512][512, 1]cuda:0" = torch.ops.aten.expm1.default(mul_4);  mul_4 = None
        mul_5: "f32[256, 512][512, 1]cuda:0" = torch.ops.aten.mul.Tensor(expm1_1, 1.7580993408473766);  expm1_1 = None
        where_1: "f32[256, 512][512, 1]cuda:0" = torch.ops.aten.where.self(gt_1, mul_3, mul_5);  gt_1 = mul_3 = mul_5 = None
        convert_element_type_9: "bf16[256, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_1, torch.bfloat16);  where_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nvidia_deeprecommender/reco_encoder/model/model.py:132 in encode, code: input=F.linear(input=x, weight=w, bias=self.encode_b[ind]),
        permute_2: "bf16[512, 1024][1, 512]cuda:0" = torch.ops.aten.permute.default(arg5_1, [1, 0]);  arg5_1 = None
        addmm_2: "bf16[256, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(arg6_1, convert_element_type_9, permute_2);  arg6_1 = convert_element_type_9 = permute_2 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nvidia_deeprecommender/reco_encoder/model/model.py:12 in activation, code: return F.selu(input)
        convert_element_type_13: "f32[256, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(addmm_2, torch.float32);  addmm_2 = None
        gt_2: "b8[256, 1024][1024, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_13, 0)
        mul_6: "f32[256, 1024][1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_13, 1.0507009873554805)
        mul_7: "f32[256, 1024][1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_13, 1.0);  convert_element_type_13 = None
        expm1_2: "f32[256, 1024][1024, 1]cuda:0" = torch.ops.aten.expm1.default(mul_7);  mul_7 = None
        mul_8: "f32[256, 1024][1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(expm1_2, 1.7580993408473766);  expm1_2 = None
        where_2: "f32[256, 1024][1024, 1]cuda:0" = torch.ops.aten.where.self(gt_2, mul_6, mul_8);  gt_2 = mul_6 = mul_8 = None
        convert_element_type_14: "bf16[256, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_2, torch.bfloat16);  where_2 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nvidia_deeprecommender/reco_encoder/model/model.py:160 in decode, code: input=F.linear(input=z, weight=w, bias=self.decode_b[ind]),
        permute_3: "bf16[1024, 512][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg7_1, [1, 0]);  arg7_1 = None
        addmm_3: "bf16[256, 512][512, 1]cuda:0" = torch.ops.aten.addmm.default(arg8_1, convert_element_type_14, permute_3);  arg8_1 = convert_element_type_14 = permute_3 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nvidia_deeprecommender/reco_encoder/model/model.py:12 in activation, code: return F.selu(input)
        convert_element_type_18: "f32[256, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(addmm_3, torch.float32);  addmm_3 = None
        gt_3: "b8[256, 512][512, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_18, 0)
        mul_9: "f32[256, 512][512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_18, 1.0507009873554805)
        mul_10: "f32[256, 512][512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_18, 1.0);  convert_element_type_18 = None
        expm1_3: "f32[256, 512][512, 1]cuda:0" = torch.ops.aten.expm1.default(mul_10);  mul_10 = None
        mul_11: "f32[256, 512][512, 1]cuda:0" = torch.ops.aten.mul.Tensor(expm1_3, 1.7580993408473766);  expm1_3 = None
        where_3: "f32[256, 512][512, 1]cuda:0" = torch.ops.aten.where.self(gt_3, mul_9, mul_11);  gt_3 = mul_9 = mul_11 = None
        convert_element_type_19: "bf16[256, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_3, torch.bfloat16);  where_3 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nvidia_deeprecommender/reco_encoder/model/model.py:160 in decode, code: input=F.linear(input=z, weight=w, bias=self.decode_b[ind]),
        permute_4: "bf16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(arg9_1, [1, 0]);  arg9_1 = None
        addmm_4: "bf16[256, 512][512, 1]cuda:0" = torch.ops.aten.addmm.default(arg10_1, convert_element_type_19, permute_4);  arg10_1 = convert_element_type_19 = permute_4 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nvidia_deeprecommender/reco_encoder/model/model.py:12 in activation, code: return F.selu(input)
        convert_element_type_23: "f32[256, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(addmm_4, torch.float32);  addmm_4 = None
        gt_4: "b8[256, 512][512, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_23, 0)
        mul_12: "f32[256, 512][512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_23, 1.0507009873554805)
        mul_13: "f32[256, 512][512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_23, 1.0);  convert_element_type_23 = None
        expm1_4: "f32[256, 512][512, 1]cuda:0" = torch.ops.aten.expm1.default(mul_13);  mul_13 = None
        mul_14: "f32[256, 512][512, 1]cuda:0" = torch.ops.aten.mul.Tensor(expm1_4, 1.7580993408473766);  expm1_4 = None
        where_4: "f32[256, 512][512, 1]cuda:0" = torch.ops.aten.where.self(gt_4, mul_12, mul_14);  gt_4 = mul_12 = mul_14 = None
        convert_element_type_24: "bf16[256, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_4, torch.bfloat16);  where_4 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nvidia_deeprecommender/reco_encoder/model/model.py:160 in decode, code: input=F.linear(input=z, weight=w, bias=self.decode_b[ind]),
        permute_5: "bf16[512, 197951][1, 512]cuda:0" = torch.ops.aten.permute.default(arg11_1, [1, 0]);  arg11_1 = None
        addmm_5: "bf16[256, 197951][197951, 1]cuda:0" = torch.ops.aten.addmm.default(arg12_1, convert_element_type_24, permute_5);  arg12_1 = convert_element_type_24 = permute_5 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nvidia_deeprecommender/reco_encoder/model/model.py:12 in activation, code: return F.selu(input)
        convert_element_type_28: "f32[256, 197951][197951, 1]cuda:0" = torch.ops.prims.convert_element_type.default(addmm_5, torch.float32);  addmm_5 = None
        gt_5: "b8[256, 197951][197951, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_28, 0)
        mul_15: "f32[256, 197951][197951, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_28, 1.0507009873554805)
        mul_16: "f32[256, 197951][197951, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_28, 1.0);  convert_element_type_28 = None
        expm1_5: "f32[256, 197951][197951, 1]cuda:0" = torch.ops.aten.expm1.default(mul_16);  mul_16 = None
        mul_17: "f32[256, 197951][197951, 1]cuda:0" = torch.ops.aten.mul.Tensor(expm1_5, 1.7580993408473766);  expm1_5 = None
        where_5: "f32[256, 197951][197951, 1]cuda:0" = torch.ops.aten.where.self(gt_5, mul_15, mul_17);  gt_5 = mul_15 = mul_17 = None
        convert_element_type_29: "bf16[256, 197951][197951, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_5, torch.bfloat16);  where_5 = None
        return (convert_element_type_29,)
