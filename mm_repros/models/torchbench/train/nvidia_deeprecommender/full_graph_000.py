class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[512, 197951][197951, 1]cuda:0", primals_2: "f32[512][1]cuda:0", primals_3: "f32[256, 197951][197951, 1]cuda:0", primals_4: "f32[512, 512][512, 1]cuda:0", primals_5: "f32[512][1]cuda:0", primals_6: "f32[1024, 512][512, 1]cuda:0", primals_7: "f32[1024][1]cuda:0", primals_8: "f32[512, 1024][1024, 1]cuda:0", primals_9: "f32[512][1]cuda:0", primals_10: "f32[512, 512][512, 1]cuda:0", primals_11: "f32[512][1]cuda:0", primals_12: "f32[197951, 512][512, 1]cuda:0", primals_13: "f32[197951][1]cuda:0"):
        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nvidia_deeprecommender/reco_encoder/model/model.py:132 in encode, code: input=F.linear(input=x, weight=w, bias=self.encode_b[ind]),
        convert_element_type: "bf16[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_2, torch.bfloat16);  primals_2 = None
        convert_element_type_1: "bf16[512, 197951][197951, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_1, torch.bfloat16);  primals_1 = None
        convert_element_type_2: "bf16[256, 197951][197951, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_3, torch.bfloat16);  primals_3 = None
        permute: "bf16[197951, 512][1, 197951]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1, [1, 0]);  convert_element_type_1 = None
        addmm: "bf16[256, 512][512, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type, convert_element_type_2, permute);  convert_element_type = permute = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nvidia_deeprecommender/reco_encoder/model/model.py:12 in activation, code: return F.selu(input)
        convert_element_type_6: "f32[256, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(addmm, torch.float32)
        gt: "b8[256, 512][512, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_6, 0)
        mul: "f32[256, 512][512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_6, 1.0507009873554805)
        mul_1: "f32[256, 512][512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_6, 1.0);  convert_element_type_6 = None
        expm1: "f32[256, 512][512, 1]cuda:0" = torch.ops.aten.expm1.default(mul_1);  mul_1 = None
        mul_2: "f32[256, 512][512, 1]cuda:0" = torch.ops.aten.mul.Tensor(expm1, 1.7580993408473766);  expm1 = None
        where: "f32[256, 512][512, 1]cuda:0" = torch.ops.aten.where.self(gt, mul, mul_2);  gt = mul = mul_2 = None
        convert_element_type_7: "bf16[256, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where, torch.bfloat16);  where = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nvidia_deeprecommender/reco_encoder/model/model.py:132 in encode, code: input=F.linear(input=x, weight=w, bias=self.encode_b[ind]),
        convert_element_type_8: "bf16[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_5, torch.bfloat16);  primals_5 = None
        convert_element_type_9: "bf16[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_4, torch.bfloat16);  primals_4 = None
        permute_1: "bf16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_9, [1, 0]);  convert_element_type_9 = None
        addmm_1: "bf16[256, 512][512, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_8, convert_element_type_7, permute_1);  convert_element_type_8 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nvidia_deeprecommender/reco_encoder/model/model.py:12 in activation, code: return F.selu(input)
        convert_element_type_13: "f32[256, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(addmm_1, torch.float32)
        gt_1: "b8[256, 512][512, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_13, 0)
        mul_3: "f32[256, 512][512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_13, 1.0507009873554805)
        mul_4: "f32[256, 512][512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_13, 1.0);  convert_element_type_13 = None
        expm1_1: "f32[256, 512][512, 1]cuda:0" = torch.ops.aten.expm1.default(mul_4);  mul_4 = None
        mul_5: "f32[256, 512][512, 1]cuda:0" = torch.ops.aten.mul.Tensor(expm1_1, 1.7580993408473766);  expm1_1 = None
        where_1: "f32[256, 512][512, 1]cuda:0" = torch.ops.aten.where.self(gt_1, mul_3, mul_5);  gt_1 = mul_3 = mul_5 = None
        convert_element_type_14: "bf16[256, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_1, torch.bfloat16);  where_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nvidia_deeprecommender/reco_encoder/model/model.py:132 in encode, code: input=F.linear(input=x, weight=w, bias=self.encode_b[ind]),
        convert_element_type_15: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_7, torch.bfloat16);  primals_7 = None
        convert_element_type_16: "bf16[1024, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_6, torch.bfloat16);  primals_6 = None
        permute_2: "bf16[512, 1024][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_16, [1, 0]);  convert_element_type_16 = None
        addmm_2: "bf16[256, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_15, convert_element_type_14, permute_2);  convert_element_type_15 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nvidia_deeprecommender/reco_encoder/model/model.py:12 in activation, code: return F.selu(input)
        convert_element_type_20: "f32[256, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(addmm_2, torch.float32)
        gt_2: "b8[256, 1024][1024, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_20, 0)
        mul_6: "f32[256, 1024][1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_20, 1.0507009873554805)
        mul_7: "f32[256, 1024][1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_20, 1.0);  convert_element_type_20 = None
        expm1_2: "f32[256, 1024][1024, 1]cuda:0" = torch.ops.aten.expm1.default(mul_7);  mul_7 = None
        mul_8: "f32[256, 1024][1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(expm1_2, 1.7580993408473766);  expm1_2 = None
        where_2: "f32[256, 1024][1024, 1]cuda:0" = torch.ops.aten.where.self(gt_2, mul_6, mul_8);  gt_2 = mul_6 = mul_8 = None
        convert_element_type_21: "bf16[256, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_2, torch.bfloat16);  where_2 = None

        # No stacktrace found for following nodes
        inductor_seeds_default: "i64[1][1]cuda:0" = torch.ops.prims.inductor_seeds.default(1, device(type='cuda', index=0))

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nvidia_deeprecommender/reco_encoder/model/model.py:136 in encode, code: x = self.drop(x)
        inductor_lookup_seed_default: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 0);  inductor_seeds_default = None
        inductor_random_default: "f32[256, 1024][1024, 1]cuda:0" = torch.ops.prims.inductor_random.default([256, 1024], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        convert_element_type_default: "bf16[256, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default, torch.bfloat16);  inductor_random_default = None
        gt_3: "b8[256, 1024][1024, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default, 0.8);  convert_element_type_default = None
        mul_9: "bf16[256, 1024][1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_3, convert_element_type_21);  convert_element_type_21 = None
        mul_10: "bf16[256, 1024][1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_9, 5.000000000000001);  mul_9 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nvidia_deeprecommender/reco_encoder/model/model.py:160 in decode, code: input=F.linear(input=z, weight=w, bias=self.decode_b[ind]),
        convert_element_type_22: "bf16[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_9, torch.bfloat16);  primals_9 = None
        convert_element_type_23: "bf16[512, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_8, torch.bfloat16);  primals_8 = None
        permute_3: "bf16[1024, 512][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_23, [1, 0]);  convert_element_type_23 = None
        addmm_3: "bf16[256, 512][512, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_22, mul_10, permute_3);  convert_element_type_22 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nvidia_deeprecommender/reco_encoder/model/model.py:12 in activation, code: return F.selu(input)
        convert_element_type_27: "f32[256, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(addmm_3, torch.float32)
        gt_4: "b8[256, 512][512, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_27, 0)
        mul_11: "f32[256, 512][512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_27, 1.0507009873554805)
        mul_12: "f32[256, 512][512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_27, 1.0);  convert_element_type_27 = None
        expm1_3: "f32[256, 512][512, 1]cuda:0" = torch.ops.aten.expm1.default(mul_12);  mul_12 = None
        mul_13: "f32[256, 512][512, 1]cuda:0" = torch.ops.aten.mul.Tensor(expm1_3, 1.7580993408473766);  expm1_3 = None
        where_3: "f32[256, 512][512, 1]cuda:0" = torch.ops.aten.where.self(gt_4, mul_11, mul_13);  gt_4 = mul_11 = mul_13 = None
        convert_element_type_28: "bf16[256, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_3, torch.bfloat16);  where_3 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nvidia_deeprecommender/reco_encoder/model/model.py:160 in decode, code: input=F.linear(input=z, weight=w, bias=self.decode_b[ind]),
        convert_element_type_29: "bf16[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_11, torch.bfloat16);  primals_11 = None
        convert_element_type_30: "bf16[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_10, torch.bfloat16);  primals_10 = None
        permute_4: "bf16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_30, [1, 0]);  convert_element_type_30 = None
        addmm_4: "bf16[256, 512][512, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_29, convert_element_type_28, permute_4);  convert_element_type_29 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nvidia_deeprecommender/reco_encoder/model/model.py:12 in activation, code: return F.selu(input)
        convert_element_type_34: "f32[256, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(addmm_4, torch.float32)
        gt_5: "b8[256, 512][512, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_34, 0)
        mul_14: "f32[256, 512][512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_34, 1.0507009873554805)
        mul_15: "f32[256, 512][512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_34, 1.0);  convert_element_type_34 = None
        expm1_4: "f32[256, 512][512, 1]cuda:0" = torch.ops.aten.expm1.default(mul_15);  mul_15 = None
        mul_16: "f32[256, 512][512, 1]cuda:0" = torch.ops.aten.mul.Tensor(expm1_4, 1.7580993408473766);  expm1_4 = None
        where_4: "f32[256, 512][512, 1]cuda:0" = torch.ops.aten.where.self(gt_5, mul_14, mul_16);  gt_5 = mul_14 = mul_16 = None
        convert_element_type_35: "bf16[256, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_4, torch.bfloat16);  where_4 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nvidia_deeprecommender/reco_encoder/model/model.py:160 in decode, code: input=F.linear(input=z, weight=w, bias=self.decode_b[ind]),
        convert_element_type_36: "bf16[197951][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_13, torch.bfloat16);  primals_13 = None
        convert_element_type_37: "bf16[197951, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_12, torch.bfloat16);  primals_12 = None
        permute_5: "bf16[512, 197951][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_37, [1, 0]);  convert_element_type_37 = None
        addmm_5: "bf16[256, 197951][197951, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_36, convert_element_type_35, permute_5);  convert_element_type_36 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nvidia_deeprecommender/reco_encoder/model/model.py:12 in activation, code: return F.selu(input)
        convert_element_type_41: "f32[256, 197951][197951, 1]cuda:0" = torch.ops.prims.convert_element_type.default(addmm_5, torch.float32)
        gt_6: "b8[256, 197951][197951, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_41, 0)
        mul_17: "f32[256, 197951][197951, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_41, 1.0507009873554805)
        mul_18: "f32[256, 197951][197951, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_41, 1.0);  convert_element_type_41 = None
        expm1_5: "f32[256, 197951][197951, 1]cuda:0" = torch.ops.aten.expm1.default(mul_18);  mul_18 = None
        mul_19: "f32[256, 197951][197951, 1]cuda:0" = torch.ops.aten.mul.Tensor(expm1_5, 1.7580993408473766);  expm1_5 = None
        where_5: "f32[256, 197951][197951, 1]cuda:0" = torch.ops.aten.where.self(gt_6, mul_17, mul_19);  gt_6 = mul_17 = mul_19 = None
        convert_element_type_42: "bf16[256, 197951][197951, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_5, torch.bfloat16);  where_5 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nvidia_deeprecommender/reco_encoder/model/model.py:160 in decode, code: input=F.linear(input=z, weight=w, bias=self.decode_b[ind]),
        permute_6: "bf16[197951, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_5, [1, 0]);  permute_5 = None
        permute_10: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_4, [1, 0]);  permute_4 = None
        permute_14: "bf16[512, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_3, [1, 0]);  permute_3 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nvidia_deeprecommender/reco_encoder/model/model.py:132 in encode, code: input=F.linear(input=x, weight=w, bias=self.encode_b[ind]),
        permute_18: "bf16[1024, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None
        permute_22: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_1, [1, 0]);  permute_1 = None
        return (convert_element_type_42, convert_element_type_2, addmm, convert_element_type_7, addmm_1, convert_element_type_14, addmm_2, gt_3, mul_10, addmm_3, convert_element_type_28, addmm_4, convert_element_type_35, addmm_5, permute_6, permute_10, permute_14, permute_18, permute_22)
