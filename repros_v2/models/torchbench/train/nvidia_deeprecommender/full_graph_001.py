class GraphModule(torch.nn.Module):
    def forward(self, convert_element_type_2: "bf16[256, 197951][197952, 1]cuda:0", addmm: "bf16[256, 512][512, 1]cuda:0", convert_element_type_7: "bf16[256, 512][512, 1]cuda:0", addmm_1: "bf16[256, 512][512, 1]cuda:0", convert_element_type_14: "bf16[256, 512][512, 1]cuda:0", addmm_2: "bf16[256, 1024][1024, 1]cuda:0", gt_3: "b8[256, 1024][1024, 1]cuda:0", mul_10: "bf16[256, 1024][1024, 1]cuda:0", addmm_3: "bf16[256, 512][512, 1]cuda:0", convert_element_type_28: "bf16[256, 512][512, 1]cuda:0", addmm_4: "bf16[256, 512][512, 1]cuda:0", convert_element_type_35: "bf16[256, 512][512, 1]cuda:0", addmm_5: "bf16[256, 197951][197952, 1]cuda:0", permute_6: "bf16[197951, 512][512, 1]cuda:0", permute_10: "bf16[512, 512][512, 1]cuda:0", permute_14: "bf16[512, 1024][1024, 1]cuda:0", permute_18: "bf16[1024, 512][512, 1]cuda:0", permute_22: "bf16[512, 512][512, 1]cuda:0", tangents_1: "bf16[256, 197951][197951, 1]cuda:0"):
        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nvidia_deeprecommender/reco_encoder/model/model.py:12 in activation, code: return F.selu(input)
        convert_element_type_43: "f32[256, 197951][197951, 1]cuda:0" = torch.ops.prims.convert_element_type.default(tangents_1, torch.float32);  tangents_1 = None
        convert_element_type_41: "f32[256, 197951][197951, 1]cuda:0" = torch.ops.prims.convert_element_type.default(addmm_5, torch.float32);  addmm_5 = None
        le: "b8[256, 197951][197951, 1]cuda:0" = torch.ops.aten.le.Scalar(convert_element_type_41, 0)
        mul_20: "f32[256, 197951][197951, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_43, 1)
        mul_21: "f32[256, 197951][197951, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_20, 1.7580993408473766);  mul_20 = None
        mul_22: "f32[256, 197951][197951, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_41, 1);  convert_element_type_41 = None
        exp: "f32[256, 197951][197951, 1]cuda:0" = torch.ops.aten.exp.default(mul_22);  mul_22 = None
        mul_23: "f32[256, 197951][197951, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_21, exp);  mul_21 = exp = None
        mul_24: "f32[256, 197951][197951, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_43, 1.0507009873554805);  convert_element_type_43 = None
        where_6: "f32[256, 197951][197951, 1]cuda:0" = torch.ops.aten.where.self(le, mul_23, mul_24);  le = mul_23 = mul_24 = None
        convert_element_type_45: "bf16[256, 197951][197951, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_6, torch.bfloat16);  where_6 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nvidia_deeprecommender/reco_encoder/model/model.py:160 in decode, code: input=F.linear(input=z, weight=w, bias=self.decode_b[ind]),
        constant_pad_nd_default_2: "bf16[256, 197952][197952, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(convert_element_type_45, [0, 1, 0, 0])
        constant_pad_nd_default_3: "bf16[197952, 512][512, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(permute_6, [0, 0, 0, 1]);  permute_6 = None
        mm_default_2: "bf16[256, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(constant_pad_nd_default_2, constant_pad_nd_default_3);  constant_pad_nd_default_2 = constant_pad_nd_default_3 = None
        permute_7: "bf16[197951, 256][1, 197951]cuda:0" = torch.ops.aten.permute.default(convert_element_type_45, [1, 0])
        constant_pad_nd_default_1: "bf16[197952, 256][256, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(permute_7, [0, 0, 0, 1]);  permute_7 = None
        mm_default_1: "bf16[197952, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(constant_pad_nd_default_1, convert_element_type_35);  constant_pad_nd_default_1 = convert_element_type_35 = None
        slice_tensor_1: "bf16[197951, 512][512, 1]cuda:0" = torch.ops.aten.slice.Tensor(mm_default_1, 0, 0, -1);  mm_default_1 = None
        sum_1: "f32[1, 197951][197951, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_45, [0], True, dtype = torch.float32);  convert_element_type_45 = None
        view: "f32[197951][1]cuda:0" = torch.ops.aten.reshape.default(sum_1, [197951]);  sum_1 = None
        convert_element_type_50: "bf16[197951][1]cuda:0" = torch.ops.prims.convert_element_type.default(view, torch.bfloat16);  view = None
        convert_element_type_51: "f32[197951, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(slice_tensor_1, torch.float32);  slice_tensor_1 = None
        convert_element_type_52: "f32[197951][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_50, torch.float32);  convert_element_type_50 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nvidia_deeprecommender/reco_encoder/model/model.py:12 in activation, code: return F.selu(input)
        convert_element_type_53: "f32[256, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_default_2, torch.float32);  mm_default_2 = None
        convert_element_type_34: "f32[256, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(addmm_4, torch.float32);  addmm_4 = None
        le_1: "b8[256, 512][512, 1]cuda:0" = torch.ops.aten.le.Scalar(convert_element_type_34, 0)
        mul_25: "f32[256, 512][512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_53, 1)
        mul_26: "f32[256, 512][512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_25, 1.7580993408473766);  mul_25 = None
        mul_27: "f32[256, 512][512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_34, 1);  convert_element_type_34 = None
        exp_1: "f32[256, 512][512, 1]cuda:0" = torch.ops.aten.exp.default(mul_27);  mul_27 = None
        mul_28: "f32[256, 512][512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_26, exp_1);  mul_26 = exp_1 = None
        mul_29: "f32[256, 512][512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_53, 1.0507009873554805);  convert_element_type_53 = None
        where_7: "f32[256, 512][512, 1]cuda:0" = torch.ops.aten.where.self(le_1, mul_28, mul_29);  le_1 = mul_28 = mul_29 = None
        convert_element_type_55: "bf16[256, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_7, torch.bfloat16);  where_7 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nvidia_deeprecommender/reco_encoder/model/model.py:160 in decode, code: input=F.linear(input=z, weight=w, bias=self.decode_b[ind]),
        mm_2: "bf16[256, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(convert_element_type_55, permute_10);  permute_10 = None
        permute_11: "bf16[512, 256][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_55, [1, 0])
        mm_3: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_11, convert_element_type_28);  permute_11 = convert_element_type_28 = None
        sum_2: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_55, [0], True, dtype = torch.float32);  convert_element_type_55 = None
        view_1: "f32[512][1]cuda:0" = torch.ops.aten.reshape.default(sum_2, [512]);  sum_2 = None
        convert_element_type_60: "bf16[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1, torch.bfloat16);  view_1 = None
        convert_element_type_61: "f32[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_3, torch.float32);  mm_3 = None
        convert_element_type_62: "f32[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_60, torch.float32);  convert_element_type_60 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nvidia_deeprecommender/reco_encoder/model/model.py:12 in activation, code: return F.selu(input)
        convert_element_type_63: "f32[256, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_2, torch.float32);  mm_2 = None
        convert_element_type_27: "f32[256, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(addmm_3, torch.float32);  addmm_3 = None
        le_2: "b8[256, 512][512, 1]cuda:0" = torch.ops.aten.le.Scalar(convert_element_type_27, 0)
        mul_30: "f32[256, 512][512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_63, 1)
        mul_31: "f32[256, 512][512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_30, 1.7580993408473766);  mul_30 = None
        mul_32: "f32[256, 512][512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_27, 1);  convert_element_type_27 = None
        exp_2: "f32[256, 512][512, 1]cuda:0" = torch.ops.aten.exp.default(mul_32);  mul_32 = None
        mul_33: "f32[256, 512][512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_31, exp_2);  mul_31 = exp_2 = None
        mul_34: "f32[256, 512][512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_63, 1.0507009873554805);  convert_element_type_63 = None
        where_8: "f32[256, 512][512, 1]cuda:0" = torch.ops.aten.where.self(le_2, mul_33, mul_34);  le_2 = mul_33 = mul_34 = None
        convert_element_type_65: "bf16[256, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_8, torch.bfloat16);  where_8 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nvidia_deeprecommender/reco_encoder/model/model.py:160 in decode, code: input=F.linear(input=z, weight=w, bias=self.decode_b[ind]),
        mm_4: "bf16[256, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(convert_element_type_65, permute_14);  permute_14 = None
        permute_15: "bf16[512, 256][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_65, [1, 0])
        mm_5: "bf16[512, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_15, mul_10);  permute_15 = mul_10 = None
        sum_3: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_65, [0], True, dtype = torch.float32);  convert_element_type_65 = None
        view_2: "f32[512][1]cuda:0" = torch.ops.aten.reshape.default(sum_3, [512]);  sum_3 = None
        convert_element_type_70: "bf16[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_2, torch.bfloat16);  view_2 = None
        convert_element_type_71: "f32[512, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_5, torch.float32);  mm_5 = None
        convert_element_type_72: "f32[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_70, torch.float32);  convert_element_type_70 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nvidia_deeprecommender/reco_encoder/model/model.py:136 in encode, code: x = self.drop(x)
        convert_element_type_73: "bf16[256, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_3, torch.bfloat16);  gt_3 = None
        mul_35: "bf16[256, 1024][1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_73, 5.000000000000001);  convert_element_type_73 = None
        mul_36: "bf16[256, 1024][1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mm_4, mul_35);  mm_4 = mul_35 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nvidia_deeprecommender/reco_encoder/model/model.py:12 in activation, code: return F.selu(input)
        convert_element_type_74: "f32[256, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_36, torch.float32);  mul_36 = None
        convert_element_type_20: "f32[256, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(addmm_2, torch.float32);  addmm_2 = None
        le_3: "b8[256, 1024][1024, 1]cuda:0" = torch.ops.aten.le.Scalar(convert_element_type_20, 0)
        mul_37: "f32[256, 1024][1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_74, 1)
        mul_38: "f32[256, 1024][1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_37, 1.7580993408473766);  mul_37 = None
        mul_39: "f32[256, 1024][1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_20, 1);  convert_element_type_20 = None
        exp_3: "f32[256, 1024][1024, 1]cuda:0" = torch.ops.aten.exp.default(mul_39);  mul_39 = None
        mul_40: "f32[256, 1024][1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_38, exp_3);  mul_38 = exp_3 = None
        mul_41: "f32[256, 1024][1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_74, 1.0507009873554805);  convert_element_type_74 = None
        where_9: "f32[256, 1024][1024, 1]cuda:0" = torch.ops.aten.where.self(le_3, mul_40, mul_41);  le_3 = mul_40 = mul_41 = None
        convert_element_type_76: "bf16[256, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_9, torch.bfloat16);  where_9 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nvidia_deeprecommender/reco_encoder/model/model.py:132 in encode, code: input=F.linear(input=x, weight=w, bias=self.encode_b[ind]),
        mm_6: "bf16[256, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(convert_element_type_76, permute_18);  permute_18 = None
        permute_19: "bf16[1024, 256][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_76, [1, 0])
        mm_7: "bf16[1024, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_19, convert_element_type_14);  permute_19 = convert_element_type_14 = None
        sum_4: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_76, [0], True, dtype = torch.float32);  convert_element_type_76 = None
        view_3: "f32[1024][1]cuda:0" = torch.ops.aten.reshape.default(sum_4, [1024]);  sum_4 = None
        convert_element_type_81: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_3, torch.bfloat16);  view_3 = None
        convert_element_type_82: "f32[1024, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_7, torch.float32);  mm_7 = None
        convert_element_type_83: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_81, torch.float32);  convert_element_type_81 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nvidia_deeprecommender/reco_encoder/model/model.py:12 in activation, code: return F.selu(input)
        convert_element_type_84: "f32[256, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_6, torch.float32);  mm_6 = None
        convert_element_type_13: "f32[256, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(addmm_1, torch.float32);  addmm_1 = None
        le_4: "b8[256, 512][512, 1]cuda:0" = torch.ops.aten.le.Scalar(convert_element_type_13, 0)
        mul_42: "f32[256, 512][512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_84, 1)
        mul_43: "f32[256, 512][512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_42, 1.7580993408473766);  mul_42 = None
        mul_44: "f32[256, 512][512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_13, 1);  convert_element_type_13 = None
        exp_4: "f32[256, 512][512, 1]cuda:0" = torch.ops.aten.exp.default(mul_44);  mul_44 = None
        mul_45: "f32[256, 512][512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_43, exp_4);  mul_43 = exp_4 = None
        mul_46: "f32[256, 512][512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_84, 1.0507009873554805);  convert_element_type_84 = None
        where_10: "f32[256, 512][512, 1]cuda:0" = torch.ops.aten.where.self(le_4, mul_45, mul_46);  le_4 = mul_45 = mul_46 = None
        convert_element_type_86: "bf16[256, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_10, torch.bfloat16);  where_10 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nvidia_deeprecommender/reco_encoder/model/model.py:132 in encode, code: input=F.linear(input=x, weight=w, bias=self.encode_b[ind]),
        mm_8: "bf16[256, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(convert_element_type_86, permute_22);  permute_22 = None
        permute_23: "bf16[512, 256][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_86, [1, 0])
        mm_9: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_23, convert_element_type_7);  permute_23 = convert_element_type_7 = None
        sum_5: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_86, [0], True, dtype = torch.float32);  convert_element_type_86 = None
        view_4: "f32[512][1]cuda:0" = torch.ops.aten.reshape.default(sum_5, [512]);  sum_5 = None
        convert_element_type_91: "bf16[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_4, torch.bfloat16);  view_4 = None
        convert_element_type_92: "f32[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_9, torch.float32);  mm_9 = None
        convert_element_type_93: "f32[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_91, torch.float32);  convert_element_type_91 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nvidia_deeprecommender/reco_encoder/model/model.py:12 in activation, code: return F.selu(input)
        convert_element_type_94: "f32[256, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_8, torch.float32);  mm_8 = None
        convert_element_type_6: "f32[256, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(addmm, torch.float32);  addmm = None
        le_5: "b8[256, 512][512, 1]cuda:0" = torch.ops.aten.le.Scalar(convert_element_type_6, 0)
        mul_47: "f32[256, 512][512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_94, 1)
        mul_48: "f32[256, 512][512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_47, 1.7580993408473766);  mul_47 = None
        mul_49: "f32[256, 512][512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_6, 1);  convert_element_type_6 = None
        exp_5: "f32[256, 512][512, 1]cuda:0" = torch.ops.aten.exp.default(mul_49);  mul_49 = None
        mul_50: "f32[256, 512][512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_48, exp_5);  mul_48 = exp_5 = None
        mul_51: "f32[256, 512][512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_94, 1.0507009873554805);  convert_element_type_94 = None
        where_11: "f32[256, 512][512, 1]cuda:0" = torch.ops.aten.where.self(le_5, mul_50, mul_51);  le_5 = mul_50 = mul_51 = None
        convert_element_type_96: "bf16[256, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_11, torch.bfloat16);  where_11 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nvidia_deeprecommender/reco_encoder/model/model.py:132 in encode, code: input=F.linear(input=x, weight=w, bias=self.encode_b[ind]),
        permute_26: "bf16[512, 256][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_96, [1, 0])
        constant_pad_nd_default: "bf16[256, 197952][197952, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(convert_element_type_2, [0, 1, 0, 0]);  convert_element_type_2 = None
        mm_default: "bf16[512, 197952][197952, 1]cuda:0" = torch.ops.aten.mm.default(permute_26, constant_pad_nd_default);  permute_26 = constant_pad_nd_default = None
        slice_tensor: "bf16[512, 197951][197952, 1]cuda:0" = torch.ops.aten.slice.Tensor(mm_default, 1, 0, -1);  mm_default = None
        sum_6: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_96, [0], True, dtype = torch.float32);  convert_element_type_96 = None
        view_5: "f32[512][1]cuda:0" = torch.ops.aten.reshape.default(sum_6, [512]);  sum_6 = None
        convert_element_type_99: "bf16[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_5, torch.bfloat16);  view_5 = None
        convert_element_type_100: "f32[512, 197951][197951, 1]cuda:0" = torch.ops.prims.convert_element_type.default(slice_tensor, torch.float32);  slice_tensor = None
        convert_element_type_101: "f32[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_99, torch.float32);  convert_element_type_99 = None
        return (convert_element_type_100, convert_element_type_101, None, convert_element_type_92, convert_element_type_93, convert_element_type_82, convert_element_type_83, convert_element_type_71, convert_element_type_72, convert_element_type_61, convert_element_type_62, convert_element_type_51, convert_element_type_52)
