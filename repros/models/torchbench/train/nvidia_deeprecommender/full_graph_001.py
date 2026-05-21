class GraphModule(torch.nn.Module):
    def forward(self, primals_3: "f32[1024, 197951]", primals_4: "f32[512, 512]", primals_6: "f32[1024, 512]", primals_8: "f32[512, 1024]", primals_10: "f32[512, 512]", primals_12: "f32[197951, 512]", addmm: "f32[1024, 512]", where: "f32[1024, 512]", addmm_1: "f32[1024, 512]", where_1: "f32[1024, 512]", addmm_2: "f32[1024, 1024]", gt_3: "b8[1024, 1024]", mul_10: "f32[1024, 1024]", addmm_3: "f32[1024, 512]", where_3: "f32[1024, 512]", addmm_4: "f32[1024, 512]", where_4: "f32[1024, 512]", addmm_5: "f32[1024, 197951]", tangents_1: "f32[1024, 197951]"):
        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nvidia_deeprecommender/reco_encoder/model/model.py:12 in activation, code: return F.selu(input)
        mul_20: "f32[1024, 197951]" = torch.ops.aten.mul.Tensor(tangents_1, 1)
        mul_21: "f32[1024, 197951]" = torch.ops.aten.mul.Tensor(mul_20, 1.7580993408473766);  mul_20 = None
        mul_22: "f32[1024, 197951]" = torch.ops.aten.mul.Tensor(addmm_5, 1)
        exp: "f32[1024, 197951]" = torch.ops.aten.exp.default(mul_22);  mul_22 = None
        mul_23: "f32[1024, 197951]" = torch.ops.aten.mul.Tensor(mul_21, exp);  mul_21 = exp = None
        mul_24: "f32[1024, 197951]" = torch.ops.aten.mul.Tensor(tangents_1, 1.0507009873554805);  tangents_1 = None
        le: "b8[1024, 197951]" = torch.ops.aten.le.Scalar(addmm_5, 0);  addmm_5 = None
        where_6: "f32[1024, 197951]" = torch.ops.aten.where.self(le, mul_23, mul_24);  le = mul_23 = mul_24 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nvidia_deeprecommender/reco_encoder/model/model.py:160 in decode, code: input=F.linear(input=z, weight=w, bias=self.decode_b[ind]),
        permute_5: "f32[512, 197951]" = torch.ops.aten.permute.default(primals_12, [1, 0]);  primals_12 = None
        permute_6: "f32[197951, 512]" = torch.ops.aten.permute.default(permute_5, [1, 0]);  permute_5 = None
        mm: "f32[1024, 512]" = torch.ops.aten.mm.default(where_6, permute_6);  permute_6 = None
        permute_7: "f32[197951, 1024]" = torch.ops.aten.permute.default(where_6, [1, 0])
        mm_1: "f32[197951, 512]" = torch.ops.aten.mm.default(permute_7, where_4);  permute_7 = where_4 = None
        sum_1: "f32[1, 197951]" = torch.ops.aten.sum.dim_IntList(where_6, [0], True);  where_6 = None
        view: "f32[197951]" = torch.ops.aten.reshape.default(sum_1, [197951]);  sum_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nvidia_deeprecommender/reco_encoder/model/model.py:12 in activation, code: return F.selu(input)
        le_1: "b8[1024, 512]" = torch.ops.aten.le.Scalar(addmm_4, 0)
        mul_25: "f32[1024, 512]" = torch.ops.aten.mul.Tensor(mm, 1)
        mul_26: "f32[1024, 512]" = torch.ops.aten.mul.Tensor(mul_25, 1.7580993408473766);  mul_25 = None
        mul_27: "f32[1024, 512]" = torch.ops.aten.mul.Tensor(addmm_4, 1);  addmm_4 = None
        exp_1: "f32[1024, 512]" = torch.ops.aten.exp.default(mul_27);  mul_27 = None
        mul_28: "f32[1024, 512]" = torch.ops.aten.mul.Tensor(mul_26, exp_1);  mul_26 = exp_1 = None
        mul_29: "f32[1024, 512]" = torch.ops.aten.mul.Tensor(mm, 1.0507009873554805);  mm = None
        where_7: "f32[1024, 512]" = torch.ops.aten.where.self(le_1, mul_28, mul_29);  le_1 = mul_28 = mul_29 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nvidia_deeprecommender/reco_encoder/model/model.py:160 in decode, code: input=F.linear(input=z, weight=w, bias=self.decode_b[ind]),
        permute_4: "f32[512, 512]" = torch.ops.aten.permute.default(primals_10, [1, 0]);  primals_10 = None
        permute_10: "f32[512, 512]" = torch.ops.aten.permute.default(permute_4, [1, 0]);  permute_4 = None
        mm_2: "f32[1024, 512]" = torch.ops.aten.mm.default(where_7, permute_10);  permute_10 = None
        permute_11: "f32[512, 1024]" = torch.ops.aten.permute.default(where_7, [1, 0])
        mm_3: "f32[512, 512]" = torch.ops.aten.mm.default(permute_11, where_3);  permute_11 = where_3 = None
        sum_2: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(where_7, [0], True);  where_7 = None
        view_1: "f32[512]" = torch.ops.aten.reshape.default(sum_2, [512]);  sum_2 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nvidia_deeprecommender/reco_encoder/model/model.py:12 in activation, code: return F.selu(input)
        le_2: "b8[1024, 512]" = torch.ops.aten.le.Scalar(addmm_3, 0)
        mul_30: "f32[1024, 512]" = torch.ops.aten.mul.Tensor(mm_2, 1)
        mul_31: "f32[1024, 512]" = torch.ops.aten.mul.Tensor(mul_30, 1.7580993408473766);  mul_30 = None
        mul_32: "f32[1024, 512]" = torch.ops.aten.mul.Tensor(addmm_3, 1);  addmm_3 = None
        exp_2: "f32[1024, 512]" = torch.ops.aten.exp.default(mul_32);  mul_32 = None
        mul_33: "f32[1024, 512]" = torch.ops.aten.mul.Tensor(mul_31, exp_2);  mul_31 = exp_2 = None
        mul_34: "f32[1024, 512]" = torch.ops.aten.mul.Tensor(mm_2, 1.0507009873554805);  mm_2 = None
        where_8: "f32[1024, 512]" = torch.ops.aten.where.self(le_2, mul_33, mul_34);  le_2 = mul_33 = mul_34 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nvidia_deeprecommender/reco_encoder/model/model.py:160 in decode, code: input=F.linear(input=z, weight=w, bias=self.decode_b[ind]),
        permute_3: "f32[1024, 512]" = torch.ops.aten.permute.default(primals_8, [1, 0]);  primals_8 = None
        permute_14: "f32[512, 1024]" = torch.ops.aten.permute.default(permute_3, [1, 0]);  permute_3 = None
        mm_4: "f32[1024, 1024]" = torch.ops.aten.mm.default(where_8, permute_14);  permute_14 = None
        permute_15: "f32[512, 1024]" = torch.ops.aten.permute.default(where_8, [1, 0])
        mm_5: "f32[512, 1024]" = torch.ops.aten.mm.default(permute_15, mul_10);  permute_15 = mul_10 = None
        sum_3: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(where_8, [0], True);  where_8 = None
        view_2: "f32[512]" = torch.ops.aten.reshape.default(sum_3, [512]);  sum_3 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nvidia_deeprecommender/reco_encoder/model/model.py:136 in encode, code: x = self.drop(x)
        convert_element_type: "f32[1024, 1024]" = torch.ops.prims.convert_element_type.default(gt_3, torch.float32);  gt_3 = None
        mul_35: "f32[1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type, 5.000000000000001);  convert_element_type = None
        mul_36: "f32[1024, 1024]" = torch.ops.aten.mul.Tensor(mm_4, mul_35);  mm_4 = mul_35 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nvidia_deeprecommender/reco_encoder/model/model.py:12 in activation, code: return F.selu(input)
        le_3: "b8[1024, 1024]" = torch.ops.aten.le.Scalar(addmm_2, 0)
        mul_37: "f32[1024, 1024]" = torch.ops.aten.mul.Tensor(mul_36, 1)
        mul_38: "f32[1024, 1024]" = torch.ops.aten.mul.Tensor(mul_37, 1.7580993408473766);  mul_37 = None
        mul_39: "f32[1024, 1024]" = torch.ops.aten.mul.Tensor(addmm_2, 1);  addmm_2 = None
        exp_3: "f32[1024, 1024]" = torch.ops.aten.exp.default(mul_39);  mul_39 = None
        mul_40: "f32[1024, 1024]" = torch.ops.aten.mul.Tensor(mul_38, exp_3);  mul_38 = exp_3 = None
        mul_41: "f32[1024, 1024]" = torch.ops.aten.mul.Tensor(mul_36, 1.0507009873554805);  mul_36 = None
        where_9: "f32[1024, 1024]" = torch.ops.aten.where.self(le_3, mul_40, mul_41);  le_3 = mul_40 = mul_41 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nvidia_deeprecommender/reco_encoder/model/model.py:132 in encode, code: input=F.linear(input=x, weight=w, bias=self.encode_b[ind]),
        permute_2: "f32[512, 1024]" = torch.ops.aten.permute.default(primals_6, [1, 0]);  primals_6 = None
        permute_18: "f32[1024, 512]" = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None
        mm_6: "f32[1024, 512]" = torch.ops.aten.mm.default(where_9, permute_18);  permute_18 = None
        permute_19: "f32[1024, 1024]" = torch.ops.aten.permute.default(where_9, [1, 0])
        mm_7: "f32[1024, 512]" = torch.ops.aten.mm.default(permute_19, where_1);  permute_19 = where_1 = None
        sum_4: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(where_9, [0], True);  where_9 = None
        view_3: "f32[1024]" = torch.ops.aten.reshape.default(sum_4, [1024]);  sum_4 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nvidia_deeprecommender/reco_encoder/model/model.py:12 in activation, code: return F.selu(input)
        le_4: "b8[1024, 512]" = torch.ops.aten.le.Scalar(addmm_1, 0)
        mul_42: "f32[1024, 512]" = torch.ops.aten.mul.Tensor(mm_6, 1)
        mul_43: "f32[1024, 512]" = torch.ops.aten.mul.Tensor(mul_42, 1.7580993408473766);  mul_42 = None
        mul_44: "f32[1024, 512]" = torch.ops.aten.mul.Tensor(addmm_1, 1);  addmm_1 = None
        exp_4: "f32[1024, 512]" = torch.ops.aten.exp.default(mul_44);  mul_44 = None
        mul_45: "f32[1024, 512]" = torch.ops.aten.mul.Tensor(mul_43, exp_4);  mul_43 = exp_4 = None
        mul_46: "f32[1024, 512]" = torch.ops.aten.mul.Tensor(mm_6, 1.0507009873554805);  mm_6 = None
        where_10: "f32[1024, 512]" = torch.ops.aten.where.self(le_4, mul_45, mul_46);  le_4 = mul_45 = mul_46 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nvidia_deeprecommender/reco_encoder/model/model.py:132 in encode, code: input=F.linear(input=x, weight=w, bias=self.encode_b[ind]),
        permute_1: "f32[512, 512]" = torch.ops.aten.permute.default(primals_4, [1, 0]);  primals_4 = None
        permute_22: "f32[512, 512]" = torch.ops.aten.permute.default(permute_1, [1, 0]);  permute_1 = None
        mm_8: "f32[1024, 512]" = torch.ops.aten.mm.default(where_10, permute_22);  permute_22 = None
        permute_23: "f32[512, 1024]" = torch.ops.aten.permute.default(where_10, [1, 0])
        mm_9: "f32[512, 512]" = torch.ops.aten.mm.default(permute_23, where);  permute_23 = where = None
        sum_5: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(where_10, [0], True);  where_10 = None
        view_4: "f32[512]" = torch.ops.aten.reshape.default(sum_5, [512]);  sum_5 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nvidia_deeprecommender/reco_encoder/model/model.py:12 in activation, code: return F.selu(input)
        le_5: "b8[1024, 512]" = torch.ops.aten.le.Scalar(addmm, 0)
        mul_47: "f32[1024, 512]" = torch.ops.aten.mul.Tensor(mm_8, 1)
        mul_48: "f32[1024, 512]" = torch.ops.aten.mul.Tensor(mul_47, 1.7580993408473766);  mul_47 = None
        mul_49: "f32[1024, 512]" = torch.ops.aten.mul.Tensor(addmm, 1);  addmm = None
        exp_5: "f32[1024, 512]" = torch.ops.aten.exp.default(mul_49);  mul_49 = None
        mul_50: "f32[1024, 512]" = torch.ops.aten.mul.Tensor(mul_48, exp_5);  mul_48 = exp_5 = None
        mul_51: "f32[1024, 512]" = torch.ops.aten.mul.Tensor(mm_8, 1.0507009873554805);  mm_8 = None
        where_11: "f32[1024, 512]" = torch.ops.aten.where.self(le_5, mul_50, mul_51);  le_5 = mul_50 = mul_51 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nvidia_deeprecommender/reco_encoder/model/model.py:132 in encode, code: input=F.linear(input=x, weight=w, bias=self.encode_b[ind]),
        permute_26: "f32[512, 1024]" = torch.ops.aten.permute.default(where_11, [1, 0])
        mm_10: "f32[512, 197951]" = torch.ops.aten.mm.default(permute_26, primals_3);  permute_26 = primals_3 = None
        sum_6: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(where_11, [0], True);  where_11 = None
        view_5: "f32[512]" = torch.ops.aten.reshape.default(sum_6, [512]);  sum_6 = None
        return (mm_10, view_5, None, mm_9, view_4, mm_7, view_3, mm_5, view_2, mm_3, view_1, mm_1, view)
