class GraphModule(torch.nn.Module):
    def forward(self, primals_10: "f32[1, 16][16, 1]cuda:0", convert_element_type_2: "f16[4, 1][1, 1]cuda:0", tanh: "f16[4, 16][16, 1]cuda:0", tanh_1: "f16[4, 16][16, 1]cuda:0", tanh_2: "f16[4, 16][16, 1]cuda:0", tanh_3: "f16[4, 16][16, 1]cuda:0", permute_9: "f16[16, 16][16, 1]cuda:0", permute_13: "f16[16, 16][16, 1]cuda:0", permute_17: "f16[16, 16][16, 1]cuda:0", tangents_1: "f16[4, 1][1, 1]cuda:0"):
        # File: /var/lib/jenkins/workspace/benchmarks/dynamo/torchbench.py:486 in torch_dynamo_resume_in_forward_and_backward_pass_at_481, code: pred = mod(*cloned_inputs)
        convert_element_type_27: "f32[4, 1][1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(tangents_1, torch.float32)
        convert_element_type_23: "f16[1, 16][16, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_10, torch.float16);  primals_10 = None
        permute_4: "f16[16, 1][1, 16]cuda:0" = torch.ops.aten.permute.default(convert_element_type_23, [1, 0]);  convert_element_type_23 = None
        permute_5: "f16[1, 16][16, 1]cuda:0" = torch.ops.aten.permute.default(permute_4, [1, 0]);  permute_4 = None
        convert_element_type_28: "f32[1, 16][16, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_5, torch.float32);  permute_5 = None
        mul_3: "f32[4, 16][16, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_27, convert_element_type_28);  convert_element_type_27 = convert_element_type_28 = None
        permute_6: "f16[1, 4][1, 1]cuda:0" = torch.ops.aten.permute.default(tangents_1, [1, 0])
        mm: "f16[1, 16][16, 1]cuda:0" = torch.ops.aten.mm.default(permute_6, tanh_3);  permute_6 = None
        sum_1: "f16[1, 1][1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True);  tangents_1 = None
        view: "f16[1][1]cuda:0" = torch.ops.aten.reshape.default(sum_1, [1]);  sum_1 = None
        convert_element_type_32: "f32[1, 16][16, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm, torch.float32);  mm = None
        convert_element_type_33: "f32[1][1]cuda:0" = torch.ops.prims.convert_element_type.default(view, torch.float32);  view = None
        convert_element_type_35: "f32[4, 16][16, 1]cuda:0" = torch.ops.prims.convert_element_type.default(tanh_3, torch.float32);  tanh_3 = None
        mul_4: "f32[4, 16][16, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_35, convert_element_type_35);  convert_element_type_35 = None
        sub: "f32[4, 16][16, 1]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_4);  mul_4 = None
        mul_5: "f32[4, 16][16, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_3, sub);  mul_3 = sub = None
        convert_element_type_36: "f16[4, 16][16, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_5, torch.float16);  mul_5 = None
        mm_1: "f16[4, 16][16, 1]cuda:0" = torch.ops.aten.mm.default(convert_element_type_36, permute_9);  permute_9 = None
        permute_10: "f16[16, 4][1, 16]cuda:0" = torch.ops.aten.permute.default(convert_element_type_36, [1, 0])
        mm_2: "f16[16, 16][16, 1]cuda:0" = torch.ops.aten.mm.default(permute_10, tanh_2);  permute_10 = None
        sum_2: "f16[1, 16][16, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_36, [0], True);  convert_element_type_36 = None
        view_1: "f16[16][1]cuda:0" = torch.ops.aten.reshape.default(sum_2, [16]);  sum_2 = None
        convert_element_type_41: "f32[16, 16][16, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_2, torch.float32);  mm_2 = None
        convert_element_type_42: "f32[16][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1, torch.float32);  view_1 = None
        convert_element_type_43: "f32[4, 16][16, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_1, torch.float32);  mm_1 = None
        convert_element_type_44: "f32[4, 16][16, 1]cuda:0" = torch.ops.prims.convert_element_type.default(tanh_2, torch.float32);  tanh_2 = None
        mul_6: "f32[4, 16][16, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_44, convert_element_type_44);  convert_element_type_44 = None
        sub_1: "f32[4, 16][16, 1]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_6);  mul_6 = None
        mul_7: "f32[4, 16][16, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_43, sub_1);  convert_element_type_43 = sub_1 = None
        convert_element_type_45: "f16[4, 16][16, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_7, torch.float16);  mul_7 = None
        mm_3: "f16[4, 16][16, 1]cuda:0" = torch.ops.aten.mm.default(convert_element_type_45, permute_13);  permute_13 = None
        permute_14: "f16[16, 4][1, 16]cuda:0" = torch.ops.aten.permute.default(convert_element_type_45, [1, 0])
        mm_4: "f16[16, 16][16, 1]cuda:0" = torch.ops.aten.mm.default(permute_14, tanh_1);  permute_14 = None
        sum_3: "f16[1, 16][16, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_45, [0], True);  convert_element_type_45 = None
        view_2: "f16[16][1]cuda:0" = torch.ops.aten.reshape.default(sum_3, [16]);  sum_3 = None
        convert_element_type_50: "f32[16, 16][16, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_4, torch.float32);  mm_4 = None
        convert_element_type_51: "f32[16][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_2, torch.float32);  view_2 = None
        convert_element_type_52: "f32[4, 16][16, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_3, torch.float32);  mm_3 = None
        convert_element_type_53: "f32[4, 16][16, 1]cuda:0" = torch.ops.prims.convert_element_type.default(tanh_1, torch.float32);  tanh_1 = None
        mul_8: "f32[4, 16][16, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_53, convert_element_type_53);  convert_element_type_53 = None
        sub_2: "f32[4, 16][16, 1]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_8);  mul_8 = None
        mul_9: "f32[4, 16][16, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_52, sub_2);  convert_element_type_52 = sub_2 = None
        convert_element_type_54: "f16[4, 16][16, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_9, torch.float16);  mul_9 = None
        mm_5: "f16[4, 16][16, 1]cuda:0" = torch.ops.aten.mm.default(convert_element_type_54, permute_17);  permute_17 = None
        permute_18: "f16[16, 4][1, 16]cuda:0" = torch.ops.aten.permute.default(convert_element_type_54, [1, 0])
        mm_6: "f16[16, 16][16, 1]cuda:0" = torch.ops.aten.mm.default(permute_18, tanh);  permute_18 = None
        sum_4: "f16[1, 16][16, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_54, [0], True);  convert_element_type_54 = None
        view_3: "f16[16][1]cuda:0" = torch.ops.aten.reshape.default(sum_4, [16]);  sum_4 = None
        convert_element_type_59: "f32[16, 16][16, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_6, torch.float32);  mm_6 = None
        convert_element_type_60: "f32[16][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_3, torch.float32);  view_3 = None
        convert_element_type_61: "f32[4, 16][16, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_5, torch.float32);  mm_5 = None
        convert_element_type_62: "f32[4, 16][16, 1]cuda:0" = torch.ops.prims.convert_element_type.default(tanh, torch.float32);  tanh = None
        mul_10: "f32[4, 16][16, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_62, convert_element_type_62);  convert_element_type_62 = None
        sub_3: "f32[4, 16][16, 1]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_10);  mul_10 = None
        mul_11: "f32[4, 16][16, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_61, sub_3);  convert_element_type_61 = sub_3 = None
        convert_element_type_63: "f16[4, 16][16, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_11, torch.float16);  mul_11 = None
        permute_21: "f16[16, 4][1, 16]cuda:0" = torch.ops.aten.permute.default(convert_element_type_63, [1, 0])
        mm_7: "f16[16, 1][1, 1]cuda:0" = torch.ops.aten.mm.default(permute_21, convert_element_type_2);  permute_21 = convert_element_type_2 = None
        sum_5: "f16[1, 16][16, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_63, [0], True);  convert_element_type_63 = None
        view_4: "f16[16][1]cuda:0" = torch.ops.aten.reshape.default(sum_5, [16]);  sum_5 = None
        convert_element_type_66: "f32[16, 1][1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_7, torch.float32);  mm_7 = None
        convert_element_type_67: "f32[16][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_4, torch.float32);  view_4 = None
        return (convert_element_type_66, convert_element_type_67, None, convert_element_type_59, convert_element_type_60, convert_element_type_50, convert_element_type_51, convert_element_type_41, convert_element_type_42, convert_element_type_32, convert_element_type_33)
