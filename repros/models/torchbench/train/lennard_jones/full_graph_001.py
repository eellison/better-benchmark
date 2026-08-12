class GraphModule(torch.nn.Module):
    def forward(self, primals_10: "f32[1, 16][16, 1]cuda:0", convert_element_type_2: "bf16[128, 1][1, 1]cuda:0", tanh: "bf16[128, 16][16, 1]cuda:0", tanh_1: "bf16[128, 16][16, 1]cuda:0", tanh_2: "bf16[128, 16][16, 1]cuda:0", tanh_3: "bf16[128, 16][16, 1]cuda:0", permute_9: "bf16[16, 16][16, 1]cuda:0", permute_13: "bf16[16, 16][16, 1]cuda:0", permute_17: "bf16[16, 16][16, 1]cuda:0", tangents_1: "bf16[128, 1][1, 1]cuda:0"):
        # No stacktrace found for following nodes
        convert_element_type_27: "f32[128, 1][1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(tangents_1, torch.float32)
        convert_element_type_23: "bf16[1, 16][16, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_10, torch.bfloat16);  primals_10 = None
        permute_4: "bf16[16, 1][1, 16]cuda:0" = torch.ops.aten.permute.default(convert_element_type_23, [1, 0]);  convert_element_type_23 = None
        permute_5: "bf16[1, 16][16, 1]cuda:0" = torch.ops.aten.permute.default(permute_4, [1, 0]);  permute_4 = None
        convert_element_type_28: "f32[1, 16][16, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_5, torch.float32);  permute_5 = None
        mul_3: "f32[128, 16][16, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_27, convert_element_type_28);  convert_element_type_27 = convert_element_type_28 = None
        convert_element_type_29: "bf16[128, 16][16, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_3, torch.bfloat16);  mul_3 = None
        permute_6: "bf16[1, 128][1, 1]cuda:0" = torch.ops.aten.permute.default(tangents_1, [1, 0])
        mm: "bf16[1, 16][16, 1]cuda:0" = torch.ops.aten.mm.default(permute_6, tanh_3);  permute_6 = None
        sum_1: "f32[1, 1][1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True, dtype = torch.float32);  tangents_1 = None
        view: "f32[1][1]cuda:0" = torch.ops.aten.reshape.default(sum_1, [1]);  sum_1 = None
        convert_element_type_32: "bf16[1][1]cuda:0" = torch.ops.prims.convert_element_type.default(view, torch.bfloat16);  view = None
        convert_element_type_33: "f32[1, 16][16, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm, torch.float32);  mm = None
        convert_element_type_34: "f32[1][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_32, torch.float32);  convert_element_type_32 = None
        convert_element_type_35: "f32[128, 16][16, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_29, torch.float32);  convert_element_type_29 = None
        convert_element_type_36: "f32[128, 16][16, 1]cuda:0" = torch.ops.prims.convert_element_type.default(tanh_3, torch.float32);  tanh_3 = None
        mul_4: "f32[128, 16][16, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_36, convert_element_type_36);  convert_element_type_36 = None
        sub: "f32[128, 16][16, 1]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_4);  mul_4 = None
        mul_5: "f32[128, 16][16, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_35, sub);  convert_element_type_35 = sub = None
        convert_element_type_37: "bf16[128, 16][16, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_5, torch.bfloat16);  mul_5 = None
        mm_1: "bf16[128, 16][16, 1]cuda:0" = torch.ops.aten.mm.default(convert_element_type_37, permute_9);  permute_9 = None
        permute_10: "bf16[16, 128][1, 16]cuda:0" = torch.ops.aten.permute.default(convert_element_type_37, [1, 0])
        mm_2: "bf16[16, 16][16, 1]cuda:0" = torch.ops.aten.mm.default(permute_10, tanh_2);  permute_10 = None
        sum_2: "f32[1, 16][16, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_37, [0], True, dtype = torch.float32);  convert_element_type_37 = None
        view_1: "f32[16][1]cuda:0" = torch.ops.aten.reshape.default(sum_2, [16]);  sum_2 = None
        convert_element_type_42: "bf16[16][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1, torch.bfloat16);  view_1 = None
        convert_element_type_43: "f32[16, 16][16, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_2, torch.float32);  mm_2 = None
        convert_element_type_44: "f32[16][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_42, torch.float32);  convert_element_type_42 = None
        convert_element_type_45: "f32[128, 16][16, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_1, torch.float32);  mm_1 = None
        convert_element_type_46: "f32[128, 16][16, 1]cuda:0" = torch.ops.prims.convert_element_type.default(tanh_2, torch.float32);  tanh_2 = None
        mul_6: "f32[128, 16][16, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_46, convert_element_type_46);  convert_element_type_46 = None
        sub_1: "f32[128, 16][16, 1]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_6);  mul_6 = None
        mul_7: "f32[128, 16][16, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_45, sub_1);  convert_element_type_45 = sub_1 = None
        convert_element_type_47: "bf16[128, 16][16, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_7, torch.bfloat16);  mul_7 = None
        mm_3: "bf16[128, 16][16, 1]cuda:0" = torch.ops.aten.mm.default(convert_element_type_47, permute_13);  permute_13 = None
        permute_14: "bf16[16, 128][1, 16]cuda:0" = torch.ops.aten.permute.default(convert_element_type_47, [1, 0])
        mm_4: "bf16[16, 16][16, 1]cuda:0" = torch.ops.aten.mm.default(permute_14, tanh_1);  permute_14 = None
        sum_3: "f32[1, 16][16, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_47, [0], True, dtype = torch.float32);  convert_element_type_47 = None
        view_2: "f32[16][1]cuda:0" = torch.ops.aten.reshape.default(sum_3, [16]);  sum_3 = None
        convert_element_type_52: "bf16[16][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_2, torch.bfloat16);  view_2 = None
        convert_element_type_53: "f32[16, 16][16, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_4, torch.float32);  mm_4 = None
        convert_element_type_54: "f32[16][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_52, torch.float32);  convert_element_type_52 = None
        convert_element_type_55: "f32[128, 16][16, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_3, torch.float32);  mm_3 = None
        convert_element_type_56: "f32[128, 16][16, 1]cuda:0" = torch.ops.prims.convert_element_type.default(tanh_1, torch.float32);  tanh_1 = None
        mul_8: "f32[128, 16][16, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_56, convert_element_type_56);  convert_element_type_56 = None
        sub_2: "f32[128, 16][16, 1]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_8);  mul_8 = None
        mul_9: "f32[128, 16][16, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_55, sub_2);  convert_element_type_55 = sub_2 = None
        convert_element_type_57: "bf16[128, 16][16, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_9, torch.bfloat16);  mul_9 = None
        mm_5: "bf16[128, 16][16, 1]cuda:0" = torch.ops.aten.mm.default(convert_element_type_57, permute_17);  permute_17 = None
        permute_18: "bf16[16, 128][1, 16]cuda:0" = torch.ops.aten.permute.default(convert_element_type_57, [1, 0])
        mm_6: "bf16[16, 16][16, 1]cuda:0" = torch.ops.aten.mm.default(permute_18, tanh);  permute_18 = None
        sum_4: "f32[1, 16][16, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_57, [0], True, dtype = torch.float32);  convert_element_type_57 = None
        view_3: "f32[16][1]cuda:0" = torch.ops.aten.reshape.default(sum_4, [16]);  sum_4 = None
        convert_element_type_62: "bf16[16][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_3, torch.bfloat16);  view_3 = None
        convert_element_type_63: "f32[16, 16][16, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_6, torch.float32);  mm_6 = None
        convert_element_type_64: "f32[16][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_62, torch.float32);  convert_element_type_62 = None
        convert_element_type_65: "f32[128, 16][16, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_5, torch.float32);  mm_5 = None
        convert_element_type_66: "f32[128, 16][16, 1]cuda:0" = torch.ops.prims.convert_element_type.default(tanh, torch.float32);  tanh = None
        mul_10: "f32[128, 16][16, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_66, convert_element_type_66);  convert_element_type_66 = None
        sub_3: "f32[128, 16][16, 1]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_10);  mul_10 = None
        mul_11: "f32[128, 16][16, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_65, sub_3);  convert_element_type_65 = sub_3 = None
        convert_element_type_67: "bf16[128, 16][16, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_11, torch.bfloat16);  mul_11 = None
        permute_21: "bf16[16, 128][1, 16]cuda:0" = torch.ops.aten.permute.default(convert_element_type_67, [1, 0])
        mm_7: "bf16[16, 1][1, 1]cuda:0" = torch.ops.aten.mm.default(permute_21, convert_element_type_2);  permute_21 = convert_element_type_2 = None
        sum_5: "f32[1, 16][16, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_67, [0], True, dtype = torch.float32);  convert_element_type_67 = None
        view_4: "f32[16][1]cuda:0" = torch.ops.aten.reshape.default(sum_5, [16]);  sum_5 = None
        convert_element_type_70: "bf16[16][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_4, torch.bfloat16);  view_4 = None
        convert_element_type_71: "f32[16, 1][1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_7, torch.float32);  mm_7 = None
        convert_element_type_72: "f32[16][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_70, torch.float32);  convert_element_type_70 = None
        return (convert_element_type_71, convert_element_type_72, None, convert_element_type_63, convert_element_type_64, convert_element_type_53, convert_element_type_54, convert_element_type_43, convert_element_type_44, convert_element_type_33, convert_element_type_34)
