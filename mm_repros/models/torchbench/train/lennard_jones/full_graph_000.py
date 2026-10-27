class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[16, 1][1, 1]cuda:0", primals_2: "f32[16][1]cuda:0", primals_3: "f32[128, 1][1, 1]cuda:0", primals_4: "f32[16, 16][16, 1]cuda:0", primals_5: "f32[16][1]cuda:0", primals_6: "f32[16, 16][16, 1]cuda:0", primals_7: "f32[16][1]cuda:0", primals_8: "f32[16, 16][16, 1]cuda:0", primals_9: "f32[16][1]cuda:0", primals_10: "f32[1, 16][16, 1]cuda:0", primals_11: "f32[1][1]cuda:0"):
        # No stacktrace found for following nodes
        convert_element_type: "bf16[16][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_2, torch.bfloat16);  primals_2 = None
        convert_element_type_1: "bf16[16, 1][1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_1, torch.bfloat16);  primals_1 = None
        convert_element_type_2: "bf16[128, 1][1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_3, torch.bfloat16);  primals_3 = None
        permute: "bf16[1, 16][1, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1, [1, 0]);  convert_element_type_1 = None
        convert_element_type_3: "f32[16][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type, torch.float32);  convert_element_type = None
        convert_element_type_4: "f32[128, 1][1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2, torch.float32)
        convert_element_type_5: "f32[1, 16][1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute, torch.float32);  permute = None
        mul: "f32[128, 16][16, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_4, convert_element_type_5);  convert_element_type_4 = convert_element_type_5 = None
        mul_1: "f32[128, 16][16, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul, 1);  mul = None
        mul_2: "f32[16][1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_3, 1);  convert_element_type_3 = None
        add: "f32[128, 16][16, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_1, mul_2);  mul_1 = mul_2 = None
        convert_element_type_6: "bf16[128, 16][16, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add, torch.bfloat16);  add = None
        tanh: "bf16[128, 16][16, 1]cuda:0" = torch.ops.aten.tanh.default(convert_element_type_6);  convert_element_type_6 = None
        convert_element_type_7: "bf16[16][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_5, torch.bfloat16);  primals_5 = None
        convert_element_type_8: "bf16[16, 16][16, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_4, torch.bfloat16);  primals_4 = None
        permute_1: "bf16[16, 16][1, 16]cuda:0" = torch.ops.aten.permute.default(convert_element_type_8, [1, 0]);  convert_element_type_8 = None
        addmm: "bf16[128, 16][16, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_7, tanh, permute_1);  convert_element_type_7 = None
        tanh_1: "bf16[128, 16][16, 1]cuda:0" = torch.ops.aten.tanh.default(addmm);  addmm = None
        convert_element_type_12: "bf16[16][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_7, torch.bfloat16);  primals_7 = None
        convert_element_type_13: "bf16[16, 16][16, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_6, torch.bfloat16);  primals_6 = None
        permute_2: "bf16[16, 16][1, 16]cuda:0" = torch.ops.aten.permute.default(convert_element_type_13, [1, 0]);  convert_element_type_13 = None
        addmm_1: "bf16[128, 16][16, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_12, tanh_1, permute_2);  convert_element_type_12 = None
        tanh_2: "bf16[128, 16][16, 1]cuda:0" = torch.ops.aten.tanh.default(addmm_1);  addmm_1 = None
        convert_element_type_17: "bf16[16][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_9, torch.bfloat16);  primals_9 = None
        convert_element_type_18: "bf16[16, 16][16, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_8, torch.bfloat16);  primals_8 = None
        permute_3: "bf16[16, 16][1, 16]cuda:0" = torch.ops.aten.permute.default(convert_element_type_18, [1, 0]);  convert_element_type_18 = None
        addmm_2: "bf16[128, 16][16, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_17, tanh_2, permute_3);  convert_element_type_17 = None
        tanh_3: "bf16[128, 16][16, 1]cuda:0" = torch.ops.aten.tanh.default(addmm_2);  addmm_2 = None
        convert_element_type_22: "bf16[1][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_11, torch.bfloat16);  primals_11 = None
        convert_element_type_23: "bf16[1, 16][16, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_10, torch.bfloat16)
        permute_4: "bf16[16, 1][1, 16]cuda:0" = torch.ops.aten.permute.default(convert_element_type_23, [1, 0]);  convert_element_type_23 = None
        addmm_3: "bf16[128, 1][1, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_22, tanh_3, permute_4);  convert_element_type_22 = permute_4 = None
        permute_9: "bf16[16, 16][16, 1]cuda:0" = torch.ops.aten.permute.default(permute_3, [1, 0]);  permute_3 = None
        permute_13: "bf16[16, 16][16, 1]cuda:0" = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None
        permute_17: "bf16[16, 16][16, 1]cuda:0" = torch.ops.aten.permute.default(permute_1, [1, 0]);  permute_1 = None
        return (addmm_3, primals_10, convert_element_type_2, tanh, tanh_1, tanh_2, tanh_3, permute_9, permute_13, permute_17)
