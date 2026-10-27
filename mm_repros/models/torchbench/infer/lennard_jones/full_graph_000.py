class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "bf16[16, 1][1, 1]cuda:0", arg1_1: "bf16[16][1]cuda:0", arg2_1: "bf16[128, 1][1, 1]cuda:0", arg3_1: "bf16[16, 16][16, 1]cuda:0", arg4_1: "bf16[16][1]cuda:0", arg5_1: "bf16[16, 16][16, 1]cuda:0", arg6_1: "bf16[16][1]cuda:0", arg7_1: "bf16[16, 16][16, 1]cuda:0", arg8_1: "bf16[16][1]cuda:0", arg9_1: "bf16[1, 16][16, 1]cuda:0", arg10_1: "bf16[1][1]cuda:0"):
        # No stacktrace found for following nodes
        convert_element_type_1: "f32[128, 1][1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(arg2_1, torch.float32);  arg2_1 = None
        permute: "bf16[1, 16][1, 1]cuda:0" = torch.ops.aten.permute.default(arg0_1, [1, 0]);  arg0_1 = None
        convert_element_type_2: "f32[1, 16][1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute, torch.float32);  permute = None
        mul: "f32[128, 16][16, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1, convert_element_type_2);  convert_element_type_1 = convert_element_type_2 = None
        mul_1: "f32[128, 16][16, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul, 1);  mul = None
        convert_element_type: "f32[16][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg1_1, torch.float32);  arg1_1 = None
        mul_2: "f32[16][1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type, 1);  convert_element_type = None
        add: "f32[128, 16][16, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_1, mul_2);  mul_1 = mul_2 = None
        convert_element_type_3: "bf16[128, 16][16, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add, torch.bfloat16);  add = None
        tanh: "bf16[128, 16][16, 1]cuda:0" = torch.ops.aten.tanh.default(convert_element_type_3);  convert_element_type_3 = None
        permute_1: "bf16[16, 16][1, 16]cuda:0" = torch.ops.aten.permute.default(arg3_1, [1, 0]);  arg3_1 = None
        addmm: "bf16[128, 16][16, 1]cuda:0" = torch.ops.aten.addmm.default(arg4_1, tanh, permute_1);  arg4_1 = tanh = permute_1 = None
        tanh_1: "bf16[128, 16][16, 1]cuda:0" = torch.ops.aten.tanh.default(addmm);  addmm = None
        permute_2: "bf16[16, 16][1, 16]cuda:0" = torch.ops.aten.permute.default(arg5_1, [1, 0]);  arg5_1 = None
        addmm_1: "bf16[128, 16][16, 1]cuda:0" = torch.ops.aten.addmm.default(arg6_1, tanh_1, permute_2);  arg6_1 = tanh_1 = permute_2 = None
        tanh_2: "bf16[128, 16][16, 1]cuda:0" = torch.ops.aten.tanh.default(addmm_1);  addmm_1 = None
        permute_3: "bf16[16, 16][1, 16]cuda:0" = torch.ops.aten.permute.default(arg7_1, [1, 0]);  arg7_1 = None
        addmm_2: "bf16[128, 16][16, 1]cuda:0" = torch.ops.aten.addmm.default(arg8_1, tanh_2, permute_3);  arg8_1 = tanh_2 = permute_3 = None
        tanh_3: "bf16[128, 16][16, 1]cuda:0" = torch.ops.aten.tanh.default(addmm_2);  addmm_2 = None
        permute_4: "bf16[16, 1][1, 16]cuda:0" = torch.ops.aten.permute.default(arg9_1, [1, 0]);  arg9_1 = None
        addmm_3: "bf16[128, 1][1, 1]cuda:0" = torch.ops.aten.addmm.default(arg10_1, tanh_3, permute_4);  arg10_1 = tanh_3 = permute_4 = None
        return (addmm_3,)
