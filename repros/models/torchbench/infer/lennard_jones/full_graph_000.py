class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "f32[16, 1]", arg1_1: "f32[16]", arg2_1: "f32[1000, 1]", arg3_1: "f32[16, 16]", arg4_1: "f32[16]", arg5_1: "f32[16, 16]", arg6_1: "f32[16]", arg7_1: "f32[16, 16]", arg8_1: "f32[16]", arg9_1: "f32[1, 16]", arg10_1: "f32[1]"):
        # No stacktrace found for following nodes
        permute: "f32[1, 16]" = torch.ops.aten.permute.default(arg0_1, [1, 0]);  arg0_1 = None
        mul: "f32[1000, 16]" = torch.ops.aten.mul.Tensor(arg2_1, permute);  arg2_1 = permute = None
        mul_1: "f32[1000, 16]" = torch.ops.aten.mul.Tensor(mul, 1);  mul = None
        mul_2: "f32[16]" = torch.ops.aten.mul.Tensor(arg1_1, 1);  arg1_1 = None
        add: "f32[1000, 16]" = torch.ops.aten.add.Tensor(mul_1, mul_2);  mul_1 = mul_2 = None
        tanh: "f32[1000, 16]" = torch.ops.aten.tanh.default(add);  add = None
        permute_1: "f32[16, 16]" = torch.ops.aten.permute.default(arg3_1, [1, 0]);  arg3_1 = None
        addmm: "f32[1000, 16]" = torch.ops.aten.addmm.default(arg4_1, tanh, permute_1);  arg4_1 = tanh = permute_1 = None
        tanh_1: "f32[1000, 16]" = torch.ops.aten.tanh.default(addmm);  addmm = None
        permute_2: "f32[16, 16]" = torch.ops.aten.permute.default(arg5_1, [1, 0]);  arg5_1 = None
        addmm_1: "f32[1000, 16]" = torch.ops.aten.addmm.default(arg6_1, tanh_1, permute_2);  arg6_1 = tanh_1 = permute_2 = None
        tanh_2: "f32[1000, 16]" = torch.ops.aten.tanh.default(addmm_1);  addmm_1 = None
        permute_3: "f32[16, 16]" = torch.ops.aten.permute.default(arg7_1, [1, 0]);  arg7_1 = None
        addmm_2: "f32[1000, 16]" = torch.ops.aten.addmm.default(arg8_1, tanh_2, permute_3);  arg8_1 = tanh_2 = permute_3 = None
        tanh_3: "f32[1000, 16]" = torch.ops.aten.tanh.default(addmm_2);  addmm_2 = None
        permute_4: "f32[16, 1]" = torch.ops.aten.permute.default(arg9_1, [1, 0]);  arg9_1 = None
        addmm_3: "f32[1000, 1]" = torch.ops.aten.addmm.default(arg10_1, tanh_3, permute_4);  arg10_1 = tanh_3 = permute_4 = None
        return (addmm_3,)
