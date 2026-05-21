class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "bf16[1152000, 512]", primals_2: "f32[512]", rsqrt: "f32[1152000, 1]", tangents_1: "bf16[]"):
        # File: /tmp/scratch_space/better_benchmark/capture_genai_kernels.py:325 in rmsnorm_bwd, code: return out.sum()
        expand: "bf16[1152000, 512]" = torch.ops.aten.expand.default(tangents_1, [1152000, 512]);  tangents_1 = None

        # File: /tmp/scratch_space/better_benchmark/capture_genai_kernels.py:324 in rmsnorm_bwd, code: ).to(x.dtype)
        convert_element_type_2: "f32[1152000, 512]" = torch.ops.prims.convert_element_type.default(expand, torch.float32);  expand = None

        # File: /tmp/scratch_space/better_benchmark/capture_genai_kernels.py:319 in rmsnorm_bwd, code: x_f32 = x.float()
        convert_element_type: "f32[1152000, 512]" = torch.ops.prims.convert_element_type.default(primals_1, torch.float32);  primals_1 = None

        # File: /tmp/scratch_space/better_benchmark/capture_genai_kernels.py:321 in rmsnorm_bwd, code: x_f32
        mul: "f32[1152000, 512]" = torch.ops.aten.mul.Tensor(convert_element_type, rsqrt)
        mul_2: "f32[1152000, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_2, mul);  mul = None
        mul_3: "f32[1152000, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_2, primals_2);  convert_element_type_2 = primals_2 = None
        sum_2: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(mul_2, [0], True);  mul_2 = None
        view: "f32[512]" = torch.ops.aten.reshape.default(sum_2, [512]);  sum_2 = None
        mul_4: "f32[1152000, 512]" = torch.ops.aten.mul.Tensor(mul_3, convert_element_type)
        mul_5: "f32[1152000, 512]" = torch.ops.aten.mul.Tensor(mul_3, rsqrt);  mul_3 = None
        sum_3: "f32[1152000, 1]" = torch.ops.aten.sum.dim_IntList(mul_4, [1], True);  mul_4 = None

        # File: /tmp/scratch_space/better_benchmark/capture_genai_kernels.py:322 in rmsnorm_bwd, code: * torch.rsqrt(torch.mean(x_f32.square(), dim=-1, keepdim=True) + 1e-6)
        pow_2: "f32[1152000, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt, 3);  rsqrt = None
        mul_6: "f32[1152000, 1]" = torch.ops.aten.mul.Scalar(sum_3, -0.5);  sum_3 = None
        mul_7: "f32[1152000, 1]" = torch.ops.aten.mul.Tensor(mul_6, pow_2);  mul_6 = pow_2 = None
        expand_1: "f32[1152000, 512]" = torch.ops.aten.expand.default(mul_7, [1152000, 512]);  mul_7 = None
        div: "f32[1152000, 512]" = torch.ops.aten.div.Scalar(expand_1, 512);  expand_1 = None
        pow_3: "f32[1152000, 512]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type, 1.0);  convert_element_type = None
        mul_8: "f32[1152000, 512]" = torch.ops.aten.mul.Scalar(pow_3, 2.0);  pow_3 = None
        mul_9: "f32[1152000, 512]" = torch.ops.aten.mul.Tensor(div, mul_8);  div = mul_8 = None
        add_1: "f32[1152000, 512]" = torch.ops.aten.add.Tensor(mul_5, mul_9);  mul_5 = mul_9 = None

        # File: /tmp/scratch_space/better_benchmark/capture_genai_kernels.py:319 in rmsnorm_bwd, code: x_f32 = x.float()
        convert_element_type_3: "bf16[1152000, 512]" = torch.ops.prims.convert_element_type.default(add_1, torch.bfloat16);  add_1 = None
        return (convert_element_type_3, view)
