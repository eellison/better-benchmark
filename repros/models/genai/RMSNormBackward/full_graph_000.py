class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "bf16[1152000, 512]", primals_2: "f32[512]"):
        # File: /tmp/scratch_space/better_benchmark/capture_genai_kernels.py:319 in rmsnorm_bwd, code: x_f32 = x.float()
        convert_element_type: "f32[1152000, 512]" = torch.ops.prims.convert_element_type.default(primals_1, torch.float32)

        # File: /tmp/scratch_space/better_benchmark/capture_genai_kernels.py:322 in rmsnorm_bwd, code: * torch.rsqrt(torch.mean(x_f32.square(), dim=-1, keepdim=True) + 1e-6)
        pow_1: "f32[1152000, 512]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type, 2)
        mean: "f32[1152000, 1]" = torch.ops.aten.mean.dim(pow_1, [-1], True);  pow_1 = None
        add: "f32[1152000, 1]" = torch.ops.aten.add.Tensor(mean, 1e-06);  mean = None
        rsqrt: "f32[1152000, 1]" = torch.ops.aten.rsqrt.default(add);  add = None

        # File: /tmp/scratch_space/better_benchmark/capture_genai_kernels.py:321 in rmsnorm_bwd, code: x_f32
        mul: "f32[1152000, 512]" = torch.ops.aten.mul.Tensor(convert_element_type, rsqrt);  convert_element_type = None
        mul_1: "f32[1152000, 512]" = torch.ops.aten.mul.Tensor(mul, primals_2);  mul = None

        # File: /tmp/scratch_space/better_benchmark/capture_genai_kernels.py:324 in rmsnorm_bwd, code: ).to(x.dtype)
        convert_element_type_1: "bf16[1152000, 512]" = torch.ops.prims.convert_element_type.default(mul_1, torch.bfloat16);  mul_1 = None

        # File: /tmp/scratch_space/better_benchmark/capture_genai_kernels.py:325 in rmsnorm_bwd, code: return out.sum()
        sum_1: "bf16[]" = torch.ops.aten.sum.default(convert_element_type_1);  convert_element_type_1 = None
        return (sum_1, primals_1, primals_2, rsqrt)
