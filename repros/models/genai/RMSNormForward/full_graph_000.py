import torch
from torch import device
from math import inf, nan

class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "bf16[1152000, 512]", arg1_1: "f32[512]"):
        # File: /tmp/scratch_space/better_benchmark/capture_genai_kernels.py:285 in rmsnorm_fwd, code: x_f32 = x.float()
        convert_element_type: "f32[1152000, 512]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.float32);  arg0_1 = None

        # File: /tmp/scratch_space/better_benchmark/capture_genai_kernels.py:288 in rmsnorm_fwd, code: * torch.rsqrt(torch.mean(x_f32.square(), dim=-1, keepdim=True) + 1e-6)
        pow_1: "f32[1152000, 512]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type, 2)
        mean: "f32[1152000, 1]" = torch.ops.aten.mean.dim(pow_1, [-1], True);  pow_1 = None
        add: "f32[1152000, 1]" = torch.ops.aten.add.Tensor(mean, 1e-06);  mean = None
        rsqrt: "f32[1152000, 1]" = torch.ops.aten.rsqrt.default(add);  add = None

        # File: /tmp/scratch_space/better_benchmark/capture_genai_kernels.py:287 in rmsnorm_fwd, code: x_f32
        mul: "f32[1152000, 512]" = torch.ops.aten.mul.Tensor(convert_element_type, rsqrt);  convert_element_type = rsqrt = None
        mul_1: "f32[1152000, 512]" = torch.ops.aten.mul.Tensor(mul, arg1_1);  mul = arg1_1 = None

        # File: /tmp/scratch_space/better_benchmark/capture_genai_kernels.py:290 in rmsnorm_fwd, code: ).to(x.dtype)
        convert_element_type_1: "bf16[1152000, 512]" = torch.ops.prims.convert_element_type.default(mul_1, torch.bfloat16);  mul_1 = None
        return (convert_element_type_1,)
