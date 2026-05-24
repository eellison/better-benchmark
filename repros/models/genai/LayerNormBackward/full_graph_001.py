import torch
from torch import device
from math import inf, nan

class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "bf16[1152000, 512]", primals_2: "f32[512]", getitem_1: "f32[1152000, 1]", rsqrt: "f32[1152000, 1]", tangents_1: "bf16[]"):
        # File: /tmp/scratch_space/better_benchmark/capture_genai_kernels.py:386 in layernorm_bwd, code: return out.sum()
        expand: "bf16[1152000, 512]" = torch.ops.aten.expand.default(tangents_1, [1152000, 512]);  tangents_1 = None

        # File: /tmp/scratch_space/better_benchmark/capture_genai_kernels.py:385 in layernorm_bwd, code: out = F.layer_norm(x_f32, w.shape, w, None, 1e-6).to(x.dtype)
        convert_element_type_2: "f32[1152000, 512]" = torch.ops.prims.convert_element_type.default(expand, torch.float32);  expand = None
        mul_3: "f32[1152000, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_2, primals_2);  primals_2 = None
        mul_4: "f32[1152000, 512]" = torch.ops.aten.mul.Tensor(mul_3, 512)
        sum_2: "f32[1152000, 1]" = torch.ops.aten.sum.dim_IntList(mul_3, [1], True)

        # File: /tmp/scratch_space/better_benchmark/capture_genai_kernels.py:384 in layernorm_bwd, code: x_f32 = x.float()
        convert_element_type: "f32[1152000, 512]" = torch.ops.prims.convert_element_type.default(primals_1, torch.float32);  primals_1 = None

        # File: /tmp/scratch_space/better_benchmark/capture_genai_kernels.py:385 in layernorm_bwd, code: out = F.layer_norm(x_f32, w.shape, w, None, 1e-6).to(x.dtype)
        sub: "f32[1152000, 512]" = torch.ops.aten.sub.Tensor(convert_element_type, getitem_1);  convert_element_type = getitem_1 = None
        mul: "f32[1152000, 512]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        mul_5: "f32[1152000, 512]" = torch.ops.aten.mul.Tensor(mul_3, mul);  mul_3 = None
        sum_3: "f32[1152000, 1]" = torch.ops.aten.sum.dim_IntList(mul_5, [1], True);  mul_5 = None
        mul_6: "f32[1152000, 512]" = torch.ops.aten.mul.Tensor(mul, sum_3);  sum_3 = None
        sub_2: "f32[1152000, 512]" = torch.ops.aten.sub.Tensor(mul_4, sum_2);  mul_4 = sum_2 = None
        sub_3: "f32[1152000, 512]" = torch.ops.aten.sub.Tensor(sub_2, mul_6);  sub_2 = mul_6 = None
        div: "f32[1152000, 1]" = torch.ops.aten.div.Tensor(rsqrt, 512);  rsqrt = None
        mul_7: "f32[1152000, 512]" = torch.ops.aten.mul.Tensor(div, sub_3);  div = sub_3 = None
        mul_8: "f32[1152000, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_2, mul);  convert_element_type_2 = mul = None
        sum_4: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_8, [0]);  mul_8 = None

        # File: /tmp/scratch_space/better_benchmark/capture_genai_kernels.py:384 in layernorm_bwd, code: x_f32 = x.float()
        convert_element_type_3: "bf16[1152000, 512]" = torch.ops.prims.convert_element_type.default(mul_7, torch.bfloat16);  mul_7 = None
        return (convert_element_type_3, sum_4)
