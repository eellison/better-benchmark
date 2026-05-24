import torch
import torch._inductor.inductor_prims  # registers inductor prims ops
from torch import device
from math import inf, nan

class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "bf16[8192, 262144]", amax: "f32[8192, 1]", sum_1: "f32[8192, 1]", tangents_1: "bf16[]"):
        # File: /tmp/scratch_space/better_benchmark/capture_genai_kernels.py:256 in sm_bwd, code: return y.sum()
        expand: "bf16[8192, 262144]" = torch.ops.aten.expand.default(tangents_1, [8192, 262144]);  tangents_1 = None

        # File: /tmp/scratch_space/better_benchmark/capture_genai_kernels.py:255 in sm_bwd, code: y = F.softmax(x, dim=-1)
        convert_element_type_2: "f32[8192, 262144]" = torch.ops.prims.convert_element_type.default(expand, torch.float32);  expand = None
        convert_element_type: "f32[8192, 262144]" = torch.ops.prims.convert_element_type.default(primals_1, torch.float32);  primals_1 = None
        sub: "f32[8192, 262144]" = torch.ops.aten.sub.Tensor(convert_element_type, amax);  convert_element_type = amax = None
        exp: "f32[8192, 262144]" = torch.ops.aten.exp.default(sub);  sub = None
        div: "f32[8192, 262144]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        convert_element_type_1: "bf16[8192, 262144]" = torch.ops.prims.convert_element_type.default(div, torch.bfloat16);  div = None
        convert_element_type_3: "f32[8192, 262144]" = torch.ops.prims.convert_element_type.default(convert_element_type_1, torch.float32);  convert_element_type_1 = None
        mul: "f32[8192, 262144]" = torch.ops.aten.mul.Tensor(convert_element_type_2, convert_element_type_3);  convert_element_type_2 = None
        sum_3: "f32[8192, 1]" = torch.ops.aten.sum.dim_IntList(mul, [-1], True)
        neg: "f32[8192, 262144]" = torch.ops.aten.neg.default(convert_element_type_3);  convert_element_type_3 = None
        fma: "f32[8192, 262144]" = torch.ops.prims.fma.default(neg, sum_3, mul);  neg = sum_3 = mul = None
        convert_element_type_4: "bf16[8192, 262144]" = torch.ops.prims.convert_element_type.default(fma, torch.bfloat16);  fma = None
        return (convert_element_type_4,)
