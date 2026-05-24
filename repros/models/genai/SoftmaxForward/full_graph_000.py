import torch
from torch import device
from math import inf, nan

class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "bf16[8192, 262144]"):
        # File: /tmp/scratch_space/better_benchmark/capture_genai_kernels.py:227 in sm_fwd, code: return F.softmax(x, dim=-1)
        convert_element_type: "f32[8192, 262144]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.float32);  arg0_1 = None
        amax: "f32[8192, 1]" = torch.ops.aten.amax.default(convert_element_type, [-1], True)
        sub: "f32[8192, 262144]" = torch.ops.aten.sub.Tensor(convert_element_type, amax);  convert_element_type = amax = None
        exp: "f32[8192, 262144]" = torch.ops.aten.exp.default(sub);  sub = None
        sum_1: "f32[8192, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div: "f32[8192, 262144]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        convert_element_type_1: "bf16[8192, 262144]" = torch.ops.prims.convert_element_type.default(div, torch.bfloat16);  div = None
        return (convert_element_type_1,)
