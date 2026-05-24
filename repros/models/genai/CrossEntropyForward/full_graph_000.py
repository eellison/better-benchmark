import torch
from torch import device
from math import inf, nan

class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "bf16[8192, 262144]", arg1_1: "i64[8192]"):
        # File: /tmp/scratch_space/better_benchmark/capture_genai_kernels.py:169 in ce_fwd, code: return F.cross_entropy(x, target, reduction="none")
        ne_1: "b8[8192]" = torch.ops.aten.ne.Scalar(arg1_1, -100)
        convert_element_type: "f32[8192, 262144]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.float32);  arg0_1 = None
        amax: "f32[8192, 1]" = torch.ops.aten.amax.default(convert_element_type, [1], True)
        sub: "f32[8192, 262144]" = torch.ops.aten.sub.Tensor(convert_element_type, amax);  convert_element_type = amax = None
        exp: "f32[8192, 262144]" = torch.ops.aten.exp.default(sub)
        sum_1: "f32[8192, 1]" = torch.ops.aten.sum.dim_IntList(exp, [1], True);  exp = None
        log: "f32[8192, 1]" = torch.ops.aten.log.default(sum_1);  sum_1 = None
        sub_1: "f32[8192, 262144]" = torch.ops.aten.sub.Tensor(sub, log);  sub = log = None
        convert_element_type_1: "bf16[8192, 262144]" = torch.ops.prims.convert_element_type.default(sub_1, torch.bfloat16);  sub_1 = None
        ne: "b8[8192]" = torch.ops.aten.ne.Scalar(arg1_1, -100)
        full_default: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "i64[8192]" = torch.ops.aten.where.self(ne, arg1_1, full_default);  ne = arg1_1 = full_default = None
        unsqueeze: "i64[8192, 1]" = torch.ops.aten.unsqueeze.default(where, 1);  where = None
        gather: "bf16[8192, 1]" = torch.ops.aten.gather.default(convert_element_type_1, 1, unsqueeze);  convert_element_type_1 = unsqueeze = None
        squeeze: "bf16[8192]" = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
        neg: "bf16[8192]" = torch.ops.aten.neg.default(squeeze);  squeeze = None
        full_default_1: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_1: "bf16[8192]" = torch.ops.aten.where.self(ne_1, neg, full_default_1);  ne_1 = neg = full_default_1 = None
        return (where_1,)
