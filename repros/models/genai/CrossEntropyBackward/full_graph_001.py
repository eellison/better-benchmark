import torch
from torch import device
from math import inf, nan

class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "bf16[8192, 262144]", primals_2: "i64[8192]", amax: "f32[8192, 1]", log: "f32[8192, 1]", tangents_1: "bf16[]"):
        # File: /tmp/scratch_space/better_benchmark/capture_genai_kernels.py:199 in ce_bwd, code: return loss.sum()
        expand: "bf16[8192]" = torch.ops.aten.expand.default(tangents_1, [8192]);  tangents_1 = None

        # File: /tmp/scratch_space/better_benchmark/capture_genai_kernels.py:198 in ce_bwd, code: loss = F.cross_entropy(x, target, reduction="none")
        unsqueeze_1: "i64[8192, 1]" = torch.ops.aten.unsqueeze.default(primals_2, 1);  primals_2 = None
        ne_2: "b8[8192, 1]" = torch.ops.aten.ne.Scalar(unsqueeze_1, -100)
        full_default: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_2: "i64[8192, 1]" = torch.ops.aten.where.self(ne_2, unsqueeze_1, full_default);  unsqueeze_1 = full_default = None

        # No stacktrace found for following nodes
        iota_default: "i64[262144]" = torch.ops.prims.iota.default(262144, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        view_default: "i64[1, 262144]" = torch.ops.aten.reshape.default(iota_default, [1, 262144]);  iota_default = None
        expand_default: "i64[8192, 262144]" = torch.ops.aten.expand.default(where_2, [8192, 262144]);  where_2 = None
        eq_tensor: "b8[8192, 262144]" = torch.ops.aten.eq.Tensor(expand_default, view_default);  expand_default = view_default = None
        scalar_tensor_default: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
        scalar_tensor_default_1: "f32[]" = torch.ops.aten.scalar_tensor.default(-1.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))

        # File: /tmp/scratch_space/better_benchmark/capture_genai_kernels.py:198 in ce_bwd, code: loss = F.cross_entropy(x, target, reduction="none")
        where_self: "f32[8192, 262144]" = torch.ops.aten.where.self(eq_tensor, scalar_tensor_default_1, scalar_tensor_default);  eq_tensor = scalar_tensor_default_1 = scalar_tensor_default = None
        unsqueeze_2: "bf16[8192, 1]" = torch.ops.aten.unsqueeze.default(expand, 1);  expand = None
        full_default_1: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_3: "bf16[8192, 1]" = torch.ops.aten.where.self(ne_2, unsqueeze_2, full_default_1);  ne_2 = unsqueeze_2 = full_default_1 = None
        mul: "f32[8192, 262144]" = torch.ops.aten.mul.Tensor(where_self, where_3);  where_self = where_3 = None
        convert_element_type_2: "f32[8192, 262144]" = torch.ops.prims.convert_element_type.default(mul, torch.float32);  mul = None
        convert_element_type: "f32[8192, 262144]" = torch.ops.prims.convert_element_type.default(primals_1, torch.float32);  primals_1 = None
        sub: "f32[8192, 262144]" = torch.ops.aten.sub.Tensor(convert_element_type, amax);  convert_element_type = amax = None
        sub_1: "f32[8192, 262144]" = torch.ops.aten.sub.Tensor(sub, log);  sub = log = None
        convert_element_type_1: "bf16[8192, 262144]" = torch.ops.prims.convert_element_type.default(sub_1, torch.bfloat16);  sub_1 = None
        convert_element_type_3: "f32[8192, 262144]" = torch.ops.prims.convert_element_type.default(convert_element_type_1, torch.float32);  convert_element_type_1 = None
        exp_1: "f32[8192, 262144]" = torch.ops.aten.exp.default(convert_element_type_3);  convert_element_type_3 = None
        sum_3: "f32[8192, 1]" = torch.ops.aten.sum.dim_IntList(convert_element_type_2, [1], True)
        mul_1: "f32[8192, 262144]" = torch.ops.aten.mul.Tensor(exp_1, sum_3);  exp_1 = sum_3 = None
        sub_2: "f32[8192, 262144]" = torch.ops.aten.sub.Tensor(convert_element_type_2, mul_1);  convert_element_type_2 = mul_1 = None
        convert_element_type_4: "bf16[8192, 262144]" = torch.ops.prims.convert_element_type.default(sub_2, torch.bfloat16);  sub_2 = None
        return (convert_element_type_4, None)
