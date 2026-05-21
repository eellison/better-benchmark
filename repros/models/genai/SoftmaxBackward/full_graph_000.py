class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "bf16[8192, 262144]"):
        # File: /tmp/scratch_space/better_benchmark/capture_genai_kernels.py:255 in sm_bwd, code: y = F.softmax(x, dim=-1)
        convert_element_type: "f32[8192, 262144]" = torch.ops.prims.convert_element_type.default(primals_1, torch.float32)
        amax: "f32[8192, 1]" = torch.ops.aten.amax.default(convert_element_type, [-1], True)
        sub: "f32[8192, 262144]" = torch.ops.aten.sub.Tensor(convert_element_type, amax);  convert_element_type = None
        exp: "f32[8192, 262144]" = torch.ops.aten.exp.default(sub);  sub = None
        sum_1: "f32[8192, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div: "f32[8192, 262144]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = None
        convert_element_type_1: "bf16[8192, 262144]" = torch.ops.prims.convert_element_type.default(div, torch.bfloat16);  div = None

        # File: /tmp/scratch_space/better_benchmark/capture_genai_kernels.py:256 in sm_bwd, code: return y.sum()
        sum_2: "bf16[]" = torch.ops.aten.sum.default(convert_element_type_1);  convert_element_type_1 = None
        return (sum_2, primals_1, amax, sum_1)
