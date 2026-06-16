class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "bf16[8192, 262144][262144, 1]cuda:0"):
        # File: /tmp/scratch_space/better_benchmark/repros/models/genai/SoftmaxBackward/full_graph_000.py:8 in forward, code: convert_element_type: "f32[8192, 262144]" = torch.ops.prims.convert_element_type.default(primals_1, torch.float32)
        convert_element_type: "f32[8192, 262144][262144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(arg0_1, torch.float32);  arg0_1 = None

        # File: /tmp/scratch_space/better_benchmark/repros/models/genai/SoftmaxBackward/full_graph_000.py:9 in forward, code: amax: "f32[8192, 1]" = torch.ops.aten.amax.default(convert_element_type, [-1], True)
        amax: "f32[8192, 1][1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type, [-1], True)

        # File: /tmp/scratch_space/better_benchmark/repros/models/genai/SoftmaxBackward/full_graph_000.py:10 in forward, code: sub: "f32[8192, 262144]" = torch.ops.aten.sub.Tensor(convert_element_type, amax);  convert_element_type = None
        sub: "f32[8192, 262144][262144, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type, amax);  convert_element_type = None

        # File: /tmp/scratch_space/better_benchmark/repros/models/genai/SoftmaxBackward/full_graph_000.py:11 in forward, code: exp: "f32[8192, 262144]" = torch.ops.aten.exp.default(sub);  sub = None
        exp: "f32[8192, 262144][262144, 1]cuda:0" = torch.ops.aten.exp.default(sub);  sub = None

        # File: /tmp/scratch_space/better_benchmark/repros/models/genai/SoftmaxBackward/full_graph_000.py:12 in forward, code: sum_1: "f32[8192, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        sum_1: "f32[8192, 1][1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)

        # File: /tmp/scratch_space/better_benchmark/repros/models/genai/SoftmaxBackward/full_graph_000.py:13 in forward, code: div: "f32[8192, 262144]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = None
        div: "f32[8192, 262144][262144, 1]cuda:0" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = None

        # File: /tmp/scratch_space/better_benchmark/repros/models/genai/SoftmaxBackward/full_graph_000.py:14 in forward, code: convert_element_type_1: "bf16[8192, 262144]" = torch.ops.prims.convert_element_type.default(div, torch.bfloat16);  div = None
        convert_element_type_1: "bf16[8192, 262144][262144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div, torch.bfloat16);  div = None

        # File: /tmp/scratch_space/better_benchmark/repros/models/genai/SoftmaxBackward/full_graph_000.py:17 in forward, code: sum_2: "bf16[]" = torch.ops.aten.sum.default(convert_element_type_1);  convert_element_type_1 = None
        sum_2: "bf16[][]cuda:0" = torch.ops.aten.sum.default(convert_element_type_1);  convert_element_type_1 = None
        return (sum_2, amax, sum_1)
