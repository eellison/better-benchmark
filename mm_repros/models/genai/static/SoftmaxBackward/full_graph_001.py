class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "bf16[][]cuda:0", arg1_1: "bf16[8192, 262144][262144, 1]cuda:0", arg2_1: "f32[8192, 1][1, 1]cuda:0", arg3_1: "f32[8192, 1][1, 1]cuda:0"):
        # File: /tmp/scratch_space/better_benchmark/repros/models/genai/SoftmaxBackward/full_graph_001.py:13 in forward, code: convert_element_type: "f32[8192, 262144]" = torch.ops.prims.convert_element_type.default(primals_1, torch.float32);  primals_1 = None
        convert_element_type_1: "f32[8192, 262144][262144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(arg1_1, torch.float32);  arg1_1 = None

        # File: /tmp/scratch_space/better_benchmark/repros/models/genai/SoftmaxBackward/full_graph_001.py:14 in forward, code: sub: "f32[8192, 262144]" = torch.ops.aten.sub.Tensor(convert_element_type, amax);  convert_element_type = amax = None
        sub: "f32[8192, 262144][262144, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_1, arg2_1);  convert_element_type_1 = arg2_1 = None

        # File: /tmp/scratch_space/better_benchmark/repros/models/genai/SoftmaxBackward/full_graph_001.py:15 in forward, code: exp: "f32[8192, 262144]" = torch.ops.aten.exp.default(sub);  sub = None
        exp: "f32[8192, 262144][262144, 1]cuda:0" = torch.ops.aten.exp.default(sub);  sub = None

        # File: /tmp/scratch_space/better_benchmark/repros/models/genai/SoftmaxBackward/full_graph_001.py:16 in forward, code: div: "f32[8192, 262144]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        div: "f32[8192, 262144][262144, 1]cuda:0" = torch.ops.aten.div.Tensor(exp, arg3_1);  exp = arg3_1 = None

        # File: /tmp/scratch_space/better_benchmark/repros/models/genai/SoftmaxBackward/full_graph_001.py:17 in forward, code: convert_element_type_1: "bf16[8192, 262144]" = torch.ops.prims.convert_element_type.default(div, torch.bfloat16);  div = None
        convert_element_type_2: "bf16[8192, 262144][262144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div, torch.bfloat16);  div = None

        # File: /tmp/scratch_space/better_benchmark/repros/models/genai/SoftmaxBackward/full_graph_001.py:18 in forward, code: convert_element_type_3: "f32[8192, 262144]" = torch.ops.prims.convert_element_type.default(convert_element_type_1, torch.float32);  convert_element_type_1 = None
        convert_element_type_3: "f32[8192, 262144][262144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2, torch.float32);  convert_element_type_2 = None

        # File: /tmp/scratch_space/better_benchmark/repros/models/genai/SoftmaxBackward/full_graph_001.py:21 in forward, code: neg: "f32[8192, 262144]" = torch.ops.aten.neg.default(convert_element_type_3);  convert_element_type_3 = None
        neg: "f32[8192, 262144][262144, 1]cuda:0" = torch.ops.aten.neg.default(convert_element_type_3)

        # File: /tmp/scratch_space/better_benchmark/repros/models/genai/SoftmaxBackward/full_graph_001.py:9 in forward, code: expand: "bf16[8192, 262144]" = torch.ops.aten.expand.default(tangents_1, [8192, 262144]);  tangents_1 = None
        expand: "bf16[8192, 262144][0, 0]cuda:0" = torch.ops.aten.expand.default(arg0_1, [8192, 262144]);  arg0_1 = None

        # File: /tmp/scratch_space/better_benchmark/repros/models/genai/SoftmaxBackward/full_graph_001.py:12 in forward, code: convert_element_type_2: "f32[8192, 262144]" = torch.ops.prims.convert_element_type.default(expand, torch.float32);  expand = None
        convert_element_type: "f32[8192, 262144][262144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(expand, torch.float32);  expand = None

        # File: /tmp/scratch_space/better_benchmark/repros/models/genai/SoftmaxBackward/full_graph_001.py:19 in forward, code: mul: "f32[8192, 262144]" = torch.ops.aten.mul.Tensor(convert_element_type_2, convert_element_type_3);  convert_element_type_2 = None
        mul: "f32[8192, 262144][262144, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type, convert_element_type_3);  convert_element_type = convert_element_type_3 = None

        # File: /tmp/scratch_space/better_benchmark/repros/models/genai/SoftmaxBackward/full_graph_001.py:20 in forward, code: sum_3: "f32[8192, 1]" = torch.ops.aten.sum.dim_IntList(mul, [-1], True)
        sum_1: "f32[8192, 1][1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul, [-1], True)

        # File: /tmp/scratch_space/better_benchmark/repros/models/genai/SoftmaxBackward/full_graph_001.py:22 in forward, code: fma: "f32[8192, 262144]" = torch.ops.prims.fma.default(neg, sum_3, mul);  neg = sum_3 = mul = None
        fma: "f32[8192, 262144][262144, 1]cuda:0" = torch.ops.prims.fma.default(neg, sum_1, mul);  neg = sum_1 = mul = None

        # File: /tmp/scratch_space/better_benchmark/repros/models/genai/SoftmaxBackward/full_graph_001.py:23 in forward, code: convert_element_type_4: "bf16[8192, 262144]" = torch.ops.prims.convert_element_type.default(fma, torch.bfloat16);  fma = None
        convert_element_type_4: "bf16[8192, 262144][262144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma, torch.bfloat16);  fma = None
        return (convert_element_type_4,)
