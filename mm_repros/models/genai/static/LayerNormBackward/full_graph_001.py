class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "bf16[][]cuda:0", arg1_1: "f32[512][1]cuda:0", arg2_1: "bf16[1152000, 512][512, 1]cuda:0", arg3_1: "f32[1152000, 1][1, 1]cuda:0", arg4_1: "f32[1152000, 1][1, 1]cuda:0"):
        # File: /tmp/scratch_space/better_benchmark/repros/models/genai/LayerNormBackward/full_graph_001.py:27 in forward, code: div: "f32[1152000, 1]" = torch.ops.aten.div.Tensor(rsqrt, 512);  rsqrt = None
        div: "f32[1152000, 1][1, 1]cuda:0" = torch.ops.aten.div.Tensor(arg4_1, 512)

        # File: /tmp/scratch_space/better_benchmark/repros/models/genai/LayerNormBackward/full_graph_001.py:8 in forward, code: expand: "bf16[1152000, 512]" = torch.ops.aten.expand.default(tangents_1, [1152000, 512]);  tangents_1 = None
        expand: "bf16[1152000, 512][0, 0]cuda:0" = torch.ops.aten.expand.default(arg0_1, [1152000, 512]);  arg0_1 = None

        # File: /tmp/scratch_space/better_benchmark/repros/models/genai/LayerNormBackward/full_graph_001.py:11 in forward, code: convert_element_type_2: "f32[1152000, 512]" = torch.ops.prims.convert_element_type.default(expand, torch.float32);  expand = None
        convert_element_type: "f32[1152000, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(expand, torch.float32);  expand = None

        # File: /tmp/scratch_space/better_benchmark/repros/models/genai/LayerNormBackward/full_graph_001.py:12 in forward, code: mul_3: "f32[1152000, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_2, primals_2);  primals_2 = None
        mul: "f32[1152000, 512][512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type, arg1_1);  arg1_1 = None

        # File: /tmp/scratch_space/better_benchmark/repros/models/genai/LayerNormBackward/full_graph_001.py:13 in forward, code: mul_4: "f32[1152000, 512]" = torch.ops.aten.mul.Tensor(mul_3, 512)
        mul_1: "f32[1152000, 512][512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul, 512)

        # File: /tmp/scratch_space/better_benchmark/repros/models/genai/LayerNormBackward/full_graph_001.py:14 in forward, code: sum_2: "f32[1152000, 1]" = torch.ops.aten.sum.dim_IntList(mul_3, [1], True)
        sum_1: "f32[1152000, 1][1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul, [1], True)

        # File: /tmp/scratch_space/better_benchmark/repros/models/genai/LayerNormBackward/full_graph_001.py:25 in forward, code: sub_2: "f32[1152000, 512]" = torch.ops.aten.sub.Tensor(mul_4, sum_2);  mul_4 = sum_2 = None
        sub_1: "f32[1152000, 512][512, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_1, sum_1);  mul_1 = sum_1 = None

        # File: /tmp/scratch_space/better_benchmark/repros/models/genai/LayerNormBackward/full_graph_001.py:17 in forward, code: convert_element_type: "f32[1152000, 512]" = torch.ops.prims.convert_element_type.default(primals_1, torch.float32);  primals_1 = None
        convert_element_type_1: "f32[1152000, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(arg2_1, torch.float32);  arg2_1 = None

        # File: /tmp/scratch_space/better_benchmark/repros/models/genai/LayerNormBackward/full_graph_001.py:20 in forward, code: sub: "f32[1152000, 512]" = torch.ops.aten.sub.Tensor(convert_element_type, getitem_1);  convert_element_type = getitem_1 = None
        sub: "f32[1152000, 512][512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_1, arg3_1);  convert_element_type_1 = arg3_1 = None

        # File: /tmp/scratch_space/better_benchmark/repros/models/genai/LayerNormBackward/full_graph_001.py:21 in forward, code: mul: "f32[1152000, 512]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        mul_2: "f32[1152000, 512][512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub, arg4_1);  sub = arg4_1 = None

        # File: /tmp/scratch_space/better_benchmark/repros/models/genai/LayerNormBackward/full_graph_001.py:22 in forward, code: mul_5: "f32[1152000, 512]" = torch.ops.aten.mul.Tensor(mul_3, mul);  mul_3 = None
        mul_3: "f32[1152000, 512][512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul, mul_2);  mul = None

        # File: /tmp/scratch_space/better_benchmark/repros/models/genai/LayerNormBackward/full_graph_001.py:23 in forward, code: sum_3: "f32[1152000, 1]" = torch.ops.aten.sum.dim_IntList(mul_5, [1], True);  mul_5 = None
        sum_2: "f32[1152000, 1][1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_3, [1], True);  mul_3 = None

        # File: /tmp/scratch_space/better_benchmark/repros/models/genai/LayerNormBackward/full_graph_001.py:24 in forward, code: mul_6: "f32[1152000, 512]" = torch.ops.aten.mul.Tensor(mul, sum_3);  sum_3 = None
        mul_4: "f32[1152000, 512][512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2, sum_2);  sum_2 = None

        # File: /tmp/scratch_space/better_benchmark/repros/models/genai/LayerNormBackward/full_graph_001.py:26 in forward, code: sub_3: "f32[1152000, 512]" = torch.ops.aten.sub.Tensor(sub_2, mul_6);  sub_2 = mul_6 = None
        sub_2: "f32[1152000, 512][512, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_1, mul_4);  sub_1 = mul_4 = None

        # File: /tmp/scratch_space/better_benchmark/repros/models/genai/LayerNormBackward/full_graph_001.py:28 in forward, code: mul_7: "f32[1152000, 512]" = torch.ops.aten.mul.Tensor(div, sub_3);  div = sub_3 = None
        mul_5: "f32[1152000, 512][512, 1]cuda:0" = torch.ops.aten.mul.Tensor(div, sub_2);  div = sub_2 = None

        # File: /tmp/scratch_space/better_benchmark/repros/models/genai/LayerNormBackward/full_graph_001.py:33 in forward, code: convert_element_type_3: "bf16[1152000, 512]" = torch.ops.prims.convert_element_type.default(mul_7, torch.bfloat16);  mul_7 = None
        convert_element_type_2: "bf16[1152000, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_5, torch.bfloat16);  mul_5 = None

        # File: /tmp/scratch_space/better_benchmark/repros/models/genai/LayerNormBackward/full_graph_001.py:29 in forward, code: mul_8: "f32[1152000, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_2, mul);  convert_element_type_2 = mul = None
        mul_6: "f32[1152000, 512][512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type, mul_2);  convert_element_type = mul_2 = None

        # File: /tmp/scratch_space/better_benchmark/repros/models/genai/LayerNormBackward/full_graph_001.py:30 in forward, code: sum_4: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_8, [0]);  mul_8 = None
        sum_3: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_6, [0]);  mul_6 = None
        return (convert_element_type_2, sum_3)
