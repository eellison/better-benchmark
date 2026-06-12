class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "bf16[][]cuda:0", arg1_1: "bf16[1152000, 512][512, 1]cuda:0", arg2_1: "f32[1152000, 1][1, 1]cuda:0", arg3_1: "f32[512][1]cuda:0"):
        # File: /tmp/scratch_space/better_benchmark/repros/models/genai/RMSNormBackward/full_graph_001.py:8 in forward, code: expand: "bf16[1152000, 512]" = torch.ops.aten.expand.default(tangents_1, [1152000, 512]);  tangents_1 = None
        expand: "bf16[1152000, 512][0, 0]cuda:0" = torch.ops.aten.expand.default(arg0_1, [1152000, 512]);  arg0_1 = None

        # File: /tmp/scratch_space/better_benchmark/repros/models/genai/RMSNormBackward/full_graph_001.py:11 in forward, code: convert_element_type_2: "f32[1152000, 512]" = torch.ops.prims.convert_element_type.default(expand, torch.float32);  expand = None
        convert_element_type: "f32[1152000, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(expand, torch.float32);  expand = None

        # File: /tmp/scratch_space/better_benchmark/repros/models/genai/RMSNormBackward/full_graph_001.py:19 in forward, code: mul_3: "f32[1152000, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_2, primals_2);  convert_element_type_2 = primals_2 = None
        mul_2: "f32[1152000, 512][512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type, arg3_1);  arg3_1 = None

        # File: /tmp/scratch_space/better_benchmark/repros/models/genai/RMSNormBackward/full_graph_001.py:23 in forward, code: mul_5: "f32[1152000, 512]" = torch.ops.aten.mul.Tensor(mul_3, rsqrt);  mul_3 = None
        mul_4: "f32[1152000, 512][512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2, arg2_1)

        # File: /tmp/scratch_space/better_benchmark/repros/models/genai/RMSNormBackward/full_graph_001.py:14 in forward, code: convert_element_type: "f32[1152000, 512]" = torch.ops.prims.convert_element_type.default(primals_1, torch.float32);  primals_1 = None
        convert_element_type_1: "f32[1152000, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(arg1_1, torch.float32);  arg1_1 = None

        # File: /tmp/scratch_space/better_benchmark/repros/models/genai/RMSNormBackward/full_graph_001.py:22 in forward, code: mul_4: "f32[1152000, 512]" = torch.ops.aten.mul.Tensor(mul_3, convert_element_type)
        mul_3: "f32[1152000, 512][512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2, convert_element_type_1);  mul_2 = None

        # File: /tmp/scratch_space/better_benchmark/repros/models/genai/RMSNormBackward/full_graph_001.py:24 in forward, code: sum_3: "f32[1152000, 1]" = torch.ops.aten.sum.dim_IntList(mul_4, [1], True);  mul_4 = None
        sum_2: "f32[1152000, 1][1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_3, [1], True);  mul_3 = None

        # File: /tmp/scratch_space/better_benchmark/repros/models/genai/RMSNormBackward/full_graph_001.py:28 in forward, code: mul_6: "f32[1152000, 1]" = torch.ops.aten.mul.Scalar(sum_3, -0.5);  sum_3 = None
        mul_5: "f32[1152000, 1][1, 1]cuda:0" = torch.ops.aten.mul.Scalar(sum_2, -0.5);  sum_2 = None

        # File: /tmp/scratch_space/better_benchmark/repros/models/genai/RMSNormBackward/full_graph_001.py:27 in forward, code: pow_2: "f32[1152000, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt, 3);  rsqrt = None
        pow_1: "f32[1152000, 1][1, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(arg2_1, 3)

        # File: /tmp/scratch_space/better_benchmark/repros/models/genai/RMSNormBackward/full_graph_001.py:29 in forward, code: mul_7: "f32[1152000, 1]" = torch.ops.aten.mul.Tensor(mul_6, pow_2);  mul_6 = pow_2 = None
        mul_6: "f32[1152000, 1][1, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_5, pow_1);  mul_5 = pow_1 = None

        # File: /tmp/scratch_space/better_benchmark/repros/models/genai/RMSNormBackward/full_graph_001.py:30 in forward, code: expand_1: "f32[1152000, 512]" = torch.ops.aten.expand.default(mul_7, [1152000, 512]);  mul_7 = None
        expand_1: "f32[1152000, 512][1, 0]cuda:0" = torch.ops.aten.expand.default(mul_6, [1152000, 512]);  mul_6 = None

        # File: /tmp/scratch_space/better_benchmark/repros/models/genai/RMSNormBackward/full_graph_001.py:31 in forward, code: div: "f32[1152000, 512]" = torch.ops.aten.div.Scalar(expand_1, 512);  expand_1 = None
        div: "f32[1152000, 512][512, 1]cuda:0" = torch.ops.aten.div.Scalar(expand_1, 512);  expand_1 = None

        # File: /tmp/scratch_space/better_benchmark/repros/models/genai/RMSNormBackward/full_graph_001.py:32 in forward, code: pow_3: "f32[1152000, 512]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type, 1.0);  convert_element_type = None
        pow_2: "f32[1152000, 512][512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_1, 1.0)

        # File: /tmp/scratch_space/better_benchmark/repros/models/genai/RMSNormBackward/full_graph_001.py:33 in forward, code: mul_8: "f32[1152000, 512]" = torch.ops.aten.mul.Scalar(pow_3, 2.0);  pow_3 = None
        mul_7: "f32[1152000, 512][512, 1]cuda:0" = torch.ops.aten.mul.Scalar(pow_2, 2.0);  pow_2 = None

        # File: /tmp/scratch_space/better_benchmark/repros/models/genai/RMSNormBackward/full_graph_001.py:34 in forward, code: mul_9: "f32[1152000, 512]" = torch.ops.aten.mul.Tensor(div, mul_8);  div = mul_8 = None
        mul_8: "f32[1152000, 512][512, 1]cuda:0" = torch.ops.aten.mul.Tensor(div, mul_7);  div = mul_7 = None

        # File: /tmp/scratch_space/better_benchmark/repros/models/genai/RMSNormBackward/full_graph_001.py:35 in forward, code: add_1: "f32[1152000, 512]" = torch.ops.aten.add.Tensor(mul_5, mul_9);  mul_5 = mul_9 = None
        add: "f32[1152000, 512][512, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_4, mul_8);  mul_4 = mul_8 = None

        # File: /tmp/scratch_space/better_benchmark/repros/models/genai/RMSNormBackward/full_graph_001.py:38 in forward, code: convert_element_type_3: "bf16[1152000, 512]" = torch.ops.prims.convert_element_type.default(add_1, torch.bfloat16);  add_1 = None
        convert_element_type_2: "bf16[1152000, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add, torch.bfloat16);  add = None

        # File: /tmp/scratch_space/better_benchmark/repros/models/genai/RMSNormBackward/full_graph_001.py:17 in forward, code: mul: "f32[1152000, 512]" = torch.ops.aten.mul.Tensor(convert_element_type, rsqrt)
        mul: "f32[1152000, 512][512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1, arg2_1);  convert_element_type_1 = arg2_1 = None

        # File: /tmp/scratch_space/better_benchmark/repros/models/genai/RMSNormBackward/full_graph_001.py:18 in forward, code: mul_2: "f32[1152000, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_2, mul);  mul = None
        mul_1: "f32[1152000, 512][512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type, mul);  convert_element_type = mul = None

        # File: /tmp/scratch_space/better_benchmark/repros/models/genai/RMSNormBackward/full_graph_001.py:20 in forward, code: sum_2: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(mul_2, [0], True);  mul_2 = None
        sum_1: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1, [0], True);  mul_1 = None

        # File: /tmp/scratch_space/better_benchmark/repros/models/genai/RMSNormBackward/full_graph_001.py:21 in forward, code: view: "f32[512]" = torch.ops.aten.reshape.default(sum_2, [512]);  sum_2 = None
        view: "f32[512][1]cuda:0" = torch.ops.aten.reshape.default(sum_1, [512]);  sum_1 = None
        return (convert_element_type_2, view)
