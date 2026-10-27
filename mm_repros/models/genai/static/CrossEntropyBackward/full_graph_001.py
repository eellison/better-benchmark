class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "bf16[][]cuda:0", arg1_1: "i64[8192][1]cuda:0", arg2_1: "bf16[8192, 262144][262144, 1]cuda:0", arg3_1: "f32[8192, 1][1, 1]cuda:0", arg4_1: "f32[8192, 1][1, 1]cuda:0"):
        # File: /tmp/scratch_space/better_benchmark/repros/models/genai/CrossEntropyBackward/full_graph_001.py:31 in forward, code: convert_element_type: "f32[8192, 262144]" = torch.ops.prims.convert_element_type.default(primals_1, torch.float32);  primals_1 = None
        convert_element_type_1: "f32[8192, 262144][262144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(arg2_1, torch.float32);  arg2_1 = None

        # File: /tmp/scratch_space/better_benchmark/repros/models/genai/CrossEntropyBackward/full_graph_001.py:32 in forward, code: sub: "f32[8192, 262144]" = torch.ops.aten.sub.Tensor(convert_element_type, amax);  convert_element_type = amax = None
        sub: "f32[8192, 262144][262144, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_1, arg3_1);  convert_element_type_1 = arg3_1 = None

        # File: /tmp/scratch_space/better_benchmark/repros/models/genai/CrossEntropyBackward/full_graph_001.py:33 in forward, code: sub_1: "f32[8192, 262144]" = torch.ops.aten.sub.Tensor(sub, log);  sub = log = None
        sub_1: "f32[8192, 262144][262144, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub, arg4_1);  sub = arg4_1 = None

        # File: /tmp/scratch_space/better_benchmark/repros/models/genai/CrossEntropyBackward/full_graph_001.py:34 in forward, code: convert_element_type_1: "bf16[8192, 262144]" = torch.ops.prims.convert_element_type.default(sub_1, torch.bfloat16);  sub_1 = None
        convert_element_type_2: "bf16[8192, 262144][262144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(sub_1, torch.bfloat16);  sub_1 = None

        # File: /tmp/scratch_space/better_benchmark/repros/models/genai/CrossEntropyBackward/full_graph_001.py:35 in forward, code: convert_element_type_3: "f32[8192, 262144]" = torch.ops.prims.convert_element_type.default(convert_element_type_1, torch.float32);  convert_element_type_1 = None
        convert_element_type_3: "f32[8192, 262144][262144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2, torch.float32);  convert_element_type_2 = None

        # File: /tmp/scratch_space/better_benchmark/repros/models/genai/CrossEntropyBackward/full_graph_001.py:36 in forward, code: exp_1: "f32[8192, 262144]" = torch.ops.aten.exp.default(convert_element_type_3);  convert_element_type_3 = None
        exp: "f32[8192, 262144][262144, 1]cuda:0" = torch.ops.aten.exp.default(convert_element_type_3);  convert_element_type_3 = None

        # File: /tmp/scratch_space/better_benchmark/repros/models/genai/CrossEntropyBackward/full_graph_001.py:11 in forward, code: unsqueeze_1: "i64[8192, 1]" = torch.ops.aten.unsqueeze.default(primals_2, 1);  primals_2 = None
        unsqueeze: "i64[8192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg1_1, 1);  arg1_1 = None

        # File: /tmp/scratch_space/better_benchmark/repros/models/genai/CrossEntropyBackward/full_graph_001.py:12 in forward, code: ne_2: "b8[8192, 1]" = torch.ops.aten.ne.Scalar(unsqueeze_1, -100)
        ne: "b8[8192, 1][1, 1]cuda:0" = torch.ops.aten.ne.Scalar(unsqueeze, -100)

        # File: /tmp/scratch_space/better_benchmark/repros/models/genai/CrossEntropyBackward/full_graph_001.py:13 in forward, code: full_default: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default: "i64[][]cuda:0" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /tmp/scratch_space/better_benchmark/repros/models/genai/CrossEntropyBackward/full_graph_001.py:14 in forward, code: where_2: "i64[8192, 1]" = torch.ops.aten.where.self(ne_2, unsqueeze_1, full_default);  unsqueeze_1 = full_default = None
        where: "i64[8192, 1][1, 1]cuda:0" = torch.ops.aten.where.self(ne, unsqueeze, full_default);  unsqueeze = full_default = None

        # File: /tmp/scratch_space/better_benchmark/repros/models/genai/CrossEntropyBackward/full_graph_001.py:19 in forward, code: expand_default: "i64[8192, 262144]" = torch.ops.aten.expand.default(where_2, [8192, 262144]);  where_2 = None
        expand_1: "i64[8192, 262144][1, 0]cuda:0" = torch.ops.aten.expand.default(where, [8192, 262144]);  where = None

        # File: /tmp/scratch_space/better_benchmark/repros/models/genai/CrossEntropyBackward/full_graph_001.py:17 in forward, code: iota_default: "i64[262144]" = torch.ops.prims.iota.default(262144, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        iota: "i64[262144][1]cuda:0" = torch.ops.prims.iota.default(262144, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)

        # File: /tmp/scratch_space/better_benchmark/repros/models/genai/CrossEntropyBackward/full_graph_001.py:18 in forward, code: view_default: "i64[1, 262144]" = torch.ops.aten.reshape.default(iota_default, [1, 262144]);  iota_default = None
        view: "i64[1, 262144][262144, 1]cuda:0" = torch.ops.aten.reshape.default(iota, [1, 262144]);  iota = None

        # File: /tmp/scratch_space/better_benchmark/repros/models/genai/CrossEntropyBackward/full_graph_001.py:20 in forward, code: eq_tensor: "b8[8192, 262144]" = torch.ops.aten.eq.Tensor(expand_default, view_default);  expand_default = view_default = None
        eq: "b8[8192, 262144][262144, 1]cuda:0" = torch.ops.aten.eq.Tensor(expand_1, view);  expand_1 = view = None

        # File: /tmp/scratch_space/better_benchmark/repros/models/genai/CrossEntropyBackward/full_graph_001.py:22 in forward, code: scalar_tensor_default_1: "f32[]" = torch.ops.aten.scalar_tensor.default(-1.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
        full_default_2: "f32[][]cuda:0" = torch.ops.aten.full.default([], -1.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /tmp/scratch_space/better_benchmark/repros/models/genai/CrossEntropyBackward/full_graph_001.py:21 in forward, code: scalar_tensor_default: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
        full_default_1: "f32[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /tmp/scratch_space/better_benchmark/repros/models/genai/CrossEntropyBackward/full_graph_001.py:25 in forward, code: where_self: "f32[8192, 262144]" = torch.ops.aten.where.self(eq_tensor, scalar_tensor_default_1, scalar_tensor_default);  eq_tensor = scalar_tensor_default_1 = scalar_tensor_default = None
        where_1: "f32[8192, 262144][262144, 1]cuda:0" = torch.ops.aten.where.self(eq, full_default_2, full_default_1);  eq = full_default_2 = full_default_1 = None

        # File: /tmp/scratch_space/better_benchmark/repros/models/genai/CrossEntropyBackward/full_graph_001.py:8 in forward, code: expand: "bf16[8192]" = torch.ops.aten.expand.default(tangents_1, [8192]);  tangents_1 = None
        expand: "bf16[8192][0]cuda:0" = torch.ops.aten.expand.default(arg0_1, [8192]);  arg0_1 = None

        # File: /tmp/scratch_space/better_benchmark/repros/models/genai/CrossEntropyBackward/full_graph_001.py:26 in forward, code: unsqueeze_2: "bf16[8192, 1]" = torch.ops.aten.unsqueeze.default(expand, 1);  expand = None
        unsqueeze_1: "bf16[8192, 1][0, 1]cuda:0" = torch.ops.aten.unsqueeze.default(expand, 1);  expand = None

        # File: /tmp/scratch_space/better_benchmark/repros/models/genai/CrossEntropyBackward/full_graph_001.py:27 in forward, code: full_default_1: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_3: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /tmp/scratch_space/better_benchmark/repros/models/genai/CrossEntropyBackward/full_graph_001.py:28 in forward, code: where_3: "bf16[8192, 1]" = torch.ops.aten.where.self(ne_2, unsqueeze_2, full_default_1);  ne_2 = unsqueeze_2 = full_default_1 = None
        where_2: "bf16[8192, 1][1, 1]cuda:0" = torch.ops.aten.where.self(ne, unsqueeze_1, full_default_3);  ne = unsqueeze_1 = full_default_3 = None

        # File: /tmp/scratch_space/better_benchmark/repros/models/genai/CrossEntropyBackward/full_graph_001.py:29 in forward, code: mul: "f32[8192, 262144]" = torch.ops.aten.mul.Tensor(where_self, where_3);  where_self = where_3 = None
        mul: "f32[8192, 262144][262144, 1]cuda:0" = torch.ops.aten.mul.Tensor(where_1, where_2);  where_1 = where_2 = None

        # File: /tmp/scratch_space/better_benchmark/repros/models/genai/CrossEntropyBackward/full_graph_001.py:37 in forward, code: sum_3: "f32[8192, 1]" = torch.ops.aten.sum.dim_IntList(convert_element_type_2, [1], True)
        sum_1: "f32[8192, 1][1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul, [1], True)

        # File: /tmp/scratch_space/better_benchmark/repros/models/genai/CrossEntropyBackward/full_graph_001.py:38 in forward, code: mul_1: "f32[8192, 262144]" = torch.ops.aten.mul.Tensor(exp_1, sum_3);  exp_1 = sum_3 = None
        mul_1: "f32[8192, 262144][262144, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp, sum_1);  exp = sum_1 = None

        # File: /tmp/scratch_space/better_benchmark/repros/models/genai/CrossEntropyBackward/full_graph_001.py:39 in forward, code: sub_2: "f32[8192, 262144]" = torch.ops.aten.sub.Tensor(convert_element_type_2, mul_1);  convert_element_type_2 = mul_1 = None
        sub_2: "f32[8192, 262144][262144, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul, mul_1);  mul = mul_1 = None

        # File: /tmp/scratch_space/better_benchmark/repros/models/genai/CrossEntropyBackward/full_graph_001.py:40 in forward, code: convert_element_type_4: "bf16[8192, 262144]" = torch.ops.prims.convert_element_type.default(sub_2, torch.bfloat16);  sub_2 = None
        convert_element_type_4: "bf16[8192, 262144][262144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(sub_2, torch.bfloat16);  sub_2 = None
        return (convert_element_type_4,)
