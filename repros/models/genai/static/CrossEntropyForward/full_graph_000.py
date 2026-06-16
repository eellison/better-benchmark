class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "i64[8192][1]cuda:0", arg1_1: "bf16[8192, 262144][262144, 1]cuda:0"):
        # File: /tmp/scratch_space/better_benchmark/repros/models/genai/CrossEntropyForward/full_graph_000.py:8 in forward, code: ne_1: "b8[8192]" = torch.ops.aten.ne.Scalar(arg1_1, -100)
        ne: "b8[8192][1]cuda:0" = torch.ops.aten.ne.Scalar(arg0_1, -100)

        # File: /tmp/scratch_space/better_benchmark/repros/models/genai/CrossEntropyForward/full_graph_000.py:9 in forward, code: convert_element_type: "f32[8192, 262144]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.float32);  arg0_1 = None
        convert_element_type: "f32[8192, 262144][262144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(arg1_1, torch.float32);  arg1_1 = None

        # File: /tmp/scratch_space/better_benchmark/repros/models/genai/CrossEntropyForward/full_graph_000.py:10 in forward, code: amax: "f32[8192, 1]" = torch.ops.aten.amax.default(convert_element_type, [1], True)
        amax: "f32[8192, 1][1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type, [1], True)

        # File: /tmp/scratch_space/better_benchmark/repros/models/genai/CrossEntropyForward/full_graph_000.py:11 in forward, code: sub: "f32[8192, 262144]" = torch.ops.aten.sub.Tensor(convert_element_type, amax);  convert_element_type = amax = None
        sub: "f32[8192, 262144][262144, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type, amax);  convert_element_type = amax = None

        # File: /tmp/scratch_space/better_benchmark/repros/models/genai/CrossEntropyForward/full_graph_000.py:12 in forward, code: exp: "f32[8192, 262144]" = torch.ops.aten.exp.default(sub)
        exp: "f32[8192, 262144][262144, 1]cuda:0" = torch.ops.aten.exp.default(sub)

        # File: /tmp/scratch_space/better_benchmark/repros/models/genai/CrossEntropyForward/full_graph_000.py:13 in forward, code: sum_1: "f32[8192, 1]" = torch.ops.aten.sum.dim_IntList(exp, [1], True);  exp = None
        sum_1: "f32[8192, 1][1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp, [1], True);  exp = None

        # File: /tmp/scratch_space/better_benchmark/repros/models/genai/CrossEntropyForward/full_graph_000.py:14 in forward, code: log: "f32[8192, 1]" = torch.ops.aten.log.default(sum_1);  sum_1 = None
        log: "f32[8192, 1][1, 1]cuda:0" = torch.ops.aten.log.default(sum_1);  sum_1 = None

        # File: /tmp/scratch_space/better_benchmark/repros/models/genai/CrossEntropyForward/full_graph_000.py:15 in forward, code: sub_1: "f32[8192, 262144]" = torch.ops.aten.sub.Tensor(sub, log);  sub = log = None
        sub_1: "f32[8192, 262144][262144, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub, log);  sub = log = None

        # File: /tmp/scratch_space/better_benchmark/repros/models/genai/CrossEntropyForward/full_graph_000.py:16 in forward, code: convert_element_type_1: "bf16[8192, 262144]" = torch.ops.prims.convert_element_type.default(sub_1, torch.bfloat16);  sub_1 = None
        convert_element_type_1: "bf16[8192, 262144][262144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(sub_1, torch.bfloat16);  sub_1 = None

        # File: /tmp/scratch_space/better_benchmark/repros/models/genai/CrossEntropyForward/full_graph_000.py:17 in forward, code: ne: "b8[8192]" = torch.ops.aten.ne.Scalar(arg1_1, -100)
        ne_1: "b8[8192][1]cuda:0" = torch.ops.aten.ne.Scalar(arg0_1, -100)

        # File: /tmp/scratch_space/better_benchmark/repros/models/genai/CrossEntropyForward/full_graph_000.py:18 in forward, code: full_default: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default: "i64[][]cuda:0" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /tmp/scratch_space/better_benchmark/repros/models/genai/CrossEntropyForward/full_graph_000.py:19 in forward, code: where: "i64[8192]" = torch.ops.aten.where.self(ne, arg1_1, full_default);  ne = arg1_1 = full_default = None
        where: "i64[8192][1]cuda:0" = torch.ops.aten.where.self(ne_1, arg0_1, full_default);  ne_1 = arg0_1 = full_default = None

        # File: /tmp/scratch_space/better_benchmark/repros/models/genai/CrossEntropyForward/full_graph_000.py:20 in forward, code: unsqueeze: "i64[8192, 1]" = torch.ops.aten.unsqueeze.default(where, 1);  where = None
        unsqueeze: "i64[8192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(where, 1);  where = None

        # File: /tmp/scratch_space/better_benchmark/repros/models/genai/CrossEntropyForward/full_graph_000.py:21 in forward, code: gather: "bf16[8192, 1]" = torch.ops.aten.gather.default(convert_element_type_1, 1, unsqueeze);  convert_element_type_1 = unsqueeze = None
        gather: "bf16[8192, 1][1, 1]cuda:0" = torch.ops.aten.gather.default(convert_element_type_1, 1, unsqueeze);  convert_element_type_1 = unsqueeze = None

        # File: /tmp/scratch_space/better_benchmark/repros/models/genai/CrossEntropyForward/full_graph_000.py:22 in forward, code: squeeze: "bf16[8192]" = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
        squeeze: "bf16[8192][1]cuda:0" = torch.ops.aten.squeeze.dim(gather, 1);  gather = None

        # File: /tmp/scratch_space/better_benchmark/repros/models/genai/CrossEntropyForward/full_graph_000.py:23 in forward, code: neg: "bf16[8192]" = torch.ops.aten.neg.default(squeeze);  squeeze = None
        neg: "bf16[8192][1]cuda:0" = torch.ops.aten.neg.default(squeeze);  squeeze = None

        # File: /tmp/scratch_space/better_benchmark/repros/models/genai/CrossEntropyForward/full_graph_000.py:24 in forward, code: full_default_1: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_1: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /tmp/scratch_space/better_benchmark/repros/models/genai/CrossEntropyForward/full_graph_000.py:25 in forward, code: where_1: "bf16[8192]" = torch.ops.aten.where.self(ne_1, neg, full_default_1);  ne_1 = neg = full_default_1 = None
        where_1: "bf16[8192][1]cuda:0" = torch.ops.aten.where.self(ne, neg, full_default_1);  ne = neg = full_default_1 = None
        return (where_1,)
