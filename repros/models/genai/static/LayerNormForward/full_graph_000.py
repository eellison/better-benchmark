class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "bf16[1152000, 512][512, 1]cuda:0", arg1_1: "f32[512][1]cuda:0"):
        # File: /tmp/scratch_space/better_benchmark/repros/models/genai/LayerNormForward/full_graph_000.py:8 in forward, code: convert_element_type: "f32[1152000, 512]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.float32);  arg0_1 = None
        convert_element_type: "f32[1152000, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(arg0_1, torch.float32);  arg0_1 = None

        # File: /tmp/scratch_space/better_benchmark/repros/models/genai/LayerNormForward/full_graph_000.py:11 in forward, code: var_mean = torch.ops.aten.var_mean.correction(convert_element_type, [1], correction = 0, keepdim = True)
        var_mean = torch.ops.aten.var_mean.correction(convert_element_type, [1], correction = 0, keepdim = True)
        getitem: "f32[1152000, 1][1, 1]cuda:0" = var_mean[0]
        getitem_1: "f32[1152000, 1][1, 1]cuda:0" = var_mean[1];  var_mean = None

        # File: /tmp/scratch_space/better_benchmark/repros/models/genai/LayerNormForward/full_graph_000.py:14 in forward, code: sub: "f32[1152000, 512]" = torch.ops.aten.sub.Tensor(convert_element_type, getitem_1);  convert_element_type = getitem_1 = None
        sub: "f32[1152000, 512][512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type, getitem_1);  convert_element_type = getitem_1 = None

        # File: /tmp/scratch_space/better_benchmark/repros/models/genai/LayerNormForward/full_graph_000.py:15 in forward, code: add: "f32[1152000, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-06);  getitem = None
        add: "f32[1152000, 1][1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem, 1e-06);  getitem = None

        # File: /tmp/scratch_space/better_benchmark/repros/models/genai/LayerNormForward/full_graph_000.py:16 in forward, code: rsqrt: "f32[1152000, 1]" = torch.ops.aten.rsqrt.default(add);  add = None
        rsqrt: "f32[1152000, 1][1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add);  add = None

        # File: /tmp/scratch_space/better_benchmark/repros/models/genai/LayerNormForward/full_graph_000.py:17 in forward, code: mul: "f32[1152000, 512]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
        mul: "f32[1152000, 512][512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None

        # File: /tmp/scratch_space/better_benchmark/repros/models/genai/LayerNormForward/full_graph_000.py:18 in forward, code: mul_1: "f32[1152000, 512]" = torch.ops.aten.mul.Tensor(mul, arg1_1);  mul = arg1_1 = None
        mul_1: "f32[1152000, 512][512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul, arg1_1);  mul = arg1_1 = None

        # File: /tmp/scratch_space/better_benchmark/repros/models/genai/LayerNormForward/full_graph_000.py:19 in forward, code: convert_element_type_1: "bf16[1152000, 512]" = torch.ops.prims.convert_element_type.default(mul_1, torch.bfloat16);  mul_1 = None
        convert_element_type_1: "bf16[1152000, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1, torch.bfloat16);  mul_1 = None
        return (convert_element_type_1,)
