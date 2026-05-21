class GraphModule(torch.nn.Module):
    def forward(self, primals_3: "Sym(s28)", convert_element_type_2: "f16[s28, 3][3, 1]cuda:0", relu: "f16[s28, 1024][1024, 1]cuda:0", relu_1: "f16[s28, 1024][1024, 1]cuda:0", tanh: "f16[s28, 1][1, 1]cuda:0", exp: "f32[s28, 1][1, 1]cuda:0", permute_3: "f16[2, 1024][1024, 1]cuda:0", permute_7: "f16[1024, 1024][1024, 1]cuda:0", tangents_1: "f16[s28, 1][1, 1]cuda:0", tangents_2: "f32[s28, 1][1, 1]cuda:0", tangents_3: "f32[s28, 1][1, 1]cuda:0", tangents_4: "f16[s28, 1][1, 1]cuda:0"):
        # No stacktrace found for following nodes
        add_56: "f32[s28, 1][1, 1]cuda:0" = torch.ops.aten.add.Tensor(tangents_2, tangents_3);  tangents_2 = tangents_3 = None
        add_57: "f16[s28, 1][1, 1]cuda:0" = torch.ops.aten.add.Tensor(tangents_1, tangents_4);  tangents_1 = tangents_4 = None

        # File: /torchbench/torchbenchmark/models/soft_actor_critic/nets.py:122 in forward, code: std = log_std.exp()
        mul_29: "f32[s28, 1][1, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_56, exp);  add_56 = exp = None
        convert_element_type_17: "f16[s28, 1][1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_29, torch.float16);  mul_29 = None

        # File: /torchbench/torchbenchmark/models/soft_actor_critic/nets.py:119 in forward, code: log_std = self.log_std_low + 0.5 * (
        mul_30: "f16[s28, 1][1, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_17, 6.0);  convert_element_type_17 = None

        # File: /torchbench/torchbenchmark/models/soft_actor_critic/nets.py:118 in forward, code: log_std = torch.tanh(log_std)
        convert_element_type_18: "f32[s28, 1][1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_30, torch.float32);  mul_30 = None
        convert_element_type_19: "f32[s28, 1][1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(tanh, torch.float32);  tanh = None
        mul_31: "f32[s28, 1][1, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_19, convert_element_type_19);  convert_element_type_19 = None
        sub_18: "f32[s28, 1][1, 1]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_31);  mul_31 = None
        mul_32: "f32[s28, 1][1, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_18, sub_18);  convert_element_type_18 = sub_18 = None
        convert_element_type_20: "f16[s28, 1][1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_32, torch.float16);  mul_32 = None

        # File: /torchbench/torchbenchmark/models/soft_actor_critic/nets.py:116 in forward, code: mu, log_std = out.chunk(2, dim=1)
        cat: "f16[s28, 2][2, 1]cuda:0" = torch.ops.aten.cat.default([add_57, convert_element_type_20], 1);  add_57 = convert_element_type_20 = None

        # File: /torchbench/torchbenchmark/models/soft_actor_critic/nets.py:115 in forward, code: out = self.fc3(x)
        mm: "f16[s28, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(cat, permute_3);  permute_3 = None
        permute_4: "f16[2, s28][1, 2]cuda:0" = torch.ops.aten.permute.default(cat, [1, 0])
        mm_1: "f16[2, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_4, relu_1);  permute_4 = None
        sum_1: "f16[1, 2][2, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(cat, [0], True);  cat = None
        view: "f16[2][1]cuda:0" = torch.ops.aten.reshape.default(sum_1, [2]);  sum_1 = None
        convert_element_type_25: "f32[2, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_1, torch.float32);  mm_1 = None
        convert_element_type_26: "f32[2][1]cuda:0" = torch.ops.prims.convert_element_type.default(view, torch.float32);  view = None

        # File: /torchbench/torchbenchmark/models/soft_actor_critic/nets.py:114 in forward, code: x = F.relu(self.fc2(x))
        le: "b8[s28, 1024][1024, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_1, 0);  relu_1 = None
        full_default: "f16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "f16[s28, 1024][1024, 1]cuda:0" = torch.ops.aten.where.self(le, full_default, mm);  le = mm = None
        mm_2: "f16[s28, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(where, permute_7);  permute_7 = None
        permute_8: "f16[1024, s28][1, 1024]cuda:0" = torch.ops.aten.permute.default(where, [1, 0])
        mm_3: "f16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_8, relu);  permute_8 = None
        sum_2: "f16[1, 1024][1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(where, [0], True);  where = None
        view_1: "f16[1024][1]cuda:0" = torch.ops.aten.reshape.default(sum_2, [1024]);  sum_2 = None
        convert_element_type_31: "f32[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_3, torch.float32);  mm_3 = None
        convert_element_type_32: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1, torch.float32);  view_1 = None

        # File: /torchbench/torchbenchmark/models/soft_actor_critic/nets.py:113 in forward, code: x = F.relu(self.fc1(state))
        le_1: "b8[s28, 1024][1024, 1]cuda:0" = torch.ops.aten.le.Scalar(relu, 0);  relu = None
        where_1: "f16[s28, 1024][1024, 1]cuda:0" = torch.ops.aten.where.self(le_1, full_default, mm_2);  le_1 = full_default = mm_2 = None
        permute_11: "f16[1024, s28][1, 1024]cuda:0" = torch.ops.aten.permute.default(where_1, [1, 0])
        mm_4: "f16[1024, 3][3, 1]cuda:0" = torch.ops.aten.mm.default(permute_11, convert_element_type_2);  permute_11 = convert_element_type_2 = None
        sum_3: "f16[1, 1024][1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(where_1, [0], True);  where_1 = None
        view_2: "f16[1024][1]cuda:0" = torch.ops.aten.reshape.default(sum_3, [1024]);  sum_3 = None
        convert_element_type_35: "f32[1024, 3][3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_4, torch.float32);  mm_4 = None
        convert_element_type_36: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_2, torch.float32);  view_2 = None
        return (convert_element_type_35, convert_element_type_36, None, None, convert_element_type_31, convert_element_type_32, convert_element_type_25, convert_element_type_26)
