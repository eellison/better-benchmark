class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[1024, 3][3, 1]cuda:0", primals_2: "f32[1024][1]cuda:0", primals_3: "Sym(s28)", primals_4: "f32[s28, 3][3, 1]cuda:0", primals_5: "f32[1024, 1024][1024, 1]cuda:0", primals_6: "f32[1024][1]cuda:0", primals_7: "f32[2, 1024][1024, 1]cuda:0", primals_8: "f32[2][1]cuda:0"):
        # File: /torchbench/torchbenchmark/models/soft_actor_critic/nets.py:113 in forward, code: x = F.relu(self.fc1(state))
        convert_element_type: "f16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_2, torch.float16);  primals_2 = None
        convert_element_type_1: "f16[1024, 3][3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_1, torch.float16);  primals_1 = None
        convert_element_type_2: "f16[s28, 3][3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_4, torch.float16);  primals_4 = None
        permute: "f16[3, 1024][1, 3]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1, [1, 0]);  convert_element_type_1 = None
        addmm: "f16[s28, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type, convert_element_type_2, permute);  convert_element_type = permute = None
        relu: "f16[s28, 1024][1024, 1]cuda:0" = torch.ops.aten.relu.default(addmm);  addmm = None

        # File: /torchbench/torchbenchmark/models/soft_actor_critic/nets.py:114 in forward, code: x = F.relu(self.fc2(x))
        convert_element_type_6: "f16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_6, torch.float16);  primals_6 = None
        convert_element_type_7: "f16[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_5, torch.float16);  primals_5 = None
        permute_1: "f16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_7, [1, 0]);  convert_element_type_7 = None
        addmm_1: "f16[s28, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_6, relu, permute_1);  convert_element_type_6 = None
        relu_1: "f16[s28, 1024][1024, 1]cuda:0" = torch.ops.aten.relu.default(addmm_1);  addmm_1 = None

        # File: /torchbench/torchbenchmark/models/soft_actor_critic/nets.py:115 in forward, code: out = self.fc3(x)
        convert_element_type_11: "f16[2][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_8, torch.float16);  primals_8 = None
        convert_element_type_12: "f16[2, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_7, torch.float16);  primals_7 = None
        permute_2: "f16[1024, 2][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_12, [1, 0]);  convert_element_type_12 = None
        addmm_2: "f16[s28, 2][2, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_11, relu_1, permute_2);  convert_element_type_11 = None

        # File: /torchbench/torchbenchmark/models/soft_actor_critic/nets.py:116 in forward, code: mu, log_std = out.chunk(2, dim=1)
        split = torch.ops.aten.split.Tensor(addmm_2, 1, 1);  addmm_2 = None
        getitem: "f16[s28, 1][2, 1]cuda:0" = split[0]
        getitem_1: "f16[s28, 1][2, 1]cuda:0" = split[1];  split = None

        # File: /torchbench/torchbenchmark/models/soft_actor_critic/nets.py:118 in forward, code: log_std = torch.tanh(log_std)
        tanh: "f16[s28, 1][1, 1]cuda:0" = torch.ops.aten.tanh.default(getitem_1);  getitem_1 = None

        # File: /torchbench/torchbenchmark/models/soft_actor_critic/nets.py:121 in forward, code: ) * (log_std + 1)
        add_36: "f16[s28, 1][1, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh, 1)

        # File: /torchbench/torchbenchmark/models/soft_actor_critic/nets.py:119 in forward, code: log_std = self.log_std_low + 0.5 * (
        mul_23: "f16[s28, 1][1, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_36, 6.0);  add_36 = None
        add_43: "f16[s28, 1][1, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_23, -10.0);  mul_23 = None

        # File: /torchbench/torchbenchmark/models/soft_actor_critic/nets.py:122 in forward, code: std = log_std.exp()
        convert_element_type_16: "f32[s28, 1][1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_43, torch.float32);  add_43 = None
        exp: "f32[s28, 1][1, 1]cuda:0" = torch.ops.aten.exp.default(convert_element_type_16);  convert_element_type_16 = None

        # File: /torchbench/torchbenchmark/models/soft_actor_critic/nets.py:115 in forward, code: out = self.fc3(x)
        permute_3: "f16[2, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None

        # File: /torchbench/torchbenchmark/models/soft_actor_critic/nets.py:114 in forward, code: x = F.relu(self.fc2(x))
        permute_7: "f16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_1, [1, 0]);  permute_1 = None
        return (getitem, exp, primals_3, exp, getitem, convert_element_type_2, relu, relu_1, tanh, exp, permute_3, permute_7, primals_3)
