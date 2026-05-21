class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "f32[1024, 3]", arg1_1: "f32[1024]", arg2_1: "f32[256, 3]", arg3_1: "f32[1024, 1024]", arg4_1: "f32[1024]", arg5_1: "f32[2, 1024]", arg6_1: "f32[2]"):
        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/soft_actor_critic/nets.py:113 in forward, code: x = F.relu(self.fc1(state))
        permute: "f32[3, 1024]" = torch.ops.aten.permute.default(arg0_1, [1, 0]);  arg0_1 = None
        addmm: "f32[256, 1024]" = torch.ops.aten.addmm.default(arg1_1, arg2_1, permute);  arg1_1 = arg2_1 = permute = None
        relu: "f32[256, 1024]" = torch.ops.aten.relu.default(addmm);  addmm = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/soft_actor_critic/nets.py:114 in forward, code: x = F.relu(self.fc2(x))
        permute_1: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg3_1, [1, 0]);  arg3_1 = None
        addmm_1: "f32[256, 1024]" = torch.ops.aten.addmm.default(arg4_1, relu, permute_1);  arg4_1 = relu = permute_1 = None
        relu_1: "f32[256, 1024]" = torch.ops.aten.relu.default(addmm_1);  addmm_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/soft_actor_critic/nets.py:115 in forward, code: out = self.fc3(x)
        permute_2: "f32[1024, 2]" = torch.ops.aten.permute.default(arg5_1, [1, 0]);  arg5_1 = None
        addmm_2: "f32[256, 2]" = torch.ops.aten.addmm.default(arg6_1, relu_1, permute_2);  arg6_1 = relu_1 = permute_2 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/soft_actor_critic/nets.py:116 in forward, code: mu, log_std = out.chunk(2, dim=1)
        split = torch.ops.aten.split.Tensor(addmm_2, 1, 1);  addmm_2 = None
        getitem: "f32[256, 1]" = split[0]
        getitem_1: "f32[256, 1]" = split[1];  split = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/soft_actor_critic/nets.py:118 in forward, code: log_std = torch.tanh(log_std)
        tanh: "f32[256, 1]" = torch.ops.aten.tanh.default(getitem_1);  getitem_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/soft_actor_critic/nets.py:121 in forward, code: ) * (log_std + 1)
        add: "f32[256, 1]" = torch.ops.aten.add.Tensor(tanh, 1);  tanh = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/soft_actor_critic/nets.py:119 in forward, code: log_std = self.log_std_low + 0.5 * (
        mul: "f32[256, 1]" = torch.ops.aten.mul.Tensor(add, 6.0);  add = None
        add_1: "f32[256, 1]" = torch.ops.aten.add.Tensor(mul, -10.0);  mul = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/soft_actor_critic/nets.py:122 in forward, code: std = log_std.exp()
        exp: "f32[256, 1]" = torch.ops.aten.exp.default(add_1);  add_1 = None
        return (getitem, exp, exp, getitem)
