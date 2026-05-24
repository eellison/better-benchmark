import torch
from torch import device
from math import inf, nan

class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[1024, 3]", primals_2: "f32[1024]", primals_3: "f32[256, 3]", primals_4: "f32[1024, 1024]", primals_5: "f32[1024]", primals_6: "f32[2, 1024]", primals_7: "f32[2]"):
        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/soft_actor_critic/nets.py:113 in forward, code: x = F.relu(self.fc1(state))
        permute: "f32[3, 1024]" = torch.ops.aten.permute.default(primals_1, [1, 0]);  primals_1 = None
        addmm: "f32[256, 1024]" = torch.ops.aten.addmm.default(primals_2, primals_3, permute);  primals_2 = permute = None
        relu: "f32[256, 1024]" = torch.ops.aten.relu.default(addmm);  addmm = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/soft_actor_critic/nets.py:114 in forward, code: x = F.relu(self.fc2(x))
        permute_1: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_4, [1, 0])
        addmm_1: "f32[256, 1024]" = torch.ops.aten.addmm.default(primals_5, relu, permute_1);  primals_5 = permute_1 = None
        relu_1: "f32[256, 1024]" = torch.ops.aten.relu.default(addmm_1);  addmm_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/soft_actor_critic/nets.py:115 in forward, code: out = self.fc3(x)
        permute_2: "f32[1024, 2]" = torch.ops.aten.permute.default(primals_6, [1, 0])
        addmm_2: "f32[256, 2]" = torch.ops.aten.addmm.default(primals_7, relu_1, permute_2);  primals_7 = permute_2 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/soft_actor_critic/nets.py:116 in forward, code: mu, log_std = out.chunk(2, dim=1)
        split = torch.ops.aten.split.Tensor(addmm_2, 1, 1);  addmm_2 = None
        getitem: "f32[256, 1]" = split[0]
        getitem_1: "f32[256, 1]" = split[1];  split = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/soft_actor_critic/nets.py:118 in forward, code: log_std = torch.tanh(log_std)
        tanh: "f32[256, 1]" = torch.ops.aten.tanh.default(getitem_1);  getitem_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/soft_actor_critic/nets.py:121 in forward, code: ) * (log_std + 1)
        add: "f32[256, 1]" = torch.ops.aten.add.Tensor(tanh, 1)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/soft_actor_critic/nets.py:119 in forward, code: log_std = self.log_std_low + 0.5 * (
        mul: "f32[256, 1]" = torch.ops.aten.mul.Tensor(add, 6.0);  add = None
        add_1: "f32[256, 1]" = torch.ops.aten.add.Tensor(mul, -10.0);  mul = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/soft_actor_critic/nets.py:122 in forward, code: std = log_std.exp()
        exp: "f32[256, 1]" = torch.ops.aten.exp.default(add_1);  add_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/soft_actor_critic/nets.py:118 in forward, code: log_std = torch.tanh(log_std)
        mul_3: "f32[256, 1]" = torch.ops.aten.mul.Tensor(tanh, tanh);  tanh = None
        sub: "f32[256, 1]" = torch.ops.aten.sub.Tensor(1, mul_3);  mul_3 = None
        return (getitem, exp, exp, getitem, primals_3, primals_4, primals_6, relu, relu_1, exp, sub)
