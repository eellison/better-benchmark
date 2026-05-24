import torch
from torch import device
from math import inf, nan

class GraphModule(torch.nn.Module):
    def forward(self, primals_3: "f32[256, 3]", primals_4: "f32[1024, 1024]", primals_6: "f32[2, 1024]", relu: "f32[256, 1024]", relu_1: "f32[256, 1024]", exp: "f32[256, 1]", sub: "f32[256, 1]", tangents_1: "f32[256, 1]", tangents_2: "f32[256, 1]", tangents_3: "f32[256, 1]", tangents_4: "f32[256, 1]"):
        # No stacktrace found for following nodes
        add_2: "f32[256, 1]" = torch.ops.aten.add.Tensor(tangents_2, tangents_3);  tangents_2 = tangents_3 = None
        add_3: "f32[256, 1]" = torch.ops.aten.add.Tensor(tangents_1, tangents_4);  tangents_1 = tangents_4 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/soft_actor_critic/nets.py:122 in forward, code: std = log_std.exp()
        mul_1: "f32[256, 1]" = torch.ops.aten.mul.Tensor(add_2, exp);  add_2 = exp = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/soft_actor_critic/nets.py:119 in forward, code: log_std = self.log_std_low + 0.5 * (
        mul_2: "f32[256, 1]" = torch.ops.aten.mul.Tensor(mul_1, 6.0);  mul_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/soft_actor_critic/nets.py:118 in forward, code: log_std = torch.tanh(log_std)
        mul_4: "f32[256, 1]" = torch.ops.aten.mul.Tensor(mul_2, sub);  mul_2 = sub = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/soft_actor_critic/nets.py:116 in forward, code: mu, log_std = out.chunk(2, dim=1)
        cat: "f32[256, 2]" = torch.ops.aten.cat.default([add_3, mul_4], 1);  add_3 = mul_4 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/soft_actor_critic/nets.py:115 in forward, code: out = self.fc3(x)
        permute_2: "f32[1024, 2]" = torch.ops.aten.permute.default(primals_6, [1, 0]);  primals_6 = None
        permute_3: "f32[2, 1024]" = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None
        mm: "f32[256, 1024]" = torch.ops.aten.mm.default(cat, permute_3);  permute_3 = None
        permute_4: "f32[2, 256]" = torch.ops.aten.permute.default(cat, [1, 0])
        mm_1: "f32[2, 1024]" = torch.ops.aten.mm.default(permute_4, relu_1);  permute_4 = None
        sum_1: "f32[1, 2]" = torch.ops.aten.sum.dim_IntList(cat, [0], True);  cat = None
        view: "f32[2]" = torch.ops.aten.reshape.default(sum_1, [2]);  sum_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/soft_actor_critic/nets.py:114 in forward, code: x = F.relu(self.fc2(x))
        le: "b8[256, 1024]" = torch.ops.aten.le.Scalar(relu_1, 0);  relu_1 = None
        full_default: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "f32[256, 1024]" = torch.ops.aten.where.self(le, full_default, mm);  le = mm = None
        permute_1: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_4, [1, 0]);  primals_4 = None
        permute_7: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_1, [1, 0]);  permute_1 = None
        mm_2: "f32[256, 1024]" = torch.ops.aten.mm.default(where, permute_7);  permute_7 = None
        permute_8: "f32[1024, 256]" = torch.ops.aten.permute.default(where, [1, 0])
        mm_3: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_8, relu);  permute_8 = None
        sum_2: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(where, [0], True);  where = None
        view_1: "f32[1024]" = torch.ops.aten.reshape.default(sum_2, [1024]);  sum_2 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/soft_actor_critic/nets.py:113 in forward, code: x = F.relu(self.fc1(state))
        le_1: "b8[256, 1024]" = torch.ops.aten.le.Scalar(relu, 0);  relu = None
        where_1: "f32[256, 1024]" = torch.ops.aten.where.self(le_1, full_default, mm_2);  le_1 = full_default = mm_2 = None
        permute_11: "f32[1024, 256]" = torch.ops.aten.permute.default(where_1, [1, 0])
        mm_4: "f32[1024, 3]" = torch.ops.aten.mm.default(permute_11, primals_3);  permute_11 = primals_3 = None
        sum_3: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(where_1, [0], True);  where_1 = None
        view_2: "f32[1024]" = torch.ops.aten.reshape.default(sum_3, [1024]);  sum_3 = None
        return (mm_4, view_2, None, mm_3, view_1, mm_1, view)
