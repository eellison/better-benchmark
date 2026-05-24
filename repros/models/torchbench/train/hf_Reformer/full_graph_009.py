import torch
import torch._inductor.inductor_prims  # registers inductor prims ops
from torch import device
from math import inf, nan

class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[512]", primals_2: "f32[512]", primals_3: "f32[8, 4096, 512]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1796 in torch_dynamo_resume_in_forward_at_1781, code: hidden_states = self.layer_norm(hidden_states)
        var_mean = torch.ops.aten.var_mean.correction(primals_3, [2], correction = 0, keepdim = True)
        getitem: "f32[8, 4096, 1]" = var_mean[0]
        getitem_1: "f32[8, 4096, 1]" = var_mean[1];  var_mean = None
        add: "f32[8, 4096, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-12);  getitem = None
        rsqrt: "f32[8, 4096, 1]" = torch.ops.aten.rsqrt.default(add);  add = None
        sub: "f32[8, 4096, 512]" = torch.ops.aten.sub.Tensor(primals_3, getitem_1)
        mul: "f32[8, 4096, 512]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        mul_1: "f32[8, 4096, 512]" = torch.ops.aten.mul.Tensor(mul, primals_1);  mul = None
        add_1: "f32[8, 4096, 512]" = torch.ops.aten.add.Tensor(mul_1, primals_2);  mul_1 = primals_2 = None

        # No stacktrace found for following nodes
        inductor_seeds_default: "i64[1]" = torch.ops.prims.inductor_seeds.default(1, device(type='cuda', index=0))

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1799 in torch_dynamo_resume_in_forward_at_1781, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        inductor_lookup_seed_default: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 0);  inductor_seeds_default = None
        inductor_random_default: "f32[8, 4096, 512]" = torch.ops.prims.inductor_random.default([8, 4096, 512], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        gt: "b8[8, 4096, 512]" = torch.ops.aten.gt.Scalar(inductor_random_default, 0.05);  inductor_random_default = None
        mul_2: "f32[8, 4096, 512]" = torch.ops.aten.mul.Tensor(gt, add_1);  add_1 = None
        mul_3: "f32[8, 4096, 512]" = torch.ops.aten.mul.Tensor(mul_2, 1.0526315789473684);  mul_2 = None
        return (mul_3, primals_1, primals_3, getitem_1, rsqrt, gt)
