import torch
from torch import device
from math import inf, nan

class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[16, 1]", primals_2: "f32[16]", primals_3: "f32[1000, 1]", primals_4: "f32[16, 16]", primals_5: "f32[16]", primals_6: "f32[16, 16]", primals_7: "f32[16]", primals_8: "f32[16, 16]", primals_9: "f32[16]", primals_10: "f32[1, 16]", primals_11: "f32[1]"):
        # No stacktrace found for following nodes
        permute: "f32[1, 16]" = torch.ops.aten.permute.default(primals_1, [1, 0]);  primals_1 = None
        mul: "f32[1000, 16]" = torch.ops.aten.mul.Tensor(primals_3, permute);  permute = None
        mul_1: "f32[1000, 16]" = torch.ops.aten.mul.Tensor(mul, 1);  mul = None
        mul_2: "f32[16]" = torch.ops.aten.mul.Tensor(primals_2, 1);  primals_2 = None
        add: "f32[1000, 16]" = torch.ops.aten.add.Tensor(mul_1, mul_2);  mul_1 = mul_2 = None
        tanh: "f32[1000, 16]" = torch.ops.aten.tanh.default(add);  add = None
        permute_1: "f32[16, 16]" = torch.ops.aten.permute.default(primals_4, [1, 0])
        addmm: "f32[1000, 16]" = torch.ops.aten.addmm.default(primals_5, tanh, permute_1);  primals_5 = permute_1 = None
        tanh_1: "f32[1000, 16]" = torch.ops.aten.tanh.default(addmm);  addmm = None
        permute_2: "f32[16, 16]" = torch.ops.aten.permute.default(primals_6, [1, 0])
        addmm_1: "f32[1000, 16]" = torch.ops.aten.addmm.default(primals_7, tanh_1, permute_2);  primals_7 = permute_2 = None
        tanh_2: "f32[1000, 16]" = torch.ops.aten.tanh.default(addmm_1);  addmm_1 = None
        permute_3: "f32[16, 16]" = torch.ops.aten.permute.default(primals_8, [1, 0])
        addmm_2: "f32[1000, 16]" = torch.ops.aten.addmm.default(primals_9, tanh_2, permute_3);  primals_9 = permute_3 = None
        tanh_3: "f32[1000, 16]" = torch.ops.aten.tanh.default(addmm_2);  addmm_2 = None
        permute_4: "f32[16, 1]" = torch.ops.aten.permute.default(primals_10, [1, 0])
        addmm_3: "f32[1000, 1]" = torch.ops.aten.addmm.default(primals_11, tanh_3, permute_4);  primals_11 = permute_4 = None
        return (addmm_3, primals_3, primals_4, primals_6, primals_8, primals_10, tanh, tanh_1, tanh_2, tanh_3)
