import torch
from torch import device
from math import inf, nan

class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "f32[64]", arg1_1: "f32[64]", arg2_1: "f32[64, 64, 16, 16]"):
        # No stacktrace found for following nodes
        view: "f32[64, 32, 2, 256]" = torch.ops.aten.reshape.default(arg2_1, [64, 32, 2, 256]);  arg2_1 = None
        var_mean = torch.ops.aten.var_mean.correction(view, [2, 3], correction = 0, keepdim = True)
        getitem: "f32[64, 32, 1, 1]" = var_mean[0]
        getitem_1: "f32[64, 32, 1, 1]" = var_mean[1];  var_mean = None
        sub: "f32[64, 32, 2, 256]" = torch.ops.aten.sub.Tensor(view, getitem_1);  view = getitem_1 = None
        add: "f32[64, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt: "f32[64, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add);  add = None
        mul: "f32[64, 32, 2, 256]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
        view_1: "f32[64, 64, 16, 16]" = torch.ops.aten.reshape.default(mul, [64, 64, 16, 16]);  mul = None
        unsqueeze: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(arg0_1, 0);  arg0_1 = None
        unsqueeze_1: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, 2);  unsqueeze = None
        unsqueeze_2: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1, 3);  unsqueeze_1 = None
        mul_1: "f32[64, 64, 16, 16]" = torch.ops.aten.mul.Tensor(view_1, unsqueeze_2);  view_1 = unsqueeze_2 = None
        unsqueeze_3: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(arg1_1, 0);  arg1_1 = None
        unsqueeze_4: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3, 2);  unsqueeze_3 = None
        unsqueeze_5: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, 3);  unsqueeze_4 = None
        add_1: "f32[64, 64, 16, 16]" = torch.ops.aten.add.Tensor(mul_1, unsqueeze_5);  mul_1 = unsqueeze_5 = None
        return (add_1,)
