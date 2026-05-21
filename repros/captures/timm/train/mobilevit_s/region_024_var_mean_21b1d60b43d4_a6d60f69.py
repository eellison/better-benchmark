"""
Standalone repro captured via capture_hook.
Label: timm_mobilevit_s_train
Pattern hash: 21b1d60b43d4
Shape hash: a6d60f69
"""
_shapes_config = "(T([128, 96, 32, 32], f32, stride=(98304, 1, 3072, 96)), T([96], f32), T([96], f32), T([128, 96, 32, 32], f32, stride=(98304, 1, 3072, 96)))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, convolution_18: "f32[128, 96, 32, 32]", primals_135: "f32[96]", primals_136: "f32[96]", add_92: "f32[128, 96, 32, 32]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_correction = torch.ops.aten.var_mean.correction(convolution_18, [0, 2, 3], correction = 0, keepdim = True)
        getitem: "f32[1, 96, 1, 1]" = var_mean_correction[0]
        getitem_1: "f32[1, 96, 1, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor: "f32[1, 96, 1, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt_default: "f32[1, 96, 1, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        sub_tensor: "f32[128, 96, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_18, getitem_1);  convolution_18 = getitem_1 = None
        mul_tensor: "f32[128, 96, 32, 32]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        unsqueeze_default: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(primals_135, -1);  primals_135 = None
        unsqueeze_default_1: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        mul_tensor_1: "f32[128, 96, 32, 32]" = torch.ops.aten.mul.Tensor(mul_tensor, unsqueeze_default_1);  mul_tensor = unsqueeze_default_1 = None
        unsqueeze_default_2: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(primals_136, -1);  primals_136 = None
        unsqueeze_default_3: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        add_tensor_1: "f32[128, 96, 32, 32]" = torch.ops.aten.add.Tensor(mul_tensor_1, unsqueeze_default_3);  mul_tensor_1 = unsqueeze_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        neg_default: "f32[128, 96, 32, 32]" = torch.ops.aten.neg.default(add_tensor_1)
        exp_default: "f32[128, 96, 32, 32]" = torch.ops.aten.exp.default(neg_default);  neg_default = None
        add_tensor_2: "f32[128, 96, 32, 32]" = torch.ops.aten.add.Tensor(exp_default, 1);  exp_default = None
        div_tensor: "f32[128, 96, 32, 32]" = torch.ops.aten.div.Tensor(add_tensor_1, add_tensor_2);  add_tensor_1 = add_tensor_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/mobilevit.py:277 in forward, code: x = self.conv_fusion(torch.cat((shortcut, x), dim=1))
        cat_default: "f32[128, 192, 32, 32]" = torch.ops.aten.cat.default([add_92, div_tensor], 1);  add_92 = div_tensor = None
        return cat_default



def make_inputs():
    return [
    torch.randn(12582912, dtype=torch.float32, device='cuda').as_strided([128, 96, 32, 32], [98304, 1, 3072, 96]),  # convolution_18
    torch.randn([96], dtype=torch.float32, device='cuda'),
    torch.randn([96], dtype=torch.float32, device='cuda'),
    torch.randn(12582912, dtype=torch.float32, device='cuda').as_strided([128, 96, 32, 32], [98304, 1, 3072, 96]),  # add_92
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
