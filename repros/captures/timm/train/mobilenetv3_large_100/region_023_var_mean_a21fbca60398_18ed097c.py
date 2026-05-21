"""
Standalone repro captured via capture_hook.
Label: timm_mobilenetv3_large_100_train
Pattern hash: a21fbca60398
Shape hash: 18ed097c
"""
_shapes_config = "(T([512, 80, 14, 14], f32, stride=(15680, 1, 1120, 80)), T([80], f32), T([80], f32))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, convolution_26: "f32[512, 80, 14, 14]", primals_138: "f32[80]", primals_139: "f32[80]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_correction = torch.ops.aten.var_mean.correction(convolution_26, [0, 2, 3], correction = 0, keepdim = True)
        getitem: "f32[1, 80, 1, 1]" = var_mean_correction[0]
        getitem_1: "f32[1, 80, 1, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor: "f32[1, 80, 1, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt_default: "f32[1, 80, 1, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        sub_tensor: "f32[512, 80, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_26, getitem_1);  convolution_26 = getitem_1 = None
        mul_tensor: "f32[512, 80, 14, 14]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        unsqueeze_default: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(primals_138, -1);  primals_138 = None
        unsqueeze_default_1: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        mul_tensor_1: "f32[512, 80, 14, 14]" = torch.ops.aten.mul.Tensor(mul_tensor, unsqueeze_default_1);  mul_tensor = unsqueeze_default_1 = None
        unsqueeze_default_2: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(primals_139, -1);  primals_139 = None
        unsqueeze_default_3: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        add_tensor_1: "f32[512, 80, 14, 14]" = torch.ops.aten.add.Tensor(mul_tensor_1, unsqueeze_default_3);  mul_tensor_1 = unsqueeze_default_3 = None
        return add_tensor_1



def make_inputs():
    return [
    torch.randn(8028160, dtype=torch.float32, device='cuda').as_strided([512, 80, 14, 14], [15680, 1, 1120, 80]),  # convolution_26
    torch.randn([80], dtype=torch.float32, device='cuda'),
    torch.randn([80], dtype=torch.float32, device='cuda'),
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
