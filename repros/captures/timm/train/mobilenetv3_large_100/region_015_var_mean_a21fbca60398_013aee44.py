"""
Standalone repro captured via capture_hook.
Label: timm_mobilenetv3_large_100_train
Pattern hash: a21fbca60398
Shape hash: 013aee44
"""
_shapes_config = "(T([512, 112, 14, 14], f32, stride=(21952, 1, 1568, 112)), T([112], f32), T([112], f32))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, convolution_40: "f32[512, 112, 14, 14]", primals_214: "f32[112]", primals_215: "f32[112]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_correction = torch.ops.aten.var_mean.correction(convolution_40, [0, 2, 3], correction = 0, keepdim = True)
        getitem: "f32[1, 112, 1, 1]" = var_mean_correction[0]
        getitem_1: "f32[1, 112, 1, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor: "f32[1, 112, 1, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt_default: "f32[1, 112, 1, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        sub_tensor: "f32[512, 112, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_40, getitem_1);  convolution_40 = getitem_1 = None
        mul_tensor: "f32[512, 112, 14, 14]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        unsqueeze_default: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(primals_214, -1);  primals_214 = None
        unsqueeze_default_1: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        mul_tensor_1: "f32[512, 112, 14, 14]" = torch.ops.aten.mul.Tensor(mul_tensor, unsqueeze_default_1);  mul_tensor = unsqueeze_default_1 = None
        unsqueeze_default_2: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(primals_215, -1);  primals_215 = None
        unsqueeze_default_3: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        add_tensor_1: "f32[512, 112, 14, 14]" = torch.ops.aten.add.Tensor(mul_tensor_1, unsqueeze_default_3);  mul_tensor_1 = unsqueeze_default_3 = None
        return add_tensor_1



def make_inputs():
    return [
    torch.randn(11239424, dtype=torch.float32, device='cuda').as_strided([512, 112, 14, 14], [21952, 1, 1568, 112]),  # convolution_40
    torch.randn([112], dtype=torch.float32, device='cuda'),
    torch.randn([112], dtype=torch.float32, device='cuda'),
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
