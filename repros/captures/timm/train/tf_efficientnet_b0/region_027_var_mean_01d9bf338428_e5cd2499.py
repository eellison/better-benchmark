"""
Standalone repro captured via capture_hook.
Label: timm_tf_efficientnet_b0_train
Pattern hash: 01d9bf338428
Shape hash: e5cd2499
"""
_shapes_config = "(T([128, 40, 28, 28], f32, stride=(31360, 1, 1120, 40)), T([40], f32), T([40], f32), T([128, 40, 28, 28], f32, stride=(31360, 1, 1120, 40)))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, convolution_24: "f32[128, 40, 28, 28]", primals_110: "f32[40]", primals_111: "f32[40]", add_72: "f32[128, 40, 28, 28]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_correction = torch.ops.aten.var_mean.correction(convolution_24, [0, 2, 3], correction = 0, keepdim = True)
        getitem: "f32[1, 40, 1, 1]" = var_mean_correction[0]
        getitem_1: "f32[1, 40, 1, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor: "f32[1, 40, 1, 1]" = torch.ops.aten.add.Tensor(getitem, 0.001);  getitem = None
        rsqrt_default: "f32[1, 40, 1, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        sub_tensor: "f32[128, 40, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_24, getitem_1);  convolution_24 = getitem_1 = None
        mul_tensor: "f32[128, 40, 28, 28]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        unsqueeze_default: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(primals_110, -1);  primals_110 = None
        unsqueeze_default_1: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        mul_tensor_1: "f32[128, 40, 28, 28]" = torch.ops.aten.mul.Tensor(mul_tensor, unsqueeze_default_1);  mul_tensor = unsqueeze_default_1 = None
        unsqueeze_default_2: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(primals_111, -1);  primals_111 = None
        unsqueeze_default_3: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        add_tensor_1: "f32[128, 40, 28, 28]" = torch.ops.aten.add.Tensor(mul_tensor_1, unsqueeze_default_3);  mul_tensor_1 = unsqueeze_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:338 in forward, code: x = self.drop_path(x) + shortcut
        add_tensor_2: "f32[128, 40, 28, 28]" = torch.ops.aten.add.Tensor(add_tensor_1, add_72);  add_tensor_1 = add_72 = None
        return add_tensor_2



def make_inputs():
    return [
    torch.randn(4014080, dtype=torch.float32, device='cuda').as_strided([128, 40, 28, 28], [31360, 1, 1120, 40]),  # convolution_24
    torch.randn([40], dtype=torch.float32, device='cuda'),
    torch.randn([40], dtype=torch.float32, device='cuda'),
    torch.randn(4014080, dtype=torch.float32, device='cuda').as_strided([128, 40, 28, 28], [31360, 1, 1120, 40]),  # add_72
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
