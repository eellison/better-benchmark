"""
Standalone repro captured via capture_hook.
Label: timm_mobilenetv2_100_train
Pattern hash: fd4b1c218321
Shape hash: 4bf3f793
"""
_shapes_config = "(T([128, 576, 14, 14], f32, stride=(112896, 1, 8064, 576)), T([576], f32), T([576], f32))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, convolution_39: "f32[128, 576, 14, 14]", primals_240: "f32[576]", primals_241: "f32[576]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_correction = torch.ops.aten.var_mean.correction(convolution_39, [0, 2, 3], correction = 0, keepdim = True)
        getitem: "f32[1, 576, 1, 1]" = var_mean_correction[0]
        getitem_1: "f32[1, 576, 1, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor: "f32[1, 576, 1, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt_default: "f32[1, 576, 1, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        sub_tensor: "f32[128, 576, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_39, getitem_1);  convolution_39 = getitem_1 = None
        mul_tensor: "f32[128, 576, 14, 14]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        unsqueeze_default: "f32[576, 1]" = torch.ops.aten.unsqueeze.default(primals_240, -1);  primals_240 = None
        unsqueeze_default_1: "f32[576, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        mul_tensor_1: "f32[128, 576, 14, 14]" = torch.ops.aten.mul.Tensor(mul_tensor, unsqueeze_default_1);  mul_tensor = unsqueeze_default_1 = None
        unsqueeze_default_2: "f32[576, 1]" = torch.ops.aten.unsqueeze.default(primals_241, -1);  primals_241 = None
        unsqueeze_default_3: "f32[576, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        add_tensor_1: "f32[128, 576, 14, 14]" = torch.ops.aten.add.Tensor(mul_tensor_1, unsqueeze_default_3);  mul_tensor_1 = unsqueeze_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        clamp_min_default: "f32[128, 576, 14, 14]" = torch.ops.aten.clamp_min.default(add_tensor_1, 0.0);  add_tensor_1 = None
        clamp_max_default: "f32[128, 576, 14, 14]" = torch.ops.aten.clamp_max.default(clamp_min_default, 6.0);  clamp_min_default = None
        return clamp_max_default



def make_inputs():
    return [
    torch.randn(14450688, dtype=torch.float32, device='cuda').as_strided([128, 576, 14, 14], [112896, 1, 8064, 576]),  # convolution_39
    torch.randn([576], dtype=torch.float32, device='cuda'),
    torch.randn([576], dtype=torch.float32, device='cuda'),
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
