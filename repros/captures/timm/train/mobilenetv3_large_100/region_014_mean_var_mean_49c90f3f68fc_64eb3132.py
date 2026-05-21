"""
Standalone repro captured via capture_hook.
Label: timm_mobilenetv3_large_100_train
Pattern hash: 49c90f3f68fc
Shape hash: 64eb3132
"""
_shapes_config = "(T([512, 672, 14, 14], f32, stride=(131712, 1, 9408, 672)), T([672], f32), T([672], f32))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, convolution_42: "f32[512, 672, 14, 14]", primals_226: "f32[672]", primals_227: "f32[672]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_correction = torch.ops.aten.var_mean.correction(convolution_42, [0, 2, 3], correction = 0, keepdim = True)
        getitem: "f32[1, 672, 1, 1]" = var_mean_correction[0]
        getitem_1: "f32[1, 672, 1, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor: "f32[1, 672, 1, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt_default: "f32[1, 672, 1, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        sub_tensor: "f32[512, 672, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_42, getitem_1);  convolution_42 = getitem_1 = None
        mul_tensor: "f32[512, 672, 14, 14]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        unsqueeze_default: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(primals_226, -1);  primals_226 = None
        unsqueeze_default_1: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        mul_tensor_1: "f32[512, 672, 14, 14]" = torch.ops.aten.mul.Tensor(mul_tensor, unsqueeze_default_1);  mul_tensor = unsqueeze_default_1 = None
        unsqueeze_default_2: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(primals_227, -1);  primals_227 = None
        unsqueeze_default_3: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        add_tensor_1: "f32[512, 672, 14, 14]" = torch.ops.aten.add.Tensor(mul_tensor_1, unsqueeze_default_3);  mul_tensor_1 = unsqueeze_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        add_tensor_2: "f32[512, 672, 14, 14]" = torch.ops.aten.add.Tensor(add_tensor_1, 3)
        clamp_min_default: "f32[512, 672, 14, 14]" = torch.ops.aten.clamp_min.default(add_tensor_2, 0);  add_tensor_2 = None
        clamp_max_default: "f32[512, 672, 14, 14]" = torch.ops.aten.clamp_max.default(clamp_min_default, 6);  clamp_min_default = None
        mul_tensor_2: "f32[512, 672, 14, 14]" = torch.ops.aten.mul.Tensor(add_tensor_1, clamp_max_default);  add_tensor_1 = clamp_max_default = None
        div_tensor: "f32[512, 672, 14, 14]" = torch.ops.aten.div.Tensor(mul_tensor_2, 6);  mul_tensor_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:79 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_dim: "f32[512, 672, 1, 1]" = torch.ops.aten.mean.dim(div_tensor, [2, 3], True);  div_tensor = None
        return mean_dim



def make_inputs():
    return [
    torch.randn(67436544, dtype=torch.float32, device='cuda').as_strided([512, 672, 14, 14], [131712, 1, 9408, 672]),  # convolution_42
    torch.randn([672], dtype=torch.float32, device='cuda'),
    torch.randn([672], dtype=torch.float32, device='cuda'),
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
