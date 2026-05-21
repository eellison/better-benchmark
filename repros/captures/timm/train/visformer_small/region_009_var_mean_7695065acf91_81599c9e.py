"""
Standalone repro captured via capture_hook.
Label: timm_visformer_small_train
Pattern hash: 7695065acf91
Shape hash: 81599c9e
"""
_shapes_config = "(T([128, 384, 14, 14], f32, stride=(75264, 1, 5376, 384)), T([128, 384, 14, 14], f32, stride=(75264, 1, 5376, 384)), T([384], f32), T([384], f32))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, add_111: "f32[128, 384, 14, 14]", convolution_37: "f32[128, 384, 14, 14]", primals_132: "f32[384]", primals_133: "f32[384]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:176 in forward, code: x = x + self.drop_path(self.attn(self.norm1(x)))
        add_tensor: "f32[128, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_111, convolution_37);  add_111 = convolution_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:177 in forward, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
        var_mean_correction = torch.ops.aten.var_mean.correction(add_tensor, [0, 2, 3], correction = 0, keepdim = True)
        getitem: "f32[1, 384, 1, 1]" = var_mean_correction[0]
        getitem_1: "f32[1, 384, 1, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor_1: "f32[1, 384, 1, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt_default: "f32[1, 384, 1, 1]" = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        sub_tensor: "f32[128, 384, 14, 14]" = torch.ops.aten.sub.Tensor(add_tensor, getitem_1);  add_tensor = getitem_1 = None
        mul_tensor: "f32[128, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        unsqueeze_default: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(primals_132, -1);  primals_132 = None
        unsqueeze_default_1: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        mul_tensor_1: "f32[128, 384, 14, 14]" = torch.ops.aten.mul.Tensor(mul_tensor, unsqueeze_default_1);  mul_tensor = unsqueeze_default_1 = None
        unsqueeze_default_2: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(primals_133, -1);  primals_133 = None
        unsqueeze_default_3: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        add_tensor_2: "f32[128, 384, 14, 14]" = torch.ops.aten.add.Tensor(mul_tensor_1, unsqueeze_default_3);  mul_tensor_1 = unsqueeze_default_3 = None
        return add_tensor_2



def make_inputs():
    return [
    torch.randn(9633792, dtype=torch.float32, device='cuda').as_strided([128, 384, 14, 14], [75264, 1, 5376, 384]),  # add_111
    torch.randn(9633792, dtype=torch.float32, device='cuda').as_strided([128, 384, 14, 14], [75264, 1, 5376, 384]),  # convolution_37
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
