"""
Standalone repro captured via capture_hook.
Label: timm_mobilenetv3_large_100_train
Pattern hash: 01d9bf338428
Shape hash: 78d08be5
"""
_shapes_config = "(T([512, 16, 112, 112], f32, stride=(200704, 1, 1792, 16)), T([16], f32), T([16], f32), T([512, 16, 112, 112], f32, stride=(200704, 1, 1792, 16)))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, convolution_2: "f32[512, 16, 112, 112]", primals_18: "f32[16]", primals_19: "f32[16]", div: "f32[512, 16, 112, 112]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_correction = torch.ops.aten.var_mean.correction(convolution_2, [0, 2, 3], correction = 0, keepdim = True)
        getitem: "f32[1, 16, 1, 1]" = var_mean_correction[0]
        getitem_1: "f32[1, 16, 1, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor: "f32[1, 16, 1, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt_default: "f32[1, 16, 1, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        sub_tensor: "f32[512, 16, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_2, getitem_1);  convolution_2 = getitem_1 = None
        mul_tensor: "f32[512, 16, 112, 112]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        unsqueeze_default: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(primals_18, -1);  primals_18 = None
        unsqueeze_default_1: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        mul_tensor_1: "f32[512, 16, 112, 112]" = torch.ops.aten.mul.Tensor(mul_tensor, unsqueeze_default_1);  mul_tensor = unsqueeze_default_1 = None
        unsqueeze_default_2: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(primals_19, -1);  primals_19 = None
        unsqueeze_default_3: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        add_tensor_1: "f32[512, 16, 112, 112]" = torch.ops.aten.add.Tensor(mul_tensor_1, unsqueeze_default_3);  mul_tensor_1 = unsqueeze_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:230 in forward, code: x = self.drop_path(x) + shortcut
        add_tensor_2: "f32[512, 16, 112, 112]" = torch.ops.aten.add.Tensor(add_tensor_1, div);  add_tensor_1 = div = None
        return add_tensor_2



def make_inputs():
    return [
    torch.randn(102760448, dtype=torch.float32, device='cuda').as_strided([512, 16, 112, 112], [200704, 1, 1792, 16]),  # convolution_2
    torch.randn([16], dtype=torch.float32, device='cuda'),
    torch.randn([16], dtype=torch.float32, device='cuda'),
    torch.randn(102760448, dtype=torch.float32, device='cuda').as_strided([512, 16, 112, 112], [200704, 1, 1792, 16]),  # div
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
