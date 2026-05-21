"""
Standalone repro captured via capture_hook.
Label: timm_mobilenetv3_large_100_train
Pattern hash: 09d71f9017f3
Shape hash: 16fe019c
"""
_shapes_config = "(T([512, 1280], f32), T([512, 1280, 1, 1], f32), S([512, 1280, 1, 1]))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, mm: "f32[512, 1280]", convolution_62: "f32[512, 1280, 1, 1]", _shape_param_0):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/mobilenetv3.py:326 in forward_head, code: x = self.flatten(x)
        reshape_default: "f32[512, 1280, 1, 1]" = torch.ops.aten.reshape.default(mm, _shape_param_0);  mm = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/mobilenetv3.py:325 in forward_head, code: x = self.act2(x)
        le_scalar: "b8[512, 1280, 1, 1]" = torch.ops.aten.le.Scalar(convolution_62, -3)
        lt_scalar: "b8[512, 1280, 1, 1]" = torch.ops.aten.lt.Scalar(convolution_62, 3)
        div_tensor: "f32[512, 1280, 1, 1]" = torch.ops.aten.div.Tensor(convolution_62, 3);  convolution_62 = None
        add_tensor: "f32[512, 1280, 1, 1]" = torch.ops.aten.add.Tensor(div_tensor, 0.5);  div_tensor = None
        mul_tensor: "f32[512, 1280, 1, 1]" = torch.ops.aten.mul.Tensor(reshape_default, add_tensor);  add_tensor = None
        where_self: "f32[512, 1280, 1, 1]" = torch.ops.aten.where.self(lt_scalar, mul_tensor, reshape_default);  lt_scalar = mul_tensor = reshape_default = None
        full_default: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_1: "f32[512, 1280, 1, 1]" = torch.ops.aten.where.self(le_scalar, full_default, where_self);  le_scalar = full_default = where_self = None
        return where_self_1



def make_inputs():
    return [
    torch.randn([512, 1280], dtype=torch.float32, device='cuda'),
    torch.randn([512, 1280, 1, 1], dtype=torch.float32, device='cuda'),
    [512, 1280, 1, 1],  # _shape_param_0
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
